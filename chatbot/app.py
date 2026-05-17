import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from agents.graph import w3_agent_graph
from agents.orchestrator import orchestrator_node
from agents.state import AgentState
from chatbot.session import session_store
from rag.retriever import build_rag_tool
from rag.store import collection_stats

app = FastAPI(title="W3Schools RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    sources: list[str]
    session_id: str


def _build_initial_state(session, messages) -> AgentState:
    return {
        "messages": messages,
        "selected_agent": None,
        "retrieved_context": [],
        "final_answer": "",
        "sources": [],
        "session_id": session.session_id,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = session_store.get_or_create(req.session_id)
    session.messages.append({"role": "user", "content": req.message})

    result = w3_agent_graph.invoke(
        _build_initial_state(session, session.messages.copy())
    )

    session.messages.append({"role": "assistant", "content": result["final_answer"]})
    session_store.save(session)

    return ChatResponse(
        answer=result["final_answer"],
        agent_used=result["selected_agent"],
        sources=result["sources"],
        session_id=session.session_id,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session = session_store.get_or_create(req.session_id)
    session.messages.append({"role": "user", "content": req.message})

    state = _build_initial_state(session, session.messages.copy())
    state = orchestrator_node(state)
    agent = state["selected_agent"]

    retrieve = build_rag_tool(agent)
    chunks = retrieve(req.message, top_k=6)
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['metadata']['url']}]\n{c['text']}" for c in chunks
    )
    sources = list({c["metadata"]["url"] for c in chunks})

    stream_llm = ChatAnthropic(model="claude-sonnet-4-6", streaming=True)

    async def generate():
        yield f"data: {json.dumps({'type': 'agent', 'agent': agent})}\n\n"

        full_response = ""
        async for chunk in stream_llm.astream([
            {
                "role": "system",
                "content": (
                    f"You are a W3Schools {agent.upper()} expert. "
                    "Answer using ONLY the provided context. "
                    "Always cite the source URL."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {req.message}",
            },
        ]):
            token = chunk.content
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        session.messages.append({"role": "assistant", "content": full_response})
        session_store.save(session)

        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'session_id': session.session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return {"messages": session_store.get_history(session_id)}


@app.get("/health")
async def health():
    return {"status": "ok", **collection_stats()}
