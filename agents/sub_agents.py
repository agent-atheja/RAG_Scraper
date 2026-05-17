from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from agents.state import AgentState
from rag.retriever import build_rag_tool

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1, max_tokens=1024)

_PERSONAS: dict[str, str] = {
    "python":     "You are a Python expert with deep knowledge of W3Schools Python tutorials.",
    "html":       "You are an HTML expert specializing in web structure, semantic markup, and HTML5.",
    "css":        "You are a CSS expert skilled in styling, layouts (flexbox/grid), and responsive design.",
    "javascript": "You are a JavaScript expert covering ES6+, async patterns, and DOM manipulation.",
    "sql":        "You are a SQL expert covering queries, joins, aggregations, and database concepts.",
    "general":    "You are a full-stack web development expert covering all W3Schools topics.",
}

_SYSTEM_TEMPLATE = """\
{persona}

Answer using ONLY the provided W3Schools context below.
If the context is insufficient, say so honestly.
Always end with: Source: <url>

Format:
1. Direct answer (2–3 sentences)
2. Code block (if applicable)
3. Source: [URL]
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def build_sub_agent(topic: str):
    retrieve = build_rag_tool(topic)
    system_msg = _SYSTEM_TEMPLATE.format(persona=_PERSONAS[topic])
    chain = _PROMPT | _llm

    def sub_agent_node(state: AgentState) -> AgentState:
        last_user_msg = next(
            (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
            "",
        )

        chunks = retrieve(last_user_msg, top_k=6)
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['metadata']['url']}]\n{c['text']}" for c in chunks
        )
        sources = list({c["metadata"]["url"] for c in chunks})

        response = chain.invoke({
            "system": system_msg,
            "context": context_text,
            "question": last_user_msg,
        })
        answer = response.content

        return {
            **state,
            "messages": [{"role": "assistant", "content": answer}],
            "retrieved_context": chunks,
            "final_answer": answer,
            "sources": sources,
        }

    sub_agent_node.__name__ = f"{topic}_agent_node"
    return sub_agent_node


python_agent_node     = build_sub_agent("python")
html_agent_node       = build_sub_agent("html")
css_agent_node        = build_sub_agent("css")
javascript_agent_node = build_sub_agent("javascript")
sql_agent_node        = build_sub_agent("sql")
general_agent_node    = build_sub_agent("general")
