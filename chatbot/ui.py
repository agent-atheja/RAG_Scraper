import os
import sys
import uuid

# Ensure repo root is on the path (needed on Streamlit Cloud)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

AGENT_EMOJIS = {
    "python": "🐍",
    "html": "🌐",
    "css": "🎨",
    "javascript": "⚡",
    "sql": "🗄️",
    "general": "🤖",
}

st.set_page_config(
    page_title="W3Schools AI Assistant",
    page_icon="📚",
    layout="wide",
)
st.title("📚 W3Schools AI Assistant")
st.caption("Ask anything about HTML, CSS, JavaScript, Python, or SQL — powered by RAG + LangGraph")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(show_spinner="Loading agent graph...")
def load_graph():
    from agents.graph import w3_agent_graph
    return w3_agent_graph


def get_db_stats():
    try:
        from rag.store import collection_stats
        return collection_stats()
    except Exception:
        return {"total_chunks": 0}


with st.sidebar:
    st.header("Session")
    st.code(st.session_state.session_id[:12] + "...", language=None)

    st.divider()
    st.subheader("Topic Agents")
    for topic, emoji in AGENT_EMOJIS.items():
        st.markdown(f"{emoji} **{topic.capitalize()}**")

    st.divider()
    stats = get_db_stats()
    chunks = stats.get("total_chunks", 0)
    if chunks > 0:
        st.success(f"DB: {chunks:,} chunks indexed")
    else:
        st.warning("DB empty — run the scraper pipeline first")

    st.divider()
    if st.button("New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("agent"):
            emoji = AGENT_EMOJIS.get(msg["agent"], "🤖")
            st.caption(f"{emoji} **{msg['agent'].upper()} Agent**")
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- [{src}]({src})")

if prompt := st.chat_input("Ask about HTML, CSS, Python, SQL, JavaScript..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        agent_placeholder = st.empty()
        text_placeholder = st.empty()
        agent_used = ""
        full_response = ""
        sources: list[str] = []

        try:
            graph = load_graph()
            initial_state = {
                "messages": [{"role": "user", "content": prompt}],
                "selected_agent": None,
                "retrieved_context": [],
                "final_answer": "",
                "sources": [],
                "session_id": st.session_state.session_id,
            }
            with st.spinner("Thinking..."):
                result = graph.invoke(initial_state)

            agent_used = result.get("selected_agent") or "general"
            full_response = result.get("final_answer", "")
            sources = result.get("sources", [])

            emoji = AGENT_EMOJIS.get(agent_used, "🤖")
            agent_placeholder.caption(f"{emoji} **{agent_used.upper()} Agent**")
            text_placeholder.markdown(full_response)

        except Exception as exc:
            full_response = f"Error: {exc}"
            text_placeholder.error(full_response)

        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    st.markdown(f"- [{src}]({src})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "agent": agent_used,
        "sources": sources,
    })
