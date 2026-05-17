import json

import httpx
import streamlit as st

from config import FASTAPI_PORT

API_URL = f"http://localhost:{FASTAPI_PORT}"

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
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Session")
    if st.session_state.session_id:
        st.code(st.session_state.session_id[:12] + "...", language=None)
    else:
        st.caption("No active session yet.")

    st.divider()
    st.subheader("Topic Agents")
    for topic, emoji in AGENT_EMOJIS.items():
        st.markdown(f"{emoji} **{topic.capitalize()}**")

    st.divider()
    if st.button("New Conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    try:
        health = httpx.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"DB: {health.get('total_chunks', '?'):,} chunks indexed")
    except Exception:
        st.warning("API not reachable")

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
        full_response = ""
        agent_used = ""
        sources: list[str] = []

        try:
            with httpx.Client(timeout=90) as client:
                with client.stream(
                    "POST",
                    f"{API_URL}/chat/stream",
                    json={"message": prompt, "session_id": st.session_state.session_id},
                ) as response:
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = json.loads(line[6:])

                        if data["type"] == "agent":
                            agent_used = data["agent"]
                            emoji = AGENT_EMOJIS.get(agent_used, "🤖")
                            agent_placeholder.caption(
                                f"Routing to: {emoji} **{agent_used.upper()} Agent**"
                            )

                        elif data["type"] == "token":
                            full_response += data["text"]
                            text_placeholder.markdown(full_response + "▌")

                        elif data["type"] == "done":
                            sources = data.get("sources", [])
                            new_sid = data.get("session_id")
                            if new_sid:
                                st.session_state.session_id = new_sid
                            text_placeholder.markdown(full_response)

        except Exception as exc:
            full_response = f"Error contacting API: {exc}"
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
