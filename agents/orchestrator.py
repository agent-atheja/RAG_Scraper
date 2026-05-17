from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from agents.state import AgentState, VALID_AGENTS

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=10)

_SYSTEM = """\
You are the Orchestrator for a W3Schools knowledge assistant.
Your ONLY job: output the name of the most appropriate specialist agent.

Agents:
- python      → Python syntax, OOP, stdlib, file I/O
- html        → HTML tags, forms, structure, semantic HTML5
- css         → Selectors, flexbox, grid, animations, responsive design
- javascript  → DOM, events, async/await, ES6+, browser APIs
- sql         → Queries, joins, aggregations, MySQL/PostgreSQL syntax
- general     → Cross-topic or ambiguous questions

Rules:
- Output ONLY one word (the agent name), lowercase, no punctuation.
- When a question spans topics pick the PRIMARY one.
- Default to: general
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Question: {question}\n\nAgent name:"),
])

_chain = _PROMPT | _llm


def orchestrator_node(state: AgentState) -> AgentState:
    last_user_msg = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )
    raw = _chain.invoke({"question": last_user_msg}).content.strip().lower()
    selected = raw if raw in VALID_AGENTS else "general"
    return {**state, "selected_agent": selected}
