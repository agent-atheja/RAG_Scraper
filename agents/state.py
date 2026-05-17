import operator
from typing import Annotated, Literal, TypedDict

TOPIC_AGENTS = Literal["python", "html", "css", "javascript", "sql", "general"]
VALID_AGENTS: set[str] = {"python", "html", "css", "javascript", "sql", "general"}


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    selected_agent: TOPIC_AGENTS | None
    retrieved_context: list[dict]
    final_answer: str
    sources: list[str]
    session_id: str
