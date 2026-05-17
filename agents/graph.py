from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.orchestrator import orchestrator_node
from agents.sub_agents import (
    css_agent_node,
    general_agent_node,
    html_agent_node,
    javascript_agent_node,
    python_agent_node,
    sql_agent_node,
)

_SUB_AGENTS = {
    "python":     python_agent_node,
    "html":       html_agent_node,
    "css":        css_agent_node,
    "javascript": javascript_agent_node,
    "sql":        sql_agent_node,
    "general":    general_agent_node,
}


def _route(state: AgentState) -> str:
    return state["selected_agent"] or "general"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    for name, node_fn in _SUB_AGENTS.items():
        graph.add_node(name, node_fn)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route,
        {name: name for name in _SUB_AGENTS},
    )

    for name in _SUB_AGENTS:
        graph.add_edge(name, END)

    return graph.compile()


w3_agent_graph = build_graph()
