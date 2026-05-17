"""CLI smoke-test for the LangGraph agent graph (no chatbot UI needed)."""
import sys
import uuid

from agents.graph import w3_agent_graph
from agents.state import AgentState


def ask(question: str) -> str:
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": question}],
        "selected_agent": None,
        "retrieved_context": [],
        "final_answer": "",
        "sources": [],
        "session_id": str(uuid.uuid4()),
    }

    result = w3_agent_graph.invoke(initial_state)

    print(f"\n[Routed to: {result['selected_agent'].upper()} Agent]")
    print("-" * 60)
    print(result["final_answer"])
    print("-" * 60)
    if result["sources"]:
        print("Sources:")
        for src in result["sources"]:
            print(f"  {src}")
    print()

    return result["final_answer"]


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "How do I use list comprehension in Python?",
        "What is CSS Grid and how does it differ from Flexbox?",
        "Write a SQL query to find duplicate rows in a table.",
    ]
    for q in questions:
        ask(q)
