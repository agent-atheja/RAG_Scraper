from rag.embedder import embed_query
from rag.store import query as chroma_query


def build_rag_tool(topic: str):
    """
    Factory: returns a retrieval callable scoped to a specific W3Schools topic.
    Pass topic='general' to search across all topics.
    """
    topic_filter = topic if topic != "general" else None

    def retrieve(user_query: str, top_k: int = 6) -> list[dict]:
        query_vec = embed_query(user_query)
        return chroma_query(query_vec, top_k=top_k, topic_filter=topic_filter)

    retrieve.__name__ = f"search_{topic}_docs"
    retrieve.__doc__ = (
        f"Search W3Schools {topic.upper()} documentation for relevant content."
    )
    return retrieve
