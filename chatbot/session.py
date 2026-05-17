import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)


class InMemorySessionStore:
    """Thread-safe in-memory session store. Swap for Redis in production."""

    def __init__(self) -> None:
        self._store: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: str | None) -> ChatSession:
        if session_id and session_id in self._store:
            session = self._store[session_id]
            session.last_active = datetime.utcnow()
            return session
        session = ChatSession(session_id=session_id or str(uuid.uuid4()))
        self._store[session.session_id] = session
        return session

    def save(self, session: ChatSession) -> None:
        self._store[session.session_id] = session

    def get_history(self, session_id: str) -> list[dict]:
        session = self._store.get(session_id)
        return session.messages if session else []


session_store = InMemorySessionStore()
