"""Where a conversation's short-term state lives between HTTP requests.

The problem this solves: HTTP is stateless — each request is independent — but a
conversation spans many requests. So we can't keep the conversation "in the handler";
we must store it under a session id and reload it on the next request.

This is the service-layer echo of Phase 0's lesson (the model is stateless) one level up:
now the *transport* is stateless too, so the conversation state has to live somewhere
both turns can reach. That "somewhere" is a session store.

`InMemorySessionStore` is the simplest backend: a dict in the process. It's enough to run
and learn, but it has exactly the limits that drive real architecture decisions:
  - lost on restart (no durability),
  - not shared across processes/replicas (so requests for one session must hit the same
    process — "session affinity" — or you lose the conversation),
  - grows forever without eviction.
In production this is **Redis** (fast, shared across replicas, with TTL eviction). The
`SessionStore` Protocol is the seam: swapping in a RedisSessionStore that serializes via
ConversationMemory.to_dict()/from_dict() touches nothing else. That's the payoff of coding
to an interface — see docs/phase-4-system-design.md §"Where state lives".
"""

from __future__ import annotations

from typing import Protocol

from ..memory.conversation import ConversationMemory


class SessionStore(Protocol):
    def load(self, session_id: str) -> ConversationMemory | None: ...
    def save(self, session_id: str, conversation: ConversationMemory) -> None: ...


class InMemorySessionStore:
    """Single-process, non-durable session store. Fine for dev; Redis for prod."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}

    def load(self, session_id: str) -> ConversationMemory | None:
        data = self._sessions.get(session_id)
        # We store the serialized form (not the live object) on purpose: it mirrors what
        # a Redis-backed store does, so the in-memory and prod paths behave the same.
        return ConversationMemory.from_dict(data) if data is not None else None

    def save(self, session_id: str, conversation: ConversationMemory) -> None:
        self._sessions[session_id] = conversation.to_dict()
