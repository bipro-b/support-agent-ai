"""Request/response models for the HTTP API.

Pydantic models give us validation at the edge (a malformed request is rejected with a
clear 422 before it ever reaches the engine) and a typed contract the API auto-documents.
Validate at the boundary, trust the inside — a core service-design habit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    customer_id: str = Field(..., description="Stable id for the customer (keys long-term memory).")
    message: str = Field(..., min_length=1, description="The customer's message this turn.")
    session_id: str | None = Field(
        default=None,
        description="Conversation id. Omit on the first turn; reuse the returned id after.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    retrieved: list[str]
    steps: list[str] | None = None
    summarized: bool = False
    new_facts: list[str] = []
    cost_usd: float = 0.0
    degraded: bool = False  # true if a dependency failed and this is a fallback/degraded reply
