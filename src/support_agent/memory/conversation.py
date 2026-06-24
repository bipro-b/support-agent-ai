"""Short-term memory: the conversation history for a single session.

This is the "the chatbot remembers what I said a moment ago" capability. As we proved
in Phase 0, the model itself remembers nothing — so a conversation is just a list of
messages that WE keep and resend every turn.

Two important design choices live here:

1. **Stored history is the source of truth, and it stays clean.** We store exactly
   what the customer said and what the agent replied — nothing else. The prompt we
   actually send (persona, retrieved sources, summaries) is BUILT from this each turn
   by the ContextAssembler. Keeping the two separate is what lets us re-assemble the
   prompt differently as the conversation grows without corrupting the record.

2. **A `summary` field for compressed older turns.** When the verbatim history gets
   too long for our token budget, the summarizer compresses the oldest turns into this
   field and drops them from `messages`. So memory = (a compact summary of the distant
   past) + (the recent turns verbatim).
"""

from __future__ import annotations

from ..llm import Message, assistant, user


class ConversationMemory:
    """Holds one session's messages plus a rolling summary of dropped older turns."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.summary: str | None = None  # compressed older turns, set by the summarizer

    def add_user(self, text: str) -> None:
        self.messages.append(user(text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(assistant(text))

    def last_user_text(self) -> str:
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return str(msg["content"])
        return ""

    def __len__(self) -> int:
        return len(self.messages)

    # Serialization — so a session store can persist this across requests/processes.
    # Conversation memory holds only clean text turns, so it's trivially JSON-safe.
    def to_dict(self) -> dict:
        return {"messages": self.messages, "summary": self.summary}

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        conv = cls()
        conv.messages = list(data.get("messages") or [])
        conv.summary = data.get("summary")
        return conv
