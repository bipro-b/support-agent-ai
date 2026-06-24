"""Long-term memory: per-customer facts that persist ACROSS sessions.

Short-term memory dies when the session ends. Long-term memory is what makes the agent
feel like it knows the customer: "Welcome back, Rahim — is this about order #1234 again?"
It's a small, durable set of facts keyed by customer, written to disk so it survives a
process restart (run the Phase 2 demo twice and watch it remember you the second time).

Two pieces:

- A `MemoryStore` interface (Protocol) and a `JsonFileMemoryStore` implementation.
  JSON files are fine for learning; in production this is a database (Postgres for
  structured facts, Redis for fast session state, or a vector store for *semantic*
  memory you retrieve by relevance). The interface is what matters — swapping the
  backend shouldn't touch the rest of the system.

- A `MemoryExtractor` that uses the cheap/fast model to pull durable facts out of a
  conversation turn ("the customer's name is Rahim", "order #1234 arrived late"). This
  is how long-term memory gets WRITTEN automatically instead of by hand. We use the
  fast model on purpose — extraction is a simple task and shouldn't cost Opus money.

Caution worth stating: long-term memory is a privacy surface. Don't persist secrets
(card numbers, passwords) and be deliberate about PII. We'll harden this in Phase 9.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from ..llm import LLMClient, user


@dataclass
class CustomerMemory:
    """The durable profile we keep for one customer."""

    customer_id: str
    name: str | None = None
    facts: list[str] = field(default_factory=list)

    def add_fact(self, fact: str) -> bool:
        """Add a fact if we don't already have it. Returns True if it was new."""
        fact = fact.strip()
        if not fact or fact in self.facts:
            return False
        self.facts.append(fact)
        return True

    def absorb_fact(self, fact: str) -> bool:
        """Store a fact, promoting a 'name is X' fact to the structured name field."""
        if self.name is None and "name is" in fact.lower():
            self.name = fact.split("is", 1)[1].strip().rstrip(".")
            return True
        return self.add_fact(fact)

    def as_prompt_block(self) -> str:
        """Render this memory for injection into the system prompt."""
        if not self.name and not self.facts:
            return "No prior information about this customer."
        lines = []
        if self.name:
            lines.append(f"Customer name: {self.name}")
        if self.facts:
            lines.append("Known facts about this customer:")
            lines.extend(f"- {f}" for f in self.facts)
        return "\n".join(lines)


class MemoryStore(Protocol):
    def get(self, customer_id: str) -> CustomerMemory: ...
    def save(self, memory: CustomerMemory) -> None: ...


class JsonFileMemoryStore:
    """Persist each customer's memory as a JSON file under `directory`."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, customer_id: str) -> Path:
        # Basic sanitization so a customer_id can't escape the directory.
        safe = "".join(c for c in customer_id if c.isalnum() or c in ("-", "_"))
        return self._dir / f"{safe}.json"

    def get(self, customer_id: str) -> CustomerMemory:
        path = self._path(customer_id)
        if not path.exists():
            return CustomerMemory(customer_id=customer_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return CustomerMemory(**data)

    def save(self, memory: CustomerMemory) -> None:
        self._path(memory.customer_id).write_text(
            json.dumps(asdict(memory), indent=2), encoding="utf-8"
        )


_EXTRACTION_SYSTEM = """You extract durable facts about a customer from a support exchange.
Output ONLY facts worth remembering for future conversations: the customer's name,
their orders/products, ongoing issues, and stated preferences.
Rules:
- One fact per line, written as a short third-person statement.
- Do NOT include transient pleasantries, the agent's answers, or anything sensitive
  (card numbers, passwords, full addresses).
- If there is nothing worth remembering, output exactly: NONE"""


class MemoryExtractor:
    """Pull durable facts from a turn using the fast model."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract(self, customer_message: str, agent_reply: str) -> list[str]:
        prompt = (
            f"Customer said: {customer_message}\n"
            f"Agent replied: {agent_reply}\n\n"
            "Facts to remember:"
        )
        result = self._llm.complete(
            [user(prompt)],
            system=_EXTRACTION_SYSTEM,
            model=self._llm.settings.fast_model,  # cheap model for a cheap task
            max_tokens=200,
        )
        text = result.text.strip()
        if text.upper() == "NONE" or not text:
            return []
        return [line.lstrip("-* ").strip() for line in text.splitlines() if line.strip()]
