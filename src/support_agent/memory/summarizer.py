"""Summarization: how a long conversation stays inside a finite budget.

The problem: every turn we resend the whole history, so the prompt grows without bound.
Eventually it gets expensive (you pay for those tokens every single turn) and, far
enough out, threatens the context window. Resending 50 turns verbatim to answer turn 51
is wasteful — turns 1-46 can usually be compressed to a paragraph.

The technique: when the verbatim history exceeds a token budget, **summarize the oldest
turns into a compact running summary and drop them**, keeping only the most recent turns
verbatim. The agent still "remembers" the early conversation — just in compressed form.

    [t1 t2 t3 ... t46] [t47 t48 t49 t50]      -> over budget
            │                  │
            ▼                  ▼
       summary("...")    keep verbatim

This is exactly the idea behind the API's server-side "compaction" feature; here we do
it ourselves so the mechanics are visible. We summarize with the FAST model — it's a
cheap task that shouldn't cost Opus money.

A note on token counting: deciding "are we over budget?" every turn with the API's
exact token counter would mean a network call each time. For a budget GATE, a fast local
estimate (≈ chars / 4) is plenty; we reserve the exact counter for when precision
actually matters. Approximate where it's cheap, exact where it counts.
"""

from __future__ import annotations

from ..llm import LLMClient, Message, user
from .conversation import ConversationMemory


def estimate_tokens(messages: list[Message]) -> int:
    """Fast, free, approximate token count for budget decisions (~4 chars/token)."""
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return chars // 4 + len(messages) * 4  # small per-message overhead


_SUMMARY_SYSTEM = """You compress a customer-support conversation into a brief summary
for the agent's own reference. Capture: the customer's goal, key facts and order/product
details mentioned, decisions or commitments made, and anything still unresolved.
Be concise (a short paragraph). Write in the third person. Omit pleasantries."""


class Summarizer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def summarize(self, messages: list[Message], *, prior_summary: str | None = None) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prior = f"Summary so far:\n{prior_summary}\n\n" if prior_summary else ""
        prompt = f"{prior}Conversation to fold in:\n{transcript}\n\nUpdated summary:"
        result = self._llm.complete(
            [user(prompt)],
            system=_SUMMARY_SYSTEM,
            model=self._llm.settings.fast_model,
            max_tokens=400,
        )
        return result.text.strip()


def compact_if_needed(
    conversation: ConversationMemory,
    summarizer: Summarizer,
    *,
    budget_tokens: int,
    keep_recent: int,
) -> bool:
    """If history exceeds the budget, summarize the oldest turns and drop them.

    Returns True if compaction happened. Mutates `conversation` in place: the dropped
    turns are folded into `conversation.summary`, and `conversation.messages` is
    trimmed to the most recent `keep_recent` messages.
    """
    if estimate_tokens(conversation.messages) <= budget_tokens:
        return False

    old = conversation.messages[:-keep_recent] if keep_recent else conversation.messages
    if not old:
        return False  # nothing old enough to summarize; recent turns alone exceed budget

    conversation.summary = summarizer.summarize(old, prior_summary=conversation.summary)
    conversation.messages = conversation.messages[-keep_recent:] if keep_recent else []
    return True
