"""Context assembly — turning stored memory into the prompt we actually send.

This is the senior skill of Phase 2. Every turn, we have several sources of context and
a finite budget. The assembler decides what goes where:

    ┌─────────────────────────── system prompt (STABLE → cacheable) ──────────────────┐
    │  persona + standing instructions                                                │
    │  long-term customer memory (name, known facts)                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────── messages (VOLATILE) ─────────────────────────────────┐
    │  recent conversation turns, verbatim                                            │
    │  current user turn, augmented with:                                             │
    │     - summary of older turns (if any)                                           │
    │     - retrieved RAG sources for THIS question                                   │
    │     - the question itself                                                       │
    └─────────────────────────────────────────────────────────────────────────────────┘

Why this split is deliberate (and a great interview answer):

- **Stable content first, volatile content last.** Prompt caching is a prefix match —
  the cached part must be byte-identical across turns. Persona + long-term memory change
  rarely, so they go in the cached system prefix. Retrieved sources and the question
  change every turn, so they go AFTER, in the messages. Put a timestamp or the question
  in the system prompt and you'd invalidate the cache every single turn.

- **Stored history ≠ the prompt.** The conversation stores the clean record; the
  assembler builds a *fresh* prompt from it each turn, injecting summary and RAG without
  mutating the record. We augment a COPY of the last user message, never the stored one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import Message
from .conversation import ConversationMemory
from .store import CustomerMemory
from .summarizer import estimate_tokens

# The agent's standing persona. This is stable across the whole session, so it lives in
# the cached system prefix. In a real product this is a large, carefully-tuned document.
PERSONA = """You are a customer-support agent for BIP Store, an online shop.
Answer using the provided sources and the conversation context.
Rules:
- If the answer is not in the sources or context, say you don't have that information
  and offer to connect the customer to a human. Do not guess.
- Use the customer's name and known facts to personalize when natural.
- Be concise and friendly (1-3 sentences). Cite source numbers you used, like [Source 2]."""


@dataclass
class AssembledContext:
    system: str
    messages: list[Message]
    estimated_tokens: int


class ContextAssembler:
    def assemble(
        self,
        *,
        customer: CustomerMemory,
        conversation: ConversationMemory,
        retrieved_context: str,
    ) -> AssembledContext:
        """Build the (system, messages) to send for the current turn.

        Assumes the current user question is already the LAST message in
        `conversation.messages` (the session adds it before calling us).
        """
        # ---- stable prefix: persona + long-term memory (cacheable) ----
        system = f"{PERSONA}\n\n[Customer profile]\n{customer.as_prompt_block()}"

        # ---- volatile body: recent turns + augmented current question ----
        messages: list[Message] = [dict(m) for m in conversation.messages]  # shallow copy
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": self._augment_question(
                    question=str(messages[-1]["content"]),
                    summary=conversation.summary,
                    retrieved_context=retrieved_context,
                ),
            }

        estimated = estimate_tokens(messages) + estimate_tokens(
            [{"role": "system", "content": system}]
        )
        return AssembledContext(system=system, messages=messages, estimated_tokens=estimated)

    @staticmethod
    def _augment_question(
        *, question: str, summary: str | None, retrieved_context: str
    ) -> str:
        parts: list[str] = []
        if summary:
            parts.append(f"[Earlier conversation summary]\n{summary}")
        if retrieved_context:
            parts.append(f"[Relevant sources]\n{retrieved_context}")
        parts.append(f"Customer question: {question}")
        return "\n\n".join(parts)
