"""Phase 2 — Memory in action: a multi-turn conversation that remembers.

This runs a single customer through a multi-turn support chat and shows, turn by turn:
  - short-term memory working (it recalls what was said earlier in THIS chat)
  - long-term memory being written (durable facts extracted and persisted to disk)
  - summarization triggering once history exceeds the token budget
  - the system prefix being cached (watch cache_read tokens once a session is warm)

Run it TWICE. The second run, the agent already knows the customer from disk
(data/memory/<id>.json) — that's long-term memory surviving across sessions.

    python examples/phase2_memory.py

We deliberately set a SMALL context budget below so summarization triggers within a
short demo. In production the budget is far larger; the mechanism is identical.
"""

from __future__ import annotations

from support_agent.config import get_settings
from support_agent.memory.session import SupportSession

CUSTOMER_ID = "cust_rahim"

# A scripted conversation. Early turns establish facts; later turns test recall and
# push the history over the budget so we can watch summarization kick in.
TURNS = [
    "Hi, my name is Rahim. My order #1234 arrived late and one item was damaged.",
    "How do I return the damaged item, and will I pay for return shipping?",
    "How long until I get the refund back on my card?",
    "Also, what payment methods do you accept? I might reorder.",
    "Do you ship internationally, and how long does that take?",
    "What was my order number again, and what was wrong with it?",  # tests memory
]


def main() -> None:
    # Shrink the budget for the demo so compaction happens within a few turns.
    settings = get_settings()
    settings.context_budget_tokens = 250

    session = SupportSession(CUSTOMER_ID, settings=settings)

    print(f"Customer profile loaded from disk: {session.customer.as_prompt_block()}")
    print("(Run this script again later and it will already know Rahim.)\n")

    total_cost = 0.0
    for i, question in enumerate(TURNS, start=1):
        result = session.ask(question)
        total_cost += result.usage.cost_usd

        print(f"{'=' * 72}\nTurn {i}")
        print(f"Customer: {question}")
        print(f"Agent:    {result.answer}")
        print(f"  retrieved: {result.retrieved}")
        if result.summarized:
            print("  >>> history exceeded budget -> older turns SUMMARIZED")
        if result.new_facts:
            print(f"  >>> new long-term facts: {result.new_facts}")
        print(f"  history now ~{session.history_tokens()} tokens | {result.usage}")
        if result.usage.cache_read_input_tokens:
            print(f"  cache hit: {result.usage.cache_read_input_tokens} tokens served from cache")

    print(f"\n{'=' * 72}")
    print(f"Total cost for {len(TURNS)} turns: ${total_cost:.5f}")
    if session.conversation.summary:
        print(f"\nRolling summary of older turns:\n{session.conversation.summary}")
    print(f"\nFinal long-term memory for {CUSTOMER_ID}:")
    print(session.customer.as_prompt_block())


if __name__ == "__main__":
    main()
