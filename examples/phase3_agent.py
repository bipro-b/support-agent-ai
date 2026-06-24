"""Phase 3 — The agent in action: tools, routing, and human-in-the-loop.

A scripted conversation that exercises the graph's paths:
  1. A general policy question     -> the model answers from RAG, no tools.
  2. "What's the status of order 1234?" -> the model calls lookup_order (safe, auto-run).
  3. "Return the damaged headphones" -> start_return is SENSITIVE -> human_review gate
     fires -> our approval policy approves -> the return executes.
  4. "Also return the laptop stand from 5678" -> our policy DENIES high-value returns
     -> the model is told it was denied and offers an alternative / escalates.

Each turn prints the graph's STEP TRACE so you can see exactly which path it took.

Run (needs ANTHROPIC_API_KEY; VOYAGE_API_KEY recommended for good retrieval):

    python examples/phase3_agent.py
"""

from __future__ import annotations

from support_agent.agent.agent_session import AgentSession
from support_agent.llm import ToolCall

CUSTOMER_ID = "cust_rahim"


def approval_policy(call: ToolCall) -> bool:
    """Stand-in for a human reviewer. Here: approve returns unless they're high-value.

    In production this is where you'd block on a real human decision (a Slack approval,
    a dashboard button) — LangGraph's durable interrupts let the graph wait for that.
    """
    item = str(call.input.get("item", "")).lower()
    high_value = any(word in item for word in ("laptop", "stand", "monitor"))
    decision = not high_value
    print(f"    [HUMAN REVIEW] {call.name}({call.input}) -> "
          f"{'APPROVE' if decision else 'DENY (high-value, needs manager)'}")
    return decision


TURNS = [
    "What's your return window for items I didn't like?",
    "Can you check the status of my order 1234?",
    "The wireless headphones from order 1234 arrived damaged — please start a return.",
    "Great. Also start a return for the laptop stand on order 5678, I changed my mind.",
]


def main() -> None:
    session = AgentSession(CUSTOMER_ID, approval_callback=approval_policy)

    for i, question in enumerate(TURNS, start=1):
        print(f"{'=' * 74}\nTurn {i}")
        print(f"Customer: {question}")
        result = session.ask(question)
        print(f"Agent:    {result.answer}")
        print("  graph path:")
        for step in result.steps or []:
            print(f"    - {step}")
        print(f"  {result.usage}")

    print(f"\n{'=' * 74}")
    from support_agent.agent.tools import RETURNS_LOG
    print(f"Returns actually created this run: {RETURNS_LOG}")
    print("Notice turn 4's return was DENIED by the review gate and never executed.")


if __name__ == "__main__":
    main()
