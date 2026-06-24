"""Phase 0 — Foundations, made concrete.

Run this after setting ANTHROPIC_API_KEY in .env:

    python examples/phase0_hello_claude.py

It walks through the five foundational ideas from docs/phase-0-foundations.md:
  1. A basic completion + what it cost.
  2. Streaming — tokens as they arrive.
  3. The model is STATELESS — prove it by forgetting, then remembering.
  4. Counting tokens before you spend them.
  5. The system prompt steers behavior (same question, different persona).

Read the doc first; then read this; then run it and change things.
"""

from __future__ import annotations

from support_agent.llm import LLMClient, assistant, user

# A first taste of the support domain: this persona is the seed of our whole agent.
SUPPORT_SYSTEM = (
    "You are a concise, friendly customer-support agent for BIP Store, an online shop. "
    "Answer in 1-2 sentences. If you don't know something, say so plainly."
)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def demo_basic_completion(llm: LLMClient) -> None:
    section("1. A basic completion — and what it cost")
    result = llm.complete(
        [user("What are your store hours?")],
        system=SUPPORT_SYSTEM,
    )
    print(f"Assistant: {result.text}")
    print(f"\n[usage] {result.usage}")
    print("Note: the model invented hours — it has no real data yet. That's exactly")
    print("the gap Phase 1 (RAG) fills: grounding answers in OUR knowledge base.")


def demo_streaming(llm: LLMClient) -> None:
    section("2. Streaming — tokens as they arrive")
    print("Assistant: ", end="", flush=True)
    for chunk in llm.stream(
        [user("Write a one-sentence apology for a late delivery.")],
        system=SUPPORT_SYSTEM,
    ):
        print(chunk, end="", flush=True)
    print()  # newline after the stream


def demo_statelessness(llm: LLMClient) -> None:
    section("3. The model is STATELESS")

    print("--- Without history (two separate calls) ---")
    llm.complete([user("My name is Rahim.")], system=SUPPORT_SYSTEM)
    forgot = llm.complete([user("What is my name?")], system=SUPPORT_SYSTEM)
    print(f"Q: What is my name?  ->  {forgot.text}")
    print("It has no idea — the second call never saw the first.")

    print("\n--- With history replayed (one conversation) ---")
    history = [
        user("My name is Rahim."),
        assistant("Nice to meet you, Rahim! How can I help with your order today?"),
        user("What is my name?"),
    ]
    remembered = llm.complete(history, system=SUPPORT_SYSTEM)
    print(f"Q: What is my name?  ->  {remembered.text}")
    print("'Memory' is just us resending the transcript. Phase 2 makes this efficient.")


def demo_token_counting(llm: LLMClient) -> None:
    section("4. Counting tokens before you spend them")
    short = [user("Hi")]
    long = [user("Tell me everything about your return policy. " * 50)]
    print(f"Short prompt tokens: {llm.count_tokens(short, system=SUPPORT_SYSTEM)}")
    print(f"Long prompt tokens:  {llm.count_tokens(long, system=SUPPORT_SYSTEM)}")
    print("This is how you answer 'will it fit?' and 'what will it cost?' up front.")


def demo_system_prompt_steers(llm: LLMClient) -> None:
    section("5. The system prompt is the program")
    question = user("A customer is angry their order is late. Respond.")

    terse = llm.complete([question], system="You are a terse support agent. One short line.")
    warm = llm.complete(
        [question],
        system="You are a warm, empathetic support agent. Acknowledge feelings first.",
    )
    print(f"Terse persona: {terse.text}")
    print(f"\nWarm persona:  {warm.text}")
    print("\nSame model, same question — behavior is set by the system prompt.")


def main() -> None:
    llm = LLMClient()
    demo_basic_completion(llm)
    demo_streaming(llm)
    demo_statelessness(llm)
    demo_token_counting(llm)
    demo_system_prompt_steers(llm)

    section("Done")
    print("Now go read docs/phase-0-foundations.md and try the exercises at the end.")


if __name__ == "__main__":
    main()
