"""Phase 2 — Prompt caching: pay for the stable prefix once, reuse it cheaply.

Prompt caching lets the API store a stable prompt prefix and serve it on later calls at
~10% of the input price. The catch: caching is a PREFIX MATCH (the cached bytes must be
identical across calls) and there's a MINIMUM cacheable size — roughly 4096 tokens for
Opus. Below that threshold, caching silently does nothing (no error, just no savings).

This demo shows BOTH realities:
  A) A small, agent-sized system prefix -> likely NO cache (below the threshold). This
     is a real production gotcha people trip over.
  B) A large stable prefix (a full policy manual inlined into the system) -> a real cache
     hit on the second call, with visible token + cost savings.

    python examples/phase2_prompt_caching.py
"""

from __future__ import annotations

from pathlib import Path

from support_agent.llm import LLMClient, user

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


def run_pair(llm: LLMClient, system: str, label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'-' * 72}")
    approx_tokens = llm.count_tokens([user("ping")], system=system)
    print(f"System prefix ≈ {approx_tokens} tokens "
          f"(Opus needs ≈4096 to cache).")

    # Call 1 — writes the cache (if the prefix is large enough).
    first = llm.complete([user("In one sentence: what is your return window?")],
                         system=system, cache_system=True)
    print(f"Call 1: {first.usage}")
    print(f"        cache_creation={first.usage.cache_creation_input_tokens} "
          f"cache_read={first.usage.cache_read_input_tokens}")

    # Call 2 — identical system prefix; should READ from cache if it was written.
    second = llm.complete([user("In one sentence: do you ship internationally?")],
                          system=system, cache_system=True)
    print(f"Call 2: {second.usage}")
    print(f"        cache_creation={second.usage.cache_creation_input_tokens} "
          f"cache_read={second.usage.cache_read_input_tokens}")

    if second.usage.cache_read_input_tokens:
        print("  -> Cache HIT on call 2: the prefix was served at ~10% price.")
    else:
        print("  -> No cache read. Prefix was below the minimum, or didn't match.")


def main() -> None:
    llm = LLMClient()

    # ---- A) small prefix: a normal short persona ----
    small_system = (
        "You are a concise support agent for BIP Store. Answer in one sentence."
    )
    run_pair(llm, small_system, "A) Small system prefix (expect NO cache)")

    # ---- B) large prefix: inline the whole policy manual into the system ----
    manual = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*.md")))
    # A real support agent often carries its full policy set in the system prompt so it
    # always has it without retrieval. We repeat it to reliably clear the ~4096-token
    # minimum for this demo; in production your genuine policy/persona text gets you there.
    big_system = (
        "You are a support agent for BIP Store. Follow this policy manual exactly.\n\n"
        + (manual + "\n\n") * 3
    )
    run_pair(llm, big_system, "B) Large system prefix (expect a cache hit on call 2)")

    print(f"\n{'=' * 72}")
    print("Takeaway: caching pays off for a LARGE, STABLE prefix reused across many calls")
    print("(a big system/policy prompt). Keep volatile content (the question, retrieved")
    print("sources, timestamps) AFTER the cached prefix, or you invalidate it every turn.")


if __name__ == "__main__":
    main()
