"""Context & memory management (Phase 2).

The model is stateless and its context window is finite. Everything here exists to
assemble the RIGHT context for each turn without blowing the token budget:

    conversation.py  short-term memory: the running message list for one session
    store.py         long-term memory: per-customer facts that persist across sessions
    summarizer.py    compress old turns into a summary when history grows too long
    context.py       budget-aware assembly: persona + memory + summary + RAG + question
    session.py       a thin orchestrator that wires it all together (uses Phase 1 RAG)

Read docs/phase-2-context-memory.md alongside these files.
"""
