"""The deployable service (Phase 4).

This is where the agent becomes a real service with an HTTP API. The central design idea:
separate the STATELESS ENGINE (built once, shared by every request) from PER-CONVERSATION
STATE (loaded and saved per request from stores).

    sessions.py   SessionStore: where a conversation's short-term state lives between turns
    engine.py     SupportEngine: the shared, stateless compute (LLM, retriever, graph, ...)
                  + handle_turn(customer_id, session_id, message)
    schemas.py    request/response models for the API
    api.py        the FastAPI app (POST /chat, GET /health)

Read docs/phase-4-system-design.md alongside these files.
"""
