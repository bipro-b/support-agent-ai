"""The FastAPI app — the transport layer.

This layer is deliberately THIN. Its only jobs are HTTP concerns: parse and validate the
request, generate a session id if needed, call the engine, shape the response, map errors
to status codes. All the intelligence lives in the engine; the API just exposes it over
HTTP. Keeping transport and domain separate means you could put the same engine behind a
WebSocket, a queue consumer, or a CLI without touching the agent logic.

Run it:
    uvicorn support_agent.service.api:app --reload
Then POST to http://127.0.0.1:8000/chat (see examples/phase4_client.py), or open
http://127.0.0.1:8000/docs for the auto-generated API explorer.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException

from ..config import get_settings
from ..security.rate_limit import RateLimiter
from .engine import SupportEngine
from .schemas import ChatRequest, ChatResponse

app = FastAPI(title="Brecx Store Support Agent", version="0.9.0")

# The engine is a singleton built lazily on first use. We expose it through a dependency
# so tests can override it (app.dependency_overrides[get_engine] = ...) without a real
# API key or network. In production you'd build it eagerly in a lifespan handler so the
# process fails fast at startup if (say) the model API is unreachable.
_engine: SupportEngine | None = None


def get_engine() -> SupportEngine:
    global _engine
    if _engine is None:
        _engine = SupportEngine()
    return _engine


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        s = get_settings()
        _rate_limiter = RateLimiter(s.rate_limit_capacity, s.rate_limit_refill_per_sec)
    return _rate_limiter


@app.get("/health")
def health() -> dict:
    """Liveness check. Stays cheap — does NOT touch the model or build the engine."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    engine: SupportEngine = Depends(get_engine),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> ChatResponse:
    # Phase 9: throttle per customer to bound abuse and cost. 429 = back off.
    if not limiter.allow(req.customer_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")

    # First turn has no session_id — mint one and return it so the client can continue.
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    result = engine.handle_turn(
        customer_id=req.customer_id, session_id=session_id, message=req.message
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        retrieved=result.retrieved,
        steps=result.steps,
        summarized=result.summarized,
        new_facts=result.new_facts,
        cost_usd=result.usage.cost_usd,
        degraded=result.degraded,
    )
