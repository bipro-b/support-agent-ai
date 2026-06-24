"""Tracing: a timed, nested tree of one request's steps.

A **trace** is one request (one customer turn). A **span** is one step within it — retrieve,
assemble, an individual model call, a tool execution — with a duration and free-form
attributes (tokens, cost, chunk count). Spans nest: a model call is a child of "generate",
which is a child of the turn. The result is a tree you can read top-to-bottom to see exactly
where the time and money went. This is the OpenTelemetry mental model, built small.

The clever part is **implicit context propagation**. Instead of passing a tracer object
through every function signature (invasive, ugly), we keep the "current span" in a
`contextvars.ContextVar`. Any code that opens `with span("x"):` automatically nests under
whatever span is active on this execution context — the LLM client, a graph node, the
engine, none of them need to know about each other. That's how real tracing libraries avoid
polluting every function with a `tracer` argument.

If no trace is active, `span()` still works — it just isn't attached to a tree (near-zero
overhead). So instrumentation is always on; collection only happens inside `start_trace`.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Span:
    name: str
    attributes: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    children: list["Span"] = field(default_factory=list)
    _start: float = 0.0

    def set(self, **attrs) -> "Span":
        """Attach attributes to this span (tokens, cost, counts, ...)."""
        self.attributes.update(attrs)
        return self


# The current span and trace id travel on the execution context, not in arguments.
_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)


def current_trace_id() -> str | None:
    """The active trace id (for log correlation), or None outside a trace."""
    return _trace_id.get()


@contextmanager
def start_trace(name: str, **attrs) -> Iterator[Span]:
    """Begin a trace. The yielded root Span becomes the tree for this request."""
    trace_id = uuid.uuid4().hex[:12]
    root = Span(name=name, attributes={"trace_id": trace_id, **attrs})
    root._start = time.perf_counter()
    span_token = _current.set(root)
    tid_token = _trace_id.set(trace_id)
    try:
        yield root
    finally:
        root.duration_ms = (time.perf_counter() - root._start) * 1000
        _current.reset(span_token)
        _trace_id.reset(tid_token)


@contextmanager
def span(name: str, **attrs) -> Iterator[Span]:
    """Open a child span under whatever span is currently active."""
    parent = _current.get()
    s = Span(name=name, attributes=dict(attrs))
    s._start = time.perf_counter()
    if parent is not None:
        parent.children.append(s)
    token = _current.set(s)
    try:
        yield s
    finally:
        s.duration_ms = (time.perf_counter() - s._start) * 1000
        _current.reset(token)


def set_attribute(**attrs) -> None:
    """Set attributes on the currently-active span, if any."""
    s = _current.get()
    if s is not None:
        s.set(**attrs)


# --------------------------------------------------------------------------- #
# Reading a trace back out                                                    #
# --------------------------------------------------------------------------- #
def render_trace(root: Span, indent: int = 0) -> str:
    """Pretty-print a span tree as an indented, timed outline."""
    pad = "  " * indent
    attrs = "  ".join(
        f"{k}={v}" for k, v in root.attributes.items() if k != "trace_id"
    )
    line = f"{pad}{root.name}  [{root.duration_ms:.0f}ms]"
    if attrs:
        line += f"  {attrs}"
    lines = [line]
    for child in root.children:
        lines.append(render_trace(child, indent + 1))
    return "\n".join(lines)


def span_to_dict(s: Span) -> dict:
    """JSON-serializable form — what you'd ship to a tracing backend."""
    return {
        "name": s.name,
        "duration_ms": round(s.duration_ms, 1),
        "attributes": s.attributes,
        "children": [span_to_dict(c) for c in s.children],
    }


def summarize_trace(root: Span) -> dict:
    """Roll a trace up into the per-turn metrics that feed dashboards/alerts."""
    agg = {
        "trace_id": root.attributes.get("trace_id"),
        "latency_ms": round(root.duration_ms, 1),
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_calls": 0,
        "tool_calls": 0,
    }

    def walk(s: Span) -> None:
        if s.name.startswith("llm."):
            agg["llm_calls"] += 1
        if s.name.startswith("tool."):
            agg["tool_calls"] += 1
        agg["cost_usd"] += float(s.attributes.get("cost_usd", 0) or 0)
        agg["input_tokens"] += int(s.attributes.get("input_tokens", 0) or 0)
        agg["output_tokens"] += int(s.attributes.get("output_tokens", 0) or 0)
        for c in s.children:
            walk(c)

    walk(root)
    agg["cost_usd"] = round(agg["cost_usd"], 6)
    return agg
