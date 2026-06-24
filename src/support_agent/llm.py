"""A thin, typed wrapper around the Anthropic Claude client.

This is the single doorway through which the rest of the system talks to the model.
Every later phase (RAG, memory, orchestration, evals, observability) calls *through*
this module rather than touching the raw SDK. Concentrating model access in one place
is what lets us, later, add caching, tracing, cost accounting, and retries in ONE spot
instead of scattered across the codebase.

For Phase 0 it stays deliberately small: complete a turn, stream a turn, count tokens,
and report cost. Read docs/phase-0-foundations.md alongside this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import anthropic

from .observability.tracing import span

from .config import Settings, get_settings

# A "message" in the Messages API is {"role": "user"|"assistant", "content": ...}.
# The API is STATELESS: it has no memory of past calls. To hold a conversation you
# resend the whole history every turn. This list IS the conversation.
Role = Literal["user", "assistant"]
Message = dict[str, object]

# Per-1M-token prices (USD). Source: the model catalog. We keep this here so cost
# accounting lives next to the client that incurs it. In a real system this would be
# data, refreshed from a pricing source — hardcoding it is fine for learning.
_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_$_per_1M, output_$_per_1M)
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class Usage:
    """Token usage + derived cost for a single model call.

    Cache fields matter from Phase 2 onward (prompt caching). For now they're
    almost always zero — but we surface them so the cost number is honest.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        in_price, out_price = _PRICING.get(self.model, (0.0, 0.0))
        # Cache reads are ~0.1x input price; cache writes ~1.25x. We approximate.
        billed_input = self.input_tokens + self.cache_read_input_tokens * 0.1
        billed_input += self.cache_creation_input_tokens * 1.25
        return (billed_input * in_price + self.output_tokens * out_price) / 1_000_000

    def __str__(self) -> str:
        return (
            f"{self.model}: in={self.input_tokens} out={self.output_tokens} "
            f"cost=${self.cost_usd:.5f}"
        )


@dataclass
class Completion:
    """The result of a (non-streaming) model call: the text plus what it cost."""

    text: str
    usage: Usage


@dataclass
class ToolCall:
    """A request from the model to run one of our tools (Phase 3)."""

    id: str            # the tool_use id; every call MUST get a matching tool_result
    name: str
    input: dict


@dataclass
class AgentTurn:
    """One turn of a tool-using agent: text, any tool calls, and the raw blocks.

    `raw_content` is the assistant's content blocks exactly as returned; we append it
    back into the message history so the model sees its own prior tool_use blocks on
    the next loop iteration (the API requires this).
    """

    text: str
    tool_calls: list[ToolCall]
    raw_content: list[object]   # assistant content blocks, appended to history verbatim
    stop_reason: str | None
    usage: Usage


class LLMClient:
    """The system's one and only entry point to Claude."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        # The SDK reads ANTHROPIC_API_KEY from the environment, but we pass it
        # explicitly so behavior doesn't depend on import-time env state.
        # timeout + max_retries (Phase 8): the SDK retries transient errors (429/5xx/
        # timeouts) with its own exponential backoff before raising.
        self._client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key,
            timeout=self.settings.request_timeout_s,
            max_retries=self.settings.max_retries,
        )

    # ------------------------------------------------------------------ #
    # Core: complete one turn                                            #
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
        cache_system: bool = False,
    ) -> Completion:
        """Send the conversation and return the assistant's reply + usage.

        `messages` is the FULL history every time — the API is stateless.
        `thinking=True` turns on adaptive thinking (the model decides how much to
        reason before answering). We default it OFF here because a simple Q&A
        doesn't need it and thinking costs tokens; we'll turn it on deliberately
        when a task is genuinely hard.

        `cache_system=True` marks the system prompt as cacheable (Phase 2). On
        repeated calls with a byte-identical system prefix, the API serves it from
        cache at ~10% of the input price. See docs/phase-2-context-memory.md.
        """
        model = model or self.settings.primary_model
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "messages": messages,
        }
        if system is not None:
            if cache_system:
                # A list of content blocks lets us attach cache_control. The marker
                # caches everything up to and including this block (the stable prefix).
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system
        if thinking:
            # Adaptive thinking is the modern mode on Opus 4.x — no fixed token budget.
            kwargs["thinking"] = {"type": "adaptive"}

        # One span per model call: this is where the latency and cost actually live, so
        # it's the most important thing to see in a trace.
        with span("llm.complete", model=model) as sp:
            response = self._client.messages.create(**kwargs)

            # response.content is a LIST of blocks (text, thinking, tool_use, ...).
            # We pull out only the text. Always check .type — never assume content[0].
            text = "".join(b.text for b in response.content if b.type == "text")

            usage = Usage(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
            )
            sp.set(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=round(usage.cost_usd, 6),
            )
        return Completion(text=text, usage=usage)

    # ------------------------------------------------------------------ #
    # Tool use: one turn of an agent loop (Phase 3)                      #
    # ------------------------------------------------------------------ #
    def complete_with_tools(
        self,
        messages: list[Message],
        *,
        tools: list[dict],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = False,
    ) -> AgentTurn:
        """Run ONE turn with tools available.

        The model either answers (stop_reason "end_turn") or asks to call one or more
        tools (stop_reason "tool_use"). This method does NOT run the loop — the graph
        (agent/graph.py) does, because the loop is exactly the control flow we want to
        make explicit and orchestrate. Here we just expose one model step.

        `cache_system=True` caches the (large, stable) system prefix — a real win in the
        agent loop, where the same persona+tools prefix is resent on every iteration and
        across turns. See Phase 7.
        """
        model = model or self.settings.primary_model
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "messages": messages,
            "tools": tools,
        }
        if system is not None:
            if cache_system:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                kwargs["system"] = system

        with span("llm.complete_with_tools", model=model) as sp:
            response = self._client.messages.create(**kwargs)
            text = "".join(b.text for b in response.content if b.type == "text")
            tool_calls = [
                ToolCall(id=b.id, name=b.name, input=dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ]
            usage = Usage(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            sp.set(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=round(usage.cost_usd, 6),
                requested_tools=[c.name for c in tool_calls],
            )
        return AgentTurn(
            text=text,
            tool_calls=tool_calls,
            raw_content=response.content,
            stop_reason=response.stop_reason,
            usage=usage,
        )

    # ------------------------------------------------------------------ #
    # Streaming: same call, tokens as they arrive                        #
    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield text chunks as the model produces them.

        Why streaming matters in production:
          - Perceived latency: the user sees words immediately instead of waiting
            for the whole answer.
          - Timeouts: long non-streaming responses can exceed HTTP idle timeouts;
            streaming keeps the connection alive.
        """
        model = model or self.settings.primary_model
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                yield chunk

    # ------------------------------------------------------------------ #
    # Token counting: know the cost BEFORE you pay it                    #
    # ------------------------------------------------------------------ #
    def count_tokens(self, messages: list[Message], *, system: str | None = None,
                     model: str | None = None) -> int:
        """Return the input-token count for a prompt without running it.

        This is the right tool to answer "will this fit in the context window?"
        and "what will this prompt cost?". Do NOT use tiktoken — it's OpenAI's
        tokenizer and is wrong for Claude.
        """
        model = model or self.settings.primary_model
        kwargs: dict[str, object] = {"model": model, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        return self._client.messages.count_tokens(**kwargs).input_tokens


def user(content: str) -> Message:
    """Helper: build a user message."""
    return {"role": "user", "content": content}


def assistant(content: str) -> Message:
    """Helper: build an assistant message (used when replaying history)."""
    return {"role": "assistant", "content": content}
