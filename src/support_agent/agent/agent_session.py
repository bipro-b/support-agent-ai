"""AgentSession — Phase 2's SupportSession with the LangGraph engine swapped in.

This is the payoff of making `_generate` an overridable hook in SupportSession. Memory,
retrieval, summarization, context assembly, fact extraction — all of Phase 1+2 — stay
exactly the same. The ONLY thing that changes is HOW the answer is produced: instead of
one model call, we run the agent graph (tool loop + human-in-the-loop) over the assembled
context.

That's a clean illustration of good system design: orchestration is a strategy you plug
into a stable turn lifecycle, not a rewrite of everything around it.

The agent needs to know it has tools (the base persona was written for plain RAG), so we
append short tool guidance to the assembled system prompt here.
"""

from __future__ import annotations

from ..memory.context import AssembledContext
from ..memory.session import GenerationResult, SupportSession
from ..security.authz import AuthContext
from .graph import ApprovalCallback, build_support_graph, run_agent_turn

TOOL_GUIDANCE = """

You have tools available:
- lookup_order: get the status and items of an order by ID.
- start_return: start a return for an item (this requires human approval before it runs).
- escalate_to_human: hand off to a human agent.
Use a tool when the customer references a specific order or wants to take an action.
Prefer answering from the provided sources for general policy questions."""


def auto_approve(_call) -> bool:
    """Default approval policy: allow everything. Replace with real review logic."""
    return True


class AgentSession(SupportSession):
    def __init__(
        self,
        customer_id: str,
        *,
        approval_callback: ApprovalCallback | None = None,
        **kwargs,
    ) -> None:
        super().__init__(customer_id, **kwargs)
        self.graph = build_support_graph(self.llm, approval_callback or auto_approve)

    def _generate(self, assembled: AssembledContext) -> GenerationResult:
        # The script path doesn't route; it always uses the primary model. Tools authorize
        # against the session's own customer.
        answer, usage, steps, tools_called = run_agent_turn(
            self.graph,
            messages=list(assembled.messages),
            system=assembled.system + TOOL_GUIDANCE,
            model=self.settings.primary_model,
            auth=AuthContext(self.customer.customer_id),
        )
        return GenerationResult(answer, usage, steps, tools_called)
