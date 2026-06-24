"""The agent as a LangGraph state machine.

In Phase 2 a turn was a straight line: assemble -> call model -> done. That can't take
actions or make decisions. Real support needs to: look at the question, maybe call a
tool, look at the result, maybe call another, maybe ask a human to approve, then answer.
That's a LOOP with BRANCHES — a state machine. LangGraph is built to express exactly
this.

Three concepts LangGraph gives us:

- **State** — a shared object every node reads and writes (here `AgentState`). Nodes
  return partial updates that get merged in. Fields marked with a reducer (e.g.
  `Annotated[list, operator.add]`) ACCUMULATE across nodes; others overwrite.
- **Nodes** — units of work (functions). Ours: `agent` (call the model), `human_review`
  (approve sensitive tools), `tools` (execute tools).
- **Edges** — transitions. Plain edges always go A->B; **conditional edges** pick the
  next node by reading state. The conditional edge after `agent` IS the routing logic.

The graph we build (the classic "ReAct" agent loop, plus a human gate):

         ┌─────────────────────────────────────────────┐
         ▼                                               │
START → agent ──(no tools)────────────────────────────► END
          │  (sensitive tool requested)                  │
          ├──────────────► human_review ──► tools ───────┘ (loop back to agent)
          │  (safe tools only)                            ▲
          └──────────────────────────────────────────────┘ → tools

Human-in-the-loop: we gate sensitive tools behind an `approval_callback`. Here that's a
plain function call (good enough to teach the pattern and stay runnable). In production
you'd often use LangGraph's *durable* interrupt + checkpointer so the graph can pause for
hours while a real human approves in a UI, then resume exactly where it left off — same
idea, persisted across processes.

A note on safety rails: the agent loop can run away (call tools forever). LangGraph's
`recursion_limit` caps the number of steps; we set it when invoking.
"""

from __future__ import annotations

import operator
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm import LLMClient, ToolCall, Usage
from ..observability.tracing import span
from ..security.authz import AuthContext
from .tools import SENSITIVE_TOOLS, TOOL_SCHEMAS, execute_tool

# A callback that decides whether a sensitive tool call may run. Returns True to allow.
ApprovalCallback = Callable[[ToolCall], bool]


class AgentState(TypedDict):
    # The model-facing transcript for THIS turn's tool loop. Accumulates (reducer).
    messages: Annotated[list, operator.add]
    system: str                       # assembled system prompt (persona + memory + tool guidance)
    model: str                        # which model to run this turn on (Phase 7 routing decides)
    auth: AuthContext                 # the authenticated caller (Phase 9: tools authorize against this)
    tool_calls: list                  # ToolCalls the model just requested (overwrite each agent step)
    decisions: dict                   # tool_use_id -> approved? (set by human_review)
    answer: str                       # final text answer (set when the model stops calling tools)
    steps: Annotated[list, operator.add]   # human-readable trace of the path taken
    tools_called: Annotated[list, operator.add]  # names of tools actually executed (for evals)
    in_tokens: Annotated[int, operator.add]
    out_tokens: Annotated[int, operator.add]


def build_support_graph(
    llm: LLMClient,
    approval_callback: ApprovalCallback,
    *,
    tool_schemas: list[dict] | None = None,
):
    """Compile and return the support agent graph. Closures capture llm + callback."""
    tool_schemas = tool_schemas or TOOL_SCHEMAS

    # ---- Node: agent — one model step with tools available ----
    def agent_node(state: AgentState) -> dict:
        turn = llm.complete_with_tools(
            state["messages"],
            system=state["system"],
            tools=tool_schemas,
            model=state["model"],          # Phase 7: the routed model
            cache_system=True,             # Phase 7: cache the stable prefix
        )
        if turn.tool_calls:
            step = "agent: requested " + ", ".join(c.name for c in turn.tool_calls)
        else:
            step = "agent: final answer"
        update: dict = {
            "messages": [{"role": "assistant", "content": turn.raw_content}],
            "tool_calls": turn.tool_calls,
            "decisions": {},  # reset for this round
            "in_tokens": turn.usage.input_tokens,
            "out_tokens": turn.usage.output_tokens,
            "steps": [step],
        }
        if not turn.tool_calls:
            update["answer"] = turn.text
        return update

    # ---- Node: human_review — approve/deny sensitive tool calls ----
    def human_review_node(state: AgentState) -> dict:
        decisions = dict(state.get("decisions") or {})
        steps: list[str] = []
        for call in state["tool_calls"]:
            if call.name in SENSITIVE_TOOLS and call.id not in decisions:
                approved = approval_callback(call)
                decisions[call.id] = approved
                steps.append(
                    f"human_review: {call.name} -> {'APPROVED' if approved else 'DENIED'}"
                )
        return {"decisions": decisions, "steps": steps}

    # ---- Node: tools — execute requested tools (respecting approvals) ----
    def tools_node(state: AgentState) -> dict:
        decisions = state.get("decisions") or {}
        results = []
        steps: list[str] = []
        ran: list[str] = []
        for call in state["tool_calls"]:
            if decisions.get(call.id, True):  # default-allow for non-sensitive tools
                with span(f"tool.{call.name}"):
                    output = execute_tool(call.name, call.input, auth=state["auth"])
                steps.append(f"tools: ran {call.name}({call.input})")
                ran.append(call.name)
            else:
                output = (
                    "This action was DENIED by a human reviewer. Do not retry it; "
                    "explain to the customer and offer an alternative or escalate."
                )
                steps.append(f"tools: {call.name} blocked (denied)")
            # Every tool_use block MUST get a matching tool_result, or the API errors.
            results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": output}
            )
        return {
            "messages": [{"role": "user", "content": results}],
            "tool_calls": [],
            "steps": steps,
            "tools_called": ran,
        }

    # ---- Conditional edge: where to go after the agent speaks ----
    def route_after_agent(state: AgentState) -> str:
        calls = state.get("tool_calls") or []
        if not calls:
            return END  # the model produced a final answer
        if any(c.name in SENSITIVE_TOOLS for c in calls):
            return "human_review"  # at least one sensitive action -> get approval first
        return "tools"  # only safe tools -> run them directly

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "human_review": "human_review", END: END},
    )
    graph.add_edge("human_review", "tools")
    graph.add_edge("tools", "agent")  # the loop: after tools, the model speaks again

    return graph.compile()


def run_agent_turn(
    graph, *, messages: list, system: str, model: str, auth: AuthContext,
    recursion_limit: int = 16,
) -> tuple[str, Usage, list[str], list[str]]:
    """Drive a compiled graph for one customer turn on the given model.

    Returns (answer, usage, steps, tools_called). Shared by AgentSession (Phase 3 script)
    and SupportEngine (Phase 4 service) so the orchestration logic lives in exactly one
    place. `model` is whatever the Phase 7 router chose; `auth` is the authenticated caller
    that tools authorize against (Phase 9). `recursion_limit` caps total graph steps.
    """
    initial_state = {
        "messages": list(messages),
        "system": system,
        "model": model,
        "auth": auth,
        "tool_calls": [],
        "decisions": {},
        "answer": "",
        "steps": [],
        "tools_called": [],
        "in_tokens": 0,
        "out_tokens": 0,
    }
    final = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
    usage = Usage(
        model=model,
        input_tokens=final["in_tokens"],
        output_tokens=final["out_tokens"],
    )
    return (
        final["answer"] or "(no answer produced)",
        usage,
        final["steps"],
        final["tools_called"],
    )
