"""Agentic orchestration (Phase 3).

Turns the straight-line turn of Phase 2 into a graph that can take ACTIONS and make
DECISIONS:

    tools.py          the actions the agent can take (look up an order, start a return,
                      escalate) plus a fake backend and the Anthropic tool schemas
    graph.py          a LangGraph state machine: agent <-> tools loop, conditional
                      routing, and a human-in-the-loop gate on sensitive actions
    agent_session.py  AgentSession — Phase 2's SupportSession with the graph swapped
                      in at the generation step (memory + RAG still apply)

Read docs/phase-3-orchestration.md alongside these files.
"""
