"""Authorization: the load-bearing security control for an agent.

The scenario that keeps you up at night: a customer (or an attacker who took over a chat, or
a prompt injection hidden in a document) gets the agent to call `lookup_order(9999)` — and
9999 belongs to a DIFFERENT customer. If the tool just trusts the order id the model passed,
you've leaked another customer's data. This is the agent version of an IDOR / broken
object-level authorization bug, and it's the most damaging because the model can be talked
into requesting anything.

The fix is a principle, not a prompt: **authorize at the tool boundary against the
AUTHENTICATED principal — never against what the model passed.** The `AuthContext` carries
who the request is really for (established by your auth layer, NOT by the conversation). Every
tool that touches customer data checks the resource belongs to that principal, and refuses
otherwise. The model can ask for anything; the tool decides what's allowed.

This is exactly why it lives in deterministic Python, outside the model's influence. No amount
of prompt injection can edit an `if order.owner != auth.customer_id` check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    """Who the request is authenticated as. Set by the auth layer, not the model."""

    customer_id: str


class AuthorizationError(Exception):
    """Raised when a tool is asked to act on a resource the principal doesn't own."""
