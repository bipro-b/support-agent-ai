"""Tools: the actions the agent can take, plus the fake backend they act on.

A "tool" is a function the model can ask us to run. We describe each tool to the model
with a JSON schema (name, description, inputs); the model decides when to call one and
with what arguments; WE execute it and feed the result back. The model never runs code
— it emits a request, our code does the work. That boundary is the whole safety model
of agents (Phase 9 leans on it hard).

Two categories matter for orchestration:

- **Read-only tools** (e.g. `lookup_order`) are safe to auto-execute.
- **Sensitive tools** (e.g. `start_return`) have side effects — they change state, cost
  money, or are hard to undo. These should pause for human approval before running.
  We mark them in SENSITIVE_TOOLS, and the graph routes them through a review gate.

`design your tool surface` is itself a senior skill: give the model a few clear,
well-described, appropriately-scoped tools — not one giant "do_anything" tool, and not
fifty overlapping ones. Descriptions are prompt engineering: they decide whether the
model calls the right tool at the right time.
"""

from __future__ import annotations

from ..observability.tracing import set_attribute
from ..security.authz import AuthContext, AuthorizationError

# --------------------------------------------------------------------------- #
# Fake backend — stands in for the order-management system an agent would call #
# --------------------------------------------------------------------------- #
#
# Each order has an OWNER (`customer_id`). Authorization checks compare this against the
# authenticated caller — never against what the model passed. Order 9999 belongs to a
# DIFFERENT customer, so any attempt by cust_rahim to read it must be denied (Phase 9).
_ORDERS: dict[str, dict] = {
    "1234": {
        "customer_id": "cust_rahim",
        "customer": "Rahim",
        "status": "delivered",
        "placed": "2026-05-28",
        "delivered": "2026-06-03",
        "items": [
            {"name": "Wireless headphones", "price": 79.99, "condition": "damaged"},
            {"name": "Phone case", "price": 14.99, "condition": "ok"},
        ],
    },
    "5678": {
        "customer_id": "cust_rahim",
        "customer": "Rahim",
        "status": "in transit",
        "placed": "2026-06-10",
        "delivered": None,
        "items": [{"name": "Laptop stand", "price": 249.00, "condition": "ok"}],
    },
    "9999": {
        "customer_id": "cust_other",
        "customer": "Sara",
        "status": "delivered",
        "placed": "2026-06-01",
        "delivered": "2026-06-05",
        "items": [{"name": "Smart watch", "price": 199.00, "condition": "ok"}],
    },
}

# Side-effect log so the demo can show that sensitive actions actually "did" something.
RETURNS_LOG: list[dict] = []


# --------------------------------------------------------------------------- #
# Tool implementations (plain Python — the model never runs these directly)   #
# --------------------------------------------------------------------------- #
def _authorize_order(order_id: str, auth: AuthContext) -> dict:
    """Fetch an order, but ONLY if it belongs to the authenticated caller."""
    order = _ORDERS.get(order_id.strip().lstrip("#"))
    if not order:
        raise KeyError(order_id)  # caller maps this to a friendly "not found"
    if order["customer_id"] != auth.customer_id:
        # The model was asked (perhaps via injection) to touch someone else's order.
        raise AuthorizationError(
            f"{auth.customer_id} attempted to access order {order_id} "
            f"owned by {order['customer_id']}"
        )
    return order


def lookup_order(order_id: str, *, auth: AuthContext) -> str:
    try:
        order = _authorize_order(order_id, auth)
    except KeyError:
        return f"No order found with ID {order_id}."
    items = "; ".join(f"{i['name']} (${i['price']}, condition: {i['condition']})"
                      for i in order["items"])
    delivered = order["delivered"] or "not yet delivered"
    return (
        f"Order {order_id}: status={order['status']}, placed {order['placed']}, "
        f"delivered {delivered}. Items: {items}."
    )


def start_return(order_id: str, item: str, reason: str, *, auth: AuthContext) -> str:
    try:
        _authorize_order(order_id, auth)  # can't start a return on someone else's order
    except KeyError:
        return f"Cannot start a return: no order {order_id}."

    # IDEMPOTENCY (Phase 8): a return is a side effect, and retries happen (network
    # blips, the agent loop re-calling). If a return for this (order, item) already
    # exists, return the SAME authorization instead of creating a duplicate. Without
    # this, a retry would refund/ship-label the customer twice. Real systems key this
    # on an explicit idempotency token; (order_id, item) is a simple stand-in.
    for existing in RETURNS_LOG:
        if existing["order_id"] == order_id and existing["item"] == item:
            return (
                f"A return for '{item}' on order {order_id} is already in progress "
                f"(authorization {existing['rma']}). No duplicate was created."
            )

    rma = f"RMA-{len(RETURNS_LOG) + 1001}"
    RETURNS_LOG.append({"order_id": order_id, "item": item, "reason": reason, "rma": rma})
    return (
        f"Return started for '{item}' on order {order_id} (reason: {reason}). "
        f"Return authorization {rma}. A prepaid label has been emailed."
    )


def escalate_to_human(reason: str) -> str:
    return (
        f"Escalated to a human support agent (reason: {reason}). "
        "A specialist will follow up by email within one business day."
    )


# --------------------------------------------------------------------------- #
# Tool schemas (what the model sees) + the registry that wires it together    #
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "lookup_order",
        "description": (
            "Look up the status, dates, and items of a customer's order by its ID. "
            "Use this whenever the customer references a specific order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. 1234"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "start_return",
        "description": (
            "Start a return for a specific item on an order. This is a real action that "
            "creates a return authorization and emails a label. Use only when the "
            "customer clearly wants to return a specific item."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item": {"type": "string", "description": "The item being returned"},
                "reason": {"type": "string", "description": "Why it's being returned"},
            },
            "required": ["order_id", "item", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a human agent. Use when the customer asks for a "
            "human, is very upset, or the request is outside your tools and knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why escalation is needed"}
            },
            "required": ["reason"],
        },
    },
]

_TOOL_FUNCS = {
    "lookup_order": lookup_order,
    "start_return": start_return,
    "escalate_to_human": escalate_to_human,
}

# Tools that change state / are hard to undo -> require human approval before running.
SENSITIVE_TOOLS = {"start_return"}

# Tools that access customer data -> must run under the caller's AuthContext (Phase 9).
_AUTHZ_TOOLS = {"lookup_order", "start_return"}


def execute_tool(name: str, tool_input: dict, *, auth: AuthContext) -> str:
    """Run a tool by name under the caller's auth. Returns what the model sees back."""
    func = _TOOL_FUNCS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'."
    try:
        if name in _AUTHZ_TOOLS:
            return func(**tool_input, auth=auth)
        return func(**tool_input)
    except AuthorizationError:
        # Authorization is enforced here, in code — not by trusting the model. Record the
        # attempt as a security signal (Phase 6) and return a safe, non-revealing message.
        set_attribute(security_event="authz_denied", denied_tool=name)
        return (
            "I can only access information and actions for your own account, and that "
            "request isn't for your account, so I can't help with it."
        )
    except TypeError as exc:
        # Bad/missing arguments — tell the model so it can correct itself.
        return f"Error calling {name}: {exc}"
