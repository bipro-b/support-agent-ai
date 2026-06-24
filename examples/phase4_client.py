"""Phase 4 — Talk to the support agent over HTTP.

This shows the agent as a real service: a client that holds NO agent state, just a
session_id, and has a multi-turn conversation over HTTP. The conversation state lives on
the server (in the session store), reloaded each turn by the session_id.

First, start the server in one terminal:
    uvicorn support_agent.service.api:app --reload

Then run this client in another:
    python examples/phase4_client.py

(Server needs ANTHROPIC_API_KEY; VOYAGE_API_KEY recommended. The first /chat call is slow
because the engine — including indexing the knowledge base — is built on first request.)
"""

from __future__ import annotations

import httpx

BASE_URL = "http://127.0.0.1:8000"
CUSTOMER_ID = "cust_rahim"

TURNS = [
    "Hi, what's your return window?",
    "Can you check the status of my order 1234?",
    "What was the order number I just asked about?",  # tests server-side memory
]


def main() -> None:
    session_id: str | None = None  # the client's ONLY piece of state

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        print("health:", client.get("/health").json())
        for i, message in enumerate(TURNS, start=1):
            payload = {"customer_id": CUSTOMER_ID, "message": message}
            if session_id:
                payload["session_id"] = session_id

            resp = client.post("/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            session_id = data["session_id"]  # reuse it next turn

            print(f"\n{'=' * 70}\nTurn {i}  (session {session_id})")
            print(f"Customer: {message}")
            print(f"Agent:    {data['answer']}")
            print(f"  retrieved: {data['retrieved']}  cost=${data['cost_usd']:.5f}")
            if data.get("new_facts"):
                print(f"  new facts remembered: {data['new_facts']}")

    print(f"\n{'=' * 70}")
    print("The client kept only a session_id. All conversation + customer state lived")
    print("on the server — exactly how a real chat service is structured.")


if __name__ == "__main__":
    main()
