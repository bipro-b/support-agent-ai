"""Security (Phase 9).

LLM agents have a threat surface normal apps don't. The big ones (OWASP LLM Top 10):

  - **Prompt injection** — a user (direct) or a retrieved document/tool result (indirect)
    smuggles instructions that try to override the agent's rules. THE defining LLM
    vulnerability, and not fully solved by any prompt trick.
  - **Excessive agency / broken authorization** — the agent acts on data or takes actions it
    shouldn't, e.g. fetching another customer's order. For an agent, this is the highest-impact
    bug, and it's the one with a real fix.
  - **Sensitive-data disclosure** — leaking PII or secrets in answers or logs.
  - **Abuse** — flooding the endpoint to run up cost or deny service.

    authz.py        AuthContext + tool authorization (enforce own-data access at the boundary)
    guardrails.py   input injection screening + PII redaction + the security preamble
    rate_limit.py   token-bucket throttling per principal

The throughline: **never trust model output or retrieved content to enforce security.** Put
the real controls in deterministic code (authz checks, rate limits) around the model. Read
docs/phase-9-security.md.
"""
