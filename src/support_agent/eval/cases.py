"""The eval dataset: labeled cases describing expected agent behavior.

An eval is only as good as its dataset. Each case states what we expect, so a check can be
objective. Building this dataset is real work — and it's the asset that lets you change the
prompt, model, or retrieval and *know* whether quality held. Curate it from real (or
realistic) customer questions, including the hard and adversarial ones.

A case can exercise several dimensions at once:
  - `reference`      key facts a correct answer must reflect (drives the answer-quality judge)
  - `expected_tools` tools the agent SHOULD call (drives the deterministic tool-use check)
  - `should_refuse`  True for out-of-scope questions the agent must decline (not guess)
  - `check_grounded` whether to run the groundedness judge against the retrieved sources
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    reference: str | None = None          # key facts a good answer reflects
    expected_tools: list[str] = field(default_factory=list)
    should_refuse: bool = False
    check_grounded: bool = False

    @property
    def dimensions(self) -> list[str]:
        """Which checks apply to this case."""
        dims: list[str] = []
        if self.should_refuse:
            dims.append("refusal")
        else:
            if self.reference:
                dims.append("answer_quality")
            if self.check_grounded:
                dims.append("groundedness")
        if self.expected_tools:
            dims.append("tool_use")
        return dims


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**d) for d in data]
