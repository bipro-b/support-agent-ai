"""Structured logging: machine-readable events, correlated by trace id.

Plain `print("retrieved 4 chunks")` is useless at scale — you can't search it, filter it,
or tie it to a specific request. **Structured logs** are JSON records with named fields, so
your log system can query them ("show all turns where cost_usd > 0.05"). Crucially, every
record carries the current `trace_id`, so you can pull every log line for one request and
line it up with its trace.

Privacy note (expanded in Phase 9): logs are a data-exfiltration and PII surface. Don't log
full prompts/answers by default — they may contain customer data. Log counts, ids, costs,
and metadata; sample or redact content deliberately if you need it.
"""

from __future__ import annotations

import json
import sys
import time
from typing import TextIO

from .tracing import current_trace_id


class StructuredLogger:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def log(self, event: str, *, level: str = "info", **fields) -> None:
        record = {
            "ts": round(time.time(), 3),
            "level": level,
            "event": event,
            "trace_id": current_trace_id(),  # ties this line to its request
            **fields,
        }
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
