from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StageTimers:
    """Lightweight hierarchical timer collector."""

    values: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(seconds)

    def time_block(self, name: str):
        return _TimerContext(self, name)


class _TimerContext:
    def __init__(self, timers: StageTimers, name: str):
        self.timers = timers
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timers.add(self.name, time.perf_counter() - self.t0)
        return False
