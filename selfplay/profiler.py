"""Lightweight hierarchical profiler for MCTS hot-path analysis.

Usage::

    profiler = Profiler()
    with profiler.measure("search"):
        with profiler.measure("eval"):
            ...
    print(profiler.report())
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class _Timer:
    count: int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


class Profiler:
    """Accumulates wall-clock timings via context-manager scopes.

    Thread-safe within a single thread; not safe across threads.
    """

    def __init__(self) -> None:
        self._timers: dict[str, _Timer] = defaultdict(_Timer)
        self._stack: list[tuple[str, float]] = []
        self._enabled = True

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time the enclosed block, accumulating into *name*."""
        if not self._enabled:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = (time.monotonic() - start) * 1000  # ms
            t = self._timers[name]
            t.count += 1
            t.total_ms += elapsed

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a formatted table of all recorded timings."""
        if not self._timers:
            return "=== Profiler Report ===  (no data collected)"

        total_ms = max(
            self._timers.get("search.total", _Timer()).total_ms, 1e-9
        )

        header = (
            f"{'Timer':<40s}  {'Calls':>6s}  {'Total (ms)':>10s}  "
            f"{'Avg (ms)':>10s}  {'%':>6s}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        # Sort: higher total first, with "search.total" always first.
        def sort_key(item: tuple[str, _Timer]) -> tuple[float, str]:
            name, timer = item
            if name == "search.total":
                return (float("inf"), "")
            return (timer.total_ms, name)

        for name, timer in sorted(self._timers.items(), key=sort_key, reverse=True):
            pct = (timer.total_ms / total_ms) * 100
            lines.append(
                f"  {name:<38s}  {timer.count:>6d}  "
                f"{timer.total_ms:>10.2f}  "
                f"{timer.avg_ms:>10.4f}  "
                f"{pct:>5.1f}%"
            )

        lines.append(sep)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._timers.clear()

    def __repr__(self) -> str:
        return f"Profiler({len(self._timers)} timers)"
