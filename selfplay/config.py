"""Strength preset configuration for MCTS search.

Provides ``StrengthConfig`` — a dataclass holding all knobs that govern
search strength — together with a set of built-in presets (Fast, Medium,
Strong, Turbo) and a lookup helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class StrengthConfig:
    """One knob set controlling the strength / speed trade-off of MCTS.

    Parameters:
        num_simulations:  Fixed simulation budget per search.
                          Ignored when *time_budget_ms* is set.
        c_puct:           PUCT exploration constant (default 2.5).
        batch_size:       Leaves per neural-evaluation batch (default 1).
        threat_override:  Whether to short-circuit on forced wins/blocks.
        time_budget_ms:   Wall-clock budget in milliseconds.
                          When set, *num_simulations* is treated as a
                          per-batch cap rather than a total limit.
        description:      Human-readable label (e.g. "Fast", "Turbo").
    """

    num_simulations: int = 800
    c_puct: float = 2.5
    batch_size: int = 8
    threat_override: bool = True
    time_budget_ms: float | None = None
    description: str = ""

    def to_mcts_kwargs(self) -> dict:
        """Convert to keyword args accepted by ``MCTS.__init__``."""
        return {
            "c_puct": self.c_puct,
            "num_simulations": self.num_simulations,
            "time_budget_ms": self.time_budget_ms,
            "batch_size": self.batch_size,
            "threat_override": self.threat_override,
        }

    @property
    def label(self) -> str:
        return self.description or (
            f"{self.num_simulations} sims" if self.time_budget_ms is None
            else f"{self.time_budget_ms:.0f}ms time budget"
        )

    def __repr__(self) -> str:
        if self.time_budget_ms is not None:
            return (
                f"StrengthConfig(time_budget={self.time_budget_ms:.1f}ms, "
                f"c_puct={self.c_puct}, batch={self.batch_size}, "
                f"threat={self.threat_override})"
            )
        return (
            f"StrengthConfig(sims={self.num_simulations}, "
            f"c_puct={self.c_puct}, batch={self.batch_size}, "
            f"threat={self.threat_override})"
        )


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

FAST = StrengthConfig(
    num_simulations=200,
    description="Fast",
)

MEDIUM = StrengthConfig(
    num_simulations=800,
    description="Medium",
)

STRONG = StrengthConfig(
    num_simulations=3000,
    description="Strong",
)

TURBO = StrengthConfig(
    num_simulations=3000,
    time_budget_ms=3000.0,
    description="Turbo",
)

BUILTIN_PRESETS: dict[str, StrengthConfig] = {
    "fast": FAST,
    "medium": MEDIUM,
    "strong": STRONG,
    "turbo": TURBO,
}


def get_preset(name: str) -> StrengthConfig:
    """Return the preset *name* (case-insensitive), or raise KeyError."""
    return BUILTIN_PRESETS[name.lower()]
