"""Explainability tools for Gomoku AI — saliency, activations, and comparison.

Each subsystem is independently importable and testable.
No explain/ module imports from another explain/ module.
"""

from .saliency import SaliencyMap, attribution_to_grid, compute_saliency
from .activations import (
    ActivationSnapshot,
    capture_activations,
    channel_to_grid,
    select_top_channels,
)
from .comparison import (
    MoveCandidate,
    MoveComparison,
    compare_move,
    compare_move_fast,
)

__all__ = [
    # saliency
    "SaliencyMap",
    "attribution_to_grid",
    "compute_saliency",
    # activations
    "ActivationSnapshot",
    "capture_activations",
    "channel_to_grid",
    "select_top_channels",
    # comparison
    "MoveCandidate",
    "MoveComparison",
    "compare_move",
    "compare_move_fast",
]
