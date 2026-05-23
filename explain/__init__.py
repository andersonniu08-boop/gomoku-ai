"""Explainability tools for Gomoku AI — saliency, activations, and comparison.

Each subsystem is independently importable and testable.
No explain/ module imports from another explain/ module.
"""

# Each subsystem is imported lazily so the package remains importable
# during parallel development before all modules exist.
# Once all three workstreams have landed, simplify to direct imports.

try:
    from .saliency import SaliencyMap, attribution_to_grid, compute_saliency
except ImportError:
    SaliencyMap = None  # type: ignore
    attribution_to_grid = None  # type: ignore
    compute_saliency = None  # type: ignore

try:
    from .activations import (
        ActivationSnapshot,
        capture_activations,
        channel_to_grid,
        select_top_channels,
    )
except ImportError:
    ActivationSnapshot = None  # type: ignore
    capture_activations = None  # type: ignore
    channel_to_grid = None  # type: ignore
    select_top_channels = None  # type: ignore

try:
    from .comparison import (
        MoveCandidate,
        MoveComparison,
        compare_move,
        compare_move_fast,
    )
except ImportError:
    MoveCandidate = None  # type: ignore
    MoveComparison = None  # type: ignore
    compare_move = None  # type: ignore
    compare_move_fast = None  # type: ignore
