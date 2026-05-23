# Workstream A — Saliency Maps

## Project Overview

You are implementing one of three parallel workstreams for the Explainability
phase of a Gomoku AI engine. This is a modular research project following the
AlphaZero paradigm. The codebase is at `/home/anderson/projects/gomoku-ai`.

## Full project instructions (CLAUDE.md)

The entire project specification is in `/home/anderson/projects/gomoku-ai/CLAUDE.md`.
Read it before starting any implementation — it contains strict rules about
modularity, imports, naming, type hints, coding standards, and testing.

Key rules:
- Imports flow one direction: `engine` <- `neural` <- `selfplay` <- `explain`
- No PyTorch in `engine/`. No game logic in `neural/`.
- All public functions/methods must have type hints and docstrings.
- No file shall exceed ~500 lines.
- Use `Optional[X]` not `X | None`.
- Use `@dataclass(slots=True)` for data containers.
- Tests use pytest, live in `tests/`, named `test_<module>.py`.

## What you are building

You own **saliency maps** — gradient-based input attribution for GomokuNet.
Given a board position and a trained model, produce a 15×15 heatmap showing
which cells most influenced the network's output. Uses Integrated Gradients
as the primary method, vanilla gradient as a fast-path option.

Also read the spec at `docs/superpowers/specs/2026-05-23-explainability-design.md`
for the full design context.

## Pre-work already done (do NOT redo)

A method `evaluate_raw(board)` has been added to `GomokuInferenceWrapper` in
`neural/wrapper.py`. It returns `(log_policy, value)` raw tensors without
`torch.no_grad()`. Use this for gradient-based computations.

## Files you will create

### `explain/saliency.py`

Public API:

```python
@dataclass(slots=True)
class SaliencyMap:
    grid: NDArray[np.float32]       # shape (15, 15), values [0, 1]
    method: str                      # "integrated_gradients" | "vanilla"
    target: str                      # e.g. "value" | "policy" | "policy_move(7,3)"
    n_steps: int | None              # None for vanilla

def compute_saliency(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    *,
    method: str = "ig",
    target: str = "value",
    n_steps: int = 50,
) -> SaliencyMap:
    """Compute a saliency map for the given board."""

def attribution_to_grid(
    raw_gradients: torch.Tensor,
) -> NDArray[np.float32]:
    """Convert (3, 15, 15) gradient tensor to (15, 15) heatmap.
    Max-pool across channels, take abs, normalize to [0, 1]."""
```

You'll also need internal helpers:

```python
def _make_target_fn(log_policy, value, target) -> torch.Tensor:
    """Return scalar tensor for gradient computation.
    - "value" -> value[0, 0]
    - "policy" -> log_policy[0, :].sum()
    - "policy_move(r,c)" -> log_policy[0, r * 15 + c]
    """

def _compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_fn: callable,
    n_steps: int,
) -> torch.Tensor:
    """Integrated Gradients: interpolate baseline->input, average gradients."""

def _compute_vanilla_gradient(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_fn: callable,
) -> torch.Tensor:
    """Single forward+backward gradient."""
```

### `tests/test_saliency.py`

Test the following invariants (all testable with random weights):

1. **Completeness axiom:** sum of IG attributions ≈ model_output(input) -
   model_output(baseline), within 1e-3 tolerance. This is the math invariant.
2. **Output shape:** `SaliencyMap.grid.shape == (15, 15)`.
3. **Output range:** all values in [0, 1].
4. **Method selection:** both "ig" and "vanilla" work and return valid maps.
5. **Target modes:** "value", "policy", and "policy_move(r,c)" all work.
6. **Empty board:** no crash, no NaN.
7. **Symmetry:** flipping input horizontally flips saliency horizontally
   (valid invariant with random weights since model has no architectural bias).
8. **Vanilla and IG are correlated:** both point to similar regions
   (Spearman rank > 0.3 with random weights, higher with trained weights).
9. **IG steps improve quality:** completeness error decreases with more steps.

## Files you must NOT modify

- `neural/wrapper.py` — pre-work is already done
- `selfplay/mcts.py` — owned by Workstream C
- `engine/board.py`, `engine/encoding.py`, `engine/threats.py` — no changes needed
- `neural/model.py` — no changes needed
- `explain/activations.py` — owned by Workstream B
- `explain/comparison.py` — owned by Workstream C

## Dependencies

- `neural/model.py` for GomokuNet
- `neural/wrapper.py` for `GomokuInferenceWrapper.evaluate_raw()` (already added)
- `engine/encoding.py` for `board_to_tensor()` (for baseline creation)
- `engine/board.py` for `Board`

## Gradient safety rules (critical)

- NEVER leave `requires_grad` set on the model's parameters after your call.
- Call `model.zero_grad(set_to_none=True)` before each backward pass.
- Use fresh `tensor.clone().requires_grad_(True)` for each IG interpolation
  step — do NOT mutate a shared tensor in-place.
- Wrap the IG loop body except the active step in `torch.no_grad()` to avoid
  accumulating computation graphs for all steps simultaneously.
- After `evaluate_raw()`, the result tensors are on the model's device. Move
  baseline tensors to the same device.

## Import conventions

Use the project's conventions:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from engine.board import Board
from neural.wrapper import GomokuInferenceWrapper
```

## Serialization

`SaliencyMap.grid` is a numpy array. For future JSON serialization, consumers
call `.tolist()`. No custom serialization needed on this workstream.

## Integration notes

- Your module will be re-exported from `explain/__init__.py` by whoever creates it.
- No other `explain/` module imports from you.
- Your output (`SaliencyMap.grid`) can be rendered as a policy heatmap overlay
  in Phase 5 — this is just a 15×15 float array with values in [0, 1].
