# Workstream B — Activation Visualization

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

You own **activation visualization** — capture intermediate feature maps from
GomokuNet's residual blocks. Register forward hooks on `model.res_blocks`,
run a forward pass, collect activations. Output is a set of 15×15 activation
grids (one per channel per block) that show what patterns each layer detects.

Also read the spec at `docs/superpowers/specs/2026-05-23-explainability-design.md`
for the full design context.

## Pre-work already done (do NOT redo)

A method `evaluate_raw(board)` has been added to `GomokuInferenceWrapper` in
`neural/wrapper.py`. It returns `(log_policy, value)` raw tensors without
`torch.no_grad()`. Use this to run forward passes for activation capture.

NOTE: You should wrap `evaluate_raw()` in `torch.no_grad()` yourself in your
code — unlike Saliency (Workstream A), you don't need gradients.

## Files you will create

### `explain/activations.py`

Public API:

```python
@dataclass(slots=True)
class ActivationSnapshot:
    """Captured activations from residual blocks.

    activations[i] has shape (num_channels, 15, 15) for block i,
    stored as float32 numpy on CPU.
    """
    activations: list[NDArray[np.float32]]
    block_indices: list[int]
    channel_count: int

def capture_activations(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    *,
    blocks: Optional[list[int]] = None,
    channels: Optional[list[int]] = None,
) -> ActivationSnapshot:
    """Run a forward pass and capture activations from residual blocks.

    Args:
        wrapper: Inference wrapper with evaluate_raw().
        board: The board position.
        blocks: Indices of blocks to capture (None = all).
        channels: Indices of channels per block (None = all).

    Returns:
        ActivationSnapshot with captured data (already moved to CPU as numpy).
    """

def select_top_channels(
    snapshot: ActivationSnapshot,
    block_idx: int,
    k: int = 16,
) -> list[int]:
    """Return the k channel indices with highest L2 norm in a given block."""

def channel_to_grid(
    snapshot: ActivationSnapshot,
    block_idx: int,
    channel_idx: int,
) -> NDArray[np.float32]:
    """Extract a single channel as a (15, 15) float32 grid."""
```

### Hook Architecture — ActivationCapture class

Use a context manager pattern for safe hook lifecycle:

```python
class ActivationCapture:
    """Context manager that registers forward hooks on model.res_blocks.

    Usage:
        with ActivationCapture(wrapper.model, blocks=[0, 1, 5]) as cap:
            wrapper.evaluate_raw(board)
        snapshot = cap.to_snapshot()
    """

    def __init__(self, model: nn.Module, block_indices: list[int]):
        self._model = model
        self._handles: list[RemovableHandle] = []
        # Initialize storage for each block
        self._activations: dict[int, torch.Tensor] = {}
        for idx in block_indices:
            block = model.res_blocks[idx]
            handle = block.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            # Detach and move to CPU inside the hook to free GPU memory immediately
            self._activations[idx] = output.detach().cpu()
        return hook

    def to_snapshot(self) -> ActivationSnapshot:
        """Convert captured activations to an ActivationSnapshot."""
        ...

    def close(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

Key design decisions:
- **Pre-allocate storage in __init__** using a dict keyed by block index.
- **Immediately .detach().cpu() inside the hook** — the hook runs during
  forward pass; GPU memory is freed before the next operation.
- **close() must always be called** — context manager guarantees this.
- **If the forward pass fails**, the context manager `__exit__` still runs
  and calls `close()`.
- After `close()`, verify that no hook residues remain on the model.

### `tests/test_activations.py`

Test the following invariants (all testable with random weights):

1. **Hook fires:** After `capture_activations(wrapper, board)`, snapshot
   has `len(activations) == model.num_res_blocks`.
2. **Shape correctness:** Each activation has shape
   `(num_hidden_channels, 15, 15)`. Note: the raw hook output has batch dim
   `(1, C, 15, 15)` — squeeze the batch dim before storing.
3. **Default model params:** The test should use a GomokuNet with default
   params (10 blocks, 128 channels) to match the real model.
4. **Channel selection:** `select_top_channels(snapshot, 0, k=16)` returns
   16 channel indices in range `[0, num_hidden_channels)`.
5. **Channel selection sort:** Top channels are sorted by L2 norm descending.
6. **Hook cleanup:** After the context manager exits, verify that hooks
   are removed (check `len(model._forward_hooks) == 0` or check that a
   second context manager run doesn't double-register).
7. **Idempotency:** Two captures on the same board produce identical
   activations (deterministic in eval mode).
8. **Empty board:** No crash, no NaN activations.
9. **Channel extraction:** `channel_to_grid(snapshot, 0, 0)` returns
   shape `(15, 15)` with finite values.
10. **Selective blocks:** `blocks=[0, 2, 4]` captures exactly 3 blocks.
11. **Block indices validation:** Invalid block index raises IndexError.

## Files you must NOT modify

- `neural/wrapper.py` — pre-work is already done
- `neural/model.py` — no changes needed (hooks are external via nn.Module API)
- `selfplay/mcts.py` — owned by Workstream C
- `engine/board.py`, `engine/encoding.py`, `engine/threats.py` — no changes
- `explain/saliency.py` — owned by Workstream A
- `explain/comparison.py` — owned by Workstream C

## Dependencies

- `neural/model.py` for `GomokuNet` (specifically `model.res_blocks`)
- `neural/wrapper.py` for `GomokuInferenceWrapper.evaluate_raw()` (already added)
- `engine/board.py` for `Board`

## Import conventions

Use the project's conventions:
```python
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from engine.board import Board
from neural.wrapper import GomokuInferenceWrapper
```

## Memory management

- Each activation tensor: `(1, 128, 15, 15) * float32 = 115 KB` per block.
  10 blocks = 1.15 MB per capture. This is negligible for single positions.
- For batch mode in the future: same 115 KB per block per board, so
  `batch_size=32` with 10 blocks = ~37 MB. Still manageable but worth noting.
- **Critical:** Always `.cpu()` inside the hook. Never hold GPU memory.

## Integration notes

- Your module will be re-exported from `explain/__init__.py` by whoever creates it.
- No other `explain/` module imports from you.
- A future Phase 5 activation viewer will read your `ActivationSnapshot`
  activations and render them as tiled images or interactive grids.
- The `select_top_channels` function is the key bridge between arbitrary
  channel count (128) and what a UI can reasonably display (16-32).
