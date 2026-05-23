# Activation Visualization (Workstream B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Capture intermediate feature maps from GomokuNet's residual blocks via forward hooks, producing 15×15 activation grids for visualization.

**Architecture:** Use PyTorch `register_forward_hook` on `model.res_blocks` entries inside a context manager (`ActivationCapture`) that guarantees cleanup. A `capture_activations()` convenience function wraps the full flow. Helper functions `select_top_channels()` and `channel_to_grid()` support downstream analysis.

**Tech Stack:** PyTorch hooks, NumPy, pytest

**Files created:**

| File | Purpose |
|------|---------|
| `explain/activations.py` | `ActivationCapture`, `ActivationSnapshot`, `capture_activations()`, `select_top_channels()`, `channel_to_grid()` |
| `tests/test_activations.py` | Tests for all 11 invariants |

**Files NOT modified:** No existing files are touched. The `explain/__init__.py` already has lazy imports for the activations module.

---

### Task 1: ActivationSnapshot dataclass

**Files:**
- Create: `explain/activations.py` (part 1 — dataclass)
- Test: `tests/test_activations.py` (part 1)

- [ ] **Step 1: Write the test skeleton and dataclass test**

```python
"""Tests for activation visualization (Workstream B)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from numpy.typing import NDArray

from engine.board import Board
from engine.encoding import board_to_tensor
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper


@pytest.fixture
def model() -> GomokuNet:
    """GomokuNet with default params and random weights."""
    torch.manual_seed(42)
    m = GomokuNet(
        board_size=15,
        in_channels=3,
        num_res_blocks=10,
        num_hidden_channels=128,
        use_se=True,
        use_attention=True,
    )
    m.eval()
    return m


@pytest.fixture
def board() -> Board:
    """Board with a few stones played."""
    b = Board()
    b.make_move(7, 7)
    b.make_move(7, 8)
    b.make_move(8, 7)
    b.make_move(8, 8)
    return b


def test_snapshot_creation() -> None:
    """ActivationSnapshot stores activations, block_indices, and channel_count."""
    from explain.activations import ActivationSnapshot

    snap = ActivationSnapshot(
        activations=[np.zeros((128, 15, 15), dtype=np.float32)],
        block_indices=[0],
        channel_count=128,
    )
    assert snap.activations[0].shape == (128, 15, 15)
    assert snap.block_indices == [0]
    assert snap.channel_count == 128
```

- [ ] **Step 2: Run test to verify it fails (module doesn't exist)**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_snapshot_creation -v 2>&1`
Expected: `ModuleNotFoundError` or `ImportError` for `explain.activations`

- [ ] **Step 3: Create initial `explain/activations.py` with the dataclass**

```python
"""Activation visualization — capture intermediate feature maps via forward hooks.

Provides a context manager (``ActivationCapture``) for safe hook lifecycle and
helper functions for filtering and extracting activation channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from engine.board import Board
from neural.wrapper import GomokuInferenceWrapper


@dataclass(slots=True)
class ActivationSnapshot:
    """Captured activations from residual blocks.

    activations[i] has shape (num_channels, 15, 15) for block i,
    stored as float32 numpy on CPU.
    """
    activations: list[NDArray[np.float32]]
    block_indices: list[int]
    channel_count: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_snapshot_creation -v 2>&1`
Expected: PASS

---

### Task 2: ActivationCapture context manager

**Files:**
- Modify: `explain/activations.py` (add ActivationCapture class)
- Modify: `tests/test_activations.py` (add hook and cleanup tests)

- [ ] **Step 1: Write hook-firing, shape, and cleanup tests**

```python
def test_hook_fires(model: GomokuNet, board: Board) -> None:
    """After capture, snapshot has len(activations) == num_res_blocks."""
    from explain.activations import ActivationCapture

    tensor = board_to_tensor(board)
    num_blocks = len(model.res_blocks)

    with ActivationCapture(model, list(range(num_blocks))) as cap:
        with torch.no_grad():
            model(tensor)
    snap = cap.to_snapshot()

    assert len(snap.activations) == num_blocks


def test_shape_correctness(model: GomokuNet, board: Board) -> None:
    """Each activation has shape (num_hidden_channels, 15, 15)."""
    from explain.activations import ActivationCapture

    tensor = board_to_tensor(board)

    with ActivationCapture(model, list(range(len(model.res_blocks)))) as cap:
        with torch.no_grad():
            model(tensor)
    snap = cap.to_snapshot()

    for activation in snap.activations:
        assert activation.shape == (128, 15, 15)


def test_hook_cleanup(model: GomokuNet, board: Board) -> None:
    """After context manager exit, hooks are removed and re-registration works."""
    from explain.activations import ActivationCapture

    tensor = board_to_tensor(board)
    block = model.res_blocks[0]
    before = len(block._forward_hooks)

    with ActivationCapture(model, [0]) as cap:
        with torch.no_grad():
            model(tensor)
    after_within = len(block._forward_hooks)
    assert after_within == before, "hooks not cleaned up"

    # Second run should work without double-registration
    with ActivationCapture(model, [0]) as cap:
        with torch.no_grad():
            model(tensor)
    snap = cap.to_snapshot()
    assert snap.activations[0].shape == (128, 15, 15)
    assert len(block._forward_hooks) == before
    assert len(cap._handles) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_hook_fires tests/test_activations.py::test_shape_correctness tests/test_activations.py::test_hook_cleanup -v 2>&1`
Expected: FAIL — `ActivationCapture` not defined

- [ ] **Step 3: Add ActivationCapture class to `explain/activations.py`**

Append after the `ActivationSnapshot` dataclass:

```python
class ActivationCapture:
    """Context manager that registers forward hooks on ``model.res_blocks``.

    Usage::

        with ActivationCapture(wrapper.model, blocks=[0, 1, 5]) as cap:
            wrapper.evaluate_raw(board)
        snapshot = cap.to_snapshot()
    """

    def __init__(self, model: nn.Module, block_indices: list[int]) -> None:
        self._model = model
        self._handles: list[RemovableHandle] = []
        self._activations: dict[int, torch.Tensor] = {}
        for idx in block_indices:
            block = model.res_blocks[idx]  # may raise IndexError
            handle = block.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._activations[idx] = output.detach().cpu()
        return hook

    def to_snapshot(self) -> ActivationSnapshot:
        """Convert captured activations to an ``ActivationSnapshot``."""
        sorted_indices = sorted(self._activations.keys())
        activations_np: list[NDArray[np.float32]] = []
        for idx in sorted_indices:
            arr = self._activations[idx].squeeze(0).numpy().astype(np.float32)
            activations_np.append(arr)
        channel_count = activations_np[0].shape[0] if activations_np else 0
        return ActivationSnapshot(
            activations=activations_np,
            block_indices=sorted_indices,
            channel_count=channel_count,
        )

    def close(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> ActivationCapture:
        return self

    def __exit__(self, *args) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_hook_fires tests/test_activations.py::test_shape_correctness tests/test_activations.py::test_hook_cleanup -v 2>&1`
Expected: 3 PASS

---

### Task 3: capture_activations convenience function

**Files:**
- Modify: `explain/activations.py` (add `capture_activations`)
- Modify: `tests/test_activations.py` (add test)

- [ ] **Step 1: Write test**

```python
@pytest.fixture
def wrapper(model: GomokuNet, tmp_path) -> GomokuInferenceWrapper:
    """Create a wrapper with random weights (no trained checkpoint needed)."""
    from neural.wrapper import GomokuInferenceWrapper

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return GomokuInferenceWrapper(
        checkpoint_path=str(checkpoint_path),
        num_res_blocks=10,
        num_hidden_channels=128,
        use_se=True,
        use_attention=True,
    )


def test_capture_activations(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """capture_activations returns the right number of blocks."""
    from explain.activations import capture_activations

    snap = capture_activations(wrapper, board)
    assert len(snap.activations) == len(wrapper.model.res_blocks)


def test_empty_board_no_crash(wrapper: GomokuInferenceWrapper) -> None:
    """Empty board does not crash and produces no NaN activations."""
    from explain.activations import capture_activations

    b = Board()
    snap = capture_activations(wrapper, b)
    for activation in snap.activations:
        assert not np.any(np.isnan(activation))
        assert not np.any(np.isinf(activation))


def test_selective_blocks(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """blocks=[0, 2, 4] captures exactly 3 blocks."""
    from explain.activations import capture_activations

    snap = capture_activations(wrapper, board, blocks=[0, 2, 4])
    assert len(snap.activations) == 3
    assert snap.block_indices == [0, 2, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_capture_activations tests/test_activations.py::test_empty_board_no_crash tests/test_activations.py::test_selective_blocks -v 2>&1`
Expected: FAIL — `capture_activations` not defined

- [ ] **Step 3: Add `capture_activations` to `explain/activations.py`**

Append after `ActivationCapture`:

```python
def capture_activations(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    *,
    blocks: Optional[list[int]] = None,
    channels: Optional[list[int]] = None,
) -> ActivationSnapshot:
    """Run a forward pass and capture activations from residual blocks.

    Args:
        wrapper: Inference wrapper with ``evaluate_raw()``.
        board: The board position to evaluate.
        blocks: Indices of blocks to capture (None = all).
        channels: Indices of channels to keep per block (None = all).

    Returns:
        ActivationSnapshot with captured data on CPU as numpy arrays.
    """
    model = wrapper.model
    num_blocks = len(model.res_blocks)
    if blocks is None:
        blocks = list(range(num_blocks))

    with ActivationCapture(model, blocks) as cap:
        with torch.no_grad():
            wrapper.evaluate_raw(board)

    snapshot = cap.to_snapshot()

    if channels is not None:
        filtered: list[NDArray[np.float32]] = []
        for activation in snapshot.activations:
            filtered.append(activation[channels, :, :].copy())
        snapshot.activations = filtered
        snapshot.channel_count = len(channels)

    return snapshot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_capture_activations tests/test_activations.py::test_empty_board_no_crash tests/test_activations.py::test_selective_blocks -v 2>&1`
Expected: 3 PASS

---

### Task 4: select_top_channels and channel_to_grid

**Files:**
- Modify: `explain/activations.py` (add `select_top_channels`, `channel_to_grid`)
- Modify: `tests/test_activations.py` (add tests)

- [ ] **Step 1: Write tests**

```python
def test_select_top_channels(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """select_top_channels returns k indices in [0, num_hidden_channels)."""
    from explain.activations import capture_activations, select_top_channels

    snap = capture_activations(wrapper, board)
    top = select_top_channels(snap, block_idx=0, k=16)

    assert len(top) == 16
    for idx in top:
        assert 0 <= idx < 128


def test_select_top_channels_sorted(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """Top channels are sorted by L2 norm descending."""
    from explain.activations import capture_activations, select_top_channels

    snap = capture_activations(wrapper, board)
    top = select_top_channels(snap, block_idx=0, k=16)

    # Verify descending L2 norm
    norms = [np.sqrt(np.sum(snap.activations[0][idx] ** 2)) for idx in top]
    for i in range(len(norms) - 1):
        assert norms[i] >= norms[i + 1]


def test_channel_to_grid(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """channel_to_grid returns (15, 15) with finite values."""
    from explain.activations import capture_activations, channel_to_grid

    snap = capture_activations(wrapper, board)
    grid = channel_to_grid(snap, block_idx=0, channel_idx=0)

    assert grid.shape == (15, 15)
    assert np.all(np.isfinite(grid))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_select_top_channels tests/test_activations.py::test_select_top_channels_sorted tests/test_activations.py::test_channel_to_grid -v 2>&1`
Expected: FAIL — functions not defined

- [ ] **Step 3: Add helper functions to `explain/activations.py`**

Append after `capture_activations`:

```python
def select_top_channels(
    snapshot: ActivationSnapshot,
    block_idx: int,
    k: int = 16,
) -> list[int]:
    """Return the ``k`` channel indices with highest L2 norm in a given block.

    Args:
        snapshot: An ActivationSnapshot from a previous capture.
        block_idx: Index into ``snapshot.activations``.
        k: Number of channels to return (default 16).

    Returns:
        List of channel indices sorted by L2 norm descending.
    """
    activations = snapshot.activations[block_idx]
    l2_norms = np.sqrt(np.sum(activations ** 2, axis=(1, 2)))
    top_indices = np.argsort(l2_norms)[::-1][:k]
    return top_indices.tolist()


def channel_to_grid(
    snapshot: ActivationSnapshot,
    block_idx: int,
    channel_idx: int,
) -> NDArray[np.float32]:
    """Extract a single channel as a (15, 15) float32 grid.

    Args:
        snapshot: An ActivationSnapshot from a previous capture.
        block_idx: Index into ``snapshot.activations``.
        channel_idx: Channel index within the block.

    Returns:
        A (15, 15) NDArray of float32 values.
    """
    return snapshot.activations[block_idx][channel_idx].copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py::test_select_top_channels tests/test_activations.py::test_select_top_channels_sorted tests/test_activations.py::test_channel_to_grid -v 2>&1`
Expected: 3 PASS

---

### Task 5: Remaining edge case tests

**Files:**
- Modify: `tests/test_activations.py` (add remaining tests)

- [ ] **Step 1: Write remaining tests**

```python
def test_idempotency(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """Two captures on same board produce identical activations (eval mode)."""
    from explain.activations import capture_activations

    snap1 = capture_activations(wrapper, board)
    snap2 = capture_activations(wrapper, board)

    for a1, a2 in zip(snap1.activations, snap2.activations):
        assert np.allclose(a1, a2)


def test_invalid_block_index(wrapper: GomokuInferenceWrapper) -> None:
    """Invalid block index raises IndexError."""
    from explain.activations import ActivationCapture

    invalid_idx = len(wrapper.model.res_blocks) + 1

    with pytest.raises(IndexError):
        cap = ActivationCapture(wrapper.model, [invalid_idx])
        cap.close()


def test_channel_filtering(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """Providing channels=[0, 5, 10] keeps only those channels."""
    from explain.activations import capture_activations

    snap = capture_activations(wrapper, board, channels=[0, 5, 10])
    assert snap.channel_count == 3
    for activation in snap.activations:
        assert activation.shape == (3, 15, 15)
```

- [ ] **Step 2: Run all tests**

Run: `cd /home/anderson/projects/gomoku-ai && python -m pytest tests/test_activations.py -v 2>&1`
Expected: 11 PASS (all green)

### Self-Review

- [ ] **Spec coverage check:** All 11 invariants from the spec have corresponding tests.
- [ ] **Placeholder scan:** No TBDs, TODOs, or vague steps.
- [ ] **Type consistency:** All function signatures match the spec. `capture_activations` accepts `wrapper: GomokuInferenceWrapper`. Hook functions accept `model: nn.Module`. Dataclass field types are consistent across all functions and tests.
