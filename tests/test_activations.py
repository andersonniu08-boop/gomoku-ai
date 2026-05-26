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


@pytest.fixture
def wrapper(model: GomokuNet, tmp_path) -> GomokuInferenceWrapper:
    """Create a wrapper with random weights (no trained checkpoint needed)."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return GomokuInferenceWrapper(
        checkpoint_path=str(checkpoint_path),
        num_res_blocks=10,
        num_hidden_channels=128,
        use_se=True,
        use_attention=True,
    )


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


def test_cleanup_on_exception(model: GomokuNet) -> None:
    """Hooks are removed even when an exception is raised in the with block."""
    from explain.activations import ActivationCapture

    block = model.res_blocks[0]
    before = len(block._forward_hooks)

    try:
        with ActivationCapture(model, [0]):
            msg = "simulated failure"
            raise RuntimeError(msg)
    except RuntimeError:
        pass

    assert len(block._forward_hooks) == before


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


def test_select_top_channels(wrapper: GomokuInferenceWrapper, board: Board) -> None:
    """select_top_channels returns k indices in [0, num_channels)."""
    from explain.activations import capture_activations, select_top_channels

    snap = capture_activations(wrapper, board)
    num_channels = snap.activations[0].shape[0]
    top = select_top_channels(snap, block_idx=0, k=16)

    assert len(top) == 16
    for idx in top:
        assert 0 <= idx < num_channels


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
