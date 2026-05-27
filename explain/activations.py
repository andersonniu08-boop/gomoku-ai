"""Activation visualization — capture intermediate feature maps via forward hooks.

Provides a context manager (``ActivationCapture``) for safe hook lifecycle and
helper functions for filtering and extracting activation channels.
"""

from __future__ import annotations

from collections.abc import Callable
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


class ActivationCapture:
    """Context manager that registers forward hooks on ``model.res_blocks``.

    Usage::

        with ActivationCapture(model, blocks=[0, 1, 5]) as cap:
            model(board_tensor)
        snapshot = cap.to_snapshot()
    """

    def __init__(self, model: nn.Module, block_indices: list[int]) -> None:
        self._model = model
        self._handles: list[RemovableHandle] = []
        self._activations: dict[int, torch.Tensor] = {}
        # Validate all indices before registering any hooks
        for idx in block_indices:
            model.res_blocks[idx]  # may raise IndexError
        try:
            for idx in block_indices:
                block = model.res_blocks[idx]
                handle = block.register_forward_hook(self._make_hook(idx))
                self._handles.append(handle)
        except Exception:
            self.close()
            raise

    def _make_hook(self, idx: int) -> Callable[..., None]:
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
    model = getattr(wrapper, "_raw_model", wrapper.model)
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
        return ActivationSnapshot(
            activations=filtered,
            block_indices=snapshot.block_indices[:],
            channel_count=len(channels),
        )

    return snapshot


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
