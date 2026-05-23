"""Gradient-based input attribution for GomokuNet.

Provides Integrated Gradients (primary) and vanilla gradient (fast-path)
saliency computation. Each method produces a 15x15 heatmap showing which
board cells most influenced the network's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from engine.board import Board
from engine.encoding import board_to_tensor
from neural.wrapper import GomokuInferenceWrapper

BOARD_SIZE = 15


@dataclass(slots=True)
class SaliencyMap:
    """Attribution heatmap for a single board position.

    Attributes:
        grid:   (15, 15) float32 array, values in [0, 1].
        method: ``"integrated_gradients"`` or ``"vanilla"``.
        target: The network output being explained, e.g. ``"value"``,
                ``"policy"``, or ``"policy_move(7,3)"``.
        n_steps: Number of interpolation steps (``None`` for vanilla).
    """

    grid: NDArray[np.float32]
    method: str
    target: str
    n_steps: int | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_saliency(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    *,
    method: str = "ig",
    target: str = "value",
    n_steps: int = 50,
) -> SaliencyMap:
    """Compute a saliency map for the given board position.

    Args:
        wrapper: Inference wrapper with a loaded GomokuNet.
        board: The board position to explain.
        method: ``"ig"`` for Integrated Gradients, ``"vanilla"`` for
                single-gradient.
        target: ``"value"``, ``"policy"``, or ``"policy_move(r,c)"``.
        n_steps: Number of interpolation steps for IG (ignored for vanilla).

    Returns:
        A ``SaliencyMap`` with a 15x15 attribution grid in [0, 1].

    Raises:
        ValueError: If ``method`` or ``target`` is unrecognised.
    """
    input_tensor = board_to_tensor(board).to(wrapper.device)
    model = wrapper.model

    target_fn = lambda log_p, v: _make_target_fn(log_p, v, target)

    if method == "ig":
        raw_grad = _compute_integrated_gradients(
            model, input_tensor, target_fn, n_steps
        )
        grid = attribution_to_grid(raw_grad)
        return SaliencyMap(
            grid=grid,
            method="integrated_gradients",
            target=target,
            n_steps=n_steps,
        )
    elif method == "vanilla":
        raw_grad = _compute_vanilla_gradient(model, input_tensor, target_fn)
        grid = attribution_to_grid(raw_grad)
        return SaliencyMap(
            grid=grid,
            method="vanilla",
            target=target,
            n_steps=None,
        )
    else:
        raise ValueError(f"Unknown saliency method: {method!r}. Use 'ig' or 'vanilla'.")


def attribution_to_grid(raw_gradients: torch.Tensor) -> NDArray[np.float32]:
    """Convert a gradient tensor to a (15, 15) normalised heatmap.

    Max-pools across the 3 input channels, takes absolute value, and
    normalises to [0, 1].

    Args:
        raw_gradients: Tensor of shape ``(3, 15, 15)`` or ``(1, 3, 15, 15)``.

    Returns:
        (15, 15) float32 array with values in [0, 1].
    """
    if raw_gradients.dim() == 4:
        raw_gradients = raw_gradients.squeeze(0)  # (3, 15, 15)

    # Absolute value, then max-pool across channels.
    grid, _ = raw_gradients.abs().max(dim=0)  # (15, 15)

    # Normalise to [0, 1].
    max_val = grid.max()
    if max_val > 1e-12:
        grid = grid / max_val

    return grid.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_target_fn(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    target: str,
) -> torch.Tensor:
    """Extract a scalar target from model outputs for gradient computation.

    Args:
        log_policy: (1, 225) log-softmax policy tensor.
        value: (1, 1) value tensor.
        target: One of ``"value"``, ``"policy"``, ``"policy_move(r,c)"``.

    Returns:
        A scalar tensor suitable for ``backward()``.
    """
    if target == "value":
        return value[0, 0]
    elif target == "policy":
        return log_policy[0, :].sum()
    elif target.startswith("policy_move(") and target.endswith(")"):
        inner = target[len("policy_move(") : -1]
        parts = inner.split(",")
        r, c = int(parts[0]), int(parts[1])
        idx = r * BOARD_SIZE + c
        return log_policy[0, idx]
    else:
        raise ValueError(
            f"Unknown target: {target!r}. "
            f"Use 'value', 'policy', or 'policy_move(r,c)'."
        )


def _compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_steps: int,
) -> torch.Tensor:
    """Integrated Gradients: interpolate from baseline to input, average gradients.

    Uses a right Riemann sum with step size 1/n_steps.

    Args:
        model: The model to explain (in eval mode).
        input_tensor: (1, 3, 15, 15) input tensor.
        target_fn: Callable ``(log_policy, value) -> scalar`` defining the
                   output to attribute.
        n_steps: Number of interpolation steps.

    Returns:
        (1, 3, 15, 15) attribution tensor.
    """
    baseline = torch.zeros_like(input_tensor, device=input_tensor.device)
    diff = input_tensor - baseline

    accumulated = torch.zeros_like(input_tensor, device=input_tensor.device)

    for k in range(1, n_steps + 1):
        alpha = k / n_steps

        # Build the interpolated input outside the active graph.
        with torch.no_grad():
            scaled = baseline + alpha * diff
        scaled = scaled.clone().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        log_policy, value = model(scaled)
        target = target_fn(log_policy, value)
        target.backward()

        if scaled.grad is not None:
            accumulated = accumulated + scaled.grad

    # Scale by (input - baseline) and normalise by n_steps.
    attributions = diff * accumulated / n_steps
    return attributions


def _compute_vanilla_gradient(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Single forward+backward pass for vanilla input gradient.

    Args:
        model: The model to explain (in eval mode).
        input_tensor: (1, 3, 15, 15) input tensor.
        target_fn: Callable ``(log_policy, value) -> scalar`` defining the
                   output to attribute.

    Returns:
        (1, 3, 15, 15) gradient tensor.
    """
    inp = input_tensor.clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    log_policy, value = model(inp)
    target = target_fn(log_policy, value)
    target.backward()

    if inp.grad is None:
        raise RuntimeError("Vanilla gradient computation failed — grad is None.")

    return inp.grad
