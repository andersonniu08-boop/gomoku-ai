"""Tests for gradient-based saliency maps (Workstream A).

All tests use a randomly initialized GomokuNet (no checkpoint needed)
running on CPU with a fixed random seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from engine.board import Board
from engine.encoding import board_to_tensor
from neural.model import GomokuNet
from explain.saliency import (
    SaliencyMap,
    attribution_to_grid,
    compute_saliency,
    _compute_integrated_gradients,
    _compute_vanilla_gradient,
    _make_target_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation on rank-transformed data (Spearman)."""
    rx = np.argsort(np.argsort(x.ravel())).astype(np.float64)
    ry = np.argsort(np.argsort(y.ravel())).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


class _TestWrapper:
    """Stand-in for GomokuInferenceWrapper using a random model."""

    def __init__(self, model: GomokuNet):
        self.model = model
        self.device = torch.device("cpu")


def _board_with_stones() -> Board:
    """Return a board with a few stones to create spatial structure."""
    b = Board()
    b.make_move(7, 7)  # Black centre
    b.make_move(7, 8)  # White
    b.make_move(8, 7)  # Black
    b.make_move(8, 8)  # White
    b.make_move(6, 7)  # Black — three in a column
    return b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
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
def wrapper(model):
    return _TestWrapper(model)


@pytest.fixture
def board():
    return _board_with_stones()


# ---------------------------------------------------------------------------
# 1. Completeness axiom
# ---------------------------------------------------------------------------


class TestCompletenessAxiom:
    """Integrated Gradients satisfies: sum(attr) ≈ F(input) - F(baseline)."""

    def test_value_target(self, model, board):
        input_tensor = board_to_tensor(board)
        baseline = torch.zeros_like(input_tensor)
        target_fn = lambda log_p, v: v[0, 0]

        attr = _compute_integrated_gradients(model, input_tensor, target_fn, n_steps=50)
        attr_sum = attr.sum().item()

        with torch.no_grad():
            _, v_in = model(input_tensor)
            _, v_base = model(baseline)
        output_diff = v_in[0, 0].item() - v_base[0, 0].item()

        assert abs(attr_sum - output_diff) < 1e-3, (
            f"Completeness error: |{attr_sum:.6f} - {output_diff:.6f}| >= 1e-3"
        )

    def test_policy_target(self, model, board):
        input_tensor = board_to_tensor(board)
        baseline = torch.zeros_like(input_tensor)
        target_fn = lambda log_p, v: log_p[0, :].sum()

        attr = _compute_integrated_gradients(model, input_tensor, target_fn, n_steps=50)
        attr_sum = attr.sum().item()

        with torch.no_grad():
            lp_in, _ = model(input_tensor)
            lp_base, _ = model(baseline)
        output_diff = lp_in[0, :].sum().item() - lp_base[0, :].sum().item()

        assert abs(attr_sum - output_diff) < 1e-3, (
            f"Completeness error: |{attr_sum:.6f} - {output_diff:.6f}| >= 1e-3"
        )

    def test_policy_move_target(self, model, board):
        input_tensor = board_to_tensor(board)
        baseline = torch.zeros_like(input_tensor)
        target_fn = lambda log_p, v: log_p[0, 7 * 15 + 7]

        attr = _compute_integrated_gradients(model, input_tensor, target_fn, n_steps=50)
        attr_sum = attr.sum().item()

        with torch.no_grad():
            lp_in, _ = model(input_tensor)
            lp_base, _ = model(baseline)
        output_diff = lp_in[0, 112].item() - lp_base[0, 112].item()

        assert abs(attr_sum - output_diff) < 1e-3, (
            f"Completeness error: |{attr_sum:.6f} - {output_diff:.6f}| >= 1e-3"
        )


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_grid_shape_15x15(self, wrapper, board):
        sm = compute_saliency(wrapper, board)
        assert sm.grid.shape == (15, 15), f"Expected (15, 15), got {sm.grid.shape}"

    def test_grid_shape_empty_board(self, wrapper):
        sm = compute_saliency(wrapper, Board())
        assert sm.grid.shape == (15, 15)


# ---------------------------------------------------------------------------
# 3. Output range
# ---------------------------------------------------------------------------


class TestOutputRange:
    def test_values_in_0_1(self, wrapper, board):
        sm = compute_saliency(wrapper, board)
        assert sm.grid.min() >= 0.0, f"Min {sm.grid.min()} < 0"
        assert sm.grid.max() <= 1.0, f"Max {sm.grid.max()} > 1"

    def test_vanilla_values_in_0_1(self, wrapper, board):
        sm = compute_saliency(wrapper, board, method="vanilla")
        assert sm.grid.min() >= 0.0
        assert sm.grid.max() <= 1.0

    def test_no_nan(self, wrapper, board):
        sm = compute_saliency(wrapper, board)
        assert not np.any(np.isnan(sm.grid))


# ---------------------------------------------------------------------------
# 4. Method selection
# ---------------------------------------------------------------------------


class TestMethodSelection:
    def test_ig_method_string(self, wrapper, board):
        sm = compute_saliency(wrapper, board, method="ig")
        assert sm.method == "integrated_gradients"
        assert sm.n_steps == 50

    def test_vanilla_method_string(self, wrapper, board):
        sm = compute_saliency(wrapper, board, method="vanilla")
        assert sm.method == "vanilla"
        assert sm.n_steps is None

    def test_invalid_method(self, wrapper, board):
        with pytest.raises(ValueError):
            compute_saliency(wrapper, board, method="invalid")


# ---------------------------------------------------------------------------
# 5. Target modes
# ---------------------------------------------------------------------------


class TestTargetModes:
    def test_value_target(self, wrapper, board):
        sm = compute_saliency(wrapper, board, target="value")
        assert sm.target == "value"
        assert sm.grid.shape == (15, 15)

    def test_policy_target(self, wrapper, board):
        sm = compute_saliency(wrapper, board, target="policy")
        assert sm.target == "policy"
        assert sm.grid.shape == (15, 15)

    def test_policy_move_target(self, wrapper, board):
        sm = compute_saliency(wrapper, board, target="policy_move(7,7)")
        assert sm.target == "policy_move(7,7)"
        assert sm.grid.shape == (15, 15)

    def test_invalid_target(self, wrapper, board):
        with pytest.raises(ValueError):
            compute_saliency(wrapper, board, target="invalid")


# ---------------------------------------------------------------------------
# 6. Empty board
# ---------------------------------------------------------------------------


class TestEmptyBoard:
    def test_no_crash(self, wrapper):
        empty = Board()
        sm = compute_saliency(wrapper, empty)
        assert sm.grid.shape == (15, 15)

    def test_no_nan(self, wrapper):
        empty = Board()
        sm = compute_saliency(wrapper, empty)
        assert not np.any(np.isnan(sm.grid)), "NaN values on empty board"

    def test_vanilla_no_crash(self, wrapper):
        sm = compute_saliency(wrapper, Board(), method="vanilla")
        assert sm.grid.shape == (15, 15)
        assert not np.any(np.isnan(sm.grid))


# ---------------------------------------------------------------------------
# 7. Symmetry — horizontal flip
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_horizontal_flip(self, wrapper, board):
        """Flipping input horizontally should flip saliency horizontally.

        With random weights and no architectural bias, the saliency map
        of the flipped board should correlate with the flipped saliency
        of the original board.
        """
        sm_orig = compute_saliency(wrapper, board)

        flipped = board.copy()
        flipped.grid = np.fliplr(flipped.grid)
        sm_flipped = compute_saliency(wrapper, flipped)

        orig_flipped_back = np.fliplr(sm_orig.grid)
        corr = _spearman_rank(orig_flipped_back, sm_flipped.grid)
        assert corr > 0.3, f"Spearman rho = {corr:.4f} (expected > 0.3)"


# ---------------------------------------------------------------------------
# 8. Vanilla and IG are correlated
# ---------------------------------------------------------------------------


class TestIgVanillaCorrelation:
    def test_correlation_with_random_weights(self, wrapper, board):
        sm_ig = compute_saliency(wrapper, board, method="ig")
        sm_vanilla = compute_saliency(wrapper, board, method="vanilla")
        corr = _spearman_rank(sm_ig.grid, sm_vanilla.grid)
        assert corr > 0.3, f"Spearman rho = {corr:.4f} (expected > 0.3)"


# ---------------------------------------------------------------------------
# 9. IG steps improve quality
# ---------------------------------------------------------------------------


class TestIgStepsQuality:
    def test_completeness_error_decreases_with_more_steps(self, model, board):
        input_tensor = board_to_tensor(board)
        baseline = torch.zeros_like(input_tensor)
        target_fn = lambda log_p, v: v[0, 0]

        with torch.no_grad():
            _, v_in = model(input_tensor)
            _, v_base = model(baseline)
        output_diff = v_in[0, 0].item() - v_base[0, 0].item()

        errors = {}
        for steps in [5, 20, 50]:
            attr = _compute_integrated_gradients(
                model, input_tensor, target_fn, n_steps=steps
            )
            attr_sum = attr.sum().item()
            errors[steps] = abs(attr_sum - output_diff)

        assert errors[50] < errors[5], (
            f"More steps should reduce error: {errors}"
        )


# ---------------------------------------------------------------------------
# Unit-level: attribution_to_grid, _make_target_fn
# ---------------------------------------------------------------------------


class TestAttributionToGrid:
    def test_squeezes_batch_dim(self):
        grad = torch.randn(1, 3, 15, 15)
        grid = attribution_to_grid(grad)
        assert grid.shape == (15, 15)

    def test_handles_no_batch_dim(self):
        grad = torch.randn(3, 15, 15)
        grid = attribution_to_grid(grad)
        assert grid.shape == (15, 15)

    def test_values_in_0_1(self):
        grad = torch.randn(3, 15, 15)
        grid = attribution_to_grid(grad)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_zero_tensor_handled(self):
        grad = torch.zeros(3, 15, 15)
        grid = attribution_to_grid(grad)
        assert grid.shape == (15, 15)
        assert np.all(grid >= 0.0)


class TestMakeTargetFn:
    def test_value_target(self):
        log_p = torch.randn(1, 225)
        v = torch.randn(1, 1)
        scalar = _make_target_fn(log_p, v, "value")
        assert scalar.shape == ()  # scalar
        assert scalar.item() == v[0, 0].item()

    def test_policy_target(self):
        log_p = torch.randn(1, 225)
        v = torch.randn(1, 1)
        scalar = _make_target_fn(log_p, v, "policy")
        assert scalar.shape == ()
        assert scalar.item() == log_p[0, :].sum().item()

    def test_policy_move_target(self):
        log_p = torch.randn(1, 225)
        v = torch.randn(1, 1)
        scalar = _make_target_fn(log_p, v, "policy_move(7,3)")
        assert scalar.shape == ()
        assert scalar.item() == log_p[0, 7 * 15 + 3].item()

    def test_invalid_target(self):
        log_p = torch.randn(1, 225)
        v = torch.randn(1, 1)
        with pytest.raises(ValueError):
            _make_target_fn(log_p, v, "invalid")
