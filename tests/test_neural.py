"""Tests for neural.model and neural.wrapper."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from engine.board import Board
from neural.model import (
    AttentionAugmentedConv,
    DropPath,
    GomokuNet,
    PreActSEResidualBlock,
    ResidualBlock,
    SELayer,
    SEResidualBlock,
)
from neural.wrapper import GomokuInferenceWrapper


# ---------------------------------------------------------------------------
# Model shape contracts
# ---------------------------------------------------------------------------


def test_residual_block_preserves_shape():
    block = ResidualBlock(64)
    x = torch.randn(2, 64, 15, 15)
    out = block(x)
    assert out.shape == x.shape


def test_se_layer_preserves_shape():
    se = SELayer(64, reduction=16)
    x = torch.randn(2, 64, 15, 15)
    out = se(x)
    assert out.shape == x.shape


def test_se_layer_modulates_channels():
    se = SELayer(64, reduction=16)
    x = torch.randn(2, 64, 15, 15)
    out = se(x)
    assert not torch.allclose(out, x)


def test_attention_augmented_conv_preserves_shape():
    attn = AttentionAugmentedConv(64, num_heads=1)
    x = torch.randn(2, 64, 15, 15)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_augmented_conv_multi_head():
    attn = AttentionAugmentedConv(64, num_heads=2)
    x = torch.randn(2, 64, 15, 15)
    out = attn(x)
    assert out.shape == x.shape


def test_se_residual_block_preserves_shape():
    block = SEResidualBlock(64)
    x = torch.randn(2, 64, 15, 15)
    out = block(x)
    assert out.shape == x.shape


def test_se_residual_block_no_se_no_attn():
    block = SEResidualBlock(64, use_se=False, use_attention=False)
    x = torch.randn(2, 64, 15, 15)
    out = block(x)
    assert out.shape == x.shape


def test_se_residual_block_variants():
    for use_se in (True, False):
        for use_attn in (True, False):
            block = SEResidualBlock(64, use_se=use_se, use_attention=use_attn)
            x = torch.randn(1, 64, 15, 15)
            out = block(x)
            assert out.shape == x.shape


def test_model_with_se_and_attention():
    model = GomokuNet()
    x = torch.randn(4, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (4, 225)
    assert value.shape == (4, 1)
    probs = torch.exp(log_policy)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    assert (-1.0 <= value).all() and (value <= 1.0).all()


def test_model_without_se_and_attention():
    model = GomokuNet(
        num_res_blocks=5,
        num_hidden_channels=64,
        use_se=False,
        use_attention=False,
    )
    x = torch.randn(4, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (4, 225)
    assert value.shape == (4, 1)


def test_model_custom_depth_and_width():
    model = GomokuNet(
        num_res_blocks=3,
        num_hidden_channels=32,
        use_se=False,
        use_attention=False,
    )
    x = torch.randn(1, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (1, 225)
    assert value.shape == (1, 1)


def test_pre_act_block_preserves_shape():
    block = PreActSEResidualBlock(64)
    x = torch.randn(2, 64, 15, 15)
    out = block(x)
    assert out.shape == x.shape


def test_pre_act_block_variants():
    for use_se in (True, False):
        for use_attn in (True, False):
            block = PreActSEResidualBlock(64, use_se=use_se, use_attention=use_attn)
            x = torch.randn(1, 64, 15, 15)
            out = block(x)
            assert out.shape == x.shape


def test_pre_act_produces_different_output():
    """Pre-activation output should differ from standard block for same input."""
    std_block = SEResidualBlock(64, use_se=False, use_attention=False)
    pre_block = PreActSEResidualBlock(64, use_se=False, use_attention=False)
    x = torch.randn(1, 64, 15, 15)
    # Initialize BN running stats to same values.
    with torch.no_grad():
        _ = std_block(x.clone())
        _ = pre_block(x.clone())
    out_std = std_block(x.clone())
    out_pre = pre_block(x.clone())
    assert not torch.allclose(out_std, out_pre)


def test_model_with_pre_activation():
    model = GomokuNet(use_pre_activation=True)
    x = torch.randn(4, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (4, 225)
    assert value.shape == (4, 1)
    probs = torch.exp(log_policy)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    assert (-1.0 <= value).all() and (value <= 1.0).all()


def test_model_with_pre_activation_and_old_defaults():
    """Pre-activation works with the old 5-block / 64-channel config."""
    model = GomokuNet(
        num_res_blocks=5,
        num_hidden_channels=64,
        use_se=False,
        use_attention=False,
        use_pre_activation=True,
    )
    x = torch.randn(2, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (2, 225)
    assert value.shape == (2, 1)


def test_value_head_no_nan():
    """Value head produces finite outputs for random input."""
    model = GomokuNet()
    x = torch.randn(4, 3, 15, 15)
    _, value = model(x)
    assert value.shape == (4, 1)
    assert not torch.isnan(value).any()
    assert not torch.isinf(value).any()


def test_value_head_range():
    """Value output is in [-1, 1]."""
    model = GomokuNet()
    x = torch.randn(4, 3, 15, 15)
    _, value = model(x)
    assert (-1.0 <= value).all() and (value <= 1.0).all()


def test_value_head_deterministic():
    """Same input produces same value in eval mode."""
    model = GomokuNet()
    model.eval()
    x = torch.randn(2, 3, 15, 15)
    with torch.no_grad():
        _, v1 = model(x)
        _, v2 = model(x)
    assert torch.allclose(v1, v2)


def test_value_dual_pooling():
    """Value head uses both avg and max pooling (verified by shape)."""
    model = GomokuNet(num_hidden_channels=128)
    # value_fc1 takes 2*C = 256 inputs
    assert model.value_fc1.in_features == 256
    assert model.value_fc1.out_features == 64


def test_model_output_shapes():
    model = GomokuNet(board_size=15, in_channels=3)
    x = torch.randn(4, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (4, 225)
    assert value.shape == (4, 1)


def test_log_policy_is_log_softmax():
    model = GomokuNet()
    x = torch.randn(1, 3, 15, 15)
    log_policy, _ = model(x)
    probs = torch.exp(log_policy)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_value_in_range():
    model = GomokuNet()
    x = torch.randn(1, 3, 15, 15)
    _, value = model(x)
    assert -1.0 <= value.item() <= 1.0


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------


def test_wrapper_save_load_and_evaluate():
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    try:
        wrapper = GomokuInferenceWrapper(
            tmp_path, device="cpu", num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False
        )
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        move_probs, value = wrapper.evaluate(board)

        assert len(move_probs) > 0
        total = sum(p for _, p in move_probs)
        assert abs(total - 1.0) < 1e-5
        assert -1.0 <= value <= 1.0
    finally:
        tmp_path.unlink()


def test_wrapper_evaluate_with_threats_no_crash():
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    try:
        wrapper = GomokuInferenceWrapper(tmp_path, device="cpu", num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        move_probs, value, info = wrapper.evaluate_with_threats(board)
        assert len(move_probs) > 0
        assert -1.0 <= value <= 1.0
        # No threats yet, so no override
        assert info is None
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper():
    """Create a wrapper around a fresh untrained model for testing."""
    model = GomokuNet(
        board_size=15,
        in_channels=3,
        num_res_blocks=5,
        num_hidden_channels=64,
        use_se=False,
        use_attention=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    try:
        wrapper = GomokuInferenceWrapper(
            tmp_path,
            device="cpu",
            num_res_blocks=5,
            num_hidden_channels=64,
            use_se=False,
            use_attention=False,
        )
        yield wrapper
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Batch evaluate
# ---------------------------------------------------------------------------


def test_batch_evaluate_returns_correct_count():
    """N boards in → N results out."""
    wrapper = next(_make_wrapper())
    board1 = Board()
    board2 = Board()
    board1.make_move(7, 7)
    board2.make_move(7, 7)
    board2.make_move(7, 8)

    results = wrapper.batch_evaluate([board1, board2])
    assert len(results) == 2
    for move_probs, value in results:
        assert isinstance(move_probs, list)
        assert len(move_probs) > 0
        assert isinstance(move_probs[0], tuple)
        assert isinstance(move_probs[0][0], tuple)  # (row, col)
        assert isinstance(move_probs[0][1], float)  # prob
        assert -1.0 <= value <= 1.0


def test_batch_evaluate_empty():
    """Empty input → empty output."""
    wrapper = next(_make_wrapper())
    results = wrapper.batch_evaluate([])
    assert results == []


def test_batch_evaluate_matches_single():
    """Each board in a batch produces the same result as calling evaluate() individually."""
    wrapper = next(_make_wrapper())
    boards = [Board() for _ in range(4)]
    for i, b in enumerate(boards):
        b.make_move(7, 7)
        if i % 2 == 0:
            b.make_move(7, 8)

    batch_results = wrapper.batch_evaluate(boards)
    single_results = [wrapper.evaluate(b) for b in boards]

    for (b_probs, b_val), (s_probs, s_val) in zip(batch_results, single_results):
        assert abs(b_val - s_val) < 1e-5
        assert len(b_probs) == len(s_probs)
        for (bm, bp), (sm, sp) in zip(sorted(b_probs), sorted(s_probs)):
            assert bm == sm
            assert abs(bp - sp) < 1e-5


# ---------------------------------------------------------------------------
# Batch evaluate with threats
# ---------------------------------------------------------------------------


def test_batch_evaluate_with_threats_empty():
    """Empty input → empty output."""
    wrapper = next(_make_wrapper())
    results = wrapper.batch_evaluate_with_threats([])
    assert results == []


def test_batch_evaluate_with_threats_matches_single():
    """Batch results match individual evaluate_with_threats calls."""
    wrapper = next(_make_wrapper())
    boards = [Board() for _ in range(4)]
    for b in boards:
        b.make_move(7, 7)
        b.make_move(7, 8)

    batch_results = wrapper.batch_evaluate_with_threats(boards)
    single_results = [wrapper.evaluate_with_threats(b) for b in boards]

    for (b_probs, b_val, b_info), (s_probs, s_val, s_info) in zip(
        batch_results, single_results
    ):
        assert abs(b_val - s_val) < 1e-5
        assert len(b_probs) == len(s_probs)
        for (bm, bp), (sm, sp) in zip(sorted(b_probs), sorted(s_probs)):
            assert bm == sm
            assert abs(bp - sp) < 1e-5
        # Both should have matching threat_info dict or both be None.
        if b_info is None or s_info is None:
            assert b_info is None and s_info is None
        else:
            assert b_info == s_info


def test_batch_evaluate_with_threats_immediate_win():
    """When the current player has an OPEN_FOUR, batch evaluate should override
    with deterministic winning moves and skip neural eval."""
    from engine.threats import ThreatDetector, ThreatType

    wrapper = next(_make_wrapper())
    board = Board()
    # Set up: Black gets 4 in a row at (7,3)-(7,6), both ends open.
    # Black moves first, then White dummies.
    board.make_move(7, 3)  # Black
    board.make_move(0, 0)  # White
    board.make_move(7, 4)  # Black
    board.make_move(0, 1)  # White
    board.make_move(7, 5)  # Black
    board.make_move(0, 2)  # White
    board.make_move(7, 6)  # Black
    board.make_move(0, 3)  # White

    # Now it's Black's turn. Black has OPEN_FOUR at (7,2) and (7,7).
    results = wrapper.batch_evaluate_with_threats([board])
    probs, value, info = results[0]
    assert info is not None
    assert info["reason"] == "immediate_win"
    assert value == 1.0
    # Winning moves should be (7,2) and (7,7).
    winning = {(7, 2), (7, 7)}
    for move, prob in probs:
        if move in winning:
            assert prob > 0.0
        else:
            assert prob == 0.0


def test_batch_evaluate_with_threats_block_boosting():
    """Blocking moves get boosted via the urgent_blocks path.

    Uses opponent OPEN_THREE (not OPEN_FOUR) so the hard-override
    ``must_block`` path is NOT triggered — only ``boosted_blocks``.
    """
    wrapper = next(_make_wrapper())
    board = Board()
    # White builds an open-three while Black plays scattered dummies
    # that form no threat of their own.
    board.make_move(0, 0)  # Black
    board.make_move(7, 3)  # White
    board.make_move(2, 2)  # Black
    board.make_move(7, 4)  # White
    board.make_move(0, 4)  # Black
    board.make_move(7, 5)  # White
    board.make_move(2, 6)  # Black
    board.make_move(6, 6)  # White — second threat pattern

    # Black's turn. White has open-three at (7,3)-(7,5).
    # TacticalSolver populates urgent_blocks → boosted_blocks kicks in.
    results = wrapper.batch_evaluate_with_threats([board])
    probs, value, info = results[0]

    assert info is not None
    assert info["reason"] == "boosted_blocks"
    assert -1.0 <= value <= 1.0
    total = sum(p for _, p in probs)
    assert abs(total - 1.0) < 1e-5
    block_moves = {(7, 2), (7, 6), (5, 6), (7, 6)}
    for m in block_moves:
        assert m in dict(probs)


# ---------------------------------------------------------------------------
# Upgraded architecture tests (v2)
# ---------------------------------------------------------------------------


def test_policy_head_fully_convolutional():
    """Policy head uses 1×1 conv projection, not an FC layer."""
    model = GomokuNet()
    # New policy head: policy_conv1 (3×3), policy_conv2 (1×1), no policy_fc
    assert hasattr(model, "policy_conv2")
    assert isinstance(model.policy_conv2, nn.Conv2d)
    assert model.policy_conv2.out_channels == 1
    assert not hasattr(model, "policy_fc")


def test_policy_head_output_shape():
    """Fully-conv policy head still produces (B, 225) log-softmax."""
    model = GomokuNet()
    x = torch.randn(4, 3, 15, 15)
    log_policy, _ = model(x)
    assert log_policy.shape == (4, 225)
    probs = torch.exp(log_policy)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_se_reduction_default():
    """Default SE reduction is 8 (bottleneck ≥ 16 for 128-ch networks)."""
    model = GomokuNet()
    se = model.res_blocks[0].se
    assert se.fc1.out_features >= 16  # 128 // 8 = 16
    assert se.fc1.out_features == se.fc2.in_features


def test_attention_heads_default():
    """Default is 4 attention heads."""
    model = GomokuNet()
    attn = model.res_blocks[0].attn
    assert attn.num_heads == 4
    assert attn.head_dim == 32  # 128 // 4


def test_attention_has_layernorm():
    """Attention module includes a pre-projection LayerNorm."""
    model = GomokuNet()
    attn = model.res_blocks[0].attn
    assert hasattr(attn, "norm")
    assert isinstance(attn.norm, nn.LayerNorm)


def test_dilation_schedule_is_pyramid():
    """Dilation schedule ramps 1→2→3→2→1 for 10 blocks."""
    model = GomokuNet(num_res_blocks=10)
    dilations = [
        b.conv1.dilation[0]  # type: ignore[attr-defined]
        for b in model.res_blocks
    ]
    # Must contain all three dilation levels
    assert 1 in dilations
    assert 2 in dilations
    assert 3 in dilations
    # Must start and end with dilation 1
    assert dilations[0] == 1
    assert dilations[-1] == 1
    # Monotonically increases then decreases
    peak_idx = dilations.index(max(dilations))
    for i in range(peak_idx):
        assert dilations[i] <= dilations[i + 1]
    for i in range(peak_idx, len(dilations) - 1):
        assert dilations[i] >= dilations[i + 1]


def test_dilation_schedule_custom():
    """Custom dilation schedule is respected."""
    custom = [1, 2, 3, 1, 2]
    model = GomokuNet(num_res_blocks=5, dilations=custom)
    dilations = [
        b.conv1.dilation[0]  # type: ignore[attr-defined]
        for b in model.res_blocks
    ]
    assert dilations == custom


def test_dilation_schedule_length_mismatch_raises():
    """ValueError when dilations length != num_res_blocks."""
    with pytest.raises(ValueError):
        GomokuNet(num_res_blocks=5, dilations=[1, 2, 3])


def test_drop_path_created():
    """Each residual block has a DropPath module."""
    model = GomokuNet(drop_path_rate=0.1)
    for block in model.res_blocks:
        assert hasattr(block, "drop_path")
        assert isinstance(block.drop_path, (DropPath, nn.Identity))


def test_drop_path_linear_schedule():
    """DropPath rates increase linearly from 0 to drop_path_rate."""
    model = GomokuNet(drop_path_rate=0.2, num_res_blocks=10)
    # Blocks with drop_prob==0 get nn.Identity; later blocks get DropPath.
    drop_probs: list[float] = []
    for block in model.res_blocks:
        if isinstance(block.drop_path, DropPath):
            drop_probs.append(block.drop_path.drop_prob)
    # At least the last half of blocks should have DropPath.
    assert len(drop_probs) > 0
    assert drop_probs[-1] == pytest.approx(0.2)
    for i in range(len(drop_probs) - 1):
        assert drop_probs[i] <= drop_probs[i + 1]


def test_drop_path_training_vs_eval():
    """DropPath is active in train() but identity in eval()."""
    drop = DropPath(drop_prob=0.5)
    x = torch.ones(1000, 1, 1, 1)

    drop.train()
    out = drop(x)
    num_nonzero = (out.abs() > 1e-6).sum().item()
    assert 350 < num_nonzero < 650  # ~50% kept, scaled by 2

    drop.eval()
    out_eval = drop(x)
    assert torch.allclose(out_eval, x)


def test_conv_policy_head_preserves_spatial_structure():
    """Policy head without FC preserves spatial locality for inference."""
    model = GomokuNet()
    model.eval()
    # The policy_conv2 is 1×1, so each output cell depends only on the
    # corresponding input feature column.
    assert model.policy_conv2.kernel_size == (1, 1)
    assert model.policy_conv2.out_channels == 1


def test_new_model_parameter_count():
    """v2 model parameter count is within expected range (3-4M)."""
    model = GomokuNet()
    total = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < total < 4_000_000
