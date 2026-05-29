"""Tests for selfplay.train — loss function, training, evaluation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from neural.model import GomokuNet
from selfplay.selfplay import TrainingExample
from selfplay.train import compute_loss, run_evaluation, save_model_checkpoint, train_on_examples


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


def test_compute_loss_returns_scalar_tensors():
    policy_target = torch.rand(4, 225)
    policy_target /= policy_target.sum(dim=1, keepdim=True)
    log_policy = torch.log(policy_target + 1e-8)
    value = torch.rand(4, 1) * 2 - 1
    value_target = torch.rand(4, 1)

    policy_loss, value_loss, total = compute_loss(
        log_policy, value, policy_target, value_target
    )
    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert total.ndim == 0


def test_compute_loss_policy_perfect_prediction():
    """Cross-entropy is minimized when prediction matches target.

    Uses a near-deterministic target so the entropy (and therefore the
    minimum achievable cross-entropy) is close to zero."""
    target = torch.zeros(1, 225)
    target[0, 0] = 1.0
    # log_policy must be a valid log-softmax output reflecting the same
    # distribution. Use a large logit for position 0 so softmax collapses
    # nearly all mass there.
    logits = torch.zeros(1, 225)
    logits[0, 0] = 100.0  # large logit → exp(100) dominates softmax
    log_policy = torch.log_softmax(logits, dim=1)
    value = torch.zeros(1, 1)
    value_target = torch.zeros(1, 1)

    policy_loss, _, _ = compute_loss(log_policy, value, target, value_target)
    assert policy_loss.item() < 0.01


def test_compute_loss_policy_wrong_prediction():
    """Uniform prediction vs one-hot target should give log(225) cross-entropy."""
    import math

    target = torch.tensor([[1.0] + [0.0] * 224])
    # Valid log-softmax representing a uniform distribution: log(1/225) per cell.
    log_policy = torch.full((1, 225), math.log(1.0 / 225))
    value = torch.zeros(1, 1)
    value_target = torch.zeros(1, 1)

    policy_loss, _, _ = compute_loss(log_policy, value, target, value_target)
    # Cross-entropy of uniform distribution against one-hot target = -log(1/225)
    assert abs(policy_loss.item() - (-torch.log(torch.tensor(1.0 / 225)).item())) < 0.01


def test_compute_loss_value_perfect():
    target = torch.ones(1, 225) / 225
    log_policy = torch.log(target + 1e-8)
    value = torch.tensor([[0.5]])
    value_target = torch.tensor([[0.5]])

    _, value_loss, _ = compute_loss(log_policy, value, target, value_target)
    assert value_loss.item() < 1e-4


def test_compute_loss_value_wrong():
    target = torch.ones(1, 225) / 225
    log_policy = torch.log(target + 1e-8)
    value = torch.tensor([[1.0]])
    value_target = torch.tensor([[-1.0]])

    _, value_loss, _ = compute_loss(log_policy, value, target, value_target)
    assert value_loss.item() > 1.0  # (1 - (-1))^2 = 4


# ---------------------------------------------------------------------------
# save_model_checkpoint & train_on_examples
# ---------------------------------------------------------------------------


def test_save_model_checkpoint_roundtrip():
    """Save → load round-trip produces identical outputs (new format)."""
    model = GomokuNet()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)

    try:
        save_model_checkpoint(model, tmp)
        data = torch.load(str(tmp), map_location="cpu", weights_only=False)
        assert isinstance(data, dict)
        assert "state_dict" in data
        assert "arch_config" in data
        assert data["arch_config"]["num_res_blocks"] == 10
        loaded = GomokuNet()
        loaded.load_state_dict(data["state_dict"])
        model.eval()
        loaded.eval()

        x = torch.randn(2, 3, 15, 15)
        with torch.no_grad():
            lp1, v1 = model(x)
            lp2, v2 = loaded(x)
        assert torch.allclose(lp1, lp2)
        assert torch.allclose(v1, v2)
    finally:
        tmp.unlink()


def test_train_on_examples_reduces_loss():
    """Training on a trivial dataset (all examples share the same input and
    target) must decrease the loss, proving the training loop works."""
    # All examples share a single state so the model can overfit quickly.
    shared_state = torch.randn(3, 15, 15)
    # One-hot target policy — all probability mass on position 0.
    target_policy = torch.zeros(225)
    target_policy[0] = 1.0
    target_value = 0.8

    examples = [
        TrainingExample(
            state=shared_state.clone(),
            policy=target_policy.clone(),
            value=target_value,
        )
        for _ in range(64)
    ]

    model = GomokuNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Capture initial eval loss
    model.eval()
    with torch.no_grad():
        s = torch.stack([ex.state for ex in examples])
        lp, v = model(s)
        tp = torch.stack([ex.policy for ex in examples])
        tv = torch.tensor([ex.value for ex in examples], dtype=torch.float32).unsqueeze(1)
        _, _, initial = compute_loss(lp, v, tp, tv)

    train_on_examples(model, optimizer, examples, batch_size=16)

    # Compute final eval loss — must be lower after training
    model.eval()
    with torch.no_grad():
        s2 = torch.stack([ex.state for ex in examples])
        lp2, v2 = model(s2)
        _, _, final = compute_loss(lp2, v2, tp, tv)

    assert final.item() < initial.item()


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------


def test_run_evaluation_smoke():
    """2-game eval should not crash and return a float in [0, 1]."""
    # Create two identical untrained models and save them as checkpoints.
    model = GomokuNet()
    with (
        tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f1,
        tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f2,
    ):
        tmp_new = Path(f1.name)
        tmp_best = Path(f2.name)

    try:
        save_model_checkpoint(model, tmp_new)
        save_model_checkpoint(model, tmp_best)

        win_rate = run_evaluation(tmp_new, tmp_best, num_games=2, device="cpu", num_simulations=10)
        assert isinstance(win_rate, float)
        assert 0.0 <= win_rate <= 1.0
    finally:
        tmp_new.unlink()
        tmp_best.unlink()


# ---------------------------------------------------------------------------
# Integration: one-iteration training loop smoke test
# ---------------------------------------------------------------------------


def test_main_one_iteration_no_crash():
    """Run main() for 1 iteration with minimal settings. Should not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            from selfplay.train import main

            main(
                num_iterations=1,
                games_per_iteration=1,
                batch_size=16,
                mcts_simulations=10,
                eval_frequency=999,  # never eval during this smoke test
            )
            # Check that checkpoints were created.
            assert Path("checkpoints/best.pt").exists()
            assert Path("checkpoints/latest.pt").exists()
        finally:
            os.chdir(orig_cwd)


def test_save_model_checkpoint_nondefault_arch_roundtrip():
    """Non-default architecture checkpoint saves and restores correctly."""
    model = GomokuNet(
        num_res_blocks=5,
        num_hidden_channels=64,
        use_se=False,
        use_attention=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)

    try:
        save_model_checkpoint(model, tmp)
        data = torch.load(str(tmp), map_location="cpu", weights_only=False)
        assert data["arch_config"]["num_res_blocks"] == 5
        assert data["arch_config"]["num_hidden_channels"] == 64
        assert data["arch_config"]["use_se"] is False
        assert data["arch_config"]["use_attention"] is False
        loaded = GomokuNet.from_config(data["arch_config"])
        loaded.load_state_dict(data["state_dict"])
        model.eval()
        loaded.eval()

        x = torch.randn(2, 3, 15, 15)
        with torch.no_grad():
            lp1, v1 = model(x)
            lp2, v2 = loaded(x)
        assert torch.allclose(lp1, lp2)
        assert torch.allclose(v1, v2)
    finally:
        tmp.unlink()
