"""Tests for selfplay.train — loss function, training, evaluation."""
from __future__ import annotations

import csv
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from neural.model import GomokuNet
from selfplay.selfplay import TrainingExample
from selfplay.train import (
    _append_csv_row,
    _init_csv_log,
    compute_loss,
    load_training_state,
    run_evaluation,
    save_model_checkpoint,
    save_training_state,
    set_seed,
    train_on_examples,
)


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
    target = torch.zeros(1, 225)
    target[0, 0] = 1.0
    logits = torch.zeros(1, 225)
    logits[0, 0] = 100.0
    log_policy = torch.log_softmax(logits, dim=1)
    value = torch.zeros(1, 1)
    value_target = torch.zeros(1, 1)

    policy_loss, _, _ = compute_loss(log_policy, value, target, value_target)
    assert policy_loss.item() < 0.01


def test_compute_loss_policy_wrong_prediction():
    import math
    target = torch.tensor([[1.0] + [0.0] * 224])
    log_policy = torch.full((1, 225), math.log(1.0 / 225))
    value = torch.zeros(1, 1)
    value_target = torch.zeros(1, 1)

    policy_loss, _, _ = compute_loss(log_policy, value, target, value_target)
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
    assert value_loss.item() > 1.0


# ---------------------------------------------------------------------------
# save_model_checkpoint & train_on_examples
# ---------------------------------------------------------------------------


def test_save_model_checkpoint_roundtrip():
    """Save -> load round-trip produces identical outputs (new format)."""
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


def test_train_on_examples_reduces_loss():
    shared_state = torch.randn(3, 15, 15)
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

    model.eval()
    with torch.no_grad():
        s = torch.stack([ex.state for ex in examples])
        lp, v = model(s)
        tp = torch.stack([ex.policy for ex in examples])
        tv = torch.tensor([ex.value for ex in examples], dtype=torch.float32).unsqueeze(1)
        _, _, initial = compute_loss(lp, v, tp, tv)

    train_on_examples(model, optimizer, examples, batch_size=16)

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
# Deterministic seeding
# ---------------------------------------------------------------------------


def test_set_seed_determinism():
    """Identical seeds must produce identical random sequences."""
    set_seed(42)
    py_a = [random.random() for _ in range(5)]
    set_seed(42)
    py_b = [random.random() for _ in range(5)]
    assert py_a == py_b, "Python random: same seed -> same sequence"

    set_seed(7)
    np_a = [np.random.random() for _ in range(3)]
    set_seed(7)
    np_b = [np.random.random() for _ in range(3)]
    assert np_a == np_b, "NumPy: same seed -> same sequence"

    set_seed(99)
    torch_a = [torch.randn(1).item() for _ in range(3)]
    set_seed(99)
    torch_b = [torch.randn(1).item() for _ in range(3)]
    assert torch_a == torch_b, "Torch: same seed -> same sequence"


# ---------------------------------------------------------------------------
# Training state save / load round-trip
# ---------------------------------------------------------------------------


def _make_training_state(device: str = "cpu") -> tuple:
    """Create a minimal training state, save it, and return the objects."""
    model = GomokuNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler("cuda") if "cuda" in device else None

    x = torch.randn(1, 3, 15, 15).to(device)
    target = torch.tensor([[0.5]]).to(device)
    for _ in range(3):
        optimizer.zero_grad()
        log_policy, value = model(x)
        loss = (value - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if scaler is not None:
            scaler.update()

    tmpdir = tempfile.mkdtemp()
    state_path = Path(tmpdir) / "training_state.pt"
    save_training_state(state_path, model, optimizer, scheduler, scaler, iteration=5)

    return model, optimizer, scheduler, scaler, 5, state_path


def test_training_state_roundtrip():
    """Save and restore training state -- weights, optimizer, iteration."""
    model, optimizer, scheduler, scaler, saved_iter, state_path = _make_training_state("cpu")

    fresh_model = GomokuNet()
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=0.001)
    fresh_sched = CosineAnnealingLR(fresh_opt, T_max=100)

    loaded_iter = load_training_state(state_path, fresh_model, fresh_opt, fresh_sched, None)
    assert loaded_iter == saved_iter

    x = torch.randn(1, 3, 15, 15)
    model.eval()
    fresh_model.eval()
    with torch.no_grad():
        lp1, v1 = model(x)
        lp2, v2 = fresh_model(x)
    assert torch.allclose(lp1, lp2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)

    try:
        state_path.unlink()
        state_path.parent.rmdir()
    except OSError:
        pass


def test_optimizer_restoration():
    """Optimizer state (Adam momentum) survives save/load round-trip."""
    model, optimizer, scheduler, scaler, saved_iter, state_path = _make_training_state("cpu")

    fresh_model = GomokuNet()
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=0.001)
    fresh_sched = CosineAnnealingLR(fresh_opt, T_max=100)

    load_training_state(state_path, fresh_model, fresh_opt, fresh_sched, None)

    for pg_orig, pg_loaded in zip(optimizer.param_groups, fresh_opt.param_groups):
        assert pg_orig["lr"] == pg_loaded["lr"]
        assert pg_orig["weight_decay"] == pg_loaded["weight_decay"]

    orig_state = optimizer.state_dict()
    loaded_state = fresh_opt.state_dict()
    assert set(orig_state.keys()) == set(loaded_state.keys())
    assert orig_state["state"].keys() == loaded_state["state"].keys()

    try:
        state_path.unlink()
        state_path.parent.rmdir()
    except OSError:
        pass


def test_scheduler_restoration():
    """LR scheduler state survives save/load round-trip."""
    model, optimizer, scheduler, scaler, saved_iter, state_path = _make_training_state("cpu")

    fresh_model = GomokuNet()
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=0.001)
    fresh_sched = CosineAnnealingLR(fresh_opt, T_max=100)
    old_lr = fresh_sched.get_last_lr()[0]

    load_training_state(state_path, fresh_model, fresh_opt, fresh_sched, None)
    restored_lr = fresh_sched.get_last_lr()[0]
    original_lr = scheduler.get_last_lr()[0]

    assert abs(restored_lr - original_lr) < 1e-6, (
        f"LR mismatch: original={original_lr:.8f}, restored={restored_lr:.8f}"
    )
    assert abs(original_lr - old_lr) > 1e-8, (
        "Scheduler should have advanced past initial LR after training steps"
    )

    try:
        state_path.unlink()
        state_path.parent.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


def test_csv_logging():
    """CSV log creation and row appending must produce a valid, parseable CSV."""
    tmpdir = tempfile.mkdtemp()
    csv_path = Path(tmpdir) / "training_log.csv"

    try:
        _init_csv_log(csv_path)
        assert csv_path.exists()

        _append_csv_row(csv_path, {
            "iteration": 1,
            "loss": "2.345678",
            "policy_loss": "1.234567",
            "value_loss": "1.111111",
            "entropy": "3.210000",
            "grad_norm": "0.500000",
            "learning_rate": "0.00100000",
            "buffer_size": 5000,
            "simulations": 800,
            "iteration_runtime_sec": "12.34",
        })

        with open(str(csv_path), newline="") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 2, f"Expected header + 1 row, got {len(reader)}"
        assert reader[0] == [
            "iteration", "loss", "policy_loss", "value_loss", "entropy",
            "grad_norm", "learning_rate", "buffer_size", "simulations",
            "iteration_runtime_sec",
        ]
        assert reader[1][0] == "1"

        _append_csv_row(csv_path, {
            "iteration": 2,
            "loss": "2.100000",
            "policy_loss": "1.100000",
            "value_loss": "1.000000",
            "entropy": "3.100000",
            "grad_norm": "0.400000",
            "learning_rate": "0.00099000",
            "buffer_size": 6000,
            "simulations": 800,
            "iteration_runtime_sec": "10.50",
        })

        with open(str(csv_path), newline="") as f:
            reader = list(csv.reader(f))
        assert len(reader) == 3, f"Expected header + 2 rows, got {len(reader)}"

    finally:
        try:
            csv_path.unlink()
            csv_path.parent.rmdir()
        except OSError:
            pass


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
                eval_frequency=999,
                seed=42,
            )
            assert Path("checkpoints/best.pt").exists()
            assert Path("checkpoints/latest.pt").exists()
            assert Path("data/training_state.pt").exists()
            assert Path("data/training_log.csv").exists()
            with open("data/training_log.csv", newline="") as f:
                rows = list(csv.reader(f))
            assert len(rows) >= 2
            assert rows[0][0] == "iteration"
        finally:
            os.chdir(orig_cwd)


def test_main_resume_from_training_state():
    """Interrupted run must resume from the correct iteration."""
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
                eval_frequency=999,
                seed=42,
            )
            assert Path("data/training_state.pt").exists()

            main(
                num_iterations=2,
                games_per_iteration=1,
                batch_size=16,
                mcts_simulations=10,
                eval_frequency=999,
                seed=42,
            )
            state = torch.load(
                "data/training_state.pt", map_location="cpu", weights_only=False
            )
            assert state["iteration"] == 2

            with open("data/training_log.csv", newline="") as f:
                rows = list(csv.reader(f))
            iterations = [int(r[0]) for r in rows[1:]]
            assert iterations == [1, 2], f"Expected iterations [1, 2], got {iterations}"
        finally:
            os.chdir(orig_cwd)
