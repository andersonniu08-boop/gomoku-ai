# Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the AlphaZero training loop — loss computation, self-play data generation, mini-batch training with cosine annealing, model evaluation, and checkpoint management.

**Architecture:** Single file `selfplay/train.py` containing: `compute_loss` (pure loss function), `_play_eval_game` / `run_evaluation` (model-vs-model matches), and `main` (two-phase training loop orchestrating self-play → train → eval). Checkpoints save raw `state_dict` for InferenceWrapper compatibility. Initial model checkpoint is bootstrapped on first run.

**Tech Stack:** PyTorch, existing selfplay/engine/neural modules

---

## File Structure

| Action | Path | Purpose |
|--------|------|---------|
| Create | `selfplay/train.py` | Loss function, training, evaluation, main loop |
| Create | `tests/test_train.py` | Tests for loss, training, eval, checkpointing |

No existing files are modified. `selfplay/__init__.py` is not updated — training is an entry point, not a library import.

---

### Task 1: `compute_loss` — pure loss function

**Files:**
- Create: `selfplay/train.py` (partial)
- Create: `tests/test_train.py` (partial)

- [ ] **Step 1: Write the test file with loss tests**

```python
"""Tests for selfplay.train — loss function, training, evaluation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from neural.model import GomokuNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.selfplay import TrainingExample
from selfplay.train import (
    compute_loss,
    run_evaluation,
    save_model_checkpoint,
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
    target = torch.ones(1, 225) / 225
    log_policy = torch.log(target + 1e-8)
    value = torch.zeros(1, 1)
    value_target = torch.zeros(1, 1)

    policy_loss, _, _ = compute_loss(log_policy, value, target, value_target)
    assert policy_loss.item() < 1e-4


def test_compute_loss_policy_wrong_prediction():
    target = torch.tensor([[1.0] + [0.0] * 224])
    log_policy = torch.zeros(1, 225)  # uniform log-prob ≈ -5.42 each
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL — ModuleNotFoundError for `selfplay.train`

- [ ] **Step 3: Write the `compute_loss` function and file header**

Create `selfplay/train.py`:

```python
"""AlphaZero training loop with self-play data generation and model evaluation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from engine.board import Board, Player
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS
from selfplay.replay_buffer import ReplayBuffer
from selfplay.selfplay import SelfPlayGame


def compute_loss(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AlphaZero training losses.

    Args:
        log_policy: (B, 225) log-softmax output from network.
        value: (B, 1) tanh value output from network.
        target_policy: (B, 225) MCTS visit-count distribution.
        target_value: (B, 1) game outcome in [-1, 1].

    Returns:
        (policy_loss, value_loss, total_loss) — scalar tensors.
    """
    policy_loss = -(target_policy * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    total_loss = policy_loss + value_loss
    return policy_loss, value_loss, total_loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: 5 passed

---

### Task 2: Model checkpoint and training helpers

**Files:**
- Modify: `selfplay/train.py` (append helpers)

- [ ] **Step 1: Append `save_model_checkpoint` and `train_on_examples` to train.py**

```python
def save_model_checkpoint(model: GomokuNet, path: str | Path) -> None:
    """Save model state_dict for InferenceWrapper compatibility."""
    torch.save(model.state_dict(), str(path))


def train_on_examples(
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    examples: list[TrainingExample],
    batch_size: int,
    scheduler: Optional[CosineAnnealingLR] = None,
    device: Optional[str] = None,
) -> float:
    """Train the model for one pass over *examples*.

    Examples are shuffled before batching.  Returns the average loss.
    """
    random.shuffle(examples)
    model.train()

    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        states = torch.stack([ex.state for ex in batch])
        target_policies = torch.stack([ex.policy for ex in batch])
        target_values = torch.tensor(
            [ex.value for ex in batch], dtype=torch.float32
        ).unsqueeze(1)

        if device is not None:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

        log_policy, value = model(states)
        _, _, loss = compute_loss(log_policy, value, target_policies, target_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
```

- [ ] **Step 2: Write tests for training and checkpointing**

Append to `tests/test_train.py`:

```python
# ---------------------------------------------------------------------------
# save_model_checkpoint & train_on_examples
# ---------------------------------------------------------------------------


def test_save_model_checkpoint_roundtrip():
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)

    try:
        save_model_checkpoint(model, tmp)
        loaded = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5)
        loaded.load_state_dict(torch.load(str(tmp), map_location="cpu", weights_only=True))

        x = torch.randn(2, 3, 15, 15)
        with torch.no_grad():
            lp1, v1 = model(x)
            lp2, v2 = loaded(x)
        assert torch.allclose(lp1, lp2)
        assert torch.allclose(v1, v2)
    finally:
        tmp.unlink()


def test_train_on_examples_reduces_loss():
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    examples = [
        TrainingExample(
            state=torch.randn(3, 15, 15),
            policy=torch.ones(225) / 225,
            value=0.0,
        )
        for _ in range(32)
    ]

    # Capture initial loss
    model.eval()
    with torch.no_grad():
        s = torch.stack([ex.state for ex in examples])
        lp, v = model(s)
        tp = torch.stack([ex.policy for ex in examples])
        tv = torch.tensor([ex.value for ex in examples], dtype=torch.float32).unsqueeze(1)
        _, _, initial = compute_loss(lp, v, tp, tv)

    final_loss = train_on_examples(model, optimizer, examples, batch_size=16)

    # Loss should decrease after training
    assert final_loss < initial.item()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_train.py -v -k "roundtrip or reduces_loss"`
Expected: FAIL — functions not defined

- [ ] **Step 4: Run all train tests to verify they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: 7 passed

---

### Task 3: Model evaluation — `run_evaluation`

**Files:**
- Modify: `selfplay/train.py` (append eval functions)

- [ ] **Step 1: Append `_play_eval_game` and `run_evaluation` to train.py**

```python
def _play_eval_game(
    black_wrapper: GomokuInferenceWrapper,
    white_wrapper: GomokuInferenceWrapper,
    num_simulations: int = 100,
) -> Player | None:
    """Play one deterministic game between two different models.

    Returns the winner (Player.BLACK, Player.WHITE, or None for draw).
    """
    board = Board()
    black_mcts = MCTS(black_wrapper, num_simulations=num_simulations, threat_override=True)
    white_mcts = MCTS(white_wrapper, num_simulations=num_simulations, threat_override=True)

    while not board.is_terminal():
        mcts = black_mcts if board.current_player == Player.BLACK else white_mcts
        move = mcts.select_move(board, temperature=0.0)
        board.make_move(*move)

    return board.check_win()


def run_evaluation(
    new_checkpoint: str | Path,
    best_checkpoint: str | Path,
    num_games: int = 100,
    device: Optional[str] = None,
) -> float:
    """Pit new model vs best model and return new model's win rate.

    Alternates which model plays Black to cancel first-move advantage.
    Each game is deterministic (temperature=0).
    """
    new_wins = 0.0

    for i in range(num_games):
        if i % 2 == 0:
            black_ckpt = new_checkpoint
            white_ckpt = best_checkpoint
        else:
            black_ckpt = best_checkpoint
            white_ckpt = new_checkpoint

        black_wrapper = GomokuInferenceWrapper(Path(black_ckpt), device=device or "cpu")
        white_wrapper = GomokuInferenceWrapper(Path(white_ckpt), device=device or "cpu")

        winner = _play_eval_game(black_wrapper, white_wrapper)

        if i % 2 == 0 and winner == Player.BLACK:
            new_wins += 1
        elif i % 2 == 1 and winner == Player.WHITE:
            new_wins += 1
        elif winner is None:
            new_wins += 0.5
        # else: new model lost

    return new_wins / num_games
```

- [ ] **Step 2: Write eval tests**

Append to `tests/test_train.py`:

```python
# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------


def test_run_evaluation_smoke():
    """2-game eval should not crash and return a float in [0, 1]."""
    # Create two identical untrained models and save them as checkpoints.
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5)
    with (
        tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f1,
        tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f2,
    ):
        tmp_new = Path(f1.name)
        tmp_best = Path(f2.name)

    try:
        save_model_checkpoint(model, tmp_new)
        save_model_checkpoint(model, tmp_best)

        win_rate = run_evaluation(tmp_new, tmp_best, num_games=2, device="cpu")
        assert isinstance(win_rate, float)
        assert 0.0 <= win_rate <= 1.0
    finally:
        tmp_new.unlink()
        tmp_best.unlink()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_train.py -v -k "eval"`
Expected: FAIL — `run_evaluation` not defined

- [ ] **Step 4: Run all train tests to verify they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: 8 passed

---

### Task 4: `main()` — training loop orchestration

**Files:**
- Modify: `selfplay/train.py` (append main function and `if __name__ == "__main__"` guard)

- [ ] **Step 1: Append `main` to train.py**

```python
def main(
    num_iterations: int = 50,
    games_per_iteration: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    eval_frequency: int = 5,
    eval_games: int = 100,
    eval_threshold: float = 0.55,
    mcts_simulations: int = 400,
    selfplay_temperature: float = 1.0,
    selfplay_temp_threshold: int = 15,
    device: Optional[str] = None,
) -> None:
    """Run the AlphaZero training loop.

    Two-phase iteration:
      1. Generate self-play games with the best model.
      2. Train on those examples, then evaluate and possibly promote.
    """
    checkpoints_dir = Path("checkpoints")
    data_dir = Path("data")
    checkpoints_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    best_path = checkpoints_dir / "best.pt"
    latest_path = checkpoints_dir / "latest.pt"
    buffer_path = data_dir / "replay_buffer.pt"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bootstrap: if no best checkpoint exists, create one from a fresh model.
    if not best_path.exists():
        model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64)
        save_model_checkpoint(model, best_path)
        save_model_checkpoint(model, latest_path)

    # Load or create replay buffer.
    if buffer_path.exists():
        buffer = torch.load(str(buffer_path), map_location="cpu", weights_only=False)
    else:
        buffer = ReplayBuffer(max_size=500_000)

    print(f"Device: {device}")
    print(f"Buffer size: {len(buffer)}")
    print(f"Iterations: {num_iterations}, games/iter: {games_per_iteration}, batch: {batch_size}")

    for iteration in range(1, num_iterations + 1):
        # --- Phase A: Self-play with best model ---
        wrapper = GomokuInferenceWrapper(best_path, device=device)
        game = SelfPlayGame(
            wrapper,
            num_simulations=mcts_simulations,
            temperature=selfplay_temperature,
            temperature_threshold=selfplay_temp_threshold,
            threat_override=True,
            augment=True,
        )

        iter_examples: list = []
        for g in range(games_per_iteration):
            examples = game.play()
            iter_examples.extend(examples)

        buffer.add_examples(iter_examples)
        torch.save(buffer, str(buffer_path))

        print(
            f"\nIteration {iteration}: {len(iter_examples)} examples from "
            f"{games_per_iteration} games, buffer now {len(buffer)}"
        )

        # --- Phase B: Train ---
        model = GomokuNet(
            board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64
        )
        model.load_state_dict(
            torch.load(str(latest_path), map_location=device, weights_only=True)
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        total_steps = (len(iter_examples) + batch_size - 1) // batch_size
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        avg_loss = train_on_examples(
            model, optimizer, iter_examples, batch_size, scheduler, device=device
        )

        save_model_checkpoint(model, latest_path)
        print(f"  Training loss: {avg_loss:.4f}  (lr={scheduler.get_last_lr()[0]:.6f})")

        # --- Evaluation ---
        if iteration % eval_frequency == 0:
            print(f"  Evaluating latest vs best ({eval_games} games)...")
            win_rate = run_evaluation(latest_path, best_path, num_games=eval_games, device=device)

            if win_rate >= eval_threshold:
                save_model_checkpoint(model, best_path)
                print(f"  ✅ Promoted!  Win rate: {win_rate:.2%}")
            else:
                print(f"  ❌ Not promoted.  Win rate: {win_rate:.2%}  (threshold {eval_threshold:.0%})")

    print(f"\nDone.  Best model: {best_path}")
```

- [ ] **Step 2: Write integration test for the full loop**

Append to `tests/test_train.py`:

```python
# ---------------------------------------------------------------------------
# Integration: one-iteration training loop smoke test
# ---------------------------------------------------------------------------


def test_main_one_iteration_no_crash():
    """Run main() for 1 iteration with minimal settings. Should not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
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
```

- [ ] **Step 3: Run integration test to verify it fails (main not defined)**

Run: `python -m pytest tests/test_train.py -v -k "integration"`
Expected: FAIL — `main` not defined

- [ ] **Step 4: Run all train tests**

Run: `python -m pytest tests/test_train.py -v`
Expected: 9 passed

---

### Task 5: Add `if __name__ == "__main__"` guard and verify full test suite

**Files:**
- Modify: `selfplay/train.py` (append guard at end)

- [ ] **Step 1: Append the entry-point guard**

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all project tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: 84 passed (75 existing + 9 new)

---

### Task 6: Commit

- [ ] **Step 1: Stage and commit all new files**

```bash
git add selfplay/train.py tests/test_train.py docs/superpowers/specs/2026-05-21-training-loop-design.md docs/superpowers/plans/2026-05-21-training-loop.md
git commit -m "feat: add AlphaZero training loop with self-play, cosine annealing, and model evaluation"
```
