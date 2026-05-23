"""AlphaZero training loop with self-play data generation and model evaluation."""

from __future__ import annotations

import random
import time
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
from selfplay.selfplay import SelfPlayGame, TrainingExample, augment_examples


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


def save_model_checkpoint(model: GomokuNet, path: str | Path) -> None:
    """Save model state_dict for InferenceWrapper compatibility."""
    torch.save(model.state_dict(), str(path))


def ingest_game_files(
    buffer: ReplayBuffer,
    game_dir: Path,
    consumed_dir: Path,
    max_consumed: int = 1000,
) -> int:
    """Ingest pending .pt game files into the replay buffer.

    Skips files already present in *consumed_dir* (dedup by filename).
    Successfully ingested files are moved to *consumed_dir*, which is
    trimmed to the most recent *max_consumed* files.

    Returns the number of files ingested.
    """
    consumed_dir.mkdir(parents=True, exist_ok=True)

    consumed_names = {
        p.name for p in consumed_dir.iterdir() if p.suffix == ".pt"
    }

    pending = sorted(
        [p for p in game_dir.glob("game_*.pt") if p.name not in consumed_names]
    )
    if not pending:
        return 0

    ingested = 0
    for path in pending:
        try:
            examples = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            print(f"  [trainer] Skipping unreadable file: {path.name}")
            continue

        if not isinstance(examples, list) or not examples:
            print(f"  [trainer] Skipping empty/invalid file: {path.name}")
            continue

        # Worker files are written raw (augment=False).  Apply D₄ symmetries
        # on ingest so the buffer always contains augmented data regardless
        # of source, matching the local fallback path (SelfPlayGame w/
        # augment=True).  8× data per game.
        examples = augment_examples(examples)
        buffer.add_examples(examples)
        path.rename(consumed_dir / path.name)
        ingested += 1

    # Trim consumed directory to *max_consumed* most-recent files.
    consumed_files = sorted(
        consumed_dir.glob("game_*.pt"), key=lambda p: p.stat().st_mtime
    )
    for old in consumed_files[:-max_consumed]:
        meta_name = old.name.replace(".pt", "_meta.json")
        (consumed_dir / meta_name).unlink(missing_ok=True)
        old.unlink(missing_ok=True)

    return ingested


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

    new_wrapper = GomokuInferenceWrapper(Path(new_checkpoint), device=device or "cpu")
    best_wrapper = GomokuInferenceWrapper(Path(best_checkpoint), device=device or "cpu")

    for i in range(num_games):
        if i % 2 == 0:
            winner = _play_eval_game(new_wrapper, best_wrapper)
        else:
            winner = _play_eval_game(best_wrapper, new_wrapper)

        if i % 2 == 0 and winner == Player.BLACK:
            new_wins += 1
        elif i % 2 == 1 and winner == Player.WHITE:
            new_wins += 1
        elif winner is None:
            new_wins += 0.5
        # else: new model lost

    return new_wins / num_games


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
    game_examples_dir: str | Path = "game_examples/",
) -> None:
    """Run the AlphaZero training loop.

    Two-phase iteration:
      1. Collect self-play data (from worker game files or local generation).
      2. Train on those examples, then evaluate and possibly promote.

    When *game_examples_dir* contains worker-produced game files they are
    ingested first.  If the replay buffer is still below *batch_size*,
    *games_per_iteration* local games are generated as a fallback.
    """
    checkpoints_dir = Path("checkpoints")
    data_dir = Path("data")
    game_dir = Path(game_examples_dir)
    consumed_dir = game_dir / "consumed"
    checkpoints_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    game_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoints_dir / "best.pt"
    latest_path = checkpoints_dir / "latest.pt"
    buffer_path = data_dir / "replay_buffer.pt"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bootstrap: if no best checkpoint exists, create one from a fresh model.
    if not best_path.exists():
        model = GomokuNet()
        save_model_checkpoint(model, best_path)
        save_model_checkpoint(model, latest_path)

    # Load or create replay buffer.
    if buffer_path.exists():
        data = torch.load(str(buffer_path), map_location="cpu", weights_only=False)
        buffer = ReplayBuffer.from_state_dict(data)
    else:
        buffer = ReplayBuffer(max_size=500_000)

    print(f"Device: {device}")
    print(f"Buffer size: {len(buffer)}")
    print(f"Iterations: {num_iterations}, batch: {batch_size}")
    print(f"Game dir: {game_dir}  (polling for worker-generated files)")
    print(f"Generating {games_per_iteration} local games/iter as fallback")

    # Persistent model, optimizer, and LR schedule across iterations.
    # Adam momentum/variance accumulates naturally rather than resetting
    # each iteration, giving stable training dynamics.
    model = GomokuNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing over the full training run (~10 batches / iteration).
    # T_max spans all batches so the LR reaches 0 at the final step,
    # avoiding per-iteration LR spikes that destabilise late-stage learning.
    batches_per_iter = max(1, (batch_size * 10 + batch_size - 1) // batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations * batches_per_iter)

    for iteration in range(1, num_iterations + 1):
        # --- Phase A: Collect self-play data ---
        # 1. Ingest any worker-generated files.
        ingested = ingest_game_files(buffer, game_dir, consumed_dir)
        if ingested > 0:
            print(f"\nIteration {iteration}: Ingested {ingested} game files, "
                  f"buffer now {len(buffer)}")

        # 2. Fallback: generate local games if workers haven't produced enough.
        if len(buffer) < batch_size:
            print(f"  Buffer too small ({len(buffer)} < {batch_size}), "
                  f"generating {games_per_iteration} local games...")
            wrapper = GomokuInferenceWrapper(latest_path, device=device)
            game_runner = SelfPlayGame(
                wrapper,
                num_simulations=mcts_simulations,
                temperature=selfplay_temperature,
                temperature_threshold=selfplay_temp_threshold,
                threat_override=True,
                augment=True,
            )

            for _ in range(games_per_iteration):
                examples = game_runner.play()
                buffer.add_examples(examples)

            torch.save(buffer.state_dict(), str(buffer_path))
            print(f"  Buffer now {len(buffer)} after local generation")
            # Use the locally-generated examples for training.
            train_examples = buffer.sample(min(len(buffer), batch_size * 10))
        else:
            # Buffer is healthy — sample from it.
            train_examples = buffer.sample(min(len(buffer), batch_size * 10))

        # --- Phase B: Train ---
        # Reload latest weights into the persistent model so optimizer
        # retains momentum across checkpoint versions.
        state_dict = torch.load(
            str(latest_path), map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)

        avg_loss = train_on_examples(
            model, optimizer, train_examples, batch_size, scheduler, device=device
        )

        save_model_checkpoint(model, latest_path)
        torch.save(buffer.state_dict(), str(buffer_path))
        print(f"  Training loss: {avg_loss:.4f}  "
              f"(lr={scheduler.get_last_lr()[0]:.6f})")

        # --- Evaluation ---
        if iteration % eval_frequency == 0:
            print(f"  Evaluating latest vs best ({eval_games} games)...")
            win_rate = run_evaluation(
                latest_path, best_path, num_games=eval_games, device=device
            )

            if win_rate >= eval_threshold:
                pct = round(win_rate * 100)
                promoted_name = f"best_iter{iteration:03d}_win{pct}.pt"
                save_model_checkpoint(model, checkpoints_dir / promoted_name)
                save_model_checkpoint(model, best_path)
                print(f"  Promoted!  Win rate: {win_rate:.2%}  → {promoted_name}")
            else:
                print(f"  Not promoted.  Win rate: {win_rate:.2%}  "
                      f"(threshold {eval_threshold:.0%})")

        # Brief pause so workers can write more files.
        time.sleep(2)

    print(f"\nDone.  Best model: {best_path}")


if __name__ == "__main__":
    main()
