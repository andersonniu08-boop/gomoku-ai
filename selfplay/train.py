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
from selfplay.elo import EloTracker
from selfplay.mcts import MCTS
from selfplay.replay_buffer import ReplayBuffer
from selfplay.selfplay import SelfPlayGame, TrainingExample


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
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), str(tmp))
    tmp.rename(path)


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

        # Worker files are written without augmentation (augment=False).
        # The ReplayBuffer applies random D₄ symmetry on retrieval,
        # giving equivalent data diversity at 1/8th the memory.
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
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 5.0,
) -> dict[str, float]:
    """Train the model for one pass over *examples*.

    Examples are shuffled before batching.  Returns a dict with keys
    ``loss``, ``policy_loss``, ``value_loss``, and ``entropy``.

    When *scaler* is provided (CUDA with mixed precision), the forward
    pass runs under ``torch.amp.autocast`` and gradients are unscaled
    before the optimizer step.
    """
    random.shuffle(examples)
    model.train()

    use_amp = scaler is not None
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    total_grad_norm = 0.0
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

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            log_policy, value = model(states)
            policy_loss, value_loss, total = compute_loss(
                log_policy, value, target_policies, target_values
            )

        # Policy entropy: H(P) = -sum(P * log P)
        probs = torch.exp(log_policy)
        entropy = -(probs * log_policy).sum(dim=1).mean()

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += total.item()
        total_policy += policy_loss.item()
        total_value += value_loss.item()
        total_entropy += entropy.item()
        total_grad_norm += float(gn.item()) if isinstance(gn, torch.Tensor) else float(gn)
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "policy_loss": total_policy / n,
        "value_loss": total_value / n,
        "entropy": total_entropy / n,
        "grad_norm": total_grad_norm / n,
    }


def _play_eval_game(
    black_wrapper: GomokuInferenceWrapper,
    white_wrapper: GomokuInferenceWrapper,
    num_simulations: int = 200,
) -> Player | None:
    """Play one deterministic game between two different models.

    Uses a strong search (800 sims by default) so that model-comparison
    games are a meaningful strength test, not a lottery.  Weak-search
    evaluation produces noisy win rates that can promote regressions.

    Returns the winner (Player.BLACK, Player.WHITE, or None for draw).
    """
    board = Board()
    black_mcts = MCTS(black_wrapper, num_simulations=num_simulations, threat_override=True, tree_reuse=False)
    white_mcts = MCTS(white_wrapper, num_simulations=num_simulations, threat_override=True, tree_reuse=False)

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
    num_simulations: int = 200,
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
            winner = _play_eval_game(new_wrapper, best_wrapper, num_simulations=num_simulations)
        else:
            winner = _play_eval_game(best_wrapper, new_wrapper, num_simulations=num_simulations)

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
    mcts_simulations: int = 800,
    eval_simulations: int = 200,
    sim_schedule: list[tuple[int, int]] | None = None,
    device: Optional[str] = None,
    game_examples_dir: str | Path = "game_examples/",
    max_grad_norm: float = 5.0,
    resignation_warmup_iters: int = 10,
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    # Cosine annealing over the full training run (~10 batches / iteration).
    # T_max spans all batches so the LR reaches 0 at the final step,
    # avoiding per-iteration LR spikes that destabilise late-stage learning.
    batches_per_iter = max(1, (batch_size * 10 + batch_size - 1) // batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations * batches_per_iter)

    # Mixed-precision GradScaler for CUDA training.
    scaler = torch.amp.GradScaler("cuda") if "cuda" in device else None

    # ------------------------------------------------------------------
    # Elo tracking — persist across training sessions.
    # ------------------------------------------------------------------
    elo_path = data_dir / "elo_state.json"
    elo_tracker = EloTracker()
    if elo_path.exists():
        elo_tracker.load(elo_path)
        print(f"Elo ratings loaded ({len(elo_tracker.known_checkpoints())} known)")
    elo_tracker.register_checkpoint("best.pt", iteration=0)
    elo_tracker.register_checkpoint("latest.pt")

    for iteration in range(1, num_iterations + 1):
        iter_start = time.monotonic()

        # --- Phase A: Collect self-play data ---
        # 1. Ingest any worker-generated files (bonus data from distributed
        #    workers, if any are running).
        ingested = ingest_game_files(buffer, game_dir, consumed_dir)
        if ingested > 0:
            print(f"\nIteration {iteration}: Ingested {ingested} game files, "
                  f"buffer now {len(buffer)}")

        # 2. Generate fresh self-play games with the LATEST model every
        #    iteration.  This is the core of AlphaZero policy iteration:
        #    improved model → improved search → improved policy targets.
        current_sims = mcts_simulations
        if sim_schedule is not None:
            for start_iter, sims in sorted(sim_schedule, reverse=True):
                if iteration >= start_iter:
                    current_sims = sims
                    break

        wrapper = GomokuInferenceWrapper(latest_path, device=device)
        resign_thresh = (
            None if iteration <= resignation_warmup_iters
            else -0.9
        )
        game_runner = SelfPlayGame(
            wrapper,
            num_simulations=current_sims,
            threat_override=True,
            augment=False,
            resignation_threshold=resign_thresh,
        )

        for _ in range(games_per_iteration):
            examples = game_runner.play()
            buffer.add_examples(examples)

        torch.save(buffer.state_dict(), str(buffer_path))

        # 3. Sample a training batch from the full replay buffer.
        #    Mixes fresh examples with older ones from previous iterations.
        train_examples = buffer.sample(min(len(buffer), batch_size * 10))

        # --- Phase B: Train ---
        # Reload latest weights into the persistent model so optimizer
        # retains momentum across checkpoint versions.
        state_dict = torch.load(
            str(latest_path), map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)

        metrics = train_on_examples(
            model, optimizer, train_examples, batch_size, scheduler,
            device=device, scaler=scaler, max_grad_norm=max_grad_norm,
        )

        save_model_checkpoint(model, latest_path)

        elapsed = time.monotonic() - iter_start
        print(f"  loss={metrics['loss']:.4f}  "
              f"policy={metrics['policy_loss']:.4f}  "
              f"value={metrics['value_loss']:.4f}  "
              f"entropy={metrics['entropy']:.4f}  "
              f"gnorm={metrics['grad_norm']:.2f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}  "
              f"buf={len(buffer)}  "
              f"sims={current_sims}  "
              f"t={elapsed:.1f}s")

        # --- Evaluation ---
        if iteration % eval_frequency == 0:
            print(f"  Evaluating latest vs best ({eval_games} games)...")
            win_rate = run_evaluation(
                latest_path, best_path, num_games=eval_games, device=device,
                num_simulations=eval_simulations,
            )

            # Record Elo update from the evaluation match.
            elo_tracker.record_match(
                "latest.pt", "best.pt", win_rate, eval_games,
                iteration=iteration,
            )
            latest_rating = elo_tracker.get_rating("latest.pt")
            best_rating = elo_tracker.get_rating("best.pt")

            if win_rate >= eval_threshold:
                pct = round(win_rate * 100)
                promoted_name = f"best_iter{iteration:03d}_win{pct}.pt"
                save_model_checkpoint(model, checkpoints_dir / promoted_name)
                save_model_checkpoint(model, best_path)
                # Register the promoted checkpoint, inheriting latest's rating.
                elo_tracker.register_checkpoint(
                    promoted_name, iteration=iteration,
                    rating=latest_rating,
                )
                print(f"  Promoted!  Win rate: {win_rate:.2%}  "
                      f"→ {promoted_name}")
            else:
                print(f"  Not promoted.  Win rate: {win_rate:.2%}  "
                      f"(threshold {eval_threshold:.0%})")

            # Show Elo summary.
            print(f"  Elo: latest={latest_rating:.1f}, "
                  f"best={best_rating:.1f}")

        # Save Elo and buffer state after each iteration.
        elo_tracker.save(elo_path)

        # Print replay diversity stats every iteration so the operator
        # can monitor training-data quality (opening convergence, move
        # distribution skew, win/loss balance).
        if len(buffer) > 0:
            stats = buffer.diversity_stats()
            opens = stats["openings_top"]
            top_opens = "  ".join(
                f"#{h[:8]}…×{c}" for h, c in opens[:3]
            ) if opens else "(none)"
            print(
                f"  replay: {stats['total_positions']} pos, "
                f"{stats['unique_openings']} openings "
                f"({stats['opening_diversity_ratio']:.1%} unique), "
                f"moves={stats['move_distribution']}, "
                f"vals={stats['value_distribution']}"
            )
            if opens:
                print(f"  top openings: {top_opens}")

        # Brief pause so workers can write more files.
        time.sleep(2)

    print(f"\nDone.  Best model: {best_path}")


if __name__ == "__main__":
    main()
