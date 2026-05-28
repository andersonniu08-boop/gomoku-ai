"""Distributed self-play worker — generates games and writes them to disk.

Run with: ``python -m selfplay.worker``
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from neural.wrapper import GomokuInferenceWrapper
from selfplay.selfplay import SelfPlayGame

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False


def _on_shutdown(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        raise SystemExit(1)
    _SHUTDOWN_REQUESTED = True
    print("\n[worker] Shutdown requested — finishing current game...")


signal.signal(signal.SIGINT, _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_worker(
    checkpoint_dir: str | Path = "checkpoints/",
    output_dir: str | Path = "game_examples/",
    num_games: int | None = None,
    worker_id: str = "auto",
    num_simulations: int = 800,
    c_puct: float = 2.5,
    temperature_stages: list[tuple[int, float]] | None = None,
    checkpoint_poll_sec: int = 5,
) -> None:
    """Run a self-play worker that writes game files for the trainer.

    Parameters:
        checkpoint_dir: Directory containing ``latest.pt`` (watched for
            changes so the worker always uses the freshest policy).
        output_dir: Directory where ``game_*.pt`` files are written.
        num_games: If set, stop after this many games.  ``None`` = run
            indefinitely until SIGINT/SIGTERM.
        worker_id: Prefix used in game file names.  ``"auto"`` resolves
            to ``<hostname>-<pid>``.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        temperature: Visit-count exponent for early-game move sampling.
        temperature_threshold: Move number after which temperature is
            annealed to 0 (deterministic).
        checkpoint_poll_sec: Seconds between mtime checks on
            ``checkpoint_dir/latest.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if worker_id == "auto":
        worker_id = f"{socket.gethostname()}-{os.getpid()}"

    # --- Auto-detect device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Wait for initial checkpoint ---
    latest_path = checkpoint_dir / "latest.pt"
    waited = 0.0
    while not latest_path.exists():
        if waited >= 60:
            raise FileNotFoundError(
                f"No checkpoint found at {latest_path} after 60s — "
                "start the trainer first."
            )
        time.sleep(1.0)
        waited += 1.0

    wrapper = GomokuInferenceWrapper(str(latest_path), device=device)
    last_mtime = latest_path.stat().st_mtime
    last_poll = time.monotonic()
    print(f"[worker {worker_id}] Loaded checkpoint from {latest_path}  (device={device})")

    games_played = 0
    seq = 0

    while not _SHUTDOWN_REQUESTED:
        if num_games is not None and games_played >= num_games:
            break

        # --- Reload checkpoint if updated (time-based polling) ---
        now = time.monotonic()
        if now - last_poll >= checkpoint_poll_sec:
            if latest_path.exists():
                cur_mtime = latest_path.stat().st_mtime
                if cur_mtime > last_mtime:
                    wrapper = GomokuInferenceWrapper(str(latest_path), device=device)
                    last_mtime = cur_mtime
                    print(f"[worker {worker_id}] Reloaded updated checkpoint")
            last_poll = now

        # --- Play one game ---
        game = SelfPlayGame(
            wrapper,
            num_simulations=num_simulations,
            c_puct=c_puct,
            threat_override=True,
            augment=False,
            temperature_stages=temperature_stages
            if temperature_stages is not None
            else [(0, 1.0), (15, 0.5), (30, 0.0)],
        )

        t0 = time.monotonic()
        examples = game.play()
        duration = time.monotonic() - t0

        if not examples:
            continue

        seq += 1
        games_played += 1

        # Infer outcome from the last example's value.
        # +1 = last mover won, -1 = last mover lost, 0 = draw.
        last_val = examples[-1].value
        if last_val > 0.5:
            winner_str = "LAST_MOVER_WON"
        elif last_val < -0.5:
            winner_str = "LAST_MOVER_LOST"
        else:
            winner_str = "DRAW"

        # --- Write game file atomically ---
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem = f"game_{worker_id}_{seq:04d}_{ts}"

        tmp_path = output_dir / f".{stem}.tmp"
        final_path = output_dir / f"{stem}.pt"
        meta_path = output_dir / f"{stem}_meta.json"

        torch.save(examples, str(tmp_path))

        meta = {
            "worker_id": worker_id,
            "game_length": len(examples),
            "duration_sec": round(duration, 3),
            "winner": winner_str,
            "num_examples": len(examples),
        }
        meta_path.write_text(json.dumps(meta))

        tmp_path.rename(final_path)
        print(
            f"[worker {worker_id}] Game {seq} ({games_played}): "
            f"{len(examples)} examples, {duration:.1f}s"
        )

    print(f"[worker {worker_id}] Stopped.  {games_played} games played.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    p = argparse.ArgumentParser(description="Gomoku self-play worker")
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--output-dir", default="game_examples/")
    p.add_argument("--num-games", type=int, default=None)
    p.add_argument("--worker-id", default="auto")
    p.add_argument("--num-simulations", type=int, default=800)
    p.add_argument("--c-puct", type=float, default=2.5)
    p.add_argument("--checkpoint-poll-sec", type=int, default=5)
    args = p.parse_args()

    run_worker(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_games=args.num_games,
        worker_id=args.worker_id,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        checkpoint_poll_sec=args.checkpoint_poll_sec,
    )


if __name__ == "__main__":
    _main()
