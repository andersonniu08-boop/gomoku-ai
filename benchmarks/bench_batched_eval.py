#!/usr/bin/env python3
"""Benchmark MCTS search throughput with different batch sizes.

Measures simulations/second, GPU evaluation time, and total search time
for various ``batch_size`` settings.

Usage:
    python benchmarks/bench_batched_eval.py                   # CPU, untrained model
    python benchmarks/bench_batched_eval.py --device cuda     # GPU, untrained model
    python benchmarks/bench_batched_eval.py --checkpoint ../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path so imports work from benchmarks/.
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import torch

from engine.board import Board
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.evaluator import BatchedLeafEvaluator
from selfplay.mcts import MCTS
from selfplay.profiler import Profiler


def _build_mid_game_board() -> Board:
    """Return a realistic mid-game position with ~20 stones per side.

    Creates a position where MCTS will have a moderate branching factor
    (not trivially small, not pathologically large) so benchmark results
    generalise well.
    """
    board = Board()
    # A spread-out opening that produces a realistic neighbour set.
    placements = [
        (7, 7), (7, 8), (8, 7), (6, 7), (7, 6),
        (8, 8), (6, 8), (9, 7), (7, 9), (6, 6),
        (8, 6), (5, 7), (7, 5), (9, 8), (8, 9),
        (5, 8), (9, 6), (10, 7), (7, 10), (6, 9),
        (8, 5), (9, 9), (4, 7), (7, 4), (10, 8),
        (5, 9), (10, 6), (11, 7), (7, 11), (6, 10),
        (9, 5), (4, 8), (7, 3), (9, 10), (10, 9),
        (4, 9), (5, 10), (8, 11), (3, 7),
    ]
    for r, c in placements:
        if not board.is_terminal():
            board.make_move(r, c)
    return board


def _make_wrapper(device: str, checkpoint: str | None = None) -> GomokuInferenceWrapper:
    """Create a wrapper, loading a checkpoint or a fresh untrained model."""
    if checkpoint:
        return GomokuInferenceWrapper(Path(checkpoint), device=device)

    model = GomokuNet(board_size=15, in_channels=3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    wrapper = GomokuInferenceWrapper(tmp_path, device=device)
    tmp_path.unlink()
    return wrapper


def bench_batch_size(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    batch_size: int,
    num_simulations: int,
    *,
    label: str = "",
) -> dict:
    """Run one benchmark and return timing statistics."""
    profiler = Profiler()

    mcts = MCTS(
        wrapper,
        num_simulations=num_simulations,
        batch_size=batch_size,
        threat_override=False,
        profiler=profiler,
    )

    torch.cuda.synchronize() if wrapper.device.type == "cuda" else None
    t0 = time.perf_counter()

    result = mcts.search_with_stats(board)

    torch.cuda.synchronize() if wrapper.device.type == "cuda" else None
    wall_time = time.perf_counter() - t0

    # Gather per-phase timings from the profiler.
    # Convert ms → s for consistent reporting.
    timers = {k: v.total_ms / 1000.0 for k, v in profiler._timers.items()}
    neural_time = timers.get("search.neural_eval", 0.0)
    descend_time = timers.get("search.descend_batch", 0.0)
    backup_time = timers.get("search.expand_backup", 0.0)
    total_time = timers.get("search.total", wall_time)

    sims = num_simulations
    return {
        "label": label or f"batch={batch_size}",
        "batch_size": batch_size,
        "simulations": sims,
        "wall_time_s": wall_time,
        "sims_per_sec": sims / wall_time,
        "total_time_profiled_s": total_time,
        "neural_time_s": neural_time,
        "descend_time_s": descend_time,
        "backup_time_s": backup_time,
        "nodes_visited": len(result.visit_counts),
    }


def print_results(results: list[dict], *, warmup_results: list[dict] | None = None) -> None:
    """Print benchmark results as a formatted table."""
    if warmup_results:
        print("=== Warm-up Runs (discarded) ===")
        for r in warmup_results:
            print(f"  {r['label']}:  {r['sims_per_sec']:.1f} sim/s  "
                  f"({r['wall_time_s']:.2f}s wall)")
        print()

    print("=== Batched Evaluation Benchmarks ===")
    header = (
        f"{'Config':<20} {'Sims':>6} {'Wall(s)':>9} {'Sim/s':>9} "
        f"{'Neural(s)':>10} {'Descend(s)':>10} {'Backup(s)':>10} {'Nodes':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<20} {r['simulations']:>6} {r['wall_time_s']:>9.3f} "
            f"{r['sims_per_sec']:>9.1f} "
            f"{r['neural_time_s']:>10.3f} "
            f"{r['descend_time_s']:>10.3f} "
            f"{r['backup_time_s']:>10.3f} "
            f"{r['nodes_visited']:>6}"
        )

    # Speed-up ratios relative to the first (baseline) result.
    if len(results) >= 2:
        baseline = results[0]["sims_per_sec"]
        print(f"\nSpeed-up (relative to {results[0]['label']}):")
        for r in results:
            ratio = r["sims_per_sec"] / baseline if baseline > 0 else float("inf")
            neural_pct = (r["neural_time_s"] / r["wall_time_s"] * 100) if r["wall_time_s"] > 0 else 0.0
            print(f"  {r['label']:<20} {ratio:>6.2f}x  "
                  f"(neural {neural_pct:.0f}% of wall)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS batch inference performance."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device (default: cpu). Use 'cuda' for GPU.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a model checkpoint .pt file.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="MCTS simulations per benchmark run (default: 800).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
        help="Batch sizes to benchmark (default: 1 8 16 32 64 128).",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        nargs="+",
        default=[32],
        help="Batch size(s) for warm-up run(s) (default: 32).",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint or '(untrained model)'}")
    print(f"Simulations: {args.simulations}")
    print(f"Batch sizes: {args.batch_sizes}")
    print()

    wrapper = _make_wrapper(args.device, args.checkpoint)
    board = _build_mid_game_board()
    print(f"Board position: {len(board.move_history)} moves played, "
          f"{len(board.get_legal_moves())} legal moves")
    print()

    # Warm-up: let CUDA kernels compile, discard these results.
    warmup_results: list[dict] = []
    for bs in args.warmup_batches or []:
        r = bench_batch_size(wrapper, board, bs, min(args.simulations, 200),
                             label=f"warmup_batch={bs}")
        warmup_results.append(r)

    # Actual benchmarks — run each batch size once.
    results: list[dict] = []
    for bs in args.batch_sizes:
        r = bench_batch_size(wrapper, board, bs, args.simulations,
                             label=f"batch={bs}")
        results.append(r)
        print(f"  {r['label']:<20} {r['sims_per_sec']:>9.1f} sim/s  "
              f"({r['wall_time_s']:.2f}s wall)  —  neural={r['neural_time_s']:.2f}s  "
              f"descend={r['descend_time_s']:.2f}s  backup={r['backup_time_s']:.2f}s")

    print()
    print_results(results, warmup_results=warmup_results)


if __name__ == "__main__":
    main()
