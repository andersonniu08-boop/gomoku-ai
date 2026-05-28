#!/usr/bin/env python3
"""Benchmark the MCTS hot-path with profiling instrumentation.

Usage:
    python -m selfplay.benchmark [--sims 200] [--batch 8] [--iterations 3]

Output: per-scenario timing breakdown and an aggregate report.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch

from engine.board import Board, Player
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_wrapper(device: str = "cpu") -> GomokuInferenceWrapper:
    """Create a wrapper around a freshly-initialised model (random weights)."""
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

    wrapper = GomokuInferenceWrapper(
        tmp_path, device=device,
        num_res_blocks=5, num_hidden_channels=64,
        use_se=False, use_attention=False,
    )
    wrapper._tmp_path = tmp_path
    return wrapper


def _cleanup(wrapper: GomokuInferenceWrapper) -> None:
    if hasattr(wrapper, "_tmp_path"):
        wrapper._tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Benchmark positions (verified no 5-in-a-row)
# ---------------------------------------------------------------------------


def empty_board() -> Board:
    return Board()


def opening_position() -> Board:
    """~10 stones — typical early game branching, ~25-35 legal moves."""
    board = Board()
    for r, c in [
        (7, 7), (8, 7),   # B, W
        (7, 8), (8, 8),   # B, W
        (6, 6), (9, 9),   # B, W
        (6, 8), (8, 6),   # B, W
        (7, 5), (5, 7),   # B, W
    ]:
        board.make_move(r, c)
    return board


def midgame_position() -> Board:
    """~26 stones — rich position with multiple groups, ~40-50 legal moves."""
    board = Board()
    for r, c in [
        (7, 7), (7, 6),
        (8, 7), (8, 6),
        (9, 7), (9, 6),
        (6, 6), (10, 6),
        (5, 5), (6, 5),
        (7, 8), (7, 9),
        (8, 8), (8, 9),
        (9, 8), (9, 9),
        (5, 7), (4, 7),
        (10, 7), (11, 7),
        (6, 10), (5, 10),
        (10, 4), (10, 5),
        (4, 8), (3, 9),
    ]:
        board.make_move(r, c)
    return board


def threat_open_four_position() -> Board:
    """Black has an open four — MCTS short-circuits via threat override."""
    board = Board()
    for r, c in [
        # B builds a line, W plays scattered
        (7, 3), (0, 0),
        (7, 4), (0, 2),
        (7, 5), (0, 4),
        (7, 6), (0, 6),
        # Black now has open four at (7,3)-(7,6), White has no threats
        (2, 7), (12, 12),
        (3, 8), (11, 11),
    ]:
        board.make_move(r, c)
    return board


def must_block_position() -> Board:
    """White has an open four — Black must block."""
    board = Board()
    for r, c in [
        (0, 0), (7, 3),   # B, W
        (0, 2), (7, 4),   # B, W
        (0, 4), (7, 5),   # B, W
        (0, 6), (7, 6),   # B, W — White has open four
        (12, 12), (2, 7), # B, W
        (11, 11), (3, 8), # B, W
    ]:
        board.make_move(r, c)
    return board


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _max_rss_mb() -> float:
    """Return approximate RSS in MB, or 0.0 if unavailable."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        return 0.0


def run_benchmark(
    name: str,
    board: Board,
    wrapper: GomokuInferenceWrapper,
    *,
    num_simulations: int,
    batch_size: int,
    threat_override: bool,
    iterations: int,
) -> dict:
    """Run MCTS *iterations* times and return aggregate metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"  sims={num_simulations}, batch={batch_size}, "
          f"threat_override={threat_override}, iterations={iterations}")
    print(f"{'='*60}")

    mcts = MCTS(
        wrapper,
        num_simulations=num_simulations,
        batch_size=batch_size,
        threat_override=threat_override,
        c_puct=2.5,
    )

    iter_times: list[float] = []
    iter_speeds: list[float] = []
    threat_triggered = 0

    for i in range(iterations):
        profiler = mcts.profiler
        profiler.reset()
        profiler.enable()
        wrapper.profiler = profiler

        board_copy = board.copy()
        start = time.monotonic()
        result = mcts.search(board_copy)
        elapsed = time.monotonic() - start

        sims_per_sec = num_simulations / elapsed
        iter_times.append(elapsed)
        iter_speeds.append(sims_per_sec)

        if mcts._check_forced(board) is not None:
            threat_triggered += 1

        if i == 0:
            report = profiler.report()
            key_timers = [
                "search.total", "search.batch", "search.descend_batch",
                "search.neural_eval", "search.expand_backup",
                "descend.puct_select", "descend.board_copy",
                "descend.make_move", "descend.is_terminal", "descend.rewind",
                "eval.tensor_construction", "eval.model_forward",
                "eval.postprocess", "eval.board_to_tensor",
                "eval.policy_to_move_probs",
                "threat_check", "expand.cutoff", "expand.create_nodes",
                "backup.walk",
                "dirichlet_noise", "descend.single",
            ]
            print(f"\n  Profile (iteration 0):")
            total_line_shown = False
            for line in report.split("\n"):
                timer_name = line.strip().split()[0] if line.strip() else ""
                if timer_name == "search.total" and not total_line_shown:
                    total_line_shown = True
                    print(f"    {line}")
                elif any(k in timer_name for k in key_timers):
                    print(f"    {line}")

    avg_time = sum(iter_times) / len(iter_times)
    avg_speed = sum(iter_speeds) / len(iter_speeds)

    print(f"\n  Results ({iterations} iterations):")
    print(f"    Avg time:      {avg_time*1000:.1f} ms")
    print(f"    Avg sims/sec:  {avg_speed:.0f}")
    print(f"    Threat override triggered: {threat_triggered}/{iterations}")

    return {
        "name": name,
        "avg_time_ms": avg_time * 1000,
        "avg_sims_per_sec": avg_speed,
        "iter_times": iter_times,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MCTS benchmark suite")
    parser.add_argument("--sims", type=int, default=800,
                        help="MCTS simulations per search (default: 800)")
    parser.add_argument("--batch", type=int, default=8,
                        help="MCTS batch size (default: 8)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Iterations per scenario (default: 3)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--full", action="store_true",
                        help="Run all scenarios (including no-threat-override)")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    rss_before = _max_rss_mb()

    wrapper = _make_wrapper(device=device)
    try:
        # Warmup: run one search on the opening position to warm caches/JIT.
        print("Warming up...")
        warmup_board = opening_position()
        warmup_mcts = MCTS(wrapper, num_simulations=50, batch_size=8, threat_override=True)
        warmup_mcts.search(warmup_board)
        print("  done.\n")

        scenarios = [
            ("empty_board", empty_board()),
            ("opening", opening_position()),
            ("midgame", midgame_position()),
            ("threat_open_four", threat_open_four_position()),
            ("must_block", must_block_position()),
        ]

        results: list[dict] = []
        for name, board in scenarios:
            result = run_benchmark(
                name, board, wrapper,
                num_simulations=args.sims,
                batch_size=args.batch,
                threat_override=True,
                iterations=args.iterations,
            )
            results.append(result)

        # Without threat override for comparison (midgame).
        if args.full:
            result_no_to = run_benchmark(
                "midgame_no_threat_override",
                midgame_position(), wrapper,
                num_simulations=args.sims,
                batch_size=args.batch,
                threat_override=False,
                iterations=min(args.iterations, 2),
            )
            results.append(result_no_to)

        # Summary table
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Scenario':<30s}  {'Avg (ms)':>10s}  {'Sims/sec':>10s}")
        print("-" * 54)
        for r in results:
            print(f"{r['name']:<30s}  {r['avg_time_ms']:>10.1f}  "
                  f"{r['avg_sims_per_sec']:>10.0f}")

        rss_after = _max_rss_mb()
        if rss_before > 0:
            print(f"\nRSS delta: {rss_after - rss_before:.1f} MB")

    finally:
        _cleanup(wrapper)


if __name__ == "__main__":
    main()
