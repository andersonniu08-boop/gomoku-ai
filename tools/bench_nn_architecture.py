#!/usr/bin/env python3
"""Neural network architecture benchmark — inference speed, memory, throughput.

Measures the upgraded architecture (10-block residual CNN, multi-head attention,
dilated convs, deeper policy head, CBAM spatial attention) across key metrics.

Usage::

    python tools/bench_nn_architecture.py
    python tools/bench_nn_architecture.py --checkpoint checkpoints/best.pt --device cuda
    python tools/bench_nn_architecture.py --batch-sizes 1,4,16,64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from engine.board import Board, Player
from engine.encoding import board_to_tensor
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS
from selfplay.selfplay import SelfPlayGame


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def param_count(model: torch.nn.Module) -> dict:
    """Return detailed parameter breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    by_component: dict[str, int] = Counter()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Group by top-level component
        top = name.split(".")[0]
        by_component[top] += p.numel()

    return {
        "total": total,
        "trainable": trainable,
        "by_component": {k: v for k, v in sorted(by_component.items())},
    }


# ---------------------------------------------------------------------------
# Inference speed
# ---------------------------------------------------------------------------


def benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int = 10,
    repeats: int = 100,
) -> list[dict]:
    """Measure raw forward-pass latency across batch sizes."""
    results = []
    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, 15, 15, device=device)

        # Warmup
        for _ in range(warmup):
            _ = model(dummy)

        # Timed
        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        mean_s = sum(times) / len(times)
        results.append({
            "batch_size": bs,
            "mean_ms": round(mean_s * 1000, 3),
            "per_sample_ms": round(mean_s * 1000 / bs, 3),
            "samples_per_sec": round(bs / mean_s, 1),
        })

    return results


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------


def benchmark_memory(model: torch.nn.Module, device: torch.device) -> dict:
    """Estimate model memory footprint."""
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )

    result = {
        "parameters_mb": round(param_bytes / (1024 * 1024), 2),
        "buffers_mb": round(buffer_bytes / (1024 * 1024), 2),
        "total_mb": round((param_bytes + buffer_bytes) / (1024 * 1024), 2),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        dummy = torch.randn(16, 3, 15, 15, device=device)
        _ = model(dummy)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        result["peak_allocated_bs16_mb"] = round(peak_mb, 2)

    return result


# ---------------------------------------------------------------------------
# Receptive field analysis
# ---------------------------------------------------------------------------


def compute_receptive_field(model: GomokuNet) -> dict:
    """Estimate the effective receptive field of each block.

    For a 3×3 conv with dilation d, the local receptive field grows by
    2*d pixels per layer (one on each side).  With k=3 and padding=dilation,
    each conv pass contributes d cells on each edge.
    """
    rf_start = 3  # initial conv: 3×3 → RF=3
    rf_by_block: list[int] = []
    last_effective = 1
    cumulative = rf_start

    for i, block in enumerate(model.res_blocks):
        if hasattr(block, "conv1"):
            d = block.conv1.dilation[0]
        else:
            d = 1
        # Two convs per block, each with 3×3 kernel and dilation d
        rf_gain = 2 * d  # per conv (k=3): (k-1)*d = 2*d
        cumulative += 2 * rf_gain
        rf_by_block.append(cumulative)
        if cumulative >= 15 and last_effective < 15:
            last_effective = i

    return {
        "final_rf": cumulative,
        "span_15x15": cumulative >= 15,
        "rf_by_block": rf_by_block,
        "first_full_span_block": last_effective,
        "dilations": [
            block.conv1.dilation[0] if hasattr(block, "conv1") else 1
            for block in model.res_blocks
        ],
    }


# ---------------------------------------------------------------------------
# Tactical benchmark
# ---------------------------------------------------------------------------


TACTICAL_SCENARIOS: list[tuple[str, list[tuple[int, int]], set[tuple[int, int]]]] = [
    (
        "diagonal open-four",
        [(3, 3), (0, 0), (4, 4), (0, 1), (5, 5), (0, 2), (6, 6), (0, 3)],
        {(2, 2), (7, 7)},
    ),
    (
        "split closed-four",
        [(7, 2), (0, 0), (7, 3), (0, 1), (7, 5), (0, 2), (7, 6), (0, 3)],
        {(7, 4)},
    ),
    (
        "block opponent open-four",
        [(2, 2), (7, 2), (4, 4), (7, 3), (6, 6), (7, 4), (8, 8), (7, 5)],
        {(7, 1), (7, 6)},
    ),
    (
        "win priority over block",
        [(7, 2), (10, 0), (7, 3), (10, 1), (7, 4), (10, 2), (7, 5), (10, 3)],
        {(7, 1), (7, 6)},
    ),
    (
        "long-range diagonal threat",
        [(1, 1), (13, 2), (2, 2), (12, 3), (3, 3), (11, 4), (4, 4), (10, 5)],
        {(5, 5), (0, 0)},
    ),
]


def benchmark_tactical(
    wrapper: GomokuInferenceWrapper,
    simulations: int = 200,
) -> dict:
    """Measure tactical correctness on known test positions."""
    passed = 0
    failed = 0
    details = []

    for name, setup, expected in TACTICAL_SCENARIOS:
        board = Board()
        for r, c in setup:
            board.make_move(r, c)

        mcts = MCTS(wrapper, num_simulations=simulations, threat_override=True)
        probs = mcts.search(board)
        actual = set(probs.keys())

        ok = actual == expected
        if ok:
            passed += 1
        else:
            failed += 1
        details.append({
            "scenario": name,
            "passed": ok,
            "expected": sorted(expected),
            "actual": sorted(actual) if not ok else None,
        })

    return {
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / max(passed + failed, 1),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Gameplay quality
# ---------------------------------------------------------------------------


def benchmark_gameplay(
    wrapper: GomokuInferenceWrapper,
    num_games: int = 3,
    simulations: int = 200,
) -> dict:
    """Generate self-play games and measure quality metrics."""
    game_lengths = []
    total_time = 0.0
    opening_moves: Counter = Counter()

    for _ in range(num_games):
        game = SelfPlayGame(
            wrapper,
            num_simulations=simulations,
            temperature=1.0,
            temperature_threshold=15,
            threat_override=True,
            augment=False,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
        )
        t0 = time.monotonic()
        examples = game.play()
        elapsed = time.monotonic() - t0

        total_time += elapsed
        game_lengths.append(len(examples))

        # Track opening diversity (first 2 moves)
        if len(examples) >= 2:
            opening_hash = f"{len(examples)}_{game_lengths[-1]}"
            opening_moves[opening_hash] += 1

    mean_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    mean_time = total_time / num_games if num_games else 0

    return {
        "games": num_games,
        "mean_game_length": round(mean_len, 1),
        "mean_game_time_s": round(mean_time, 1),
        "games_per_hour": round(3600 / mean_time, 1) if mean_time > 0 else float("inf"),
        "game_lengths": game_lengths,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Neural network architecture benchmark"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to trained checkpoint (uses untrained model if not set)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--batch-sizes", default="1,4,16,64",
        help="Comma-separated batch sizes for inference benchmark",
    )
    parser.add_argument(
        "--repeats", type=int, default=100,
        help="Inference timing repeats per batch size",
    )
    parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS simulations for tactical/gameplay benchmarks",
    )
    parser.add_argument(
        "--games", type=int, default=3,
        help="Self-play games for gameplay benchmark",
    )
    parser.add_argument(
        "--json", default="data/bench_nn_results.json",
        help="Write JSON results to this path",
    )
    parser.add_argument(
        "--skip-gameplay", action="store_true",
        help="Skip self-play gameplay benchmark (slow)",
    )

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # --- Model setup ---
    model = GomokuNet().to(device)
    model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

    wrapper = GomokuInferenceWrapper(
        args.checkpoint or _make_temp_checkpoint(model, device),
        device=str(device),
    ) if args.checkpoint else _make_wrapper_from_model(model, device)

    print(f"Device: {device}")
    print(f"Batch sizes: {batch_sizes}")

    all_results = {}

    # --- Parameter count ---
    print("\n" + "=" * 60)
    print("PARAMETER COUNT")
    print("=" * 60)
    params = param_count(model)
    all_results["parameters"] = params
    print(f"  Total:     {params['total']:>10,}")
    print(f"  Trainable: {params['trainable']:>10,}")
    for comp, count in params["by_component"].items():
        print(f"  {comp:<20s} {count:>10,}")

    # --- Memory footprint ---
    print("\n" + "=" * 60)
    print("MEMORY FOOTPRINT")
    print("=" * 60)
    mem = benchmark_memory(model, device)
    all_results["memory"] = mem
    for k, v in mem.items():
        print(f"  {k}: {v}")

    # --- Inference speed ---
    print("\n" + "=" * 60)
    print("INFERENCE SPEED")
    print("=" * 60)
    inf_results = benchmark_inference(model, device, batch_sizes, repeats=args.repeats)
    all_results["inference"] = inf_results
    print(f"  {'Batch':>6s}  {'Mean ms':>10s}  {'Per-sample':>10s}  {'Samples/s':>10s}")
    print("-" * 48)
    for r in inf_results:
        print(f"  {r['batch_size']:>6d}  {r['mean_ms']:>10.3f}  "
              f"{r['per_sample_ms']:>10.3f}  {r['samples_per_sec']:>10.1f}")

    # --- Receptive field ---
    print("\n" + "=" * 60)
    print("RECEPTIVE FIELD")
    print("=" * 60)
    rf = compute_receptive_field(model)
    all_results["receptive_field"] = rf
    print(f"  Final RF:     {rf['final_rf']} cells")
    print(f"  Spans 15×15:  {rf['span_15x15']}")
    print(f"  Full span at: block {rf['first_full_span_block']}")
    print(f"  Dilations:    {rf['dilations']}")

    # --- Tactical ---
    print("\n" + "=" * 60)
    print("TACTICAL CORRECTNESS")
    print("=" * 60)
    tac = benchmark_tactical(wrapper, simulations=args.simulations)
    all_results["tactical"] = tac
    print(f"  Passed: {tac['passed']}/{tac['total']} ({tac['pass_rate']:.1%})")
    for d in tac["details"]:
        sym = "✓" if d["passed"] else "✗"
        print(f"    {sym} {d['scenario']}")

    # --- Gameplay (optional) ---
    if not args.skip_gameplay:
        print("\n" + "=" * 60)
        print("SELF-PLAY GAMEPLAY")
        print("=" * 60)
        gp = benchmark_gameplay(wrapper, num_games=args.games, simulations=args.simulations)
        all_results["gameplay"] = gp
        print(f"  Games:             {gp['games']}")
        print(f"  Mean game length:  {gp['mean_game_length']} moves")
        print(f"  Mean game time:    {gp['mean_game_time_s']}s")
        print(f"  Games per hour:    {gp['games_per_hour']}")
        print(f"  Game lengths:      {gp['game_lengths']}")

    # --- Write JSON ---
    json_path = Path(args.json)
    json_path.parent.mkdir(exist_ok=True)
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results: {json_path}")


def _make_temp_checkpoint(model: torch.nn.Module, device: torch.device) -> str:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(model.state_dict(), tmp.name)
    return tmp.name


def _make_wrapper_from_model(model: torch.nn.Module, device: torch.device) -> GomokuInferenceWrapper:
    tmp = _make_temp_checkpoint(model, device)
    wrapper = GomokuInferenceWrapper(tmp, device=str(device))
    Path(tmp).unlink(missing_ok=True)
    return wrapper


if __name__ == "__main__":
    main()
