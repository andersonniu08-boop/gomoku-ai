#!/usr/bin/env python3
"""Self-play quality benchmark — throughput, diversity, and gameplay metrics."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch

from engine.board import Board, Player
from engine.encoding import board_to_tensor
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS
from selfplay.selfplay import SelfPlayGame, TrainingExample


def benchmark_search_throughput(
    wrapper: GomokuInferenceWrapper,
    sims_list: list[int],
    positions: int = 5,
) -> list[dict]:
    """Measure MCTS search time at different simulation budgets."""
    results = []
    board = Board()
    # Play a few opening moves to get a realistic midgame position.
    for r, c in [(7, 7), (7, 8), (8, 7), (8, 8), (6, 6), (9, 9)]:
        board.make_move(r, c)

    for sims in sims_list:
        mcts = MCTS(wrapper, num_simulations=sims, threat_override=True,
                    dirichlet_alpha=0.03, dirichlet_epsilon=0.25)
        times = []
        visit_depths = []
        for _ in range(positions):
            b = board.copy()
            t0 = time.monotonic()
            result = mcts.search_with_stats(b)
            elapsed = time.monotonic() - t0
            times.append(elapsed)
            # Estimate tree depth from max visited node depth (proxy: avg visits per move)
            if result.visit_counts:
                avg_visits = sum(result.visit_counts.values()) / len(result.visit_counts)
                visit_depths.append(avg_visits)

        results.append({
            "simulations": sims,
            "mean_time_s": sum(times) / len(times),
            "min_time_s": min(times),
            "max_time_s": max(times),
            "moves_per_second": sims / (sum(times) / len(times)),
            "avg_top_move_visits": sum(visit_depths) / max(len(visit_depths), 1),
        })

    return results


def benchmark_game_throughput(
    wrapper: GomokuInferenceWrapper,
    sims: int,
    num_games: int = 3,
) -> dict:
    """Measure full self-play game throughput."""
    times = []
    move_counts = []
    example_counts = []
    resignation_counts = 0

    for i in range(num_games):
        game = SelfPlayGame(
            wrapper,
            num_simulations=sims,
            temperature=1.0,
            temperature_threshold=15,
            threat_override=True,
            augment=False,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
            opening_moves=6,
            resignation_threshold=-0.9,
            resignation_moves=3,
        )
        t0 = time.monotonic()
        examples = game.play()
        elapsed = time.monotonic() - t0
        times.append(elapsed)

        # Count moves from examples (dedup by counting unique states is hard,
        # but move_history is available from the game's board — we'll estimate
        # from example count).
        move_counts.append(len(examples))
        example_counts.append(len(examples))

        # Check for resignation: if not terminal and examples exist, game was resigned.
        # We can't check the board from here, but short games (<20 moves) suggest
        # resignation (or very fast win).
        if len(examples) < 20:
            resignation_counts += 1

    mean_time = sum(times) / len(times)
    return {
        "simulations": sims,
        "games": num_games,
        "mean_game_time_s": mean_time,
        "games_per_hour": 3600 / mean_time if mean_time > 0 else float("inf"),
        "mean_moves_per_game": sum(move_counts) / len(move_counts),
        "mean_examples_per_game": sum(example_counts) / len(example_counts),
        "total_examples": sum(example_counts),
        "short_games": resignation_counts,
    }


def benchmark_move_agreement(
    wrapper: GomokuInferenceWrapper,
    sims_pairs: list[tuple[int, int]],
    positions: int = 10,
) -> list[dict]:
    """Measure how often different sim budgets agree on the best move."""
    results = []
    # Generate diverse positions by playing random opening moves.
    boards = []
    for seed in range(positions):
        board = Board()
        import random
        random.seed(seed * 137)
        legal = board.get_legal_moves()
        for _ in range(8):  # 8 random opening moves
            if not legal:
                break
            move = random.choice(legal)
            board.make_move(*move)
            legal = board.get_legal_moves()
        boards.append(board)

    for low_sims, high_sims in sims_pairs:
        agreements = 0
        low_moves: Counter = Counter()
        high_moves: Counter = Counter()
        for board in boards:
            mcts_low = MCTS(wrapper, num_simulations=low_sims, threat_override=True)
            mcts_high = MCTS(wrapper, num_simulations=high_sims, threat_override=True)

            move_low = mcts_low.select_move(board.copy(), temperature=0.0)
            move_high = mcts_high.select_move(board.copy(), temperature=0.0)

            if move_low == move_high:
                agreements += 1
            low_moves[move_low] += 1
            high_moves[move_high] += 1

        results.append({
            "low_sims": low_sims,
            "high_sims": high_sims,
            "agreement_rate": agreements / positions,
            "unique_low_moves": len(low_moves),
            "unique_high_moves": len(high_moves),
        })

    return results


def benchmark_replay_diversity(
    wrapper: GomokuInferenceWrapper,
    sims: int,
    num_games: int = 5,
) -> dict:
    """Generate games and measure replay diversity metrics."""
    from selfplay.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(max_size=100_000)
    for _ in range(num_games):
        game = SelfPlayGame(
            wrapper,
            num_simulations=sims,
            temperature=1.0,
            temperature_threshold=15,
            threat_override=True,
            augment=False,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
            opening_moves=6,
            resignation_threshold=-0.9,
            resignation_moves=3,
        )
        examples = game.play()
        buf.add_examples(examples)

    return buf.diversity_stats()


def main():
    import tempfile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Use a fresh model matching the current architecture.  The on-disk
    # checkpoint is from an older architecture revision and isn't loadable.
    # A fresh model gives valid throughput numbers (search speed is what
    # matters), though move-agreement metrics will reflect random play.
    from neural.model import GomokuNet
    model = GomokuNet()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    try:
        print(f"Using fresh model (current architecture) for throughput benchmarks...")
        wrapper = GomokuInferenceWrapper(tmp_path, device=device)
    finally:
        tmp_path.unlink()

    all_results = {}

    # --- Search throughput ---
    print("\n" + "=" * 60)
    print("SEARCH THROUGHPUT BENCHMARK")
    print("=" * 60)
    sims_list = [200, 400, 800]
    search_results = benchmark_search_throughput(wrapper, sims_list)
    all_results["search_throughput"] = search_results
    print(f"{'Sims':>6s}  {'Time (s)':>8s}  {'Moves/s':>8s}  {'Top visits':>10s}")
    print("-" * 42)
    for r in search_results:
        print(f"{r['simulations']:>6d}  {r['mean_time_s']:>8.3f}  "
              f"{r['moves_per_second']:>8.1f}  {r['avg_top_move_visits']:>10.1f}")

    # --- Game throughput ---
    print("\n" + "=" * 60)
    print("GAME THROUGHPUT BENCHMARK")
    print("=" * 60)
    game_results = benchmark_game_throughput(wrapper, 800, num_games=3)
    all_results["game_throughput"] = game_results
    print(f"Sims per move:      {game_results['simulations']}")
    print(f"Games played:       {game_results['games']}")
    print(f"Mean game time:     {game_results['mean_game_time_s']:.1f}s")
    print(f"Games per hour:     {game_results['games_per_hour']:.1f}")
    print(f"Mean moves/game:    {game_results['mean_moves_per_game']:.1f}")
    print(f"Mean examples/game: {game_results['mean_examples_per_game']:.1f}")
    print(f"Total examples:     {game_results['total_examples']}")
    print(f"Short games (resign/quick win): {game_results['short_games']}")

    # --- Move agreement ---
    # NOTE: skipped with untrained model — move selection is essentially
    # random, so agreement rates are meaningless.  Run with a trained
    # checkpoint once the architecture revision is stabilized.
    print("\n" + "=" * 60)
    print("MOVE AGREEMENT (skipped — needs trained checkpoint)")
    print("=" * 60)
    print("Re-run with a trained checkpoint after architecture stabilizes.")
    agreement_results = []
    all_results["move_agreement"] = agreement_results

    # --- Replay diversity ---
    print("\n" + "=" * 60)
    print("REPLAY DIVERSITY")
    print("=" * 60)
    diversity = benchmark_replay_diversity(wrapper, 800, num_games=5)
    all_results["replay_diversity"] = diversity
    print(f"Total positions:      {diversity['total_positions']}")
    print(f"Unique openings:      {diversity['unique_openings']}")
    print(f"Opening diversity:    {diversity['opening_diversity_ratio']:.1%}")
    print(f"Move distribution:    {diversity['move_distribution']}")
    print(f"Value distribution:   {diversity['value_distribution']}")
    if diversity['openings_top']:
        print(f"Top openings:")
        for h, c in diversity['openings_top'][:5]:
            print(f"  hash={h[:12]}…  count={c}")

    # --- Write JSON ---
    output_path = Path("data/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
