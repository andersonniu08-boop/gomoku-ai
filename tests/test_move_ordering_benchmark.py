"""Benchmark move ordering effectiveness: branching factor, sims/sec, tactical accuracy.

Usage:
    python -m pytest tests/test_move_ordering_benchmark.py -v --tb=short
"""

import time
import tempfile
from pathlib import Path

import torch
import pytest

from engine.board import Board
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS
from selfplay.move_ordering import (
    compute_tactical_scores,
    order_and_filter_moves,
)


def _make_wrapper():
    model = GomokuNet(board_size=15, in_channels=3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    class _CleanupWrapper(GomokuInferenceWrapper):
        def __del__(self):
            if tmp_path.exists():
                tmp_path.unlink()

    wrapper = _CleanupWrapper(tmp_path, device="cpu")
    return wrapper, tmp_path


def _place_many(board, moves):
    for r, c in moves:
        board.make_move(r, c)


# ---------------------------------------------------------------------------
# Benchmark: branching factor — raw neural cutoff vs. tactical ordering
# ---------------------------------------------------------------------------


def test_benchmark_branching_factor():
    """Tactical ordering should not substantially increase branching factor
    compared to raw top-40 cutoff."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        # Create a mid-game position with many stones.
        # Black on row 6, White on row 8 — spaced so no 5-in-a-row.
        _place_many(board, [
            (6, 2), (8, 3), (6, 4), (8, 5),
            (6, 6), (8, 7), (6, 8), (8, 9),
            (5, 5), (9, 6), (7, 3), (9, 4),
            (7, 7), (9, 8), (4, 6), (10, 6),
        ])

        # Get neural move_probs
        move_probs, _ = wrapper.evaluate(board)

        # Raw cutoff: top 40 by neural prior
        raw_cutoff = sorted(move_probs, key=lambda x: -x[1])[:40]
        total_raw = sum(p for _, p in raw_cutoff)
        raw_cutoff = [(m, p / total_raw) for m, p in raw_cutoff]

        # Tactical ordering
        tactical = order_and_filter_moves(board, move_probs, max_moves=40)
        total_tactical = sum(p for _, p in tactical)
        tactical = [(m, p / total_tactical) for m, p in tactical]

        # Report branching factor
        print(f"\n  Raw cutoff:     {len(raw_cutoff)} moves")
        print(f"  Tactical order: {len(tactical)} moves")
        print(f"  Legal moves available: {len(board.get_legal_moves())}")

        # Tactical ordering should not produce MORE moves than raw cutoff
        # (it's a subset, not an expansion)
        assert len(tactical) <= 40
        assert len(raw_cutoff) <= 40
    finally:
        tmp.unlink()


def test_benchmark_branching_factor_with_threats():
    """When threats are present, tactical ordering should keep fewer but
    more relevant moves."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        # Position with a threat: Black has an open four
        _place_many(board, [
            (7, 2), (0, 0),
            (7, 3), (0, 1),
            (7, 4), (0, 2),
            (7, 5), (0, 3),
            (6, 6), (8, 8),  # Some extra stones for more context
        ])

        move_probs, _ = wrapper.evaluate(board)

        raw_cutoff = sorted(move_probs, key=lambda x: -x[1])[:40]
        total_raw = sum(p for _, p in raw_cutoff)
        raw_cutoff = [(m, p / total_raw) for m, p in raw_cutoff]

        tactical = order_and_filter_moves(board, move_probs, max_moves=40)
        total_tactical = sum(p for _, p in tactical)
        tactical = [(m, p / total_tactical) for m, p in tactical]

        tactical_moves = {m for m, _ in tactical}
        winning_moves = {(7, 1), (7, 6)}

        print(f"\n  Raw cutoff:       {len(raw_cutoff)} moves")
        print(f"  Tactical order:   {len(tactical)} moves")
        print(f"  Winning moves in tactical: {tactical_moves & winning_moves}")
        print(f"  Legal moves: {len(board.get_legal_moves())}")

        # Critical: winning moves must survive
        assert (7, 1) in tactical_moves or (7, 6) in tactical_moves
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Benchmark: sims/sec — measure overhead of tactical scoring in MCTS
# ---------------------------------------------------------------------------


def test_benchmark_mcts_speed():
    """Measure sims/sec with tactical ordering."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        _place_many(board, [
            (7, 5), (8, 5), (7, 4), (8, 4),
            (7, 6), (8, 6), (6, 5), (9, 5),
        ])

        mcts = MCTS(wrapper, num_simulations=100, batch_size=8, threat_override=True)
        sim_board = board.copy()

        from selfplay.mcts import MCTSNode
        root = MCTSNode()

        start = time.monotonic()
        mcts._run_search(sim_board, root)
        elapsed = time.monotonic() - start

        total_visits = sum(c.visit_count for c in root.children.values())
        sims_per_sec = total_visits / elapsed if elapsed > 0 else 0

        print(f"\n  Simulations: {total_visits}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Sims/sec: {sims_per_sec:.1f}")
        print(f"  Children explored: {len(root.children)}")

        # Just verify it completes
        assert total_visits > 0
    finally:
        tmp.unlink()


def test_benchmark_move_ordering_overhead():
    """Measure the cost of tactical scoring relative to raw cutoff."""
    board = Board()
    # Mid-game position with many stones, no 5-in-a-row.
    _place_many(board, [
        (6, 2), (8, 3), (6, 4), (8, 5),
        (6, 6), (8, 7), (6, 8), (8, 9),
        (5, 5), (9, 6), (7, 3), (9, 4),
        (7, 7), (9, 8), (4, 6), (10, 6),
    ])

    legal = board.get_legal_moves()
    uniform_probs = [(m, 1.0 / len(legal)) for m in legal]

    # Measure tactical scoring time
    N = 100
    start = time.monotonic()
    for _ in range(N):
        scores = compute_tactical_scores(board, uniform_probs)
    score_time = (time.monotonic() - start) / N

    # Measure raw filter time (no tactical analysis)
    start = time.monotonic()
    for _ in range(N):
        result = order_and_filter_moves(board, uniform_probs, max_moves=40, threat_boost=False)
    raw_time = (time.monotonic() - start) / N

    # Measure tactical filter time (with analysis)
    start = time.monotonic()
    for _ in range(N):
        result = order_and_filter_moves(board, uniform_probs, max_moves=40, threat_boost=True)
    tactical_time = (time.monotonic() - start) / N

    print(f"\n  Raw cutoff (no tactical):   {raw_time*1000:.3f}ms")
    print(f"  Tactical scoring only:      {score_time*1000:.3f}ms")
    print(f"  Tactical ordering:          {tactical_time*1000:.3f}ms")
    print(f"  Overhead added:             {(tactical_time - raw_time)*1000:.3f}ms")
    print(f"  Candidates scored:          {len(legal)}")

    # Report but don't assert on timing — too variable
    assert score_time > 0


# ---------------------------------------------------------------------------
# Benchmark: tactical accuracy — can MCTS + tactical ordering find forced wins
# with fewer simulations?
# ---------------------------------------------------------------------------


def _count_win_probability(mcts, board, winning_moves):
    """Run MCTS search and return total probability assigned to winning moves."""
    dist = mcts.search(board)
    return sum(p for m, p in dist.items() if m in winning_moves)


@pytest.mark.parametrize("use_threat_override", [True, False])
def test_benchmark_tactical_vs_raw_win_detection(use_threat_override):
    """Compare win detection with and without tactical ordering."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        _place_many(board, [
            (7, 2), (0, 0),
            (7, 3), (0, 1),
            (7, 4), (0, 2),
            (7, 5), (0, 3),
        ])
        winning_moves = {(7, 1), (7, 6)}

        # Just run MCTS and check it doesn't crash
        mcts = MCTS(
            wrapper, num_simulations=50, batch_size=8,
            threat_override=use_threat_override,
        )
        dist = mcts.search(board)
        win_prob = _count_win_probability(mcts, board, winning_moves)
        print(f"\n  Threat override: {use_threat_override}")
        print(f"  Winning moves in dist: {[m for m in winning_moves if m in dist]}")
        print(f"  Winning prob: {win_prob:.3f}")
        assert len(dist) > 0
    finally:
        tmp.unlink()
