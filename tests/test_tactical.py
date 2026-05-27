"""Tests for engine.tactical — TacticalSolver and TacticalAnalysis.

Verifies:
- Immediate win detection (open four, closed four contiguous/split)
- Must-block detection (including opponent CLOSED_FOUR — the key bug fix)
- Double-threat move detection
- Forced-sequence search
- Tactical prior boosting
- Integration with MCTS
"""

from __future__ import annotations

import pytest

from engine.board import Board, Player
from engine.tactical import TacticalAnalysis, TacticalSolver
from engine.threats import ThreatDetector, ThreatType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place_many(board: Board, moves: list[tuple[int, int]]) -> None:
    """Play a sequence of alternating moves."""
    for r, c in moves:
        board.make_move(r, c)


# ===========================================================================
# TacticalSolver.analyze — immediate wins
# ===========================================================================


def test_detects_open_four_win():
    """Black has open four — both open ends should be winning moves."""
    board = Board()
    _place_many(board, [
        (7, 2), (0, 0),
        (7, 3), (0, 1),
        (7, 4), (0, 2),
        (7, 5), (0, 3),
    ])
    analysis = TacticalSolver.analyze(board)
    assert analysis.winning_moves == {(7, 1), (7, 6)}
    assert analysis.has_forced_win
    dist = analysis.get_forced_distribution()
    assert dist is not None
    assert set(dist.keys()) == {(7, 1), (7, 6)}


def test_detects_contiguous_closed_four_win():
    """Contiguous closed four (XXXX_) — the one open end wins."""
    board = Board()
    _place_many(board, [
        (7, 1), (7, 0),  # X O — O blocks left of the four
        (7, 2), (8, 0),  # X O
        (7, 3), (8, 1),  # X O
        (7, 4), (8, 2),  # X O
    ])
    # Black has X at (7,1)(7,2)(7,3)(7,4). Left blocked by O at (7,0).
    # Right end (7,5) is open → closed four, winning move.
    analysis = TacticalSolver.analyze(board)
    assert (7, 5) in analysis.winning_moves
    assert analysis.has_forced_win


def test_detects_split_closed_four_win():
    """Split closed four (XX_XX) — only the gap wins."""
    board = Board()
    _place_many(board, [
        (7, 2), (0, 0),
        (7, 3), (0, 1),
        (7, 5), (0, 2),
        (7, 6), (0, 3),
    ])
    # Black has XX_XX at cols 2,3,5,6, gap at 4.
    analysis = TacticalSolver.analyze(board)
    assert (7, 4) in analysis.winning_moves
    # External ends (7,1) and (7,7) should NOT be winning moves.
    assert (7, 1) not in analysis.winning_moves
    assert (7, 7) not in analysis.winning_moves
    assert analysis.has_forced_win
    dist = analysis.get_forced_distribution()
    assert dist is not None
    assert set(dist.keys()) == {(7, 4)}


def test_no_false_win_on_empty_board():
    """Empty board should have no forced win."""
    board = Board()
    analysis = TacticalSolver.analyze(board)
    assert not analysis.has_forced_win
    assert len(analysis.winning_moves) == 0


def test_no_false_win_on_scattered_board():
    """Scattered stones with no threats should have no forced win."""
    board = Board()
    board.make_move(3, 3)
    board.make_move(10, 10)
    board.make_move(7, 7)
    analysis = TacticalSolver.analyze(board)
    assert not analysis.has_forced_win
    assert len(analysis.winning_moves) == 0


# ===========================================================================
# TacticalSolver.analyze — must-block (the critical CLOSED_FOUR fix)
# ===========================================================================


def test_detects_must_block_opponent_open_four():
    """Opponent has open four — must block at either open end."""
    board = Board()
    _place_many(board, [
        (0, 0), (7, 2),  # Black, White starts open four
        (0, 1), (7, 3),
        (0, 2), (7, 4),
        (0, 3), (7, 5),
    ])
    # White has open four at (7,2)-(7,5). Black must block.
    analysis = TacticalSolver.analyze(board)
    assert analysis.has_forced_defense
    assert analysis.must_block == {(7, 1), (7, 6)}


def test_detects_must_block_opponent_closed_four():
    """Opponent has CLOSED_FOUR — must block their one open end.

    This is the key bug fix: previously, opponent CLOSED_FOUR was not
    treated as a must-block threat.
    """
    board = Board()
    _place_many(board, [
        (0, 0), (7, 2),  # Black, White starts building
        (0, 1), (7, 3),
        (0, 2), (7, 4),
        (0, 3), (7, 5),
        (7, 1), (8, 0),  # Black blocks left end, White dummy
    ])
    # White has closed four at (7,2)-(7,5), left blocked by B at (7,1).
    # Right end (7,6) is open.  Black MUST block at (7,6).
    analysis = TacticalSolver.analyze(board)
    assert (7, 6) in analysis.must_block, (
        "CRITICAL BUG: opponent CLOSED_FOUR is not being detected as must-block!"
    )
    assert analysis.has_forced_defense


def test_detects_must_block_opponent_split_closed_four():
    """Opponent has split CLOSED_FOUR — must block the gap."""
    board = Board()
    _place_many(board, [
        (0, 0), (7, 2),  # Black, White starts
        (0, 1), (7, 3),
        (0, 2), (7, 5),
        (0, 3), (7, 6),
    ])
    # White has XX_XX at cols 2,3,5,6, gap at 4. Black must block gap.
    analysis = TacticalSolver.analyze(board)
    assert (7, 4) in analysis.must_block, (
        "CRITICAL BUG: opponent split CLOSED_FOUR gap must be blocked!"
    )
    assert analysis.has_forced_defense


def test_must_block_takes_priority_over_other_moves():
    """When must-block exists, forced distribution returns only blocking moves."""
    board = Board()
    # White has open four at (7,2)-(7,5).  Black stones spread out
    # (not forming any threats) so Black has no winning move of its own.
    _place_many(board, [
        (10, 0), (7, 2),  # Black, White starts open four
        (12, 3), (7, 3),
        (10, 6), (7, 4),
        (12, 9), (7, 5),
    ])
    analysis = TacticalSolver.analyze(board)
    dist = analysis.get_forced_distribution()
    assert dist is not None
    # Should only include blocking moves, not all legal moves.
    assert set(dist.keys()) == {(7, 1), (7, 6)}


def test_no_must_block_when_opponent_has_no_threats():
    """When opponent has no forcing threats, must_block should be empty."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(8, 8)
    analysis = TacticalSolver.analyze(board)
    assert not analysis.has_forced_defense
    assert len(analysis.must_block) == 0


# ===========================================================================
# TacticalSolver.analyze — urgent blocks (opponent OPEN_THREE)
# ===========================================================================


def test_detects_urgent_block_opponent_open_three():
    """Opponent has open three — blocking ends should be urgent."""
    board = Board()
    _place_many(board, [
        (0, 0), (7, 3),  # Black, White starts open three
        (0, 1), (7, 4),
        (0, 2), (7, 5),
    ])
    analysis = TacticalSolver.analyze(board)
    # Open three at cols 3,4,5 has open ends (7,2) and (7,6).
    assert (7, 2) in analysis.urgent_blocks
    assert (7, 6) in analysis.urgent_blocks


# ===========================================================================
# TacticalSolver.analyze — double-threat detection
# ===========================================================================


def test_detects_double_threat_move():
    """A move that creates a double threat should be detected."""
    board = Board()
    # Set up a position where playing (7,5) creates two open threes
    # simultaneously: horizontal at (7,3)-(7,5) and vertical at (6,4)-(8,4).
    # Black stones: (7,3),(7,4),(6,4),(8,4). Playing at (7,5) joins the
    # horizontal pair (7,3)(7,4) into a three AND the vertical already
    # forms a three at (6,4)(7,4)(8,4) — together they are two open threes
    # that cannot both be blocked.
    #
    # White stones scattered on row 10 (no threats).
    _place_many(board, [
        (7, 3), (10, 0),
        (7, 4), (10, 1),
        (6, 4), (10, 2),
        (8, 4), (10, 3),
    ])
    analysis = TacticalSolver.analyze(board)
    # (7,5) creates horizontal three AND shares (7,4) with the existing
    # vertical three → two open threes = double threat.
    assert (7, 5) in analysis.double_threat_moves, (
        f"Expected (7,5) as double-threat move, got: "
        f"{analysis.double_threat_moves}"
    )


def test_no_double_threat_when_no_threats():
    """Scattered board should have no double-threat moves."""
    board = Board()
    board.make_move(3, 3)
    board.make_move(10, 10)
    analysis = TacticalSolver.analyze(board)
    assert len(analysis.double_threat_moves) == 0


# ===========================================================================
# TacticalSolver.analyze — forced-sequence search
# ===========================================================================


def test_forced_sequence_open_four_chain():
    """Create open four → opponent blocks → we win at other end.

    Pattern: we have XXX at (7,3)(7,4)(7,5). Playing (7,6) creates open
    four with two open ends.  Opponent can only block one — we play the
    other and win.  This is a forced win sequence, not an immediate win
    (the open four doesn't exist yet on the board).
    """
    board = Board()
    # White stones scattered (no threats).
    _place_many(board, [
        (7, 3), (10, 0),
        (7, 4), (12, 3),
        (7, 5), (10, 7),
    ])
    # Black has XXX at (7,3)-(7,5).  Winning moves check pre-move threats
    # only — the three is not yet a win.  But the forced-sequence search
    # discovers that (7,6) creates an unblockable open four.
    analysis = TacticalSolver.analyze(board)
    assert analysis.has_forced_win
    assert analysis.forced_sequence is not None
    # The first move of the forced sequence is one of the open-four
    # creation cells: either (7,2) or (7,6).
    assert analysis.forced_sequence[0] in {(7, 2), (7, 6)}


# ===========================================================================
# TacticalAnalysis — get_move_boost
# ===========================================================================


def test_winning_move_gets_huge_boost():
    """Winning moves should get an enormous prior boost."""
    board = Board()
    _place_many(board, [
        (7, 2), (0, 0),
        (7, 3), (0, 1),
        (7, 4), (0, 2),
        (7, 5), (0, 3),
    ])
    analysis = TacticalSolver.analyze(board)
    boost = analysis.get_move_boost((7, 1))
    assert boost >= 100.0, f"Winning move boost too low: {boost}"


def test_must_block_move_gets_huge_boost():
    """Must-block moves should get an enormous prior boost."""
    board = Board()
    _place_many(board, [
        (0, 0), (7, 2),  # White open four
        (0, 1), (7, 3),
        (0, 2), (7, 4),
        (0, 3), (7, 5),
    ])
    analysis = TacticalSolver.analyze(board)
    boost = analysis.get_move_boost((7, 1))
    assert boost >= 100.0, f"Must-block move boost too low: {boost}"


def test_non_tactical_move_gets_baseline_boost():
    """Non-tactical moves should get a boost near 1.0."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(8, 8)
    analysis = TacticalSolver.analyze(board)
    # A distant move like (0, 0) has no tactical value.
    boost = analysis.get_move_boost((0, 0))
    assert 0.5 <= boost < 10.0, f"Non-tactical boost out of range: {boost}"


# ===========================================================================
# TacticalAnalysis — get_forced_distribution
# ===========================================================================


def test_forced_distribution_returns_none_when_not_forced():
    """When the position is not forced, get_forced_distribution returns None."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(8, 8)
    analysis = TacticalSolver.analyze(board)
    assert analysis.get_forced_distribution() is None


def test_forced_distribution_is_valid():
    """Forced distributions should sum to 1 and only include legal moves."""
    board = Board()
    _place_many(board, [
        (7, 2), (0, 0),
        (7, 3), (0, 1),
        (7, 4), (0, 2),
        (7, 5), (0, 3),
    ])
    analysis = TacticalSolver.analyze(board)
    dist = analysis.get_forced_distribution()
    assert dist is not None
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-5
    legal = set(board.get_legal_moves())
    for m in dist:
        assert m in legal


# ===========================================================================
# TacticalAnalysis — priority ordering
# ===========================================================================


def test_priority_order_puts_winning_moves_first():
    """Winning moves should appear first in priority order."""
    board = Board()
    _place_many(board, [
        (7, 2), (0, 0),
        (7, 3), (0, 1),
        (7, 4), (0, 2),
        (7, 5), (0, 3),
    ])
    analysis = TacticalSolver.analyze(board)
    order = analysis.get_priority_order()
    # First entries should be winning moves.
    assert order[0] in {(7, 1), (7, 6)}
    assert order[1] in {(7, 1), (7, 6)}


def test_priority_order_puts_must_block_first():
    """Must-block moves should appear first in priority order."""
    board = Board()
    # White has open four.  Black stones spread out (no winning move).
    _place_many(board, [
        (10, 0), (7, 2),  # Black scattered, White open four
        (12, 3), (7, 3),
        (10, 6), (7, 4),
        (12, 9), (7, 5),
    ])
    analysis = TacticalSolver.analyze(board)
    order = analysis.get_priority_order()
    assert order[0] in {(7, 1), (7, 6)}
    assert order[1] in {(7, 1), (7, 6)}


# ===========================================================================
# TacticalSolver.analyze_lightweight
# ===========================================================================


def test_lightweight_detects_winning_moves():
    """analyze_lightweight should detect immediate wins."""
    board = Board()
    _place_many(board, [
        (7, 2), (0, 0),
        (7, 3), (0, 1),
        (7, 4), (0, 2),
        (7, 5), (0, 3),
    ])
    analysis = TacticalSolver.analyze_lightweight(board)
    assert analysis.has_forced_win
    assert analysis.winning_moves == {(7, 1), (7, 6)}


def test_lightweight_detects_must_block():
    """analyze_lightweight should detect must-block threats including
    opponent CLOSED_FOUR."""
    board = Board()
    _place_many(board, [
        (0, 0), (7, 2),  # White starts building four
        (0, 1), (7, 3),
        (0, 2), (7, 4),
        (0, 3), (7, 5),
        (7, 1), (8, 0),  # Black blocks left, White dummy
    ])
    analysis = TacticalSolver.analyze_lightweight(board)
    assert (7, 6) in analysis.must_block
    assert analysis.has_forced_defense


def test_lightweight_does_not_compute_double_threats():
    """analyze_lightweight should not compute double threats (for speed)."""
    board = Board()
    _place_many(board, [
        (7, 3), (0, 0),
        (7, 4), (0, 1),
        (6, 4), (0, 2),
        (8, 4), (0, 3),
    ])
    analysis = TacticalSolver.analyze_lightweight(board)
    # Lightweight doesn't compute double_threat_moves or creation_scores.
    assert len(analysis.double_threat_moves) == 0
    assert len(analysis.creation_scores) == 0


# ===========================================================================
# Integration with MCTS — verify the critical bug fixes work end-to-end
# ===========================================================================


def _make_mcts_wrapper():
    """Create a wrapper for MCTS integration tests."""
    import tempfile
    from pathlib import Path
    import torch
    from neural.model import GomokuNet
    from neural.wrapper import GomokuInferenceWrapper

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


def test_mcts_blocks_opponent_closed_four():
    """MCTS must block opponent's CLOSED_FOUR — the critical bug fix.

    Before the fix, opponent CLOSED_FOUR was not treated as a must-block
    threat, causing the engine to play random moves while the opponent
    wins next turn.
    """
    from selfplay.mcts import MCTS

    wrapper, tmp = _make_mcts_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        # White has closed four at (7,2)-(7,5), left blocked by Black at (7,1).
        _place_many(board, [
            (7, 1), (7, 2),  # Black blocks left, White starts four
            (0, 0), (7, 3),
            (0, 1), (7, 4),
            (0, 2), (7, 5),
        ])
        # Black's turn. White has CLOSED_FOUR at (7,2)-(7,5).
        # Black MUST block at (7,6).
        dist = mcts.search(board)
        assert set(dist.keys()) == {(7, 6)}, (
            f"MCTS failed to block opponent CLOSED_FOUR! "
            f"Got moves: {set(dist.keys())}"
        )
    finally:
        tmp.unlink()


def test_mcts_blocks_opponent_split_closed_four():
    """MCTS must block the gap in opponent's split CLOSED_FOUR."""
    from selfplay.mcts import MCTS

    wrapper, tmp = _make_mcts_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        # White has XX_XX at (7,2)(7,3)(7,5)(7,6) with gap at (7,4).
        # Black stones spread out (no threats).
        _place_many(board, [
            (10, 0), (7, 2),
            (12, 3), (7, 3),
            (10, 7), (7, 5),
            (12, 9), (7, 6),
        ])
        dist = mcts.search(board)
        assert set(dist.keys()) == {(7, 4)}, (
            f"Failed to detect split CLOSED_FOUR block at gap! "
            f"Got moves: {set(dist.keys())}"
        )
    finally:
        tmp.unlink()


def test_mcts_wins_when_opponent_cannot_block():
    """When we have a winning move, MCTS should find it immediately."""
    from selfplay.mcts import MCTS

    wrapper, tmp = _make_mcts_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        _place_many(board, [
            (7, 2), (0, 0),
            (7, 3), (0, 1),
            (7, 4), (0, 2),
            (7, 5), (0, 3),
        ])
        dist = mcts.search(board)
        assert set(dist.keys()) == {(7, 1), (7, 6)}
    finally:
        tmp.unlink()


def test_mcts_double_threat_creates_forced_win():
    """When we can create a double threat, MCTS should select that move."""
    from selfplay.mcts import MCTS

    wrapper, tmp = _make_mcts_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        # Set up position where (7,5) creates two open threes.
        # Black: (7,3),(7,4),(6,4),(8,4).  Playing (7,5) creates
        # horizontal three at (7,3)-(7,5) AND vertical three at
        # (6,4)-(8,4) — double threat.
        # White stones spread out (no threats).
        _place_many(board, [
            (7, 3), (10, 0),
            (7, 4), (12, 3),
            (6, 4), (10, 7),
            (8, 4), (12, 9),
        ])
        dist = mcts.search(board)
        assert (7, 5) in dist, (
            f"Double-threat move (7,5) not in MCTS distribution: "
            f"{set(dist.keys())}"
        )
    finally:
        tmp.unlink()


# ===========================================================================
# Edge cases
# ===========================================================================


def test_terminal_board_returns_empty_analysis():
    """A won board should return empty tactical analysis."""
    board = Board()
    for i in range(5):
        board.make_move(7, i)
        if i < 4:
            board.make_move(8, i)
    assert board.is_terminal()
    analysis = TacticalSolver.analyze(board)
    assert not analysis.has_forced_win
    assert not analysis.has_forced_defense
    assert analysis.get_forced_distribution() is None


def test_full_board_no_crash():
    """Analysis should not crash on any board state."""
    board = Board()
    # Fill most of the board without creating five-in-a-row.
    for r in range(15):
        for c in range(15):
            if (r + c) % 3 == 0 and board.grid[r, c] == 0:
                try:
                    if not board.is_terminal():
                        board.make_move(r, c)
                except ValueError:
                    pass
    # Should not crash.
    analysis = TacticalSolver.analyze(board)
    assert isinstance(analysis, TacticalAnalysis)


def test_get_priority_order_includes_all_scored_moves():
    """Priority ordering should include all moves with scores."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(8, 8)
    board.make_move(6, 6)
    analysis = TacticalSolver.analyze(board)
    order = analysis.get_priority_order()
    legal = set(board.get_legal_moves())
    # All legal moves should be in the priority order.
    for m in legal:
        assert m in order, f"Move {m} missing from priority order"


def test_get_move_boost_for_unscored_move():
    """A move not in any tactical set should still get a valid boost."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(8, 8)
    analysis = TacticalSolver.analyze(board)
    # (0, 14) is far from stones but should still get a boost value.
    boost = analysis.get_move_boost((0, 14))
    assert boost >= 1.0, f"Baseline boost should be ≥ 1.0, got {boost}"
