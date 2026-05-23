"""Tests for engine.threats — verify each threat pattern is detected."""

import pytest
from engine.board import Board, Player
from engine.threats import ThreatDetector, ThreatType, Threat


def _place_many(board: Board, moves: list[tuple[int, int]]) -> None:
    """Helper: play a sequence of alternating moves."""
    for r, c in moves:
        board.make_move(r, c)


def test_detects_five():
    board = Board()
    # X X X X X horizontal
    _place_many(board, [
        (7, 3), (8, 0),  # X O
        (7, 4), (8, 1),  # X O
        (7, 5), (8, 2),  # X O
        (7, 6), (8, 3),  # X O
        (7, 7),           # X wins
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    fives = [t for t in threats if t.threat_type == ThreatType.FIVE]
    assert len(fives) == 1
    assert fives[0].stones == [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]


def test_detects_open_four():
    board = Board()
    # _ X X X X _  (both ends open)
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (8, 1),
        (7, 5), (8, 2),
        (7, 6), (8, 3),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_fours = [t for t in threats if t.threat_type == ThreatType.OPEN_FOUR]
    assert len(open_fours) == 1
    assert len(open_fours[0].open_ends) == 2
    assert set(open_fours[0].open_ends) == {(7, 2), (7, 7)}


def test_detects_closed_four_left_blocked():
    board = Board()
    # O X X X X _  (left blocked by opponent, right open)
    board.make_move(7, 2)  # X (will be overwritten... need proper setup)
    # Let's set up more carefully
    board2 = Board()
    _place_many(board2, [
        (7, 2), (7, 1),  # X O (O blocks left of our four)
        (7, 3), (8, 0),  # X O
        (7, 4), (8, 1),  # X O
        (7, 5), (8, 2),  # X O
    ])
    threats = ThreatDetector.detect_all(board2, Player.BLACK)
    closed_fours = [t for t in threats if t.threat_type == ThreatType.CLOSED_FOUR]
    assert len(closed_fours) == 1
    assert len(closed_fours[0].open_ends) == 1


def test_detects_closed_four_right_blocked():
    board = Board()
    # _ X X X X O  (right blocked by opponent, left open)
    _place_many(board, [
        (7, 2), (8, 0),  # X O
        (7, 3), (7, 6),  # X O (O blocks right of our four)
        (7, 4), (8, 1),  # X O
        (7, 5), (8, 2),  # X O
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    closed_fours = [t for t in threats if t.threat_type == ThreatType.CLOSED_FOUR]
    assert len(closed_fours) == 1


def test_detects_open_three():
    board = Board()
    # _ _ X X X _ _  (both ends open with space to extend)
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (8, 1),
        (7, 5), (8, 2),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_threes = [t for t in threats if t.threat_type == ThreatType.OPEN_THREE]
    assert len(open_threes) == 1
    assert open_threes[0].stones == [(7, 3), (7, 4), (7, 5)]


def test_not_open_three_when_ends_tight():
    board = Board()
    # O _ X X X _ O  (both immediate ends open but wall/opponent beyond)
    _place_many(board, [
        (7, 2), (7, 1),  # X O (blocks left area)
        (7, 3), (8, 0),  # X O
        (7, 4), (7, 6),  # X O (blocks right area)
        (7, 5), (8, 1),  # X O
    ])
    # The three is at (7,3)(7,4)(7,5). Left-adjacent (7,2) is X (self),
    # right-adjacent (7,6) is O (opponent). So both ends blocked → not open.
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_threes = [t for t in threats if t.threat_type == ThreatType.OPEN_THREE]
    assert len(open_threes) == 0


def test_no_false_open_three_tight_space():
    board = Board()
    # _ X X X _  but with only 1 free cell on each side
    # Place so that the three has empty immediate neighbours but the
    # "beyond" cells are off-board or occupied.
    # columns: 0  1  2  3  4
    #           _  X  X  X  _
    # Place an opponent stone at column 0 to limit space.
    _place_many(board, [
        (7, 1), (7, 0),  # X at 1, O at 0 (blocks left space)
        (7, 2), (7, 4),  # X at 2, O at 4 (blocks right space)
        (7, 3), (8, 0),  # X at 3
    ])
    # Three at (7,1)(7,2)(7,3). Left: (7,0) is O → blocked.
    # So this isn't even 3 open ends... both ends are blocked.
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_threes = [t for t in threats if t.threat_type == ThreatType.OPEN_THREE]
    assert len(open_threes) == 0


def test_detects_diagonal_open_three():
    board = Board()
    # Diagonal: __XXX__ on main diagonal
    _place_many(board, [
        (5, 5), (8, 0),
        (6, 6), (8, 1),
        (7, 7), (8, 2),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_threes = [t for t in threats if t.threat_type == ThreatType.OPEN_THREE]
    assert len(open_threes) >= 1
    diag_three = [t for t in open_threes if t.direction == (1, 1)]
    assert len(diag_three) == 1


def test_detects_vertical_open_four():
    board = Board()
    # Vertical open four
    _place_many(board, [
        (3, 5), (8, 0),
        (4, 5), (8, 1),
        (5, 5), (8, 2),
        (6, 5), (8, 3),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_fours = [t for t in threats if t.threat_type == ThreatType.OPEN_FOUR]
    assert len(open_fours) == 1
    assert open_fours[0].direction == (1, 0)


def test_double_open_three():
    board = Board()
    # Create two open threes for Black: one horizontal, one vertical
    # Horizontal: __XXX__ at row 7
    # Vertical:   __XXX__ at col 3
    # They share the middle stone at (7,3).
    _place_many(board, [
        (7, 3), (10, 10),  # X O
        (7, 4), (10, 11),  # X O
        (7, 5), (10, 12),  # X O
        (5, 3), (10, 13),  # X O
        (6, 3), (10, 14),  # X O
    ])
    assert ThreatDetector.has_double_threat(board, Player.BLACK)
    assert ThreatDetector.has_open_three(board, Player.BLACK)


def test_open_four_plus_open_three():
    board = Board()
    # Black: horizontal open four at row 7 (cols 3-6) + vertical open three
    # at col 3 (rows 5-7, sharing (7,3)).
    # White: scattered stones — no threats, no 5-in-a-row.
    _place_many(board, [
        (7, 3), (8, 0),   # X O
        (7, 4), (8, 2),   # X O
        (7, 5), (8, 4),   # X O
        (7, 6), (8, 6),   # X O
        (5, 3), (8, 8),   # X O (White scattered; Black adds vertical start)
        (6, 3), (8, 10),  # X O (Black adds vertical cont; White scattered)
    ])
    assert ThreatDetector.has_double_threat(board, Player.BLACK)
    assert ThreatDetector.has_open_four(board, Player.BLACK)
    assert ThreatDetector.has_open_three(board, Player.BLACK)


def test_no_double_threat_with_single_open_three():
    board = Board()
    # Only one open three, no double threat
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (8, 1),
        (7, 5), (8, 2),
    ])
    assert not ThreatDetector.has_double_threat(board, Player.BLACK)
    assert ThreatDetector.has_open_three(board, Player.BLACK)


def test_evaluate_scores_black_ahead():
    board = Board()
    # Black has an open four, White has scattered stones (no threats)
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (8, 2),
        (7, 5), (8, 4),
        (7, 6), (8, 6),
    ])
    score = ThreatDetector.evaluate(board, Player.BLACK)
    assert score > 5000  # open four is worth 10000


def test_evaluate_scores_white_ahead():
    board = Board()
    # White has an open four, Black has scattered stones (no threats)
    _place_many(board, [
        (8, 0), (7, 3),   # X O
        (8, 2), (7, 4),   # X O
        (8, 4), (7, 5),   # X O
        (8, 6), (7, 6),   # X O
    ])
    score = ThreatDetector.evaluate(board, Player.BLACK)
    assert score < -5000  # opponent's open four


def test_count_threats():
    board = Board()
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (8, 1),
        (7, 5), (8, 2),
        (7, 6), (8, 3),
    ])
    counts = ThreatDetector.count_threats(board, Player.BLACK)
    assert counts[ThreatType.OPEN_FOUR] == 1
    assert counts[ThreatType.FIVE] == 0
    assert counts[ThreatType.OPEN_THREE] == 0


def test_empty_board_no_threats():
    board = Board()
    for player in (Player.BLACK, Player.WHITE):
        threats = ThreatDetector.detect_all(board, player)
        assert len(threats) == 0
        assert not ThreatDetector.has_double_threat(board, player)


def test_win_priority_over_other_threats():
    board = Board()
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (8, 1),
        (7, 5), (8, 2),
        (7, 6), (8, 3),
        (7, 7),           # five — wins
    ])
    assert board.check_win() == Player.BLACK
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    fives = [t for t in threats if t.threat_type == ThreatType.FIVE]
    assert len(fives) == 1


def test_multiple_open_threes():
    board = Board()
    # Horizontal open three at row 7 (cols 2-4) + another at row 9 (cols 6-8)
    # White: scattered stones with no threats
    _place_many(board, [
        (7, 2), (8, 0),
        (7, 3), (8, 2),
        (7, 4), (8, 4),
        (9, 6), (8, 6),
        (9, 7), (8, 8),
        (9, 8), (8, 10),
    ])
    counts = ThreatDetector.count_threats(board, Player.BLACK)
    assert counts[ThreatType.OPEN_THREE] == 2
    assert ThreatDetector.has_double_threat(board, Player.BLACK)


# -------------------------------------------------------------------
# Split / gap patterns
# -------------------------------------------------------------------


def test_split_four_xx_xx():
    board = Board()
    # XX_XX with both ends open — gap at col 2
    _place_many(board, [
        (7, 0), (8, 0),
        (7, 1), (8, 1),
        (7, 3), (8, 2),  # X skips col 2, plays at 3; O at (8,2)
        (7, 4), (8, 3),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    closed_fours = [t for t in threats if t.threat_type == ThreatType.CLOSED_FOUR]
    assert len(closed_fours) == 1
    assert closed_fours[0].gap == (7, 2)


def test_split_four_xxx_x():
    board = Board()
    # XXX_X with right end open — gap at col 3
    _place_many(board, [
        (7, 0), (8, 0),
        (7, 1), (8, 1),
        (7, 2), (8, 2),
        (7, 4), (8, 3),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    closed_fours = [t for t in threats if t.threat_type == ThreatType.CLOSED_FOUR]
    assert len(closed_fours) == 1
    assert closed_fours[0].gap == (7, 3)


def test_open_three_at_board_edge():
    board = Board()
    # Stones at (0,0)(0,1)(0,2) — left end is wall (blocked), right end
    # has space.  This cannot be an open three because one end is blocked.
    _place_many(board, [
        (0, 0), (7, 0),
        (0, 1), (7, 1),
        (0, 2), (7, 2),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_threes = [t for t in threats if t.threat_type == ThreatType.OPEN_THREE]
    assert len(open_threes) == 0  # left end is wall → not open


def test_five_at_board_edge():
    board = Board()
    _place_many(board, [
        (0, 0), (7, 0),
        (0, 1), (7, 1),
        (0, 2), (7, 2),
        (0, 3), (7, 3),
        (0, 4),           # win at edge
    ])
    assert ThreatDetector.has_five(board, Player.BLACK)


def test_closed_four_at_board_edge():
    board = Board()
    # Four stones against left wall: [0,0][0,1][0,2][0,3]
    # Right end (0,4) is open → closed four
    _place_many(board, [
        (0, 0), (7, 0),
        (0, 1), (7, 1),
        (0, 2), (7, 2),
        (0, 3), (7, 3),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    closed_fours = [t for t in threats if t.threat_type == ThreatType.CLOSED_FOUR]
    assert len(closed_fours) == 1


def test_anti_diagonal_open_three():
    board = Board()
    # Anti-diagonal (dr=1, dc=-1): __XXX__ from (5,6) → (7,4)
    _place_many(board, [
        (5, 6), (8, 0),
        (6, 5), (8, 1),
        (7, 4), (8, 2),
    ])
    threats = ThreatDetector.detect_all(board, Player.BLACK)
    open_threes = [t for t in threats if t.threat_type == ThreatType.OPEN_THREE]
    diag = [t for t in open_threes if t.direction == (1, -1)]
    assert len(diag) == 1


def test_no_threats_for_wrong_player():
    board = Board()
    _place_many(board, [
        (7, 3), (8, 0),
        (7, 4), (9, 2),
        (7, 5), (10, 4),
        (7, 6), (11, 6),
    ])
    # Black has an open four; White has scattered stones with no threats
    threats = ThreatDetector.detect_all(board, Player.WHITE)
    assert len(threats) == 0
