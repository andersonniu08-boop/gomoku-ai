"""Tests for engine.board — board state management and move legality."""

from engine.board import Board


def test_empty_board_has_225_legal_moves():
    """On an empty board, all 225 squares are legal moves."""
    board = Board()
    moves = board.get_legal_moves()
    assert len(moves) == 225
    positions = set(moves)
    for r in range(15):
        for c in range(15):
            assert (r, c) in positions


def test_non_adjacent_positions_are_legal():
    """Positions far from existing stones are still legal moves."""
    board = Board()
    board.make_move(7, 7)
    moves = board.get_legal_moves()
    assert (0, 0) in moves
    assert (0, 14) in moves
    assert (14, 0) in moves
    assert (14, 14) in moves


def test_occupied_positions_are_not_legal():
    """Cells with stones must not appear in legal moves."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(0, 0)
    moves = board.get_legal_moves()
    assert (7, 7) not in moves
    assert (0, 0) not in moves


def test_legal_moves_count_decreases_with_moves():
    """Each move should reduce the legal-move count by one."""
    board = Board()
    for i in range(5):
        before = len(board.get_legal_moves())
        board.make_move(i, i)
        after = len(board.get_legal_moves())
        assert after == before - 1


def test_legal_moves_after_undo():
    """After undo, legal moves should include the restored position."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(0, 0)
    board.undo_move()
    moves = board.get_legal_moves()
    assert (0, 0) in moves
    assert (7, 7) not in moves


def test_copy_preserves_legal_moves():
    """A copied board must have the same legal moves."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(0, 0)
    board.make_move(3, 3)
    copy = board.copy()
    assert board.get_legal_moves() == copy.get_legal_moves()
    copy.make_move(10, 10)
    assert (10, 10) in board.get_legal_moves()
    assert (10, 10) not in copy.get_legal_moves()


def test_legal_moves_after_complex_undo():
    """Undoing back to start restores full 225 legal moves."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(0, 0)
    board.make_move(14, 14)
    board.undo_move()
    board.undo_move()
    board.undo_move()
    assert len(board.get_legal_moves()) == 225


def test_make_move_only_checks_bounds_and_occupancy():
    """make_move validates bounds and occupancy, not adjacency."""
    board = Board()
    board.make_move(0, 0)
    assert board.grid[0, 0] != 0
    board.make_move(14, 14)
    assert board.grid[14, 14] != 0
