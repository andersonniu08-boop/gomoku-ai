"""Tests for engine.board — verify board state management and move legality."""

from engine.board import Board, Player


def test_empty_board_has_225_legal_moves():
    """On an empty board, all 225 squares are legal moves."""
    board = Board()
    moves = board.get_legal_moves()
    assert len(moves) == 225
    # Every position (r,c) for 0 <= r,c < 15 should be present.
    positions = set(moves)
    assert len(positions) == 225
    for r in range(15):
        for c in range(15):
            assert (r, c) in positions


def test_non_adjacent_positions_are_legal():
    """Positions far from existing stones are still legal moves."""
    board = Board()
    board.make_move(7, 7)  # one stone in the center
    moves = board.get_legal_moves()
    # The empty corner (0,0) is far from (7,7) but must be legal.
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
    board.undo_move()  # removes (0,0)
    moves = board.get_legal_moves()
    assert (0, 0) in moves
    assert (7, 7) not in moves  # still occupied


def test_move_count_decreases_in_sync():
    """Each play reduces legal-move count by 1 regardless of position."""
    board = Board()
    moves_done = 0
    for r in range(4):
        for c in range(4):
            if board.grid[r, c] == 0:
                before = len(board.get_legal_moves())
                board.make_move(r, c)
                after = len(board.get_legal_moves())
                assert after == before - 1
                moves_done += 1
                if board.is_terminal():
                    break
        if board.is_terminal():
            break
    assert moves_done > 0


def test_copy_preserves_legal_moves():
    """A copied board must have the same legal moves."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(0, 0)
    board.make_move(3, 3)
    copy = board.copy()
    assert board.get_legal_moves() == copy.get_legal_moves()
    # Modifying the copy must not affect the original.
    copy.make_move(10, 10)
    # Original should still have (10,10) as legal (independence).
    assert (10, 10) in board.get_legal_moves()
    # Copy should no longer have (10,10) as legal (just played).
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
    moves = board.get_legal_moves()
    assert len(moves) == 225


def test_make_move_only_checks_bounds_and_occupancy():
    """make_move only validates bounds and occupancy, not adjacency."""
    board = Board()
    # A move in an empty corner far from any stone must be accepted.
    board.make_move(0, 0)
    assert board.grid[0, 0] != 0
    # A move anywhere else on the empty board (aside from occupied) works.
    board.make_move(14, 14)
    assert board.grid[14, 14] != 0
