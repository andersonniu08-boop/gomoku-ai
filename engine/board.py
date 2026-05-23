from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class Player(IntEnum):
    BLACK = 1
    WHITE = -1


# Directions for win-checking: (row_delta, col_delta)
# Only need 4 directions — each covers both senses along an axis.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    """15x15 Gomoku board with NumPy backing.

    Optimized for MCTS:
    - O(1) make/undo via move history stack
    - get_legal_moves returns neighbours of existing stones (sparse)
    - check_win uses incremental axis scan from last move
    - Board state is copyable for parallel simulations
    """

    SIZE = 15
    WIN_LENGTH = 5

    def __init__(self) -> None:
        self.grid: NDArray[np.int8] = np.zeros(
            (self.SIZE, self.SIZE), dtype=np.int8
        )
        self.current_player: Player = Player.BLACK
        self.move_history: list[tuple[int, int]] = []
        self._winner: Optional[Player] = None
        # Track which positions are adjacent to at least one stone,
        # used to narrow legal moves to meaningful candidates.
        self._neighbor_set: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def make_move(self, row: int, col: int) -> None:
        """Place current player's stone and advance turn."""
        if self._winner is not None:
            raise ValueError("Game is already decided")
        if not self._in_bounds(row, col):
            raise ValueError(f"({row}, {col}) is out of bounds")
        if self.grid[row, col] != 0:
            raise ValueError(f"({row}, {col}) is already occupied")

        self.grid[row, col] = self.current_player
        self.move_history.append((row, col))

        self._update_neighbors(row, col)
        self._neighbor_set.discard((row, col))

        if self._check_win_at(row, col):
            self._winner = self.current_player

        self.current_player = Player(-self.current_player)

    def undo_move(self) -> None:
        """Undo the last move, restoring the previous board state."""
        if not self.move_history:
            raise ValueError("No moves to undo")

        row, col = self.move_history.pop()
        player = Player(self.grid[row, col])
        self.grid[row, col] = 0

        # Restore neighbor set: recompute from scratch only around
        # the undone cell — remove it, then re-add for remaining
        # stones in its neighbourhood.
        self._remove_stone_neighbors(row, col)
        self._add_neighbors_for(row, col)

        # If the board is now empty, neighbor set should be empty too.
        if not self.move_history:
            self._neighbor_set.clear()

        self._winner = None
        self.current_player = player

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Return candidate moves ordered for MCTS expansion.

        Returns positions adjacent to existing stones. If the board is
        empty, returns the center. This keeps the branching factor
        manageable (~20-40 instead of 225).
        """
        if not self.move_history:
            center = self.SIZE // 2
            return [(center, center)]

        # Filter to truly empty cells (undo may have invalidated some)
        moves = [pos for pos in self._neighbor_set if self.grid[pos] == 0]
        return moves

    def check_win(self) -> Optional[Player]:
        """Return the winner, or None if no winner yet."""
        return self._winner

    def is_terminal(self) -> bool:
        """True if the game has been won or the board is full."""
        return self._winner is not None or len(self.move_history) == self.SIZE**2

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def copy(self) -> Board:
        """Return an independent copy for parallel MCTS rollouts."""
        new = Board.__new__(Board)
        new.grid = self.grid.copy()
        new.current_player = self.current_player
        new.move_history = self.move_history.copy()
        new._winner = self._winner
        new._neighbor_set = self._neighbor_set.copy()
        return new

    def __repr__(self) -> str:
        symbols = {0: ".", Player.BLACK: "X", Player.WHITE: "O"}
        lines = []
        for r in range(self.SIZE):
            line = " ".join(symbols[int(self.grid[r, c])] for c in range(self.SIZE))
            lines.append(line)
        board_str = "\n".join(lines)
        return f"Board(turn={len(self.move_history)}, player={'X' if self.current_player == Player.BLACK else 'O'})\n{board_str}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.SIZE and 0 <= col < self.SIZE

    def _update_neighbors(self, row: int, col: int) -> None:
        """Add empty neighbours of (row, col) to the candidate set."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self._in_bounds(nr, nc) and self.grid[nr, nc] == 0:
                    self._neighbor_set.add((nr, nc))

    def _remove_stone_neighbors(self, row: int, col: int) -> None:
        """Remove neighbour entries that were only reachable via (row, col)."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self._in_bounds(nr, nc) and self.grid[nr, nc] == 0:
                    # Keep only if another stone still neighbours it
                    if not self._has_adjacent_stone(nr, nc):
                        self._neighbor_set.discard((nr, nc))

    def _add_neighbors_for(self, row: int, col: int) -> None:
        """Re-add neighbour entries for stones surrounding (row, col)."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self._in_bounds(nr, nc) and self.grid[nr, nc] != 0:
                    self._update_neighbors(nr, nc)

    def _has_adjacent_stone(self, row: int, col: int) -> bool:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self._in_bounds(nr, nc) and self.grid[nr, nc] != 0:
                    return True
        return False

    def _check_win_at(self, row: int, col: int) -> bool:
        """Check if the stone at (row, col) completes 5-in-a-row."""
        player = self.grid[row, col]
        for dr, dc in _DIRECTIONS:
            count = 1
            # Scan in the positive direction
            r, c = row + dr, col + dc
            while self._in_bounds(r, c) and self.grid[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Scan in the negative direction
            r, c = row - dr, col - dc
            while self._in_bounds(r, c) and self.grid[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= self.WIN_LENGTH:
                return True
        return False
