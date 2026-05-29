from __future__ import annotations

import random
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

_ZOBRIST_TABLE: Optional[list[list[list[int]]]] = None


def _init_zobrist_table() -> list[list[list[int]]]:
    """Lazily generate deterministic Zobrist keys.

    Uses a fixed seed so keys are reproducible across processes.
    ``_ZOBRIST_TABLE[r][c][p_idx]`` where *p_idx* is 0 for Black, 1 for White.
    """
    rng = random.Random(42)
    return [
        [[rng.getrandbits(64) for _ in range(2)] for _ in range(15)]
        for _ in range(15)
    ]


def _zobrist_player_index(player: Player) -> int:
    return 0 if player == Player.BLACK else 1


class Board:
    """15x15 Gomoku board with NumPy backing.

    Optimized for MCTS:
    - O(1) make/undo via move history stack
    - get_legal_moves returns all empty board positions (standard Gomoku)
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
        self._zobrist_key: int = 0

    @property
    def zobrist_key(self) -> int:
        """64-bit Zobrist hash of the current board state.

        Incrementally updated on every ``make_move`` / ``undo_move``.
        """
        return self._zobrist_key

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

        self._update_zobrist(row, col, self.current_player)

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

        self._update_zobrist(row, col, player)

        self._winner = None
        self.current_player = player

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Return all empty board positions.

        Any empty square is a legal move in standard Gomoku.  The MCTS
        layer applies its own pruning (via ``order_and_filter_moves``) to
        keep the branching factor manageable during search.
        """
        return [
            (r, c)
            for r in range(self.SIZE)
            for c in range(self.SIZE)
            if self.grid[r, c] == 0
        ]

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
        new._zobrist_key = self._zobrist_key
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

    def _update_zobrist(self, row: int, col: int, player: Player) -> None:
        """XOR the Zobrist entry for *player* at *(row, col)* in/out."""
        global _ZOBRIST_TABLE
        if _ZOBRIST_TABLE is None:
            _ZOBRIST_TABLE = _init_zobrist_table()
        p_idx = _zobrist_player_index(player)
        self._zobrist_key ^= _ZOBRIST_TABLE[row][col][p_idx]
