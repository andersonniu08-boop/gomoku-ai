"""Tensor encoding for Board → 3-channel neural network input."""

from __future__ import annotations

import numpy as np
import torch
from .board import Board, Player


def board_to_tensor(board: Board) -> torch.FloatTensor:
    """Convert the current board state into a (1, 3, 15, 15) FloatTensor.

    Encoded from the perspective of ``board.current_player``:

    - **Channel 0** – current player's stones  (1.0 = own stone,  0.0 = empty/opponent)
    - **Channel 1** – opponent's stones        (1.0 = opp stone,  0.0 = empty/own)
    - **Channel 2** – turn indicator           (1.0 everywhere if Black to move, 0.0 if White)
    """
    cp = board.current_player
    grid = board.grid  # int8: +1 Black, -1 White, 0 empty

    ch0 = (grid == cp).astype("float32")
    ch1 = (grid == -cp).astype("float32")
    ch2_val = 1.0 if cp == Player.BLACK else 0.0
    ch2 = np.full((Board.SIZE, Board.SIZE), ch2_val, dtype="float32")

    # Stack into (3, 15, 15) then add batch dim → (1, 3, 15, 15)
    tensor = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)
    return tensor


def policy_to_move_probs(
    log_policy: torch.Tensor, board: Board
) -> list[tuple[tuple[int, int], float]]:
    """Convert log-softmax policy output to a list of ((row, col), prob).

    Filters to only legal moves and normalises the distribution over them.
    """
    probs = torch.exp(log_policy).view(Board.SIZE, Board.SIZE).cpu().numpy()
    legal = board.get_legal_moves()

    result = [(move, float(probs[move])) for move in legal]
    total = sum(p for _, p in result)
    if total > 0:
        result = [(move, p / total) for move, p in result]
    else:
        # Degenerate model output — fall back to uniform.
        result = [(move, 1.0 / len(legal)) for move in legal]

    return result
