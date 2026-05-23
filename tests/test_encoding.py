"""Tests for engine.encoding — verify board → tensor conversion."""

import torch
from engine.board import Board, Player
from engine.encoding import board_to_tensor, policy_to_move_probs


def test_tensor_shape():
    board = Board()
    t = board_to_tensor(board)
    assert t.shape == (1, 3, 15, 15)
    assert t.dtype == torch.float32


def test_channel_0_current_player_stones():
    board = Board()
    board.make_move(7, 7)  # Black
    board.make_move(8, 8)  # White
    # Now current_player is Black again
    t = board_to_tensor(board)
    assert t[0, 0, 7, 7] == 1.0  # Black's own stone
    assert t[0, 0, 8, 8] == 0.0  # not Black's stone


def test_channel_1_opponent_stones():
    board = Board()
    board.make_move(7, 7)  # Black
    board.make_move(8, 8)  # White
    # current_player is Black, opponent is White
    t = board_to_tensor(board)
    assert t[0, 1, 8, 8] == 1.0  # White's stone (opponent)
    assert t[0, 1, 7, 7] == 0.0  # Black's stone (not opponent)


def test_channel_2_turn_indicator_black():
    board = Board()
    t = board_to_tensor(board)
    assert torch.all(t[0, 2] == 1.0).item()  # Black to move


def test_channel_2_turn_indicator_white():
    board = Board()
    board.make_move(7, 7)  # Black moves, now it's White's turn
    t = board_to_tensor(board)
    assert torch.all(t[0, 2] == 0.0).item()  # White to move


def test_empty_board_all_channels():
    board = Board()
    t = board_to_tensor(board)
    assert torch.all(t[0, 0] == 0.0).item()  # no stones for current player
    assert torch.all(t[0, 1] == 0.0).item()  # no stones for opponent
    assert torch.all(t[0, 2] == 1.0).item()  # Black to move


def test_policy_to_move_probs_normalizes():
    board = Board()
    log_policy = torch.zeros(1, 225).log_softmax(dim=1)  # uniform
    probs = policy_to_move_probs(log_policy, board)
    assert len(probs) == 1  # only center is legal on empty board
    assert probs[0][0] == (7, 7)
    assert abs(probs[0][1] - 1.0) < 1e-6
