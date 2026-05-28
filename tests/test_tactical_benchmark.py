"""Tactical benchmark suite — evaluates MCTS tactical correctness."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pytest
import torch
from engine.board import Board
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS

@pytest.fixture(scope="module")
def wrapper():
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        p = Path(f.name)
    w = GomokuInferenceWrapper(p, device="cpu", num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
    yield w
    p.unlink(missing_ok=True)

def pm(board, moves):
    for r, c in moves:
        board.make_move(r, c)

class TestWinInOne:
    def test_open_four_both_ends(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(7,2),(0,0),(7,3),(0,1),(7,4),(0,2),(7,5),(0,3)])
        assert set(mcts.search(b).keys()) == {(7,1),(7,6)}
    def test_contiguous_closed_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(7,2),(7,1),(7,3),(8,0),(7,4),(8,2),(7,5),(8,4)])
        assert set(mcts.search(b).keys()) == {(7,6)}
    def test_split_closed_four_xx_xx(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(7,2),(0,0),(7,3),(0,1),(7,5),(0,2),(7,6),(0,3)])
        assert set(mcts.search(b).keys()) == {(7,4)}
    def test_split_closed_four_xxx_x(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(7,2),(0,0),(7,3),(0,1),(7,4),(0,2),(7,6),(0,3)])
        assert set(mcts.search(b).keys()) == {(7,5)}
    def test_vertical_open_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(3,7),(0,0),(4,7),(0,1),(5,7),(0,2),(6,7),(0,3)])
        assert set(mcts.search(b).keys()) == {(2,7),(7,7)}
    def test_diagonal_open_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(3,3),(0,0),(4,4),(0,1),(5,5),(0,2),(6,6),(0,3)])
        assert set(mcts.search(b).keys()) == {(2,2),(7,7)}
    def test_anti_diagonal_open_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(3,6),(0,0),(4,5),(0,1),(5,4),(0,2),(6,3),(0,3)])
        assert set(mcts.search(b).keys()) == {(2,7),(7,2)}

class TestForcedDefense:
    def test_block_open_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(2,2),(7,2),(4,4),(7,3),(6,6),(7,4),(8,8),(7,5)])
        assert set(mcts.search(b).keys()) == {(7,1),(7,6)}
    def test_block_contiguous_closed_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(7,1),(7,2),(8,0),(7,3),(8,2),(7,4),(8,4),(7,5)])
        assert set(mcts.search(b).keys()) == {(7,6)}
    def test_block_split_closed_four(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(10,0),(7,2),(12,3),(7,3),(10,6),(7,5),(12,9),(7,6)])
        assert set(mcts.search(b).keys()) == {(7,4)}
    def test_block_split_closed_four_xxx_x(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(10,0),(7,2),(12,3),(7,3),(10,6),(7,4),(12,9),(7,6)])
        assert set(mcts.search(b).keys()) == {(7,5)}
    def test_win_takes_priority(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(7,2),(10,0),(7,3),(10,1),(7,4),(10,2),(7,5),(10,3)])
        assert set(mcts.search(b).keys()) == {(7,1),(7,6)}

class TestDoubleThreat:
    def test_create_double_open_three(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=80, threat_override=True)
        b = Board(); pm(b, [(7,3),(13,0),(7,4),(13,2),(5,5),(13,4),(6,5),(13,6)])
        d = mcts.search(b)
        assert (7,5) in set(d.keys())
    def test_open_four_plus_open_three(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=80, threat_override=True)
        b = Board(); pm(b, [(7,2),(1,0),(7,3),(1,2),(7,4),(1,4),(7,5),(1,6),(5,9),(1,8),(6,9),(1,10)])
        assert set(mcts.search(b).keys()) == {(7,1),(7,6)}

class TestEdgeCases:
    def test_empty_board(self, wrapper):
        d = MCTS(wrapper, num_simulations=10, threat_override=True).search(Board())
        assert len(d) > 0 and abs(sum(d.values()) - 1.0) < 1e-5
    def test_terminal_board(self, wrapper):
        b = Board()
        for i in range(5):
            b.make_move(7, i)
            if i < 4: b.make_move(8, i)
        assert len(MCTS(wrapper, num_simulations=10, threat_override=True).search(b)) == 0
    def test_near_edge_threats(self, wrapper):
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        b = Board(); pm(b, [(0,0),(7,0),(0,1),(7,1),(0,2),(7,2),(0,3),(7,3)])
        assert set(mcts.search(b).keys()) == {(0,4)}
