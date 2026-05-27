"""Tests for explain.comparison — human vs AI move comparison pipeline."""

import tempfile
from pathlib import Path

import torch

from engine.board import Board, Player
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from explain.comparison import (
    MoveCandidate,
    MoveComparison,
    compare_move,
    compare_move_fast,
)


def _make_wrapper():
    """Create a wrapper around a freshly-initialised (untrained) model."""
    model = GomokuNet(board_size=15, in_channels=3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    wrapper = GomokuInferenceWrapper(tmp_path, device="cpu")
    return wrapper, tmp_path


def _board_with_legal_move() -> tuple[Board, tuple[int, int]]:
    """Return a board with at least one legal move and a known legal move."""
    board = Board()
    board.make_move(7, 7)
    board.make_move(6, 6)
    return board, (7, 6)  # (7,6) is adjacent to existing stones


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


def test_legal_human_move():
    """A legal human move produces legal=True and the human move is ranked."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        # Use top_k large enough to guarantee the human move appears.
        legal = board.get_legal_moves()
        comp = compare_move(wrapper, board, human_move, num_simulations=10, top_k=len(legal))
        assert comp.legal is True
        assert comp.human_candidate is not None
        assert comp.human_candidate.move == human_move
        assert comp.human_rank is not None
        assert comp.human_rank >= 1
    finally:
        tmp.unlink()


def test_illegal_human_move():
    """An illegal (occupied) human move produces legal=False and no human_candidate."""
    wrapper, tmp = _make_wrapper()
    try:
        board, _ = _board_with_legal_move()  # stones at (7,7) and (6,6)
        # An occupied cell is always an illegal move.
        illegal = (7, 7)
        comp = compare_move(wrapper, board, illegal, num_simulations=10)
        assert comp.legal is False
        assert comp.human_candidate is None
    finally:
        tmp.unlink()


def test_ai_recommended_is_legal():
    """ai_recommended is always a legal move."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=10)
        legal = board.get_legal_moves()
        assert comp.ai_recommended in legal

        # Fast path.
        comp_fast = compare_move_fast(wrapper, board, human_move)
        assert comp_fast.ai_recommended in legal
    finally:
        tmp.unlink()


def test_value_range():
    """value_before and value_after are in [-1, 1]."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=10)
        assert -1.0 <= comp.value_before <= 1.0
        if comp.value_after is not None:
            assert -1.0 <= comp.value_after <= 1.0
    finally:
        tmp.unlink()


def test_top_candidates_sorted_by_visits():
    """Top candidates are sorted descending by visit_count (MCTS path)."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=30)
        assert len(comp.top_candidates) > 0
        for i in range(len(comp.top_candidates) - 1):
            assert comp.top_candidates[i].visit_count >= comp.top_candidates[i + 1].visit_count
    finally:
        tmp.unlink()


def test_top_candidates_sorted_by_prior_fast():
    """Top candidates are sorted descending by prior (fast path)."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move_fast(wrapper, board, human_move)
        assert len(comp.top_candidates) > 0
        for i in range(len(comp.top_candidates) - 1):
            assert comp.top_candidates[i].prior >= comp.top_candidates[i + 1].prior
    finally:
        tmp.unlink()


def test_top_k_length():
    """len(top_candidates) <= top_k."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        for top_k in (1, 3, 5, 10):
            comp = compare_move(wrapper, board, human_move, num_simulations=20, top_k=top_k)
            assert len(comp.top_candidates) <= top_k

            comp_fast = compare_move_fast(wrapper, board, human_move, top_k=top_k)
            assert len(comp_fast.top_candidates) <= top_k
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Fast path
# ---------------------------------------------------------------------------


def test_fast_path_structure():
    """Fast path produces same structure with visit_count=0 and q_value=0.0."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move_fast(wrapper, board, human_move)
        for c in comp.top_candidates:
            assert c.visit_count == 0
            assert c.q_value == 0.0
        # Fast path doesn't set search_stats; it's an empty dict.
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


def test_to_dict_does_not_raise():
    """to_dict() serialization completes without error."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=10)
        d = comp.to_dict()
        assert isinstance(d, dict)
        assert "human_move" in d
        assert "legal" in d
        assert "value_before" in d
    finally:
        tmp.unlink()


def test_from_dict_roundtrip():
    """MoveComparison.from_dict(d.to_dict()) == d."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=10)
        d = comp.to_dict()
        reconstructed = MoveComparison.from_dict(d)
        assert reconstructed == comp
    finally:
        tmp.unlink()


def test_from_dict_roundtrip_fast():
    """Fast path roundtrip also works."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move_fast(wrapper, board, human_move)
        d = comp.to_dict()
        reconstructed = MoveComparison.from_dict(d)
        assert reconstructed == comp
    finally:
        tmp.unlink()


def test_from_dict_roundtrip_illegal():
    """Roundtrip with illegal (occupied) move works."""
    wrapper, tmp = _make_wrapper()
    try:
        board, _ = _board_with_legal_move()  # stones at (7,7) and (6,6)
        illegal = (6, 6)  # occupied cell
        comp = compare_move(wrapper, board, illegal, num_simulations=10)
        d = comp.to_dict()
        reconstructed = MoveComparison.from_dict(d)
        assert reconstructed == comp
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_terminal_board():
    """Terminal board returns immediately, no crash."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        # Black gets five in a row.
        for i in range(5):
            board.make_move(7, i)
            if i < 4:
                board.make_move(8, i)

        assert board.is_terminal()
        comp = compare_move(wrapper, board, (0, 0), num_simulations=10)
        assert comp.legal is False
        assert comp.top_candidates == []
        assert comp.human_candidate is None
    finally:
        tmp.unlink()


def test_empty_board():
    """Empty board: AI picks a legal move (center)."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        comp = compare_move(wrapper, board, (7, 7), num_simulations=20)
        legal = board.get_legal_moves()
        assert comp.ai_recommended in legal
        assert comp.legal is True
    finally:
        tmp.unlink()


def test_threat_override():
    """Threat override is detected when a forced win exists."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        # Set up Black with open four at (7,2)-(7,5) — needs (7,1) or (7,6) to win.
        white_cols = [0, 2, 4, 6]
        for r, c in [(7, 2), (7, 3), (7, 4), (7, 5)]:
            board.make_move(r, c)
            board.make_move(8, white_cols.pop(0))

        # Human plays one of the winning moves.
        comp = compare_move(wrapper, board, (7, 1), num_simulations=10)
        # With threat_override=True, the MCTS should short-circuit.
        assert comp.threat_overridden is True
        assert comp.ai_recommended in {(7, 1), (7, 6)}
    finally:
        tmp.unlink()


def test_search_stats_contains_num_simulations():
    """Search stats includes num_simulations matching the argument."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=25)
        assert comp.search_stats.get("num_simulations") == 25
    finally:
        tmp.unlink()


def test_human_rank_in_top_k():
    """Human rank is >= 1 when the human move is legal (may be outside top-k)."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        comp = compare_move(wrapper, board, human_move, num_simulations=30, top_k=10)
        if comp.human_rank is not None:
            assert comp.human_rank >= 1
    finally:
        tmp.unlink()


def test_value_after_improves_on_win():
    """When human plays a winning move, value_after > value_before."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        # Black builds an open four at row 7, cols 2-5.
        # Black is about to win by playing (7,1) or (7,6).
        white_cols = [0, 2, 4, 6]
        for r, c in [(7, 2), (7, 3), (7, 4), (7, 5)]:
            board.make_move(r, c)
            board.make_move(8, white_cols.pop(0))

        # Human plays (7,6) — the winning move.
        comp = compare_move(wrapper, board, (7, 6), num_simulations=10)
        # After playing (7,6), Black wins → value_after should be 1.0.
        # A random model's value_before is almost never exactly 1.0.
        assert comp.value_after is not None
        assert comp.value_after > comp.value_before
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# MCTS search_with_stats integration
# ---------------------------------------------------------------------------


def test_search_with_stats_returns_valid_data():
    """Verify search_with_stats produces structured output."""
    wrapper, tmp = _make_wrapper()
    try:
        from selfplay.mcts import MCTS, SearchResult

        board, _ = _board_with_legal_move()
        mcts = MCTS(wrapper, num_simulations=30)
        result = mcts.search_with_stats(board)

        assert isinstance(result, SearchResult)
        assert len(result.visit_counts) > 0
        assert len(result.q_values) > 0
        assert len(result.priors) > 0
        # Every visited move should have a matching q_value and prior.
        for move in result.visit_counts:
            assert move in result.q_values
            assert move in result.priors
        assert result.total_simulations == 30
    finally:
        tmp.unlink()


def test_search_with_stats_threat_override():
    """search_with_stats with threat override returns forced distribution."""
    wrapper, tmp = _make_wrapper()
    try:
        from selfplay.mcts import MCTS

        board = Board()
        white_cols = [0, 2, 4, 6]
        for r, c in [(7, 2), (7, 3), (7, 4), (7, 5)]:
            board.make_move(r, c)
            board.make_move(8, white_cols.pop(0))

        mcts = MCTS(wrapper, num_simulations=100, threat_override=True)
        result = mcts.search_with_stats(board)

        # Should detect open-four → forced win at both ends.
        assert result.total_simulations == 0  # Overridden
        assert set(result.visit_counts.keys()) == {(7, 1), (7, 6)}
        for move in result.visit_counts:
            assert result.visit_counts[move] == 1
            assert result.q_values[move] == 0.0
    finally:
        tmp.unlink()


def test_search_with_stats_terminal():
    """search_with_stats on a terminal board returns empty result."""
    wrapper, tmp = _make_wrapper()
    try:
        from selfplay.mcts import MCTS

        board = Board()
        for i in range(5):
            board.make_move(7, i)
            if i < 4:
                board.make_move(8, i)

        assert board.is_terminal()
        mcts = MCTS(wrapper)
        result = mcts.search_with_stats(board)
        assert result.visit_counts == {}
        assert result.q_values == {}
        assert result.priors == {}
        assert result.total_simulations == 0
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Fast path additional verification
# ---------------------------------------------------------------------------


def test_fast_path_value_before_matches():
    """Fast path value_before should equal wrapper.evaluate()."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        _, expected_value = wrapper.evaluate(board)
        comp = compare_move_fast(wrapper, board, human_move)
        assert comp.value_before == expected_value
    finally:
        tmp.unlink()


def test_value_after_none_on_illegal_move():
    """value_after should be None when the human move is illegal (occupied)."""
    wrapper, tmp = _make_wrapper()
    try:
        board, _ = _board_with_legal_move()  # stones at (7,7) and (6,6)
        illegal = (7, 7)  # occupied cell is always illegal
        comp = compare_move(wrapper, board, illegal, num_simulations=10)
        assert comp.legal is False
        assert comp.value_after is None
    finally:
        tmp.unlink()


def test_compare_move_fast_is_equivalent_to_use_mcts_false():
    """compare_move_fast produces the same result as compare_move(use_mcts=False)."""
    wrapper, tmp = _make_wrapper()
    try:
        board, human_move = _board_with_legal_move()
        fast = compare_move_fast(wrapper, board, human_move)
        explicit = compare_move(wrapper, board, human_move, use_mcts=False)
        assert fast.to_dict() == explicit.to_dict()
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# MoveCandidate dataclass defaults
# ---------------------------------------------------------------------------


def test_move_candidate_defaults():
    """MoveCandidate has sensible defaults for optional fields."""
    c = MoveCandidate(move=(3, 4), prior=0.5)
    assert c.visit_count == 0
    assert c.q_value == 0.0
    assert c.is_human_move is False
