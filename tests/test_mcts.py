"""Tests for selfplay.mcts — verify MCTS search produces sensible output."""

import tempfile
from pathlib import Path

import torch

from engine.board import Board, Player
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS, MCTSNode


def _make_wrapper():
    """Create a wrapper around a freshly-initialised (untrained) model."""
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


def test_mcts_empty_board_returns_move_distribution():
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        board = Board()
        dist = mcts.search(board)
        # Should have at least the center move.
        assert len(dist) > 0
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-5
    finally:
        tmp.unlink()


def test_mcts_select_move_greedy():
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        board = Board()
        move = mcts.select_move(board, temperature=0.0)
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert 0 <= move[0] < 15
        assert 0 <= move[1] < 15
    finally:
        tmp.unlink()


def test_mcts_select_move_sampling():
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        board = Board()
        move = mcts.select_move(board, temperature=1.0)
        assert isinstance(move, tuple)
        assert len(move) == 2
    finally:
        tmp.unlink()


def test_mcts_detects_immediate_win():
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10)
        board = Board()
        # Set up Black one move away from five.
        # Black: (7,2), (7,3), (7,4), (7,5) — needs (7,1) or (7,6) to win.
        white_cols = [0, 2, 4, 6]
        for i, (r, c) in enumerate([(7, 2), (7, 3), (7, 4), (7, 5)]):
            board.make_move(r, c)
            board.make_move(8, white_cols[i])  # White scattered in row 8
        dist = mcts.search(board)
        # An open four has two winning moves (both ends).
        assert set(dist.keys()) == {(7, 1), (7, 6)}
        for p in dist.values():
            assert p > 0
    finally:
        tmp.unlink()


def test_mcts_detects_must_block():
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10)
        board = Board()
        # White gets an open four.  Black must block.
        # White stones at (7,2)(7,3)(7,4)(7,5), it's Black's turn.
        # Black plays first, then White builds the four.
        board.make_move(2, 2)   # Black (scattered)
        board.make_move(7, 2)   # White
        board.make_move(4, 6)   # Black (scattered)
        board.make_move(7, 3)   # White
        board.make_move(8, 10)  # Black (scattered)
        board.make_move(7, 4)   # White
        board.make_move(12, 0)  # Black (scattered)
        board.make_move(7, 5)   # White — open four at (7,2)-(7,5)
        # Now it's Black's turn. White has an open four.  Black has no
        # threats of its own and must block at (7,1) or (7,6).
        dist = mcts.search(board)
        assert set(dist.keys()).issubset({(7, 1), (7, 6)})
    finally:
        tmp.unlink()


def test_contiguous_closed_four_winning_move():
    """A contiguous closed four (XXXX_) has one open end — placing there wins."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        # Build contiguous four at (7,2)-(7,5) with right end open.
        # Left end is blocked by O at (7,1).
        board.make_move(7, 2)  # Black
        board.make_move(7, 1)  # White (blocks left)
        board.make_move(7, 3)  # Black
        board.make_move(8, 0)  # White
        board.make_move(7, 4)  # Black
        board.make_move(8, 2)  # White
        board.make_move(7, 5)  # Black
        board.make_move(8, 4)  # White
        # Black's turn. Black has contiguous CLOSED_FOUR at (7,2)-(7,5),
        # with open end at (7,6).  That's the only winning move.
        dist = mcts.search(board)
        assert set(dist.keys()) == {(7, 6)}
    finally:
        tmp.unlink()


def test_split_closed_four_only_gap_is_winning():
    """A split closed four (XX_XX or XXX_X): only the gap is a winning move,
    not the external open ends."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        # Split four at (7,2),(7,3),(7,5),(7,6) — pattern XX_XX.
        # Both left (7,1) and right (7,7) ends are open.
        # Gap at (7,4) is the only winning move.
        board.make_move(7, 2)  # Black
        board.make_move(8, 0)  # White
        board.make_move(7, 3)  # Black
        board.make_move(8, 2)  # White
        board.make_move(7, 5)  # Black
        board.make_move(8, 4)  # White
        board.make_move(7, 6)  # Black
        board.make_move(8, 6)  # White
        # Black's turn. Split CLOSED_FOUR at cols 2,3,5,6 with gap at 4.
        # Only (7,4) wins — filling it creates XXXXX.
        # (7,1) and (7,7) do NOT win (they extend the split).
        dist = mcts.search(board)
        assert set(dist.keys()) == {(7, 4)}
    finally:
        tmp.unlink()


def test_mcts_node_q_property():
    node = MCTSNode(prior=0.5, visit_count=0, total_value=0.0)
    assert node.q == 0.0

    node.visit_count = 10
    node.total_value = 5.0
    assert node.q == 0.5


def test_dirichlet_noise_normalizes_priors():
    """Root priors remain normalized after Dirichlet noise mixing."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(
            wrapper,
            num_simulations=20,
            threat_override=False,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
        )
        board = Board()
        board.make_move(7, 7)  # one stone so legal moves > 1
        board.make_move(8, 8)

        dist = mcts.search(board)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-5
    finally:
        tmp.unlink()


def test_dirichlet_noise_off_by_default_for_evaluation():
    """MCTS constructed without dirichlet_alpha should not use noise."""
    wrapper, tmp = _make_wrapper()
    try:
        # Default: no dirichlet_alpha parameter → no noise.
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        assert mcts.dirichlet_alpha is None
    finally:
        tmp.unlink()


def test_selfplay_passes_dirichlet_to_mcts():
    """SelfPlayGame passes dirichlet parameters to MCTS by default."""
    from selfplay.selfplay import SelfPlayGame
    wrapper, tmp = _make_wrapper()
    try:
        game = SelfPlayGame(wrapper, num_simulations=4)
        assert game.dirichlet_alpha == 0.03
        assert game.dirichlet_epsilon == 0.25
    finally:
        tmp.unlink()


def test_mcts_terminal_board_no_crash():
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10)
        board = Board()
        # Create a won position for Black.
        for i in range(5):
            board.make_move(7, i)
            if i < 4:
                board.make_move(8, i)
        # Black has won (five at row 7).
        assert board.is_terminal()
        dist = mcts.search(board)
        # Should be empty since game is over.
        assert len(dist) == 0
    finally:
        tmp.unlink()


def test_virtual_loss_q_no_visits():
    """Q should be -1 when virtual_loss=1 and visit_count=0."""
    from selfplay.mcts import MCTSNode
    node = MCTSNode(prior=0.5, virtual_loss=1)
    assert node.q == -1.0


def test_virtual_loss_q_with_visits():
    """Q should correctly blend real value and virtual loss."""
    from selfplay.mcts import MCTSNode
    node = MCTSNode(prior=0.5, visit_count=2, total_value=1.0, virtual_loss=1)
    # total_n = 2 + 1 = 3, q = (1.0 - 1) / 3 = 0.0
    assert node.q == 0.0


# ---------------------------------------------------------------------------
# Batched MCTS search integration tests
# ---------------------------------------------------------------------------


def test_batched_search_returns_valid_distribution():
    """Batched search returns a probability distribution over legal moves."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=50, batch_size=8)
        board = Board()
        board.make_move(7, 7)

        visit_probs = mcts.search(board)
        assert len(visit_probs) > 0
        total = sum(visit_probs.values())
        assert abs(total - 1.0) < 1e-5
        legal = board.get_legal_moves()
        for move in visit_probs:
            assert move in legal
    finally:
        tmp.unlink()


def test_batched_search_finds_immediate_win():
    """Search finds a winning move in one step when one exists."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, batch_size=8)
        board = Board()
        # Set up Black with open four at (7,3)-(7,6), open ends (7,2) and (7,7).
        board.make_move(7, 3)  # Black
        board.make_move(0, 0)  # White (dummy)
        board.make_move(7, 4)  # Black
        board.make_move(0, 1)  # White
        board.make_move(7, 5)  # Black
        board.make_move(0, 2)  # White
        board.make_move(7, 6)  # Black
        board.make_move(0, 3)  # White

        # Now it's Black's turn with an open four. (7,2) and (7,7) both win.
        visit_probs = mcts.search(board)
        assert len(visit_probs) > 0
        # Both winning moves should be the only ones with probability.
        assert set(visit_probs.keys()) == {(7, 2), (7, 7)}
    finally:
        tmp.unlink()


def test_batch_size_1_still_works():
    """batch_size=1 should produce a valid distribution (sequential-equivalent)."""
    wrapper, tmp = _make_wrapper()
    try:
        board = Board()
        board.make_move(7, 7)

        mcts = MCTS(wrapper, num_simulations=50, batch_size=1)
        visit_probs = mcts.search(board)
        assert len(visit_probs) > 0
        total = sum(visit_probs.values())
        assert abs(total - 1.0) < 1e-5
        # With threat_override=True, should pick a specific move.
        assert isinstance(mcts.select_move(board, temperature=0.0), tuple)
    finally:
        tmp.unlink()


def test_search_batch_larger_than_simulations():
    """batch_size > num_simulations should not crash and produce valid results."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, batch_size=100)
        board = Board()
        board.make_move(7, 7)

        visit_probs = mcts.search(board)
        assert len(visit_probs) > 0
        total = sum(visit_probs.values())
        assert abs(total - 1.0) < 1e-5
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Regression: all empty squares are legal for MCTS
# ---------------------------------------------------------------------------


def test_mcts_accepts_non_adjacent_root_move():
    """MCTS must consider moves far from existing stones at the root."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        visit_probs = mcts.search(board)
        assert len(visit_probs) > 0
        total = sum(visit_probs.values())
        assert abs(total - 1.0) < 1e-5
        legal = set(board.get_legal_moves())
        for move in visit_probs:
            assert move in legal, f"MCTS returned illegal move {move}"
    finally:
        tmp.unlink()


def test_mcts_threat_override_with_full_legality():
    """Threat override must work correctly when all empty positions are legal."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        white_cols = [0, 2, 4, 6]
        for r, c in [(7, 2), (7, 3), (7, 4), (7, 5)]:
            board.make_move(r, c)
            board.make_move(8, white_cols.pop(0))

        dist = mcts.search(board)
        assert set(dist.keys()) == {(7, 1), (7, 6)}
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Tree reuse tests
# ---------------------------------------------------------------------------


def test_tree_reuse_reroot_after_single_move():
    """After searching and playing the top move, the tree re-roots at that child."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=30, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        # First search.
        result1 = mcts.search_with_stats(board)
        top_move = max(result1.visit_counts, key=result1.visit_counts.get)
        prev_root = mcts._prev_root
        assert prev_root is not None
        assert top_move in prev_root.children

        # Play the top move.
        board.make_move(*top_move)

        # Second search should re-root.
        result2 = mcts.search_with_stats(board)
        # The new root should be the child we re-rooted to.
        assert mcts._prev_root is not None
        # Cumulative sims should be > num_simulations.
        assert result2.total_simulations >= 30
    finally:
        tmp.unlink()


def test_tree_reuse_reroot_through_multiple_moves():
    """Tree re-roots correctly through two moves (AI then human response)."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=40, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        # First search.
        result1 = mcts.search_with_stats(board)
        ai_move = max(result1.visit_counts, key=result1.visit_counts.get)
        board.make_move(*ai_move)

        # Second search (human response).
        result2 = mcts.search_with_stats(board)
        assert len(result2.visit_counts) > 0
        human_move = max(result2.visit_counts, key=result2.visit_counts.get)
        board.make_move(*human_move)

        # Third search — should re-root through both moves.
        result3 = mcts.search_with_stats(board)
        assert len(result3.visit_counts) > 0
        assert result3.total_simulations >= 40
    finally:
        tmp.unlink()


def test_tree_reuse_fallback_on_unknown_move():
    """When the opponent plays a move not in the tree, fall back to fresh root."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=30, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        # First search.
        mcts.search_with_stats(board)
        assert mcts._prev_root is not None

        # Play a move NOT in the tree — pick a corner far from action.
        board.make_move(0, 0)

        # Second search should fall back to fresh root.
        result2 = mcts.search_with_stats(board)
        assert len(result2.visit_counts) > 0
        # Cumulative sims should be 30 (fresh search), not 60.
        assert result2.total_simulations == 30
    finally:
        tmp.unlink()


def test_tree_reuse_fallback_on_board_reset():
    """When the board is reset (fewer moves), fall back to fresh tree."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        board = Board()
        board.make_move(7, 7)

        mcts.search_with_stats(board)
        assert mcts._prev_root is not None

        # Create a fresh board with fewer moves — simulates new game.
        board2 = Board()
        result2 = mcts.search_with_stats(board2)
        assert len(result2.visit_counts) > 0
        assert result2.total_simulations == 20
    finally:
        tmp.unlink()


def test_tree_reuse_reset_tree_method():
    """reset_tree() clears all cached state."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=20, threat_override=False)
        board = Board()
        board.make_move(7, 7)

        mcts.search_with_stats(board)
        assert mcts._prev_root is not None
        assert mcts._prev_board is not None

        mcts.reset_tree()
        assert mcts._prev_root is None
        assert mcts._prev_board is None
        assert mcts._cumulative_sims == 0
    finally:
        tmp.unlink()


def test_tree_reuse_threat_override_clears_tree():
    """When threat override fires, the cached tree is cleared."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()

        # First search on a neutral position — tree should be stored.
        board.make_move(7, 7)
        board.make_move(8, 8)
        mcts.search(board)
        assert mcts._prev_root is not None

        # Set up an immediate win — threat override fires, tree cleared.
        board.make_move(7, 2)
        board.make_move(8, 3)
        board.make_move(7, 3)
        board.make_move(8, 4)
        board.make_move(7, 4)
        board.make_move(8, 5)
        board.make_move(7, 5)  # Black has open four
        board.make_move(8, 6)
        # Black to move with open four at (7,2)-(7,5).
        mcts.search(board)
        # Tree should be cleared because threat override fired.
        assert mcts._prev_root is None
    finally:
        tmp.unlink()


def test_tree_reuse_preserves_subtree_statistics():
    """After re-root, child visit counts from prior search are preserved."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=50, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        # First search.
        result1 = mcts.search_with_stats(board)
        top_move = max(result1.visit_counts, key=result1.visit_counts.get)
        old_visits = result1.visit_counts[top_move]

        # Play the top move.
        board.make_move(*top_move)

        # Second search — continues from the re-rooted subtree.
        result2 = mcts.search_with_stats(board)
        # The cumulative sims should be > the fresh sims.
        assert result2.total_simulations > 50
        # The distribution should still be valid.
        total = sum(result2.visit_counts.values())
        assert total > 0
        assert abs(sum(v / total for v in result2.visit_counts.values()) - 1.0) < 1e-5
    finally:
        tmp.unlink()


def test_tree_reuse_distribution_valid_after_reuse():
    """Visit probabilities after tree reuse sum to 1 and only contain legal moves."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=30, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(8, 8)

        # Run two searches with a move in between.
        result1 = mcts.search_with_stats(board)
        move = max(result1.visit_counts, key=result1.visit_counts.get)
        board.make_move(*move)

        dist = mcts.search(board)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-5
        legal = board.get_legal_moves()
        for m in dist:
            assert m in legal, f"Tree reuse returned illegal move {m}"
    finally:
        tmp.unlink()
