"""Tests for Zobrist hashing (engine/board.py) and BoundedEvalCache (neural/wrapper.py)."""

import tempfile
from pathlib import Path

import torch

from engine.board import Board, Player
from neural.wrapper import BoundedEvalCache, GomokuInferenceWrapper
from neural.model import GomokuNet


# ---------------------------------------------------------------------------
# Zobrist hash tests
# ---------------------------------------------------------------------------


def test_zobrist_initial_zero():
    b = Board()
    assert b.zobrist_key == 0


def test_zobrist_changes_on_move():
    b = Board()
    key0 = b.zobrist_key
    b.make_move(7, 7)
    assert b.zobrist_key != key0


def test_zobrist_undo_restores():
    b = Board()
    b.make_move(7, 7)
    b.make_move(8, 8)
    key = b.zobrist_key
    b.undo_move()
    assert b.zobrist_key != key
    b.undo_move()
    assert b.zobrist_key == 0


def test_zobrist_same_sequence_same_key():
    b1 = Board()
    b1.make_move(7, 7)
    b1.make_move(0, 0)
    b1.make_move(14, 14)
    key1 = b1.zobrist_key

    b2 = Board()
    b2.make_move(7, 7)
    b2.make_move(0, 0)
    b2.make_move(14, 14)
    assert b2.zobrist_key == key1


def test_zobrist_different_sequence_different_key():
    b1 = Board()
    b1.make_move(7, 7)
    b1.make_move(0, 0)
    key1 = b1.zobrist_key

    b2 = Board()
    b2.make_move(0, 0)
    b2.make_move(7, 7)
    assert b2.zobrist_key != key1


def test_zobrist_different_position_different_key():
    b1 = Board()
    b1.make_move(0, 0)
    key1 = b1.zobrist_key

    b2 = Board()
    b2.make_move(1, 1)
    assert b2.zobrist_key != key1


def test_zobrist_copy_preserves_key():
    b = Board()
    b.make_move(7, 7)
    b.make_move(3, 3)
    c = b.copy()
    assert c.zobrist_key == b.zobrist_key
    assert c.current_player == b.current_player


def test_zobrist_copy_is_independent():
    b = Board()
    b.make_move(7, 7)
    c = b.copy()
    c.make_move(0, 0)
    assert c.zobrist_key != b.zobrist_key


def test_zobrist_same_position_different_player_different_key():
    b = Board()
    b.make_move(7, 7)  # Black
    key_black = b.zobrist_key
    b.make_move(0, 0)  # White
    b.undo_move()
    b.undo_move()
    assert b.zobrist_key == 0

    b.make_move(7, 7)  # Black again
    assert b.zobrist_key == key_black


# ---------------------------------------------------------------------------
# BoundedEvalCache unit tests (no model needed)
# ---------------------------------------------------------------------------


def test_cache_size_zero_disabled():
    cache = BoundedEvalCache(max_size=0)
    b = Board()
    b.make_move(7, 7)
    assert cache.get(b) is None
    cache.put(b, [((0, 0), 1.0)], 0.5)
    assert cache.get(b) is None
    assert len(cache) == 0


def test_cache_basic_get_put():
    cache = BoundedEvalCache(max_size=10)
    b = Board()
    b.make_move(7, 7)

    probs = [((0, 0), 1.0)]
    cache.put(b, probs, 0.5)
    result = cache.get(b)
    assert result is not None
    assert result == (probs, 0.5)
    assert cache.hits == 1


def test_cache_key_distinguishes_player():
    cache = BoundedEvalCache(max_size=10)

    b_black = Board()
    b_black.make_move(7, 7)  # Black moves, now White's turn
    # Read the key for White's turn
    key_white = (b_black.zobrist_key, int(b_black.current_player))
    cache.put(b_black, [((8, 8), 1.0)], 0.3)

    b_black2 = Board()
    b_black2.make_move(7, 7)  # Same board and turn
    assert cache.get(b_black2) is not None
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_key_player_matters():
    cache = BoundedEvalCache(max_size=10)
    b = Board()
    key_black = (b.zobrist_key, int(b.current_player))
    cache.put(b, [((7, 7), 1.0)], 0.2)

    b.make_move(7, 7)  # Now White's turn, different key
    assert cache.get(b) is None  # miss
    assert cache.misses == 1


def test_cache_fifo_eviction():
    cache = BoundedEvalCache(max_size=3)

    for i in range(5):
        b = Board()
        b.make_move(i, 0)
        cache.put(b, [((0, 0), 1.0)], 0.0)

    assert cache.hits == 0
    assert len(cache) == 3

    # First entries (0, 0) and (1, 0) should be evicted
    b0 = Board()
    b0.make_move(0, 0)
    assert cache.get(b0) is None  # evicted

    b1 = Board()
    b1.make_move(1, 0)
    assert cache.get(b1) is None  # evicted

    # Later entries should still be in cache
    b4 = Board()
    b4.make_move(4, 0)
    assert cache.get(b4) is not None  # still present


def test_cache_clear():
    cache = BoundedEvalCache(max_size=5)
    b = Board()
    cache.put(b, [((7, 7), 1.0)], 0.5)
    cache.get(b)  # hit
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_lru_move_to_end():
    cache = BoundedEvalCache(max_size=2)

    b0 = Board()
    b0.make_move(0, 0)
    b1 = Board()
    b1.make_move(1, 1)
    b2 = Board()
    b2.make_move(2, 2)

    cache.put(b0, [((0, 1), 1.0)], 0.0)
    cache.put(b1, [((1, 2), 1.0)], 0.0)
    cache.get(b0)  # refresh b0 — moves to end
    cache.put(b2, [((2, 3), 1.0)], 0.0)  # evicts b1

    assert cache.get(b0) is not None  # still there
    assert cache.get(b1) is None  # evicted
    assert cache.get(b2) is not None  # newest


# ---------------------------------------------------------------------------
# Cache integration tests (with model)
# ---------------------------------------------------------------------------


def _make_wrapper(cache_size: int = 10, num_res_blocks: int = 5,
                  num_hidden_channels: int = 64) -> GomokuInferenceWrapper:
    model = GomokuNet(
        board_size=15, in_channels=3,
        num_res_blocks=num_res_blocks,
        num_hidden_channels=num_hidden_channels,
        use_se=False, use_attention=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    wrapper = GomokuInferenceWrapper(
        tmp_path, device="cpu",
        num_res_blocks=num_res_blocks,
        num_hidden_channels=num_hidden_channels,
        use_se=False, use_attention=False,
        cache_size=cache_size,
    )
    return wrapper


def test_cache_hit_on_repeated_evaluate():
    wrapper = _make_wrapper(cache_size=10)
    b = Board()
    b.make_move(7, 7)
    b.make_move(8, 8)

    result1 = wrapper.evaluate(b)
    hits_before = wrapper.eval_cache.hits

    result2 = wrapper.evaluate(b)
    assert wrapper.eval_cache.hits == hits_before + 1
    assert result1 == result2


def test_cache_equivalence():
    """Verify the cached result matches the direct evaluation result."""
    wrapper = _make_wrapper(cache_size=10)
    b = Board()
    b.make_move(7, 7)
    b.make_move(8, 8)

    direct_result = wrapper.evaluate(b)

    cached = wrapper.eval_cache.get(b)
    assert cached is not None
    assert cached == direct_result


def test_cache_equivalence_after_make_undo():
    """Same board state reached via make/undo should produce cache hit."""
    wrapper = _make_wrapper(cache_size=10)
    b = Board()
    b.make_move(7, 7)
    b.make_move(8, 8)

    result1 = wrapper.evaluate(b)

    b.make_move(9, 9)
    b.undo_move()

    result2 = wrapper.evaluate(b)
    assert wrapper.eval_cache.hits >= 1
    assert result1 == result2


def test_cache_different_boards_different_keys():
    wrapper = _make_wrapper(cache_size=10)
    b1 = Board()
    b1.make_move(7, 7)
    b2 = Board()
    b2.make_move(0, 0)

    wrapper.evaluate(b1)
    misses_before = wrapper.eval_cache.misses

    wrapper.evaluate(b2)
    assert wrapper.eval_cache.misses == misses_before + 1


def test_cache_size_zero_never_hits():
    wrapper = _make_wrapper(cache_size=0)
    b = Board()
    b.make_move(7, 7)

    wrapper.evaluate(b)
    hits_before = wrapper.eval_cache.hits

    wrapper.evaluate(b)
    assert wrapper.eval_cache.hits == hits_before
    assert len(wrapper.eval_cache) == 0


def test_cache_batch_evaluate_uses_cache():
    wrapper = _make_wrapper(cache_size=10)
    b1 = Board()
    b1.make_move(7, 7)
    b2 = Board()
    b2.make_move(8, 8)

    # Pre-populate cache for b1
    wrapper.evaluate(b1)
    hits_before = wrapper.eval_cache.hits

    results = wrapper.batch_evaluate([b1, b2])
    assert wrapper.eval_cache.hits >= hits_before + 1  # b1 should be cache hit

    # Verify results are correct
    assert results[0] is not None
    assert results[1] is not None
    total = sum(p for _, p in results[0][0])
    assert abs(total - 1.0) < 1e-5


def test_cache_batch_evaluate_caches_new():
    wrapper = _make_wrapper(cache_size=10)
    b = Board()
    b.make_move(7, 7)

    wrapper.batch_evaluate([b])

    cached = wrapper.eval_cache.get(b)
    assert cached is not None


def test_cache_respects_max_size():
    wrapper = _make_wrapper(cache_size=3)

    for i in range(10):
        b = Board()
        b.make_move(i % 15, i // 15)
        wrapper.evaluate(b)

    assert len(wrapper.eval_cache) <= 3


def test_cache_overwrite_updates():
    wrapper = _make_wrapper(cache_size=10)
    b = Board()
    b.make_move(7, 7)

    result1 = wrapper.evaluate(b)
    # Evaluate again — should be a cache hit (same result)
    result2 = wrapper.evaluate(b)
    assert result1 == result2
