"""Tests for selfplay.selfplay and selfplay.replay_buffer."""

import tempfile
from pathlib import Path

import torch

from engine.board import Board, Player
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS
from selfplay.replay_buffer import ReplayBuffer
from selfplay.selfplay import (
    SYMMETRIES,
    SelfPlayGame,
    TrainingExample,
    _assign_values,
    _visit_probs_to_tensor,
    augment_examples,
)



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


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


def test_training_example_construction():
    state = torch.zeros(3, 15, 15)
    policy = torch.zeros(225)
    ex = TrainingExample(state=state, policy=policy, value=1.0)
    assert ex.state.shape == (3, 15, 15)
    assert ex.policy.shape == (225,)
    assert ex.value == 1.0


# ---------------------------------------------------------------------------
# Symmetry transforms
# ---------------------------------------------------------------------------


def test_symmetries_count():
    assert len(SYMMETRIES) == 8


def test_symmetry_identity_preserves_state():
    state = torch.randn(3, 15, 15)
    state_fn, policy_fn = SYMMETRIES[0]
    assert torch.allclose(state_fn(state), state)


def test_symmetry_rot90_then_rot270_is_identity():
    state = torch.randn(3, 15, 15)
    rot90_s, rot90_p = SYMMETRIES[1]  # rot90
    rot270_s, rot270_p = SYMMETRIES[3]  # rot270
    result = rot270_s(rot90_s(state))
    assert torch.allclose(result, state)


def test_symmetry_state_shape_preserved():
    state = torch.randn(3, 15, 15)
    for state_fn, _ in SYMMETRIES:
        assert state_fn(state).shape == (3, 15, 15)


def test_symmetry_policy_shape_preserved():
    policy_grid = torch.randn(15, 15)
    for _, policy_fn in SYMMETRIES:
        assert policy_fn(policy_grid).shape == (15, 15)


def test_symmetry_flip_twice_is_identity():
    state = torch.randn(3, 15, 15)
    flip_s, flip_p = SYMMETRIES[4]  # horizontal flip
    result = flip_s(flip_s(state))
    assert torch.allclose(result, state)


# ---------------------------------------------------------------------------
# augment_examples
# ---------------------------------------------------------------------------


def test_augment_examples_8x_expansion():
    state = torch.randn(3, 15, 15)
    policy = torch.zeros(225)
    policy[7 * 15 + 7] = 1.0  # all mass on center
    ex = TrainingExample(state=state, policy=policy, value=1.0)
    augmented = augment_examples([ex])
    assert len(augmented) == 8


def test_augment_examples_preserves_value():
    state = torch.randn(3, 15, 15)
    policy = torch.zeros(225)
    policy[0] = 1.0
    ex = TrainingExample(state=state, policy=policy, value=-1.0)
    augmented = augment_examples([ex])
    for a in augmented:
        assert a.value == -1.0


def test_augment_multiple_examples():
    examples = [
        TrainingExample(
            state=torch.randn(3, 15, 15),
            policy=torch.zeros(225),
            value=1.0,
        )
        for _ in range(3)
    ]
    augmented = augment_examples(examples)
    assert len(augmented) == 24  # 3 * 8


# ---------------------------------------------------------------------------
# _visit_probs_to_tensor
# ---------------------------------------------------------------------------


def test_visit_probs_to_tensor_empty():
    result = _visit_probs_to_tensor({})
    assert result.shape == (225,)
    assert torch.all(result == 0.0)


def test_visit_probs_to_tensor_maps_correctly():
    probs = {(0, 0): 0.3, (7, 7): 0.7}
    tensor = _visit_probs_to_tensor(probs)
    assert tensor[0] == 0.3
    assert tensor[7 * 15 + 7] == 0.7
    assert tensor.sum() == 1.0


# ---------------------------------------------------------------------------
# _assign_values
# ---------------------------------------------------------------------------


def test_assign_values_winner_gets_positive():
    state = torch.zeros(3, 15, 15)
    policy = torch.zeros(225)
    raw = [(state, policy, Player.BLACK)]
    examples = _assign_values(raw, Player.BLACK)
    assert examples[0].value == 1.0


def test_assign_values_loser_gets_negative():
    state = torch.zeros(3, 15, 15)
    policy = torch.zeros(225)
    raw = [(state, policy, Player.WHITE)]
    examples = _assign_values(raw, Player.BLACK)
    assert examples[0].value == -1.0


def test_assign_values_draw_gets_zero():
    state = torch.zeros(3, 15, 15)
    policy = torch.zeros(225)
    raw = [(state, policy, Player.BLACK)]
    examples = _assign_values(raw, None)
    assert examples[0].value == 0.0


def test_assign_values_mixed_sides():
    state = torch.zeros(3, 15, 15)
    policy = torch.zeros(225)
    raw = [
        (state, policy, Player.BLACK),
        (state, policy, Player.WHITE),
        (state, policy, Player.BLACK),
    ]
    examples = _assign_values(raw, Player.BLACK)
    assert examples[0].value == 1.0  # Black won → Black's move is +1
    assert examples[1].value == -1.0  # Black won → White's move is -1
    assert examples[2].value == 1.0


# ---------------------------------------------------------------------------
# SelfPlayGame
# ---------------------------------------------------------------------------


def test_selfplay_produces_examples():
    wrapper, tmp = _make_wrapper()
    try:
        game = SelfPlayGame(
            wrapper,
            num_simulations=10,
            temperature=1.0,
            temperature_threshold=0,
            augment=False,
        )
        examples = game.play()
        assert len(examples) > 0
        for ex in examples:
            assert ex.state.shape == (3, 15, 15)
            assert ex.policy.shape == (225,)
            assert abs(ex.policy.sum().item() - 1.0) < 1e-5
            assert -1.0 <= ex.value <= 1.0
    finally:
        tmp.unlink()


def test_selfplay_with_augmentation():
    wrapper, tmp = _make_wrapper()
    try:
        game = SelfPlayGame(
            wrapper,
            num_simulations=10,
            temperature=1.0,
            temperature_threshold=0,
            augment=True,
        )
        examples = game.play()
        # Raw moves × 8 symmetries
        assert len(examples) >= 8
        assert len(examples) % 8 == 0
    finally:
        tmp.unlink()


def test_selfplay_temperature_annealing():
    wrapper, tmp = _make_wrapper()
    try:
        game = SelfPlayGame(wrapper, temperature=1.0, temperature_threshold=5)
        # First 5 moves should use temperature 1.0
        for i in range(5):
            assert game._temperature_for_move(i) == 1.0
        # Move 5 onward should use 0.0
        assert game._temperature_for_move(5) == 0.0
        assert game._temperature_for_move(100) == 0.0
    finally:
        tmp.unlink()


def test_selfplay_greedy_game():
    """Game with temperature=0 from the start should still complete."""
    wrapper, tmp = _make_wrapper()
    try:
        game = SelfPlayGame(
            wrapper,
            num_simulations=10,
            temperature=1.0,
            temperature_threshold=0,
            augment=False,
        )
        examples = game.play()
        assert len(examples) > 0
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


def _make_example(value=1.0):
    return TrainingExample(
        state=torch.randn(3, 15, 15),
        policy=torch.zeros(225),
        value=value,
    )


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(max_size=100)
    examples = [_make_example() for _ in range(10)]
    buf.add_examples(examples)
    assert len(buf) == 10


def test_replay_buffer_sample():
    buf = ReplayBuffer(max_size=100)
    examples = [_make_example(i) for i in range(50)]
    buf.add_examples(examples)
    batch = buf.sample(8)
    assert len(batch) == 8
    assert all(isinstance(ex, TrainingExample) for ex in batch)


def test_replay_buffer_sample_more_than_contents():
    buf = ReplayBuffer(max_size=100)
    buf.add_examples([_make_example() for _ in range(5)])
    batch = buf.sample(100)
    assert len(batch) == 5


def test_replay_buffer_max_size_eviction():
    buf = ReplayBuffer(max_size=10)
    buf.add_examples([_make_example(i) for i in range(15)])
    assert len(buf) == 10


def test_replay_buffer_get_batch_shapes():
    buf = ReplayBuffer(max_size=100)
    buf.add_examples([_make_example() for _ in range(20)])
    states, policies, values = buf.get_batch(16)
    assert states.shape == (16, 3, 15, 15)
    assert policies.shape == (16, 225)
    assert values.shape == (16, 1)


def test_replay_buffer_get_batch_empty():
    buf = ReplayBuffer(max_size=100)
    states, policies, values = buf.get_batch(8)
    assert states.shape == (0, 3, 15, 15)
    assert policies.shape == (0, 225)
    assert values.shape == (0, 1)


def test_replay_buffer_clear():
    buf = ReplayBuffer(max_size=100)
    buf.add_examples([_make_example() for _ in range(30)])
    buf.clear()
    assert len(buf) == 0


def test_replay_buffer_iter():
    buf = ReplayBuffer(max_size=100)
    examples = [_make_example() for _ in range(5)]
    buf.add_examples(examples)
    items = list(buf)
    assert len(items) == 5


def test_replay_buffer_state_dict_roundtrip():
    buf = ReplayBuffer(max_size=50)
    buf.add_examples([_make_example(i) for i in range(10)])
    restored = ReplayBuffer.from_state_dict(buf.state_dict())
    assert len(restored) == 10
    assert restored.max_size == 50


# ---------------------------------------------------------------------------
# Integration: small self-play game → replay buffer → training batch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression: no double-search during self-play
# ---------------------------------------------------------------------------


def test_select_move_uses_provided_visit_probs():
    """select_move must use provided visit_probs instead of calling search()."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=False)
        board = Board()
        board.make_move(7, 7)
        board.make_move(0, 0)

        legal = board.get_legal_moves()
        fake_probs = {m: 1.0 / len(legal) for m in legal}

        move = mcts.select_move(board, temperature=0.0, visit_probs=fake_probs)
        assert move in legal
    finally:
        tmp.unlink()


def test_select_move_fallback_calls_search():
    """select_move without visit_probs should still call search() (backward compat)."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=False)
        board = Board()
        # Use a board with at least one move so search returns a distribution.
        board.make_move(7, 7)

        # Should not crash and return a valid move.
        move = mcts.select_move(board, temperature=0.0)
        assert isinstance(move, tuple)
        assert len(move) == 2
    finally:
        tmp.unlink()


def test_selfplay_no_double_search():
    """SelfPlayGame.play() must not re-run MCTS via select_move.

    Regression test: play() was calling both search() and select_move(),
    and select_move() called search() again internally.  The fix passes
    visit_probs from search() into select_move() to avoid the duplicate.

    We verify by counting how many times MCTS.search() is called during
    a game and asserting it equals the number of moves (one per move).
    """
    import unittest.mock as mock

    wrapper, tmp = _make_wrapper()
    try:
        original_search = MCTS.search

        call_count = 0

        def counting_search(self, board):
            nonlocal call_count
            call_count += 1
            return original_search(self, board)

        with mock.patch.object(MCTS, "search", counting_search):
            game = SelfPlayGame(
                wrapper,
                num_simulations=10,
                temperature=1.0,
                temperature_threshold=0,
                opening_moves=0,
                augment=False,
            )
            examples = game.play()

        # One search per move, no extras from select_move.
        num_moves = len(examples)
        assert call_count == num_moves, (
            f"MCTS.search() called {call_count}x for {num_moves} moves. "
            f"Expected exactly 1 call per move ({num_moves}). "
            f"This means the double-search bug has regressed."
        )
    finally:
        tmp.unlink()
