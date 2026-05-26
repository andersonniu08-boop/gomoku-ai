"""AlphaZero-style self-play game generator.

Produces (state, policy_target, value_target) training triples by running
MCTS-guided games where two copies of the same AI play each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from engine.board import Board, Player
from engine.encoding import board_to_tensor
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS

# ---------------------------------------------------------------------------
# Training example
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainingExample:
    """One labelled position for neural-network training.

    Attributes:
        state:   Encoded board tensor, shape ``(3, 15, 15)``, from the
                 perspective of the player who is about to move.
        policy:  Target policy (MCTS visit proportions), shape ``(225,)``.
        value:   Final game outcome from *this* player's perspective
                 (+1 = win, -1 = loss, 0 = draw).
    """

    state: torch.Tensor
    policy: torch.Tensor
    value: float


# ---------------------------------------------------------------------------
# D₄ symmetry transforms  (rotations + reflections of a square grid)
# ---------------------------------------------------------------------------

# Each entry is a pair of callables: (state_transform, policy_transform).
# The state transform operates on a ``(3, 15, 15)`` tensor (dims 1,2 are
# spatial).  The policy transform operates on a ``(15, 15)`` tensor (the
# policy reshaped from its 225-element flat form).  Value targets are
# unchanged by any symmetry.


def _make_symmetries() -> list[
    tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]
]:
    """Build the 8 D₄ transforms as (state_fn, policy_fn) pairs."""
    syms: list[
        tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ]
    ] = []

    # Identity
    syms.append((lambda t: t, lambda p: p))

    # Rotations (k = 1, 2, 3 quarters clockwise)
    for k in (1, 2, 3):
        syms.append((
            lambda t, _k=k: torch.rot90(t, _k, dims=(1, 2)),
            lambda p, _k=k: torch.rot90(p, _k, dims=(0, 1)),
        ))

    # Horizontal flip, then each rotation
    for k in (0, 1, 2, 3):
        syms.append((
            lambda t, _k=k: torch.rot90(torch.flip(t, dims=(2,)), _k, dims=(1, 2)),
            lambda p, _k=k: torch.rot90(torch.flip(p, dims=(1,)), _k, dims=(0, 1)),
        ))

    return syms


SYMMETRIES = _make_symmetries()


def augment_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Apply all 8 D₄ symmetries to every example, returning 8× as many."""
    augmented: list[TrainingExample] = []
    for ex in examples:
        policy_grid = ex.policy.view(15, 15)
        for state_fn, policy_fn in SYMMETRIES:
            augmented.append(
                TrainingExample(
                    state=state_fn(ex.state).clone(),
                    policy=policy_fn(policy_grid).reshape(-1).clone(),
                    value=ex.value,
                )
            )
    return augmented


# ---------------------------------------------------------------------------
# Self-play game runner
# ---------------------------------------------------------------------------


class SelfPlayGame:
    """Play one game of the current AI against itself, collecting training data.

    Parameters:
        wrapper:              Neural-network inference wrapper shared by both
                              sides (the model is stateless, so one is enough).
        num_simulations:      MCTS iterations per move.
        c_puct:               PUCT exploration constant.
        temperature:          Visit-count exponent for stochastic move selection
                              in the early phase (1.0 = proportional sampling).
        temperature_threshold:  Number of moves after which temperature is
                                annealed to 0 (deterministic / greedy).
        threat_override:      When True, MCTS short-circuits on forced
                              wins / blocks.
        augment:              If True, apply D₄ symmetry augmentation to the
                              output examples (8× data per game).
        dirichlet_alpha:      Root Dirichlet noise concentration (default 0.03).
                              ``None`` disables noise.
        dirichlet_epsilon:    Noise mixing proportion (default 0.25).
    """

    def __init__(
        self,
        wrapper: GomokuInferenceWrapper,
        *,
        num_simulations: int = 400,
        c_puct: float = 2.5,
        temperature: float = 1.0,
        temperature_threshold: int = 15,
        threat_override: bool = True,
        augment: bool = True,
        dirichlet_alpha: float | None = 0.03,
        dirichlet_epsilon: float = 0.25,
    ):
        self.wrapper = wrapper
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.threat_override = threat_override
        self.augment = augment
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self) -> list[TrainingExample]:
        """Run one self-play game and return labelled training examples."""
        board = Board()
        mcts = MCTS(
            self.wrapper,
            c_puct=self.c_puct,
            num_simulations=self.num_simulations,
            threat_override=self.threat_override,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )

        # Phase 1 — collect raw (state, policy, player) per turn
        raw: list[tuple[torch.Tensor, torch.Tensor, Player]] = []

        while not board.is_terminal():
            move_num = len(board.move_history)
            temp = self._temperature_for_move(move_num)

            visit_probs = mcts.search(board)

            # Encode state *before* the move, from the current player's view.
            # board_to_tensor produces (1, 3, 15, 15); we squeeze the batch
            # dim so each example is stored as (3, 15, 15).
            state = board_to_tensor(board).squeeze(0)

            # Build a flat 225-element policy target from the visit counts.
            policy = _visit_probs_to_tensor(visit_probs, board_size=15)

            # Record whose turn it is so we can assign the correct value
            # sign once the game concludes.
            raw.append((state, policy, board.current_player))

            move = mcts.select_move(board, temperature=temp, visit_probs=visit_probs)
            board.make_move(*move)

        # Phase 2 — convert to (state, policy, value_target)
        winner = board.check_win()
        examples = _assign_values(raw, winner)

        # Phase 3 — optional symmetry augmentation
        if self.augment:
            examples = augment_examples(examples)

        return examples

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _temperature_for_move(self, move_num: int) -> float:
        """Return the temperature to use for a given move index.

        Early moves use positive temperature (stochastic sampling) to
        encourage exploration.  Late moves use temperature 0 (argmax) so
        the AI plays its strongest line in critical positions.
        """
        if move_num < self.temperature_threshold:
            return self.temperature
        return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _visit_probs_to_tensor(
    visit_probs: dict[tuple[int, int], float], *, board_size: int = 15
) -> torch.Tensor:
    """Convert a ``{(r,c): prob}`` dict into a flat ``(board_size²,)`` FloatTensor."""
    policy = torch.zeros(board_size * board_size, dtype=torch.float32)
    for (r, c), prob in visit_probs.items():
        policy[r * board_size + c] = prob
    return policy


def _assign_values(
    raw: list[tuple[torch.Tensor, torch.Tensor, Player]],
    winner: Player | None,
) -> list[TrainingExample]:
    """Convert per-turn records into labelled examples.

    Value is +1 for the winner, -1 for the loser.  Draws produce 0 for
    both sides.
    """
    examples: list[TrainingExample] = []
    for state, policy, player in raw:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        examples.append(TrainingExample(state=state, policy=policy, value=value))
    return examples
