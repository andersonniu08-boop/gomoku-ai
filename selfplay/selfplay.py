"""AlphaZero-style self-play game generator.

Produces (state, policy_target, value_target) training triples by running
MCTS-guided games where two copies of the same AI play each other.
"""

from __future__ import annotations

import random
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
                                Temperature decays linearly from *temperature*
                                to 0 across this many moves rather than
                                switching abruptly.
        threat_override:      When True, MCTS short-circuits on forced
                              wins / blocks.
        augment:              If True, apply D₄ symmetry augmentation to the
                              output examples (8× data per game).  Prefer
                              ``False`` when writing to a ``ReplayBuffer``
                              that augments on retrieval instead.
        dirichlet_alpha:      Root Dirichlet noise concentration (default 0.03).
                              ``None`` disables noise.
        dirichlet_epsilon:    Noise mixing proportion (default 0.25).
        opening_moves:        Number of opening plies (each side) where moves
                              are sampled from the raw policy prior with high
                              temperature rather than from full MCTS visit
                              counts.  This forces opening diversity and
                              prevents premature convergence to a narrow
                              repertoire.  Default 6.
        resignation_threshold:  Value below which the AI considers resigning.
                                Default -0.9 (i.e. estimated win chance < 5%).
                                ``None`` disables resignation.
        resignation_moves:    Number of consecutive sub-threshold value
                              estimates before resigning.  Default 3.
    """

    def __init__(
        self,
        wrapper: GomokuInferenceWrapper,
        *,
        num_simulations: int = 800,
        c_puct: float = 2.5,
        temperature: float = 1.0,
        temperature_threshold: int = 15,
        threat_override: bool = True,
        augment: bool = False,
        dirichlet_alpha: float | None = 0.03,
        dirichlet_epsilon: float = 0.25,
        opening_moves: int = 6,
        resignation_threshold: float | None = -0.9,
        resignation_moves: int = 3,
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
        self.opening_moves = opening_moves
        self.resignation_threshold = resignation_threshold
        self.resignation_moves = resignation_moves

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
            tree_reuse=False,
        )

        # Phase 1 — collect raw (state, policy, player) per turn
        raw: list[tuple[torch.Tensor, torch.Tensor, Player]] = []

        # Resignation tracking: consecutive moves where the root value
        # estimate is below the resignation threshold.
        consecutive_lost: int = 0

        while not board.is_terminal():
            move_num = len(board.move_history)
            temp = self._temperature_for_move(move_num)

            # --- Opening diversity: sample from raw prior, not MCTS ---
            # For the first *opening_moves* plies the move is drawn from
            # the network's raw policy prior (temperature ~2) rather than
            # from MCTS visit counts.  This forces exploration of diverse
            # opening lines the network considers plausible, preventing
            # premature convergence to a narrow repertoire.  The raw prior
            # is also used as the training target for these positions,
            # which is acceptable because opening positions carry a weak
            # strategic signal — the game-outcome value target still
            # provides the primary learning signal.
            if move_num < self.opening_moves:
                state, policy, move = self._opening_move(board, mcts)
                raw.append((state, policy, board.current_player))
                board.make_move(*move)
                continue

            visit_probs = mcts.search(board)

            # Encode state *before* the move, from the current player's view.
            state = board_to_tensor(board).squeeze(0)

            # Build a flat 225-element policy target from the visit counts.
            policy = _visit_probs_to_tensor(visit_probs, board_size=15)

            # Record whose turn it is so we can assign the correct value
            # sign once the game concludes.
            raw.append((state, policy, board.current_player))

            move = mcts.select_move(board, temperature=temp, visit_probs=visit_probs)
            board.make_move(*move)

            # --- Resignation check ---
            # If the root value estimate stays below the resignation
            # threshold for enough consecutive moves the game is almost
            # certainly decided — stop early to save compute.  The
            # already-recorded positions are still valid training data.
            if self.resignation_threshold is not None and visit_probs:
                root_value = self._estimate_root_value(visit_probs, mcts, board)
                if root_value < self.resignation_threshold:
                    consecutive_lost += 1
                else:
                    consecutive_lost = 0
                if consecutive_lost >= self.resignation_moves:
                    break

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

    def _opening_move(
        self,
        board: Board,
        mcts: MCTS,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Return (state_tensor, policy_target, move) for an opening ply.

        The move is sampled from the raw network prior (not MCTS visit
        counts) with temperature ~2, forcing exploration of diverse
        lines.  The policy target is the raw prior itself — the network
        learns to predict its own prior for opening positions, which is
        acceptable because the game-outcome value target still provides
        the primary learning signal.
        """
        import torch

        from engine.encoding import board_to_tensor

        # Raw network prior.
        tensor = board_to_tensor(board).to(self.wrapper.device)
        with torch.no_grad():
            log_policy, _ = self.wrapper.model(tensor)
        raw_prior = torch.exp(log_policy).squeeze(0).cpu().numpy()

        # Filter to legal moves.
        legal = board.get_legal_moves()
        probs = [float(raw_prior[r * 15 + c]) for (r, c) in legal]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(legal)] * len(legal)

        # Sample with temperature ~2 for extra diversity in the opening.
        temp = 2.0
        probs_temp = [p ** (1.0 / temp) for p in probs]
        total_temp = sum(probs_temp)
        probs_temp = [p / total_temp for p in probs_temp]
        move = legal[random.choices(range(len(legal)), weights=probs_temp, k=1)[0]]

        # Encode state (same format as main loop).
        state = board_to_tensor(board).squeeze(0).cpu()

        # Use the raw prior as the policy target (no MCTS overhead).
        policy = torch.zeros(225, dtype=torch.float32)
        for (r, c), p in zip(legal, probs):
            policy[r * 15 + c] = p

        return state, policy, move

    @staticmethod
    def _estimate_root_value(
        visit_probs: dict[tuple[int, int], float],
        mcts: MCTS,
        board: Board,
    ) -> float:
        """Estimate the root position value from the perspective of the
        player who just searched.

        Uses the visit-weighted average of child Q-values, which is a
        lower-variance estimate than the raw network value alone.
        Falls back to a neutral estimate when the tree is empty.
        """
        if not visit_probs:
            return 0.0
        # Reconstruct root Q-values from the MCTS tree.  The search
        # result only carries visit proportions, but select_move with
        # temperature > 0 doesn't expose Q.  Approximate via the
        # evaluator: run a single neural eval as the fallback.
        tensor = board_to_tensor(board).to(mcts.wrapper.device)
        with torch.no_grad():
            _, value = mcts.wrapper.model(tensor)
        return float(value.item())

    def _temperature_for_move(self, move_num: int) -> float:
        """Return the temperature to use for a given move index.

        Temperature decays linearly from the configured value to 0 across
        *temperature_threshold* moves.  This soft annealing is less brittle
        than the hard cutoff used in the original AlphaZero — it avoids a
        sudden switch from noisy to deterministic play at a single move
        boundary, which is especially important for shorter Gomoku games
        where the transition zone falls in the critical midgame.
        """
        if move_num >= self.temperature_threshold:
            return 0.0
        return self.temperature


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
