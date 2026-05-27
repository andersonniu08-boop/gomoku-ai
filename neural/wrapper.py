"""Inference wrapper for GomokuNet — loads checkpoints and runs evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from engine.board import Board, Player
from engine.encoding import board_to_tensor, policy_to_move_probs
from engine.tactical import TacticalSolver
from neural.model import GomokuNet


class GomokuInferenceWrapper:
    """Loads a trained ``GomokuNet`` checkpoint and provides a clean
    ``evaluate(board)`` method that returns policy priors and a value
    estimate for MCTS consumption."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        num_res_blocks: int = 10,
        num_hidden_channels: int = 128,
        use_se: bool = True,
        use_attention: bool = True,
        use_pre_activation: bool = False,
        value_global_pool: bool = True,
        profiler: Optional[Profiler] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        from selfplay.profiler import Profiler as _Profiler
        self.profiler: _Profiler = profiler or _Profiler()
        self.profiler.disable()  # off by default; MCTS enables when needed

        self.model = GomokuNet(
            board_size=15,
            in_channels=3,
            num_res_blocks=num_res_blocks,
            num_hidden_channels=num_hidden_channels,
            use_se=use_se,
            use_attention=use_attention,
            use_pre_activation=use_pre_activation,
            value_global_pool=value_global_pool,
        ).to(self.device)

        checkpoint = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _BLOCK_BOOST_FACTOR = 5.0

    def evaluate(
        self, board: Board
    ) -> tuple[list[tuple[tuple[int, int], float]], float]:
        """Run a single inference step.

        Returns:
            move_probs: list of ``((row, col), prior_prob)`` for legal moves,
                        normalised to sum to 1.
            value:      scalar in [-1, 1] — expected outcome for
                        ``board.current_player``.
        """
        with self.profiler.measure("eval.board_to_tensor"):
            tensor = board_to_tensor(board).to(self.device)

        with self.profiler.measure("eval.model_forward"):
            with torch.no_grad():
                log_policy, value = self.model(tensor)

        with self.profiler.measure("eval.policy_to_move_probs"):
            move_probs = policy_to_move_probs(log_policy, board)
        value = float(value.item())

        return move_probs, value

    def evaluate_raw(
        self, board: Board
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference without ``torch.no_grad()``.

        Returns raw ``(log_policy, value)`` tensors with the computational
        graph intact — required for gradient-based explainability methods
        such as saliency maps and for activation capture via forward hooks.

        Callers must manage gradients and device placement on the result.
        """
        tensor = board_to_tensor(board).to(self.device)
        return self.model(tensor)

    def batch_evaluate(
        self, boards: list[Board]
    ) -> list[tuple[list[tuple[tuple[int, int], float]], float]]:
        """Evaluate multiple boards in one forward pass.

        Returns one ``(move_probs, value)`` per board, in the same order.
        """
        if not boards:
            return []

        with self.profiler.measure("eval.board_to_tensor"):
            tensors = torch.cat(
                [board_to_tensor(b) for b in boards], dim=0
            ).to(self.device)

        with self.profiler.measure("eval.model_forward"):
            with torch.no_grad():
                log_policy, value = self.model(tensors)

        results: list[tuple[list[tuple[tuple[int, int], float]], float]] = []
        for i, board in enumerate(boards):
            with self.profiler.measure("eval.policy_to_move_probs"):
                move_probs = policy_to_move_probs(log_policy[i : i + 1], board)
            results.append((move_probs, float(value[i].item())))

        return results

    def evaluate_with_threats(
        self,
        board: Board,
        *,
        hard_override: bool = True,
    ) -> tuple[list[tuple[tuple[int, int], float]], float, Optional[dict]]:
        """Like ``evaluate()``, but overrides priors for forcing moves.

        Returns:
            move_probs, value, threat_info   – *threat_info* is ``None`` when
            no override was applied, otherwise a dict with keys ``"overridden"``
            and ``"reason"``.
        """
        legal = board.get_legal_moves()

        if hard_override:
            analysis = TacticalSolver.analyze(board)
            forced = analysis.get_forced_distribution()
            if forced is not None:
                probs = [
                    (m, forced.get(m, 0.0))
                    for m in legal
                ]
                reason = "immediate_win" if analysis.winning_moves else \
                         "must_block" if analysis.must_block else \
                         "double_threat" if analysis.double_threat_moves else \
                         "forced_sequence"
                return probs, 1.0, {"overridden": True, "reason": reason}

        move_probs, value = self.evaluate(board)

        # Boost blocking moves using tactical analysis.
        analysis = TacticalSolver.analyze(board)
        urgent = analysis.must_block | analysis.urgent_blocks
        if urgent:
            move_probs = [
                (m, p * self._BLOCK_BOOST_FACTOR if m in urgent else p)
                for m, p in move_probs
            ]
            total = sum(p for _, p in move_probs)
            move_probs = [(m, p / total) for m, p in move_probs]
            return move_probs, value, {"overridden": True, "reason": "boosted_blocks"}

        return move_probs, value, None

    def batch_evaluate_with_threats(
        self,
        boards: list[Board],
        *,
        hard_override: bool = True,
    ) -> list[
        tuple[
            list[tuple[tuple[int, int], float]],
            float,
            Optional[dict],
        ]
    ]:
        """Batch ``evaluate_with_threats`` — one GPU call for all boards
        that need neural evaluation after immediate-win filtering.

        Returns one ``(move_probs, value, threat_info)`` per board.
        """
        if not boards:
            return []

        results: list[
            tuple[
                list[tuple[tuple[int, int], float]],
                float,
                Optional[dict],
            ]
        ] = []

        neural_indices: list[int] = []
        neural_boards: list[Board] = []
        per_board_urgent: list[set[tuple[int, int]] | None] = []

        for i, board in enumerate(boards):
            legal = board.get_legal_moves()

            if hard_override:
                analysis = TacticalSolver.analyze(board)
                forced = analysis.get_forced_distribution()
                if forced is not None:
                    probs = [
                        (m, forced.get(m, 0.0))
                        for m in legal
                    ]
                    reason = "immediate_win" if analysis.winning_moves else \
                             "must_block" if analysis.must_block else \
                             "double_threat" if analysis.double_threat_moves else \
                             "forced_sequence"
                    results.append(
                        (probs, 1.0, {"overridden": True, "reason": reason})
                    )
                    per_board_urgent.append(None)
                    continue

                urgent = analysis.must_block | analysis.urgent_blocks
            else:
                urgent = set()

            neural_indices.append(i)
            neural_boards.append(board)
            results.append(([], 0.0, None))  # placeholder
            per_board_urgent.append(urgent)

        if neural_boards:
            neural_results = self.batch_evaluate(neural_boards)
        else:
            neural_results = []

        for j, (i, board) in enumerate(zip(neural_indices, neural_boards)):
            move_probs, value = neural_results[j]
            urgent = per_board_urgent[i]
            if urgent:
                move_probs = [
                    (m, p * self._BLOCK_BOOST_FACTOR if m in urgent else p)
                    for m, p in move_probs
                ]
                total = sum(p for _, p in move_probs)
                move_probs = [(m, p / total) for m, p in move_probs]
                results[i] = (
                    move_probs,
                    value,
                    {"overridden": True, "reason": "boosted_blocks"},
                )
            else:
                results[i] = (move_probs, value, None)

        return results
