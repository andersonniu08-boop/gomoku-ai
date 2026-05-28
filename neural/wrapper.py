"""Inference wrapper for GomokuNet — loads checkpoints and runs evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from engine.board import Board, Player
from engine.encoding import board_to_tensor, policy_to_move_probs
from engine.threats import ThreatDetector, ThreatType
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
        ).to(self.device)

        checkpoint = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Enable TF32 tensor cores for ~2× matmul throughput on Ampere+.
        torch.set_float32_matmul_precision("high")

        # Keep a reference to the uncompiled model for hook-based
        # introspection (explainability tools register forward hooks
        # on res_blocks, which a torch.compiled model would bypass).
        self._raw_model = self.model

    # ------------------------------------------------------------------
    # Shared threat helpers used by evaluate_with_threats and
    # batch_evaluate_with_threats.
    # ------------------------------------------------------------------

    @staticmethod
    def _winning_moves(
        threats: list, legal_set: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Return legal moves that complete a winning threat.

        Handles FIVE, OPEN_FOUR, and CLOSED_FOUR.  For closed fours:
        * Contiguous (XXXX_): the one open end is a winning move.
        * Split (XX_XX, XXX_X): only the gap creates five-in-a-row.
        """
        moves: set[tuple[int, int]] = set()
        for t in threats:
            if t.threat_type == ThreatType.FIVE:
                if t.gap is not None:
                    moves.add(t.gap)
                for end in t.open_ends:
                    moves.add(end)
            elif t.threat_type == ThreatType.OPEN_FOUR:
                for end in t.open_ends:
                    moves.add(end)
            elif t.threat_type == ThreatType.CLOSED_FOUR:
                if t.gap is not None:
                    # Split closed four — only the gap wins.
                    moves.add(t.gap)
                else:
                    # Contiguous closed four — the one open end wins.
                    for end in t.open_ends:
                        moves.add(end)
        return moves & legal_set

    @staticmethod
    def _block_moves(
        opp_threats: list, legal_set: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Return legal moves that block opponent threats.

        Covers every opponent threat that can win on the next turn
        (FIVE, OPEN_FOUR, CLOSED_FOUR) as well as credible developing
        threats (OPEN_THREE) that demand a response.
        """
        moves: set[tuple[int, int]] = set()
        for t in opp_threats:
            if t.threat_type == ThreatType.CLOSED_FOUR:
                if t.gap is not None:
                    moves.add(t.gap)
                else:
                    for end in t.open_ends:
                        moves.add(end)
            elif t.threat_type in (ThreatType.FIVE, ThreatType.OPEN_FOUR, ThreatType.OPEN_THREE):
                if t.gap is not None:
                    moves.add(t.gap)
                for end in t.open_ends:
                    moves.add(end)
        return moves & legal_set

    _BLOCK_BOOST_FACTOR = 5.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        return self._raw_model(tensor)

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
        legal_set = set(legal)

        our_threats = ThreatDetector.detect_all(board, board.current_player)

        if hard_override:
            winning_moves = self._winning_moves(our_threats, legal_set)
            if winning_moves:
                probs = [
                    (m, 1.0 / len(winning_moves) if m in winning_moves else 0.0)
                    for m in legal
                ]
                return probs, 1.0, {"overridden": True, "reason": "immediate_win"}

        move_probs, value = self.evaluate(board)

        opp_threats = ThreatDetector.detect_all(board, Player(-board.current_player))
        block_set = self._block_moves(opp_threats, legal_set)
        if block_set:
            move_probs = [
                (m, p * self._BLOCK_BOOST_FACTOR if m in block_set else p)
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
        # Cache per-board legal sets and opp threats for the fill-in pass.
        per_board: list[
            tuple[list[tuple[int, int]], set[tuple[int, int]]] | None
        ] = []

        for i, board in enumerate(boards):
            legal = board.get_legal_moves()
            legal_set = set(legal)

            our_threats = ThreatDetector.detect_all(board, board.current_player)
            opp_threats = ThreatDetector.detect_all(
                board, Player(-board.current_player)
            )

            if hard_override:
                winning_moves = self._winning_moves(our_threats, legal_set)
                if winning_moves:
                    probs = [
                        (m, 1.0 / len(winning_moves) if m in winning_moves else 0.0)
                        for m in legal
                    ]
                    results.append(
                        (probs, 1.0, {"overridden": True, "reason": "immediate_win"})
                    )
                    per_board.append(None)
                    continue

            neural_indices.append(i)
            neural_boards.append(board)
            results.append(([], 0.0, None))  # placeholder
            per_board.append((legal, opp_threats))

        if neural_boards:
            neural_results = self.batch_evaluate(neural_boards)
        else:
            neural_results = []

        for j, (i, board) in enumerate(zip(neural_indices, neural_boards)):
            move_probs, value = neural_results[j]
            legal, opp_threats = per_board[
                i
            ]  # per_board[i] is always non-None when i in neural_indices

            block_set = self._block_moves(opp_threats, set(legal))
            if block_set:
                move_probs = [
                    (m, p * self._BLOCK_BOOST_FACTOR if m in block_set else p)
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
