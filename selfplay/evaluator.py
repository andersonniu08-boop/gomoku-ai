"""Batched leaf evaluator for efficient GPU inference during MCTS.

Provides ``BatchedLeafEvaluator`` — a drop-in for ``wrapper.batch_evaluate()``
that significantly improves GPU utilisation by:

- Building batched tensors directly from numpy arrays (avoiding per-board
  ``torch.tensor`` allocations and ``torch.cat`` overhead).
- Running the model once per batch (amortising kernel-launch latency).
- Extracting values with a single ``.tolist()`` call (one H2D sync instead
  of one per board).
- Converting policies in batch on GPU then transferring once to CPU.
- Warming up CUDA kernels at construction time.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import torch

from engine.board import Board, Player
from neural.wrapper import GomokuInferenceWrapper
from selfplay.profiler import Profiler


class EvaluationResult(NamedTuple):
    """Result of evaluating a single leaf position.

    Attributes:
        move_probs:  ``[(move, prior), ...]`` for legal moves, normalised.
        value:       Scalar position evaluation in [-1, 1] from the
                     perspective of ``board.current_player``.
    """
    move_probs: list[tuple[tuple[int, int], float]]
    value: float


class BatchedLeafEvaluator:
    """Evaluates leaf board positions in efficient GPU batches.

    Optimises the inner loop of MCTS leaf evaluation through three
    mechanisms:

    1. **Batch tensor construction** — a single float32 ``ndarray`` is
       pre-allocated and filled per board via boolean indexing.  One
       ``torch.from_numpy`` + one ``.to(device)`` for the entire batch.

    2. **Batched post-processing** — all value scalars are extracted via
       ``.tolist()`` (one CPU-GPU sync for the batch instead of per-board
       ``.item()`` calls).  All policy probabilities are ``exp``-ed on the
       device then transferred to CPU in one shot.

    3. **CUDA warmup** — a dummy forward pass at construction time
       initialises CUDA kernels so the first real batch doesn't pay the
       cold-start cost.

    Parameters:
        wrapper:          Inference wrapper providing the loaded model.
        target_batch_size:  Maximum boards per GPU forward pass.  Larger
                            sets are split into chunks; smaller sets are
                            evaluated in whatever size they arrive.
    """

    def __init__(
        self,
        wrapper: GomokuInferenceWrapper,
        *,
        target_batch_size: int = 64,
        profiler: Optional[Profiler] = None,
    ):
        self.wrapper = wrapper
        self.device = wrapper.device
        self.target_batch_size = target_batch_size
        self.profiler = profiler or Profiler()
        self.profiler.disable()  # off by default; MCTS enables when needed

        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self, boards: list[Board]
    ) -> list[EvaluationResult]:
        """Evaluate *boards* and return one result per board (same order).

        When the input exceeds ``target_batch_size`` the list is split
        into chunks, each evaluated with a separate forward pass.
        """
        if not boards:
            return []

        results: list[EvaluationResult] = []
        for start in range(0, len(boards), self.target_batch_size):
            chunk = boards[start:start + self.target_batch_size]
            results.extend(self._evaluate_chunk(chunk))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run one dummy forward pass so CUDA kernels are compiled and
        cached before the first real MCTS evaluation."""
        if self.device.type != "cuda":
            return
        dummy = torch.zeros(
            (1, 3, Board.SIZE, Board.SIZE), device=self.device
        )
        with torch.no_grad():
            self.wrapper.model(dummy)
        torch.cuda.synchronize()

    def _evaluate_chunk(
        self, boards: list[Board]
    ) -> list[EvaluationResult]:
        """Run one batched forward pass and convert outputs."""
        batch = len(boards)

        # --- Batch tensor construction --------------------------------
        with self.profiler.measure("eval.tensor_construction"):
            arr = np.zeros(
                (batch, 3, Board.SIZE, Board.SIZE), dtype=np.float32
            )
            for i, board in enumerate(boards):
                cp = board.current_player
                grid = board.grid  # int8: +1 Black, -1 White, 0 empty
                arr[i, 0] = (grid == cp)      # bool → float32
                arr[i, 1] = (grid == -cp)
                if cp == Player.BLACK:
                    arr[i, 2] = 1.0            # channel 2 stays 0.0 for White

            tensor = torch.from_numpy(arr).to(self.device)

        # --- Batched forward pass -------------------------------------
        with self.profiler.measure("eval.model_forward"):
            with torch.no_grad():
                log_policy, value = self.wrapper.model(tensor)

        # --- Efficient result extraction ------------------------------
        with self.profiler.measure("eval.postprocess"):
            values = value.view(-1).tolist()

            probs = torch.exp(log_policy)  # (B, 225) on GPU
            probs_cpu = probs.cpu().numpy()

            results: list[EvaluationResult] = []
            for i, board in enumerate(boards):
                legal = board.get_legal_moves()
                prob_grid = probs_cpu[i].reshape(Board.SIZE, Board.SIZE)

                result_probs = [
                    (move, float(prob_grid[move])) for move in legal
                ]
                total = sum(p for _, p in result_probs)
                if total > 0:
                    result_probs = [
                        (move, p / total) for move, p in result_probs
                    ]
                else:
                    uniform = 1.0 / max(len(legal), 1)
                    result_probs = [(move, uniform) for move in legal]

                results.append(EvaluationResult(result_probs, values[i]))

        return results
