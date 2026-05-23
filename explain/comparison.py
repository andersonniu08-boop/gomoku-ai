"""Human vs AI move comparison tool.

Given a board position and a human-chosen move, produces a structured
report: the AI's top-k moves with visit counts, priors, and Q values;
the human move's rank and statistics; the position value before and
after the move.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from engine.board import Board, Player
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS, SearchResult


@dataclass(slots=True)
class MoveCandidate:
    """Statistics for a single candidate move."""

    move: tuple[int, int]
    prior: float
    visit_count: int = 0
    q_value: float = 0.0
    is_human_move: bool = False


@dataclass(slots=True)
class MoveComparison:
    """Structured comparison between a human move and AI recommendation."""

    board: Board
    human_move: tuple[int, int]
    legal: bool
    top_candidates: list[MoveCandidate]
    human_candidate: Optional[MoveCandidate]
    human_rank: Optional[int]
    value_before: float
    value_after: Optional[float]
    ai_recommended: Optional[tuple[int, int]]
    threat_overridden: bool
    search_stats: dict

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        def _move_to_list(m: tuple[int, int]) -> list[int]:
            return [m[0], m[1]]

        def _candidate_to_dict(c: MoveCandidate) -> dict:
            return {
                "move": _move_to_list(c.move),
                "prior": c.prior,
                "visit_count": c.visit_count,
                "q_value": c.q_value,
                "is_human_move": c.is_human_move,
            }

        return {
            "human_move": _move_to_list(self.human_move),
            "legal": self.legal,
            "top_candidates": [_candidate_to_dict(c) for c in self.top_candidates],
            "human_candidate": (
                _candidate_to_dict(self.human_candidate) if self.human_candidate is not None else None
            ),
            "human_rank": self.human_rank,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "ai_recommended": (
                _move_to_list(self.ai_recommended) if self.ai_recommended is not None else None
            ),
            "threat_overridden": self.threat_overridden,
            "search_stats": dict(self.search_stats),
            "board": {
                "current_player": int(self.board.current_player),
                "move_history": [[r, c] for r, c in self.board.move_history],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> MoveComparison:
        """Reconstruct a ``MoveComparison`` from a dict produced by ``to_dict()``."""

        def _list_to_move(lst: list[int]) -> tuple[int, int]:
            return (lst[0], lst[1])

        def _dict_to_candidate(d: dict) -> MoveCandidate:
            return MoveCandidate(
                move=_list_to_move(d["move"]),
                prior=d["prior"],
                visit_count=d["visit_count"],
                q_value=d["q_value"],
                is_human_move=d["is_human_move"],
            )

        # Reconstruct board from move history.
        board = Board()
        for r, c in data.get("board", {}).get("move_history", []):
            board.make_move(r, c)

        top_candidates = [_dict_to_candidate(c) for c in data["top_candidates"]]

        human_candidate = (
            _dict_to_candidate(data["human_candidate"])
            if data.get("human_candidate") is not None
            else None
        )

        ai_recommended = (
            _list_to_move(data["ai_recommended"])
            if data.get("ai_recommended") is not None
            else None
        )

        return cls(
            board=board,
            human_move=_list_to_move(data["human_move"]),
            legal=data["legal"],
            top_candidates=top_candidates,
            human_candidate=human_candidate,
            human_rank=data.get("human_rank"),
            value_before=data["value_before"],
            value_after=data.get("value_after"),
            ai_recommended=ai_recommended,
            threat_overridden=data.get("threat_overridden", False),
            search_stats=data.get("search_stats", {}),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MoveComparison):
            return NotImplemented
        # Compare to_dict output to handle Board identity differences
        # and tuple/list coercion from serialization roundtrip.
        return self.to_dict() == other.to_dict()


def _build_candidates(
    sorted_moves: list[tuple[tuple[int, int], float]],
    human_move: tuple[int, int],
    legal: bool,
    top_k: int,
    *,
    visit_counts: Optional[dict[tuple[int, int], int]] = None,
    q_values: Optional[dict[tuple[int, int], float]] = None,
) -> tuple[list[MoveCandidate], Optional[MoveCandidate], Optional[int]]:
    """Build top-k candidate list and locate the human move.

    Args:
        sorted_moves:  Moves sorted by descending priority (visit count or prior).
        human_move:    The human-chosen move.
        legal:         Whether the human move is legal.
        top_k:         Maximum number of candidates to return.
        visit_counts:  Visit counts per move (None for fast path).
        q_values:      Q values per move (None for fast path).

    Returns:
        (top_candidates, human_candidate, human_rank)
    """
    top_candidates: list[MoveCandidate] = []
    human_candidate: Optional[MoveCandidate] = None
    human_rank: Optional[int] = None

    for rank, (move, score) in enumerate(sorted_moves[:top_k]):
        vc = (visit_counts or {}).get(move, 0)
        qv = (q_values or {}).get(move, 0.0)
        is_hm = legal and move == human_move
        candidate = MoveCandidate(
            move=move,
            prior=score,
            visit_count=vc,
            q_value=qv,
            is_human_move=is_hm,
        )
        top_candidates.append(candidate)
        if is_hm:
            human_candidate = candidate
            human_rank = rank + 1

    if legal and human_rank is None:
        # Human move is legal but not in top-k — find its full rank.
        for rank, (move, _) in enumerate(sorted_moves):
            if move == human_move:
                human_rank = rank + 1
                break

    return top_candidates, human_candidate, human_rank


def compare_move(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    human_move: tuple[int, int],
    *,
    use_mcts: bool = True,
    num_simulations: int = 400,
    top_k: int = 5,
) -> MoveComparison:
    """Compare a human move against the AI's recommendation.

    Args:
        wrapper:   Inference wrapper.
        board:     Board position before the move.
        human_move: (row, col) the human played.
        use_mcts:  If True, run MCTS for visit-based comparison.
        num_simulations: MCTS iterations (only if ``use_mcts``).
        top_k:     Number of top AI candidates to include.

    Returns:
        ``MoveComparison`` with structured comparison data.
    """
    # 1. Early exit if terminal.
    if board.is_terminal():
        winner = board.check_win()
        if winner is None:
            v = 0.0
        elif winner == board.current_player:
            v = 1.0
        else:
            v = -1.0
        return MoveComparison(
            board=board,
            human_move=human_move,
            legal=False,
            top_candidates=[],
            human_candidate=None,
            human_rank=None,
            value_before=v,
            value_after=None,
            ai_recommended=None,
            threat_overridden=False,
            search_stats={},
        )

    # 2. Legal move check.
    legal_moves = board.get_legal_moves()
    legal = human_move in legal_moves

    # 3. Value before.
    _, value_before = wrapper.evaluate(board)

    # 4. Search with MCTS or fast-path.
    threat_overridden = False
    top_candidates: list[MoveCandidate] = []
    human_candidate: Optional[MoveCandidate] = None
    human_rank: Optional[int] = None
    ai_recommended: Optional[tuple[int, int]] = None
    search_stats: dict = {}

    if use_mcts:
        mcts = MCTS(
            wrapper,
            num_simulations=num_simulations,
            threat_override=True,
        )
        result = mcts.search_with_stats(board)

        # Detect threat override: total_simulations == 0 when forced.
        threat_overridden = result.total_simulations == 0

        search_stats = {
            "num_simulations": num_simulations,
            "nodes_visited": len(result.visit_counts),
            "total_simulations_actual": result.total_simulations,
        }

        # Sort by visit count descending.
        sorted_moves = sorted(
            result.visit_counts.items(),
            key=lambda x: (-x[1], -result.priors.get(x[0], 0.0)),
        )

        top_candidates, human_candidate, human_rank = _build_candidates(
            sorted_moves,
            human_move,
            legal,
            top_k,
            visit_counts=result.visit_counts,
            q_values=result.q_values,
        )

        ai_recommended = sorted_moves[0][0] if sorted_moves else None

    else:
        # Fast path: policy head only, no MCTS.
        move_probs, value_before = wrapper.evaluate(board)
        sorted_probs = sorted(move_probs, key=lambda x: -x[1])

        top_candidates, human_candidate, human_rank = _build_candidates(
            sorted_probs,
            human_move,
            legal,
            top_k,
        )

        ai_recommended = sorted_probs[0][0] if sorted_probs else None

    # 5. Value after.
    value_after: Optional[float] = None
    if legal:
        board_copy = board.copy()
        board_copy.make_move(*human_move)
        if board_copy.is_terminal():
            winner = board_copy.check_win()
            if winner is None:
                value_after = 0.0
            elif winner == board.current_player:
                value_after = 1.0
            else:
                value_after = -1.0
        else:
            _, value_after = wrapper.evaluate(board_copy)

    return MoveComparison(
        board=board,
        human_move=human_move,
        legal=legal,
        top_candidates=top_candidates,
        human_candidate=human_candidate,
        human_rank=human_rank,
        value_before=value_before,
        value_after=value_after,
        ai_recommended=ai_recommended,
        threat_overridden=threat_overridden,
        search_stats=search_stats,
    )


def compare_move_fast(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    human_move: tuple[int, int],
    *,
    top_k: int = 5,
) -> MoveComparison:
    """Fast comparison using policy head only (no MCTS).

    Useful for quick feedback when MCTS overhead is undesirable.

    Args:
        wrapper:    Inference wrapper.
        board:      Board position before the move.
        human_move: (row, col) the human played.
        top_k:      Number of top AI candidates to include.

    Returns:
        ``MoveComparison`` with structured comparison data (visit_count=0,
        q_value=0.0 for all candidates).
    """
    return compare_move(
        wrapper,
        board,
        human_move,
        use_mcts=False,
        top_k=top_k,
    )
