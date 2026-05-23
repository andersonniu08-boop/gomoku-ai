# Workstream C — Human vs AI Move Comparison

## Project Overview

You are implementing one of three parallel workstreams for the Explainability
phase of a Gomoku AI engine. This is a modular research project following the
AlphaZero paradigm. The codebase is at `/home/anderson/projects/gomoku-ai`.

## Full project instructions (CLAUDE.md)

The entire project specification is in `/home/anderson/projects/gomoku-ai/CLAUDE.md`.
Read it before starting any implementation — it contains strict rules about
modularity, imports, naming, type hints, coding standards, and testing.

Key rules:
- Imports flow one direction: `engine` <- `neural` <- `selfplay` <- `explain`
- No PyTorch in `engine/`. No game logic in `neural/`.
- All public functions/methods must have type hints and docstrings.
- No file shall exceed ~500 lines.
- Use `Optional[X]` not `X | None`.
- Use `@dataclass(slots=True)` for data containers.
- Tests use pytest, live in `tests/`, named `test_<module>.py`.

## What you are building

You own the **human vs AI move comparison tool**. Given a board position and
a human-chosen move, produce a structured report: the AI's top-k moves with
visit counts, priors, and Q values; the human move's rank and statistics;
the position value before and after the move.

This is the most integration-heavy workstream. You will:
1. Add a `SearchResult` dataclass and `search_with_stats()` method to MCTS.
2. Build the comparison pipeline in `explain/comparison.py`.

Also read the spec at `docs/superpowers/specs/2026-05-23-explainability-design.md`
for the full design context.

## Pre-work already done (do NOT redo)

Nothing special — the pre-work (`evaluate_raw()` on the wrapper) is for
Workstreams A and B. You use the existing `evaluate()` and `evaluate_with_threats()`
APIs. However, you DO need to add `search_with_stats()` to MCTS yourself.

## Files you will modify or create

### Modify: `selfplay/mcts.py`

Add the following BEFORE the `MCTS` class (or inline with `MCTSNode`):

```python
@dataclass(slots=True)
class SearchResult:
    """Full MCTS search statistics for a root position."""
    visit_counts: dict[tuple[int, int], int]
    q_values: dict[tuple[int, int], float]
    priors: dict[tuple[int, int], float]
    total_simulations: int
```

Add a new method to the `MCTS` class:

```python
def search_with_stats(self, board: Board) -> SearchResult:
    """Like search() but also returns Q-values and priors.

    Calls the same internal search loop as search() but exposes
    root children's full statistics instead of just visit proportions.
    """
    # The cleanest approach: refactor the shared search loop into
    # a private helper _run_search(board, root) -> None that mutates
    # the tree. Both search() and search_with_stats() call it, then
    # read different data from the root.
    #
    # If you refactor:
    #   _run_search(self, board: Board, root: MCTSNode) -> None:
    #       # same search loop as current search() method
    #       ...
    #
    #   search(self, board: Board) -> dict[tuple[int, int], float]:
    #       root = MCTSNode()
    #       self._run_search(board, root)
    #       # return visit proportions as before
    #
    #   search_with_stats(self, board: Board) -> SearchResult:
    #       root = MCTSNode()
    #       self._run_search(board, root)
    #       # return full SearchResult with visits, q_values, priors
    #
    # If you DON'T refactor (prefer minimal changes), just duplicate
    # the search loop. The spec prefers refactoring if it's clean.
```

Do NOT change the signature or behavior of `search()` — existing callers
(self-play, select_move) must continue to work unchanged.

### Create: `explain/comparison.py`

Public API:

```python
@dataclass(slots=True)
class MoveCandidate:
    move: tuple[int, int]
    prior: float
    visit_count: int           # 0 if MCTS not used
    q_value: float             # 0.0 if MCTS not used
    is_human_move: bool

@dataclass(slots=True)
class MoveComparison:
    board: Board
    human_move: tuple[int, int]
    legal: bool
    top_candidates: list[MoveCandidate]
    human_candidate: Optional[MoveCandidate]
    human_rank: Optional[int]
    value_before: float
    value_after: Optional[float]
    ai_recommended: tuple[int, int]
    threat_overridden: bool
    search_stats: dict

    def to_dict(self) -> dict:
        """JSON-serializable dict."""

    @classmethod
    def from_dict(cls, data: dict) -> MoveComparison:
        """Reconstruct from dict."""

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
        wrapper: Inference wrapper.
        board: Board position before the move.
        human_move: (row, col) the human played.
        use_mcts: If True, run MCTS for visit-based comparison.
        num_simulations: MCTS iterations (only if use_mcts).
        top_k: Number of top AI candidates to include.

    Returns:
        MoveComparison with structured comparison data.
    """

def compare_move_fast(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    human_move: tuple[int, int],
    *,
    top_k: int = 5,
) -> MoveComparison:
    """Fast comparison using policy head only (no MCTS).

    Useful for quick feedback when MCTS overhead is undesirable.
    Sets use_mcts=False internally.
    """
```

### Comparison Pipeline (internal logic)

1. **Early exit if terminal:** `board.is_terminal()`. Return immediately.
2. **Legal move check.** If `human_move` not in `board.get_legal_moves()`,
   set `legal=False`. AI recommendation still works — just no human candidate.
3. **Value before.** `wrapper.evaluate(board)` -> `value_before`.
4. **MCTS path:**
   - Detect threat override: if `threat_override=True`, first check
     `_check_forced()`. If forced, `search_with_stats()` returns the forced
     distribution. Set `threat_overridden=True`.
   - Otherwise, run `search_with_stats()` normally.
   - Sort moves by `visit_count` descending.
   - Build `MoveCandidate` list (top_k items).
   - Find human move rank.
5. **Fast path:**
   - `wrapper.evaluate(board)` -> `(move_probs, value)`.
   - Sort by prior descending.
   - Build `MoveCandidate` list (visit_count=0, q_value=0.0).
   - Find human move rank.
6. **Value after.** Copy board: `board_copy = board.copy()`;
   `board_copy.make_move(*human_move)`.
   Skip if board is now terminal.
   `wrapper.evaluate(board_copy)` -> `value_after`.
7. **Assemble MoveComparison** with all fields populated.

### Edge cases to handle

- **Illegal human move:** `legal=False`. `human_candidate=None`.
  `human_rank=None`. Still show AI top-k.
- **Terminal board:** Return immediately (value_before reflects game outcome).
- **Forced win detected by MCTS:** `threat_overridden=True`.
- **Human move not in top-k:** `human_rank=None`, `human_candidate=None`.
  This is normal — it just means the human chose a move the AI considered low.
- **Zero legal moves:** Return immediately (full board draw).
- **Board after move is terminal:** `value_after=None` (or +1.0 for human win,
  -1.0 for AI win from the AI's perspective).

### `tests/test_comparison.py`

Test the following invariants (all testable with random weights):

1. **Legal human move:** `comparison.legal == True`,
   `comparison.human_candidate is not None`.
2. **Illegal human move:** `comparison.legal == False`,
   `comparison.human_candidate is None`.
3. **AI recommended is a legal move.** Verify `ai_recommended` is in
   `board.get_legal_moves()`.
4. **Value range:** `value_before` and `value_after` in [-1, 1].
5. **Top candidates sorted:** descending by visit_count (or prior for fast path).
6. **Top-k length:** `len(top_candidates) <= top_k`.
7. **Fast path:** same structure as MCTS path (with visit_count=0, q_value=0.0).
8. **JSON roundtrip:** `MoveComparison.from_dict(d.to_dict()) == d` (equals
   comparison compares fields).
9. **Terminal board:** returns immediately, no crash.
10. **Empty board:** center move is recommended (model may have random bias,
    but AI picks SOMETHING legal).
11. **Threat override:** if `threat_override=True`, the result reflects it.
    (Test by mocking or using a board with a forced win pattern.)
12. **Search stats:** `"num_simulations"` matches the argument.
13. **Human rank:** 1 <= rank <= top_k when human move is in top-k.
14. **Value after improves:** on a board where you're about to win,
    `value_after` > `value_before` (roughly).
15. **to_dict doesn't raise:** verify serialization works.
16. **from_dict roundtrip:** verify deserialization matches.

## Files you must NOT modify

- `neural/wrapper.py` — pre-work is already done for other sessions
- `neural/model.py` — no changes needed
- `engine/board.py`, `engine/encoding.py`, `engine/threats.py` — no changes
- `explain/saliency.py` — owned by Workstream A
- `explain/activations.py` — owned by Workstream B

Files you DO modify:
- `selfplay/mcts.py` — add `SearchResult` and `search_with_stats()`.
  Do NOT change existing `search()` or `select_move()` signatures.

## Dependencies

- `selfplay/mcts.py` for `MCTS`, `MCTSNode`, `SearchResult`, `search_with_stats()`
- `neural/wrapper.py` for `GomokuInferenceWrapper.evaluate()`
- `engine/board.py` for `Board.copy()`, `make_move()`, `get_legal_moves()`

## Import conventions

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from engine.board import Board
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS, SearchResult
```

For `comparison.py`, import `Board` from `engine.board`, NOT from `selfplay`.

## JSON serialization

`MoveComparison.to_dict()` should produce a dict with only JSON-native types:
- Tuples become `[r, c]` lists
- Board -> board serialization (can use `repr` for debugging, or store
  `current_player` + `move_history` for reconstruction)
- Optional[float] -> None maps to JSON null
- numpy types converted to Python native types

`from_dict()` must reconstruct the full `MoveComparison`. Board
reconstruction just needs `current_player` and `move_history` to call
individual `make_move()` calls on a fresh board.

## Integration notes

- Your module will be re-exported from `explain/__init__.py` by whomever
  creates the `__init__.py` later.
- No other `explain/` module imports from you.
- A future Phase 5 web UI will call `compare_move()` and render the
  `MoveComparison` as a side-by-side comparison panel.
- The move comparison can optionally display saliency maps (Workstream A)
  alongside the move data — this is a render-level concern, not a
  dependency, so just leave a slot for it.
