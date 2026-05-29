# TacticalSolver — Usage Audit & Role Assessment

## Decision: **Option A — Document and preserve behavior.**

TacticalSolver is well-designed, fully tested, and serves a distinct role
from the hot-path tactical components. It is kept as-is.

---

## Production Usage

| Location | Call | Purpose |
|----------|------|---------|
| `selfplay/selfplay.py:260` | `TacticalSolver.analyze_lightweight(board)` | Resignation safety guard — verifies the value head is not incorrectly resigning a tactically urgent position. **This is the only production call.** |

## Test-Only Usage

| File | Lines | Scope |
|------|-------|-------|
| `tests/test_tactical.py` | 1–641 (32 tests) | Full coverage: wins, blocks, double-threats, forced sequences, prior boosts |
| `tests/test_selfplay.py` | 472–529 (2 tests) | Mocks `analyze_lightweight` to verify resignation safety |
| `tests/test_neural.py` | 467 (comment only) | References TacticalSolver for context on `boosted_blocks` |

---

## Component Comparison

### 1. TacticalSolver (`engine/tactical.py`, 505 lines)
**Role:** Heavyweight tactical analysis. Stateless, pure engine-layer, no
neural/MCTS dependency. Builds a `TacticalAnalysis` dataclass with winning
moves, must-block, urgent blocks, double threats, per-move creation/blocking
scores, and forced-sequence search.

- Used in production ONLY for `analyze_lightweight` (resignation guard).
- Full `analyze()` is UNUSED in production but fully tested.
- `_score_all_moves` copies the board per move and runs full
  `ThreatDetector.detect_all` — intentionally slower but more accurate than
  the incremental approach in `move_ordering.py`.
- `_find_forced_sequence` is UNUSED in production but fully tested.

### 2. threats.py (`engine/threats.py`, 463 lines)
**Role:** Low-level threat pattern detection engine. Finds FIVE, OPEN_FOUR,
CLOSED_FOUR, OPEN_THREEs. Provides `detect_all`, `has_double_threat`,
`count_threats`, `evaluate`, `get_completion_cells`.

- Used by ALL other tactical components (TacticalSolver, MCTS `_check_forced`,
  `move_ordering` via `evaluate_with_threats`).
- The foundation layer — no duplication.

### 3. move_ordering.py (`selfplay/move_ordering.py`, 306 lines)
**Role:** Per-move tactical scoring and filtering for MCTS expansion. Uses
incremental line scanning (O(board_size) per candidate, no board copies).

- Used in MCTS `_run_search` expansion (hot path) to reorder neural priors.
- `hard_override=True` catches deep wins/must-blocks during tree expansion.
- Scoring constants mirror TacticalSolver's but are cheaper to compute.
- **Does NOT depend on TacticalSolver** — completely independent algorithm.

### 4. MCTS `_check_forced` (`selfplay/mcts.py`, lines 538–607)
**Role:** Pre-MCTS short-circuit. Uses `ThreatDetector` directly to find
immediate wins and must-blocks. Returns a uniform distribution over forced
moves, bypassing all neural evaluation and tree search.

- Called at the top of `search()` and `search_with_stats()`.
- Simpler and faster than TacticalSolver for this specific task — no
  `TacticalAnalysis` object, no extra field computation.
- **Does NOT depend on TacticalSolver.**

---

## Overlap Analysis

| Component Pair | Overlap | Assessment |
|---------------|---------|------------|
| `_check_forced` vs `analyze_lightweight` | Both find wins + must-blocks via ThreatDetector | Intentional. `_check_forced` is optimized for the MCTS hot path; `analyze_lightweight` is used in the resignation check. Same logic, different call sites. No consolidation needed. |
| `move_ordering` vs `TacticalSolver._score_all_moves` | Both score moves tactically | Different algorithms. `move_ordering` is incremental (fast, in MCTS hot path). TacticalSolver uses board copies (accurate but slower). The constants are similar but not identical — each tuned to its use case. |
| `evaluate_with_threats` vs TacticalSolver | Both boost block moves | `evaluate_with_threats` (in `wrapper.py`) uses ThreatDetector directly, boosting block moves by 5×. TacticalSolver's `get_move_boost` returns 10000.0 for must-block. These serve different tiers: the wrapper is for neural evaluation, TacticalSolver for deterministic short-circuit. |

---

## Rationale for Option A

1. **No dead code concern:** `analyze_lightweight` is actively used.
   The unused pieces (`analyze`, `forced_sequence`) are fully tested and
   provide a documented capability for future use (explainability, tactics
   in self-play training targets, debugging).

2. **Clean architecture:** TacticalSolver lives in `engine/` with no
   higher-layer dependencies. It's a pure engine module per AGENTS.md rules.

3. **No risk:** Not integrating it into MCTS avoids adding overhead to
   the hot path. The existing `_check_forced` and `move_ordering` are
   faster and sufficient.

4. **Test coverage is excellent:** 32 dedicated tests plus integration
   tests for the resignation guard. No maintenance burden.

5. **No architectural change needed:** The components are cleanly
   separated with distinct responsibilities. Consolidation would create
   coupling without clear benefit.
