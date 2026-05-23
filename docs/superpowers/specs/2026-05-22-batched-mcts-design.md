# Batched MCTS Inference — Design Spec

## Overview

Replace one-at-a-time neural inference in MCTS with batched forward passes,
using virtual loss to enable parallel descent within a single search.

**Motivation:** Current MCTS runs 400 sequential `(1, 3, 15, 15)` forward passes
per move. An RTX 3050 can handle batches of 8-32 boards in the same wall-clock
time as one. Virtual loss steers concurrent simulations down different tree paths
so multiple leaves are ready for the GPU at once.

## Scope

Two changes, delivered in order:

1. **`neural/wrapper.py`** — add `batch_evaluate()` and
   `batch_evaluate_with_threats()` methods.
2. **`selfplay/mcts.py`** — add virtual loss to `MCTSNode`, rewrite
   `MCTS.search()` to descend in batches.

## Non-scope

- `engine/` — zero changes. Board, threats, encoding are unchanged.
- `neural/model.py` — already accepts `(B, 3, 15, 15)`; no changes needed.
- `selfplay/selfplay.py` and `selfplay/train.py` — MCTS API (`search()`,
  `select_move()`) is unchanged. Existing callers work without modification.

---

## Part 1 — Batched Wrapper Methods

### `batch_evaluate(boards)`

```python
def batch_evaluate(
    self, boards: list[Board]
) -> list[tuple[list[tuple[tuple[int, int], float]], float]]:
    """Evaluate N boards in one forward pass.
    
    Returns one (move_probs, value) per board, in the same order.
    """
```

**Implementation:**
1. Stack `N` tensors via `board_to_tensor()` → `(N, 3, 15, 15)`.
2. Single `model(tensor)` call under `torch.no_grad()`.
3. For each row `i`: call `policy_to_move_probs(log_policy[i], boards[i])`
   and extract `value[i].item()`.
4. Empty input → empty list.

### `batch_evaluate_with_threats(boards, hard_override=True)`

Same as above, but applies the per-board threat-detection logic from the
existing `evaluate_with_threats()`: check for immediate wins (override policy,
set value=1.0), boost blocking moves for opponent open-fours.

---

## Part 2 — Virtual Loss + Batched MCTS Search

### `MCTSNode` changes

One new field:

```python
virtual_loss: int = 0
```

Modified `q` property:

```python
@property
def q(self) -> float:
    total_n = self.visit_count + self.virtual_loss
    if total_n == 0:
        return 0.0
    return (self.total_value - self.virtual_loss) / total_n
```

A virtual loss of 1 on a node pulls Q toward -1 when no real visits exist,
steering subsequent PUCT selections away from that path.

### `MCTS` changes

New constructor parameter: `batch_size: int = 8`.

### Rewritten `search(board)` algorithm

```
1. If terminal → return {}
2. _check_forced(board) → if forced, return immediately (unchanged)
3. sim_board = board.copy()
4. root = MCTSNode()

5. While sims_done < num_simulations:
     a. N = min(batch_size, num_simulations - sims_done)
     b. Descend N times from root (sequential, not parallel threads):
        - Each descent runs PUCT with virtual-loss-adjusted Q, so descent
          i+1 steers away from paths already taken by descents 1..i.
        - On reaching a leaf:
            * Record (path, leaf_board)
            * Add virtual_loss += 1 to each node on the path
            * If leaf is already terminal: mark for immediate backup
     c. Collect non-terminal leaf boards, call wrapper.batch_evaluate(boards)
     d. For each leaf:
        - Expand node: create MCTSNode children with priors from batch result
        - Backup value up the path (negating at each level)
        - Remove virtual_loss from path nodes
        - Update visit_count and total_value
     e. sims_done += N

6. Return visit-count proportions over legal moves (unchanged logic)
```

**Key invariant:** Virtual loss is added during descent and removed during
backup. Visit counts and total_values are only updated during the real backup
phase. A simulation that is "in flight" (virtual loss added, not yet backed up)
impacts PUCT selection but does not affect the final visit distribution.

### Backward compatibility

`batch_size=1` recovers sequential behavior: one descent, one eval, one backup
per iteration. Virtual loss is added then immediately removed before the next
iteration, so it has zero impact.

---

## Error Handling & Edge Cases

| Case | Behavior |
|------|----------|
| Empty batch (0 boards) | Return `[]` |
| Terminal leaf found mid-descent | Skip neural eval; back up game result directly. Virtual loss still added/removed for consistency. |
| All leaves in batch are terminal | Skip `batch_evaluate` call entirely; back up all via terminal values. |
| Batch size > remaining sims | `min(batch_size, remaining)` handles this. |
| Batch size > `POLICY_CUTOFF` or legal moves | No interaction — batch size controls simulation parallelism, not branching factor. |

## What Stays the Same

- `select_move()` — identical API and behavior.
- `_puct_select()` — unchanged (already reads `q` property, which now includes virtual loss).
- `_check_forced()` — unchanged, called once before search begins.
- `_terminal_value()` — unchanged.
- `_expand_and_evaluate()` — **removed**, replaced by inlined expand+backup in the batched loop.

## Test Plan

| Test | Location | What It Verifies |
|------|----------|-----------------|
| `test_batch_evaluate_returns_correct_count` | `test_neural.py` | N boards → N results |
| `test_batch_evaluate_empty` | `test_neural.py` | Empty list → empty list |
| `test_batch_evaluate_matches_single` | `test_neural.py` | Each board's result matches individual `evaluate()` call |
| `test_virtual_loss_q_no_visits` | `test_mcts.py` | Q = -1 when virtual_loss=1, visit_count=0 |
| `test_virtual_loss_q_with_visits` | `test_mcts.py` | Q adjusts correctly with mix of real + virtual |
| `test_batched_search_distribution` | `test_mcts.py` | Returns valid probability distribution over legal moves |
| `test_batched_search_finds_immediate_win` | `test_mcts.py` | Single search call finds winning move in one step |
| `test_batch_size_1_still_works` | `test_mcts.py` | Regression: sequential behavior preserved |
| `test_batched_search_deterministic` | `test_mcts.py` | Same seed + same batch size = same result |
| `test_search_batch_larger_than_simulations` | `test_mcts.py` | batch_size=800, num_simulations=100 — correct |

Existing tests (84) must continue to pass without modification.

## Design Decisions

- **Batch size as constructor parameter, not `search()` parameter.**
  The batch size is a search-engine configuration, not a per-search tuning
  knob. It belongs on the MCTS instance, matching `num_simulations` and
  `c_puct`.

- **No threat detection on every leaf.**
  Threat detection at the start of `search()` handles the opening case
  (opponent just made a winning threat). Adding per-leaf threat detection
  would add CPU overhead that eats into the batching speedup. Terminals are
  handled naturally by `board.is_terminal()`.

- **Virtual loss = 1 (not a tunable parameter).**
  A virtual loss of 1 means "assume this path is a loss until proven
  otherwise." This is the standard AlphaZero approach. Making it tunable
  adds complexity without evidence it helps.

- **`_expand_and_evaluate` removed rather than kept alongside batched path.**
  The batched loop subsumes its functionality. Keeping both would be dead
  code and a maintenance burden. The `batch_size=1` fallback provides the
  equivalent behavior for testing.
