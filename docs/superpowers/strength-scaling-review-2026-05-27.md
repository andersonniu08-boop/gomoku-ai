# Gomoku AI — Strength-Scaling Architecture Review

**Date:** 2026-05-27
**Scope:** Full-system engineering review of current bottlenecks and ten proposed improvement directions
**Goal:** Evolve from "tactically improving experimental engine" to "genuinely strong scalable Gomoku system"

---

## Current State Summary

**Tactical system (strong):** Three-layer threat handling — root override (`_check_forced`), leaf expansion override (`order_and_filter_moves` with `hard_override=True`), and per-candidate incremental line scoring. Open-four and must-block detection works reliably.

**MCTS (adequate but shallow):** 400 simulations default at `selfplay/mcts.py:92`. PUCT selection, virtual loss for batched descent, Dirichlet noise at root. Single-board mutation + restore model avoids per-node copies. Correct but slow.

**Network (reasonable baseline):** 10-block residual CNN, 128 channels, SE channel attention, single-head spatial self-attention, value global pooling. Four spec'd improvements (multi-head attention, dilated convs, deeper policy head, CBAM spatial attention) are ready to implement.

**Training (minimal-viable):** 10 self-play games per iteration, 400 sims each. 500k replay buffer with symmetry augmentation. Adam + cosine annealing. Single-process.

---

## Bottleneck Hierarchy

```
Current:   Search budget (400 sims) >>> Network capacity > Training data volume
           ↓ (after Phase 1)
Next:      Training data volume > Network capacity > Search budget
           ↓ (after Phase 2-3)
Then:      Network architecture > Inference throughput > Search budget
           ↓ (after Phase 4)
Future:    Opening theory / endgame precision > Everything else
```

---

## Proposal Evaluations

### 1. Stronger MCTS Scaling

**Current:** 400 sims, hardcoded default.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Very High — doubling sims gives reliable +100-200 Elo on the steep part of the curve |
| Implementation | Trivial — `num_simulations` is already a constructor param; time_budget is ~10 lines |
| Compute cost | Linear with sims; amortized by tree reuse (#2) |
| Engineering complexity | None |
| Training implications | None (search is inference-only) |
| Scalability | Already scales |
| Destabilization risk | Zero |
| Verdict | **NOW** |

This is the single highest-leverage, lowest-effort improvement in the entire project. Moving from 400→800→1600 sims produces more Elo gain than any architectural change.

Recommendations:
- Raise `num_simulations` default to 800
- Add `time_budget_ms` parameter that loops search until budget exhausted
- Add "Strong" mode preset (1600+ sims)

---

### 2. Persistent Search-Tree Reuse

**Current:** Every `MCTS.search()` builds a tree from scratch. ~80% of simulations are under the played move and get discarded.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Very High — 2-5× effective sim budget at same wall-clock |
| Implementation | Moderate — re-root tree under played move, promote child to root, discard siblings |
| Compute cost | Negative (saves work) |
| Engineering complexity | ~100-150 lines; careful board/tree state management required |
| Training implications | Compound benefit in self-play (each game is faster) |
| Scalability | Works with all sim budgets |
| Destabilization risk | Medium — tree corruption bugs produce subtly wrong search results |
| Verdict | **NOW** |

Implementation sketch:
1. `select_move()` returns the chosen child node alongside the move
2. After opponent plays, `re_root(child_node)` — that node becomes the new root
3. Child's `board` must match the post-opponent-move state
4. Fall back to fresh search when opponent plays a move not in the tree

Every strong engine (Leela, KataGo, Stockfish NNUE) does this. Not implementing it leaves a 2-5× multiplier unused.

---

### 3. Tactical Search Expansion

**Current:** Three-layer 0-ply tactical system. No "if I play X, they block Y, then I have double threat at Z" reasoning.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Moderate — catches 2-3 move forced sequences that 0-ply misses |
| Implementation | Moderate — shallow alpha-beta on forcing moves only, ~200-300 lines |
| Compute cost | Low (forcing-move branching factor is 3-5, not 30) |
| Engineering complexity | Medium — threat classification correctness is critical |
| Training implications | None |
| Scalability | Optional per-move |
| Destabilization risk | Medium — misclassification cascades |
| Verdict | **LATER** |

The current 0-ply system already catches the most common cases. Multi-ply tactical search becomes more valuable when the network is stronger and can accurately evaluate non-forcing leaf nodes. Existing `ThreatDetector` and incremental line scanning provide all the building blocks.

---

### 4. Better Move Ordering

**Current:** `order_and_filter_moves` already has tactical scoring (five/open-four/closed-four/open-three creation and blocking), connectivity bonuses, proximity bonuses, hard overrides, and top-40 cutoff.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Low — 5-10% simulation efficiency gain at best |
| Implementation | Easy — killer/history heuristic tables are 30-50 lines |
| Compute cost | Negligible |
| Engineering complexity | Low |
| Training implications | None |
| Scalability | Already adequate |
| Destabilization risk | Low |
| Verdict | **LATER (low priority)** |

PUCT already balances exploration — move ordering matters far less than in alpha-beta. The existing ordering is ~80% of maximum. Killer/history heuristics won't move the needle.

---

### 5. Self-Play Quality Improvements

**Current:** 400 sims per move, temperature=1.0 for first 15 moves, Dirichlet α=0.03 with ε=0.25. 10 games/iteration.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Very High — better self-play data is a compound improvement over training iterations |
| Implementation | Parameter changes for sims/temperature; moderate for replay diversity |
| Compute cost | Linear with sim budget increase |
| Engineering complexity | Low for sim increase; medium for advanced sampling |
| Training implications | Direct — this IS the training signal quality |
| Scalability | Benefits from tree reuse and distributed workers |
| Destabilization risk | Low for sim increase; medium for aggressive temperature rescheduling |
| Verdict | **NOW (self-play sim budget), LATER (advanced tuning)** |

**The feedback loop:** Better search → better policy targets → better network → better priors → even better search. A 2× increase in self-play sims compounds over iterations and might produce 3-4× improvement in final model strength.

**Recommendations:**
- Increase self-play sims to 800 (match inference sim budget)
- Tune Dirichlet α based on legal move count: α = 10/N (more concentrated when fewer moves)
- Extend temperature annealing window (15→30 moves) for richer opening exploration

**Replay diversity is adequate** — uniform sampling with symmetry augmentation on retrieval is correct at this stage. Prioritized/windowed sampling adds complexity without clear benefit for Gomoku's state distribution.

---

### 6. Neural Network Expansion

**Current:** `GomokuNet` and spec'd improvements at `neural/model.py`.

**Evaluation of four planned improvements:**

| Improvement | Prior accuracy gain (est.) | Elo gain (est.) | Params added |
|---|---|---|---|
| Multi-head attention (1→2) | +1-3% | +10-30 | ~0 |
| Dilated convs (blocks 7-9) | +2-5% | +20-40 | ~0 |
| Deeper policy head | +1-3% | +10-20 | ~37k |
| CBAM spatial attention | +0.5-2% | +5-15 | ~50 |
| **Combined** | +5-10% | +40-80 | ~37k total |

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Low-Medium — real but incremental (~40-80 combined) |
| Implementation | Already spec'd out, ~60 lines total across model.py changes |
| Compute cost | Negligible (same conv FLOPs, attention head dim halves, SE/spatial are lightweight) |
| Engineering complexity | Low — all fit within existing architecture family |
| Training implications | Checkpoints incompatible with old architecture; retrain from scratch |
| Scalability | Already adequate |
| Destabilization risk | Low |
| Verdict | **NOW** |

These are cheap and correct. They should be done. But they won't transform the engine from "clearly weak" to "strong" — search budget dominates.

**On transformer trunks:** For Gomoku's 15×15 grid, full self-attention (225² = 50k scores) is trivially cheap. A 6-8 layer transformer trunk would give ideal line-pattern recognition. But transformers need more training data than CNNs to converge (less inductive bias). Defer until self-play throughput scales up.

**On deeper/wider networks:** Going to 20+ blocks or 256 channels is diminishing returns for Gomoku's complexity. The current 10×128 with SE+attention is already in the right regime. Capacity is not the binding constraint.

---

### 7. Training Infrastructure Scaling

**Current:** Single-process training loop. Polls for optional worker game files. Self-play dominates iteration time by 20-40× over GPU training time. Training GPU is idle during self-play.

**Scaling math:**
- 1 self-play game: ~60-120s (30-60 moves × 400 sims × ~5ms/sim)
- 10 games: ~10-20 min
- Training epoch: ~30s
- **CPU-bound self-play is the bottleneck, not GPU-bound training**

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Very High — N workers = ~N× self-play throughput = ~N× training data |
| Implementation | Major — distributed systems engineering |
| Compute cost | High (in infrastructure, not code) |
| Engineering complexity | High — worker orchestration, batch queue, fault tolerance, file I/O races |
| Training implications | Fundamental — this IS how you scale AlphaZero |
| Scalability | The whole point |
| Destabilization risk | Medium-High — distributed systems have distributed failure modes |
| Verdict | **LATER (after per-worker efficiency is maximized)** |

**Architecture reference (standard AlphaZero distributed design):**

```
Worker 1 ──┐
Worker 2 ──┼──→ GPU Inference Server (batched leaf eval) ──→ Results
Worker 3 ──┤                                                    │
   ...     ──┘                                                  │
Central Trainer ←── Replay Buffer ←── Game files ←──────────────┘
```

Multiple CPU workers run MCTS independently, submitting leaf evaluation requests to a shared GPU inference server. The server batches requests across workers, keeping the GPU saturated. Game results flow into the replay buffer. The trainer trains on accumulated data independently.

**Key insight:** Scaling a slow worker just gives you more slow games. Fix per-worker efficiency first (tree reuse, sim budget), then scale horizontally. A worker that's 5× more efficient plus 4× more workers = 20× training throughput.

---

### 8. Search Efficiency Optimizations

**Current:** The `_run_search` hot loop at `selfplay/mcts.py:218`. With `batch_size=1`: serial descent (PUCT select, make_move, board copy, terminal check), GPU eval, expand, backup. Repeat 400×.

**Batching failure analysis:**

`batch_size=128` is 23× slower than `batch_size=1` on RTX 3050 because:

1. **Virtual loss contamination:** 128 concurrent descents with stale tree statistics. Virtual loss pushes later descents to explore different branches, but with 128 concurrent descents in a tree of depth 20-40, virtual loss creates excessive exploration noise — many descents waste simulations.

2. **Serial descent cost dominates:** 128 full descents (PUCT select + make_move × depth) take far longer than one GPU forward call for 128 boards. The batched forward saves 127 GPU calls (~5ms each = ~635ms), but the 128 serial descents cost ~50ms+.

3. **Tree statistics staleness:** With batch_size=1, each simulation updates the tree before the next descent. With batch_size=128, all 128 descents use the same pre-batch statistics.

**Python overhead:** 400 sims × ~20 function calls × profiling overhead = thousands of Python function calls per search. Each is individually fast but cumulatively significant.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | High (micro-optimizations: +20-50% faster search), Very High (Cython/Rust: 5-10×) |
| Implementation | Easy for profiling/micro-opt; major for compiled-language rewrite |
| Compute cost | Negative (saves time) |
| Engineering complexity | Low for micro-opt; Very High for compiled rewrite |
| Training implications | Faster self-play = more training data |
| Scalability | Directly enables more sims |
| Destabilization risk | Low for micro-opt; Medium for rewrite |
| Verdict | **NOW (profile and micro-optimize), LATER (compiled rewrite)** |

**Specific micro-optimizations to profile:**
- `_puct_select`: `sqrt(parent_n)` recomputed per call but only changes when children are visited — cache it
- `board.is_terminal()`: called at every leaf; already incremental but verify no waste
- `node.children.items()`: Python dict iteration bottleneck in PUCT selection

**On GPU batching for MCTS:** Don't optimize for GPU utilization when the GPU isn't the bottleneck. The benchmark proves batching is counterproductive at this scale. Focus on search speed, not GPU occupancy.

---

### 9. Evaluation / Benchmarking Systems

**Current:** `run_evaluation` — 100 games latest vs best every 5 iterations, win rate ≥ 55% promotion. No Elo tracking, no tactical benchmarks, no regression tests.

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Zero directly, High for development velocity |
| Implementation | Elo tracking ~30 lines; benchmark suites require position curation |
| Compute cost | Evaluation games already run |
| Engineering complexity | Low for Elo; Medium for benchmark suites |
| Training implications | Enables data-driven decisions about what works |
| Scalability | Runs on the same infrastructure |
| Destabilization risk | Zero |
| Verdict | **NOW (Elo tracking), LATER (benchmark suites)** |

Elo tracking is essential — without it, you can't tell if training is improving the model or oscillating. Compute Elo from the existing evaluation games (Bayesian Elo or simple logistic regression). Store per-checkpoint ratings and plot the trend.

Tactical benchmark suite: 20-30 positions with known correct moves (win-in-1, win-in-2, must-block, double threat, etc.). Run on every checkpoint to detect regressions.

---

### 10. Gameplay UX Improvements

**Evaluation:**

| Dimension | Rating |
|---|---|
| Elo impact | Zero |
| Implementation | Varies |
| Compute cost | N/A |
| Engineering complexity | Medium-High (frontend work) |
| Training implications | None |
| Scalability | N/A |
| Destabilization risk | Zero |
| Verdict | **LATER** |

UX work does nothing for playing strength. Do not spend engineering time here until the engine is strong.

---

## What's Already Good Enough

- **Threat detection / tactical override:** Three-layer system catches immediate wins and blocks reliably. Don't over-invest.
- **Board encoding:** Correct, efficient, perspective-relative.
- **Replay buffer design:** Symmetry augmentation on retrieval (not storage). Correct.
- **Profiler:** Lightweight, integrated. Good foundation.
- **Training loss / optimizer:** Standard AlphaZero (CE policy + MSE value), Adam + cosine annealing. Correct.
- **Move ordering for expansion:** Existing `order_and_filter_moves` with incremental line scanning covers ~80% of achievable benefit.

## What's Dangerously Underdeveloped

- **Search budget:** 400 sims is barely enough to find good moves beyond immediate tactics. This is the #1 strength ceiling.
- **No tree reuse:** Discarding the entire search tree after every move is the biggest efficiency waste.
- **No Elo tracking:** Can't measure improvement. Can't detect regressions. Flying blind.
- **Self-play search quality:** Training on 400-sim data creates a low ceiling on policy target quality regardless of architecture.

## What's Overrated (at this stage)

- **Batched GPU inference for MCTS:** The benchmark proves batching is counterproductive. GPU isn't the bottleneck.
- **More complex neural architectures:** Collectively maybe +50-80 Elo. Doubling simulations is +100-200 Elo for zero architecture change.
- **Transposition tables:** 15×15 board is small; move sequences rarely transpose in MCTS.
- **Move ordering sophistication:** Killer/history heuristics are the last 20%; won't move the needle.

## What's Underestimated

- **Tree reuse:** Effectively "free" simulations. 2-5× multiplier sitting unused.
- **Self-play sim budget as a feedback loop:** Better search → better targets → better network → better priors → compound effect over iterations. A 2× self-play sim increase may give 3-4× final model improvement.
- **Python overhead:** Death by a thousand cuts. 400 sims × 20 calls × overhead. Moving the inner loop to compiled code could double search speed.

---

## Ranked Recommendations

### Top 3 Highest-Leverage Immediate Improvements

**1. Increase MCTS simulations + think-time budget (#1)**
- Effort: ~10 lines
- Expected gain: +100-200 Elo (400→800-1600 sims)
- Risk: zero
- Why first: Free strength. Everything else builds on this.

**2. Search-tree reuse between moves (#2)**
- Effort: ~100-150 lines
- Expected gain: 2-5× effective sim budget at same wall-clock
- Risk: medium (tree corruption bugs)
- Why second: Multiplies the benefit of #1. Combined: "1600+ effective sims" for ~400 sims of wall-clock cost.

**3. Self-play search budget increase (#5)**
- Effort: parameter change (400→800)
- Expected gain: Higher quality training targets → compound effect over iterations
- Risk: low
- Why third: Better training data is a force multiplier for everything downstream.

### Top 3 Longer-Term Scaling Improvements

**1. Multi-worker distributed self-play (#7)**
- How every strong open-source engine scales. Maximize per-worker efficiency first, then scale horizontally.
- Target: 4-8 workers → 4-8× training throughput.

**2. Python hot-loop → compiled language (#8)**
- 5-10× search speedup enables 5000+ sims per move or 5× more self-play games per hour.
- Most impactful single optimization, but most expensive to build.

**3. Transformer trunk (#6, deferred)**
- Once training data volume is sufficient, a transformer trunk gives ideal line-pattern recognition for Gomoku.
- Prerequisite: scaled training pipeline producing millions of positions.

---

## Recommended Development Sequence

```
Phase 1 (Now — ~1-2 days):
├── Increase num_simulations default 400→800, add time_budget_ms parameter
├── Implement search-tree reuse (re-root after opponent move)
├── Increase self-play sims 400→800
├── Add simple Elo tracking to training loop
└── Implement 4 planned network improvements
    (multi-head attention, dilated convs, deeper policy head, CBAM spatial attention)

Phase 2 (Next — ~1-2 weeks):
├── Profile and micro-optimize MCTS hot loop (cache sqrt(parent_n), reduce allocations)
├── Tactical benchmark suite (20-30 positions with known correct moves)
├── Tune Dirichlet noise and temperature schedule
└── Multi-ply tactical search (shallow forced-sequence solver, depth 4-6)

Phase 3 (Medium-term — ~1 month):
├── Multi-worker self-play with shared GPU inference queue
├── Cloud GPU for training (A100/H100 on Runpod)
├── Opening book / opening diversity system
└── Automated game analysis pipeline

Phase 4 (Long-term — ~2-3 months):
├── Python hot loop → Cython/Rust
├── Transformer trunk experimentation
├── Full distributed training with parameter server
└── Rich UI/UX (search visualization, strength presets)
```

---

## Core Insight

**The engine's weakness is primarily a search budget problem, not an architecture problem.**

A 10-block/128-channel CNN with SE and attention, given 5000 sims per move with tree reuse, would play dramatically stronger than the same network with 400 sims. The four planned neural improvements are worth doing because they're cheap, but the overwhelming priority should be making search faster and deeper.

The AlphaZero scaling law: better networks → better priors → fewer sims needed → faster search. But you need sufficient sims first to generate training targets good enough for the network to learn anything useful. At 400 sims, the policy targets contain significant noise. Increasing the self-play sim budget is the single change most likely to create a virtuous cycle of improvement.
