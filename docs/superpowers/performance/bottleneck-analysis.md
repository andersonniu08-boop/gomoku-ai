# MCTS Performance Bottleneck Analysis

Date: 2026-05-26
Model: 5-block 64-channel ResNet / 10-block 128-channel SE+Attention ResNet
Device: CPU (Intel), no CUDA available

## Executive Summary

The MCTS engine spends **~83–89% of its time in neural network forward passes**.
Tree traversal, board operations, and data postprocessing together account for
only **~10–12%**. This means:

1. **The neural network is the only real bottleneck.** Everything else is cheap
   enough that optimizing it won't materially improve sims/sec.
2. **Any optimization that reduces the number of forward passes per search**
   (virtual loss, better batch efficiency, graph reuse) has 5–8× more leverage
   than any optimization to board operations.
3. **Playing-strength bottlenecks are more concerning than throughput
   bottlenecks.** A search that runs 1100 sims/sec can already play well if
   those simulations are well-directed.

---

## 1. Measured Performance

### Small Model (5 blocks, 64 channels, no SE, no Attention)

| Position | Batch | Sims | Avg Time | Sims/Sec | NN eval % |
|----------|-------|------|----------|----------|-----------|
| Empty board | 8 | 200 | 171 ms | 1,172 | 83% |
| Opening (10 stones) | 8 | 200 | 280 ms | 895 | 89% |
| Opening | 1 | 400 | 838 ms | 477 | — |
| Opening | 2 | 400 | 714 ms | 560 | — |
| Opening | 4 | 400 | 491 ms | 815 | — |
| Opening | 8 | 400 | 432 ms | 927 | — |
| Opening | 16 | 400 | 445 ms | 898 | — |
| Opening | 32 | 400 | 411 ms | 972 | — |
| Midgame (has threats) | 8 | 200 | **1.6 ms** | 163,685 | 0%* |
| Threat open-four | 8 | 200 | **0.6 ms** | 318,156 | 0%* |
| Must-block | 8 | 200 | **0.6 ms** | 328,002 | 0%* |

*\*Threat override short-circuits MCTS entirely, no NN evaluation needed.*

### Full Model (10 blocks, 128 channels, SE + Attention)

| Position | Batch | Sims | Avg Time | Sims/Sec |
|----------|-------|------|----------|----------|
| Opening | 1 | 200 | 4,467 ms | 45 |
| Opening | 8 | 200 | 1,733 ms | 115 |
| Opening | 32 | 200 | 1,629 ms | 123 |

---

## 2. Bottleneck Breakdown

### Small Model — Empty Board (200 sims, batch=8, 25 NN passes)

```
search.total            171 ms   100%
├── search.neural_eval  146 ms    85%
│   ├── model_forward   140 ms    82%    ← PRIMARY BOTTLENECK
│   ├── postprocess       3 ms     2%
│   └── tensor_construct  2 ms     1%
├── search.descend       21 ms    12%
│   ├── make_move         4 ms     2%
│   ├── puct_select       2 ms     1%
│   ├── board_copy        0.4 ms   0.2%
│   └── is_terminal       0.1 ms   0.1%
├── search.expand_backup  2 ms     1%
│   ├── create_nodes      1 ms     0.7%
│   └── backup_walk       0.2 ms   0.1%
└── threat_check          0.4 ms   0.2%
```

### Key Ratios

- **NN forward pass vs tree traversal: 7:1** (140 ms vs 21 ms)
- **NN forward pass vs board copy: 350:1** (140 ms vs 0.4 ms)
- **NN forward pass vs PUCT selection: 70:1** (140 ms vs 2 ms)

---

## 3. Throughput Bottlenecks (ranked by impact)

### 🔴 P0 — Neural network forward pass (83–89% of time)

The single dominating cost. Each forward pass through the 5-block network takes
~5.6 ms (batch=8). The full 10-block SE+Attention model takes ~13.3 ms per
batch (batch=8) — roughly 2.4× slower.

**nn forward pass cost per simulation:**
- Small model, batch=8: **5.6 ms / 8 ≈ 0.7 ms/sim**
- Full model, batch=8: **13.3 ms / 8 ≈ 1.7 ms/sim**

**What this means:**
- On CPU, we get ~1100 sims/sec with the small model, ~115 with full model.
- On GPU, these numbers would be dramatically higher (100×+ for batch inference).
- GPU is the single highest-leverage optimization.

### 🟡 P1 — Batch size tuning (1.5–2× headroom)

Batch size significantly impacts throughput:
- batch=1: 477 sims/sec
- batch=8: 927 sims/sec  (1.9× improvement)
- batch=32: 972 sims/sec (2.0× improvement from batch=1)

The diminishing returns beyond batch=8 suggest the CPU forward pass is
compute-bound rather than memory-bound at this batch size.

### 🟢 P2 — Tensor construction + postprocessing (3–4% of time)

Building the numpy array, transferring to torch tensor, computing exp on GPU,
and copying back to CPU total ~5 ms per search (3%). Not worth optimizing yet.

### 🟢 P3 — Board copy (0.2% of time)

At 0.002 ms per copy, 200 copies total 0.4 ms per search. Not a bottleneck.

### 🟢 P4 — PUCT selection (1.1% of time)

At 0.003 ms per selection. Not a bottleneck.

### 🟢 P5 — Threat detection (0.2% per normal search)

`ThreatDetector.detect_all` takes ~290 µs per call. Called twice in
`_check_forced` = ~585 µs. Negligible for normal search, but very impactful
when it saves 171 ms of full search via short-circuit.

---

## 4. Search-Quality Bottlenecks

These are distinct from throughput and affect **playing strength** rather than
speed.

### 🔴 P0 — Double MCTS search in SelfPlayGame

**Found bug:** `SelfPlayGame.play()` calls `mcts.search(board)` on line 174 to
get visit counts, then calls `mcts.select_move(board, temp)` on line 188.
`select_move()` calls `self.search(board)` again internally, running the full
MCTS a second time.

**Impact:** +111% overhead per move during self-play. For a game of ~30 moves
with 400 sims each, this wastes ~12,000 simulations per game.

**Fix:** Store the search result and select the move from it without re-running.

### 🟡 P1 — Policy cutoff at 40 moves

`_POLICY_CUTOFF = 40` caps the branching factor by dropping low-prior moves.
For positions with 40+ legal moves (common in midgame), this prunes potentially
good moves based solely on the (untrained) policy head's ranking. Early in
training when the policy head is random, this could prune winning moves.

**Mitigation:** 40 is generous enough for most positions. Not urgent.

### 🟡 P2 — No search-tree reuse between moves

Each `search()` call starts from scratch with a new root. The previous search's
tree is discarded even though the chosen move's subtree is directly reusable.

**Impact:** ~400 sims discarded per move. For a 30-move game, ~12,000 sims
wasted.

**Mitigation:** Tree reuse is a moderately complex refactor. Worth doing after
baseline optimizations.

### 🟢 P3 — No virtual loss in non-batched mode

With batch_size=1, there's no virtual loss applied, so all parallel descents in
a batch would collapse to the same leaf. The batching code uses virtual loss
correctly when batch_size > 1.

---

## 5. GPU Underutilization Risks

The current architecture transfers data to/from device per batch:

**Per batch GPU cost (hypothetical):**
- Tensor transfer: ~0.01 ms (negligible for 3×15×15 = 675 float32 values)
- Kernel launch: ~0.02–0.1 ms per layer
- Total overhead: ~0.5–1 ms per forward pass on GPU
- Compute time: ~0.05–0.1 ms for small model on GPU

**On GPU, the bottleneck shifts from compute to:**
1. **Kernel launch overhead** — 25 batches × ~20 layers = 500 kernel launches
   per search. This dominates on GPU.
2. **Tensor construction on CPU** — numpy alloc + torch.from_numpy + .to(device)
   is CPU-bound work that blocks GPU.
3. **Postprocessing on CPU** — policy renormalization happens on CPU after
   transferring data back.

**The BatchedLeafEvaluator already addresses some of these concern:**
- Single `torch.from_numpy` per batch avoids per-board overhead
- Batched `exp()` on GPU reduces CPU work
- Batch `values.tolist()` reduces CPU-GPU sync points

---

## 6. Optimization Priority Ranking

| Priority | Optimization | Est. Speedup | Complexity | Category |
|----------|-------------|:---:|:---:|----------|
| P0 | **GPU inference** (CUDA) | 50–200× | Medium | Throughput |
| P0 | **Fix double-search bug** in SelfPlayGame | 2× | Trivial | Throughput |
| P1 | **Search-tree reuse** between moves | 1.5–2× | Medium | Quality |
| P2 | **Batch size tuning** (32–64) | 1.5–2× | Trivial | Throughput |
| P2 | **Larger/efficient batching** in BatchedLeafEvaluator | 1.2× | Trivial | Throughput |
| P3 | **NUMA-aware tensor construction** | 1.1× | Low | Throughput |
| P4 | **Threaded board copy** | 1.01× | High | Throughput |
| P4 | **PUCT sqrt_parent_n caching** | 1.005× | Low | Throughput |

---

## 7. Recommended Optimization Roadmap

### Phase 1 — Fixed bugs (this week)
1. Fix double-search in SelfPlayGame — 10 lines, 2× self-play speedup
2. Set default batch_size to 32 (already done in evaluator branch)

### Phase 2 — GPU acceleration (1–2 weeks)
1. Add CUDA support detection and tensor pinning
2. Move policy postprocessing to GPU fully (exp + renormalization)
3. Allocate pinned memory for host-to-device transfers

### Phase 3 — Search quality (2–4 weeks)
1. Implement tree reuse between moves
2. Evaluate whether _POLICY_CUTOFF is harming strength
3. Profile with the FULL model (128ch, SE, Attention)

### Phase 4 — Self-play throughput (1–2 weeks)
1. Reduce MCTS simulations during self-play (200 instead of 400)
2. Increase self-play games per iteration to compensate
3. Add batched board-to-tensor conversion in BatchedLeafEvaluator

### Avoid
- Premature optimization of board operations (<1% of runtime)
- C++ extensions for board/move generation
- Threat scanning optimization (already fast enough)
- PUCT formula changes without empirical validation

---

## 8. Concrete Measured Findings

| Finding | Evidence | Impact |
|---------|----------|--------|
| NN forward pass is 83% of search time | Empty board profile: 140/171 ms | GPU is the main lever |
| Double-search bug in selfplay | 117ms vs 247ms measured | 111% overhead |
| Threat override is extremely effective | 0.6 ms vs 171 ms for threat positions | Don't skip threat detection |
| Batch size 8→32 gives 1.05× on CPU | 927 vs 972 sims/sec | Diminishing returns on CPU |
| Board copy is negligible | 0.4 ms per 200 copies (0.2%) | Don't optimize |
| PUCT selection is negligible | 2 ms per search (1.1%) | Don't optimize |
| Full model is 8× slower than small model | 115 vs 895 sims/sec | Use small model for self-play |
| BatchedLeafEvaluator matches wrapper | ~13 ms vs ~13 ms for batch=8 | Good, proceed with it |
