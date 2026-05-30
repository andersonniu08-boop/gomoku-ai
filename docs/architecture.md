# NeuralGomoku — Architecture Document

## Project Overview

Hybrid Gomoku (15×15, 5-in-a-row) engine combining Monte Carlo Tree Search with
deep neural networks for position evaluation and policy guidance, following the
self-play reinforcement learning paradigm. Built with PyTorch and NumPy.

## Directory Structure

```
neural-gomoku/
│
├── engine/                         # Game logic layer (zero framework deps)
│   ├── board.py                    #   Board, Player — state, moves, win detection
│   ├── threats.py                  #   ThreatDetector, Threat — pattern recognition
│   └── encoding.py                 #   board_to_tensor(), policy_to_move_probs()
│
├── neural/                         # Neural network layer (PyTorch only)
│   ├── model.py                    #   GomokuNet, ResidualBlock — dual-headed CNN
│   └── wrapper.py                  #   GomokuInferenceWrapper — checkpoint loading,
│                                   #     board→tensor→inference→(policy, value)
│
├── selfplay/                       # Search & training layer
│   ├── mcts.py                     #   MCTS, MCTSNode — PUCT search, move selection
│   ├── selfplay.py                 #   SelfPlayGame — game generation, D₄ symmetries
│   ├── replay_buffer.py            #   ReplayBuffer — FIFO storage, batch sampling
│   └── train.py                    #   compute_loss, run_evaluation, main() loop
│
├── explain/                         # Explainability layer (Phase 4)
│   ├── saliency.py                  #   Integrated Gradients / vanilla gradient attribution
│   ├── activations.py               #   Residual block forward-hook activation capture
│   └── comparison.py                #   Human vs AI move comparison with MCTS statistics
│
├── tests/                           # 105 tests across 9 files
│   ├── test_encoding.py            #   7 tests
│   ├── test_mcts.py                #   18 tests
│   ├── test_neural.py              #   6 tests
│   ├── test_selfplay.py            #   30 tests
│   ├── test_threats.py             #   25 tests
│   ├── test_train.py               #   9 tests
│   ├── test_saliency.py            #   29 tests
│   ├── test_activations.py         #   13 tests
│   └── test_comparison.py          #   24 tests
│
├── checkpoints/                    # Model weight files (gitignored)
│   ├── best.pt                     #   Strongest model — used for self-play
│   └── latest.pt                   #   Most recently trained — candidate promotion
│
├── data/                           # Training data (gitignored)
│   └── replay_buffer.pt            #   Serialized ReplayBuffer state
│
├── docs/                           # Documentation
├── CLAUDE.md                       # Project rules and standards
├── main.py                         # Entry point
└── requirements.txt                # Python dependencies
```

## Layer Architecture

### Import Discipline

Imports flow strictly downward. Higher layers may import from lower layers;
lower layers must **never** import from higher layers.

```
explain/      ──┐
                │  may import from selfplay/, neural/, and engine/
selfplay/     ──┤
                │  may import from neural/ and engine/
neural/       ──┤
                │  may import from engine/
engine/       ──┘
                │  zero internal project imports
```

Circular imports are forbidden. If two modules need each other, extract the
shared concern into a third module that both import.

### Layer Responsibilities

#### `engine/` — Game Logic

Zero framework dependencies. NumPy is acceptable; PyTorch is not.

| File | Class/Function | Responsibility |
|------|---------------|----------------|
| `board.py` | `Player` (IntEnum) | BLACK (+1), WHITE (-1) |
| | `Board` | 15×15 grid, `make_move()`, `undo_move()`, `check_win()`, `is_terminal()`, `get_legal_moves()`, `copy()` |
| `threats.py` | `ThreatType` (IntEnum) | FIVE, OPEN_FOUR, CLOSED_FOUR, OPEN_THREE |
| | `Threat` | Pattern descriptor: type, position, open ends, gap |
| | `ThreatDetector` | `detect_all()`, `evaluate()`, double-threat detection |
| `encoding.py` | `board_to_tensor()` | Board → `(1, 3, 15, 15)` FloatTensor (current player stones, opponent stones, turn indicator) |
| | `policy_to_move_probs()` | Log-policy → legal-move-filtered probability distribution |

**Key invariants:**
- Board is deterministic: same move sequence always produces same state
- Win detection is incremental (scans only from last-placed stone)
- Legal moves are neighbor-based (adjacent to existing stones + center for opening)
- No search, no neural inference, no UI code

#### `neural/` — Neural Network

PyTorch only. Knows nothing about game rules, board representation, or search.

| File | Class | Responsibility |
|------|-------|---------------|
| `model.py` | `ResidualBlock` | Two 3×3 convs with batch norm and skip connection |
| | `GomokuNet` | Dual-headed residual CNN (see Architecture Diagram below) |
| `wrapper.py` | `GomokuInferenceWrapper` | Checkpoint loading, device placement, `evaluate(board)` → `(move_probs, value)`, optional threat-aware evaluation |

**GomokuNet architecture:**

```
Input: (B, 3, 15, 15)
  │
  ▼
Conv2d(3→64, 3×3) → BatchNorm → ReLU
  │
  ▼
ResidualBlock(64) × 5
  │
  ├─────────────────────┐
  ▼                     ▼
Policy Head          Value Head
Conv2d(64→2, 1×1)   Conv2d(64→1, 1×1)
BatchNorm            BatchNorm
ReLU                 ReLU
Flatten → FC(450, 225)  Flatten → FC(225, 64) → ReLU → FC(64, 1) → Tanh
LogSoftmax           Tanh scalar
```

**Output contract:**
- Policy: log-softmax over 225 cells — `(B, 225)`
- Value: tanh scalar in [-1, 1] — `(B, 1)`

#### `selfplay/` — Search & Training

May import from both `engine` and `neural`. Consumed by the training loop;
no UI or API dependencies.

| File | Class/Function | Responsibility |
|------|---------------|----------------|
| `mcts.py` | `MCTSNode` | Edge statistics: prior, visit_count, total_value, children, Q(s,a) |
| | `MCTS` | PUCT search, `search(board)` → visit distribution, `select_move(board, temp)`. Uses `BatchedLeafEvaluator` for GPU-efficient leaf evaluation. |
| `evaluator.py` | `EvaluationResult` | (move_probs, value) named tuple returned by leaf evaluation |
| | `BatchedLeafEvaluator` | Optimized batched neural evaluation: pre-allocated tensor array, single `.tolist()` sync, on-device policy softmax, CUDA warmup |
| `selfplay.py` | `TrainingExample` | (state (3,15,15), policy (225,), value float) |
| | `SelfPlayGame` | Plays one game: MCTS per move, records training triples, temperature annealing |
| | `SYMMETRIES` / `augment_examples()` | D₄ dihedral group: 8 rotations/reflections applied to states and policies |
| `replay_buffer.py` | `ReplayBuffer` | FIFO deque (max 500K), `sample()`, `get_batch()`, `state_dict()` / `from_state_dict()` |
| `train.py` | `compute_loss()` | Cross-entropy policy loss + MSE value loss |
| | `save_model_checkpoint()` | Save raw `state_dict` for InferenceWrapper compatibility |
| | `train_on_examples()` | Shuffle → mini-batch → forward/backward/optimizer step → avg loss |
| | `_play_eval_game()` | One deterministic game between two different wrappers |
| | `run_evaluation()` | 100-game match, alternating colors, returns new model win rate |
| | `main()` | Two-phase training loop orchestrator |

#### `explain/` — Explainability (Phase 4)

May import from `selfplay/`, `neural/`, and `engine/`. Read-only consumers of
model outputs and MCTS statistics. Produces structured data (dataclass instances,
NumPy arrays) with no UI dependencies — ready for Phase 5 rendering.

| File | Class/Function | Responsibility |
|------|---------------|----------------|
| `saliency.py` | `SaliencyMap` | 15×15 attribution heatmap in [0, 1] |
| | `compute_saliency()` | Integrated Gradients or vanilla gradient attribution |
| | `attribution_to_grid()` | (3, 15, 15) gradient → (15, 15) single-channel heatmap |
| `activations.py` | `ActivationSnapshot` | Captured feature maps per residual block |
| | `ActivationCapture` | Context manager for safe forward hook lifecycle |
| | `capture_activations()` | High-level: board → forward pass → snapshot |
| | `select_top_channels()` | Top-k channels by L2 activation norm |
| | `channel_to_grid()` | Extract single channel as 15×15 grid |
| `comparison.py` | `MoveCandidate` | Single move statistics: prior, visits, Q, is_human_move |
| | `MoveComparison` | Full comparison: top-k candidates, human rank, values before/after |
| | `compare_move()` | MCTS-powered comparison pipeline |
| | `compare_move_fast()` | Policy-head-only fast path |

## Data Flow

### Training Loop

```
Iteration:
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Phase A — Self-Play                                        │
  │  ┌──────┐    ┌─────────┐    ┌──────┐    ┌──────────────┐   │
  │  │ best │───▶│ wrapper │───▶│ MCTS │───▶│ SelfPlayGame │   │
  │  │ .pt  │    │evaluate │    │search│    │   .play()    │   │
  │  └──────┘    └─────────┘    └──────┘    └──────┬───────┘   │
  │                                                │           │
  │                                          TrainingExamples  │
  │                                                │           │
  │                                          ┌─────▼────────┐  │
  │                                          │ ReplayBuffer │  │
  │                                          │  .add()      │  │
  │                                          └──────────────┘  │
  │                                                             │
  │  Phase B — Training                                         │
  │  ┌──────────┐    ┌──────────┐    ┌───────────────┐         │
  │  │ latest   │───▶│ GomokuNet│───▶│ train_on_     │         │
  │  │ .pt      │    │ .train() │    │ examples()    │         │
  │  └──────────┘    └──────────┘    └───────┬───────┘         │
  │                                          │                 │
  │                                    Adam + CosineAnnealing  │
  │                                          │                 │
  │                                    ┌─────▼────────┐        │
  │                                    │ latest.pt    │        │
  │                                    │ (updated)    │        │
  │                                    └──────────────┘        │
  │                                                             │
  │  Evaluation (every 5 iterations)                            │
  │  ┌──────────┐    ┌──────────┐                              │
  │  │ latest.pt│ vs │ best.pt  │  100 games, deterministic    │
  │  └────┬─────┘    └────┬─────┘                              │
  │       │               │                                    │
  │       └───────┬───────┘                                    │
  │               ▼                                            │
  │        win_rate ≥ 55%?                                     │
  │        YES → promote latest → best.pt                      │
  │        NO  → keep best.pt                                  │
  └─────────────────────────────────────────────────────────────┘
```

### Inference (per MCTS leaf evaluation)

```
Board ──▶ board_to_tensor() ──▶ GomokuNet.forward() ──▶ (log_policy, value)
                                      │
                              ┌───────┴───────┐
                        policy_to_move_probs()   float(value)
                              │
                        [(move, prob), ...]
```

### MCTS Search

```
┌──────────────────────────────────────────────────┐
│  root = MCTSNode()                               │
│                                                  │
│  for _ in range(num_simulations):                │
│    1. Select: traverse tree via PUCT             │
│       a* = argmax [Q + c_puct * P * √N / (1+n)] │
│    2. Expand: if leaf not terminal, call         │
│       wrapper.evaluate(board) → (priors, value)  │
│       Create child MCTSNode per legal move       │
│    3. Backup: negate value up the path,          │
│       update visit_count and total_value         │
│                                                  │
│  Threat short-circuit: if forced win/block       │
│  exists, skip neural evaluation entirely         │
└──────────────────────────────────────────────────┘
```
## Batched Neural Evaluation

The `BatchedLeafEvaluator` (`selfplay/evaluator.py`) replaces direct
`wrapper.batch_evaluate()` calls in the MCTS loop.  It provides three
optimisations over the naive per-board evaluation path:

### 1. Batch tensor construction

Instead of converting each board independently (numpy → torch → unsqueeze → cat → device),
the evaluator pre-allocates a single `(B, 3, 15, 15)` float32 NumPy array and fills
each board's channels via boolean indexing into the board grid.  One `torch.from_numpy()`
and one `.to(device)` suffices for the entire batch.

### 2. Batched post-processing

- **Values**: extracted with a single `.tolist()` call — one CPU–GPU synchronisation
  instead of one per board.
- **Policies**: `torch.exp()` computed on-device, transferred to CPU in one `.cpu().numpy()`
  call, then per-board legal-move filtering in NumPy.

### 3. CUDA kernel warmup

A dummy forward pass at construction time initialises CuDNN heuristics and kernel
launch infrastructure so the first real batch does not pay cold-start latency.

### Architecture tradeoffs

| Decision | Rationale |
|----------|-----------|
| Larger batch size (default 32, was 8) | Underutilised GPU on small batches was the primary bottleneck.  Batch=32 exercises GPU compute units substantially better while adding only moderate virtual-loss pressure. |
| Synchronous evaluation (no async accumulation) | Async accumulation across MCTS iterations would change search semantics.  Current design keeps zero-overhead semantics: MCTS descends, evaluates, backs up — no pending evaluations. |
| Parallel descent via virtual loss | Virtual loss (1 per in-flight path) steers descents toward diverse branches naturally as batch size grows.  With batch ≤ 128 the distortion is negligible. |
| Per-board legal-move filtering on CPU | Legal moves are board-structure-dependent; filtering in NumPy after the GPU transfer is simpler and fast enough (~0.01 ms per board). |

### Remaining bottlenecks (post-batching)

1. **Board copying during descent** — each leaf requires a full `Board.copy()` call.
   For batch_size=32 this is 32 copies per iteration.  A copy-on-write or
   reference-counted board state would reduce overhead but adds complexity.

2. **Serial undo rewinding** — after each descent, the virtual board is rewound
   move-by-move.  This is O(depth) per descent and grows with search depth.
   A board state stack (push/pop) would be O(1).

3. **Python overhead in the hot loop** — the inner MCTS loop makes many Python-level
   calls (`_puct_select`, `make_move`, `undo_move`).  These are individually fast but
   cumulatively significant at 400+ simulations.

### Future scaling opportunities

- **CUDA graphs**: capture the forward pass as a CUDA graph for zero-overhead kernel launch.
  Particularly beneficial once batch sizes stabilise.
- **Torch compile**: `torch.compile` the model forward pass for fused kernel execution.
- **Multi-stream evaluation**: overlap data transfer (H2D) with MCTS descent using
  CUDA streams (advanced, adds complexity).
- **Dynamic batching**: if self-play produces many terminal leaves (few non-terminal
  evaluations per batch), pad or reschedule to maintain batch utilisation.

## Key Design Decisions

### Checkpoint Format

`save_model_checkpoint()` saves raw `model.state_dict()` only — no optimizer
state, no epoch metadata. This keeps checkpoints compatible with
`GomokuInferenceWrapper`, which expects a plain state dict.

### Replay Buffer Serialization

The buffer uses its own `state_dict()` / `from_state_dict()` API rather than
raw pickling. This decouples serialization from internal representation
(`deque` vs `list`) and follows the pattern established by PyTorch modules.

### Symmetry Augmentation

All 8 D₄ dihedral symmetries (identity, 3 rotations, horizontal flip, and
flip + 3 rotations) are applied to both board states and policy targets during
self-play. This provides 8× effective data without additional games. Value
targets are invariant under symmetry.

### Temperature Annealing

First 15 moves use temperature > 0 (stochastic sampling from visit
distribution) to encourage opening diversity. After move 15, temperature = 0
(argmax) so the AI plays its strongest line in critical positions.

### Evaluation Protocol

Deterministic games (temperature = 0) with each model playing Black in half
the games to cancel first-move advantage. 55% win rate threshold for promotion
— balances confidence with iteration speed.

## MCTS Hot Path

The MCTS selection formula at internal nodes:

```
a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * √(Σ_b N(s,b)) / (1 + N(s,a)) ]
```

Where:
- `Q(s,a)` = mean action value (total_value / visit_count)
- `P(s,a)` = prior probability from policy head
- `N(s,a)` = visit count for this edge
- `c_puct` = exploration constant (default 2.5)

Value backup negates at each level (win for opponent = loss for current player).

## Test Coverage

| File | Tests | Focus |
|------|-------|-------|
| `test_encoding.py` | 7 | board_to_tensor, policy_to_move_probs |
| `test_mcts.py` | 18 | search distribution, move selection, win/block detection, search_with_stats |
| `test_neural.py` | 6 | model output shapes, value range, forward pass |
| `test_selfplay.py` | 30 | symmetries, TrainingExample, SelfPlayGame, ReplayBuffer |
| `test_threats.py` | 25 | threat types, detection, edge cases, scoring |
| `test_train.py` | 9 | loss functions, checkpoint roundtrip, training, eval, integration |
| `test_saliency.py` | 29 | completeness axiom, shapes, ranges, methods, targets, symmetry, IG quality |
| `test_activations.py` | 13 | hook lifecycle, shapes, cleanup, channel selection, idempotency |
| `test_comparison.py` | 24 | move ranking, values, serialization, edge cases, threat override |
| **Total** | **161** | |

## Roadmap

### Phase 1 — Core Training ✅
- [x] `selfplay/selfplay.py` — self-play game generation
- [x] `selfplay/replay_buffer.py` — replay buffer with symmetry augmentation
- [x] `selfplay/train.py` — training loop with loss computation and checkpointing
- [x] Model evaluation — pit new model vs best model

### Phase 2 — Performance ✅
- [x] Batched neural inference for multiple MCTS leaf evaluations
- [x] GPU-accelerated MCTS (batch evaluation on GPU)
- [x] Virtual loss for parallel MCTS within a single search
- [x] Profile and optimize hot path (Profiler, optimized tensor pipeline)
- [ ] CUDA graphs for zero-overhead model invocation
- [ ] `torch.compile` for fused kernel execution

### Phase 3 — Scaling
- [ ] Distributed self-play (multiple workers, central trainer)
- [ ] Stronger architectures (SE blocks, attention, deeper resnets)
- [ ] Support for larger board sizes

### Phase 4 — Explainability ✅
- [x] Saliency maps
- [x] Activation visualization
- [x] Human vs AI move comparison tool

### Phase 5 — UI and Visualization
- [ ] Web-based game UI
- [ ] Search tree visualization
- [ ] Policy heatmap overlay
- [ ] Value landscape visualization
