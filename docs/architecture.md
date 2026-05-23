# Gomoku AI — Architecture Document

## Project Overview

Hybrid Gomoku (15×15, 5-in-a-row) engine combining Monte Carlo Tree Search with
deep neural networks for position evaluation and policy guidance, following the
AlphaZero paradigm. Built with PyTorch and NumPy.

## Directory Structure

```
gomoku-ai/
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
├── tests/                          # 84 tests across 6 files
│   ├── test_encoding.py            #   7 tests
│   ├── test_mcts.py                #   7 tests
│   ├── test_neural.py              #   6 tests
│   ├── test_selfplay.py            #   30 tests
│   ├── test_threats.py             #   25 tests
│   └── test_train.py               #   9 tests
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
selfplay/     ──┐
                │  may import from engine/ and neural/
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
| | `MCTS` | PUCT search, `search(board)` → visit distribution, `select_move(board, temp)` |
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

## Data Flow

### Training Loop (AlphaZero cycle)

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
| `test_mcts.py` | 7 | search distribution, move selection, win/block detection |
| `test_neural.py` | 6 | model output shapes, value range, forward pass |
| `test_selfplay.py` | 30 | symmetries, TrainingExample, SelfPlayGame, ReplayBuffer |
| `test_threats.py` | 25 | threat types, detection, edge cases, scoring |
| `test_train.py` | 9 | loss functions, checkpoint roundtrip, training, eval, integration |
| **Total** | **84** | |

## Roadmap

### Phase 1 — Core Training ✅
- [x] `selfplay/selfplay.py` — self-play game generation
- [x] `selfplay/replay_buffer.py` — replay buffer with symmetry augmentation
- [x] `selfplay/train.py` — training loop with loss computation and checkpointing
- [x] Model evaluation — pit new model vs best model

### Phase 2 — Performance
- [ ] Batched neural inference for multiple MCTS leaf evaluations
- [ ] GPU-accelerated MCTS
- [ ] Virtual loss for parallel MCTS within a single search
- [ ] Profile and optimize hot path

### Phase 3 — Scaling
- [ ] Distributed self-play (multiple workers, central trainer)
- [ ] Stronger architectures (SE blocks, attention, deeper resnets)
- [ ] Support for larger board sizes

### Phase 4 — Explainability
- [ ] Saliency maps
- [ ] Activation visualization
- [ ] Human vs AI move comparison tool

### Phase 5 — UI and Visualization
- [ ] Web-based game UI
- [ ] Search tree visualization
- [ ] Policy heatmap overlay
- [ ] Value landscape visualization
