# NeuralGomoku (神经五子棋)

A complete neural-network-powered Gomoku engine with Monte Carlo Tree Search,
self-play reinforcement learning, and distributed training infrastructure.
Built with PyTorch.

## Overview

NeuralGomoku plays Gomoku (15×15, five-in-a-row) using a dual-headed residual
convolutional neural network that evaluates positions and guides a Monte Carlo
Tree Search. The engine was trained entirely through self-play — it generates
its own training data by playing against itself, stores experiences in a replay
buffer, and periodically evaluates new models against the previous best to
decide whether to promote.

The project includes a Flask-based web UI where you can play against the
trained model, a distributed worker system for scaling self-play across
multiple machines, and a full suite of 440 tests.

## Features

- **Dual-headed residual CNN** — 10 residual blocks, 128 channels, ~3.7M
  parameters with multi-head self-attention, squeeze-and-excitation channel
  gating, CBAM spatial attention, and stochastic depth regularization
- **Monte Carlo Tree Search** — PUCT selection with batched GPU leaf
  evaluation, virtual loss for parallel descent, tree reuse across moves, and
  tactical threat-override short-circuit
- **Self-play training pipeline** — MCTS-guided game generation with
  temperature annealing, Dirichlet exploration noise, and resignation
  heuristics; 500K-capacity FIFO replay buffer with D₄ symmetry augmentation
- **Distributed workers** — File-based coordination allowing workers on
  separate machines (including cloud GPUs) to generate self-play games while a
  central trainer consumes them
- **Web UI** — Canvas-rendered board with heatmap overlay (neural-guided search
  priorities), MCTS search tree panel, and strength presets (Fast/Medium/Strong)
- **Automatic evaluation** — 100-game matches between new and best model,
  Elo tracking, win-rate-based promotion at 55% threshold
- **Checkpoint system** — Atomic saves (write `.tmp` → rename), training
  resume, architecture-compatibility validation

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Web UI (Flask)                        │
│  Browser │ Canvas board │ Search tree panel │ Strength presets │
└──────────────────────────┬───────────────────────────────────┘
                           │ /api/search  /api/new-game
┌──────────────────────────▼───────────────────────────────────┐
│                      MCTS Search Engine                       │
│  PUCT selection │ Batched GPU eval │ Tree reuse │ Threat override │
└──────────────────────────┬───────────────────────────────────┘
                           │ evaluate(board) → (policy, value)
┌──────────────────────────▼───────────────────────────────────┐
│                   Neural Network (PyTorch)                    │
│  Dual-headed CNN │ 10 blocks │ Attention │ SE/CBAM │ DropPath │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                     Game Engine (NumPy)                       │
│  15×15 Board │ Win detection │ Threat analysis │ Move encoding │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  Training Pipeline                            │
│                                                              │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐           │
│  │ Workers │───▶│ game_*.pt │───▶│    Trainer   │           │
│  │ (A100)  │    │   files   │    │ (checkpoints) │           │
│  └─────────┘    └───────────┘    └──────┬───────┘           │
│                                         │                    │
│                    ┌────────────────────┼────────────┐       │
│                    ▼                    ▼            ▼       │
│             Self-Play Games     Replay Buffer    Training    │
│             (MCTS-guided)       (500K FIFO)     (Adam+CLR)  │
│                                         │                    │
│                                    ┌────▼─────┐             │
│                                    │ Evaluate │             │
│                                    │ 100 games│             │
│                                    └────┬─────┘             │
│                                    win ≥ 55%?               │
│                                    YES → promote            │
└──────────────────────────────────────────────────────────────┘
```

Layers are strictly separated. `engine/` (NumPy, zero framework deps) →
`neural/` (PyTorch only) → `selfplay/` (search + training). Higher layers
may import from lower layers; the reverse is never allowed.

## Neural Network

**File:** `neural/model.py`

A dual-headed residual convolutional neural network:

| Component | Detail |
|---|---|
| Input | `(B, 3, 15, 15)` — current player stones, opponent stones, turn indicator |
| Stem | Conv2d(3→128, 3×3) → BatchNorm → ReLU |
| Residual blocks | 10 × SEResidualBlock(128) with dilated pyramid 1→2→3→2→1 |
| Self-attention | 4-head scaled dot-product attention with LayerNorm |
| Channel gating | Squeeze-and-Excitation (reduction=8) |
| Spatial attention | CBAM with 3×3 kernel |
| Regularization | DropPath stochastic depth (0.0→0.1 linear schedule) |
| Policy head | Conv2d(128→32, 3×3) → BN → ReLU → Conv2d(32→1, 1×1) → Flatten → LogSoftmax |
| Value head | Dual global pooling (avg+max) → FC(256→64) → ReLU → FC(64→1) → Tanh |
| Parameters | ~3.7M |

Policy output: log-softmax over 225 board cells. Value output: tanh scalar
in [-1, +1] estimating the expected outcome from the current player's
perspective.

**File:** `neural/wrapper.py` — The `GomokuInferenceWrapper` handles
checkpoint loading, device placement (CUDA/CPU auto-detection), and
board-to-tensor conversion. It validates architecture compatibility at load
time and provides `evaluate(board)` returning `(move_probs, value)`.

**File:** `engine/encoding.py` — `board_to_tensor()` converts a Board into
the 3-channel tensor format. `policy_to_move_probs()` filters network output
to only legal moves.

## Monte Carlo Tree Search

**File:** `selfplay/mcts.py`

The MCTS implementation uses the PUCT (Predictor + Upper Confidence Bound
applied to Trees) formula for node selection:

```
a* = argmaxₐ [ Q(s,a) + c_puct × P(s,a) × √(Σ_b N(s,b)) / (1 + N(s,a)) ]
```

Where `Q(s,a)` is the mean action value, `P(s,a)` is the prior probability
from the neural network, `N(s,a)` is the visit count, and `c_puct` is an
exploration constant (default 2.5).

### Key features

- **Batched leaf evaluation** — `BatchedLeafEvaluator` (`selfplay/evaluator.py`)
  collects multiple leaf boards into a single GPU batch, runs one forward pass,
  and extracts values with a single `.tolist()` call — avoiding per-board
  CPU-GPU synchronization overhead
- **Virtual loss** — During parallel descent, each in-progress path
  temporarily subtracts a virtual loss from Q values, steering concurrent
  descents toward different branches
- **Tree reuse** — Between consecutive moves in a game, the previous search
  tree is re-rooted at the played move, retaining accumulated statistics
- **Threat override** — If a forced win or must-block exists (detected by
  `ThreatDetector` in `engine/threats.py`), neural evaluation is skipped
  and the tactical response is played immediately
- **Move ordering** — `selfplay/move_ordering.py` prunes candidate moves
  using incremental line scanning, prioritizing critical tactical responses
  and filtering low-probability neural priors

### Configurable presets

| Preset | Simulations | Use case |
|---|---|---|
| Fast | 200 | Quick casual play |
| Medium | 800 | Balanced strength and speed |
| Strong | 2000 | Strongest tactical play |

## Self-Play Training Pipeline

**File:** `selfplay/train.py`

The training loop runs in two-phase iterations:

### Phase A — Self-Play Generation

1. The current best model plays games against itself using MCTS (800 sims/move)
2. Temperature annealing: moves 1–15 use stochastic sampling from the visit
   distribution for opening diversity; move 16+ uses greedy selection
3. Dirichlet noise (α=0.03) is added to root priors during self-play to
   encourage exploration
4. Resignation: if the root value drops below −0.9 (after 30+ moves), the
   game is terminated early
5. Each game produces `TrainingExample` triples: `(board_state, policy_target, value_target)`
6. D₄ symmetry augmentation is applied — each position is stored with all 8
   rotations and reflections, giving 8× effective data

### Phase B — Training

1. Mini-batches (size 256) are sampled from the replay buffer
2. Loss = cross-entropy (policy) + MSE (value), equal weight
3. Adam optimizer with CosineAnnealingLR scheduler
4. Mixed-precision training via AMP `GradScaler`
5. Gradient clipping at max_norm=5.0

### Evaluation and Promotion

Every 5 iterations, the latest model plays a 100-game match against the
current best model (both models play Black 50 times, temperature=0 for
deterministic play). If the new model wins ≥55% of games, it replaces
the best model. Checkpoints are saved atomically (`.tmp` → rename).

**File:** `selfplay/replay_buffer.py` — 500K capacity FIFO buffer with
`state_dict()`/`from_state_dict()` serialization for persistence.

**File:** `selfplay/elo.py` — Elo tracker with K=96, persisted to
`data/elo_state.json`.

## Distributed Worker System

**File:** `selfplay/worker.py`

The worker system enables horizontal scaling of self-play generation:

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Trainer    │────▶│ checkpoints/  │◀────│   Worker 1   │
│              │     │  latest.pt   │     │  (machine 1) │
└──────┬───────┘     └───────────────┘     └──────┬───────┘
       │                                          │
       │  polls game_examples/                    │  generates games
       ▼                                          ▼
┌──────────────┐                          ┌──────────────┐
│    Replay    │◀─────────────────────────│  game_*.pt   │
│    Buffer    │     file system           │    files     │
└──────────────┘                          └──────────────┘
```

- Workers poll `checkpoints/latest.pt` for new model versions
- Each worker generates games independently and writes `game_*.pt` files
  to `game_examples/`
- The trainer ingests game files into the replay buffer each iteration
- Consumed files are archived to `game_examples/consumed/`
- Workers handle SIGINT/SIGTERM for graceful shutdown

This design was used to run workers on an NVIDIA A100-80GB cloud instance
(RunPod) while the trainer ran locally. File-based coordination requires no
orchestration layer — the filesystem serves as the coordination primitive.

## Training Results

Training was run for 6 iterations with 800 MCTS simulations per move during
self-play. Training metrics from `data/training_log.csv`:

| Iteration | Total Loss | Policy Loss | Value Loss | Buffer Size | Runtime |
|---|---|---|---|---|---|
| 1 | 4.32 | 3.85 | 0.47 | 618 | 109s |
| 2 | 3.31 | 3.10 | 0.21 | 1,484 | 371s |
| 3 | 3.29 | 3.02 | 0.26 | 2,203 | 178s |
| 4 | 3.14 | 2.93 | 0.21 | 2,967 | 173s |
| 5 | 3.01 | 2.85 | 0.16 | 3,867 | 199s |
| 6 | 2.94 | 2.83 | 0.11 | 4,767 | 293s |

Both policy and value losses decreased steadily across iterations. The first
evaluation match (iteration 5) ended in a 50% draw between latest and best.
Further training iterations would improve strength — professional-strength
Gomoku engines typically require 100+ iterations with larger batch sizes.

**Test suite:** 440 tests across 9+ files covering board logic, threat
detection, neural network output shapes, MCTS search, self-play generation,
replay buffer, training losses, saliency maps, activation visualization, and
move comparison.

## Trained Model

A trained checkpoint from the 6-iteration run is included at
`checkpoints/best.pt` and `checkpoints/latest.pt` (~15 MiB each). The web UI
automatically loads `checkpoints/best.pt` on startup. Training can resume from
the checkpoint using `python -m selfplay.train`.

> This repository intentionally includes a trained checkpoint for
> demonstration purposes. Larger production ML systems typically store model
> artifacts outside the primary source repository using artifact storage,
> model registries, releases, or dedicated model hosting solutions.

## Running the Project

```bash
pip install -r requirements.txt

# Play against the trained model in your browser
python -m ui.server
# Open http://localhost:5000

# Run the full test suite
python -m pytest tests/ -v --timeout=60

# Resume training from the checkpoint
python -m selfplay.train --num-iterations 50 --games-per-iteration 10

# Start a distributed worker (on any machine with the repo)
python -m selfplay.worker

# Run benchmarks
python -m tools.benchmark_runner
```

## Repository Structure

```
neural-gomoku/
├── engine/                  Game logic (NumPy, zero framework deps)
│   ├── board.py             Board state, moves, win detection
│   ├── threats.py           Pattern-based threat detection
│   ├── encoding.py          Board ↔ tensor conversion
│   └── tactical.py          Deterministic forced-line solver
├── neural/                  PyTorch model
│   ├── model.py             Dual-headed residual CNN
│   └── wrapper.py           Checkpoint loading, inference interface
├── selfplay/                Search and training
│   ├── mcts.py              MCTS with PUCT, batched evaluation
│   ├── evaluator.py         Batched GPU leaf evaluator
│   ├── move_ordering.py     Tactical move ordering and pruning
│   ├── selfplay.py          Self-play game generation
│   ├── replay_buffer.py     500K FIFO replay buffer
│   ├── train.py             Training loop, evaluation, promotion
│   ├── worker.py            Distributed self-play worker
│   ├── elo.py               Elo rating tracker
│   ├── eval_registry.py     Evaluation coordination
│   ├── config.py            Strength presets
│   ├── bench_suite.py       Tactical benchmark suite
│   └── profiler.py          Hierarchical profiler
├── ui/                      Web interface
│   ├── server.py            Flask API server
│   └── static/              HTML, CSS, JavaScript frontend
├── explain/                 Model explainability
│   ├── saliency.py          Gradient-based attribution maps
│   ├── activations.py       Forward-hook activation capture
│   └── comparison.py        Human vs AI move comparison
├── tools/                   CLI utilities
├── checkpoints/             Trained model weights
├── data/                    Training artifacts (replay buffer, logs)
├── tests/                   440 tests
├── docs/                    Additional documentation
└── benchmarks/              Benchmark results
```

## Engineering Highlights

- **End-to-end ML pipeline** — From board representation to neural network to
  search to self-play to training to evaluation to web serving, every
  component is implemented and tested
- **GPU-optimized MCTS** — Batched leaf evaluation with pre-allocated tensors,
  single `.tolist()` synchronization, and CUDA kernel warmup avoids the
  per-board overhead that bottlenecks naive implementations
- **Distributed self-play** — File-based coordination between workers and
  trainer enables horizontal scaling without an orchestration layer; workers
  ran on an A100-80GB cloud GPU
- **Modern CNN architecture** — Multi-head attention, squeeze-and-excitation
  channel gating, CBAM spatial attention, dilated convolution pyramid, and
  DropPath stochastic depth — all standard in production vision models
- **Training infrastructure** — Mixed-precision training (AMP), gradient
  clipping, cosine annealing, checkpoint atomicity, training resume, Elo
  tracking, and evaluation match system
- **Tactical correctness guarantee** — Threat detection with forced-win and
  must-block recognition short-circuits neural search for tactical positions
- **Modular design** — Strict layer separation with unidirectional imports;
  every component is independently testable and swappable
- **Comprehensive test suite** — 440 tests covering all major subsystems
