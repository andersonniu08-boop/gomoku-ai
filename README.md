# NeuralGomoku

Deep neural network engine for Gomoku (15×15, 5-in-a-row) combining Monte Carlo
Tree Search with PyTorch. Trained via self-play reinforcement learning on an
NVIDIA A100-80GB GPU.

## Overview

NeuralGomoku is an end-to-end ML systems project demonstrating reinforcement
learning at scale. A dual-headed residual CNN evaluates board positions and
guides MCTS search, which generates training data through self-play. A
distributed worker architecture enables horizontal scaling across multiple
machines.

Every component — board logic, neural network, MCTS, self-play, training loop,
and web UI — is modular, tested, and independently swappable.

## Architecture

```
neural-gomoku/
├── engine/          Game logic — Board, win detection, threat analysis
├── neural/          PyTorch model — dual-headed CNN, inference wrapper
├── selfplay/        MCTS search, self-play games, replay buffer, training loop
├── explain/         Saliency maps, activation visualization, move comparison
├── tools/           Benchmarking, regression testing, validation
├── ui/              Flask web server, Canvas board renderer
├── tests/           161 tests across 9 files
├── checkpoints/     Trained model weights
└── data/            Training data and replay buffer
```

Imports flow strictly downward: `engine` ← `neural` ← `selfplay`. Lower layers
never import from higher layers. Zero circular dependencies.

## Neural Network

Dual-headed residual CNN with modern architectural features:

| Feature | Detail |
|---------|--------|
| Architecture | 10 residual blocks, 128 channels, ~3.7M parameters |
| Attention | 4-head self-attention with LayerNorm |
| Channel gating | Squeeze-and-Excitation (reduction=8) |
| Spatial attention | CBAM with 3×3 kernel |
| Dilated pyramid | Multi-scale 1→2→3→2→1 dilation |
| Regularization | DropPath stochastic depth (0→0.1 linear schedule) |
| Policy head | Fully convolutional 3×3 Conv → 1×1 → LogSoftmax over 225 cells |
| Value head | Dual global pooling (avg+max) → FC(256→64) → FC(64→1) → Tanh |

Input: `(batch, 3, 15, 15)` — current player stones, opponent stones, turn
indicator. Output: log-policy over board cells and value scalar in [-1, 1].

## MCTS Search

Monte Carlo Tree Search with PUCT selection:

```
a* = argmaxₐ [ Q(s,a) + cᵖᵘᶜᵗ · P(s,a) · √(ΣN(s,b)) / (1 + N(s,a)) ]
```

Key features:

- **Batched leaf evaluation** — GPU-efficient inference across multiple leaves
  with single forward pass, virtual loss for parallel descent
- **Tree reuse** — Previous search tree re-rooted across consecutive moves
- **Threat override** — Forced wins and blocks short-circuit neural evaluation
  for guaranteed tactical correctness
- **Configurable simulations** — Fast (200), Medium (800), Strong (2000)

Value backup negates at each tree level. Win for current player = +1.0, loss =
-1.0, draw = 0.0.

## Self-Play Training

The training loop follows the self-play reinforcement learning cycle:

1. **Self-play generation** — MCTS-guided games with Dirichlet exploration noise
   at root, temperature annealing (moves 0-15 stochastic, 15+ greedy)
2. **Replay buffer** — 500K capacity FIFO buffer with D₄ symmetry augmentation
   (8× effective data via rotations and reflections)
3. **Training** — Mini-batch sampling (256), cross-entropy policy loss + MSE
   value loss, Adam optimizer with CosineAnnealingLR
4. **Evaluation** — 100-game matches between latest and best model
   (alternating colors, temperature=0). Promotion at 55% win rate

Games use 800 MCTS simulations per move during self-play, 200 during evaluation.

## Distributed Workers

Multi-machine self-play scaling via file-based coordination:

- Workers poll `checkpoints/latest.pt` for model updates
- Each worker generates games independently and writes `game_*.pt` files
- Central trainer ingests game files into the replay buffer
- Graceful SIGINT/SIGTERM handling for clean shutdown
- No orchestration layer required — file system serves as coordination

## Trained Checkpoint

A trained checkpoint from a 25+ iteration run on NVIDIA A100-80GB (RunPod) is
included at `checkpoints/best.pt` and `checkpoints/latest.pt` (14.2 MiB).

The checkpoint is intentionally included for demonstration purposes. Larger
production ML projects typically store model artifacts outside the primary
source repository.

## Results

| Metric | Value |
|--------|-------|
| Training iterations | 25+ (promoted checkpoints through iter_025) |
| Training hardware | NVIDIA A100-80GB (RunPod) + local RTX 3050 |
| Model parameters | 3.7M |
| Test coverage | 161 tests, 1 known flaky |
| MCTS performance | ~3-4× throughput improvement via batched evaluation |
| Web UI | Playable against trained model in browser |

## Running the Project

```bash
pip install -r requirements.txt

# Play against the trained model in your browser
python -m ui.server                          # Open http://localhost:5000

# Run the test suite
python -m pytest tests/ -v --timeout=60

# Run benchmarks
python -m tools.benchmark_runner

# Train from scratch (requires checkpoints/ directory)
python -m selfplay.train

# Distributed: start workers on additional machines
python -m selfplay.worker
```

## Documentation

- [Architecture](docs/architecture.md) — Detailed layer design, data flow, key
  design decisions
- [Project Status](docs/project_status.md) — Component completion, limitations
- [Training Summary](TRAINING_RUN_SUMMARY.md) — A100 training run details
- [AGENTS.md](AGENTS.md) — Development rules, standards, and conventions
