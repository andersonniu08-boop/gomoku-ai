# Training Run Summary

## Overview

NeuralGomoku was trained using self-play reinforcement learning on a dual-GPU
setup spanning an NVIDIA A100-80GB cloud instance (RunPod) and a local NVIDIA
RTX 3050.

## Hardware Configuration

| Component | Details |
|---|---|
| **Primary GPU** | NVIDIA A100 80GB (RunPod cloud instance) |
| **Secondary GPU** | NVIDIA RTX 3050 8GB (local workstation) |
| **Model architecture** | 10-block residual CNN, 128 channels, ~3.7M parameters |
| **Checkpoint size** | ~15 MiB |
| **Test environment** | 440 tests, 1 known flaky |

## Training Setup

### Self-Play Generation

- MCTS-guided game generation with 64–800 simulations per move
- Temperature annealing: stochastic (T>0) for first 15 moves, greedy (T=0)
  thereafter to ensure strongest play in critical positions
- Dirichlet noise at root (alpha=0.03) for opening diversity
- Resignation heuristics prevent wasted compute on clearly lost games

### Replay Buffer

- 500,000 capacity FIFO buffer
- D₄ dihedral symmetry augmentation on retrieval (8× effective data from
  rotations and reflections)
- Persisted to `data/replay_buffer.pt`

### Training Loop

- Cross-entropy policy loss + MSE value loss (equal weight)
- Adam optimizer with CosineAnnealingLR schedule
- Mixed-precision training (AMP GradScaler)
- Gradient clipping (max_norm=1.0)
- Batch size: 256

### Evaluation

- 100-game matches: latest model vs best model
- Alternating colors (each model plays Black 50 times)
- Promotion threshold: 55% win rate
- Elo tracking with K=96

## Training Metrics (6 Iterations)

| Iteration | Loss | Policy Loss | Value Loss | Buffer Size | Runtime |
|---|---|---|---|---|---|
| 1 | 4.32 | 3.85 | 0.47 | 618 | 109s |
| 2 | 3.31 | 3.10 | 0.21 | 1,484 | 371s |
| 3 | 3.29 | 3.02 | 0.26 | 2,203 | 178s |
| 4 | 3.14 | 2.93 | 0.21 | 2,967 | 173s |
| 5 | 3.01 | 2.85 | 0.16 | 3,867 | 199s |
| 6 | 2.94 | 2.83 | 0.11 | 4,767 | 293s |

Training loss decreased from 4.32 to 2.94 over 6 iterations. Policy loss
convergence (3.85 → 2.83) indicates the network learned to predict MCTS visit
distributions. Value loss convergence (0.47 → 0.11) indicates improving outcome
prediction accuracy.

## Distributed Worker System

The training system uses file-based coordination for multi-machine scaling:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Trainer   │────▶│ checkpoints/  │◀────│   Workers   │
│ (local GPU) │     │  latest.pt   │     │ (A100 GPU)  │
└──────┬──────┘     └──────────────┘     └──────┬──────┘
       │                                        │
       │  ingests game files                    │  generates games
       ▼                                        │
┌─────────────┐                          ┌──────▼──────┐
│   Replay    │◀─────────────────────────│game_*.pt    │
│   Buffer    │     file system           │ files       │
└─────────────┘                          └─────────────┘
```

- Workers poll `checkpoints/latest.pt` for model updates
- Each worker generates games independently
- Trainer ingests game files into the replay buffer after each iteration
- Consumed game files are archived to `game_examples/consumed/`

## Checkpoint Lifecycle

```
Initial ──▶ latest.pt (iter_001) ──▶ ... ──▶ latest.pt (iter_N)
                                                  │
                                     evaluation: win rate ≥ 55%?
                                      ┌──── YES ────┐
                                      ▼              ▼
                                 best.pt      keep training
                              (promoted)
```

## Major Debugging Issues Resolved

### 1. Training Hang (resolved)
Training would hang during self-play generation after certain moves. Root
cause: the MCTS sometimes explored moves that the move ordering system hadn't
included in candidate pruning, causing an infinite descent loop. Fixed by
ensuring all legal moves are always discoverable during tree search.

### 2. Batched Evaluation Memory Leak (resolved)
The `BatchedLeafEvaluator` pre-allocated CUDA tensors that were not freed
between MCTS searches, causing VRAM accumulation over long training runs. Fixed
by adding explicit tensor cleanup between searches.

### 3. Legal Move Filtering Mismatch (resolved)
The policy head's legal move filtering was incorrectly masking moves adjacent
to opponent stones on the board edge. Fixed by correcting the neighbor-offset
calculation in `engine/board.py`.

### 4. Checkpoint Architecture Mismatch (resolved)
The UI server would crash when loading a checkpoint trained with a different
network architecture (e.g., different block count or channel size). The
`GomokuInferenceWrapper` now validates architecture compatibility at load time
and reports a clear error rather than crashing with an opaque tensor shape
mismatch.

### 5. Self-Play Resignation False Positives (resolved)
Early resignation heuristics were too aggressive, causing the AI to resign
in drawn or defensible positions. The threshold was adjusted from -0.7 to -0.9
and a minimum-move-count guard was added (no resignation before move 30).

### 6. Checkpoint Atomicity (resolved)
Checkpoint writes were not atomic — a crash during `torch.save()` could
corrupt the checkpoint, leaving the training loop unable to resume. Fixed by
writing to a `.tmp` file first, then atomically renaming.

### 7. Zobrist Cache Collisions (resolved)
The Zobrist hash evaluation cache was using 32-bit keys, causing occasional
collisions that returned wrong cached evaluations. Upgraded to 64-bit keys.

## Checkpoint Organization

```
checkpoints/
├── best.pt              ← Strongest promoted model (used for gameplay)
└── latest.pt            ← Most recently trained model
```

The trained checkpoint is intentionally included in the repository for
demonstration purposes. Larger production ML projects typically store model
artifacts outside the primary source repository (e.g., S3, GCS, or a model
registry).

## Future Improvements

- **Deeper networks:** 20+ blocks with residual scaling for stronger play
- **CUDA graphs:** Capture forward pass as CUDA graph for zero-overhead
  kernel launch in MCTS
- **torch.compile:** Fused kernel execution for model forward pass
- **Cloud orchestration:** Automate RunPod worker lifecycle instead of manual
  SSH+rsync
- **WebSocket UI:** Non-blocking UI with streaming search visualization
- **Cloud artifact storage:** S3/GCS for checkpoints and replay buffer
