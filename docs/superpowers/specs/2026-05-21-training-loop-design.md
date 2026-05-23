# AlphaZero Training Loop — Design Spec

**Date:** 2026-05-21
**Status:** approved

## Scope

Complete Phase 1 of the Gomoku AI roadmap:
1. `selfplay/train.py` — training loop with loss computation, checkpointing, LR scheduling
2. Model evaluation — pit new model vs best model to decide promotion

## Architecture

Three modules inside `selfplay/train.py` plus eval:

```
selfplay/
  train.py          # Loss computation, training loop, evaluation matches
```

### Two-Phase Iteration

Each training iteration runs:

**Phase A — Self-play:**
- Create MCTS game runner using current best model (`checkpoints/best.pt`)
- Play N games, adding all examples to ReplayBuffer
- Save buffer state to disk

**Phase B — Training:**
- Load model from latest checkpoint
- Shuffle all examples from this iteration's self-play games
- Train on them in mini-batches (batch size 256)
- Cosine-anneal LR from 0.001 to 0 over the training steps
- Save `checkpoints/latest.pt`

### Evaluation (every 5 iterations)

- Pit `latest.pt` vs `best.pt` in 100 games
- Both sides use `temperature_threshold=0` (deterministic)
- Alternate which model plays Black to offset first-move advantage
- If `latest` wins ≥55%: promote to `best.pt`
- Otherwise `best.pt` remains unchanged

## Losses

- **Policy loss:** cross-entropy `-(target * log_policy).sum(dim=1).mean()`
  where target is the MCTS visit-count distribution and log_policy is the network's log-softmax output
- **Value loss:** `F.mse_loss(value, target_value)` where target_value is the game outcome
- **Total:** `policy_loss + value_loss` (equal weight, standard AlphaZero)

## Checkpoints

| Path | Purpose |
|------|---------|
| `checkpoints/best.pt` | Strongest model — used for self-play generation |
| `checkpoints/latest.pt` | Most recently trained — candidate for promotion |

## Key Defaults

| Parameter | Default |
|-----------|---------|
| Games per iteration | 10 |
| Batch size | 256 |
| Optimizer | Adam, lr=0.001 |
| LR schedule | Cosine annealing to 0 over training steps |
| Eval frequency | Every 5 iterations |
| Eval games | 100 |
| Promotion threshold | 55% win rate |
| Buffer path | `data/replay_buffer.pt` |

## File Responsibilities

### `selfplay/train.py`

- `compute_loss(log_policy, value, target_policy, target_value)` — pure function returning (policy_loss, value_loss, total_loss)
- `run_evaluation(new_ckpt, best_ckpt, num_games, device)` — plays 100 deterministic games, returns win rate
- `run_training_epoch(model, optimizer, examples, batch_size)` — single pass over examples
- `main()` — top-level orchestration loop

## Non-Responsibilities

- No UI, no web, no pygame
- No modifications to engine/ or neural/ modules
- SelfPlayGame and ReplayBuffer are consumed as-is (already built and tested)

## Testing

- `test_compute_loss_shapes` — verify scalar outputs
- `test_compute_loss_policy_perfect` — loss near zero when prediction matches target
- `test_compute_loss_value_zero_for_perfect_prediction`
- `test_train_one_step_no_crash` — smoke test: train on a small buffer batch
- `test_evaluation_match` — smoke test: run 2 games and verify win rate is between 0 and 1
- `test_checkpoint_save_load_roundtrip` — model saved and reloaded produces same output
