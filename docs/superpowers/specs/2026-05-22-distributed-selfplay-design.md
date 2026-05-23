# Distributed Self-Play — Design Spec

**Date:** 2026-05-22
**Status:** approved

## Overview

Scale AlphaZero self-play by running multiple worker processes that generate
games independently and write them as files. A central trainer polls for new
game files, ingests them into the replay buffer, and trains the model.
Workers periodically reload the latest checkpoint so they always use the
freshest policy.

## Architecture

```
Worker 1 (CPU)    Worker 2 (CPU)    ... Worker N (CPU)
    │ write+rename     │ write+rename         │
    ▼                  ▼                      ▼
┌──────────────────────────────────────────────────┐
│              game_examples/                       │
│  game_w0_0001_20260522T120000.pt                 │
│  game_w0_0001_meta.json                          │
│  game_w1_0001_20260522T120100.pt                 │
│  consumed/   ← trainer archives processed files   │
│    (capped at last 1000)                          │
└──────────────────────┬───────────────────────────┘
                       │ Trainer polls every 2-3s
                       ▼
┌──────────────────────────────────────────────────┐
│                   Trainer                         │
│  1. Glob game_examples/*.pt                       │
│  2. Load each → ReplayBuffer.add_examples()       │
│  3. Move files → consumed/, trim retention cap    │
│  4. Train one pass over recent examples           │
│  5. Evaluate & promote model periodically         │
│  6. Write checkpoints/best_<iter>_<winrate>.pt    │
└──────────────────────────────────────────────────┘
```

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Communication | Shared filesystem | No new dependencies; workers are fully decoupled from trainer |
| File format | `.pt` (torch.save) | Consistent with existing buffer snapshots |
| Atomic writes | Write to `.tmp`, then `os.rename` | Atomic on Linux, trainer ignores `.tmp` files |
| File naming | `game_<worker_id>_<seq>_<timestamp>.pt` | Unique without coordination, sortable, debuggable |
| Dedup | Check `consumed/` for filename already archived | Handles worker restart without hash overhead |
| Model refresh | Poll checkpoint mtime every N seconds | No coordination; workers reload when mtime changes |
| Trainer polling | `time.sleep(2-3)` between globs | Portable, no new dependency; training cycle is minutes so 2s lag is invisible |
| Worker device | CPU-only | Neural inference is microsecond-scale per board; GPU reserved for trainer's batched gradient updates |
| Augmentation | Raw on disk, applied at trainer | Saves 8× disk; buffer stores unique positions not rotations |
| Backup strategy | Trainer ingests then moves to `consumed/` with 1000-file cap | Debuggable history without unbounded disk growth |
| Worker entry point | `run_worker()` function + `argparse` CLI | Importable, testable, invocable as `python -m selfplay.worker` |
| Trainer entry point | Extended `train.main()` with new `game_examples_dir` parameter | No new files for training; existing structure gains polling |
| Distributed sync | `rsync` between machines (ops concern, not code) | Keeps codebase lean; documented in `docs/distributed.md` |

## New file: `selfplay/worker.py`

### Public interface

```python
def run_worker(
    checkpoint_dir: str | Path = "checkpoints/",
    output_dir: str | Path = "game_examples/",
    num_games: int | None = None,   # None = run indefinitely
    worker_id: str = "auto",        # "auto" → f"{hostname}-{pid}"
    num_simulations: int = 400,
    c_puct: float = 2.5,
    temperature: float = 1.0,
    temperature_threshold: int = 15,
    checkpoint_poll_sec: int = 5,
) -> None:
```

### Behavior

1. **Startup:** Resolve `worker_id` from hostname + PID if `"auto"`. Wait up to
   60s for `checkpoint_dir/latest.pt` to exist, then load the model wrapper.
2. **Main loop:** For each game:
   - Check if `checkpoint_dir/latest.pt` mtime changed since last load; if so,
     reload the wrapper.
   - Run `SelfPlayGame.play()` (augment=False — raw examples only).
   - Write game to a temp file, then `os.rename` into `output_dir/`.
   - Write a metadata sidecar JSON with `{worker_id, game_length, duration_sec,
     winner, num_examples}`.
   - If `num_games` is set and reached, exit.
3. **Shutdown:** On SIGINT/SIGTERM, finish the in-progress game, write it, then
   exit. A second signal forces immediate exit.

### File format

- **Game file:** `game_<worker_id>_<seq>_<iso8601>.pt`
  - Content: `list[TrainingExample]` saved via `torch.save`
- **Metadata:** `game_<worker_id>_<seq>_<iso8601>_meta.json`
  - Content: `{"worker_id": str, "game_length": int, "duration_sec": float,
    "winner": "BLACK"|"WHITE"|"DRAW", "num_examples": int}`

### CLI

```
python -m selfplay.worker \
    --checkpoint-dir checkpoints/ \
    --output-dir game_examples/ \
    --num-games 100 \
    --num-simulations 400 \
    --temperature 1.0 \
    --temperature-threshold 15
```

All arguments are optional with sensible defaults.

## Modified file: `selfplay/train.py`

### New function: `ingest_game_files()`

```python
def ingest_game_files(
    buffer: ReplayBuffer,
    game_dir: Path,
    consumed_dir: Path,
    max_consumed: int = 1000,
) -> int:
    """Ingest pending .pt game files into the replay buffer.

    Skips files already present in consumed_dir (dedup by filename).
    Successfully ingested files are moved to consumed_dir, which is
    trimmed to the most recent *max_consumed* files.

    Returns the number of files ingested.
    """
```

### Changes to `main()`

New parameter `game_examples_dir: str | Path = "game_examples/"`.

Before each training iteration:
1. Call `ingest_game_files()` to collect pending game files.
2. If no new files and buffer has fewer than `batch_size` examples, skip
   training this iteration (not enough data).
3. Otherwise, sample a training batch from the buffer and train as before.

The trainer maintains two checkpoint files:

- `latest.pt` — updated every iteration (always the freshest policy). Workers
  poll this file's mtime to decide when to reload.
- `best_<iter>_<winrate>.pt` — written only on promotion (win rate >
  threshold). Historical snapshots for rollback and visibility.

### Evaluation (unchanged)

Evaluation tournaments still use `threat_override=True` with deterministic MCTS
(temperature=0). No changes to the evaluation logic.

## Entry points

| Command | Role |
|---------|------|
| `python -m selfplay.train` | Start the trainer (polls for game files, trains, evaluates) |
| `python -m selfplay.worker` | Start a worker (generates self-play games, writes files) |
| `python -m selfplay.worker --num-games 100` | Run a fixed number of games, then exit |

No new top-level `main.py` — `python -m` is the canonical invocation pattern.

## Tests

### `tests/test_worker.py`

1. **`test_worker_writes_valid_game_file`**
   - Create temp checkpoint (fresh model) + temp output dir.
   - Run `run_worker(games=1)`.
   - Assert one `.pt` file exists and `torch.load` returns `list[TrainingExample]`.
   - Assert metadata sidecar exists and has expected keys.

2. **`test_worker_respects_num_games`**
   - Run with `num_games=3`, assert exactly 3 game files written, then worker exits.

3. **`test_trainer_ingests_and_archives`**
   - Create temp `game_examples/` with 3 game files.
   - Call `ingest_game_files()` into a fresh `ReplayBuffer`.
   - Assert buffer length increased, files moved to `consumed/`, no files remain in root.

4. **`test_trainer_skips_already_consumed`**
   - Place same filename in both `game_examples/` and `consumed/`.
   - Call `ingest_game_files()` — assert file skipped (not double-ingested).

5. **`test_consumed_cap_enforced`**
   - Fill `consumed/` beyond `max_consumed`, run ingest, assert oldest files deleted.

## Non-goals (explicitly out of scope)

- Real-time networking between workers and trainer (TCP, gRPC, message queues)
- Automatic worker discovery or orchestration (Kubernetes, docker-compose)
- Cloud bucket integration (S3, GCS) — `rsync` is documented for cross-machine sync
- Worker pause/resume or checkpointing of worker state
- GPU inference in workers
- Metrics dashboard or structured logging aggregation
