# Distributed Self-Play

## Quick start (single machine)

Terminal 1 — trainer:
```
python -m selfplay.train
```

Terminal 2+ — workers:
```
python -m selfplay.worker
```

That's it. The trainer polls `game_examples/` for worker-generated files.
Workers automatically pick up new checkpoints from `checkpoints/latest.pt`.

## With a cloud machine

1. Copy the repo and a checkpoint to the cloud machine.
2. Start workers on the cloud machine:
   ```
   python -m selfplay.worker --output-dir game_examples/
   ```
3. Periodically sync game files back:
   ```
   rsync -avz cloud:~/gomoku-ai/game_examples/*.pt ~/gomoku-ai/game_examples/
   ```
4. The local trainer ingests them automatically.

## Directory layout

```
checkpoints/
  latest.pt          ← updated every iteration (workers watch this)
  best.pt            ← the strongest promoted model
  best_iter050_win68.pt  ← historical promoted snapshots

game_examples/
  game_w0_0001_20260522T120000.pt       ← game file
  game_w0_0001_20260522T120000_meta.json  ← metadata sidecar
  consumed/                             ← trainer archives processed files here
    (capped at 1000 most-recent files)
```

## Worker CLI

```
python -m selfplay.worker \
    --checkpoint-dir checkpoints/ \
    --output-dir game_examples/ \
    --num-games 100 \           # omit for infinite loop
    --num-simulations 400 \
    --temperature 1.0 \
    --temperature-threshold 15
```
