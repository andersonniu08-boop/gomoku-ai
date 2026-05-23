# Distributed Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale AlphaZero self-play by running multiple worker processes that generate games independently and write them as files, while a central trainer polls and ingests those files.

**Architecture:** Workers call `SelfPlayGame.play()` in a loop and write raw `TrainingExample` lists to `.pt` files via atomic rename. The trainer polls a `game_examples/` directory, loads new files into the `ReplayBuffer`, archives consumed files to `consumed/`, then trains as before. Workers watch `checkpoints/latest.pt` mtime to reload the freshest policy.

**Tech Stack:** Python 3.14, PyTorch, no new dependencies

---

### File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `selfplay/worker.py` | Worker process: checkpoint polling, game loop, file writing |
| Modify | `selfplay/train.py` | Add `ingest_game_files()`, update `main()` for file polling and checkpoint naming |
| Modify | `selfplay/__init__.py` | Export `run_worker` |
| Create | `tests/test_worker.py` | Tests for worker file output and trainer ingestion |
| Create | `docs/distributed.md` | Operational guide for running workers + trainer |

---

### Task 1: `ingest_game_files` — trainer-side file ingestion

**Files:**
- Modify: `selfplay/train.py`

- [ ] **Step 1: Add `ingest_game_files()` function**

Add this function after `save_model_checkpoint` (line 46), before `train_on_examples`:

```python
def ingest_game_files(
    buffer: ReplayBuffer,
    game_dir: Path,
    consumed_dir: Path,
    max_consumed: int = 1000,
) -> int:
    """Ingest pending .pt game files into the replay buffer.

    Skips files already present in *consumed_dir* (dedup by filename).
    Successfully ingested files are moved to *consumed_dir*, which is
    trimmed to the most recent *max_consumed* files.

    Returns the number of files ingested.
    """
    consumed_dir.mkdir(parents=True, exist_ok=True)

    consumed_names = {
        p.name for p in consumed_dir.iterdir() if p.suffix == ".pt"
    }

    pending = sorted(
        [p for p in game_dir.glob("game_*.pt") if p.name not in consumed_names]
    )
    if not pending:
        return 0

    ingested = 0
    for path in pending:
        try:
            examples = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            print(f"  [trainer] Skipping unreadable file: {path.name}")
            continue

        if not isinstance(examples, list) or not examples:
            print(f"  [trainer] Skipping empty/invalid file: {path.name}")
            continue

        buffer.add_examples(examples)
        path.rename(consumed_dir / path.name)
        ingested += 1

    # Trim consumed directory to *max_consumed* most-recent files.
    consumed_files = sorted(
        consumed_dir.glob("game_*.pt"), key=lambda p: p.stat().st_mtime
    )
    for old in consumed_files[:-max_consumed]:
        old.unlink(missing_ok=True)
        meta = old.with_suffix(old.suffix.replace(".pt", "_meta.json"))
        meta.unlink(missing_ok=True)

    return ingested
```

- [ ] **Step 2: Add `import json` and `import time` at the top of train.py**

```python
import json
import time
```

(Add after the existing imports; `json` is for metadata sidecar reading in a later task, `time` is for the polling sleep.)

- [ ] **Step 3: Verify the file still parses**

Run: `python -c "from selfplay.train import ingest_game_files; print('OK')"`
Expected: `OK`

---

### Task 2: `run_worker` — worker process

- [ ] **Step 1: Write the failing test for `run_worker`**

Create file `tests/test_worker.py`:

```python
"""Tests for selfplay.worker — distributed self-play worker process."""

import json
import tempfile
from pathlib import Path

import torch

from neural.model import GomokuNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.selfplay import TrainingExample
from selfplay.train import ingest_game_files


def _make_checkpoint(output_dir: str) -> Path:
    """Create a minimal best.pt and latest.pt in *output_dir*."""
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=10,
                      num_hidden_channels=128, use_se=True, use_attention=True)
    checkpoint_path = Path(output_dir) / "latest.pt"
    torch.save(model.state_dict(), str(checkpoint_path))
    # Also write best.pt so the bootstrap path works.
    best_path = Path(output_dir) / "best.pt"
    torch.save(model.state_dict(), str(best_path))
    return checkpoint_path


def test_worker_writes_valid_game_file():
    from selfplay.worker import run_worker

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt_dir = tmpdir / "checkpoints"
        out_dir = tmpdir / "game_examples"
        ckpt_dir.mkdir()
        out_dir.mkdir()
        _make_checkpoint(str(ckpt_dir))

        run_worker(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
            num_games=1,
            num_simulations=4,
            temperature=1.0,
            temperature_threshold=0,
            checkpoint_poll_sec=0,
        )

        files = list(out_dir.glob("game_*.pt"))
        assert len(files) == 1

        examples = torch.load(str(files[0]), map_location="cpu", weights_only=False)
        assert isinstance(examples, list)
        assert len(examples) > 0
        for ex in examples:
            assert isinstance(ex, TrainingExample)
            assert ex.state.shape == (3, 15, 15)
            assert ex.policy.shape == (225,)
            assert abs(float(ex.policy.sum()) - 1.0) < 1e-5
            assert -1.0 <= ex.value <= 1.0


def test_worker_respects_num_games():
    from selfplay.worker import run_worker

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt_dir = tmpdir / "checkpoints"
        out_dir = tmpdir / "game_examples"
        ckpt_dir.mkdir()
        out_dir.mkdir()
        _make_checkpoint(str(ckpt_dir))

        run_worker(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
            num_games=3,
            num_simulations=4,
            temperature=1.0,
            temperature_threshold=0,
            checkpoint_poll_sec=0,
        )

        files = list(out_dir.glob("game_*.pt"))
        assert len(files) == 3


def test_worker_writes_metadata_sidecar():
    from selfplay.worker import run_worker

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt_dir = tmpdir / "checkpoints"
        out_dir = tmpdir / "game_examples"
        ckpt_dir.mkdir()
        out_dir.mkdir()
        _make_checkpoint(str(ckpt_dir))

        run_worker(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
            num_games=1,
            num_simulations=4,
            temperature=1.0,
            temperature_threshold=0,
            checkpoint_poll_sec=0,
        )

        meta_files = list(out_dir.glob("*_meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert "worker_id" in meta
        assert "game_length" in meta
        assert "duration_sec" in meta
        assert meta["winner"] in ("BLACK", "WHITE", "DRAW")
        assert "num_examples" in meta


def test_ingest_game_files_loads_into_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        game_dir = tmpdir / "game_examples"
        consumed_dir = game_dir / "consumed"
        game_dir.mkdir(parents=True)

        # Write a synthetic game file.
        examples = [
            TrainingExample(
                state=torch.randn(3, 15, 15),
                policy=torch.zeros(225),
                value=1.0,
            )
            for _ in range(5)
        ]
        path = game_dir / "game_test_0001_20260522T120000.pt"
        torch.save(examples, str(path))

        buf = ReplayBuffer(max_size=1000)
        n = ingest_game_files(buf, game_dir, consumed_dir)
        assert n == 1
        assert len(buf) == 5
        assert not path.exists()
        assert (consumed_dir / path.name).exists()


def test_ingest_game_files_skips_consumed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        game_dir = tmpdir / "game_examples"
        consumed_dir = game_dir / "consumed"
        game_dir.mkdir(parents=True)
        consumed_dir.mkdir()

        filename = "game_w0_0001_20260522T120000.pt"
        examples = [TrainingExample(
            state=torch.randn(3, 15, 15),
            policy=torch.zeros(225),
            value=-1.0,
        )]

        # Write same filename to both directories.
        torch.save(examples, str(game_dir / filename))
        torch.save(examples, str(consumed_dir / filename))

        buf = ReplayBuffer(max_size=1000)
        n = ingest_game_files(buf, game_dir, consumed_dir)
        assert n == 0
        assert len(buf) == 0


def test_ingest_game_files_caps_consumed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        game_dir = tmpdir / "game_examples"
        consumed_dir = game_dir / "consumed"
        game_dir.mkdir(parents=True)
        consumed_dir.mkdir()

        buf = ReplayBuffer(max_size=10000)

        # Create and ingest 12 files with a cap of 10.
        for i in range(12):
            examples = [TrainingExample(
                state=torch.randn(3, 15, 15),
                policy=torch.zeros(225),
                value=1.0,
            )]
            path = game_dir / f"game_test_{i:04d}_20260522T120000.pt"
            torch.save(examples, str(path))

        n = ingest_game_files(buf, game_dir, consumed_dir, max_consumed=10)
        assert n == 12
        assert len(buf) == 12
        assert len(list(consumed_dir.glob("game_*.pt"))) == 10
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_worker.py -v`
Expected: All 6 tests FAIL — `ModuleNotFoundError: No module named 'selfplay.worker'`

- [ ] **Step 3: Write `selfplay/worker.py`**

```python
"""Distributed self-play worker — generates games and writes them to disk.

Run with: ``python -m selfplay.worker``
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from neural.wrapper import GomokuInferenceWrapper
from selfplay.selfplay import SelfPlayGame

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False
"""Module-level flag set by SIGINT/SIGTERM handler so the worker can
finish its current game before exiting."""


def _on_shutdown(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        # Second signal — hard exit.
        raise SystemExit(1)
    _SHUTDOWN_REQUESTED = True
    print("\n[worker] Shutdown requested — finishing current game...")


signal.signal(signal.SIGINT, _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)


def run_worker(
    checkpoint_dir: str | Path = "checkpoints/",
    output_dir: str | Path = "game_examples/",
    num_games: int | None = None,
    worker_id: str = "auto",
    num_simulations: int = 400,
    c_puct: float = 2.5,
    temperature: float = 1.0,
    temperature_threshold: int = 15,
    checkpoint_poll_sec: int = 5,
) -> None:
    """Run a self-play worker that writes game files for the trainer.

    Parameters:
        checkpoint_dir: Directory containing ``latest.pt`` (watched for
            changes so the worker always uses the freshest policy).
        output_dir: Directory where ``game_*.pt`` files are written.
        num_games: If set, stop after this many games.  ``None`` = run
            indefinitely until SIGINT/SIGTERM.
        worker_id: Prefix used in game file names.  ``"auto"`` resolves
            to ``<hostname>-<pid>``.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        temperature: Visit-count exponent for early-game move sampling.
        temperature_threshold: Move number after which temperature is
            annealed to 0 (deterministic).
        checkpoint_poll_sec: Seconds between mtime checks on
            ``checkpoint_dir/latest.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if worker_id == "auto":
        worker_id = f"{socket.gethostname()}-{os.getpid()}"

    # --- Wait for initial checkpoint ---
    latest_path = checkpoint_dir / "latest.pt"
    waited = 0.0
    while not latest_path.exists():
        if waited >= 60:
            raise FileNotFoundError(
                f"No checkpoint found at {latest_path} after 60s — "
                "start the trainer first."
            )
        time.sleep(1.0)
        waited += 1.0

    wrapper = GomokuInferenceWrapper(str(latest_path), device="cpu")
    last_mtime = latest_path.stat().st_mtime
    print(f"[worker {worker_id}] Loaded checkpoint from {latest_path}")

    games_played = 0
    seq = 0

    while not _SHUTDOWN_REQUESTED:
        if num_games is not None and games_played >= num_games:
            break

        # --- Reload checkpoint if updated ---
        if latest_path.exists():
            cur_mtime = latest_path.stat().st_mtime
            if cur_mtime > last_mtime:
                wrapper = GomokuInferenceWrapper(str(latest_path), device="cpu")
                last_mtime = cur_mtime
                print(f"[worker {worker_id}] Reloaded updated checkpoint")

        # --- Play one game ---
        game = SelfPlayGame(
            wrapper,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            temperature_threshold=temperature_threshold,
            threat_override=True,
            augment=False,
        )

        t0 = time.monotonic()
        examples = game.play()
        duration = time.monotonic() - t0

        if not examples:
            continue

        seq += 1
        games_played += 1

        # Determine winner from the last example's value.
        last_val = examples[-1].value
        if last_val > 0:
            winner = examples[-1].value > 0
            # The value is from the perspective of the mover.  The last
            # mover is the winner if value=+1 from their perspective,
            # which means the game's winner is that player. But since
            # we don't track the player identity in examples, infer:
            winner_str = "BLACK" if examples[-1].value > 0 else "WHITE"
        else:
            winner_str = "DRAW"

        # Actually, fix the winner inference: examples[-1].value is +1
        # if the last player to move won.  We can't easily tell which
        # colour that was without tracking it, so record from the value.
        if examples[-1].value > 0.5:
            winner_str = "LAST_MOVER"
        elif examples[-1].value < -0.5:
            winner_str = "LAST_MOVER_LOST"
        else:
            winner_str = "DRAW"

        # --- Write game file atomically ---
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem = f"game_{worker_id}_{seq:04d}_{ts}"

        tmp_path = output_dir / f".{stem}.tmp"
        final_path = output_dir / f"{stem}.pt"
        meta_path = output_dir / f"{stem}_meta.json"

        torch.save(examples, str(tmp_path))

        meta = {
            "worker_id": worker_id,
            "game_length": len(examples),
            "duration_sec": round(duration, 3),
            "winner": winner_str,
            "num_examples": len(examples),
        }
        meta_path.write_text(json.dumps(meta))

        tmp_path.rename(final_path)
        print(
            f"[worker {worker_id}] Game {seq} ({games_played}): "
            f"{len(examples)} examples, {duration:.1f}s"
        )

    print(f"[worker {worker_id}] Stopped.  {games_played} games played.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    p = argparse.ArgumentParser(description="Gomoku self-play worker")
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--output-dir", default="game_examples/")
    p.add_argument("--num-games", type=int, default=None)
    p.add_argument("--worker-id", default="auto")
    p.add_argument("--num-simulations", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=2.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-threshold", type=int, default=15)
    p.add_argument("--checkpoint-poll-sec", type=int, default=5)
    args = p.parse_args()

    run_worker(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_games=args.num_games,
        worker_id=args.worker_id,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_threshold=args.temperature_threshold,
        checkpoint_poll_sec=args.checkpoint_poll_sec,
    )


if __name__ == "__main__":
    _main()
```

Wait — there's a bug in the winner inference. Let me re-examine. The last example in the list is the last move of the game. The `value` in a `TrainingExample` is assigned by `_assign_values` based on `winner == player`. So if the last mover won, their example has value=+1. We should track the player during the game. But `SelfPlayGame.play()` doesn't expose player identity per example.

Let me simplify: store `player` in the example metadata by looking at who the last example's value refers to — actually, we can't know which colour the player was. Let me just track it properly in the worker by retrieving the board state after the game.

Actually, wait. SelfPlayGame.play() returns TrainingExamples but doesn't tell us which colour won. Let me look at `_assign_values` again:

```python
for state, policy, player in raw:
    if winner is None:
        value = 0.0
    elif winner == player:
        value = 1.0
    else:
        value = -1.0
```

So the value is +1 if the player making that move is the eventual winner, -1 otherwise. The last example's value tells us: if +1, the last mover won; if -1, the last mover lost.

But to know who "won" in terms of BLACK/WHITE, we need to know which colour the last mover was. For the metadata sidecar, let me just record the outcome in terms of the last mover (not BLACK/WHITE) and note it. That's fine — the metadata is for debugging anyway.

Actually, let me simplify even further. Let me record the outcome using the board's `check_win()` result. But the worker doesn't have access to the board after `SelfPlayGame.play()` returns.

The cleanest fix: just use the value sign on the last example to determine winner for metadata. Value > 0 means last mover won, value < 0 means last mover lost, value == 0 means draw. Let me write:

```
if last_val > 0.5:
    winner_str = "LAST_MOVER_WON"
elif last_val < -0.5:
    winner_str = "LAST_MOVER_LOST"
else:
    winner_str = "DRAW"
```

This is simpler and doesn't fabricate colour information. Let me fix the code in the plan.

Actually, hmm, I just wrote a very long plan step. Let me re-examine the code I wrote. There's a logic issue - I compute `winner_str` twice (first a draft, then the fix). Let me clean this up.

Let me rewrite the entire plan with clean code.<｜end▁of▁thinking｜>Let me rewrite that plan step with clean, working code instead of drafting in place.

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Write">
<｜｜DSML｜｜parameter name="content" string="true"># Distributed Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale AlphaZero self-play by running multiple worker processes that generate games independently and write them as files, while a central trainer polls and ingests those files.

**Architecture:** Workers call `SelfPlayGame.play()` in a loop and write raw `TrainingExample` lists to `.pt` files via atomic rename. The trainer polls a `game_examples/` directory, loads new files into the `ReplayBuffer`, archives consumed files to `consumed/`, then trains as before. Workers watch `checkpoints/latest.pt` mtime to reload the freshest policy.

**Tech Stack:** Python 3.14, PyTorch, no new dependencies

---

### File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `selfplay/worker.py` | Worker process: checkpoint polling, game loop, file writing |
| Modify | `selfplay/train.py` | Add `ingest_game_files()`, update `main()` for file polling and checkpoint naming |
| Modify | `selfplay/__init__.py` | Export `run_worker` |
| Create | `tests/test_worker.py` | Tests for worker file output and trainer ingestion |
| Create | `docs/distributed.md` | Operational guide for running workers + trainer |

---

### Task 1: `ingest_game_files` — trainer-side file ingestion

**Files:**
- Modify: `selfplay/train.py`

- [ ] **Step 1: Add `ingest_game_files()` function**

Insert after `save_model_checkpoint` (line 46), before `train_on_examples`:

```python
def ingest_game_files(
    buffer: ReplayBuffer,
    game_dir: Path,
    consumed_dir: Path,
    max_consumed: int = 1000,
) -> int:
    """Ingest pending .pt game files into the replay buffer.

    Skips files already present in *consumed_dir* (dedup by filename).
    Successfully ingested files are moved to *consumed_dir*, which is
    trimmed to the most recent *max_consumed* files.

    Returns the number of files ingested.
    """
    consumed_dir.mkdir(parents=True, exist_ok=True)

    consumed_names = {
        p.name for p in consumed_dir.iterdir() if p.suffix == ".pt"
    }

    pending = sorted(
        [p for p in game_dir.glob("game_*.pt") if p.name not in consumed_names]
    )
    if not pending:
        return 0

    ingested = 0
    for path in pending:
        try:
            examples = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            print(f"  [trainer] Skipping unreadable file: {path.name}")
            continue

        if not isinstance(examples, list) or not examples:
            print(f"  [trainer] Skipping empty/invalid file: {path.name}")
            continue

        buffer.add_examples(examples)
        path.rename(consumed_dir / path.name)
        ingested += 1

    # Trim consumed directory to *max_consumed* most-recent files.
    consumed_files = sorted(
        consumed_dir.glob("game_*.pt"), key=lambda p: p.stat().st_mtime
    )
    for old in consumed_files[:-max_consumed]:
        meta_name = old.name.replace(".pt", "_meta.json")
        (consumed_dir / meta_name).unlink(missing_ok=True)
        old.unlink(missing_ok=True)

    return ingested
```

- [ ] **Step 2: Verify import**

Run: `python -c "from selfplay.train import ingest_game_files; print('OK')"`
Expected: `OK`

---

### Task 2: Write tests for worker and ingestion

**Files:**
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write all tests**

```python
"""Tests for selfplay.worker — distributed self-play worker process."""

import json
import tempfile
from pathlib import Path

import torch

from neural.model import GomokuNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.selfplay import TrainingExample
from selfplay.train import ingest_game_files


def _make_checkpoint(output_dir: Path) -> None:
    """Create a minimal GomokuNet checkpoint (latest.pt) in *output_dir*."""
    model = GomokuNet(
        board_size=15,
        in_channels=3,
        num_res_blocks=10,
        num_hidden_channels=128,
        use_se=True,
        use_attention=True,
    )
    torch.save(model.state_dict(), str(output_dir / "latest.pt"))


def test_worker_writes_valid_game_file():
    from selfplay.worker import run_worker

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ckpt_dir = root / "checkpoints"
        out_dir = root / "game_examples"
        ckpt_dir.mkdir()
        out_dir.mkdir()
        _make_checkpoint(ckpt_dir)

        run_worker(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
            num_games=1,
            num_simulations=4,
            temperature=1.0,
            temperature_threshold=0,
            checkpoint_poll_sec=0,
        )

        files = list(out_dir.glob("game_*.pt"))
        assert len(files) == 1

        examples = torch.load(str(files[0]), map_location="cpu", weights_only=False)
        assert isinstance(examples, list)
        assert len(examples) > 0
        for ex in examples:
            assert isinstance(ex, TrainingExample)
            assert ex.state.shape == (3, 15, 15)
            assert ex.policy.shape == (225,)
            assert abs(float(ex.policy.sum()) - 1.0) < 1e-5
            assert -1.0 <= ex.value <= 1.0


def test_worker_respects_num_games():
    from selfplay.worker import run_worker

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ckpt_dir = root / "checkpoints"
        out_dir = root / "game_examples"
        ckpt_dir.mkdir()
        out_dir.mkdir()
        _make_checkpoint(ckpt_dir)

        run_worker(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
            num_games=3,
            num_simulations=4,
            temperature=1.0,
            temperature_threshold=0,
            checkpoint_poll_sec=0,
        )

        files = list(out_dir.glob("game_*.pt"))
        assert len(files) == 3


def test_worker_writes_metadata_sidecar():
    from selfplay.worker import run_worker

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ckpt_dir = root / "checkpoints"
        out_dir = root / "game_examples"
        ckpt_dir.mkdir()
        out_dir.mkdir()
        _make_checkpoint(ckpt_dir)

        run_worker(
            checkpoint_dir=str(ckpt_dir),
            output_dir=str(out_dir),
            num_games=1,
            num_simulations=4,
            temperature=1.0,
            temperature_threshold=0,
            checkpoint_poll_sec=0,
        )

        meta_files = list(out_dir.glob("*_meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert "worker_id" in meta
        assert "game_length" in meta
        assert "duration_sec" in meta
        assert meta["winner"] in ("LAST_MOVER_WON", "LAST_MOVER_LOST", "DRAW")
        assert "num_examples" in meta


def test_ingest_game_files_loads_into_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        game_dir = root / "game_examples"
        consumed_dir = game_dir / "consumed"
        game_dir.mkdir(parents=True)

        examples = [
            TrainingExample(
                state=torch.randn(3, 15, 15),
                policy=torch.zeros(225),
                value=1.0,
            )
            for _ in range(5)
        ]
        path = game_dir / "game_test_0001_20260522T120000.pt"
        torch.save(examples, str(path))

        buf = ReplayBuffer(max_size=1000)
        n = ingest_game_files(buf, game_dir, consumed_dir)
        assert n == 1
        assert len(buf) == 5
        assert not path.exists()
        assert (consumed_dir / path.name).exists()


def test_ingest_game_files_skips_consumed():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        game_dir = root / "game_examples"
        consumed_dir = game_dir / "consumed"
        game_dir.mkdir(parents=True)
        consumed_dir.mkdir()

        filename = "game_w0_0001_20260522T120000.pt"
        examples = [
            TrainingExample(
                state=torch.randn(3, 15, 15),
                policy=torch.zeros(225),
                value=-1.0,
            )
        ]

        torch.save(examples, str(game_dir / filename))
        torch.save(examples, str(consumed_dir / filename))

        buf = ReplayBuffer(max_size=1000)
        n = ingest_game_files(buf, game_dir, consumed_dir)
        assert n == 0
        assert len(buf) == 0


def test_ingest_game_files_caps_consumed():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        game_dir = root / "game_examples"
        consumed_dir = game_dir / "consumed"
        game_dir.mkdir(parents=True)
        consumed_dir.mkdir()

        buf = ReplayBuffer(max_size=10000)

        for i in range(12):
            examples = [
                TrainingExample(
                    state=torch.randn(3, 15, 15),
                    policy=torch.zeros(225),
                    value=1.0,
                )
            ]
            path = game_dir / f"game_test_{i:04d}_20260522T120000.pt"
            torch.save(examples, str(path))

        n = ingest_game_files(buf, game_dir, consumed_dir, max_consumed=10)
        assert n == 12
        assert len(buf) == 12
        assert len(list(consumed_dir.glob("game_*.pt"))) == 10
```

- [ ] **Step 2: Run tests — verify they fail on missing worker module**

Run: `pytest tests/test_worker.py::test_worker_writes_valid_game_file -v`
Expected: `ModuleNotFoundError: No module named 'selfplay.worker'`

---

### Task 3: Implement `selfplay/worker.py`

**Files:**
- Create: `selfplay/worker.py`

- [ ] **Step 1: Write the worker module**

```python
"""Distributed self-play worker — generates games and writes them to disk.

Run with: ``python -m selfplay.worker``
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from neural.wrapper import GomokuInferenceWrapper
from selfplay.selfplay import SelfPlayGame

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False


def _on_shutdown(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        raise SystemExit(1)
    _SHUTDOWN_REQUESTED = True
    print("\n[worker] Shutdown requested — finishing current game...")


signal.signal(signal.SIGINT, _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_worker(
    checkpoint_dir: str | Path = "checkpoints/",
    output_dir: str | Path = "game_examples/",
    num_games: int | None = None,
    worker_id: str = "auto",
    num_simulations: int = 400,
    c_puct: float = 2.5,
    temperature: float = 1.0,
    temperature_threshold: int = 15,
    checkpoint_poll_sec: int = 5,
) -> None:
    """Run a self-play worker that writes game files for the trainer.

    Parameters:
        checkpoint_dir: Directory containing ``latest.pt`` (watched for
            changes so the worker always uses the freshest policy).
        output_dir: Directory where ``game_*.pt`` files are written.
        num_games: If set, stop after this many games.  ``None`` = run
            indefinitely until SIGINT/SIGTERM.
        worker_id: Prefix used in game file names.  ``"auto"`` resolves
            to ``<hostname>-<pid>``.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        temperature: Visit-count exponent for early-game move sampling.
        temperature_threshold: Move number after which temperature is
            annealed to 0 (deterministic).
        checkpoint_poll_sec: Seconds between mtime checks on
            ``checkpoint_dir/latest.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if worker_id == "auto":
        worker_id = f"{socket.gethostname()}-{os.getpid()}"

    # --- Wait for initial checkpoint ---
    latest_path = checkpoint_dir / "latest.pt"
    waited = 0.0
    while not latest_path.exists():
        if waited >= 60:
            raise FileNotFoundError(
                f"No checkpoint found at {latest_path} after 60s — "
                "start the trainer first."
            )
        time.sleep(1.0)
        waited += 1.0

    wrapper = GomokuInferenceWrapper(str(latest_path), device="cpu")
    last_mtime = latest_path.stat().st_mtime
    print(f"[worker {worker_id}] Loaded checkpoint from {latest_path}")

    games_played = 0
    seq = 0

    while not _SHUTDOWN_REQUESTED:
        if num_games is not None and games_played >= num_games:
            break

        # --- Reload checkpoint if updated ---
        if latest_path.exists():
            cur_mtime = latest_path.stat().st_mtime
            if cur_mtime > last_mtime:
                wrapper = GomokuInferenceWrapper(str(latest_path), device="cpu")
                last_mtime = cur_mtime
                print(f"[worker {worker_id}] Reloaded updated checkpoint")

        # --- Play one game ---
        game = SelfPlayGame(
            wrapper,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            temperature_threshold=temperature_threshold,
            threat_override=True,
            augment=False,
        )

        t0 = time.monotonic()
        examples = game.play()
        duration = time.monotonic() - t0

        if not examples:
            continue

        seq += 1
        games_played += 1

        # Infer outcome from the last example's value.
        # +1 = last mover won, -1 = last mover lost, 0 = draw.
        last_val = examples[-1].value
        if last_val > 0.5:
            winner_str = "LAST_MOVER_WON"
        elif last_val < -0.5:
            winner_str = "LAST_MOVER_LOST"
        else:
            winner_str = "DRAW"

        # --- Write game file atomically ---
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem = f"game_{worker_id}_{seq:04d}_{ts}"

        tmp_path = output_dir / f".{stem}.tmp"
        final_path = output_dir / f"{stem}.pt"
        meta_path = output_dir / f"{stem}_meta.json"

        torch.save(examples, str(tmp_path))

        meta = {
            "worker_id": worker_id,
            "game_length": len(examples),
            "duration_sec": round(duration, 3),
            "winner": winner_str,
            "num_examples": len(examples),
        }
        meta_path.write_text(json.dumps(meta))

        tmp_path.rename(final_path)
        print(
            f"[worker {worker_id}] Game {seq} ({games_played}): "
            f"{len(examples)} examples, {duration:.1f}s"
        )

    print(f"[worker {worker_id}] Stopped.  {games_played} games played.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    p = argparse.ArgumentParser(description="Gomoku self-play worker")
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--output-dir", default="game_examples/")
    p.add_argument("--num-games", type=int, default=None)
    p.add_argument("--worker-id", default="auto")
    p.add_argument("--num-simulations", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=2.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-threshold", type=int, default=15)
    p.add_argument("--checkpoint-poll-sec", type=int, default=5)
    args = p.parse_args()

    run_worker(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_games=args.num_games,
        worker_id=args.worker_id,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_threshold=args.temperature_threshold,
        checkpoint_poll_sec=args.checkpoint_poll_sec,
    )


if __name__ == "__main__":
    _main()
```

- [ ] **Step 2: Run the worker test — verify it passes**

Run: `pytest tests/test_worker.py::test_worker_writes_valid_game_file -v`
Expected: `PASS`

- [ ] **Step 3: Run all worker tests**

Run: `pytest tests/test_worker.py -v`
Expected: 6/6 PASS (worker tests pass; 3 ingestion tests were already passing from Task 1)

- [ ] **Step 4: Commit worker module and tests**

```bash
git add selfplay/worker.py tests/test_worker.py
git commit -m "feat: add distributed self-play worker with file-based game output"
```

---

### Task 4: Update `selfplay/__init__.py` to export worker

**Files:**
- Modify: `selfplay/__init__.py`

- [ ] **Step 1: Add worker export**

Change the file from:

```python
from .mcts import MCTS, MCTSNode
from .selfplay import SYMMETRIES, SelfPlayGame, TrainingExample, augment_examples
from .replay_buffer import ReplayBuffer
```

To:

```python
from .mcts import MCTS, MCTSNode
from .selfplay import SYMMETRIES, SelfPlayGame, TrainingExample, augment_examples
from .replay_buffer import ReplayBuffer
from .worker import run_worker
```

- [ ] **Step 2: Verify import**

Run: `python -c "from selfplay.worker import run_worker; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add selfplay/__init__.py
git commit -m "feat: export run_worker from selfplay package"
```

---

### Task 5: Update `train.main()` for distributed file polling

**Files:**
- Modify: `selfplay/train.py`

- [ ] **Step 1: Add `import time` at the top of train.py**

After the existing imports, add:

```python
import time
```

- [ ] **Step 2: Update the `main()` function signature and body**

The key changes to `main()`:
1. New parameter `game_examples_dir: str | Path = "game_examples/"` (default preserves backward compatibility — if you use `games_per_iteration > 0` with no external dir, it still generates its own games)
2. Replace the inline self-play Phase A with a polling approach when `game_examples_dir` has files
3. Checkpoint naming: `best_iter<iter>_win<winrate>.pt` on promotion

Replace the `main()` function body. The current loop (lines 197-253) becomes:

```python
    checkpoints_dir = Path("checkpoints")
    data_dir = Path("data")
    game_dir = Path(game_examples_dir)
    consumed_dir = game_dir / "consumed"
    checkpoints_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    game_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoints_dir / "best.pt"
    latest_path = checkpoints_dir / "latest.pt"
    buffer_path = data_dir / "replay_buffer.pt"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bootstrap: if no best checkpoint exists, create one from a fresh model.
    if not best_path.exists():
        model = GomokuNet(
            board_size=15,
            in_channels=3,
            num_res_blocks=10,
            num_hidden_channels=128,
            use_se=True,
            use_attention=True,
        )
        save_model_checkpoint(model, best_path)
        save_model_checkpoint(model, latest_path)

    # Load or create replay buffer.
    if buffer_path.exists():
        data = torch.load(str(buffer_path), map_location="cpu", weights_only=False)
        buffer = ReplayBuffer.from_state_dict(data)
    else:
        buffer = ReplayBuffer(max_size=500_000)

    print(f"Device: {device}")
    print(f"Buffer size: {len(buffer)}")
    print(f"Iterations: {num_iterations}, batch: {batch_size}")
    print(f"Game dir: {game_dir}  (polling for worker-generated files)")
    print(f"Generating {games_per_iteration} local games/iter as fallback")

    for iteration in range(1, num_iterations + 1):
        # --- Phase A: Collect self-play data ---
        # 1. Ingest any worker-generated files.
        ingested = ingest_game_files(buffer, game_dir, consumed_dir)
        if ingested > 0:
            print(f"\nIteration {iteration}: Ingested {ingested} game files, "
                  f"buffer now {len(buffer)}")

        # 2. Fallback: generate local games if workers haven't produced enough.
        if len(buffer) < batch_size:
            print(f"  Buffer too small ({len(buffer)} < {batch_size}), "
                  f"generating {games_per_iteration} local games...")
            wrapper = GomokuInferenceWrapper(latest_path, device=device)
            game = SelfPlayGame(
                wrapper,
                num_simulations=mcts_simulations,
                temperature=selfplay_temperature,
                temperature_threshold=selfplay_temp_threshold,
                threat_override=True,
                augment=True,
            )

            for _ in range(games_per_iteration):
                examples = game.play()
                buffer.add_examples(examples)

            torch.save(buffer.state_dict(), str(buffer_path))
            print(f"  Buffer now {len(buffer)} after local generation")

        # --- Phase B: Train ---
        model = GomokuNet(
            board_size=15,
            in_channels=3,
            num_res_blocks=10,
            num_hidden_channels=128,
            use_se=True,
            use_attention=True,
        )
        model.load_state_dict(
            torch.load(str(latest_path), map_location=device, weights_only=True)
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Sample a training batch from the full buffer.
        if len(buffer) < batch_size:
            print("  Not enough examples to train — skipping iteration")
            continue

        train_examples = buffer.sample(min(len(buffer), batch_size * 10))
        total_steps = (len(train_examples) + batch_size - 1) // batch_size
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        avg_loss = train_on_examples(
            model, optimizer, train_examples, batch_size, scheduler, device=device
        )

        save_model_checkpoint(model, latest_path)
        torch.save(buffer.state_dict(), str(buffer_path))
        print(f"  Training loss: {avg_loss:.4f}  "
              f"(lr={scheduler.get_last_lr()[0]:.6f})")

        # --- Evaluation ---
        if iteration % eval_frequency == 0:
            print(f"  Evaluating latest vs best ({eval_games} games)...")
            win_rate = run_evaluation(
                latest_path, best_path, num_games=eval_games, device=device
            )

            if win_rate >= eval_threshold:
                # Save promoted checkpoint with iteration and win rate in name.
                pct = round(win_rate * 100)
                promoted_name = f"best_iter{iteration:03d}_win{pct}.pt"
                save_model_checkpoint(model, checkpoints_dir / promoted_name)
                save_model_checkpoint(model, best_path)
                print(f"  Promoted!  Win rate: {win_rate:.2%}  → {promoted_name}")
            else:
                print(f"  Not promoted.  Win rate: {win_rate:.2%}  "
                      f"(threshold {eval_threshold:.0%})")

        # Brief pause so workers can write more files.
        time.sleep(2)

    print(f"\nDone.  Best model: {best_path}")
```

The complete `main()` function signature becomes:

```python
def main(
    num_iterations: int = 50,
    games_per_iteration: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    eval_frequency: int = 5,
    eval_games: int = 100,
    eval_threshold: float = 0.55,
    mcts_simulations: int = 400,
    selfplay_temperature: float = 1.0,
    selfplay_temp_threshold: int = 15,
    device: Optional[str] = None,
    game_examples_dir: str | Path = "game_examples/",
) -> None:
```

- [ ] **Step 2: Check that existing tests still pass**

Run: `pytest tests/test_selfplay.py tests/test_train.py -v`
Expected: All existing tests pass

- [ ] **Step 3: Commit**

```bash
git add selfplay/train.py
git commit -m "feat: add distributed game-file ingestion to training loop"
```

---

### Task 6: Write operational docs

**Files:**
- Create: `docs/distributed.md`

- [ ] **Step 1: Write `docs/distributed.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/distributed.md
git commit -m "docs: add distributed self-play operational guide"
```

---

### Task 7: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new worker tests)

- [ ] **Step 2: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: final cleanup for distributed self-play"
```
