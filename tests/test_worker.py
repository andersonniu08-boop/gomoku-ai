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
        # 5 examples × 8 D₄ symmetries = 40
        assert len(buf) == 40
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
        # 12 files × 1 example × 8 symmetries = 96
        assert n == 12
        assert len(buf) == 96
        assert len(list(consumed_dir.glob("game_*.pt"))) == 10
