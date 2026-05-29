from __future__ import annotations
import torch
import pytest
from pathlib import Path
from selfplay.train import main, save_training_state
from neural.model import GomokuNet
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def test_fresh_start_clears_stale_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    stale = [
        tmp_path / "data" / "training_state.pt",
        tmp_path / "data" / "replay_buffer.pt",
        tmp_path / "data" / "elo_state.json",
        tmp_path / "data" / "training_log.csv",
    ]
    for f in stale:
        f.write_text("dummy")
    main(
        num_iterations=1,
        games_per_iteration=1,
        eval_frequency=999,
        mcts_simulations=8,
        fresh_start=True,
    )
    for f in stale:
        assert f.read_bytes() != b"dummy", f"{f.name} was not cleared"


def test_resume_past_end_warns_and_exits(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "checkpoints").mkdir()
    model = GomokuNet()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    save_training_state(
        tmp_path / "data" / "training_state.pt",
        model, optimizer, scheduler, None, iteration=5,
    )
    main(num_iterations=5, games_per_iteration=1, eval_frequency=999, mcts_simulations=8)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "exceeds" in captured.out
