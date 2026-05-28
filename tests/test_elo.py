"""Tests for the Elo rating system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from selfplay.elo import (
    INITIAL_RATING,
    EloTracker,
    MatchRecord,
    RatingSnapshot,
    expected_score,
    update_rating,
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


class TestExpectedScore:
    def test_equal_ratings(self) -> None:
        assert expected_score(1500, 1500) == 0.5

    def test_stronger_player(self) -> None:
        # A 400-point difference → expected score ~0.91 for the stronger
        # player (1 / (1 + 10^(-1)) ≈ 0.909).
        score = expected_score(1900, 1500)
        assert 0.90 < score < 0.92

    def test_weaker_player(self) -> None:
        score = expected_score(1500, 1900)
        assert 0.08 < score < 0.10

    def test_symmetric(self) -> None:
        a = expected_score(1600, 1400)
        b = expected_score(1400, 1600)
        assert abs(a + b - 1.0) < 1e-10


class TestUpdateRating:
    def test_win_against_equal(self) -> None:
        # Win (actual=1) against equal opponent (expected=0.5)
        new = update_rating(1500, 0.5, 1.0, k=96)
        assert new == 1500 + 96 * 0.5  # 1548

    def test_loss_against_equal(self) -> None:
        new = update_rating(1500, 0.5, 0.0, k=96)
        assert new == 1500 - 96 * 0.5  # 1452

    def test_draw_against_equal(self) -> None:
        new = update_rating(1500, 0.5, 0.5, k=96)
        assert new == 1500

    def test_custom_k(self) -> None:
        new = update_rating(1500, 0.5, 1.0, k=32)
        assert new == 1500 + 32 * 0.5  # 1516


# ---------------------------------------------------------------------------
# EloTracker
# ---------------------------------------------------------------------------


class TestEloTracker:
    def test_initial_rating(self) -> None:
        tracker = EloTracker()
        assert tracker.get_rating("unknown") == INITIAL_RATING

    def test_register_checkpoint(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("best.pt")
        assert tracker.get_rating("best.pt") == INITIAL_RATING
        assert "best.pt" in tracker.known_checkpoints()

    def test_register_with_custom_rating(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("best.pt", rating=1800.0)
        assert tracker.get_rating("best.pt") == 1800.0

    def test_double_register_preserves_first(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("best.pt")
        tracker.register_checkpoint("best.pt")  # no-op
        assert tracker.get_rating("best.pt") == INITIAL_RATING

    def test_record_match_auto_registers(self) -> None:
        tracker = EloTracker()
        tracker.record_match("a.pt", "b.pt", 0.6, 100)
        assert tracker.get_rating("a.pt") != INITIAL_RATING
        assert tracker.get_rating("b.pt") != INITIAL_RATING

    def test_winner_gains_rating(self) -> None:
        tracker = EloTracker()
        tracker.record_match("a.pt", "b.pt", 0.6, 100)
        a_rating = tracker.get_rating("a.pt")
        b_rating = tracker.get_rating("b.pt")
        assert a_rating > INITIAL_RATING
        assert b_rating < INITIAL_RATING
        # Zero-sum: total rating change is zero.
        a_delta = a_rating - INITIAL_RATING
        b_delta = b_rating - INITIAL_RATING
        assert abs(a_delta + b_delta) < 1e-10

    def test_loser_loses_rating(self) -> None:
        tracker = EloTracker()
        tracker.record_match("a.pt", "b.pt", 0.3, 100)
        assert tracker.get_rating("a.pt") < INITIAL_RATING
        assert tracker.get_rating("b.pt") > INITIAL_RATING

    def test_equal_strength_no_change(self) -> None:
        tracker = EloTracker()
        tracker.record_match("a.pt", "b.pt", 0.5, 100)
        assert tracker.get_rating("a.pt") == INITIAL_RATING
        assert tracker.get_rating("b.pt") == INITIAL_RATING

    def test_match_record_created(self) -> None:
        tracker = EloTracker()
        record = tracker.record_match("a.pt", "b.pt", 0.55, 100)
        assert isinstance(record, MatchRecord)
        assert record.model_a == "a.pt"
        assert record.model_b == "b.pt"
        assert record.score_a == 0.55
        assert record.num_games == 100
        assert record.rating_a_before == INITIAL_RATING
        assert record.rating_b_before == INITIAL_RATING

    def test_games_played_tracked(self) -> None:
        tracker = EloTracker()
        tracker.record_match("a.pt", "b.pt", 0.5, 50)
        assert tracker.get_games_played("a.pt") == 50
        assert tracker.get_games_played("b.pt") == 50
        tracker.record_match("a.pt", "b.pt", 0.5, 30)
        assert tracker.get_games_played("a.pt") == 80

    def test_rating_history_snapshots(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("a.pt", iteration=1)
        tracker.record_match("a.pt", "b.pt", 0.6, 100, iteration=2)
        history = tracker.get_rating_history("a.pt")
        assert len(history) == 2
        assert history[0].iteration == 1
        assert history[1].iteration == 2
        assert history[0].rating == INITIAL_RATING

    def test_known_checkpoints(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("c.pt")
        tracker.register_checkpoint("a.pt")
        tracker.register_checkpoint("b.pt")
        known = tracker.known_checkpoints()
        assert known == ["a.pt", "b.pt", "c.pt"]  # sorted

    def test_summary_contains_names(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("a.pt")
        summary = tracker.summary()
        assert "a.pt" in summary
        assert str(INITIAL_RATING) in summary


class TestEloTrackerPersistence:
    def test_save_load_roundtrip(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("a.pt")
        tracker.register_checkpoint("b.pt")
        tracker.record_match("a.pt", "b.pt", 0.55, 100, iteration=1)
        tracker.record_match("a.pt", "b.pt", 0.52, 100, iteration=2)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tracker.save(path)
            loaded = EloTracker()
            loaded.load(path)

            assert loaded.get_rating("a.pt") == tracker.get_rating("a.pt")
            assert loaded.get_rating("b.pt") == tracker.get_rating("b.pt")
            assert loaded.get_games_played("a.pt") == tracker.get_games_played("a.pt")
            assert len(loaded.match_history) == len(tracker.match_history)
            assert len(loaded.rating_history) == len(tracker.rating_history)

            # Verify a specific match record.
            orig = tracker.match_history[0]
            loaded_match = loaded.match_history[0]
            assert loaded_match.model_a == orig.model_a
            assert loaded_match.score_a == orig.score_a
            assert loaded_match.rating_a_before == orig.rating_a_before
            assert loaded_match.rating_a_after == orig.rating_a_after
        finally:
            path.unlink(missing_ok=True)

    def test_save_is_valid_json(self) -> None:
        tracker = EloTracker()
        tracker.register_checkpoint("m.pt")
        tracker.record_match("m.pt", "n.pt", 0.6, 50)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tracker.save(path)
            data = json.loads(path.read_text())
            assert "ratings" in data
            assert "match_history" in data
            assert "rating_history" in data
            assert len(data["match_history"]) == 1
        finally:
            path.unlink(missing_ok=True)

    def test_load_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = Path(f.name)

        try:
            tracker = EloTracker()
            tracker.load(path)
            assert tracker.known_checkpoints() == []
            assert tracker.match_history == []
        finally:
            path.unlink(missing_ok=True)

    def test_load_missing_file(self) -> None:
        tracker = EloTracker()
        tracker.load("/nonexistent/path.json")  # should not crash
        assert tracker.known_checkpoints() == []
