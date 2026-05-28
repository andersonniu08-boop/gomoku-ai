"""Elo rating system for the Gomoku engine.

Tracks per-checkpoint ratings, match history, and rating trends.
Uses the standard Elo formula with a logistic distribution:

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

Ratings are updated once per multi-game match using the aggregate score,
which converges to the same result as per-game updates in expectation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Default starting rating for new checkpoints.
INITIAL_RATING = 1500.0

# Development coefficient — applied once per match (not per game).
# K=96 means a 55% win rate over 100 games against an equal opponent
# gives a rating change of ~4.8 points, which keeps ratings meaningful
# over many evaluation cycles without excessive volatility.
_K_FACTOR = 96.0


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B, ∈ (0, 1)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_rating(
    rating: float,
    expected: float,
    actual: float,
    k: float = _K_FACTOR,
) -> float:
    """Return ``rating`` adjusted after observing ``actual`` vs ``expected``."""
    return rating + k * (actual - expected)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MatchRecord:
    """One evaluation match between two model checkpoints."""

    model_a: str
    model_b: str
    score_a: float         # win rate for model_a (0..1, draws = 0.5)
    num_games: int
    timestamp: float
    rating_a_before: float
    rating_b_before: float
    rating_a_after: float
    rating_b_after: float

    @property
    def delta_a(self) -> float:
        return self.rating_a_after - self.rating_a_before

    @property
    def delta_b(self) -> float:
        return self.rating_b_after - self.rating_b_before


@dataclass(slots=True)
class RatingSnapshot:
    """A single point in a checkpoint's rating history."""

    name: str
    rating: float
    iteration: int
    timestamp: float


# ---------------------------------------------------------------------------
# EloTracker
# ---------------------------------------------------------------------------


class EloTracker:
    """Track Elo ratings for model checkpoints over time.

    Typical usage::

        tracker = EloTracker()
        if elo_path.exists():
            tracker.load(elo_path)

        tracker.register_checkpoint("latest.pt")
        tracker.register_checkpoint("best.pt")

        # After a match (100 games, latest scored 0.55):
        tracker.record_match("latest.pt", "best.pt", 0.55, 100)

        tracker.save(elo_path)
    """

    def __init__(self, k_factor: float = _K_FACTOR) -> None:
        self.k_factor = k_factor
        self._ratings: dict[str, float] = {}
        self._games_played: dict[str, int] = {}
        self.match_history: list[MatchRecord] = []
        self.rating_history: list[RatingSnapshot] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_checkpoint(
        self,
        name: str,
        *,
        iteration: int = 0,
        rating: Optional[float] = None,
    ) -> None:
        """Register a checkpoint, preserving existing rating if already known."""
        if name in self._ratings:
            return  # already registered
        self._ratings[name] = rating if rating is not None else INITIAL_RATING
        self._games_played[name] = 0
        self.rating_history.append(
            RatingSnapshot(name, self._ratings[name], iteration, time.time())
        )

    # ------------------------------------------------------------------
    # Match recording
    # ------------------------------------------------------------------

    def record_match(
        self,
        model_a: str,
        model_b: str,
        score_a: float,
        num_games: int,
        *,
        iteration: int = 0,
    ) -> MatchRecord:
        """Update ratings after a match between two models.

        Args:
            model_a:  Name of the first checkpoint (e.g. ``"latest.pt"``).
            model_b:  Name of the second checkpoint.
            score_a:  Win rate for *model_a* across *num_games* games.
            num_games: Number of games played in the match.

        Returns:
            The :class:`MatchRecord` that was appended to ``match_history``.
        """
        # Auto-register if either model is unknown (avoids crashes from
        # stale elo_state files or manual CLI usage).
        if model_a not in self._ratings:
            self.register_checkpoint(model_a)
        if model_b not in self._ratings:
            self.register_checkpoint(model_b)

        rating_a = self._ratings[model_a]
        rating_b = self._ratings[model_b]
        score_b = 1.0 - score_a

        expected_a = expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        new_a = update_rating(rating_a, expected_a, score_a, self.k_factor)
        new_b = update_rating(rating_b, expected_b, score_b, self.k_factor)

        self._ratings[model_a] = new_a
        self._ratings[model_b] = new_b
        self._games_played[model_a] += num_games
        self._games_played[model_b] += num_games

        now = time.time()
        record = MatchRecord(
            model_a=model_a,
            model_b=model_b,
            score_a=score_a,
            num_games=num_games,
            timestamp=now,
            rating_a_before=rating_a,
            rating_b_before=rating_b,
            rating_a_after=new_a,
            rating_b_after=new_b,
        )
        self.match_history.append(record)

        self.rating_history.append(
            RatingSnapshot(model_a, new_a, iteration, now)
        )
        self.rating_history.append(
            RatingSnapshot(model_b, new_b, iteration, now)
        )

        return record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_rating(self, name: str) -> float:
        """Return current rating for *name*, or ``INITIAL_RATING``."""
        return self._ratings.get(name, INITIAL_RATING)

    def get_games_played(self, name: str) -> int:
        return self._games_played.get(name, 0)

    def get_rating_history(self, name: str) -> list[RatingSnapshot]:
        """Return all historical rating points for *name*, ordered by time."""
        return [s for s in self.rating_history if s.name == name]

    def known_checkpoints(self) -> list[str]:
        return sorted(self._ratings.keys())

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line summary of all known ratings (sorted descending)."""
        lines = ["Elo Ratings", "=" * 60]
        lines.append(
            f"{'Checkpoint':<40s} {'Rating':>8s} {'Games':>6s}  {'Last Match':>20s}"
        )
        lines.append("-" * 76)

        sorted_names = sorted(
            self._ratings, key=lambda n: self._ratings[n], reverse=True
        )
        for name in sorted_names:
            rating = self._ratings[name]
            games = self._games_played[name]
            last = ""
            for m in reversed(self.match_history):
                if m.model_a == name or m.model_b == name:
                    last = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.timestamp))
                    break
            lines.append(
                f"{name:<40s} {rating:>8.1f} {games:>6d}  {last:>20s}"
            )
        return "\n".join(lines) + "\n"

    def recent_matches(self, n: int = 10) -> str:
        """Return a summary of the *n* most recent matches."""
        recent = self.match_history[-n:]
        lines = [f"Recent Matches (last {len(recent)})", "=" * 60]
        for m in reversed(recent):
            lines.append(
                f"  {m.model_a} vs {m.model_b}: "
                f"{m.score_a:.1%} over {m.num_games} games  "
                f"(Δa={m.delta_a:+.1f}, Δb={m.delta_b:+.1f})"
            )
        return "\n".join(lines) + "\n" if recent else "(no matches)\n"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize to JSON."""
        data = {
            "ratings": dict(self._ratings),
            "games_played": dict(self._games_played),
            "match_history": [
                {
                    "model_a": m.model_a,
                    "model_b": m.model_b,
                    "score_a": m.score_a,
                    "num_games": m.num_games,
                    "timestamp": m.timestamp,
                    "rating_a_before": m.rating_a_before,
                    "rating_b_before": m.rating_b_before,
                    "rating_a_after": m.rating_a_after,
                    "rating_b_after": m.rating_b_after,
                }
                for m in self.match_history
            ],
            "rating_history": [
                {
                    "name": s.name,
                    "rating": s.rating,
                    "iteration": s.iteration,
                    "timestamp": s.timestamp,
                }
                for s in self.rating_history
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Deserialize from JSON written by :meth:`save`."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        self._ratings = data.get("ratings", {})
        self._games_played = data.get("games_played", {})

        self.match_history = [
            MatchRecord(
                model_a=m["model_a"],
                model_b=m["model_b"],
                score_a=m["score_a"],
                num_games=m["num_games"],
                timestamp=m["timestamp"],
                rating_a_before=m["rating_a_before"],
                rating_b_before=m["rating_b_before"],
                rating_a_after=m["rating_a_after"],
                rating_b_after=m["rating_b_after"],
            )
            for m in data.get("match_history", [])
        ]

        self.rating_history = [
            RatingSnapshot(
                name=s["name"],
                rating=s["rating"],
                iteration=s["iteration"],
                timestamp=s["timestamp"],
            )
            for s in data.get("rating_history", [])
        ]
