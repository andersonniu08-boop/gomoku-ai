"""Evaluation registry — ties together Elo tracking, tactical benchmarks,
and regression detection for the Gomoku engine.

The :class:`EvalRegistry` coordinates:
- Elo rating updates after head-to-head evaluation matches.
- Tactical benchmark runs against any checkpoint.
- Regression detection by comparing current benchmark results against a
  stored baseline.
- Summary reports covering all three dimensions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from neural.wrapper import GomokuInferenceWrapper
from selfplay.bench_suite import (
    BENCHMARKS,
    BenchmarkCategory,
    BenchmarkResult,
    CATEGORY_NAMES,
    format_results,
    run_benchmark_suite,
    run_single_benchmark,
)
from selfplay.elo import EloTracker


@dataclass(slots=True)
class BenchmarkRun:
    """A snapshot of benchmark results for a specific checkpoint at a point in time."""

    checkpoint_name: str
    timestamp: float
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_run(self) -> int:
        return len(self.results)


class EvalRegistry:
    """Central registry for all evaluation and benchmarking data.

    Typical usage::

        registry = EvalRegistry()
        if reg_path.exists():
            registry.load(reg_path)

        # After a training evaluation:
        registry.record_match("latest.pt", "best.pt", 0.55, 100)

        # Run tactical benchmarks:
        results = registry.run_benchmarks(wrapper)
        print(registry.benchmark_summary())

        registry.save(reg_path)
    """

    def __init__(self, k_factor: float = 96.0) -> None:
        self.elo = EloTracker(k_factor=k_factor)
        self.benchmark_runs: list[BenchmarkRun] = []

        # Optional baseline path for regression detection.
        self._baseline: Optional[BenchmarkRun] = None

    # ------------------------------------------------------------------
    # Elo tracking (delegates to EloTracker)
    # ------------------------------------------------------------------

    def register_checkpoint(
        self, name: str, *, iteration: int = 0, rating: Optional[float] = None
    ) -> None:
        self.elo.register_checkpoint(name, iteration=iteration, rating=rating)

    def record_match(
        self,
        model_a: str,
        model_b: str,
        score_a: float,
        num_games: int,
        *,
        iteration: int = 0,
    ) -> None:
        self.elo.record_match(model_a, model_b, score_a, num_games, iteration=iteration)

    # ------------------------------------------------------------------
    # Tactical benchmarks
    # ------------------------------------------------------------------

    def run_benchmarks(
        self,
        wrapper: GomokuInferenceWrapper,
        *,
        checkpoint_name: str = "current",
        num_simulations: int = 400,
        categories: Optional[set[BenchmarkCategory]] = None,
        tests: Optional[list] = None,
    ) -> list[BenchmarkResult]:
        """Run tactical benchmarks against *wrapper* and store the run."""
        results = run_benchmark_suite(
            wrapper,
            num_simulations=num_simulations,
            categories=categories,
            tests=tests,
        )
        self.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name=checkpoint_name,
                timestamp=time.time(),
                results=results,
            )
        )
        return results

    def benchmark_summary(self) -> str:
        """Return a formatted summary of the most recent benchmark run."""
        if not self.benchmark_runs:
            return "(no benchmark runs)\n"
        return format_results(self.benchmark_runs[-1].results)

    def all_benchmark_runs_summary(self) -> str:
        """Return a line per benchmark run (for quick trend overview)."""
        lines = ["Benchmark Runs", "=" * 60]
        lines.append(
            f"{'Checkpoint':<30s} {'Passed':>8s} {'Total':>6s} {'Date':>20s}"
        )
        lines.append("-" * 64)
        for run in self.benchmark_runs:
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(run.timestamp))
            lines.append(
                f"{run.checkpoint_name:<30s} {run.total_passed:>4d}/{run.total_run:<2d}"
                f" {date:>20s}"
            )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    def set_baseline(self, run_index: int = -1) -> None:
        """Set the baseline for regression detection.

        Args:
            run_index: Index into ``benchmark_runs`` (default: most recent).
        """
        if not self.benchmark_runs:
            raise ValueError("No benchmark runs available to set as baseline")
        self._baseline = self.benchmark_runs[run_index]
        self._baseline_index = run_index

    def check_regressions(
        self, run_index: int = -1
    ) -> list[tuple[str, str]]:
        """Compare *run_index* against the baseline.

        Returns:
            List of ``(test_name, description)`` tuples for each regression
            (a test that passed in the baseline but failed in the comparison).
        """
        if self._baseline is None:
            raise ValueError("No baseline set. Call set_baseline() first.")

        target = self.benchmark_runs[run_index]

        baseline_map: dict[str, BenchmarkResult] = {
            r.name: r for r in self._baseline.results
        }

        regressions: list[tuple[str, str]] = []
        for result in target.results:
            base = baseline_map.get(result.name)
            if base is not None and base.passed and not result.passed:
                regressions.append(
                    (result.name, f"was PASS, now FAIL ({result.details})")
                )

        return regressions

    def regression_summary(self, run_index: int = -1) -> str:
        """Return a human-readable regression report."""
        if self._baseline is None:
            return "(no baseline set — call set_baseline())\n"

        regressions = self.check_regressions(run_index)
        lines = ["Regression Check", "=" * 60]
        if regressions:
            for name, details in regressions:
                lines.append(f"  [REGRESSION] {name}")
                lines.append(f"    {details}")
            lines.append(f"\n  {len(regressions)} regression(s) found.")
        else:
            lines.append("  No regressions detected.\n")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self, *, show_recent_matches: int = 10) -> str:
        """Generate a comprehensive evaluation report."""
        parts = [
            "=" * 60,
            "NeuralGomoku — Evaluation Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            self.elo.summary(),
        ]

        if self.elo.match_history:
            parts.append(self.elo.recent_matches(show_recent_matches))

        parts.append(self.all_benchmark_runs_summary())

        if self.benchmark_runs:
            parts.append(self.benchmark_summary())

        if self._baseline is not None:
            parts.append(self.regression_summary())

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the entire registry state."""
        # Save Elo state alongside.
        elo_path = Path(path).with_suffix(".elo.json")
        self.elo.save(elo_path)

        # Save benchmark runs.
        data = {
            "benchmark_runs": [
                {
                    "checkpoint_name": run.checkpoint_name,
                    "timestamp": run.timestamp,
                    "results": [
                        {
                            "name": r.name,
                            "category": int(r.category),
                            "passed": r.passed,
                            "found_moves": list(r.found_moves),
                            "expected_moves": list(r.expected_moves),
                            "total_simulations": r.total_simulations,
                            "elapsed": r.elapsed,
                        }
                        for r in run.results
                    ],
                }
                for run in self.benchmark_runs
            ],
            "baseline_index": (
                self._baseline_index if hasattr(self, "_baseline_index") else -1
            ),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Deserialize from JSON written by :meth:`save`."""
        # Load Elo state.
        elo_path = Path(path).with_suffix(".elo.json")
        if elo_path.exists():
            self.elo.load(elo_path)

        data = json.loads(Path(path).read_text())

        # Build a lookup from benchmark result dicts back to BenchmarkResult.
        from selfplay.bench_suite import (
            TacticalBenchmark,
        )  # for type matching

        test_lookup = {t.name: t for t in BENCHMARKS}

        self.benchmark_runs = []
        for run_data in data.get("benchmark_runs", []):
            results = []
            for r_data in run_data.get("results", []):
                test = test_lookup.get(r_data["name"])
                results.append(
                    BenchmarkResult(
                        name=r_data["name"],
                        category=BenchmarkCategory(r_data["category"]),
                        passed=r_data["passed"],
                        found_moves=set(
                            tuple(m) for m in r_data.get("found_moves", [])
                        ),
                        top_moves=[],
                        expected_moves=set(
                            tuple(m) for m in r_data.get("expected_moves", [])
                        ),
                        total_simulations=r_data.get("total_simulations", 0),
                        elapsed=r_data.get("elapsed", 0.0),
                    )
                )
            self.benchmark_runs.append(
                BenchmarkRun(
                    checkpoint_name=run_data["checkpoint_name"],
                    timestamp=run_data["timestamp"],
                    results=results,
                )
            )

        baseline_idx = data.get("baseline_index", -1)
        if baseline_idx >= 0 and baseline_idx < len(self.benchmark_runs):
            self._baseline = self.benchmark_runs[baseline_idx]
            self._baseline_index = baseline_idx
