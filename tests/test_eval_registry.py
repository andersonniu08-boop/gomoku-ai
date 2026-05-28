"""Tests for the evaluation registry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from selfplay.bench_suite import (
    BENCHMARKS,
    BenchmarkCategory,
    BenchmarkResult,
    TacticalBenchmark,
)
from selfplay.eval_registry import BenchmarkRun, EvalRegistry


class TestBenchmarkRun:
    def test_defaults(self) -> None:
        run = BenchmarkRun(checkpoint_name="test.pt", timestamp=100.0)
        assert run.results == []
        assert run.total_passed == 0
        assert run.total_run == 0

    def test_counts(self) -> None:
        results = [
            BenchmarkResult(
                name="t1",
                category=BenchmarkCategory.WIN_IN_1,
                passed=True,
                found_moves={(1, 1)},
                top_moves=[],
                expected_moves={(1, 1)},
                total_simulations=10,
                elapsed=0.1,
            ),
            BenchmarkResult(
                name="t2",
                category=BenchmarkCategory.WIN_IN_1,
                passed=False,
                found_moves=set(),
                top_moves=[],
                expected_moves={(2, 2)},
                total_simulations=10,
                elapsed=0.1,
            ),
        ]
        run = BenchmarkRun(
            checkpoint_name="test.pt", timestamp=100.0, results=results
        )
        assert run.total_passed == 1
        assert run.total_run == 2


class TestEvalRegistry:
    def test_initial_state(self) -> None:
        reg = EvalRegistry()
        assert reg.benchmark_runs == []
        assert reg._baseline is None

    def test_register_checkpoint(self) -> None:
        reg = EvalRegistry()
        reg.register_checkpoint("a.pt")
        assert reg.elo.get_rating("a.pt") == 1500.0

    def test_record_match(self) -> None:
        reg = EvalRegistry()
        reg.register_checkpoint("a.pt")
        reg.register_checkpoint("b.pt")
        reg.record_match("a.pt", "b.pt", 0.55, 100)
        assert reg.elo.get_rating("a.pt") > 1500.0
        assert len(reg.elo.match_history) == 1

    def test_run_benchmarks_append(self) -> None:
        reg = EvalRegistry()
        # We need a wrapper. Since run_benchmarks needs a real wrapper,
        # test that the append mechanism and reporting work with
        # pre-constructed results.
        results = [
            BenchmarkResult(
                name="test",
                category=BenchmarkCategory.WIN_IN_1,
                passed=True,
                found_moves={(1, 1)},
                top_moves=[((1, 1), 10)],
                expected_moves={(1, 1)},
                total_simulations=100,
                elapsed=0.5,
            ),
        ]
        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="test.pt", timestamp=100.0, results=results
            )
        )
        assert len(reg.benchmark_runs) == 1
        assert "test" in reg.benchmark_summary()

    def test_benchmark_summary_empty(self) -> None:
        reg = EvalRegistry()
        summary = reg.benchmark_summary()
        assert "no benchmark runs" in summary

    def test_all_benchmark_runs_summary(self) -> None:
        reg = EvalRegistry()
        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="v1.pt", timestamp=100.0, results=[]
            )
        )
        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="v2.pt", timestamp=200.0, results=[]
            )
        )
        text = reg.all_benchmark_runs_summary()
        assert "v1.pt" in text
        assert "v2.pt" in text

    def test_set_baseline(self) -> None:
        reg = EvalRegistry()
        reg.benchmark_runs.append(
            BenchmarkRun(checkpoint_name="base", timestamp=100.0, results=[])
        )
        reg.set_baseline(0)
        assert reg._baseline is not None
        assert reg._baseline.checkpoint_name == "base"

    def test_set_baseline_raises_on_empty(self) -> None:
        reg = EvalRegistry()
        try:
            reg.set_baseline()
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_check_regressions_finds_none(self) -> None:
        reg = EvalRegistry()
        results_pass = [
            BenchmarkResult(
                name="t1",
                category=BenchmarkCategory.WIN_IN_1,
                passed=True,
                found_moves={(1, 1)},
                top_moves=[],
                expected_moves={(1, 1)},
                total_simulations=10,
                elapsed=0.1,
            ),
        ]
        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="base", timestamp=100.0, results=results_pass
            )
        )
        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="current", timestamp=200.0, results=results_pass
            )
        )
        reg.set_baseline(0)
        regressions = reg.check_regressions(1)
        assert regressions == []

    def test_check_regressions_detects_regression(self) -> None:
        reg = EvalRegistry()
        results_pass = [
            BenchmarkResult(
                name="t1",
                category=BenchmarkCategory.WIN_IN_1,
                passed=True,
                found_moves={(1, 1)},
                top_moves=[],
                expected_moves={(1, 1)},
                total_simulations=10,
                elapsed=0.1,
            ),
        ]
        results_fail = [
            BenchmarkResult(
                name="t1",
                category=BenchmarkCategory.WIN_IN_1,
                passed=False,
                found_moves=set(),
                top_moves=[],
                expected_moves={(1, 1)},
                total_simulations=10,
                elapsed=0.1,
            ),
        ]

        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="base", timestamp=100.0, results=results_pass
            )
        )
        reg.benchmark_runs.append(
            BenchmarkRun(
                checkpoint_name="bad", timestamp=200.0, results=results_fail
            )
        )
        reg.set_baseline(0)
        regressions = reg.check_regressions(1)
        assert len(regressions) == 1
        name, details = regressions[0]
        assert name == "t1"
        assert "FAIL" in details

    def test_regression_summary_no_baseline(self) -> None:
        reg = EvalRegistry()
        summary = reg.regression_summary()
        assert "no baseline" in summary

    def test_regression_summary_clean(self) -> None:
        reg = EvalRegistry()
        result = BenchmarkResult(
            name="test", category=BenchmarkCategory.WIN_IN_1, passed=True,
            found_moves={(1, 1)}, top_moves=[], expected_moves={(1, 1)},
            total_simulations=10, elapsed=0.1,
        )
        reg.benchmark_runs.append(
            BenchmarkRun(checkpoint_name="base", timestamp=100.0, results=[result])
        )
        reg.benchmark_runs.append(
            BenchmarkRun(checkpoint_name="cur", timestamp=200.0, results=[result])
        )
        reg.set_baseline(0)
        summary = reg.regression_summary()
        assert "No regressions" in summary

    def test_full_report(self) -> None:
        reg = EvalRegistry()
        reg.register_checkpoint("best.pt")
        reg.register_checkpoint("latest.pt")
        reg.record_match("latest.pt", "best.pt", 0.55, 100, iteration=1)
        report = reg.full_report()
        assert "Evaluation Report" in report
        assert "best.pt" in report
        assert "latest.pt" in report
        assert "55.0%" in report

    def test_save_load_roundtrip(self) -> None:
        reg = EvalRegistry()
        reg.register_checkpoint("a.pt")
        reg.register_checkpoint("b.pt")
        reg.record_match("a.pt", "b.pt", 0.6, 100, iteration=1)

        result = BenchmarkResult(
            name="test", category=BenchmarkCategory.WIN_IN_1, passed=True,
            found_moves={(1, 1)}, top_moves=[((1, 1), 10)],
            expected_moves={(1, 1)}, total_simulations=100, elapsed=0.5,
        )
        reg.benchmark_runs.append(
            BenchmarkRun(checkpoint_name="v1.pt", timestamp=100.0, results=[result])
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        elo_path = path.with_suffix(".elo.json")

        try:
            reg.save(path)

            loaded = EvalRegistry()
            loaded.load(path)

            assert loaded.elo.get_rating("a.pt") == reg.elo.get_rating("a.pt")
            assert len(loaded.benchmark_runs) == 1
            assert loaded.benchmark_runs[0].checkpoint_name == "v1.pt"
            assert loaded.benchmark_runs[0].results[0].passed is True
        finally:
            path.unlink(missing_ok=True)
            elo_path.unlink(missing_ok=True)
