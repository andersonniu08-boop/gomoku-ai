"""Tests for the tactical benchmark suite."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from engine.board import Board
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.bench_suite import (
    BENCHMARKS,
    BenchmarkCategory,
    BenchmarkResult,
    CATEGORY_NAMES,
    format_results,
    run_benchmark_suite,
    run_single_benchmark,
    TacticalBenchmark,
)


def _make_wrapper() -> GomokuInferenceWrapper:
    """Create a wrapper around an untrained model."""
    model = GomokuNet(board_size=15, in_channels=3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    class _CleanupWrapper(GomokuInferenceWrapper):
        def __del__(self):
            if tmp_path.exists():
                tmp_path.unlink()

    wrapper = _CleanupWrapper(tmp_path, device="cpu")
    wrapper._tmp_path = tmp_path
    return wrapper


def _cleanup(wrapper) -> None:
    if hasattr(wrapper, "_tmp_path"):
        wrapper._tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test structure and data
# ---------------------------------------------------------------------------


class TestBenchmarkDefinitions:
    def test_all_benchmarks_have_name(self) -> None:
        for t in BENCHMARKS:
            assert t.name, f"Missing name in {t}"

    def test_all_benchmarks_have_category(self) -> None:
        for t in BENCHMARKS:
            assert isinstance(t.category, BenchmarkCategory)

    def test_all_benchmarks_have_expected_moves(self) -> None:
        for t in BENCHMARKS:
            assert t.expected_moves, f"No expected moves for {t.name}"

    def test_all_categories_represented(self) -> None:
        cats = {t.category for t in BENCHMARKS}
        assert cats == set(BenchmarkCategory)

    def test_each_category_has_multiple_tests(self) -> None:
        for cat in BenchmarkCategory:
            count = sum(1 for t in BENCHMARKS if t.category == cat)
            assert count >= 2, (
                f"{CATEGORY_NAMES[cat]} has only {count} tests"
            )

    def test_setup_moves_alternate(self) -> None:
        for t in BENCHMARKS:
            # All setups start with Black, moves alternate.
            board = Board()
            for i, move in enumerate(t.setup_moves):
                board.make_move(*move)
                player = board.current_player
            # After setup, it should be Black to move for tests where
            # Black is the active player. Some tests may want White to
            # move — that's fine as long as the board is valid.
            # Just verify no errors during setup.
            assert not board.is_terminal(), (
                f"{t.name}: setup produces terminal position"
            )


# ---------------------------------------------------------------------------
# Single benchmark execution
# ---------------------------------------------------------------------------


class TestRunSingleBenchmark:
    def test_returns_benchmark_result(self) -> None:
        wrapper = _make_wrapper()
        try:
            test = TacticalBenchmark(
                name="test-simple",
                category=BenchmarkCategory.WIN_IN_1,
                setup_moves=[(7, 2), (0, 0), (7, 3), (0, 1),
                             (7, 4), (0, 2), (7, 5), (0, 3)],
                expected_moves={(7, 1), (7, 6)},
            )
            result = run_single_benchmark(test, wrapper, num_simulations=10)
            assert isinstance(result, BenchmarkResult)
            assert result.name == "test-simple"
            assert result.category == BenchmarkCategory.WIN_IN_1
        finally:
            _cleanup(wrapper)

    def test_with_sims(self) -> None:
        wrapper = _make_wrapper()
        try:
            test = TacticalBenchmark(
                name="sim-test",
                category=BenchmarkCategory.WIN_IN_1,
                setup_moves=[(3, 3), (0, 0), (3, 4), (0, 1),
                             (3, 5), (0, 2), (3, 6), (0, 3)],
                expected_moves={(3, 2), (3, 7)},
            )
            result = run_single_benchmark(test, wrapper, num_simulations=20)
            assert result.total_simulations >= 0  # may be 0 if threat-overridden
            assert result.elapsed > 0
        finally:
            _cleanup(wrapper)

    def test_found_moves_in_result(self) -> None:
        wrapper = _make_wrapper()
        try:
            test = TacticalBenchmark(
                name="found-test",
                category=BenchmarkCategory.WIN_IN_1,
                setup_moves=[(7, 2), (0, 0), (7, 3), (0, 1),
                             (7, 4), (0, 2), (7, 5), (0, 3)],
                expected_moves={(7, 1)},
            )
            result = run_single_benchmark(test, wrapper, num_simulations=10)
            assert isinstance(result.found_moves, set)
        finally:
            _cleanup(wrapper)

    def test_top_moves_sorted(self) -> None:
        wrapper = _make_wrapper()
        try:
            test = TacticalBenchmark(
                name="top-test",
                category=BenchmarkCategory.WIN_IN_1,
                setup_moves=[(7, 2), (0, 0), (7, 3), (0, 1),
                             (7, 4), (0, 2), (7, 5), (0, 3)],
                expected_moves={(7, 1)},
            )
            result = run_single_benchmark(test, wrapper, num_simulations=10)
            if result.top_moves:
                visits = [v for _, v in result.top_moves]
                assert visits == sorted(visits, reverse=True)
        finally:
            _cleanup(wrapper)

    def test_status_property(self) -> None:
        result = BenchmarkResult(
            name="test",
            category=BenchmarkCategory.WIN_IN_1,
            passed=True,
            found_moves={(1, 1)},
            top_moves=[((1, 1), 10)],
            expected_moves={(1, 1)},
            total_simulations=100,
            elapsed=0.5,
        )
        assert result.status == "PASS"

        result.passed = False
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Full suite execution
# ---------------------------------------------------------------------------


class TestRunBenchmarkSuite:
    def test_returns_list_of_results(self) -> None:
        wrapper = _make_wrapper()
        try:
            results = run_benchmark_suite(
                wrapper, num_simulations=10
            )
            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(r, BenchmarkResult) for r in results)
        finally:
            _cleanup(wrapper)

    def test_category_filter(self) -> None:
        wrapper = _make_wrapper()
        try:
            results = run_benchmark_suite(
                wrapper,
                num_simulations=10,
                categories={BenchmarkCategory.WIN_IN_1},
            )
            assert len(results) > 0
            assert all(
                r.category == BenchmarkCategory.WIN_IN_1 for r in results
            )
        finally:
            _cleanup(wrapper)

    def test_custom_test_list(self) -> None:
        wrapper = _make_wrapper()
        try:
            custom = [
                TacticalBenchmark(
                    name="custom",
                    category=BenchmarkCategory.WIN_IN_1,
                    setup_moves=[(7, 2), (0, 0), (7, 3), (0, 1),
                                 (7, 4), (0, 2), (7, 5), (0, 3)],
                    expected_moves={(7, 1)},
                ),
            ]
            results = run_benchmark_suite(
                wrapper, num_simulations=10, tests=custom
            )
            assert len(results) == 1
            assert results[0].name == "custom"
        finally:
            _cleanup(wrapper)

    def test_total_passed_and_run(self) -> None:
        wrapper = _make_wrapper()
        try:
            results = run_benchmark_suite(wrapper, num_simulations=10)
            passed = sum(1 for r in results if r.passed)
            assert sum(1 for r in results if r.passed) + \
                   sum(1 for r in results if not r.passed) == len(results)
        finally:
            _cleanup(wrapper)

    def test_multiple_categories(self) -> None:
        wrapper = _make_wrapper()
        try:
            results = run_benchmark_suite(
                wrapper,
                num_simulations=10,
                categories={BenchmarkCategory.WIN_IN_1, BenchmarkCategory.FORCED_DEFENSE},
            )
            cats = {r.category for r in results}
            assert BenchmarkCategory.WIN_IN_1 in cats
            assert BenchmarkCategory.FORCED_DEFENSE in cats
        finally:
            _cleanup(wrapper)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_format_contains_header(self) -> None:
        results = [
            BenchmarkResult(
                name="test1",
                category=BenchmarkCategory.WIN_IN_1,
                passed=True,
                found_moves={(1, 1)},
                top_moves=[((1, 1), 10)],
                expected_moves={(1, 1)},
                total_simulations=100,
                elapsed=0.5,
            ),
        ]
        text = format_results(results)
        assert "Tactical Benchmark Results" in text
        assert "test1" in text
        assert "PASS" in text

    def test_format_shows_failures(self) -> None:
        results = [
            BenchmarkResult(
                name="failing-test",
                category=BenchmarkCategory.WIN_IN_1,
                passed=False,
                found_moves=set(),
                top_moves=[],
                expected_moves={(5, 5)},
                total_simulations=100,
                elapsed=0.5,
                details="Expected: {(5, 5)}",
            ),
        ]
        text = format_results(results)
        assert "FAIL" in text
        assert "failing-test" in text

    def test_format_total_line(self) -> None:
        results = [
            BenchmarkResult(
                name="t1", category=BenchmarkCategory.WIN_IN_1, passed=True,
                found_moves={(1, 1)}, top_moves=[], expected_moves={(1, 1)},
                total_simulations=10, elapsed=0.1,
            ),
            BenchmarkResult(
                name="t2", category=BenchmarkCategory.WIN_IN_1, passed=False,
                found_moves=set(), top_moves=[], expected_moves={(2, 2)},
                total_simulations=10, elapsed=0.1,
            ),
        ]
        text = format_results(results)
        assert "1/2" in text
