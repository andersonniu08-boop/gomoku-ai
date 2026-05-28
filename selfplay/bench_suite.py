"""Tactical benchmark suite for the Gomoku engine.

Each benchmark is a known position built by replaying ``setup_moves``
onto a fresh board.  The suite measures whether the engine finds the
expected move(s) with a given search budget, providing regression
detection across checkpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from engine.board import Board, Player
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class BenchmarkCategory(IntEnum):
    WIN_IN_1 = 1
    FORCED_DEFENSE = 2
    DOUBLE_THREAT = 3
    TACTICAL_SEQUENCE = 4
    OPENING = 5
    ENDGAME = 6


CATEGORY_NAMES: dict[BenchmarkCategory, str] = {
    BenchmarkCategory.WIN_IN_1: "Win in 1",
    BenchmarkCategory.FORCED_DEFENSE: "Forced Defense",
    BenchmarkCategory.DOUBLE_THREAT: "Double Threat",
    BenchmarkCategory.TACTICAL_SEQUENCE: "Tactical Sequence",
    BenchmarkCategory.OPENING: "Opening",
    BenchmarkCategory.ENDGAME: "Endgame",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkResult:
    """Outcome of a single tactical benchmark test."""

    name: str
    category: BenchmarkCategory
    passed: bool
    found_moves: set[tuple[int, int]] = field(default_factory=set)
    top_moves: list[tuple[tuple[int, int], float]] = field(default_factory=list)
    expected_moves: set[tuple[int, int]] = field(default_factory=set)
    total_simulations: int = 0
    elapsed: float = 0.0
    details: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TacticalBenchmark:
    """One tactical test position.

    The board is constructed by replaying ``setup_moves`` (alternating
    Black, White starting with Black) onto a fresh ``Board``.  The
    expected move(s) should contain all moves that solve the position
    (e.g. both block squares for an open-four).

    Attributes:
        name:           Unique test identifier.
        category:       Category for grouping in reports.
        setup_moves:    Sequence of (row, col) moves to reach the position.
                        Alternates Black, White starting with Black.
        expected_moves: Set of moves the engine should find (any one is
                        sufficient for a passing grade).
        description:    Human-readable test description.
    """

    name: str
    category: BenchmarkCategory
    setup_moves: list[tuple[int, int]] = field(default_factory=list)
    expected_moves: set[tuple[int, int]] = field(default_factory=set)
    description: str = ""


def build_board(benchmark: TacticalBenchmark) -> Board:
    """Replay ``setup_moves`` onto a fresh board and return it."""
    board = Board()
    for move in benchmark.setup_moves:
        board.make_move(*move)
    return board


# ---------------------------------------------------------------------------
# Built-in benchmarks
# ---------------------------------------------------------------------------

BENCHMARKS: list[TacticalBenchmark] = []

# --- Win-in-1: Horizontal ---
BENCHMARKS.append(TacticalBenchmark(
    name="win-horizontal",
    category=BenchmarkCategory.WIN_IN_1,
    setup_moves=[
        (7, 5), (0, 0),
        (7, 6), (0, 1),
        (7, 7), (0, 2),
        (7, 8), (0, 3),
    ],
    expected_moves={(7, 4), (7, 9)},
    description="Black wins horizontally at (7,4) or (7,9)",
))

# --- Win-in-1: Vertical ---
BENCHMARKS.append(TacticalBenchmark(
    name="win-vertical",
    category=BenchmarkCategory.WIN_IN_1,
    setup_moves=[
        (5, 3), (0, 0),
        (6, 3), (0, 1),
        (7, 3), (0, 2),
        (8, 3), (0, 3),
    ],
    expected_moves={(4, 3), (9, 3)},
    description="Black wins vertically at (4,3) or (9,3)",
))

# --- Win-in-1: Diagonal ---
BENCHMARKS.append(TacticalBenchmark(
    name="win-diagonal",
    category=BenchmarkCategory.WIN_IN_1,
    setup_moves=[
        (2, 2), (0, 0),
        (3, 3), (0, 1),
        (4, 4), (0, 2),
        (5, 5), (0, 3),
    ],
    expected_moves={(1, 1), (6, 6)},
    description="Black wins diagonal at (1,1) or (6,6)",
))

# --- Win-in-1: Anti-diagonal ---
BENCHMARKS.append(TacticalBenchmark(
    name="win-anti-diag",
    category=BenchmarkCategory.WIN_IN_1,
    setup_moves=[
        (2, 6), (0, 0),
        (3, 5), (0, 1),
        (4, 4), (0, 2),
        (5, 3), (0, 3),
    ],
    expected_moves={(1, 7), (6, 2)},
    description="Black wins anti-diagonal at (1,7) or (6,2)",
))

# --- Must-block: opponent open-four ---
BENCHMARKS.append(TacticalBenchmark(
    name="block-open-four",
    category=BenchmarkCategory.FORCED_DEFENSE,
    setup_moves=[
        (0, 0), (7, 4),
        (0, 2), (7, 5),
        (0, 3), (7, 6),
        (0, 4), (7, 7),
    ],
    expected_moves={(7, 3), (7, 8)},
    description="Black must block White open-four at (7,3) or (7,8)",
))

# --- Must-block: opponent split-four ---
BENCHMARKS.append(TacticalBenchmark(
    name="block-split-four",
    category=BenchmarkCategory.FORCED_DEFENSE,
    setup_moves=[
        (0, 0), (7, 3),
        (0, 2), (7, 4),
        (7, 5), (8, 4),
        (0, 3), (7, 6),
        (0, 4), (7, 7),
    ],
    expected_moves={(7, 5)},
    description="Black must block White split-four gap at (7,5)",
))

# --- Double threat ---
BENCHMARKS.append(TacticalBenchmark(
    name="double-threat",
    category=BenchmarkCategory.DOUBLE_THREAT,
    setup_moves=[
        (7, 3), (6, 4),
        (7, 4), (8, 4),
        (7, 6), (0, 0),
    ],
    expected_moves={(7, 5)},
    description="Black plays (7,5) creating double threat",
))

# --- Double threat: fork ---
BENCHMARKS.append(TacticalBenchmark(
    name="double-threat-fork",
    category=BenchmarkCategory.DOUBLE_THREAT,
    setup_moves=[
        (3, 3), (0, 1),
        (3, 4), (0, 2),
        (4, 4), (0, 3),
        (4, 5), (0, 4),
    ],
    expected_moves={(5, 5)},
    description="Black plays (5,5) creating double threat",
))


# --- Tactical sequence ---
BENCHMARKS.append(TacticalBenchmark(
    name="seq-fork-attack",
    category=BenchmarkCategory.TACTICAL_SEQUENCE,
    setup_moves=[
        (7, 3), (8, 4),
        (7, 4), (8, 5),
        (7, 5), (8, 6),
        (6, 4), (9, 7),
    ],
    expected_moves={(7, 6)},
    description="Black extends threat with (7,6) setting up a fork",
))

BENCHMARKS.append(TacticalBenchmark(
    name="seq-pincer",
    category=BenchmarkCategory.TACTICAL_SEQUENCE,
    setup_moves=[
        (3, 3), (4, 4),
        (3, 4), (4, 5),
        (5, 3), (6, 4),
        (5, 4), (6, 5),
    ],
    expected_moves={(7, 3)},
    description="Black pincer attack at (7,3) extending the line",
))

# --- Opening positions ---
BENCHMARKS.append(TacticalBenchmark(
    name="opening-empty",
    category=BenchmarkCategory.OPENING,
    setup_moves=[],
    expected_moves={(7, 7)},
    description="Empty board — center opening",
))

BENCHMARKS.append(TacticalBenchmark(
    name="opening-respond",
    category=BenchmarkCategory.OPENING,
    setup_moves=[(7, 7)],
    expected_moves={(6, 6), (6, 7), (6, 8), (7, 6), (7, 8), (8, 6), (8, 7), (8, 8)},
    description="Black responds near White's center stone",
))

# --- Endgame ---
BENCHMARKS.append(TacticalBenchmark(
    name="endgame-block-win",
    category=BenchmarkCategory.ENDGAME,
    setup_moves=[
        (7, 0), (7, 3),
        (7, 1), (7, 4),
        (7, 2), (7, 5),
        (0, 0), (7, 6),
        (0, 2),
    ],
    expected_moves={(7, 7), (7, 8)},
    description="Endgame: White must block Black's five threat",
))

BENCHMARKS.append(TacticalBenchmark(
    name="endgame-finish",
    category=BenchmarkCategory.ENDGAME,
    setup_moves=[
        (3, 3), (4, 4),
        (3, 4), (4, 5),
        (3, 5), (4, 6),
        (2, 2), (5, 7),
        (2, 3),
    ],
    expected_moves={(3, 6), (3, 2)},
    description="Endgame: Black one move from victory",
))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_single_benchmark(
    benchmark: TacticalBenchmark,
    wrapper: GomokuInferenceWrapper,
    *,
    num_simulations: int = 400,
) -> BenchmarkResult:
    """Run a single tactical benchmark test.

    The benchmark is considered PASSED if the top move from MCTS search
    (at temperature=0) is in the set of expected moves.
    """
    mcts = MCTS(
        wrapper,
        num_simulations=num_simulations,
        threat_override=True,
    )

    board = build_board(benchmark)
    t0 = time.monotonic()
    visit_probs = mcts.search(board)
    elapsed = time.monotonic() - t0

    if not visit_probs:
        return BenchmarkResult(
            name=benchmark.name,
            category=benchmark.category,
            passed=False,
            found_moves=set(),
            expected_moves=benchmark.expected_moves,
            elapsed=elapsed,
            details=f"Expected: {benchmark.expected_moves}, found: (none)",
        )

    sorted_moves = sorted(visit_probs, key=visit_probs.get, reverse=True)
    top_moves = [(m, visit_probs[m]) for m in sorted_moves[:5]]
    found = set(sorted_moves[:1])
    passed = bool(found & benchmark.expected_moves)

    details = ""
    if not passed:
        details = f"Expected: {benchmark.expected_moves}, found: {found}"

    return BenchmarkResult(
        name=benchmark.name,
        category=benchmark.category,
        passed=passed,
        found_moves=found,
        top_moves=top_moves,
        expected_moves=benchmark.expected_moves,
        total_simulations=sum(visit_probs.values()),
        elapsed=elapsed,
        details=details,
    )


def run_benchmark_suite(
    wrapper: GomokuInferenceWrapper,
    *,
    num_simulations: int = 400,
    categories: Optional[set[BenchmarkCategory]] = None,
    tests: Optional[list[TacticalBenchmark]] = None,
) -> list[BenchmarkResult]:
    """Run all (or filtered) tactical benchmarks."""
    benchmarks = tests if tests is not None else BENCHMARKS
    results: list[BenchmarkResult] = []
    for bm in benchmarks:
        if categories is not None and bm.category not in categories:
            continue
        result = run_single_benchmark(bm, wrapper, num_simulations=num_simulations)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_results(results: list[BenchmarkResult]) -> str:
    """Return a human-readable table of benchmark results."""
    if not results:
        return "(no results)\n"

    lines = [
        "Tactical Benchmark Results",
        "=" * 60,
        f"{'Test':<20s} {'Category':<18s} {'Status':<6s}  {'Details'}",
        "-" * 60,
    ]
    for r in results:
        cat = CATEGORY_NAMES.get(r.category, "Unknown")
        detail = r.details if not r.passed else ""
        lines.append(f"{r.name:<20s} {cat:<18s} {r.status:<6s}  {detail}")
    passed = sum(1 for r in results if r.passed)
    lines.append("-" * 60)
    lines.append(f"  {passed}/{len(results)} passed  ({passed / len(results):.0%})")
    return "\n".join(lines) + "\n"
