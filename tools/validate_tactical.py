"""Validation script — demonstrates tactical bug fixes with concrete scenarios.

Each scenario was a known failure mode before the tactical-override fix.
Run with: python tools/validate_tactical.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from engine.board import Board, Player
from engine.threats import ThreatDetector, ThreatType
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS


def _make_wrapper():
    """Create a wrapper around an untrained model (same as test helper)."""
    model = GomokuNet(board_size=15, in_channels=3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    class _CleanupWrapper(GomokuInferenceWrapper):
        def __del__(self):
            if tmp_path.exists():
                tmp_path.unlink()

    return _CleanupWrapper(tmp_path, device="cpu"), tmp_path


def _place_many(board: Board, moves: list[tuple[int, int]]) -> None:
    for r, c in moves:
        board.make_move(r, c)


def describe_move(move: tuple[int, int], dist: dict | None = None) -> str:
    if dist and move in dist:
        return f"{move} (prob={dist[move]:.2f})"
    return str(move)


def run_scenario(name: str, setup_moves: list[tuple[int, int]],
                expected_moves: set[tuple[int, int]], description: str) -> bool:
    """Run one tactical scenario and report pass/fail."""
    wrapper, tmp = _make_wrapper()
    try:
        mcts = MCTS(wrapper, num_simulations=10, threat_override=True)
        board = Board()
        _place_many(board, setup_moves)

        dist = mcts.search(board)
        actual = set(dist.keys())
        overlap = actual & expected_moves

        status = "PASS" if overlap == expected_moves else (
            "PARTIAL" if overlap else "FAIL"
        )

        print(f"\n{'='*60}")
        print(f"[{status}] {name}")
        print(f"  {description}")
        print(f"  Expected: {expected_moves}")
        print(f"  Got:      {actual}")
        if overlap and overlap != expected_moves:
            print(f"  Overlap:  {overlap}")

        if status == "FAIL":
            # Diagnose
            print(f"  --- Diagnosis ---")
            print(f"  Board:\n{board}")
            threats = ThreatDetector.detect_all(board, board.current_player)
            opp_threats = ThreatDetector.detect_all(board, Player(-board.current_player))
            print(f"  Our threats: {[(t.threat_type.name, t.stones, t.open_ends, t.gap) for t in threats]}")
            print(f"  Opp threats: {[(t.threat_type.name, t.stones, t.open_ends, t.gap) for t in opp_threats]}")

        return status == "PASS"
    finally:
        tmp.unlink()


def main():
    print("Gomoku Tactical Override — Validation Suite")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    # ------------------------------------------------------------------
    # Scenario 1: Immediate win — open four
    # ------------------------------------------------------------------
    results.append((
        "Our open four → immediate win",
        run_scenario(
            "Our open four",
            [(7, 2), (0, 0), (7, 3), (0, 1), (7, 4), (0, 2), (7, 5), (0, 3)],
            {(7, 1), (7, 6)},
            "Black has X at (7,2)-(7,5) — both ends win.",
        ),
    ))

    # ------------------------------------------------------------------
    # Scenario 2: Must block opponent open four
    # ------------------------------------------------------------------
    results.append((
        "Block opponent open four",
        run_scenario(
            "Block opponent open four",
            [(10, 0), (7, 2), (12, 3), (7, 3), (10, 6), (7, 4), (12, 9), (7, 5)],
            {(7, 1), (7, 6)},
            "White has open four at (7,2)-(7,5). Black MUST block at either end.",
        ),
    ))

    # ------------------------------------------------------------------
    # Scenario 3 (CRITICAL FIX): Must block opponent CLOSED_FOUR
    # ------------------------------------------------------------------
    results.append((
        "Block opponent CLOSED_FOUR (was the key bug)",
        run_scenario(
            "Block opponent CLOSED_FOUR",
            [(7, 1), (7, 2), (8, 0), (7, 3), (8, 2), (7, 4), (8, 4), (7, 5)],
            {(7, 6)},
            "White has closed four at (7,2)-(7,5), left blocked. Right end (7,6) "
            "MUST be blocked or White wins next turn. Previously this was IGNORED.",
        ),
    ))

    # ------------------------------------------------------------------
    # Scenario 4: Our split closed four → only gap wins
    # ------------------------------------------------------------------
    results.append((
        "Split closed four (XX_XX) — only gap wins",
        run_scenario(
            "Split closed four gap detection",
            [(7, 2), (0, 0), (7, 3), (0, 1), (7, 5), (0, 2), (7, 6), (0, 3)],
            {(7, 4)},
            "Black has XX_XX at cols 2,3,5,6. Only the gap (7,4) creates five.",
        ),
    ))

    # ------------------------------------------------------------------
    # Scenario 5 (CRITICAL FIX): Block opponent split CLOSED_FOUR
    # ------------------------------------------------------------------
    results.append((
        "Block opponent split CLOSED_FOUR gap",
        run_scenario(
            "Block opponent split CLOSED_FOUR",
            [(10, 0), (7, 2), (12, 3), (7, 3), (10, 6), (7, 5), (12, 9), (7, 6)],
            {(7, 4)},
            "White has XX_XX at (7,2)(7,3)(7,5)(7,6). Black MUST block the gap "
            "at (7,4) or White creates five on next turn. Previously IGNORED.",
        ),
    ))

    # ------------------------------------------------------------------
    # Scenario 6: Winning move takes priority over blocking
    # ------------------------------------------------------------------
    results.append((
        "Win takes priority over block",
        run_scenario(
            "Win before block",
            [(7, 2), (0, 0), (7, 3), (0, 1), (7, 4), (0, 2), (7, 5), (7, 0),
             (8, 0), (7, 1)],
            {(7, 6)},
            "Black has open four at (7,2)-(7,5) plus a block at (7,1). "
            "Black should WIN at (7,6), not continue blocking.",
        ),
    ))

    # ------------------------------------------------------------------
    # Scenario 7: Double-threat detection
    # ------------------------------------------------------------------
    results.append((
        "Double threat creation detected",
        run_scenario(
            "Double threat",
            [(7, 3), (10, 0), (7, 4), (12, 3), (6, 4), (10, 7), (8, 4), (12, 9)],
            {(7, 5)},
            "Playing (7,5) creates two open threes (horizontal + vertical) = "
            "double threat. Opponent cannot block both → forced win.",
        ),
    ))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
    print(f"\n  {passed}/{len(results)} passed, {failed} failed")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
