from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from .board import Board, Player


class ThreatType(IntEnum):
    """Threat severity ordered from weakest to strongest."""

    OPEN_THREE = 1
    CLOSED_FOUR = 2
    OPEN_FOUR = 3
    FIVE = 4


@dataclass(slots=True)
class Threat:
    """A single threat pattern found on the board.

    Attributes:
        threat_type: The kind of threat.
        player: Who owns it.
        stones: Positions of the stones forming the pattern.
        gap: The single empty cell within a split pattern, or None.
        open_ends: Empty cells adjacent to the pattern ends (for blocking).
        direction: (dr, dc) of the line.
    """

    threat_type: ThreatType
    player: Player
    stones: list[tuple[int, int]]
    direction: tuple[int, int]
    open_ends: list[tuple[int, int]]
    gap: Optional[tuple[int, int]] = None


class ThreatDetector:
    """Detects Gomoku threats for AI evaluation.

    Scans the four line directions from each stone, allowing at most one
    single-cell gap per segment, and classifies patterns into open-threes,
    closed-fours, open-fours, and fives.

    All methods are static — no state is kept across calls.
    """

    _DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def detect_all(board: Board, player: Player) -> list[Threat]:
        """Return every threat for *player*, ordered by severity (strongest last)."""
        threats: list[Threat] = []
        seen: set[tuple[int, int, int, int]] = set()

        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                if board.grid[r, c] != player:
                    continue
                for dr, dc in ThreatDetector._DIRECTIONS:
                    # Only scan from the logical start of a run.
                    pr, pc = r - dr, c - dc
                    if board._in_bounds(pr, pc) and board.grid[pr, pc] == player:
                        continue
                    key = (r, c, dr, dc)
                    if key in seen:
                        continue
                    threat = ThreatDetector._classify_segment(board, r, c, dr, dc, player)
                    if threat is not None:
                        for sr, sc in threat.stones:
                            seen.add((sr, sc, dr, dc))
                        threats.append(threat)

        # Sort so stronger threats come last (useful for iteration).
        threats.sort(key=lambda t: t.threat_type)
        return threats

    @staticmethod
    def has_five(board: Board, player: Player) -> bool:
        return ThreatDetector._has_type(board, player, ThreatType.FIVE)

    @staticmethod
    def has_open_four(board: Board, player: Player) -> bool:
        return ThreatDetector._has_type(board, player, ThreatType.OPEN_FOUR)

    @staticmethod
    def has_closed_four(board: Board, player: Player) -> bool:
        return ThreatDetector._has_type(board, player, ThreatType.CLOSED_FOUR)

    @staticmethod
    def has_open_three(board: Board, player: Player) -> bool:
        return ThreatDetector._has_type(board, player, ThreatType.OPEN_THREE)

    @staticmethod
    def has_double_threat(board: Board, player: Player) -> bool:
        """True when *player* has two threats that cannot both be blocked in one move.

        Includes: double open-three, open-four + open-three, open-four +
        closed-four, double closed-four.
        """
        threats = ThreatDetector.detect_all(board, player)
        n_open_four = 0
        n_closed_four = 0
        n_open_three = 0
        for t in threats:
            if t.threat_type == ThreatType.OPEN_FOUR:
                n_open_four += 1
            elif t.threat_type == ThreatType.CLOSED_FOUR:
                n_closed_four += 1
            elif t.threat_type == ThreatType.OPEN_THREE:
                n_open_three += 1

        if n_open_four >= 2:
            return True
        if n_open_four >= 1 and (n_open_three >= 1 or n_closed_four >= 1):
            return True
        if n_open_three >= 2:
            return True
        if n_closed_four >= 2:
            return True
        return False

    @staticmethod
    def count_threats(board: Board, player: Player) -> dict[ThreatType, int]:
        counts: dict[ThreatType, int] = {t: 0 for t in ThreatType}
        for t in ThreatDetector.detect_all(board, player):
            counts[t.threat_type] += 1
        return counts

    @staticmethod
    def evaluate(board: Board, player: Player) -> float:
        """Heuristic score from *player*'s perspective.

        Positive = *player* is ahead.  Weights are chosen so that a
        stronger threat always outweighs any number of weaker ones.
        """
        ours = ThreatDetector.count_threats(board, player)
        opp = ThreatDetector.count_threats(board, Player(-player))

        score = 0.0
        score += ours[ThreatType.FIVE] * 1_000_000
        score += ours[ThreatType.OPEN_FOUR] * 10_000
        score += ours[ThreatType.CLOSED_FOUR] * 1_000
        score += ours[ThreatType.OPEN_THREE] * 100

        score -= opp[ThreatType.FIVE] * 1_000_000
        score -= opp[ThreatType.OPEN_FOUR] * 10_000
        score -= opp[ThreatType.CLOSED_FOUR] * 1_000
        score -= opp[ThreatType.OPEN_THREE] * 100

        return float(score)

    # ------------------------------------------------------------------
    # Internal – segment scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_segment(
        board: Board,
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> tuple[list[tuple[int, int]], Optional[tuple[int, int]], list[tuple[int, int]]]:
        """Scan a line segment starting at *(row, col)*.

        Walks in the positive *(dr, dc)* direction, collecting consecutive
        stones of *player* and allowing at most **one** single-cell gap.

        Returns ``(stones, gap, open_ends)``:
            *stones*   – every position belonging to *player* in this segment.
            *gap*      – the empty cell sitting between two stones, or ``None``.
            *open_ends* – empty cells immediately outside each end of the segment.
        """
        stones: list[tuple[int, int]] = []
        r, c = row, col

        # --- first contiguous block ---
        while board._in_bounds(r, c) and board.grid[r, c] == player:
            stones.append((r, c))
            r += dr
            c += dc

        # --- optional single gap + second block ---
        gap: Optional[tuple[int, int]] = None
        if board._in_bounds(r, c) and board.grid[r, c] == 0:
            gap_r, gap_c = r, c
            r += dr
            c += dc
            if board._in_bounds(r, c) and board.grid[r, c] == player:
                gap = (gap_r, gap_c)
                while board._in_bounds(r, c) and board.grid[r, c] == player:
                    stones.append((r, c))
                    r += dr
                    c += dc

        # --- open ends ---
        open_ends: list[tuple[int, int]] = []

        # positive end (after last stone)
        last_r, last_c = stones[-1]
        pos_r, pos_c = last_r + dr, last_c + dc
        if board._in_bounds(pos_r, pos_c) and board.grid[pos_r, pos_c] == 0:
            open_ends.append((pos_r, pos_c))

        # negative end (before first stone)
        neg_r, neg_c = row - dr, col - dc
        if board._in_bounds(neg_r, neg_c) and board.grid[neg_r, neg_c] == 0:
            open_ends.append((neg_r, neg_c))

        return stones, gap, open_ends

    # ------------------------------------------------------------------
    # Internal – classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_segment(
        board: Board,
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> Optional[Threat]:
        stones, gap, open_ends = ThreatDetector._scan_segment(
            board, row, col, dr, dc, player
        )
        n = len(stones)
        num_open = len(open_ends)

        # Five (or more) — immediate win.
        if n >= 5:
            return Threat(
                threat_type=ThreatType.FIVE,
                player=player,
                stones=stones[:5],
                direction=(dr, dc),
                open_ends=[],
                gap=None,
            )

        if n == 4:
            return ThreatDetector._classify_four(stones, gap, open_ends, num_open,
                                                  dr, dc, player)

        if n == 3:
            return ThreatDetector._classify_three(
                board, stones, gap, open_ends, dr, dc, player
            )

        return None

    @staticmethod
    def _classify_four(
        stones: list[tuple[int, int]],
        gap: Optional[tuple[int, int]],
        open_ends: list[tuple[int, int]],
        num_open: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> Optional[Threat]:
        if gap is not None:
            # Split four (XXX_X, XX_XX, etc.).  The gap is the only
            # winning cell → at least a closed-four-level threat.
            if num_open >= 1:
                return Threat(
                    ThreatType.CLOSED_FOUR, player, stones, (dr, dc), open_ends, gap=gap
                )
        else:
            # Contiguous four.
            if num_open == 2:
                return Threat(ThreatType.OPEN_FOUR, player, stones, (dr, dc), open_ends)
            elif num_open == 1:
                return Threat(ThreatType.CLOSED_FOUR, player, stones, (dr, dc), open_ends)
        return None

    @staticmethod
    def _classify_three(
        board: Board,
        stones: list[tuple[int, int]],
        gap: Optional[tuple[int, int]],
        open_ends: list[tuple[int, int]],
        dr: int,
        dc: int,
        player: Player,
    ) -> Optional[Threat]:
        if gap is not None:
            # Split three — XX_X or X_XX.  The three stones are split by
            # a single gap.  Filling the gap turns it into a four.
            return ThreatDetector._classify_split_three(
                board, stones, gap, open_ends, dr, dc, player
            )
        else:
            # Contiguous three — XXX with both ends open.
            return ThreatDetector._classify_contiguous_three(
                board, stones, open_ends, dr, dc, player
            )

    @staticmethod
    def _classify_split_three(
        board: Board,
        stones: list[tuple[int, int]],
        gap: tuple[int, int],
        open_ends: list[tuple[int, int]],
        dr: int,
        dc: int,
        player: Player,
    ) -> Optional[Threat]:
        """Classify a split three (XX_X or X_XX)."""
        if len(open_ends) < 1:
            return None  # entirely blocked — not a threat

        # Determine which side has 2 stones and which has 1.
        # stones[0] is the first stone of the segment.
        first = stones[0]
        count_first_block = 1
        r, c = first[0] + dr, first[1] + dc
        while (r, c) != gap and board._in_bounds(r, c) and board.grid[r, c] == player:
            count_first_block += 1
            r += dr
            c += dc

        # For a split three to be open, placing at the gap must produce an
        # open four.  After filling the gap, the new four's ends are the
        # current open_ends on the outside plus any newly-exposed cells.
        # Since filling the gap joins the blocks, the open ends of the
        # resulting four are exactly the current open_ends.  For it to be
        # an *open* four we need both ends open.
        if len(open_ends) == 2:
            return Threat(
                ThreatType.OPEN_THREE, player, stones, (dr, dc), open_ends, gap=gap
            )

        # One end open — filling the gap gives a closed four, which is
        # still a threat but a weaker one.  We still classify it as an
        # open three because it forces a response (opponent must block the
        # gap or the open end).  However, the opponent CAN block the
        # single open end after we fill the gap, so this is not a
        # "true" open three.
        #
        # Check: can the resulting closed four be blocked in one move?
        # Yes — block the one open end.  So this is NOT an open three.
        # It's a "half" three at best.
        return None

    @staticmethod
    def _classify_contiguous_three(
        board: Board,
        stones: list[tuple[int, int]],
        open_ends: list[tuple[int, int]],
        dr: int,
        dc: int,
        player: Player,
    ) -> Optional[Threat]:
        """Classify a contiguous three-stone run (XXX)."""
        if len(open_ends) < 2:
            return None  # at least one end blocked — not an open three

        # Both immediate neighbours are empty.  For the three to be
        # "open" a player must be able to place at one end and produce an
        # open four.  That requires at least one side to have a *second*
        # empty cell beyond the immediate neighbour.
        first_r, first_c = stones[0]
        last_r, last_c = stones[-1]

        can_extend_left = False
        can_extend_right = False

        # Left extension check: cell two steps before first stone
        left2_r, left2_c = first_r - 2 * dr, first_c - 2 * dc
        # The cell immediately before first stone is open_ends[?]
        # We need: the cell before first is empty (true, it's in open_ends)
        # AND the cell after last is empty (true, it's in open_ends)
        # AND the cell two-before first is also empty.
        if board._in_bounds(left2_r, left2_c) and board.grid[left2_r, left2_c] == 0:
            can_extend_left = True

        # Right extension check: cell two steps after last stone
        right2_r, right2_c = last_r + 2 * dr, last_c + 2 * dc
        if board._in_bounds(right2_r, right2_c) and board.grid[right2_r, right2_c] == 0:
            can_extend_right = True

        if can_extend_left or can_extend_right:
            return Threat(
                ThreatType.OPEN_THREE, player, stones, (dr, dc), open_ends
            )

        return None

    # ------------------------------------------------------------------
    # Internal – quick existence checks
    # ------------------------------------------------------------------

    @staticmethod
    def _has_type(board: Board, player: Player, threat_type: ThreatType) -> bool:
        """Return True as soon as one threat of *threat_type* is found."""
        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                if board.grid[r, c] != player:
                    continue
                for dr, dc in ThreatDetector._DIRECTIONS:
                    pr, pc = r - dr, c - dc
                    if board._in_bounds(pr, pc) and board.grid[pr, pc] == player:
                        continue
                    threat = ThreatDetector._classify_segment(
                        board, r, c, dr, dc, player
                    )
                    if threat is not None and threat.threat_type == threat_type:
                        return True
        return False
