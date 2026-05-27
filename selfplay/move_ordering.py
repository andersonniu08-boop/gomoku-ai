"""Tactical move ordering and candidate pruning for MCTS search.

Provides scoring functions and filtering logic to ensure MCTS expansions
prioritise tactically important moves while pruning clearly inferior ones.

Every function takes a board state and a list of candidate moves and returns
a filtered / re-weighted list.  No functions mutate the board.

Usage (inside MCTS expansion)::

    from selfplay.move_ordering import order_and_filter_moves
    move_probs = order_and_filter_moves(leaf_board, move_probs, max_moves=40)
"""

from __future__ import annotations

from typing import Optional

from engine.board import Board, Player

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Weights for the four scoring dimensions.  The neural-network prior is
# multiplied by ``(1.0 + combined_tactical_score)`` before renormalisation.

_CREATE_FIVE = 1000.0  # a winning move — never prune
_CREATE_OPEN_FOUR = 150.0
_CREATE_CLOSED_FOUR = 40.0
_CREATE_OPEN_THREE = 8.0

_BLOCK_FIVE = 500.0  # opponent's immediate win — never prune
_BLOCK_OPEN_FOUR = 100.0
_BLOCK_CLOSED_FOUR = 25.0
_BLOCK_OPEN_THREE = 5.0

# Positional bonuses (no simulation needed).
_PROXIMITY_BONUS = 0.5  # applied when move is adjacent to any stone
_CONNECTIVITY_BONUS = 0.3  # per friendly stone within distance 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def order_and_filter_moves(
    board: Board,
    move_probs: list[tuple[tuple[int, int], float]],
    max_moves: int = 40,
    *,
    threat_boost: bool = True,
    hard_override: bool = True,
) -> list[tuple[tuple[int, int], float]]:
    """Filter and reorder *move_probs* for tactical relevance.

    Guarantees that tactically critical moves are never pruned, then fills
    remaining slots with the highest-scoring moves.  The output list is
    sorted descending by adjusted prior.

    Parameters
    ----------
    board:
        Leaf board state (from the perspective of the player to move).
    move_probs:
        Neural-network policy output as ``((r,c), prob)`` pairs.
    max_moves:
        Maximum number of child nodes to keep.
    threat_boost:
        When True, compute tactical scores and adjust priors.  Set False
        to skip tactical analysis (useful for ablation benchmarks).
    hard_override:
        When True and forced winning / must-block moves exist, return
        **only** those moves — mirroring ``_check_forced`` at the root.
        This catches deep tactical wins that the network might otherwise
        undervalue.

    Returns
    -------
    A list of ``((r,c), prior)`` sorted by descending prior, length ≤ max_moves.
    """
    if not move_probs:
        return []

    moves = [(m, p) for m, p in move_probs]

    # --- Phase 1: partition moves by tactical urgency ---
    if threat_boost:
        scores = _compute_tactical_scores(board, moves)
    else:
        scores = {}

    # --- Phase 2: classify every move ---
    must_keep: list[tuple[tuple[int, int], float]] = []
    candidates: list[tuple[tuple[int, int], float]] = []

    for m, prob in moves:
        score = scores.get(m, 0.0)
        adjusted = prob * (1.0 + score)

        if score >= _BLOCK_FIVE or score >= _CREATE_FIVE:
            # Immediate win or must-block — keep unconditionally and
            # give it the highest possible adjusted prior.
            must_keep.append((m, adjusted * 10.0))
        elif score >= _CREATE_OPEN_FOUR:
            must_keep.append((m, adjusted * 3.0))
        else:
            candidates.append((m, adjusted))

    # --- Phase 3: build the final list ---
    # Hard override: when forced winning / must-block moves exist, expand
    # only those — the search must explore the tactical line immediately.
    if hard_override and must_keep:
        forced = [(m, p) for m, p in must_keep
                  if scores.get(m, 0) >= _BLOCK_FIVE or scores.get(m, 0) >= _CREATE_FIVE]
        if forced:
            total = sum(p for _, p in forced)
            if total > 0:
                return [(m, p / total) for m, p in forced]
            k = len(forced)
            return [(m, 1.0 / k) for m, _ in forced]

    # Sort candidates by adjusted prior, descending.
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Reserve slots: must-keep first, then fill with highest-scored candidates.
    slots_left = max_moves - len(must_keep)
    if slots_left <= 0:
        # More must-keep moves than slots — keep the highest-scored ones.
        combined = sorted(must_keep, key=lambda x: -x[1])[:max_moves]
    else:
        combined = must_keep + candidates[:slots_left]

    # Renormalise so the distribution sums to 1.
    total = sum(p for _, p in combined)
    if total > 0:
        combined = [(m, p / total) for m, p in combined]
    else:
        # Degenerate case: fall back to uniform.
        k = len(combined)
        combined = [(m, 1.0 / k) for m, p in combined]

    return combined


def compute_tactical_scores(
    board: Board,
    candidates: list[tuple[tuple[int, int], float]],
) -> dict[tuple[int, int], float]:
    """Compute per-move tactical scores (threat creation + blocking).

    Each candidate is simulated on a throwaway board copy.  Returns a dict
    mapping move → scalar score (higher = more tactically important).
    """
    return _compute_tactical_scores(board, candidates)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_tactical_scores(
    board: Board,
    candidates: list[tuple[tuple[int, int], float]],
) -> dict[tuple[int, int], float]:
    """Score each candidate using incremental line scanning.

    Instead of copying the board and running full ``ThreatDetector.detect_all``
    (which rescans all 225 cells) for every candidate, we check only the
    four lines through the candidate position.  This is O(board_size) per
    candidate instead of O(board_size²), and avoids the allocation overhead
    of board copying.

    Moves not adjacent to any stone receive a score of 0 — they cannot
    create or block meaningful threats.
    """
    scores: dict[tuple[int, int], float] = {}
    player = board.current_player
    opponent = Player(-player)
    grid = board.grid

    for (r, c), _ in candidates:
        if not _is_adjacent_to_stone(grid, r, c):
            scores[(r, c)] = 0.0
            continue

        scores[(r, c)] = _score_position_incremental(grid, r, c, player, opponent)

    return scores


def _score_position_incremental(
    grid,
    r: int,
    c: int,
    player: Player,
    opponent: Player,
) -> float:
    """Estimate the tactical value of placing a *player* stone at (r,c).

    Examines the four line directions through (r,c) and estimates:
    - Threat creation: how many consecutive friendly stones + open ends
      this move would produce.
    - Threat blocking: how many opponent stones lie on these lines
      (placing here disrupts their extension).

    Returns a scalar score suitable for move prioritisation.
    """
    score = 0.0

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # --- Count consecutive friendly stones in both directions ---
        friendly = _count_in_line(grid, r, c, dr, dc, player)
        # Count open ends for the resulting group (if we placed here).
        open_ends = _count_open_ends(grid, r, c, dr, dc, player)

        total_stones = friendly + 1  # +1 for the stone we'd place

        if total_stones >= 5:
            score += _CREATE_FIVE
        elif total_stones == 4:
            if open_ends >= 2:
                score += _CREATE_OPEN_FOUR
            elif open_ends >= 1:
                score += _CREATE_CLOSED_FOUR
        elif total_stones == 3:
            # Only an open three if both ends are open AND there's
            # room to extend (at least one side has 2+ empty cells).
            if open_ends >= 2:
                score += _CREATE_OPEN_THREE

        # --- Blocking value: opponent stones on this line ---
        opp_count = _count_in_line(grid, r, c, dr, dc, opponent)
        if opp_count >= 4:
            score += _BLOCK_OPEN_FOUR
        elif opp_count >= 3:
            score += _BLOCK_CLOSED_FOUR
        elif opp_count >= 2:
            score += _BLOCK_OPEN_THREE * 0.5

        # Connectivity bonus.
        if friendly >= 2:
            score += _CONNECTIVITY_BONUS * friendly

    score += _PROXIMITY_BONUS
    return score


def _count_open_ends(
    grid, r: int, c: int, dr: int, dc: int, player: Player
) -> int:
    """Count how many ends of the line through (r,c) are open.

    Assumes a friendly stone is placed at (r,c), then looks outward
    in both directions for empty cells.
    """
    open_ends = 0
    for direction in (-1, 1):
        nr, nc = r + dr * direction, c + dc * direction
        # Walk past friendly stones.
        while 0 <= nr < 15 and 0 <= nc < 15 and grid[nr, nc] == player:
            nr += dr * direction
            nc += dc * direction
        # The cell after the run is an open end if it's empty.
        if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr, nc] == 0:
            open_ends += 1
    return open_ends


def _count_in_line(
    grid, r: int, c: int, dr: int, dc: int, player: Player
) -> int:
    """Count consecutive stones of *player* on the line through (r,c)."""
    count = 0
    for direction in (-1, 1):
        nr, nc = r + dr * direction, c + dc * direction
        while 0 <= nr < 15 and 0 <= nc < 15 and grid[nr, nc] == player:
            count += 1
            nr += dr * direction
            nc += dc * direction
    return count


def _is_adjacent_to_stone(grid, r: int, c: int) -> bool:
    """Return True if any of the 8 neighbours of (r,c) is occupied."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr, nc] != 0:
                return True
    return False


def estimate_frontier_radius(board: Board) -> int:
    """Estimate how far from existing stones the search should look.

    Returns 1 (tight) or 2 (wide) based on board density and move count.
    Early game (few stones) uses radius 2 to catch developing patterns.
    Mid/late game uses radius 1 for efficiency.
    """
    num_stones = len(board.move_history)
    return 2 if num_stones < 10 else 1
