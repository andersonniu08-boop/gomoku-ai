"""Deterministic tactical solver for Gomoku positions.

Pure game-logic analysis with no neural-network or MCTS dependency.
Resolves threat patterns into concrete moves (wins, blocks, threat
creation) and performs shallow forced-sequence search.

All methods are static — the solver is stateless and callable from any
layer that imports from ``engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .board import Board, Player
from .threats import ThreatDetector, ThreatType, Threat


@dataclass(slots=True)
class TacticalAnalysis:
    """Structured tactical assessment of a position.

    Every field is from the perspective of ``board.current_player`` at
    the time :meth:`TacticalSolver.analyze` was called.
    """

    # ---- move sets -------------------------------------------------------

    winning_moves: set[tuple[int, int]] = field(default_factory=set)
    """Moves that immediately create five-in-a-row and win the game."""

    must_block: set[tuple[int, int]] = field(default_factory=set)
    """Moves we MUST play to prevent opponent from winning next turn.

    Covers opponent FIVE, OPEN_FOUR, and CLOSED_FOUR threats."""

    urgent_blocks: set[tuple[int, int]] = field(default_factory=set)
    """Moves that block opponent OPEN_THREE threats — urgent but not
    immediately forced."""

    double_threat_moves: set[tuple[int, int]] = field(default_factory=set)
    """Moves that create a double threat the opponent cannot fully block
    in one move — a guaranteed forced win."""

    # ---- per-move scores (0.0 = no tactical value) -----------------------

    creation_scores: dict[tuple[int, int], float] = field(default_factory=dict)
    """How much threat-creation value each move generates."""

    blocking_scores: dict[tuple[int, int], float] = field(default_factory=dict)
    """How much threat-blocking value each move provides."""

    # ---- multi-move sequences --------------------------------------------

    forced_sequence: Optional[list[tuple[int, int]]] = None
    """Sequence of moves for current_player that forces a win, or None."""

    # ---- derived properties ----------------------------------------------

    @property
    def has_forced_win(self) -> bool:
        return (
            bool(self.winning_moves)
            or bool(self.double_threat_moves)
            or self.forced_sequence is not None
        )

    @property
    def has_forced_defense(self) -> bool:
        return bool(self.must_block)

    @property
    def is_tactically_urgent(self) -> bool:
        return self.has_forced_win or self.has_forced_defense or bool(self.urgent_blocks)

    # ---- public helpers --------------------------------------------------

    def get_move_boost(self, move: tuple[int, int]) -> float:
        """Return a multiplier (>= 1.0) for the neural prior of *move*.

        Higher values cause PUCT to explore the move more aggressively.
        Winning / must-block moves get an enormous boost so they dominate
        any neural prior, no matter how small.
        """
        if move in self.winning_moves or move in self.must_block:
            return 10_000.0
        if move in self.double_threat_moves:
            return 500.0
        if move in self.urgent_blocks:
            return 100.0
        create = self.creation_scores.get(move, 0.0)
        block = self.blocking_scores.get(move, 0.0)
        return 1.0 + create + block

    def get_forced_distribution(
        self,
    ) -> Optional[dict[tuple[int, int], float]]:
        """Return a deterministic move distribution for forced positions,
        or ``None`` when the position is not forced (caller should fall
        back to full MCTS).
        """
        if self.winning_moves:
            moves = list(self.winning_moves)
            return {m: 1.0 / len(moves) for m in moves}
        if self.must_block:
            moves = list(self.must_block)
            return {m: 1.0 / len(moves) for m in moves}
        if self.double_threat_moves:
            moves = list(self.double_threat_moves)
            return {m: 1.0 / len(moves) for m in moves}
        if self.forced_sequence:
            first = self.forced_sequence[0]
            return {first: 1.0}
        return None

    def get_priority_order(self) -> list[tuple[int, int]]:
        """Return all scored moves ordered by tactical priority (highest first)."""
        all_moves = set(self.creation_scores.keys()) | set(self.blocking_scores.keys())
        all_moves |= self.winning_moves | self.must_block
        all_moves |= self.double_threat_moves | self.urgent_blocks

        def _key(m: tuple[int, int]) -> float:
            # Highest-priority first: wins > must-block > double-threat >
            # urgent-block > creation + blocking score
            if m in self.winning_moves:
                return 1_000_000.0 + self.creation_scores.get(m, 0.0)
            if m in self.must_block:
                return 100_000.0 + self.blocking_scores.get(m, 0.0)
            if m in self.double_threat_moves:
                return 10_000.0 + self.creation_scores.get(m, 0.0)
            if m in self.urgent_blocks:
                return 1_000.0 + self.blocking_scores.get(m, 0.0)
            return self.creation_scores.get(m, 0.0) + self.blocking_scores.get(m, 0.0)

        return sorted(all_moves, key=_key, reverse=True)


# =========================================================================
# TacticalSolver
# =========================================================================


class TacticalSolver:
    """Deterministic shallow tactical solver for Gomoku.

    Analyses a board position without any neural-network or MCTS
    dependency.  Answers questions like:

    * Can I win immediately?
    * Must I block an opponent threat?
    * Which moves create or block threats?
    * Is there a forced winning sequence?

    All methods are static.  Typical usage::

        analysis = TacticalSolver.analyze(board)
        if analysis.has_forced_win:
            return analysis.get_forced_distribution()
        # else run full MCTS with tactical prior boosting …
    """

    # Maximum threat moves to examine per position during forced-sequence
    # search.  Keeps the solver fast (< 1 ms for any position).
    _MAX_THREAT_MOVES = 12

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(board: Board) -> TacticalAnalysis:
        """Full tactical assessment of *board* for ``board.current_player``.

        Returns a :class:`TacticalAnalysis` with all fields populated.
        Callers should check ``analysis.has_forced_win`` and
        ``analysis.has_forced_defense`` first — these indicate positions
        that can be resolved deterministically without MCTS.
        """
        if board.is_terminal():
            return TacticalAnalysis()

        player = board.current_player
        opponent = Player(-player)
        legal = board.get_legal_moves()
        legal_set = set(legal)

        our_threats = ThreatDetector.detect_all(board, player)
        opp_threats = ThreatDetector.detect_all(board, opponent)

        analysis = TacticalAnalysis()

        # Phase 1 — deterministic forced moves
        analysis.winning_moves = TacticalSolver._find_winning_moves(
            our_threats, legal_set
        )
        analysis.must_block = TacticalSolver._find_must_block_moves(
            opp_threats, legal_set
        )

        # Phase 2 — urgent but not immediately forced
        if not analysis.winning_moves and not analysis.must_block:
            analysis.urgent_blocks = TacticalSolver._find_urgent_block_moves(
                board, player, opponent, opp_threats, legal_set
            )

        # Phase 3 — double threats (guaranteed forced win in 2+ moves)
        if not analysis.has_forced_win:
            analysis.double_threat_moves = TacticalSolver._find_double_threat_moves(
                board, player, legal_set
            )

        # Phase 4 — per-move tactical scores (for prior boosting)
        analysis.creation_scores, analysis.blocking_scores = (
            TacticalSolver._score_all_moves(board, player, opponent, legal)
        )

        # Phase 5 — forced-sequence search (multi-move tactical lines)
        if not analysis.has_forced_win and not analysis.has_forced_defense:
            analysis.forced_sequence = TacticalSolver._find_forced_sequence(
                board, player, opponent, max_depth=4
            )

        return analysis

    @staticmethod
    def analyze_lightweight(board: Board) -> TacticalAnalysis:
        """Fast tactical check using only current threats (zero board copies).

        Only populates ``winning_moves`` and ``must_block`` — the two
        fields needed for tactically-decided value injection.  Callers
        that need full scoring should use :meth:`analyze` instead.
        """
        if board.is_terminal():
            return TacticalAnalysis()

        player = board.current_player
        opponent = Player(-player)
        legal_set = set(board.get_legal_moves())

        our_threats = ThreatDetector.detect_all(board, player)
        opp_threats = ThreatDetector.detect_all(board, opponent)

        analysis = TacticalAnalysis()
        analysis.winning_moves = TacticalSolver._find_winning_moves(
            our_threats, legal_set
        )
        analysis.must_block = TacticalSolver._find_must_block_moves(
            opp_threats, legal_set
        )
        return analysis

    @staticmethod
    def find_forced_sequence(
        board: Board, max_depth: int = 4
    ) -> Optional[list[tuple[int, int]]]:
        """Search for a forced winning sequence for ``board.current_player``.

        Returns a list of moves (first move is next to play), or ``None``
        if no forced win is found within *max_depth* plies.
        """
        if board.is_terminal():
            return None
        return TacticalSolver._find_forced_sequence(
            board, board.current_player, Player(-board.current_player), max_depth
        )

    # ------------------------------------------------------------------
    # Winning-move detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_winning_moves(
        threats: list[Threat], legal_set: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Return legal moves that immediately create five-in-a-row.

        Only FIVE, OPEN_FOUR, and CLOSED_FOUR threats produce immediate
        winning moves.  OPEN_THREE cells extend to a four, not a win.
        """
        moves: set[tuple[int, int]] = set()
        for t in threats:
            if t.threat_type not in (
                ThreatType.FIVE,
                ThreatType.OPEN_FOUR,
                ThreatType.CLOSED_FOUR,
            ):
                continue
            cells = ThreatDetector.get_completion_cells(t)
            moves.update(cells)
        return moves & legal_set

    # ------------------------------------------------------------------
    # Defensive-move detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_must_block_moves(
        opp_threats: list[Threat], legal_set: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Return moves we MUST play to prevent opponent from winning next turn.

        Covers opponent FIVE, OPEN_FOUR, and CLOSED_FOUR.  A CLOSED_FOUR has
        exactly one completion cell — if the opponent fills it, they win.
        """
        moves: set[tuple[int, int]] = set()
        for t in opp_threats:
            if t.threat_type in (
                ThreatType.FIVE,
                ThreatType.OPEN_FOUR,
                ThreatType.CLOSED_FOUR,
            ):
                cells = ThreatDetector.get_completion_cells(t)
                moves.update(cells)
        return moves & legal_set

    @staticmethod
    def _find_urgent_block_moves(
        board: Board,
        player: Player,
        opponent: Player,
        opp_threats: list[Threat],
        legal_set: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Return moves that block opponent OPEN_THREE threats.

        These are urgent because an unblocked open three becomes an open
        four on the opponent's next turn.  However, not every open three
        end is a valid block — the block must actually reduce the threat.
        """
        moves: set[tuple[int, int]] = set()
        for t in opp_threats:
            if t.threat_type != ThreatType.OPEN_THREE:
                continue
            # Block cells are the open ends and the gap (if split).
            for end in t.open_ends:
                moves.add(end)
            if t.gap is not None:
                moves.add(t.gap)
        return moves & legal_set

    # ------------------------------------------------------------------
    # Double-threat detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_double_threat_moves(
        board: Board, player: Player, legal_set: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Find moves that create a double threat.

        A double threat consists of two or more threats that cannot all be
        blocked in a single move — playing such a move guarantees a win.
        """
        moves: set[tuple[int, int]] = set()
        for move in legal_set:
            copy = board.copy()
            copy.make_move(*move)
            if ThreatDetector.has_double_threat(copy, player):
                moves.add(move)
        return moves

    # ------------------------------------------------------------------
    # Per-move tactical scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_all_moves(
        board: Board,
        player: Player,
        opponent: Player,
        legal: list[tuple[int, int]],
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """Score every legal move for threat creation and blocking value.

        Returns ``(creation_scores, blocking_scores)``.
        """
        creation: dict[tuple[int, int], float] = {}
        blocking: dict[tuple[int, int], float] = {}

        # Pre-compute opponent threat counts so we can detect reductions.
        orig_opp_threats = ThreatDetector.detect_all(board, opponent)
        orig_counts: dict[ThreatType, int] = {t: 0 for t in ThreatType}
        for t in orig_opp_threats:
            orig_counts[t.threat_type] += 1

        for r, c in legal:
            copy = board.copy()
            copy.make_move(r, c)

            # --- threat creation ---
            create = 0.0
            for t in ThreatDetector.detect_all(copy, player):
                if t.threat_type == ThreatType.FIVE:
                    create += 1000.0
                elif t.threat_type == ThreatType.OPEN_FOUR:
                    create += 150.0
                elif t.threat_type == ThreatType.CLOSED_FOUR:
                    create += 40.0
                elif t.threat_type == ThreatType.OPEN_THREE:
                    create += 8.0
            creation[(r, c)] = create

            # --- threat blocking ---
            block = 0.0
            new_opp_threats = ThreatDetector.detect_all(copy, opponent)
            new_counts: dict[ThreatType, int] = {t: 0 for t in ThreatType}
            for t in new_opp_threats:
                new_counts[t.threat_type] += 1

            for tt in ThreatType:
                reduced = orig_counts[tt] - new_counts[tt]
                if reduced > 0:
                    if tt == ThreatType.OPEN_FOUR:
                        block += reduced * 100.0
                    elif tt == ThreatType.CLOSED_FOUR:
                        block += reduced * 25.0
                    elif tt == ThreatType.OPEN_THREE:
                        block += reduced * 5.0
            blocking[(r, c)] = block

        return creation, blocking

    # ------------------------------------------------------------------
    # Forced-sequence search
    # ------------------------------------------------------------------

    @staticmethod
    def _find_forced_sequence(
        board: Board,
        player: Player,
        opponent: Player,
        max_depth: int,
    ) -> Optional[list[tuple[int, int]]]:
        """Recursive threat-space search for a forced-win sequence.

        On *player*'s turn: try threat-creating moves.  If a move creates
        a forcing threat (OPEN_FOUR / CLOSED_FOUR) and every opponent
        blocking response still leads to a forced win, return the sequence.

        Depth is counted in plies.  *max_depth* ≤ 0 stops the search.
        """
        if max_depth <= 0:
            return None

        legal = board.get_legal_moves()

        for move in legal:
            copy = board.copy()
            copy.make_move(*move)

            # Immediate win?
            if copy.check_win() == player:
                return [move]

            # Double threat? Opponent cannot block both → forced win.
            if ThreatDetector.has_double_threat(copy, player):
                return [move]

            # Find forcing threats that require an immediate response.
            threats = ThreatDetector.detect_all(copy, player)
            forcing = [
                t
                for t in threats
                if t.threat_type in (ThreatType.OPEN_FOUR, ThreatType.CLOSED_FOUR)
            ]
            if not forcing:
                continue

            # Collect cells the opponent MUST fill to block.
            block_cells: set[tuple[int, int]] = set()
            for t in forcing:
                block_cells.update(ThreatDetector.get_completion_cells(t))
            block_cells &= set(copy.get_legal_moves())

            if len(block_cells) > 3:
                continue  # too many defensive options — not forcing enough

            # If the opponent has any winning move of their own, our
            # forcing move is too slow — they can just win instead.
            opp_threats = ThreatDetector.detect_all(copy, opponent)
            opp_winning = TacticalSolver._find_winning_moves(
                opp_threats, set(copy.get_legal_moves())
            )
            if opp_winning:
                continue

            # Check every blocking response.  If any block stops our
            # attack, this move is not a forced win.
            all_blocked = True
            for block in block_cells:
                copy2 = copy.copy()
                copy2.make_move(*block)

                followup = TacticalSolver._find_forced_sequence(
                    copy2, player, opponent, max_depth - 2
                )
                if followup is None:
                    all_blocked = False
                    break

            if all_blocked and block_cells:
                return [move]

        return None
