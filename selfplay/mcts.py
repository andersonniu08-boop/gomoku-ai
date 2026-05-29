"""AlphaZero-style MCTS with PUCT, driven by a dual-headed CNN."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from engine.board import Board, Player
from engine.threats import ThreatDetector, ThreatType
from neural.wrapper import GomokuInferenceWrapper
from selfplay.evaluator import BatchedLeafEvaluator
from selfplay.move_ordering import order_and_filter_moves
from selfplay.profiler import Profiler

# When there are more legal moves than this, keep only the top-N priors
# to bound the branching factor.
_POLICY_CUTOFF = 40


@dataclass(slots=True)
class MCTSNode:
    """Edge statistics for one action leading from a parent state.

    Children are stored directly on this node keyed by ``(row, col)``.
    """

    prior: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0  # sum of backed-up values (negated per level)
    virtual_loss: int = 0
    children: dict[tuple[int, int], MCTSNode] = field(default_factory=dict)

    @property
    def q(self) -> float:
        """Mean action value Q(s,a) ∈ [-1, 1].

        Virtual loss penalises in-flight nodes: it adds one phantom loss
        per pending evaluation below this node, steering subsequent
        PUCT selections toward other branches.
        """
        total_n = self.visit_count + self.virtual_loss
        if total_n == 0:
            return 0.0
        return (self.total_value - self.virtual_loss) / total_n


@dataclass(slots=True)
class SearchResult:
    """Full MCTS search statistics for a root position.

    Attributes:
        visit_counts:      Number of times each child move was visited.
        q_values:          Mean action value Q(s,a) for each child move.
        priors:            Prior probability P(s,a) from the policy head.
        total_simulations: Total MCTS simulations run (0 if threat-overridden).
    """

    visit_counts: dict[tuple[int, int], int]
    q_values: dict[tuple[int, int], float]
    priors: dict[tuple[int, int], float]
    total_simulations: int


class MCTS:
    """Monte-Carlo Tree Search powered by a neural value/policy network.

    Uses a single *virtual board* that is mutated during selection and
    restored during back-propagation, avoiding per-node board copies.

    The search can be bounded in two ways (mutually exclusive):

    1. **Fixed simulations** — ``num_simulations`` iterations, regardless
       of wall-clock time.
    2. **Time budget** — ``time_budget_ms`` milliseconds of wall-clock
       time (ignores ``num_simulations``).

    When neither is explicitly provided, the default is 800 simulations.

    Parameters:
        wrapper:           Trained ``GomokuInferenceWrapper``.
        c_puct:            Exploration constant for the PUCT formula.
        num_simulations:   MCTS iterations per search (default: 800).
                           Ignored when *time_budget_ms* is set.
        time_budget_ms:    Optional wall-clock budget in milliseconds.
                           When set, the search runs as many simulations
                           as fit within this time window.
        batch_size:        Leaves collected per neural-evaluation batch.
                           Larger values improve GPU utilisation for
                           multi-threaded inference but add virtual-loss
                           overhead in single-threaded MCTS (default: 1).
        threat_override:   When True, use ``ThreatDetector`` to short-circuit
                           search on immediate wins / must-block threats.
        dirichlet_alpha:   Concentration parameter for root Dirichlet noise.
                           ``None`` (default) = no noise.
        dirichlet_epsilon: Mixing proportion: ``(1-ε) · prior + ε · noise``.
        profiler:          Optional ``Profiler`` for detailed timing breakdown.
        evaluator:         Optional ``BatchedLeafEvaluator``.  Created
                           automatically from *wrapper* when not provided.
        tree_reuse:        When True (default), the search tree from a
                           previous search is re-rooted and reused so
                           simulation effort accumulates across moves.
                           Disable for self-play training and model
                           evaluation where each position needs an
                           independent, identically-budgeted search.
    """

    def __init__(
        self,
        wrapper: GomokuInferenceWrapper,
        *,
        c_puct: float = 2.5,
        num_simulations: int = 800,
        time_budget_ms: float | None = None,
        batch_size: int = 8,
        threat_override: bool = True,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.25,
        tree_reuse: bool = True,
        profiler: Profiler | None = None,
        evaluator: BatchedLeafEvaluator | None = None,
    ):
        self.wrapper = wrapper
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.time_budget_ms = time_budget_ms
        self.batch_size = batch_size
        self.threat_override = threat_override
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.tree_reuse = tree_reuse
        self.profiler = profiler or Profiler()
        self.evaluator = evaluator or BatchedLeafEvaluator(
            wrapper,
            target_batch_size=batch_size,
            profiler=self.profiler,
        )

        self._prev_root: Optional[MCTSNode] = None
        self._prev_board: Optional[Board] = None
        self._cumulative_sims: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_tree(self) -> None:
        """Discard any cached search tree (e.g. on new game)."""
        self._prev_root = None
        self._prev_board = None
        self._cumulative_sims = 0

    def search(self, board: Board) -> dict[tuple[int, int], float]:
        """Run batched MCTS from *board*.

        Descends *batch_size* leaves in parallel (using virtual loss to
        diversify paths), evaluates them with one GPU call, then backs up.

        When a previous search tree exists and the board's new moves can
        be matched to children of that tree, the subtree is reused —
        preserving prior simulation statistics for dramatically improved
        effective search depth.

        Returns a probability distribution over legal moves.
        """
        if board.is_terminal():
            return {}

        # Wire profiler into wrapper so eval timings are unified.
        self.wrapper.profiler = self.profiler

        if self.threat_override:
            with self.profiler.measure("threat_check"):
                forced = self._check_forced(board)
            if forced is not None:
                self._prev_root = None
                self._prev_board = None
                return forced

        sim_board = board.copy()
        root, is_reused = self._try_reroot(board)
        fresh_sims = self._sim_budget()
        with self.profiler.measure("search.total"):
            self._run_search(sim_board, root, fresh_sims, is_reused=is_reused)

        if self.tree_reuse:
            self._prev_root = root
            self._prev_board = board.copy()
        if is_reused:
            self._cumulative_sims += fresh_sims
        else:
            self._cumulative_sims = fresh_sims

        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            legal = board.get_legal_moves()
            return {m: 1.0 / len(legal) for m in legal}
        return {m: c.visit_count / total_visits for m, c in root.children.items()}

    def search_with_stats(self, board: Board) -> SearchResult:
        """Like :meth:`search` but also returns Q-values and priors.

        Uses the same internal search loop as :meth:`search` but exposes
        root children's full statistics (visit counts, Q values, priors)
        instead of just visit proportions.  Also benefits from tree reuse
        when a prior search tree is available.
        """
        if board.is_terminal():
            return SearchResult({}, {}, {}, 0)

        self.wrapper.profiler = self.profiler

        if self.threat_override:
            with self.profiler.measure("threat_check"):
                forced = self._check_forced(board)
            if forced is not None:
                self._prev_root = None
                self._prev_board = None
                moves = list(forced)
                visit_counts = {m: 1 for m in moves}
                q_values = {m: 0.0 for m in moves}
                priors = {m: forced[m] for m in moves}
                return SearchResult(visit_counts, q_values, priors, 0)

        sim_board = board.copy()
        root, is_reused = self._try_reroot(board)
        fresh_sims = self._sim_budget()
        with self.profiler.measure("search.total"):
            self._run_search(sim_board, root, fresh_sims, is_reused=is_reused)

        if self.tree_reuse:
            self._prev_root = root
            self._prev_board = board.copy()
        if is_reused:
            self._cumulative_sims += fresh_sims
        else:
            self._cumulative_sims = fresh_sims

        visit_counts = {m: c.visit_count for m, c in root.children.items()}
        q_values = {m: c.q for m, c in root.children.items()}
        priors = {m: c.prior for m, c in root.children.items()}
        return SearchResult(visit_counts, q_values, priors, self._cumulative_sims)

    def select_move(
        self,
        board: Board,
        *,
        temperature: float = 0.0,
        visit_probs: Optional[dict[tuple[int, int], float]] = None,
    ) -> tuple[int, int]:
        """Return the best move after search.

        When *visit_probs* is provided the caller has already run
        :meth:`search` on this *board* and is passing the result in to
        avoid a redundant second search.  When omitted (default) a fresh
        search is run.

        *temperature=0*  → greedy (most visits).
        *temperature>0*  → sample proportionally to visit counts.
        """
        if visit_probs is None:
            visit_probs = self.search(board)

        if temperature == 0.0:
            return max(visit_probs, key=visit_probs.get)

        moves = list(visit_probs)
        probs = [visit_probs[m] ** (1.0 / temperature) for m in moves]
        total = sum(probs)
        probs = [p / total for p in probs]
        return moves[random.choices(range(len(moves)), weights=probs, k=1)[0]]

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------

    def _try_reroot(self, board: Board) -> tuple[MCTSNode, bool]:
        """Attempt to re-root the previous search tree at *board*.

        Compares *board*'s ``move_history`` against the stored previous
        board to discover which moves were played between searches.
        Walks the tree along those moves, returning the new root and
        ``is_reused=True``.  If any move is missing from the tree — or
        if no prior tree exists — returns a fresh ``MCTSNode`` and
        ``is_reused=False``.

        Returns ``(root, is_reused)``.
        """
        if not self.tree_reuse:
            return MCTSNode(), False

        if self._prev_root is None or self._prev_board is None:
            return MCTSNode(), False

        prev_moves = self._prev_board.move_history
        curr_moves = board.move_history

        # If the board was reset or moves were undone, fall back.
        if len(curr_moves) < len(prev_moves):
            return MCTSNode(), False

        new_moves = curr_moves[len(prev_moves):]
        if not new_moves:
            # Same board — just reuse the existing root.
            return self._prev_root, True

        root = self._prev_root
        for move in new_moves:
            if move not in root.children:
                return MCTSNode(), False
            root = root.children[move]

        return root, True

    def _sim_budget(self) -> int:
        """Return the number of fresh simulations to run this search.

        In fixed-budget mode this is ``num_simulations``.  In time-budget
        mode we return a large sentinel — the search loop exits on wall
        clock instead.
        """
        if self.time_budget_ms is not None:
            return 1_000_000_000  # effectively unbounded; time-budget stops it
        return self.num_simulations

    # ------------------------------------------------------------------
    # Search loop (shared by search() and search_with_stats())
    # ------------------------------------------------------------------

    def _run_search(
        self,
        sim_board: Board,
        root: MCTSNode,
        fresh_sims: int,
        *,
        is_reused: bool = False,
    ) -> None:
        """Run the main MCTS loop, populating *root* with visited statistics.

        Mutates *sim_board* during descent and restores it after each batch
        of descents.  *root* is expanded and its children accumulate visit
        counts, total values, and virtual loss markers.

        When *is_reused* is True the root was obtained via tree reuse —
        Dirichlet noise is skipped (priors were already mixed in a prior
        search) and the existing subtree statistics provide a warm start.

        The loop terminates either when *fresh_sims* simulations have run
        (fixed-budget mode) or when ``time_budget_ms`` wall-clock time
        has elapsed (time-budget mode).  When both are set, time budget
        takes precedence.
        """
        sims_done = 0
        use_time_budget = self.time_budget_ms is not None
        deadline = None
        if use_time_budget:
            deadline = time.monotonic() + self.time_budget_ms / 1000.0

        while True:
            # --- Check termination ---
            if use_time_budget:
                if time.monotonic() >= deadline:
                    break
                # Compute how many sims we can attempt this batch without
                # blowing the budget.  Use at least 1.
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    break
            else:
                if sims_done >= fresh_sims:
                    break

            # --- Batch size ---
            if use_time_budget:
                n = self.batch_size
            else:
                n = min(self.batch_size, fresh_sims - sims_done)

            with self.profiler.measure("search.batch"):
                # ---- descend n times, collecting leaves ----
                with self.profiler.measure("search.descend_batch"):
                    paths: list[list[MCTSNode]] = []
                    leaf_boards: list[Board] = []
                    leaf_terminal: list[bool] = []

                    for _ in range(n):
                        with self.profiler.measure("descend.single"):
                            path, leaf_b, is_term = self._descend_and_tag(
                                sim_board, root
                            )
                        paths.append(path)
                        leaf_boards.append(leaf_b)
                        leaf_terminal.append(is_term)
                        # Rewind sim_board to root.
                        with self.profiler.measure("descend.rewind"):
                            for _ in range(len(path) - 1):
                                sim_board.undo_move()

                # ---- batch-evaluate non-terminal leaves ----
                with self.profiler.measure("search.neural_eval"):
                    eval_indices = [i for i, t in enumerate(leaf_terminal) if not t]
                    eval_boards = [leaf_boards[i] for i in eval_indices]
                    if eval_boards:
                        batch_results = self.evaluator.evaluate(eval_boards)
                    else:
                        batch_results = []

                # ---- expand & backup ----
                with self.profiler.measure("search.expand_backup"):
                    result_idx = 0
                    for i in range(n):
                        path = paths[i]
                        leaf_node = path[-1]

                        if leaf_terminal[i]:
                            value = self._terminal_value(leaf_boards[i])
                        else:
                            move_probs, value = batch_results[result_idx]
                            result_idx += 1
                            # The evaluator returns value from the leaf board's
                            # current_player perspective, which is the OPPONENT of
                            # the player who made the move to reach this leaf.
                            # Child nodes store Q from the mover's perspective, so
                            # we negate here.  Terminal values from _terminal_value
                            # already use the correct convention (+1 = mover won).
                            value = -value
                            # Expand leaf.  Always compute tactical scores
                            # when threat_override is enabled so forced
                            # wins / must-blocks deep in the tree are
                            # caught immediately rather than relying on
                            # the network's value head alone.
                            with self.profiler.measure("expand.order_and_filter"):
                                move_probs = order_and_filter_moves(
                                    leaf_boards[i],
                                    move_probs,
                                    max_moves=_POLICY_CUTOFF,
                                    hard_override=self.threat_override,
                                )
                            with self.profiler.measure("expand.create_nodes"):
                                for (r, c), prior in move_probs:
                                    leaf_node.children[(r, c)] = MCTSNode(prior=prior)

                        # Backup: walk up, update stats, remove virtual loss.
                        with self.profiler.measure("backup.walk"):
                            current_value = value
                            for j in range(len(path) - 1, 0, -1):
                                node = path[j]
                                node.visit_count += 1
                                node.total_value += current_value
                                node.virtual_loss -= 1
                                current_value = -current_value

                # ---- inject Dirichlet noise at root after first expansion ----
                # Only apply to freshly-created roots.  Reused trees already
                # have priors mixed from a prior search.
                if (
                    not is_reused
                    and self.dirichlet_alpha is not None
                    and root.children
                    and sims_done == 0
                ):
                    with self.profiler.measure("dirichlet_noise"):
                        self._apply_dirichlet_noise(root)

                sims_done += n

    # ------------------------------------------------------------------
    # Batched descent
    # ------------------------------------------------------------------

    def _descend_and_tag(
        self, board: Board, root: MCTSNode
    ) -> tuple[list[MCTSNode], Board, bool]:
        """Descend from *root* to a leaf via PUCT, adding virtual loss.

        Mutates *board* as it goes, then copies it at the leaf.

        Returns:
            path_nodes:  [root, child1, child2, ..., leaf] — every node
                         except root has had virtual_loss incremented.
            leaf_board:  A copy of *board* at the leaf.
            is_terminal: True if the leaf board is terminal.
        """
        path_nodes = [root]
        node = root
        while node.children:
            with self.profiler.measure("descend.puct_select"):
                action = self._puct_select(node)
            with self.profiler.measure("descend.make_move"):
                board.make_move(*action)
            node = node.children[action]
            node.virtual_loss += 1
            path_nodes.append(node)

        with self.profiler.measure("descend.board_copy"):
            leaf_board = board.copy()
        with self.profiler.measure("descend.is_terminal"):
            is_terminal = board.is_terminal()
        return path_nodes, leaf_board, is_terminal

    # ------------------------------------------------------------------
    # PUCT
    # ------------------------------------------------------------------

    def _puct_select(self, node: MCTSNode) -> tuple[int, int]:
        """Return the child action that maximises the PUCT score."""
        best_action = None
        best_score = -float("inf")

        sqrt_parent_n = math.sqrt(
            sum(c.visit_count for c in node.children.values()) + 1
        )

        for action, child in node.children.items():
            exploit = child.q
            explore = (
                self.c_puct
                * child.prior
                * sqrt_parent_n
                / (1 + child.visit_count)
            )
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None
        return best_action

    # ------------------------------------------------------------------
    # Threat short-circuit
    # ------------------------------------------------------------------

    def _check_forced(self, board: Board) -> Optional[dict[tuple[int, int], float]]:
        """Return a deterministic move distribution when the position contains
        an immediate win or must-block threat, bypassing MCTS entirely."""
        threats = ThreatDetector.detect_all(board, board.current_player)
        opp_threats = ThreatDetector.detect_all(
            board, Player(-board.current_player)
        )

        legal = board.get_legal_moves()
        legal_set = set(legal)

        # 1) Immediate winning moves for the current player — play one.
        #
        # FIVE and OPEN_FOUR: every open end and/or gap is a winning move
        #   (placing there completes five-in-a-row).
        #
        # CLOSED_FOUR behaviour depends on the pattern:
        #   * Contiguous four with one open end (XXXX_): the open end IS a
        #     winning move — placing there creates five-in-a-row.
        #   * Split four (XX_XX, XXX_X): the *gap* is a winning move — filling
        #     it gives five-in-a-row.  The external open ends are NOT winning
        #     moves because they extend the split pattern without producing
        #     five consecutive stones.
        win_moves: set[tuple[int, int]] = set()
        for t in threats:
            if t.threat_type == ThreatType.CLOSED_FOUR:
                if t.gap is not None:
                    # Split closed four — only the gap creates five-in-a-row.
                    win_moves.add(t.gap)
                else:
                    # Contiguous closed four — the one open end creates five.
                    for end in t.open_ends:
                        win_moves.add(end)
            elif t.threat_type in (ThreatType.FIVE, ThreatType.OPEN_FOUR):
                for end in t.open_ends:
                    win_moves.add(end)
                if t.gap is not None:
                    win_moves.add(t.gap)

        win_moves &= legal_set
        if win_moves:
            return {m: 1.0 / len(win_moves) for m in win_moves}

        # 2) Opponent threats that demand an immediate block.
        #
        #     An opponent FIVE (gapped) or OPEN_FOUR means the opponent
        #     can win on their next move.  An opponent CLOSED_FOUR
        #     (either contiguous with one open end, or split with a gap)
        #     also wins on their next turn — we must block every cell
        #     that completes five-in-a-row for the opponent.
        block_set: set[tuple[int, int]] = set()
        for t in opp_threats:
            if t.threat_type in (ThreatType.FIVE, ThreatType.OPEN_FOUR):
                if t.gap is not None:
                    block_set.add(t.gap)
                for end in t.open_ends:
                    block_set.add(end)
            elif t.threat_type == ThreatType.CLOSED_FOUR:
                if t.gap is not None:
                    # Split closed four — gap creates five.
                    block_set.add(t.gap)
                else:
                    # Contiguous closed four — open end creates five.
                    for end in t.open_ends:
                        block_set.add(end)
        block_moves = list(block_set & legal_set)
        if block_moves:
            return {m: 1.0 / len(block_moves) for m in block_moves}

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into root children's priors for exploration.

        The mixed prior for each child ``a`` at the root is:

            P'(s, a) = (1 - ε) · P(s, a) + ε · η_a

        where ``η ~ Dir(α)`` is a symmetric Dirichlet sample.  With a
        small ``α`` (e.g. 0.03) the Dirichlet mass is concentrated on
        a few entries, encouraging the search to try diverse lines.

        Priors remain normalized after mixing because both components
        sum to 1 and the convex combination preserves the sum.
        """
        if not root.children:
            return

        moves = list(root.children)
        k = len(moves)
        alpha = torch.full((k,), self.dirichlet_alpha)
        noise = torch.distributions.Dirichlet(alpha).sample().tolist()

        for (r, c), eta in zip(moves, noise):
            child = root.children[(r, c)]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + \
                          self.dirichlet_epsilon * eta

    @staticmethod
    def _terminal_value(board: Board) -> float:
        """Value of a terminal state from the perspective of the player who
        **just moved** (the winner, or 0 for a draw)."""
        winner = board.check_win()
        if winner is None:
            return 0.0  # draw (full board)
        return 1.0  # the player who just moved won
