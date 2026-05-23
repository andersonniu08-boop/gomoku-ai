# Batched MCTS Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one-at-a-time neural inference in MCTS with batched forward passes using virtual loss for parallel descent.

**Architecture:** Two changes in sequence: (1) add `batch_evaluate` and `batch_evaluate_with_threats` to `GomokuInferenceWrapper`, (2) add virtual loss to `MCTSNode` and rewrite `MCTS.search()` to descend N leaves in parallel, batch-evaluate them with one GPU call, then back up. Existing `search()` and `select_move()` APIs are unchanged.

**Tech Stack:** PyTorch, NumPy, Python 3.14

---

### Task 1: Add `batch_evaluate` to `GomokuInferenceWrapper`

**Files:**
- Modify: `neural/wrapper.py` — add method after `evaluate()` (line 69)
- Modify: `tests/test_neural.py` — add 3 tests after the existing wrapper tests

- [ ] **Step 1: Write the tests**

```python
def test_batch_evaluate_returns_correct_count():
    """N boards in → N results out."""
    from engine.board import Board
    from neural.wrapper import GomokuInferenceWrapper
    wrapper = _make_wrapper()
    board1 = Board()
    board2 = Board()
    board1.make_move(7, 7)
    board2.make_move(7, 7)
    board2.make_move(7, 8)

    results = wrapper.batch_evaluate([board1, board2])
    assert len(results) == 2
    for move_probs, value in results:
        assert isinstance(move_probs, list)
        assert len(move_probs) > 0
        assert isinstance(move_probs[0], tuple)
        assert isinstance(move_probs[0][0], tuple)  # (row, col)
        assert isinstance(move_probs[0][1], float)  # prob
        assert -1.0 <= value <= 1.0


def test_batch_evaluate_empty():
    """Empty input → empty output."""
    wrapper = _make_wrapper()
    results = wrapper.batch_evaluate([])
    assert results == []


def test_batch_evaluate_matches_single():
    """Each board in a batch produces the same result as calling evaluate() individually."""
    from engine.board import Board
    from neural.wrapper import GomokuInferenceWrapper
    wrapper = _make_wrapper()
    boards = [Board() for _ in range(4)]
    for i, b in enumerate(boards):
        b.make_move(7, 7)
        if i % 2 == 0:
            b.make_move(7, 8)

    batch_results = wrapper.batch_evaluate(boards)
    single_results = [wrapper.evaluate(b) for b in boards]

    for (b_probs, b_val), (s_probs, s_val) in zip(batch_results, single_results):
        assert b_val == s_val
        assert len(b_probs) == len(s_probs)
        for (bm, bp), (sm, sp) in zip(sorted(b_probs), sorted(s_probs)):
            assert bm == sm
            assert abs(bp - sp) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_neural.py::test_batch_evaluate_returns_correct_count tests/test_neural.py::test_batch_evaluate_empty tests/test_neural.py::test_batch_evaluate_matches_single -v`
Expected: FAIL with `AttributeError: 'GomokuInferenceWrapper' object has no attribute 'batch_evaluate'`

- [ ] **Step 3: Write the method**

Add this method to `GomokuInferenceWrapper` in `neural/wrapper.py`, after the `evaluate()` method (after line 69):

```python
    def batch_evaluate(
        self, boards: list[Board]
    ) -> list[tuple[list[tuple[tuple[int, int], float]], float]]:
        """Evaluate multiple boards in one forward pass.

        Returns one ``(move_probs, value)`` per board, in the same order.
        """
        if not boards:
            return []

        tensors = torch.cat([board_to_tensor(b) for b in boards], dim=0).to(
            self.device
        )

        with torch.no_grad():
            log_policy, value = self.model(tensors)

        results: list[tuple[list[tuple[tuple[int, int], float]], float]] = []
        for i, board in enumerate(boards):
            move_probs = policy_to_move_probs(log_policy[i : i + 1], board)
            results.append((move_probs, float(value[i].item())))

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_neural.py::test_batch_evaluate_returns_correct_count tests/test_neural.py::test_batch_evaluate_empty tests/test_neural.py::test_batch_evaluate_matches_single -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add neural/wrapper.py tests/test_neural.py
git commit -m "feat: add batch_evaluate to GomokuInferenceWrapper"
```

---

### Task 2: Add `batch_evaluate_with_threats`

**Files:**
- Modify: `neural/wrapper.py` — add method after `evaluate_with_threats()` (line 145)
- Modify: `tests/test_neural.py` — add 2 tests

- [ ] **Step 1: Write the tests**

```python
def test_batch_evaluate_with_threats_empty():
    """Empty input → empty output."""
    wrapper = _make_wrapper()
    results = wrapper.batch_evaluate_with_threats([])
    assert results == []


def test_batch_evaluate_with_threats_matches_single():
    """Batch results match individual evaluate_with_threats calls."""
    from engine.board import Board
    from neural.wrapper import GomokuInferenceWrapper
    wrapper = _make_wrapper()
    boards = [Board() for _ in range(4)]
    for b in boards:
        b.make_move(7, 7)
        b.make_move(7, 8)

    batch_results = wrapper.batch_evaluate_with_threats(boards)
    single_results = [wrapper.evaluate_with_threats(b) for b in boards]

    for (b_probs, b_val, b_info), (s_probs, s_val, s_info) in zip(
        batch_results, single_results
    ):
        assert b_val == s_val
        assert len(b_probs) == len(s_probs)
        for (bm, bp), (sm, sp) in zip(sorted(b_probs), sorted(s_probs)):
            assert bm == sm
            assert abs(bp - sp) < 1e-6
        # threat_info may differ (None vs dict) but both should be None or both not
        assert (b_info is None) == (s_info is None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_neural.py::test_batch_evaluate_with_threats_empty tests/test_neural.py::test_batch_evaluate_with_threats_matches_single -v`
Expected: FAIL

- [ ] **Step 3: Write the method**

Add this method to `GomokuInferenceWrapper` in `neural/wrapper.py`, after the `evaluate_with_threats()` method (after line 145):

```python
    def batch_evaluate_with_threats(
        self,
        boards: list[Board],
        *,
        hard_override: bool = True,
    ) -> list[
        tuple[
            list[tuple[tuple[int, int], float]],
            float,
            Optional[dict],
        ]
    ]:
        """Batch version of ``evaluate_with_threats``.

        Returns one ``(move_probs, value, threat_info)`` per board.
        """
        if not boards:
            return []

        results: list[
            tuple[
                list[tuple[tuple[int, int], float]],
                float,
                Optional[dict],
            ]
        ] = []

        # Gather boards that need neural evaluation (no immediate win override).
        neural_indices: list[int] = []
        neural_boards: list[Board] = []

        for i, board in enumerate(boards):
            threat_info = None
            our_threats = ThreatDetector.detect_all(board, board.current_player)
            opp_threats = ThreatDetector.detect_all(
                board, Player(-board.current_player)
            )

            # Immediate win override.
            winning = [
                t
                for t in our_threats
                if t.threat_type in (ThreatType.FIVE, ThreatType.OPEN_FOUR)
            ]
            if hard_override and winning:
                winning_moves: set[tuple[int, int]] = set()
                for t in winning:
                    if t.threat_type == ThreatType.FIVE:
                        if t.gap is not None:
                            winning_moves.add(t.gap)
                        for end in t.open_ends:
                            winning_moves.add(end)
                    elif t.threat_type == ThreatType.OPEN_FOUR:
                        for end in t.open_ends:
                            winning_moves.add(end)

                legal = board.get_legal_moves()
                legal_set = set(legal)
                winning_moves &= legal_set

                if winning_moves:
                    probs = [
                        (m, 1.0 / len(winning_moves) if m in winning_moves else 0.0)
                        for m in legal
                    ]
                    threat_info = {"overridden": True, "reason": "immediate_win"}
                    results.append((probs, 1.0, threat_info))
                    continue

            # No immediate win → needs neural evaluation.
            neural_indices.append(i)
            neural_boards.append(board)
            results.append(([], 0.0, None))  # placeholder

        # Batch neural evaluation.
        if neural_boards:
            neural_results = self.batch_evaluate(neural_boards)
        else:
            neural_results = []

        # Fill in neural results, applying block-boosting.
        for j, (i, board) in enumerate(zip(neural_indices, neural_boards)):
            move_probs, value = neural_results[j]
            threat_info = None

            opp_threats = ThreatDetector.detect_all(
                board, Player(-board.current_player)
            )
            opp_fours = [
                t for t in opp_threats if t.threat_type == ThreatType.OPEN_FOUR
            ]
            if opp_fours:
                block_moves: set[tuple[int, int]] = set()
                for t in opp_fours:
                    if t.gap is not None:
                        block_moves.add(t.gap)
                    for end in t.open_ends:
                        block_moves.add(end)

                legal_set = set(board.get_legal_moves())
                block_moves &= legal_set

                if block_moves:
                    BOOST_FACTOR = 5.0
                    move_probs = [
                        (m, p * BOOST_FACTOR if m in block_moves else p)
                        for m, p in move_probs
                    ]
                    total = sum(p for _, p in move_probs)
                    move_probs = [(m, p / total) for m, p in move_probs]
                    threat_info = {"overridden": True, "reason": "boosted_blocks"}

            results[i] = (move_probs, value, threat_info)

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_neural.py::test_batch_evaluate_with_threats_empty tests/test_neural.py::test_batch_evaluate_with_threats_matches_single -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add neural/wrapper.py tests/test_neural.py
git commit -m "feat: add batch_evaluate_with_threats to GomokuInferenceWrapper"
```

---

### Task 3: Add `virtual_loss` to `MCTSNode` and modify `q` property

**Files:**
- Modify: `selfplay/mcts.py` — add field to `MCTSNode` dataclass (line 29), modify `q` property (line 31)
- Modify: `tests/test_mcts.py` — add 2 tests

- [ ] **Step 1: Write the tests**

```python
def test_virtual_loss_q_no_visits():
    """Q should be -1 when virtual_loss=1 and visit_count=0."""
    from selfplay.mcts import MCTSNode
    node = MCTSNode(prior=0.5, virtual_loss=1)
    assert node.q == -1.0


def test_virtual_loss_q_with_visits():
    """Q should correctly blend real value and virtual loss."""
    from selfplay.mcts import MCTSNode
    node = MCTSNode(prior=0.5, visit_count=2, total_value=1.0, virtual_loss=1)
    # total_n = 2 + 1 = 3, q = (1.0 - 1) / 3 = 0.0
    assert node.q == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcts.py::test_virtual_loss_q_no_visits tests/test_mcts.py::test_virtual_loss_q_with_visits -v`
Expected: FAIL (TypeError or AssertionError — Q currently ignores virtual_loss)

- [ ] **Step 3: Add the field and modify `q`**

In `selfplay/mcts.py`, add the `virtual_loss` field to the `MCTSNode` dataclass. Replace line 29:

```python
    prior: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0  # sum of backed-up values (negated per level)
    children: dict[tuple[int, int], MCTSNode] = field(default_factory=dict)
```

With:

```python
    prior: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0  # sum of backed-up values (negated per level)
    virtual_loss: int = 0
    children: dict[tuple[int, int], MCTSNode] = field(default_factory=dict)
```

Then replace the `q` property (lines 31-36):

```python
    @property
    def q(self) -> float:
        """Mean action value Q(s,a) ∈ [-1, 1]."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
```

With:

```python
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
```

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/test_mcts.py::test_virtual_loss_q_no_visits tests/test_mcts.py::test_virtual_loss_q_with_visits -v`
Expected: 2 PASS

- [ ] **Step 5: Run existing MCTS tests to check no regression**

Run: `pytest tests/test_mcts.py -v`
Expected: All existing 7 tests still PASS (virtual_loss defaults to 0, `q` formula unchanged when virtual_loss=0)

- [ ] **Step 6: Commit**

```bash
git add selfplay/mcts.py tests/test_mcts.py
git commit -m "feat: add virtual_loss to MCTSNode"
```

---

### Task 4: Rewrite `MCTS.search()` to use batched descent

**Files:**
- Modify: `selfplay/mcts.py` — add `batch_size` to `__init__`, add `_descend_and_tag`, rewrite `search()`, remove `_select` and `_expand_and_evaluate`

- [ ] **Step 1: Add `batch_size` to `MCTS.__init__`**

In `selfplay/mcts.py`, change the `__init__` signature (line 53):

```python
    def __init__(
        self,
        wrapper: GomokuInferenceWrapper,
        *,
        c_puct: float = 2.5,
        num_simulations: int = 400,
        batch_size: int = 8,
        threat_override: bool = True,
    ):
        self.wrapper = wrapper
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.threat_override = threat_override
```

- [ ] **Step 2: Add `_descend_and_tag` helper**

Add this method to the `MCTS` class, after `_puct_select` (after line 190). This descends from root to a leaf, recording the path and adding virtual loss:

```python
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
            action = self._puct_select(node)
            board.make_move(*action)
            node = node.children[action]
            node.virtual_loss += 1
            path_nodes.append(node)

        leaf_board = board.copy()
        is_terminal = board.is_terminal()
        return path_nodes, leaf_board, is_terminal
```

- [ ] **Step 3: Rewrite `search()`**

Replace the entire `search()` method (lines 70-93) with:

```python
    def search(self, board: Board) -> dict[tuple[int, int], float]:
        """Run batched MCTS from *board*.

        Descends *batch_size* leaves in parallel (using virtual loss to
        diversify paths), evaluates them with one GPU call, then backs up.
        """
        if board.is_terminal():
            return {}

        if self.threat_override:
            forced = self._check_forced(board)
            if forced is not None:
                return forced

        sim_board = board.copy()
        root = MCTSNode()

        sims_done = 0
        while sims_done < self.num_simulations:
            n = min(self.batch_size, self.num_simulations - sims_done)

            # ---- descend n times, collecting leaves ----
            paths: list[list[MCTSNode]] = []
            leaf_boards: list[Board] = []
            leaf_terminal: list[bool] = []

            for _ in range(n):
                path, leaf_b, is_term = self._descend_and_tag(sim_board, root)
                paths.append(path)
                leaf_boards.append(leaf_b)
                leaf_terminal.append(is_term)
                # Rewind sim_board to root.
                for _ in range(len(path) - 1):
                    sim_board.undo_move()

            # ---- batch-evaluate non-terminal leaves ----
            eval_indices = [i for i, t in enumerate(leaf_terminal) if not t]
            eval_boards = [leaf_boards[i] for i in eval_indices]
            if eval_boards:
                batch_results = self.wrapper.batch_evaluate(eval_boards)

            # ---- expand & backup ----
            result_idx = 0
            for i in range(n):
                path = paths[i]
                leaf_node = path[-1]

                if leaf_terminal[i]:
                    value = self._terminal_value(leaf_boards[i])
                else:
                    move_probs, value = batch_results[result_idx]
                    result_idx += 1
                    # Expand leaf with capped, renormalised priors.
                    if len(move_probs) > _POLICY_CUTOFF:
                        move_probs.sort(key=lambda x: x[1], reverse=True)
                        move_probs = move_probs[:_POLICY_CUTOFF]
                        total = sum(p for _, p in move_probs)
                        move_probs = [(m, p / total) for m, p in move_probs]
                    for (r, c), prior in move_probs:
                        leaf_node.children[(r, c)] = MCTSNode(prior=prior)

                # Backup: walk up, update stats, remove virtual loss.
                current_value = value
                for j in range(len(path) - 1, 0, -1):
                    node = path[j]
                    node.visit_count += 1
                    node.total_value += current_value
                    node.virtual_loss -= 1
                    current_value = -current_value

            sims_done += n

        # ---- visit proportions ----
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            legal = board.get_legal_moves()
            return {m: 1.0 / len(legal) for m in legal}
        return {m: c.visit_count / total_visits for m, c in root.children.items()}
```

- [ ] **Step 4: Remove `_select` and `_expand_and_evaluate`**

Delete the `_select` method (lines 118-145) and the `_expand_and_evaluate` method (lines 147-161) from `selfplay/mcts.py`. These are replaced by the batched loop in `search()` and the `_descend_and_tag` helper.

- [ ] **Step 5: Run existing MCTS tests**

Run: `pytest tests/test_mcts.py -v`
Expected: All 9 tests (7 existing + 2 new from Task 3) PASS. Existing callers (`select_move`, `SelfPlayGame`) use unchanged public API.

- [ ] **Step 6: Commit**

```bash
git add selfplay/mcts.py
git commit -m "feat: rewrite MCTS.search() with batched descent and virtual loss"
```

---

### Task 5: Tests for batched MCTS search

**Files:**
- Modify: `tests/test_mcts.py` — add 4 tests

- [ ] **Step 1: Write the tests**

Add these imports at the top of `tests/test_mcts.py` if not already present (check existing imports first — `Board`, `MCTS`, `Player`, `torch`, `_make_wrapper` should already be available):

```python
def test_batched_search_returns_valid_distribution():
    """Batched search returns a probability distribution over legal moves."""
    wrapper = _make_wrapper()
    mcts = MCTS(wrapper, num_simulations=50, batch_size=8)
    board = Board()
    board.make_move(7, 7)

    visit_probs = mcts.search(board)
    assert len(visit_probs) > 0
    total = sum(visit_probs.values())
    assert abs(total - 1.0) < 1e-6
    legal = board.get_legal_moves()
    for move in visit_probs:
        assert move in legal


def test_batched_search_finds_immediate_win():
    """Search finds a winning move in one step when one exists."""
    wrapper = _make_wrapper()
    mcts = MCTS(wrapper, num_simulations=20, batch_size=8)
    board = Board()
    # Set up a position where Black has 4 in a row with one open end.
    for col in range(4):
        board.make_move(7, col)      # Black
        board.make_move(14, col)     # White (arbitrary, not interfering)
    # board.make_move(7, 4) would win for Black.
    visit_probs = mcts.search(board)
    assert len(visit_probs) > 0
    best_move = max(visit_probs, key=visit_probs.get)
    assert best_move == (7, 4)


def test_batch_size_1_still_works():
    """batch_size=1 should produce the same move as a separate MCTS with equivalent settings."""
    wrapper = _make_wrapper()
    board = Board()
    board.make_move(7, 7)

    mcts_batched = MCTS(wrapper, num_simulations=50, batch_size=1)
    probs_batched = mcts_batched.search(board)
    move_batched = max(probs_batched, key=probs_batched.get)

    mcts_seq = MCTS(wrapper, num_simulations=50, batch_size=1)
    probs_seq = mcts_seq.search(Board())  # fresh board, same first move
    # Apply the same opening move then search.
    b2 = Board()
    b2.make_move(7, 7)
    probs_seq = mcts_seq.search(b2)
    move_seq = max(probs_seq, key=probs_seq.get)

    # Both should select the same move (deterministic at temp=0, same NN weights).
    assert move_batched == move_seq


def test_search_batch_larger_than_simulations():
    """batch_size > num_simulations should not crash and should work correctly."""
    wrapper = _make_wrapper()
    mcts = MCTS(wrapper, num_simulations=10, batch_size=100)
    board = Board()
    board.make_move(7, 7)

    visit_probs = mcts.search(board)
    assert len(visit_probs) > 0
    total = sum(visit_probs.values())
    assert abs(total - 1.0) < 1e-6
```

- [ ] **Step 2: Run the new MCTS tests**

Run: `pytest tests/test_mcts.py::test_batched_search_returns_valid_distribution tests/test_mcts.py::test_batched_search_finds_immediate_win tests/test_mcts.py::test_batch_size_1_still_works tests/test_mcts.py::test_search_batch_larger_than_simulations -v`
Expected: 4 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcts.py
git commit -m "test: add batched MCTS search tests"
```

---

### Task 6: Run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `./venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (84 existing + 9 new = 93 tests). Zero failures, zero errors.

- [ ] **Step 2: Verify self-play integration**

Run a quick smoke test that `SelfPlayGame.play()` works with the new batched MCTS:

```bash
./venv/bin/python -c "
from selfplay.selfplay import SelfPlayGame
from neural.wrapper import GomokuInferenceWrapper
from pathlib import Path
import tempfile, os, torch
from neural.model import GomokuNet

# Create a temp checkpoint.
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    tmp = f.name
model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5)
torch.save(model.state_dict(), tmp)

try:
    wrapper = GomokuInferenceWrapper(tmp, device='cpu')
    game = SelfPlayGame(wrapper, num_simulations=20)
    examples = game.play()
    print(f'Self-play produced {len(examples)} training examples (OK)')
    assert len(examples) > 0
finally:
    os.unlink(tmp)
"
```
Expected: Prints training example count with no errors.

- [ ] **Step 3: Commit (if any final cleanup needed)**

No files to commit at this step unless the smoke test revealed issues that needed fixes.
