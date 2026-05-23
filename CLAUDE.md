# Gomoku AI — AlphaZero-Style Engine

Hybrid Gomoku (15×15, 5-in-a-row) engine combining Monte Carlo Tree Search with
deep neural networks for position evaluation and policy guidance. Built with
PyTorch and NumPy for research, education, and experimentation.

## Project Purpose

This engine plays Gomoku at superhuman strength by fusing classical tree search
with learned neural intuition, following the AlphaZero paradigm. It is designed
as a modular research platform where every component (board, search, network,
training) can be studied, replaced, or extended independently.

**Architectural goals:**

- Strict separation between game logic, search, neural inference, and training.
- Every component is independently testable and swappable.
- The board engine is deterministic, UI-agnostic, and free of search logic.
- MCTS operates through a clean interface that accepts any board and any
  inference backend — it knows nothing about PyTorch internals.
- The neural network is a pure PyTorch module with no dependency on the game
  rules or search code.
- Self-play and training pipelines are isolated from frontend / API concerns.

**Long-term roadmap (ordered by priority):**

1. Complete self-play training loop with replay buffer.
2. Symmetry-augmented policy targets from MCTS visit counts.
3. GPU-batched MCTS inference for faster self-play.
4. Distributed self-play across multiple processes / machines.
5. Stronger CNN architectures (SE blocks, attention, deeper resnets).
6. Explainable AI visualizations (policy heatmaps, value landscapes).
7. Web-based UI with search visualization.

## Core Architecture Rules

These rules are non-negotiable. Violations should be treated as bugs.

### Modularity

- Each directory (`engine/`, `neural/`, `selfplay/`) is a self-contained
  Python package with its own `__init__.py`.
- Files within a package must have a single, well-defined responsibility.
- No file shall exceed ~500 lines. If it approaches that limit, split it.

### Separation of Concerns

- **No UI logic in `engine/` or `selfplay/`.** The board, MCTS, and neural
  code must never import or depend on frontend frameworks, display code, or
  API serialization formats.
- **No search logic in `engine/board.py`.** The board is a pure game-state
  container. It provides legal moves and terminal checks; it does not search.
- **No PyTorch references in `engine/`.** The engine layer must remain
  framework-agnostic. NumPy is acceptable; PyTorch tensors are not.
- **No game-rule logic in `neural/`.** The neural network takes tensors in,
  produces log-probabilities and values out. It does not know what a "move"
  or "player" is.

### Import Discipline

- Imports must flow in one direction: `engine` ← `neural` ← `selfplay`.
  Higher layers may import from lower layers; lower layers must never import
  from higher layers.
- `engine` has zero internal imports from other project packages.
- `neural` may import from `engine` (for board encoding).
- `selfplay` may import from both `engine` and `neural`.
- Circular imports are forbidden. If two modules need each other, extract the
  shared concern into a third module that both import.

### Extensibility

- The neural network architecture must be swappable via configuration or
  constructor arguments, not by editing the model file.
- The MCTS search must accept any callable that returns (policy, value) for a
  board — the inference wrapper is an implementation detail, not a hard
  dependency of the search algorithm.
- Board size, win length, and encoding scheme should be configurable, with
  sensible defaults for standard 15×15 Gomoku.

## Coding Standards

### Type Hints

All function signatures must include Python type hints. Use `from __future__
import annotations` at the top of every file to enable deferred evaluation.

```python
def search(self, board: Board) -> dict[tuple[int, int], float]:
    ...
```

Use `Optional[X]` rather than `X | None` for consistency with the existing
codebase. Use `NDArray[np.int8]` for NumPy arrays.

### Object-Oriented Design

- Prefer small, focused classes with a single responsibility.
- Use `@dataclass` with `slots=True` for plain data containers.
- Use `IntEnum` for discrete enumerated types (players, threat types).
- Static methods are acceptable for pure functions that logically belong to a
  class but need no instance state (e.g., `ThreatDetector.detect_all`).

### Naming

- Classes: `PascalCase` (`MCTS`, `GomokuNet`, `ThreatDetector`).
- Functions and methods: `snake_case` (`make_move`, `check_win`, `board_to_tensor`).
- Constants: `UPPER_SNAKE_CASE` (`WIN_LENGTH`, `POLICY_CUTOFF`).
- Private members: prefix with single underscore (`_neighbor_set`, `_check_win_at`).
- Avoid abbreviations unless they are universally understood in the domain
  (`num_` for "number of", `ch_` for "channels" in tensor context).

### Docstrings

- Every public class and method must have a docstring.
- Use triple-double-quote `"""` style.
- First line is a concise summary. Blank line, then details.
- Document parameters and return values in the body when the signature alone
  is insufficient.
- Private helpers may omit docstrings when the name makes the purpose obvious.

### Functions

- Keep functions short and focused. A function that does two distinct things
  should be two functions.
- Avoid boolean flag parameters that change what a function returns. Split
  into two functions instead.
- Pure functions (no side effects, deterministic output for given input) are
  preferred wherever possible.

### Premature Optimization

- Write for clarity first. Profile before optimizing.
- The one exception: operations inside the MCTS hot loop (selection, expansion,
  backup) may be optimized aggressively, but must be documented as such.

## Engine Design Philosophy

### Board (`engine/board.py`)

The board is the single source of truth for game state. It must remain:

- **Deterministic:** Given the same sequence of `make_move` calls, the board
  state must be identical every time.
- **UI-independent:** The board has no concept of "rendering," "display," or
  "user interface." Its `__repr__` exists for debugging only.
- **Framework-agnostic:** The board uses NumPy arrays internally but exposes
  no framework-specific types in its public API. Callers should not need to
  know the storage layout.

### MCTS (`selfplay/mcts.py`)

The MCTS implementation must be independent of the neural network
implementation. It depends on an inference interface (a callable that maps
board → (policy priors, value)), not on `GomokuNet` or `GomokuInferenceWrapper`
directly. This allows:

- Swapping the neural backend without touching search code.
- Testing MCTS with hand-crafted evaluation functions.
- Running MCTS with a different model architecture transparently.

### Neural Network (`neural/model.py`)

The network is a pure PyTorch `nn.Module`. It knows nothing about the game
rules, board representation, or search algorithm. It accepts a tensor of shape
`(batch, channels, 15, 15)` and returns `(log_policy, value)`. This contract
is the only coupling between the neural and engine layers.

The `neural/wrapper.py` module bridges the gap: it handles checkpoint loading,
device placement, and the board-to-tensor conversion. MCTS talks to the
wrapper, not to the raw model.

### Self-Play Pipeline

The self-play pipeline (game generation, MCTS-guided move selection, replay
buffer management, training loop) must remain isolated from any frontend or
API code. It is a pure data-producing and model-training system. The output is
a trained model checkpoint; how that checkpoint is served to users is a
separate concern.

## File Responsibilities

### `engine/board.py`

The canonical game-state container. Responsibilities:

- Store the 15×15 grid (NumPy `int8` array: +1 Black, -1 White, 0 empty).
- Track `current_player` and `move_history`.
- Provide `make_move(row, col)` and `undo_move()` with O(1) complexity.
- Provide `get_legal_moves()` — returns only positions adjacent to existing
  stones (plus center for the opening move).
- Provide `check_win()` and `is_terminal()` — win detection is incremental,
  scanning only the four line directions from the last-placed stone.
- Provide `copy()` for parallel MCTS simulations.

Non-responsibilities: search, evaluation, threat detection, UI, serialization.

### `engine/threats.py`

Pattern-based threat detection for forced-move heuristics. Responsibilities:

- Scan all four line directions for five, open-four, closed-four, and
  open-three patterns.
- Detect double threats (combinations that cannot be blocked in one move).
- Provide heuristic position scoring for evaluation fallback.
- Support the MCTS threat-override shortcut: when a forced win or must-block
  exists, the search short-circuits without neural evaluation.

Non-responsibilities: search, neural inference, move generation.

### `engine/encoding.py`

Bridge between board state and neural network input. Responsibilities:

- `board_to_tensor(board)` → `(1, 3, 15, 15)` FloatTensor with channels:
  channel 0 (current player stones), channel 1 (opponent stones), channel 2
  (turn indicator: 1.0 if Black, 0.0 if White).
- `policy_to_move_probs(log_policy, board)` → legal-move-filtered,
  normalized probability distribution.

### `neural/model.py`

Pure PyTorch module. Responsibilities:

- Define `GomokuNet`: a dual-headed residual CNN.
  - Shared trunk: initial conv → batch norm → N residual blocks.
  - Policy head: conv → batch norm → FC → log-softmax over 225 cells.
  - Value head: conv → batch norm → FC → tanh scalar in [-1, 1].
- Define `ResidualBlock`: two 3×3 convs with batch norm and skip connection.
- Be serializable via `state_dict()` / `load_state_dict()`.

Non-responsibilities: board encoding, move filtering, checkpoint management,
inference orchestration.

### `neural/wrapper.py`

Inference wrapper that loads a trained checkpoint and presents a clean
`evaluate(board)` API. Responsibilities:

- Load model weights from disk.
- Handle device placement (CUDA, MPS, CPU).
- Convert board → tensor via `board_to_tensor`.
- Run inference and convert output → `(move_probs, value)`.
- Optionally apply threat-detection overrides via `evaluate_with_threats`.

### `selfplay/mcts.py`

AlphaZero-style MCTS with PUCT. Responsibilities:

- `MCTSNode`: dataclass storing `prior`, `visit_count`, `total_value`, and
  `children` dict. The `q` property returns mean action value.
- `MCTS.search(board)` → visit-count distribution over legal moves.
- `MCTS.select_move(board, temperature)` → single move for game play.
- Virtual-board mutation: the search mutates and restores a single board copy
  rather than copying per node.
- Threat short-circuit: detect forced wins and must-block positions before
  entering the neural search loop.

### `selfplay/selfplay.py` (planned)

Self-play game generation. Responsibilities:

- Run complete games: both players use MCTS-guided move selection.
- Store game trajectories: (board state → MCTS visit distribution, outcome).
- Apply Dirichlet noise to root priors for exploration.
- Handle resignation heuristics.

### `selfplay/replay_buffer.py` (planned)

Fixed-size replay buffer for training. Responsibilities:

- Store (state, policy_target, value_target) tuples.
- Support random sampling of batches.
- Handle symmetry augmentation on retrieval (rotations, reflections).
- Manage buffer capacity with FIFO eviction.

### `api.py` or `frontend/` (planned)

HTTP API or WebSocket server for UI consumption. Responsibilities:

- Accept move requests, return engine responses.
- Serve search visualization data.
- Manage game sessions.

Non-responsibilities: game logic, search, training.

## MCTS Requirements

### Node Statistics

Each `MCTSNode` must track:

| Field         | Type                           | Description                           |
|---------------|--------------------------------|---------------------------------------|
| `prior`       | `float`                        | Prior probability P(s,a) from network |
| `visit_count` | `int`                          | N(s,a) — number of times this edge was traversed |
| `total_value` | `float`                        | W(s,a) — sum of backed-up values      |
| `children`    | `dict[(r,c), MCTSNode]`       | Child nodes keyed by move coordinates |

The `q` property returns `total_value / visit_count` (or 0.0 if unvisited).

### PUCT Selection

The selection formula at an internal node is:

```
a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
```

Where `c_puct` is a tunable exploration constant (default 2.5).

Selection must be deterministic (no tie-breaking noise at internal nodes).

### Expansion and Evaluation

When a leaf node is reached:

1. If the board is terminal, back up the game result immediately (no network
   call).
2. Otherwise, call the inference wrapper to get (policy priors, value).
3. Create child `MCTSNode` instances for each legal move with non-negligible
   prior, capped at `POLICY_CUTOFF` (40) moves.
4. Back up the value estimate.

### Value Backup

Values are negated at each level of the tree. The value backed up to a node
is the negation of the value from the child's perspective. Win for the player
who just moved = +1.0, loss = -1.0, draw = 0.0.

### Neural Priors

The MCTS must support future integration of neural network priors. The
current implementation already uses priors from the policy head. Future work
may include:
- Dirichlet noise at the root for exploration during self-play.
- Temperature-annealed move selection.
- Virtual loss for parallel MCTS.

## Neural Network Requirements

### Architecture

Dual-headed residual CNN:

```
Input: (batch, 3, 15, 15)
  │
  ▼
Conv2d(3→64, 3×3) → BatchNorm → ReLU
  │
  ▼
ResidualBlock(64) × N      (N=5 default)
  │
  ├─────────────────────┐
  ▼                     ▼
Policy Head          Value Head
Conv2d(64→2, 1×1)   Conv2d(64→1, 1×1)
BatchNorm            BatchNorm
ReLU                 ReLU
Flatten → FC(450, 225)  Flatten → FC(225, 64) → ReLU → FC(64, 1) → Tanh
LogSoftmax           Tanh scalar
```

### Tensor Encoding

The 3-channel input is always from the perspective of the current player:

- **Channel 0:** 1.0 where current player has a stone, 0.0 elsewhere.
- **Channel 1:** 1.0 where opponent has a stone, 0.0 elsewhere.
- **Channel 2:** 1.0 everywhere if Black to move, 0.0 if White to move.

This encoding is invariant to the player identity — the network learns
position patterns independent of which color it plays.

### Training Compatibility

The network must produce:

- **Policy output:** log-softmax over 225 cells. Training uses KL-divergence
  or cross-entropy loss against MCTS visit-count distributions.
- **Value output:** tanh scalar in [-1, 1]. Training uses MSE loss against
  game outcomes (+1 win, -1 loss, 0 draw).

The model must serialize cleanly via `torch.save(model.state_dict(), path)` and
`model.load_state_dict(torch.load(path))`.

## Self-Play Requirements

### AlphaZero Training Loop

The training pipeline follows the standard AlphaZero self-play cycle:

1. **Self-play:** Current best model plays games against itself. Moves are
   selected by MCTS guided by the model's own policy and value outputs.
2. **Data generation:** Each game produces training examples: (board state,
   MCTS visit-count distribution, final game outcome from the perspective of
   the player to move).
3. **Training:** Sample mini-batches from the replay buffer. Train the
   network to predict both the MCTS policy targets (cross-entropy) and game
   outcomes (MSE). Sum the two losses with equal weight.
4. **Evaluation:** Periodically pit the new model against the previous best.
   If the new model wins significantly more than 50% of games, it replaces
   the old model.
5. **Repeat** from step 1.

### Replay Buffer

- Fixed capacity (e.g., 500,000 positions).
- Stores tuples of `(board_tensor, policy_target, value_target)`.
- Supports uniform random sampling for training batches.
- FIFO eviction when capacity is exceeded.
- Symmetry augmentation: on retrieval, randomly apply one of the 8
  dihedral symmetries (rotations × reflections) to both the board tensor
  and the policy target. This provides 8× effective data.

### Policy Target Generation

Policy targets for training are the MCTS visit-count distributions after
search, not the raw network priors. The search produces improved policy
targets that the network learns to imitate — this is the "policy iteration"
core of AlphaZero.

### Exploration

During self-play, Dirichlet noise is added to the root node's prior
distribution to encourage exploration:

```
P_root = (1 - epsilon) * P_network + epsilon * Dirichlet(alpha)
```

Typical values: epsilon = 0.25, alpha = 0.03 (or 10/num_legal_moves).

## Performance Expectations

- **Board copying** must be fast. The current `Board.copy()` uses NumPy array
  copy and set copy — this is acceptable. Future parallel MCTS may benefit
  from a more compact state representation.
- **Legal move generation** must avoid full-board scans. The current
  neighbor-set approach keeps the branching factor at ~20-40 instead of 225.
- **Win detection** is incremental (scan from last move only), not a full
  board scan.
- **Batched inference:** Currently the wrapper evaluates one board at a time.
  For self-play throughput, batches of positions should be evaluated together.
  This is a planned optimization.
- **Profiling:** Run `cProfile` or `torch.profiler` before optimizing. Do not
  guess where the bottleneck is.

## Git & GitHub Rules

### Never Commit

- `venv/` — virtual environment directory (already in `.gitignore`).
- `__pycache__/` and `*.pyc` — compiled Python bytecode.
- `.pytest_cache/` — pytest cache.
- `checkpoints/*.pt` — model weight files (use `.gitignore` exception for
  small test fixtures only).
- `data/` — training data and replay buffer dumps.
- Any file larger than 10 MB.

### Commit Discipline

- Each commit should be a single logical change: a bug fix, a feature
  addition, a refactor. Do not bundle unrelated changes.
- Commit messages should be descriptive: explain the "why," not just the
  "what."
- Keep the repository structure clean. When adding a new module, place it in
  the correct package. Do not add files to the repository root except for
  top-level configuration (`requirements.txt`, `CLAUDE.md`, `main.py`).

### Branch Strategy

- `main` branch should always be in a working state. All tests pass.
- Feature branches are named `feature/<description>` or `fix/<description>`.
- Merge via pull request, not direct push to main.

## Testing Expectations

### Unit Tests

- **Board logic:** Every method on `Board` must have test coverage. Test
  `make_move`, `undo_move`, `get_legal_moves`, `check_win`, `is_terminal`,
  `copy`, and all edge cases (board edges, full board, undo from empty board).
- **Win detection:** Test all four directions (horizontal, vertical, both
  diagonals). Test edge-adjacent wins. Test that four-in-a-row is NOT a win.
  Test full-board draw.
- **Legal move generation:** Test empty board (center only), test with stones
  (neighbors only), test that occupied cells are excluded.
- **MCTS:** Test that search returns a valid probability distribution. Test
  that immediate wins are found. Test that must-block threats are respected.
  Test that terminal boards return empty distributions.
- **Threat detection:** Test all four threat types in all four directions.
  Test edge cases (board edges, blocked ends, split patterns). Test
  double-threat detection.
- **Neural network:** Test output shapes, log-softmax property, value range.
  Test that forward pass does not crash on various batch sizes.

### Determinism

- Search behavior should be reproducible given the same random seed where
  possible. Neural inference with `torch.no_grad()` on CPU with fixed weights
  is deterministic. MCTS with threat_override=True and no Dirichlet noise is
  deterministic.
- Tests that rely on an untrained network's random outputs should tolerate
  some variance (e.g., only check that the output is a valid distribution,
  not that it matches specific values).

### Test Organization

- Tests live in `tests/` at the project root.
- File naming: `test_<module>.py` mirrors the source module under test.
- Use `pytest` as the test runner. No `unittest.TestCase` subclassing.
- Use fixtures and helpers (like `_make_wrapper()`) to reduce boilerplate.

## Documentation Expectations

### Code Comments

- Comment the **why**, not the **what**. Code should be self-documenting for
  what it does; comments explain why that choice was made.
- Mathematical formulas (PUCT, value backup, Dirichlet noise) should be
  documented in comments near the implementation, with plain-English
  explanations of each term.
- Non-obvious optimizations (neighbor-set management, virtual board mutation)
  must have comments explaining the approach and why it is correct.

### Markdown Documentation

- `CLAUDE.md` (this file): project overview, architecture, rules, and
  standards. The authoritative reference for contributors and AI assistants.
- `README.md`: user-facing overview, setup instructions, usage examples.
  Keep it concise.
- Architecture docs for major subsystems (MCTS, neural network, self-play
  pipeline) should live in `docs/` as separate markdown files when they
  become too detailed for `CLAUDE.md`.

### For Future Contributors

- Assume the next person reading the code is a competent engineer who has
  never seen this project before.
- Every module's `__init__.py` or module docstring should explain what the
  module provides and where to look for the entry points.
- Avoid jargon specific to this project without defining it.

## Future Roadmap

These features are planned but not yet implemented. Contributions should
align with this roadmap.

### Phase 1 — Core Training (current focus)

- [ ] `selfplay/selfplay.py` — self-play game generation with MCTS.
- [ ] `selfplay/replay_buffer.py` — replay buffer with symmetry augmentation.
- [ ] `selfplay/train.py` — training loop with loss computation and checkpointing.
- [ ] Model evaluation (pit new model vs old model).

### Phase 2 — Performance

- [ ] Batched neural inference for multiple MCTS leaf evaluations.
- [ ] GPU-accelerated MCTS where leaf evaluations run on GPU.
- [ ] Virtual loss for parallel MCTS within a single search.
- [ ] Profile and optimize the hot path.

### Phase 3 — Scaling

- [ ] Distributed self-play (multiple worker processes generating games,
  central trainer consuming the replay buffer).
- [ ] Stronger architectures: Squeeze-and-Excitation blocks, attention layers,
  deeper residual networks.

### Phase 4 — Explainability

- [ ] Saliency maps showing which board regions influenced the network's decision.
- [ ] Activation visualization for residual blocks.
- [ ] Comparison tool: human move vs. AI recommended move with explanation.

### Phase 5 — UI and Visualization

- [ ] Web-based game UI (React or similar).
- [ ] Search tree visualization (which moves MCTS considered, visit counts).
- [ ] Policy heatmap overlay on the board.
- [ ] Value landscape visualization (how the evaluation changes across the board).
