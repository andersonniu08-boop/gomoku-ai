# NeuralGomoku — ML Engine

Hybrid Gomoku (15×15, 5-in-a-row) engine combining Monte Carlo Tree Search with
deep neural networks for position evaluation and policy guidance. Built with
PyTorch and NumPy for research, education, and experimentation.

## Project Purpose

This engine plays Gomoku at superhuman strength by fusing classical tree search
with learned neural intuition. It is designed
as a modular research platform where every component (board, search, network,
training) can be studied, replaced, or extended independently.

**Architectural goals:**

- Strict separation between game logic, search, neural inference, and training.
- Every component is independently testable and swappable.
- The board engine is deterministic, UI-agnostic, and free of search logic.
- MCTS operates through a clean interface that accepts any board and any
  inference backend — it knows nothing about PyTorch internals.

## Core Architecture Rules

These rules are non-negotiable. Violations should be treated as bugs.

### Modularity

- Each directory (`engine/`, `neural/`, `selfplay/`) is a self-contained
  Python package with its own `__init__.py`.
- Files within a package must have a single, well-defined responsibility.
- Most files should stay under ~500 lines. A few hot-path files (MCTS, threat
  detection, tactical solver) may exceed this — document why when they do.

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
- `neural` may import from `engine` (for board encoding). Only exception:
  `wrapper.py` uses a local `_NoopProfiler` stub rather than importing from
  `selfplay.profiler`, preserving the dependency direction.
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

## Build / Test / Run Commands

```bash
pip install -r requirements.txt
python -m pytest tests/                    # full test suite
python -m selfplay.train                   # start training loop
python -m selfplay.worker                  # start self-play worker
python -m ui.server                        # start web UI
python -m tools.benchmark_runner           # run benchmarks
```

Run tests before committing. All tests must pass on main.

## Coding Standards

### Type Hints

All function signatures must include Python type hints. Use `from __future__
import annotations` at the top of every file to enable deferred evaluation.

```python
def search(self, board: Board) -> dict[tuple[int, int], float]:
    ...
```

Use `Optional[X]` for consistency with the existing codebase (not `X | None`).
Use `NDArray[np.int8]` for NumPy arrays.

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

The MCTS implementation depends on an inference interface (a callable that maps
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

### Self-Play & Training Pipeline

The self-play pipeline (game generation, MCTS-guided move selection, replay
buffer management, training loop) is isolated from any frontend or API code.
It is a pure data-producing and model-training system. The output is a trained
model checkpoint; how that checkpoint is served to users is a separate concern.

## File Map

### `engine/board.py`
15×15 board with NumPy backing. `get_legal_moves()` returns all empty squares
(standard Gomoku). Win detection is incremental (scans from last move).

### `engine/threats.py`
Pattern-based threat detection (FIVE, OPEN_FOUR, CLOSED_FOUR, OPEN_THREE).
Supports the MCTS threat-override shortcut.

### `engine/encoding.py`
`board_to_tensor(board)` → `(1, 3, 15, 15)` FloatTensor.
`policy_to_move_probs(log_policy, board)` → legal-move-filtered distribution.

### `engine/tactical.py`
Deterministic tactical solver with deep forced-line search. Used by tools and
explainability modules. Not part of the main MCTS pipeline (MCTS uses
`ThreatDetector` + `move_ordering` directly for speed).

### `neural/model.py`
`GomokuNet`: dual-headed residual CNN with multi-head self-attention (4 heads),
dilated convolution pyramid (1→2→3→2→1), CBAM spatial attention, SE channel
gating, and stochastic depth (DropPath). Default: 10 blocks, 128 channels.

Policy head is fully convolutional (3×3 → 1×1 → log-softmax). Value head uses
dual global pooling (avg + max) → FC → tanh.

### `neural/wrapper.py`
Checkpoint loading, device placement, board→tensor conversion, inference.
Provides `evaluate()`, `batch_evaluate()`, `evaluate_with_threats()`.

### `selfplay/mcts.py`
MCTS with PUCT. Key features:
- Batched descent with virtual loss for GPU-efficient parallel leaf evaluation.
- Tree reuse: re-roots previous search tree across consecutive moves.
- Threat override short-circuit at root.
- Dirichlet noise at root for self-play exploration.
- Time-budget mode (wall-clock limit) and fixed-simulation mode.
- `search()` returns visit distribution; `search_with_stats()` also returns
  Q-values and priors.

### `selfplay/move_ordering.py`
Tactical move ordering and candidate pruning. Filters neural priors using
incremental line scanning (no board copies). Forces critical moves (wins /
must-blocks) into the expansion set.

### `selfplay/evaluator.py`
`BatchedLeafEvaluator`: efficient batched GPU inference for MCTS leaves.
Builds batched tensors from numpy arrays, runs one forward pass per batch,
extracts values with a single `.tolist()` call.

### `selfplay/selfplay.py`
`SelfPlayGame`: runs one self-play game with MCTS-guided move selection.
Uses temperature stages for exploration, Dirichlet noise at root, opening
diversity via high-temperature sampling, and resignation heuristics.
Outputs `TrainingExample` (state, policy_target, value_target) triples.

### `selfplay/replay_buffer.py`
Fixed-capacity FIFO buffer (500K default). Applies random D₄ symmetry
augmentation on retrieval (8× effective data at 1/8 memory). Persists via
`state_dict()` / `from_state_dict()`.

### `selfplay/train.py`
Training loop:
1. Ingest worker-generated game files.
2. Generate local self-play games with current model.
3. Sample from replay buffer and train (cross-entropy policy + MSE value).
4. Evaluate latest vs best periodically; promote if win rate > threshold.
5. Elo tracking, checkpointing, replay diversity monitoring.
Uses CosineAnnealingLR, mixed-precision (GradScaler), gradient clipping.

### `selfplay/worker.py`
Distributed self-play worker. Polls `checkpoints/latest.pt` for updates,
generates games, writes `game_*.pt` files for the trainer to consume.
Graceful SIGINT/SIGTERM handling.

### `selfplay/config.py`
`StrengthConfig` dataclass and built-in presets (Fast=200, Medium=800,
Strong=3000, Turbo=3000+3s time budget). `to_mcts_kwargs()` for MCTS
construction.

### `selfplay/elo.py`
Elo rating tracker: `EloTracker` registers checkpoints, records matches,
persists to JSON. Uses standard Elo formula with K=96.

### `selfplay/eval_registry.py`
`EvalRegistry`: coordinates Elo tracking, tactical benchmarks, and regression
detection. `load()`/`save()` for persistence across training sessions.

### `selfplay/bench_suite.py`
Tactical benchmark suite: known positions that test win-in-1, forced defense,
double threats, tactical sequences, opening, and endgame categories.
`run_benchmark_suite()` for CI / regression checking.

### `selfplay/profiler.py`
Lightweight hierarchical profiler with context-manager API. Used for MCTS
hot-path timing in `benchmark.py`.

### `selfplay/benchmark.py`
MCTS benchmarking script. Measures sims/sec across opening, midgame, and
threat positions. Outputs profiler breakdown.

### `ui/server.py`
Flask server with `/api/search`, `/api/new-game`, `/api/config` endpoints.
Strength presets (Fast/Medium/Strong), side selection, tree reuse across
moves. Lazily loads model from `checkpoints/best.pt`.

### `explain/` directory
Saliency maps, activation visualization, and human-vs-AI comparison tools.

### `tools/` directory
`benchmark_runner.py` — aggregate benchmark harness.
`regression_test.py` — regression detection against baselines.
`validate_tactical.py` — tactical correctness validation.
`report.py` — evaluation report generator.
`bench_nn_architecture.py` — neural architecture benchmarks.
`bench_selfplay_quality.py` — self-play quality metrics.

## Key Conventions

- `MCTS.__init__` default `batch_size=8` for GPU-efficient batched inference.
- Self-play: `tree_reuse=False`, Dirichlet noise enabled.
- Evaluation: `tree_reuse=False`, temperature=0.0 (greedy).
- Server/UI: `tree_reuse=True` for cumulative search across moves.
- Checkpoints saved atomically (write `.tmp` → rename).
- All checkpoint paths relative to project root (`checkpoints/`, `data/`).

## MCTS Requirements

### Node Statistics

Each `MCTSNode` must track:

| Field         | Type                           | Description                           |
|---------------|--------------------------------|---------------------------------------|
| `prior`       | `float`                        | Prior probability P(s,a) from network |
| `visit_count` | `int`                          | N(s,a) — number of times this edge was traversed |
| `total_value` | `float`                        | W(s,a) — sum of backed-up values      |
| `children`    | `dict[(r,c), MCTSNode]`       | Child nodes keyed by move coordinates |

The `q` property returns `(total_value - virtual_loss) / (visit_count + virtual_loss)`.

### PUCT Selection

```
a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
```

Where `c_puct` is a tunable exploration constant (default 2.5).

### Value Backup

Values are negated at each level of the tree. Win for the player who just
moved = +1.0, loss = -1.0, draw = 0.0.

## Neural Network Architecture

```
Input: (batch, 3, 15, 15)
  │
  ▼
Conv2d(3→128, 3×3) → BatchNorm → ReLU
  │
  ▼
SEResidualBlock(128) × 10  (dilated pyramid: 1→2→3→2→1)
  ├── Multi-head self-attention (4 heads, LayerNorm)
  ├── SE channel gating (reduction=8)
  ├── CBAM spatial attention (3×3 kernel)
  └── DropPath stochastic depth (0→0.1 linear schedule)
  │
  ├──────────────────────────────┐
  ▼                              ▼
Policy Head                   Value Head
3×3 Conv(128→32) → BN → ReLU  Dual global pool (avg+max, 2C)
1×1 Conv(32→1)                 FC(256→64) → ReLU
Flatten → LogSoftmax           FC(64→1) → Tanh
```

- **Policy output:** log-softmax over 225 cells.
- **Value output:** tanh scalar in [-1, 1].
- **Tensor encoding:** channel 0 = current player stones, channel 1 = opponent
  stones, channel 2 = turn indicator (1.0 Black, 0.0 White).

## Self-Play & Training

### Training Loop

1. **Self-play:** Current model plays games against itself via MCTS.
2. **Data:** Each game produces (state, MCTS visit dist, outcome) triples.
3. **Training:** Sample mini-batches from replay buffer. Loss = cross-entropy
   (policy) + MSE (value).
4. **Evaluation:** Pit latest vs best (temperature=0, alternating colors).
   Promote if win rate > threshold (default 55%).
5. **Repeat** from step 1.

### Replay Buffer

- 500K capacity, FIFO eviction.
- D₄ symmetry applied on retrieval (random rotation + flip).
- Persisted to `data/replay_buffer.pt`.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mcts_simulations` | 800 | Sims per move during self-play |
| `eval_simulations` | 200 | Sims per move during evaluation |
| `batch_size` | 256 | Training mini-batch size |
| `learning_rate` | 0.001 | Adam initial LR |
| `temperature_stages` | [(0,1.0), (15,0.5), (30,0.0)] | Move-indexed temperature schedule |
| `dirichlet_alpha` | 0.03 | Root Dirichlet noise concentration |
| `resignation_threshold` | -0.9 | Root value below which AI considers resigning |

## Testing

```bash
python -m pytest tests/ -v --timeout=60
```

- Tests live in `tests/`, named `test_<module>.py`.
- Use `pytest` (no `unittest.TestCase`).
- MCTS tests use random-weight models (no training required).
- Server tests need `checkpoints/best.pt` to exist (pre-existing failures
  if architecture mismatches).

### Known Flaky Tests

- `test_tree_reuse_fallback_on_unknown_move` — the "unknown" move may
  sometimes land in the tree due to random network priors. Retry if it fails.

## Performance

- Default `batch_size=8` in MCTS gives ~3-4x self-play throughput vs batch=1.
- Batched evaluator builds tensors from numpy arrays (avoids per-board `torch.cat`).
- Model forward at ~5ms on A5000 (10-block, 128-channel, full attention).
- Training loop bottleneck is game generation, not gradient updates.
- For multi-worker setups: one trainer, N workers writing `game_*.pt` files.

## Git Rules

### Never Commit

- `venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`
- `checkpoints/*.pt` (except test fixtures)
- `data/` — replay buffer dumps
- `game_examples/` — worker-produced game files
- Any file larger than 10 MB

### Commit Discipline

- One logical change per commit.
- Descriptive messages explaining "why," not just "what."
- Main branch always passes all tests.

## Future Roadmap

### Phase 1 — Core Training ✅
- [x] Self-play game generation with MCTS
- [x] Replay buffer with FIFO + symmetry augmentation
- [x] Training loop with checkpointing + evaluation
- [x] Model evaluation (new vs best)

### Phase 2 — Stability & Scaling (in progress)
- [x] Batched neural inference (`BatchedLeafEvaluator`)
- [x] Virtual loss for parallel MCTS
- [x] Strength presets (`StrengthConfig`)
- [ ] Distributed self-play (multi-worker game generation)
- [ ] LR warmup + schedule tuning for stable convergence

### Phase 3 — Architecture
- [x] Multi-head self-attention
- [x] Multi-scale dilated convolutions
- [x] SE + CBAM spatial attention
- [x] Stochastic depth (DropPath)
- [x] Fully convolutional policy head
- [ ] Deeper networks (20+ blocks) with residual scaling
- [ ] Mixture-of-experts policy head

### Phase 4 — Explainability ✅
- [x] Saliency maps
- [x] Activation visualization
- [x] Human-vs-AI comparison tool

### Phase 5 — UI & Infrastructure
- [x] Web-based game UI (Flask + Canvas)
- [x] Search tree visualization (visit counts, Q-values)
- [x] Elo tracking + benchmarking
- [x] Regression detection
- [ ] Policy heatmap overlay on the board
- [ ] Runpod deployment template
