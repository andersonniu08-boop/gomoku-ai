# Explainability Phase — Design Spec

## Overview

Phase 4 adds explainability tools to the Gomoku AI engine. Three independent
subsystems let users understand *why* the AI chose a move, *how* the neural
network processes a board position internally, and *how good* a human move is
compared to the AI recommendation.

All three subsystems live in a new `explain/` package and share zero runtime
dependencies between them. They are UI-independent: each produces structured
data (NumPy arrays, dataclass instances) that a future web frontend or CLI
can render.

## Architecture

```
explain/
├── __init__.py          # re-exports: compute_saliency, capture_activations, compare_move
├── saliency.py          # Workstream A — gradient-based input attribution
├── activations.py       # Workstream B — intermediate feature map capture
└── comparison.py        # Workstream C — human move vs AI recommendation
```

### Shared pre-requisites (already landed)

1. **`GomokuInferenceWrapper.evaluate_raw(board)`** in `neural/wrapper.py`:
   Returns `(log_policy, value)` tensors without `torch.no_grad()`.
   Used by Workstreams A and B for live-graph inference.

2. **`MCTS.search_with_stats(board)`** in `selfplay/mcts.py`:
   Returns a `SearchResult` dataclass with visit counts, Q values, and priors.
   Used by Workstream C.

### Shared data conventions

All modules adhere to these contracts:

- **15×15 grids**: `NDArray[np.float32]` with shape `(15, 15)`, values in `[0, 1]`.
- **Move representation**: `tuple[int, int]` where ints are row, col (0-indexed).
- **Move lists**: `list[tuple[tuple[int, int], float]]` for policy outputs.
- **Tensors**: PyTorch tensors shape `(B, 3, 15, 15)` for input, `(B, 225)` for policy, `(B, 1)` for value.
- **Package isolation**: No `explain/` module imports from another `explain/` module.

---

## Workstream A — Saliency Maps

### Purpose

Attribution heatmaps showing which board regions influenced the network's
decision. Uses Integrated Gradients as the primary method with vanilla
gradients as a fast-path option.

### File: `explain/saliency.py`

### Public API

```python
def compute_saliency(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    *,
    method: str = "ig",
    target: str = "value",
    n_steps: int = 50,
) -> SaliencyMap:
    """Compute a saliency map for the given board.

    Args:
        wrapper: Inference wrapper with evaluate_raw().
        board: The board position to explain.
        method: "ig" for Integrated Gradients, "vanilla" for single-gradient.
        target: "value", "policy", or "policy_move(r,c)".
        n_steps: Number of interpolation steps for IG (ignored for vanilla).

    Returns:
        SaliencyMap with 15x15 attribution grid.
    """

def attribution_to_grid(
    raw_gradients: torch.Tensor,
) -> NDArray[np.float32]:
    """Convert a (3, 15, 15) gradient tensor to a single (15, 15) heatmap.

    Max-pools across channels, takes absolute value, normalizes to [0, 1].
    """
```

### Data Structures

```python
@dataclass(slots=True)
class SaliencyMap:
    grid: NDArray[np.float32]       # shape (15, 15), values [0, 1]
    method: str                      # "integrated_gradients" | "vanilla"
    target: str                      # e.g. "value" | "policy" | "policy_move(7,3)"
    n_steps: int | None              # None for vanilla
```

### Internal design

**Target function factory:**

```python
def _make_target_fn(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    target: str,
) -> torch.Tensor:
    """Return a scalar tensor representing the target to attribute.

    - "value" -> value[0, 0]
    - "policy" -> log_policy[0, :].sum()
    - "policy_move(r,c)" -> log_policy[0, r * BOARD_SIZE + c]
    """
```

**Integrated Gradients algorithm:**

1. Create baseline tensor: all zeros -> `(1, 3, 15, 15)`.
2. `diff = input_tensor - baseline`.
3. For `i` in `linspace(0, 1, n_steps)`:
   - `scaled = baseline + i * diff`, clone, `requires_grad_(True)`.
   - Forward pass (raw model call).
   - Target scalar -> `backward()`.
   - `accumulated_grads += scaled.grad`.
4. `attributions = (input_tensor - baseline) * accumulated_grads / n_steps`.
5. Verify completeness: `sum(attributions)` ≈ `model_output(input) - model_output(baseline)`.

**Vanilla gradient algorithm:**

1. `input_tensor.clone().requires_grad_(True)`.
2. Forward pass, target scalar -> `backward()`.
3. `attributions = input_tensor.grad`.

**Channel aggregation (both methods):**

1. Take `abs()` of attributions (3 channels).
2. Max-pool across channel dim: `(3, 15, 15)` -> `(15, 15)`.
3. Divide by `max(abs().max())` -> `[0, 1]`.

### Gradient safety rules

- Never leave `requires_grad` set on the wrapper's model parameters.
- Call `model.zero_grad(set_to_none=True)` before each backward pass.
- Use a fresh `tensor.clone().requires_grad_(True)` for each interpolation
  step (not in-place operations on a shared tensor).
- Wrap IG loop body except the active step in `torch.no_grad()` to avoid
  storing computation graphs for all steps simultaneously.

---

## Workstream B — Activation Visualization

### Purpose

Capture intermediate feature maps from GomokuNet's residual blocks. The output
is a set of 15×15 activation grids — one per channel per block — that can be
rendered to show what patterns each layer detects.

### File: `explain/activations.py`

### Public API

```python
def capture_activations(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    *,
    blocks: Optional[list[int]] = None,
    channels: Optional[list[int]] = None,
) -> ActivationSnapshot:
    """Run a forward pass and capture activations from residual blocks.

    Args:
        wrapper: Inference wrapper with evaluate_raw().
        board: The board position.
        blocks: Indices of blocks to capture (None = all).
        channels: Indices of channels per block (None = all).

    Returns:
        ActivationSnapshot with captured data (already moved to CPU as numpy).
    """

def select_top_channels(
    snapshot: ActivationSnapshot,
    block_idx: int,
    k: int = 16,
) -> list[int]:
    """Return the k channel indices with highest L2 norm in a given block."""

def channel_to_grid(
    snapshot: ActivationSnapshot,
    block_idx: int,
    channel_idx: int,
) -> NDArray[np.float32]:
    """Extract a single channel as a (15, 15) float32 grid."""
```

### Data Structures

```python
@dataclass(slots=True)
class ActivationSnapshot:
    """Captured activations from residual blocks.

    activations[i] has shape (num_channels, 15, 15) for block i,
    stored as float32 numpy on CPU.
    """
    activations: list[NDArray[np.float32]]
    block_indices: list[int]
    channel_count: int
```

### Hook Architecture

Use `register_forward_hook` on individual `SEResidualBlock` / `ResidualBlock`
instances within `model.res_blocks` (an `nn.ModuleList`).

Context manager pattern:

```python
class ActivationCapture:
    def __init__(self, model: nn.Module, block_indices: list[int]):
        self._model = model
        self._handles: list[RemovableHandle] = []
        self.activations: list[torch.Tensor] = []
        for idx in block_indices:
            block = model.res_blocks[idx]
            handle = block.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self.activations[idx] = output.detach().cpu()
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
```

### Cleanup guarantees

- `ActivationCapture.close()` must always be called (use context manager).
- After cleanup, verify: `model._forward_hooks` dict is empty (or no foreign
  hooks remain).
- If the forward pass fails, cleanup still runs.

### Memory management

- Each activation tensor: `(1, num_channels, 15, 15) * float32 = 115 KB` at
  default 128 channels. 10 blocks = 1.15 MB per board.
- Immediately `.cpu().numpy()` inside the hook to free GPU memory.
- Store as contiguous numpy arrays.

---

## Workstream C — Human vs AI Move Comparison

### Purpose

Compare a human-chosen move against the AI recommendation. Given a board
position and a move, produces a structured report: the AI's top-k moves with
visit counts, priors, and Q values; the human move's rank and statistics; the
position value before and after the move.

### File: `explain/comparison.py`

### Prerequisites (to add to `selfplay/mcts.py`)

Add a `SearchResult` dataclass and `search_with_stats()` method to MCTS:

```python
@dataclass(slots=True)
class SearchResult:
    visit_counts: dict[tuple[int, int], int]
    q_values: dict[tuple[int, int], float]
    priors: dict[tuple[int, int], float]
    total_simulations: int

class MCTS:
    # ... existing code ...

    def search_with_stats(self, board: Board) -> SearchResult:
        """Like search() but also returns Q-values and priors.

        Uses the same internal search loop as search() but exposes
        root children's full statistics instead of just visit proportions.
        """
        # Same search loop as search() but returns richer result
```

**Implementation note:** The cleanest approach is to factor the search loop
into a shared helper that both `search()` and `search_with_stats()` call,
avoiding code duplication.

### Public API

```python
def compare_move(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    human_move: tuple[int, int],
    *,
    use_mcts: bool = True,
    num_simulations: int = 400,
    top_k: int = 5,
) -> MoveComparison:
    """Compare a human move against the AI's recommendation.

    Args:
        wrapper: Inference wrapper.
        board: Board position before the move.
        human_move: The (row, col) the human played.
        use_mcts: If True, run MCTS for visit-based comparison.
        num_simulations: MCTS iterations (only if use_mcts).
        top_k: Number of top AI moves to include.

    Returns:
        MoveComparison with structured comparison data.
    """

def compare_move_fast(
    wrapper: GomokuInferenceWrapper,
    board: Board,
    human_move: tuple[int, int],
    *,
    top_k: int = 5,
) -> MoveComparison:
    """Fast comparison using policy head only (no MCTS).

    Useful for quick feedback or when MCTS overhead is unwanted.
    Sets use_mcts=False internally.
    """
```

### Data Structures

```python
@dataclass(slots=True)
class MoveCandidate:
    move: tuple[int, int]
    prior: float             # policy head probability
    visit_count: int         # 0 if not MCTS
    q_value: float           # 0.0 if not MCTS
    is_human_move: bool

@dataclass(slots=True)
class MoveComparison:
    board: Board
    human_move: tuple[int, int]
    legal: bool                          # was the human move legal?
    top_candidates: list[MoveCandidate]  # sorted by visits (or prior if no MCTS)
    human_candidate: Optional[MoveCandidate]
    human_rank: Optional[int]            # rank of human move (1 = best), None if not in top-k
    value_before: float
    value_after: Optional[float]         # None if terminal after move
    ai_recommended: tuple[int, int]
    threat_overridden: bool              # MCTS short-circuited by forced move
    search_stats: dict                   # {"num_simulations": N, "nodes_visited": V}

    def to_dict(self) -> dict:
        """JSON-serializable dict for future web API use."""

    @classmethod
    def from_dict(cls, data: dict) -> MoveComparison:
        """Reconstruct from dict."""
```

### Comparison Pipeline

1. **Legal move check.** If `human_move` not in `board.get_legal_moves()`,
   set `legal=False` and return early (AI recommendation still works).

2. **Value before.** `wrapper.evaluate(board)` -> `value_before`.

3. **MCTS path:**
   - `mcts.search_with_stats(board)` -> `SearchResult`.
   - Sort moves by `visit_count` descending.
   - Build `MoveCandidate` list.
   - Find human move rank.

4. **Fast path:**
   - `wrapper.evaluate(board)` -> `(move_probs, value)`.
   - Sort by prior descending.
   - Build `MoveCandidate` list (visit_count=0, q_value=0.0).
   - Find human move rank.

5. **Value after.** Copy board, `board_copy.make_move(*human_move)`.
   `wrapper.evaluate(board_copy)` -> `value_after`.

6. **Threat detection.** Check if MCTS returned a forced-move result
   (exactly 1/n distribution for winning moves, or block moves).

7. **Assemble MoveComparison** and return.

### Edge Cases

- **Illegal move:** Set `legal=False`. Show AI top-k but human_candidate=None.
- **Terminal board:** `compare_move` returns immediately with value_before=1.0 or -1.0.
- **Forced win:** Threat_overridden=True. The top candidate is the winning move.
- **Full board draw:** Value_before ≈ 0.0, no legal moves.

---

## Workstream Compatibility Contract

All three workstreams must conform to:

| Contract | A: Saliency | B: Activations | C: Comparison |
|----------|-------------|----------------|---------------|
| Uses `evaluate_raw()` | Yes | Yes | No (uses `evaluate()`) |
| Uses `search_with_stats()` | No | No | Yes |
| Creates files in `explain/` | saliency.py | activations.py | comparison.py |
| Creates files in `tests/` | test_saliency.py | test_activations.py | test_comparison.py |
| Modifies existing files | None | None | mcts.py (add SearchResult + search_with_stats) |
| Imports from `explain/` | None | None | None |
| Can test without trained weights | Completeness axiom | Shape/hooks/pipeline | Pipeline structure |
| Produces NDArray[np.float32] (15,15) | Yes | Yes (per channel) | No |
| Produces dataclass result | SaliencyMap | ActivationSnapshot | MoveComparison |
| JSON-serializable | SaliencyMap.grid.tolist() | No (requires rendering) | MoveComparison.to_dict() |

---

## Integration Path to Phase 5 (UI)

When a web frontend is built in Phase 5, the integration is:

1. **Policy heatmap overlay** -> reads `SaliencyMap.grid`, renders as
   semi-transparent color overlay on the 15×15 board canvas.

2. **Activation viewer** -> reads `ActivationSnapshot.activations`,
   renders selected channels as tiled 15×15 grids or grayscale heatmaps.

3. **Search tree visualization** -> currently not part of explainability
   (it's a Phase 5 feature). Workstream C's `MoveComparison` provides the
   structured data for a "move analysis panel" — top-k candidates with stats.

4. **Value landscape** -> not part of this phase. Can use
   `wrapper.evaluate()` per cell on a board to build a value grid, or use
   saliency as a proxy.

---

## Testing Guidelines

All tests should:
- Use a `GomokuNet` with default params (10 blocks, 128 channels) to match
  the real model.
- Use a randomly initialized model (no checkpoint needed).
- Run on CPU (GPU not required for tests).
- Be deterministic (fixed seed for reproducibility).

### Workstream-specific invariants

| Invariant | A | B | C |
|-----------|---|---|---|
| Output shapes correct | ✅ | ✅ | ✅ |
| Values in expected range | ✅ | ✅ | ✅ |
| No NaN / Inf | ✅ | ✅ | ✅ |
| Completeness axiom | ✅ | | |
| Hook cleanup verified | | ✅ | |
| Human move rank ∈ [1, top_k] | | | ✅ |
| Legal/illegal handling | | | ✅ |
| JSON roundtrip | | | ✅ |
| Context manager cleanup | | ✅ | |
| Empty board no crash | ✅ | ✅ | ✅ |
| Terminal board no crash | ✅ | | ✅ |
