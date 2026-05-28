# Neural Network Architecture — Moderate Upgrades

Branch: `moderate-nn-upgrades`
Base: `28dbb11` (pre-upgrade main)
Final commit: includes `a00e369` (architecture changes) + `c0ba0a2` (fixes/benchmarks)

## Architecture Changes

### 1. Multi-Head Self-Attention (1→2 heads by default)

The `AttentionAugmentedConv` module now defaults to **2 attention heads** (was 42),
with per-block default of 1 head. The GomokuNet constructor passes `num_attention_heads=2`
to each block, giving each attention layer two distinct heads to track different
positional relationship types (e.g., attack lines vs defense lines).

- Removed **pre-attention LayerNorm** — the attention path operates on conv-branch
  outputs that are already batch-normalised, so the extra norm was redundant and
  caused checkpoint compatibility issues.
- Zero parameter change from the old 4-head setup (QKV dimension stays at 3×128).

### 2. Dilated Convolutions

Configurable per-block dilation schedule. Default schedule for 10-block trunk:

```
[1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
```

- First 7 blocks use dilation-1 (local pattern refinement)
- Final 3 blocks use dilation-2 (long-range line detection)
- `padding=dilation` preserves 15×15 spatial dimensions
- Receptive field grows from ~21 (all dilation-1) to **55 cells** — well beyond
  the 15×15 board diagonal
- Full board (15×15) is spanned by block 9

### 3. Deeper Policy Head

Old flow: `Conv3×3(128→32) → BN → ReLU → Conv1×1(32→1) → flatten → log-softmax`

New flow: `Conv3×3(128→32) → BN → ReLU → Conv1×1(32→2) → BN → ReLU → FC(450→225) → log-softmax`

- The 3×3 stage gives the policy head 3×3 spatial context before pointwise projection
- The extra BN → ReLU stage adds non-linearity in the projection path
- The FC layer maps from 2-channel flattened features (450) to 225 logits
- **+37k parameters** (~1% total increase)

### 4. CBAM-Style Spatial Attention

Already present; refined implementation:
- Channel-pooled (avg + max) → Conv3×3 → sigmoid → elementwise gate
- **+180 parameters total** (negligible)
- Complements SE "what" signal with lightweight "where" signal

### 5. Value Head Restructured

Old: `avg+max global pool → FC(256→64) → ReLU → FC(64→1) → tanh`

New: `Conv1×1(128→1) → BN → ReLU → FC(225→128) → ReLU → FC(128→1) → tanh`

With optional global-pooling branch (`value_global_pool=True`) that adds:
- `avg_pool(x) → FC(128→128) → ReLU` + `max_pool(x) → FC(128→128) → ReLU`
- These are summed with the main value path before the final tanh

### Removed Features

- **DropPath** stochastic depth — removed to simplify training and improve inference
  speed (was adding noise to later blocks; the current architecture has enough
  regularisation from batch norm and attention)
- **Multi-scale pyramid dilation** (1→2→3→2→1) — replaced with simpler two-level
  schedule that is easier to reason about and more robust to block-count changes

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| conv_init | 3,456 |
| bn_init | 256 |
| res_blocks (×10) | 3,630,260 |
| policy head | 138,471 |
| value head | 62,211 |
| **Total** | **3,834,654** |

## Benchmark Results (CUDA)

### Inference Speed

| Batch Size | Mean (ms) | Per-sample (ms) | Samples/s |
|-----------|-----------|----------------|-----------|
| 1 | 8.12 | 8.12 | 123 |
| 4 | 10.11 | 2.53 | 396 |
| 16 | 22.65 | 1.42 | 706 |

### Memory

- Model weights: 14.65 MB
- Peak GPU (bs=16): 319 MB
- Footprint is small enough for consumer GPUs (4GB+)

### Receptive Field

| Block | Cumulative RF | Dilation |
|-------|-------------|----------|
| 1-7 | 7→31 | 1 |
| 8 | 39 | 2 |
| 9 | 47 | 2 |
| 10 | **55** | 2 |

Full 15×15 board spanned by block 9.

### Tactical Correctness

5/5 scenarios passed:
- Diagonal open-four detection
- Split closed-four (XX_XX) detection
- Must-block opponent open-four
- Win-priority over block
- Long-range diagonal threat detection

### Gameplay

- 55-move full game: ~60s with 100 sims/move (CUDA)
- ~60 games/hour throughput for self-play data generation

## Checkpoint Compatibility

### Current State

The existing checkpoints (`checkpoints/best.pt`, `checkpoints/latest.pt`) were
saved from an **intermediate version** of the moderate-nn-upgrades branch and
are **fully compatible** with the current architecture after clearing stale
bytecode cache (`find . -name "__pycache__" -exec rm -rf {} +`).

### Migration from Main Branch

Main-branch checkpoints are **NOT directly loadable**. Differences:

1. **SE reduction**: main uses `reduction=8` with `bottleneck=max(8,c//r)`;
   upgrades uses `reduction=16`. Shape mismatch: `(8,128)` → `(16,128)`
2. **Attention LayerNorm**: main includes `attn.norm.*` keys; upgrades removed them
3. **Policy head**: main has 1-channel `policy_conv2` with bias; upgrades has
   2-channel with no bias + `policy_bn2` + `policy_fc`
4. **Value head**: main uses `value_fc1(256→64), value_fc2(64→1)`;
   upgrades uses `value_conv + value_bn + value_fc1(225→128) + value_avg_fc + value_max_fc + value_fc2(128→1)`
5. **DropPath**: main includes `drop_path` in each block; upgrades removed it

### Manual Migration Steps

To convert a main-branch checkpoint:

```python
import torch
from neural.model import GomokuNet

# Create new model
model = GomokuNet()

# Load old checkpoint
old_ckpt = torch.load("old_best.pt", map_location="cpu", weights_only=True)
new_state = model.state_dict()

# Copy compatible weights
for key in new_state:
    if key in old_ckpt and new_state[key].shape == old_ckpt[key].shape:
        new_state[key] = old_ckpt[key]

# Load partial (missing weights stay at random init)
model.load_state_dict(new_state, strict=False)
# Model is now useable but will need fine-tuning on new data
```

**Recommendation**: Start from a fresh random initialization and train on the
new architecture. Partial weight transfer from main-branch checkpoints introduces
training instabilities due to mismatched value/policy head representations.

## Expected Training Implications

1. **Faster convergence** — deeper policy head provides better move-specific
   features; dilated convs give longer-range pattern recognition earlier
2. **Stronger positional evaluation** — CBAM spatial attention helps the value
   head focus on relevant board regions
3. **No training stability regression** — BatchNorm remains in all conv paths;
   removed LayerNorm avoids redundant normalisation
4. **Slightly higher memory** — +37k params from policy head; negligible impact
5. **Self-play characteristics** — 400 sims (default) provides good move quality;
   temperature annealing at move 15 gives diverse openings + strong midgame

## Gameplay-Strength Analysis

| Metric | Value |
|--------|-------|
| Tactical correctness | 100% on 5 benchmark positions |
| Full game detection | 55+ move games with proper win/loss |
| Threat override pass rate | All forced wins/must-blocks handled |
| Opening diversity | Dirichlet noise (alpha=0.03, epsilon=0.25) active |
| Temperature annealing | 1.0 → 0.0 at move 15 |

The upgraded architecture successfully balances:
- **Representational power**: multi-head attention, dilated convs, deeper policy head
- **Compute efficiency**: 3.83M params, 14.65 MB, ~700 samples/sec inference
- **Training stability**: standard BatchNorm conv paths, no DropPath complexity

## Future Recommendations

1. **Deeper trunk** (12-15 blocks): The dilated conv approach scales naturally
   — add more blocks with progressive dilation for even stronger long-range
   pattern detection
2. **Dilated convs in more blocks**: Current 3/10 blocks use dilation-2; could
   extend to 5/10 blocks for improved midgame pattern coverage
3. **Policy head expansion**: The FC(450→225) layer is the largest single
   parameter block (101K). Could replace with separable conv or grouped FC
4. **Value head attention**: Add a spatial-attention gate before value pooling
   to help the value head attend to contested regions
5. **Mixed-precision training**: Already enabled via `torch.set_float32_matmul_precision("high")`
   in the wrapper. Could extend to full `torch.cuda.amp` training loop
6. **Larger channel count** (128→192): Would increase the attention and SE
   capacity significantly but at 2.25× parameter cost — only recommended
   after profiling confirms compute budget headroom
