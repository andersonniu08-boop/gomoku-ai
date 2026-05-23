# Stronger Neural Network Architectures — Design

Date: 2026-05-22
Status: approved

## Goal

Upgrade `GomokuNet` from a basic 5-block / 64-channel residual CNN to a
configurable architecture with Squeeze-and-Excitation, attention-augmented
convolutions, and deeper/wider defaults.

## New Defaults

| Parameter            | Old   | New   |
|----------------------|-------|-------|
| `num_res_blocks`     | 5     | 10    |
| `num_hidden_channels`| 64    | 128   |
| `use_se`             | —     | True  |
| `use_attention`      | —     | True  |

## New Building Blocks

### SELayer

Standard channel-attention block. Global avg pool → FC(C → C/r) → ReLU →
FC(C/r → C) → Sigmoid. Multiplicative gating on channel dim. Reduction ratio
defaults to 16, configurable via `se_reduction`.

### AttentionAugmentedConv

Lightweight self-attention over the 15×15 spatial grid. Operates in parallel
with the conv path inside each residual block:
- Reshape input (B, C, H, W) → (B, H*W, C)
- Multi-head self-attention (default 1 head) with learned QKV projections
- Output projection maps back to C channels
- Reshape to (B, C, H, W)

This is attention-augmented convolution (option A), not a separate transformer
trunk — attention lives inside each residual block, complementing local conv
features with global pairwise relationships.

### SEResidualBlock (replaces ResidualBlock)

```
Input x
  ├── Conv3x3 → BN → ReLU → Conv3x3 → BN ─┐
  ├── AttentionAugmentedConv(x) ───────────┤ (if use_attention)
  │                                        ↓
  │                              element-wise sum
  │                                        ↓
  │                              SELayer (if use_se)
  │                                        ↓
  └────────── skip connection ──────────── (+) → ReLU → output
```

When `use_attention=False`, the attention branch is skipped.
When `use_se=False`, the SE gating is skipped.
Both off → equivalent to original `ResidualBlock`.

## Constructor Contract

```python
GomokuNet(
    board_size=15,
    in_channels=3,
    num_res_blocks=10,        # was 5
    num_hidden_channels=128,  # was 64
    use_se=True,              # new
    use_attention=True,       # new
    se_reduction=16,          # new
    num_attention_heads=1,    # new
)
```

Policy head and value head architectures are unchanged.

## Backward Compatibility

- `ResidualBlock` class retained (used in tests, zero maintenance cost).
- Existing checkpoints won't load on new architecture (parameter shape
  mismatch) — expected and acceptable for an architecture upgrade.
- `GomokuInferenceWrapper` gains `use_se` and `use_attention` kwargs
  (default `True`) passed through to `GomokuNet`.

## Test Updates

- Existing tests for `ResidualBlock` and `GomokuNet` (with explicit
  `num_res_blocks=5`) continue to pass.
- `_make_wrapper()` helper uses `num_res_blocks=5` — still valid.
- New tests: shape preservation for `SEResidualBlock`, output shapes with
  SE + attention enabled, full forward pass with new defaults.

## Non-Goals

- Full transformer trunk (separate from conv backbone)
- Changing policy/value head architectures
- Checkpoint format migration
- Batched inference changes (separate concern, Phase 2)
