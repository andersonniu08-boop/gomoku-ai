# Stronger Neural Network Architectures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade GomokuNet with SE blocks, attention-augmented convolutions, and deeper/wider defaults while preserving the existing input/output contract.

**Architecture:** Three new building blocks (`SELayer`, `AttentionAugmentedConv`, `SEResidualBlock`) added to `neural/model.py`. `GomokuNet` constructor gains `use_se`, `use_attention`, `se_reduction`, `num_attention_heads` kwargs. Defaults bump to 10 blocks / 128 channels. `GomokuInferenceWrapper` gains matching pass-through kwargs.

**Tech Stack:** PyTorch, existing codebase conventions (type hints, docstrings, slots dataclasses where applicable)

**Files:**
- Modify: `neural/model.py` — add SELayer, AttentionAugmentedConv, SEResidualBlock; update GomokuNet
- Modify: `neural/wrapper.py` — add use_se, use_attention kwargs
- Modify: `tests/test_neural.py` — new tests, updated helper

---

### Task 1: Add SELayer to neural/model.py

**Files:**
- Modify: `neural/model.py` — insert SELayer class after imports, before ResidualBlock

- [ ] **Step 1: Add SELayer class**

Insert after the docstring and imports (line 8), before `ResidualBlock`:

```python
class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Global avg pool → FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid.
    Multiplicative gating on the channel dimension.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: global average pool
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        return x * y.view(b, c, 1, 1)
```

- [ ] **Step 2: Verify the file still imports cleanly**

Run: `python -c "from neural.model import SELayer; print('OK')"`
Expected: `OK`

---

### Task 2: Add AttentionAugmentedConv to neural/model.py

**Files:**
- Modify: `neural/model.py` — insert after SELayer, before ResidualBlock

- [ ] **Step 1: Add AttentionAugmentedConv class**

Insert after `SELayer`, before `ResidualBlock`:

```python
class AttentionAugmentedConv(nn.Module):
    """Lightweight multi-head self-attention over the spatial grid.

    Runs in parallel with the conv branch inside a residual block,
    providing global pairwise position interactions to complement
    local convolution features.
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # (B, C, H, W) → (B, H*W, C)
        x_flat = x.view(b, c, h * w).transpose(1, 2)

        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H*W, C) → (B, H*W, heads, head_dim) → (B, heads, H*W, head_dim)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, heads, H*W, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(b, c, h, w)

        return out
```

- [ ] **Step 2: Verify import**

Run: `python -c "from neural.model import AttentionAugmentedConv; print('OK')"`
Expected: `OK`

---

### Task 3: Add SEResidualBlock to neural/model.py

**Files:**
- Modify: `neural/model.py` — insert after AttentionAugmentedConv, before ResidualBlock

- [ ] **Step 1: Add SEResidualBlock class**

Insert after `AttentionAugmentedConv`, before `ResidualBlock`:

```python
class SEResidualBlock(nn.Module):
    """Residual block with optional SE channel attention and attention-augmented conv.

    Conv path: Conv3×3 → BN → ReLU → Conv3×3 → BN
    Attention path (optional): self-attention over spatial grid on input
    SE (optional): channel gating after conv+attention merge
    Skip: input added to merged result, then ReLU
    """

    def __init__(
        self,
        channels: int,
        use_se: bool = True,
        use_attention: bool = True,
        se_reduction: int = 16,
        num_attention_heads: int = 1,
    ):
        super().__init__()
        self.use_se = use_se
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if use_se:
            self.se = SELayer(channels, reduction=se_reduction)
        if use_attention:
            self.attn = AttentionAugmentedConv(channels, num_heads=num_attention_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_attention:
            out = out + self.attn(x)

        if self.use_se:
            out = self.se(out)

        out = out + residual
        return F.relu(out)
```

- [ ] **Step 2: Verify import**

Run: `python -c "from neural.model import SEResidualBlock; print('OK')"`
Expected: `OK`

---

### Task 4: Update GomokuNet to use SEResidualBlock with new defaults

**Files:**
- Modify: `neural/model.py` — update GomokuNet.__init__ signature and body

- [ ] **Step 1: Update GomokuNet constructor**

Replace the existing `GomokuNet.__init__` (lines 33-61) with:

```python
class GomokuNet(nn.Module):
    """Dual-headed CNN for 15×15 Gomoku.

    Policy head  → log-softmax over 225 cells.
    Value head   → tanh  scalar in [-1, 1].

    Supports configurable SE channel attention, attention-augmented
    convolutions, and variable depth/width.
    """

    def __init__(
        self,
        board_size: int = 15,
        in_channels: int = 3,
        num_res_blocks: int = 10,
        num_hidden_channels: int = 128,
        use_se: bool = True,
        use_attention: bool = True,
        se_reduction: int = 16,
        num_attention_heads: int = 1,
    ):
        super().__init__()
        self.board_size = board_size
        action_space = board_size * board_size  # 225

        self.conv_init = nn.Conv2d(
            in_channels, num_hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_init = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.ModuleList(
            [
                SEResidualBlock(
                    num_hidden_channels,
                    use_se=use_se,
                    use_attention=use_attention,
                    se_reduction=se_reduction,
                    num_attention_heads=num_attention_heads,
                )
                for _ in range(num_res_blocks)
            ]
        )

        # --- Policy head ---
        self.policy_conv = nn.Conv2d(num_hidden_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_space)

        # --- Value head ---
        self.value_conv = nn.Conv2d(num_hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_hidden_channels)
        self.value_fc2 = nn.Linear(num_hidden_channels, 1)
```

The `forward` method is unchanged.

- [ ] **Step 2: Quick smoke test — instantiate with new defaults**

Run: `python -c "
from neural.model import GomokuNet
import torch
m = GomokuNet()
x = torch.randn(2, 3, 15, 15)
p, v = m(x)
print(f'Policy shape: {p.shape}, Value shape: {v.shape}')
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
"`
Expected: Policy shape: torch.Size([2, 225]), Value shape: torch.Size([2, 1]), Params > 500k

- [ ] **Step 3: Verify old defaults still work (backward compat)**

Run: `python -c "
from neural.model import GomokuNet
import torch
m = GomokuNet(num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
x = torch.randn(2, 3, 15, 15)
p, v = m(x)
print(f'Policy: {p.shape}, Value: {v.shape}')
"`
Expected: Policy: torch.Size([2, 225]), Value: torch.Size([2, 1])

---

### Task 5: Update GomokuInferenceWrapper with new kwargs

**Files:**
- Modify: `neural/wrapper.py` — add use_se, use_attention kwargs to __init__

- [ ] **Step 1: Update wrapper constructor signature and model instantiation**

In `neural/wrapper.py`, update `GomokuInferenceWrapper.__init__` (lines 21-48):

```python
def __init__(
    self,
    checkpoint_path: str | Path,
    device: Optional[str] = None,
    num_res_blocks: int = 10,
    num_hidden_channels: int = 128,
    use_se: bool = True,
    use_attention: bool = True,
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    self.device = torch.device(device)

    self.model = GomokuNet(
        board_size=15,
        in_channels=3,
        num_res_blocks=num_res_blocks,
        num_hidden_channels=num_hidden_channels,
        use_se=use_se,
        use_attention=use_attention,
    ).to(self.device)

    checkpoint = torch.load(
        str(checkpoint_path), map_location=self.device, weights_only=True
    )
    self.model.load_state_dict(checkpoint)
    self.model.eval()
```

The rest of the file is unchanged.

- [ ] **Step 2: Verify wrapper import**

Run: `python -c "from neural.wrapper import GomokuInferenceWrapper; print('OK')"`
Expected: `OK`

---

### Task 6: Add new tests for SELayer, AttentionAugmentedConv, and SEResidualBlock

**Files:**
- Modify: `tests/test_neural.py` — add new test functions

- [ ] **Step 1: Add imports for new classes**

Update the import line (line 9) in `tests/test_neural.py`:

```python
from neural.model import (
    AttentionAugmentedConv,
    GomokuNet,
    ResidualBlock,
    SELayer,
    SEResidualBlock,
)
```

- [ ] **Step 2: Add new test functions**

Append after `test_residual_block_preserves_shape` (after line 23):

```python
def test_se_layer_preserves_shape():
    se = SELayer(64, reduction=16)
    x = torch.randn(2, 64, 15, 15)
    out = se(x)
    assert out.shape == x.shape


def test_se_layer_modulates_channels():
    """SE layer should not be a no-op — output should differ from input."""
    se = SELayer(64, reduction=16)
    x = torch.randn(2, 64, 15, 15)
    out = se(x)
    assert not torch.allclose(out, x)


def test_attention_augmented_conv_preserves_shape():
    attn = AttentionAugmentedConv(64, num_heads=1)
    x = torch.randn(2, 64, 15, 15)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_augmented_conv_multi_head():
    attn = AttentionAugmentedConv(64, num_heads=2)
    x = torch.randn(2, 64, 15, 15)
    out = attn(x)
    assert out.shape == x.shape


def test_se_residual_block_preserves_shape():
    block = SEResidualBlock(64)
    x = torch.randn(2, 64, 15, 15)
    out = block(x)
    assert out.shape == x.shape


def test_se_residual_block_no_se_no_attn():
    """With both features off, behaves like plain residual block."""
    block = SEResidualBlock(64, use_se=False, use_attention=False)
    x = torch.randn(2, 64, 15, 15)
    out = block(x)
    assert out.shape == x.shape


def test_se_residual_block_variants():
    """All four combinations of use_se/use_attention should work."""
    for use_se in (True, False):
        for use_attn in (True, False):
            block = SEResidualBlock(64, use_se=use_se, use_attention=use_attn)
            x = torch.randn(1, 64, 15, 15)
            out = block(x)
            assert out.shape == x.shape


def test_model_with_se_and_attention():
    """New defaults: 10 blocks, 128 channels, SE + attention."""
    model = GomokuNet()
    x = torch.randn(4, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (4, 225)
    assert value.shape == (4, 1)
    # Policy is still log-softmax
    probs = torch.exp(log_policy)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    # Value still in [-1, 1]
    assert (-1.0 <= value).all() and (value <= 1.0).all()


def test_model_without_se_and_attention():
    """Old-style config: 5 blocks, 64 channels, no SE, no attention."""
    model = GomokuNet(
        num_res_blocks=5,
        num_hidden_channels=64,
        use_se=False,
        use_attention=False,
    )
    x = torch.randn(4, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (4, 225)
    assert value.shape == (4, 1)


def test_model_custom_depth_and_width():
    """Arbitrary block count and channel width should work."""
    model = GomokuNet(
        num_res_blocks=3,
        num_hidden_channels=32,
        use_se=False,
        use_attention=False,
    )
    x = torch.randn(1, 3, 15, 15)
    log_policy, value = model(x)
    assert log_policy.shape == (1, 225)
    assert value.shape == (1, 1)
```

- [ ] **Step 3: Run the new tests**

Run: `pytest tests/test_neural.py -v`
Expected: All existing + new tests pass

---

### Task 7: Update test helpers and existing tests for compatibility

**Files:**
- Modify: `tests/test_neural.py` — update `_make_wrapper` and `test_wrapper_save_load_and_evaluate`

- [ ] **Step 1: Update _make_wrapper to use old-style defaults**

The `_make_wrapper` helper (line 103-116) creates a `GomokuNet` with `num_res_blocks=5`. Since the wrapper defaults changed to 128 channels, we need to be explicit. Update:

```python
def _make_wrapper():
    """Create a wrapper around a fresh untrained model for testing."""
    model = GomokuNet(
        board_size=15,
        in_channels=3,
        num_res_blocks=5,
        num_hidden_channels=64,
        use_se=False,
        use_attention=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f)
        tmp_path = Path(f.name)

    try:
        wrapper = GomokuInferenceWrapper(
            tmp_path,
            device="cpu",
            num_res_blocks=5,
            num_hidden_channels=64,
            use_se=False,
            use_attention=False,
        )
        yield wrapper
    finally:
        tmp_path.unlink()
```

- [ ] **Step 2: Update test_wrapper_save_load_and_evaluate**

The test at line 53-74 creates a `GomokuNet` with `num_res_blocks=5` (old default implied). Make it explicit:

Replace the model creation line (line 54):
```python
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
```

And the wrapper creation (line 60-62):
```python
        wrapper = GomokuInferenceWrapper(
            tmp_path, device="cpu", num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False
        )
```

- [ ] **Step 3: Update test_wrapper_evaluate_with_threats_no_crash**

Replace model creation (line 78):
```python
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
```

And wrapper creation (line 84):
```python
        wrapper = GomokuInferenceWrapper(tmp_path, device="cpu", num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/test_neural.py -v`
Expected: All 17 tests pass

- [ ] **Step 5: Run all project tests**

Run: `pytest tests/ -v`
Expected: All tests pass across the project

---

### Task 8: Final verification — new defaults end-to-end

**Files:**
- None (verification only)

- [ ] **Step 1: Verify new defaults save/load cycle**

Run:
```bash
python -c "
import tempfile, torch
from pathlib import Path
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper

model = GomokuNet()  # new defaults
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    torch.save(model.state_dict(), f)
    tmp_path = Path(f.name)

try:
    wrapper = GomokuInferenceWrapper(tmp_path, device='cpu')
    print(f'Model loaded with {sum(p.numel() for p in wrapper.model.parameters()):,} params')
    print('Save/load cycle OK')
finally:
    tmp_path.unlink()
"
```
Expected: Model loaded with ~1M+ params, Save/load cycle OK

- [ ] **Step 2: Verify new defaults inference works on a board**

Run:
```bash
python -c "
import tempfile, torch
from pathlib import Path
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from engine.board import Board

model = GomokuNet()
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    torch.save(model.state_dict(), f)
    tmp_path = Path(f.name)

try:
    wrapper = GomokuInferenceWrapper(tmp_path, device='cpu')
    board = Board()
    board.make_move(7, 7)
    board.make_move(8, 8)
    probs, value = wrapper.evaluate(board)
    total = sum(p for _, p in probs)
    print(f'Moves: {len(probs)}, Total prob: {total:.4f}, Value: {value:.4f}')
    assert abs(total - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
    print('Inference OK')
finally:
    tmp_path.unlink()
"
```
Expected: Moves > 0, Total prob: 1.0000, Value between -1 and 1, Inference OK
