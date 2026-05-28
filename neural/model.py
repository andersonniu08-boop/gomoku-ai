"""Dual-headed residual CNN for Gomoku position evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth (DropPath) regularisation.

    Randomly drops entire residual branches during training, forcing the
    network to build redundancy across blocks.  At inference all branches
    are active — the training-time stochasticity acts as a model ensemble.

    Drop probability is typically scheduled linearly from 0 in the first
    block to *drop_prob* in the last block (Huang et al., 2016).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binary mask: 0 (drop) or 1 (keep)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Global avg pool → FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid.
    Multiplicative gating on the channel dimension.

    The reduction ratio *r* is capped so the bottleneck never drops below
    8 channels even for modest channel counts (e.g. 128 → bottleneck 16
    with the default reduction of 8).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention gate.

    Computes a 2-D attention map by pooling across channels (avg + max),
    convolving the stacked (2, H, W) descriptor with a small kernel, and
    applying sigmoid.  The resulting (1, H, W) map is broadcast-multiplied
    with the input to suppress irrelevant spatial regions.

    Applied after SE channel gating and before the residual addition,
    giving the block a lightweight "where to attend" signal that
    complements the channel-wise "what to attend" from SE.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        pooled = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(pooled))
        return x * attn


class AttentionAugmentedConv(nn.Module):
    """Multi-head self-attention over the spatial grid.

    Runs in parallel with the conv branch inside a residual block,
    providing global pairwise position interactions to complement
    local convolution features.

    A pre-attention LayerNorm stabilises training and removes the
    dependency on the caller providing normalised inputs.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # (B, C, H, W) → (B, H*W, C)
        x_flat = x.view(b, c, h * w).transpose(1, 2)

        x_norm = self.norm(x_flat)
        qkv = self.qkv(x_norm)
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


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class SEResidualBlock(nn.Module):
    """Residual block with optional SE, spatial attention, and self-attention.

    Conv path:      Conv3×3 → BN → ReLU → Conv3×3 → BN
    Attention path: self-attention over spatial grid on input (optional)
    SE:             channel gating after conv+attention merge (optional)
    Spatial:        CBAM spatial attention after SE (optional)
    DropPath:       stochastic depth on the residual branch (optional)
    Skip:           input added to merged result, then ReLU

    Dilation support expands the receptive field in later blocks
    without increasing parameter count — critical for detecting
    long-range five-in-a-row patterns on a 15×15 board.
    """

    def __init__(
        self,
        channels: int,
        use_se: bool = True,
        use_attention: bool = True,
        use_spatial: bool = True,
        se_reduction: int = 8,
        num_attention_heads: int = 4,
        dilation: int = 1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_se = use_se
        self.use_attention = use_attention
        self.use_spatial = use_spatial

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        if use_se:
            self.se = SELayer(channels, reduction=se_reduction)
        if use_attention:
            self.attn = AttentionAugmentedConv(channels, num_heads=num_attention_heads)
        if use_spatial:
            self.spatial = SpatialAttention(kernel_size=3)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_attention:
            out = out + self.attn(x)

        if self.use_se:
            out = self.se(out)

        if self.use_spatial:
            out = self.spatial(out)

        out = self.drop_path(out)
        out = out + residual
        return F.relu(out)


class PreActSEResidualBlock(nn.Module):
    """Pre-activation (ResNet-v2) residual block with optional SE, spatial,
    and self-attention.

    BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 → (+ Attn on preact) →
    (SE) → (Spatial) → DropPath → + skip

    Pre-activation places BN → ReLU before convolutions rather than after,
    improving gradient flow for deeper networks (He et al., 2016).
    """

    def __init__(
        self,
        channels: int,
        use_se: bool = True,
        use_attention: bool = True,
        use_spatial: bool = True,
        se_reduction: int = 8,
        num_attention_heads: int = 4,
        dilation: int = 1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_se = use_se
        self.use_attention = use_attention
        self.use_spatial = use_spatial

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )

        if use_se:
            self.se = SELayer(channels, reduction=se_reduction)
        if use_attention:
            self.attn = AttentionAugmentedConv(channels, num_heads=num_attention_heads)
        if use_spatial:
            self.spatial = SpatialAttention(kernel_size=3)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        preact = F.relu(self.bn1(x))
        out = self.conv1(preact)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        if self.use_attention:
            out = out + self.attn(preact)

        if self.use_se:
            out = self.se(out)

        if self.use_spatial:
            out = self.spatial(out)

        out = self.drop_path(out)
        return out + residual  # no final ReLU — absorbed by next block's BN → ReLU


class ResidualBlock(nn.Module):
    """Plain residual block (no attention, no SE) — kept for compatibility
    with lightweight test fixtures."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dilation_schedule(num_blocks: int) -> list[int]:
    """Multi-scale pyramid dilation schedule.

    Ramps from local (dilation 1) through medium (2) to long-range (3)
    and back down, giving the network access to features at three
    spatial scales rather than just two.

    For the default 10 blocks: ``[1, 1, 1, 2, 2, 3, 3, 2, 2, 1]``.
    """
    if num_blocks <= 3:
        return [1] * num_blocks

    n_local1 = max(1, round(num_blocks * 0.30))
    n_dil2_up = max(1, round(num_blocks * 0.20))
    n_dil3 = max(1, round(num_blocks * 0.20))
    n_dil2_down = max(1, round(num_blocks * 0.15))
    n_local2 = num_blocks - n_local1 - n_dil2_up - n_dil3 - n_dil2_down

    if n_local2 < 1:
        # Not enough blocks for full pyramid → two-phase fallback.
        n_dil2 = max(1, round(num_blocks * 0.3))
        return [1] * (num_blocks - n_dil2) + [2] * n_dil2

    return [1] * n_local1 + [2] * n_dil2_up + [3] * n_dil3 + [2] * n_dil2_down + [1] * n_local2


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------


class GomokuNet(nn.Module):
    """Dual-headed CNN for 15×15 Gomoku.

    Policy head  → log-softmax over 225 cells.
    Value head   → tanh  scalar in [-1, 1].

    Architecture highlights (v2):

    * **Multi-head self-attention** (default 4 heads) complements local
      convolutions with global pairwise position interactions, helping
      detect disjoint threats across the board.  A pre-attention
      LayerNorm stabilises the QKV projections.
    * **Multi-scale dilated convolutions** follow a pyramid schedule
      (1 → 2 → 3 → 2 → 1), giving the trunk access to local, medium,
      and long-range features at different depths.
    * **CBAM spatial attention** after SE channel gating gives each
      block a lightweight "where" signal alongside the "what" signal.
    * **Fully convolutional policy head** projects trunk features to a
      single-channel logit map via 3×3 → 1×1 convolutions, preserving
      spatial structure without a costly FC layer.
    * **Dual-pooling value head** concatenates global average and max
      poolings over the full 128-channel feature map, giving the value
      head access to rich spatial statistics without a 1×1 bottleneck.
    * **Stochastic depth (DropPath)** regularises the residual trunk
      during training, acting as an implicit model ensemble.
    """

    def __init__(
        self,
        board_size: int = 15,
        in_channels: int = 3,
        num_res_blocks: int = 10,
        num_hidden_channels: int = 128,
        use_se: bool = True,
        use_attention: bool = True,
        use_pre_activation: bool = False,
        se_reduction: int = 8,
        num_attention_heads: int = 4,
        use_spatial: bool = True,
        dilations: list[int] | None = None,
        policy_hidden_channels: int = 32,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.board_size = board_size
        action_space = board_size * board_size  # 225

        # --- dilation schedule ---
        if dilations is None:
            dilations = _make_dilation_schedule(num_res_blocks)

        if len(dilations) != num_res_blocks:
            raise ValueError(
                f"dilations length ({len(dilations)}) must match "
                f"num_res_blocks ({num_res_blocks})"
            )

        block_cls = (
            PreActSEResidualBlock if use_pre_activation else SEResidualBlock
        )

        # Linear DropPath schedule: 0 in the first block, *drop_path_rate*
        # in the last.  Early blocks learn reliable low-level features;
        # later blocks are regularised more aggressively.
        drop_path_rates = [
            drop_path_rate * i / max(num_res_blocks - 1, 1)
            for i in range(num_res_blocks)
        ]

        self.conv_init = nn.Conv2d(
            in_channels, num_hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_init = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.ModuleList(
            [
                block_cls(
                    num_hidden_channels,
                    use_se=use_se,
                    use_attention=use_attention,
                    use_spatial=use_spatial,
                    se_reduction=se_reduction,
                    num_attention_heads=num_attention_heads,
                    dilation=dilations[i],
                    drop_path=drop_path_rates[i],
                )
                for i in range(num_res_blocks)
            ]
        )

        # --- Policy head (fully convolutional) ---
        # 3×3 conv → BN → ReLU extracts local move-relevant features from
        # the trunk output.  A 1×1 conv then projects to a single-channel
        # logit map, preserving spatial correspondence — each of the 225
        # output logits depends on the corresponding trunk feature column.
        self.policy_conv1 = nn.Conv2d(
            num_hidden_channels, policy_hidden_channels, kernel_size=3,
            padding=1, bias=False,
        )
        self.policy_bn1 = nn.BatchNorm2d(policy_hidden_channels)
        self.policy_conv2 = nn.Conv2d(
            policy_hidden_channels, 1, kernel_size=1, bias=True,
        )

        # --- Value head (dual global pooling) ---
        # Concatenates global average and max poolings over the full
        # 128-channel feature map, avoiding the information bottleneck
        # of a 1×1 conv-to-1-channel projection.  Two FC layers map the
        # 2C pooled descriptor to a scalar.
        self.value_fc1 = nn.Linear(num_hidden_channels * 2, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy: 3×3 → BN → ReLU → 1×1 → flatten → log-softmax
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = self.policy_conv2(p)          # (B, 1, 15, 15)
        p = p.view(p.size(0), -1)         # (B, 225)
        log_policy = F.log_softmax(p, dim=1)

        # Value: dual global pooling → FC → ReLU → FC → tanh
        avg_pool = x.mean(dim=[2, 3])                     # (B, C)
        max_pool = x.max(dim=3)[0].max(dim=2)[0]          # (B, C)
        v = torch.cat([avg_pool, max_pool], dim=1)        # (B, 2C)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_policy, value
