"""Dual-headed residual CNN for Gomoku position evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention gate.

    Computes a 2D attention map by pooling across channels (avg + max),
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
        avg = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        mx = x.max(dim=1, keepdim=True)[0]      # (B, 1, H, W)
        pooled = torch.cat([avg, mx], dim=1)    # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(pooled)) # (B, 1, H, W)
        return x * attn


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


class SEResidualBlock(nn.Module):
    """Residual block with optional SE, spatial attention, and self-attention.

    Conv path:      Conv3×3 → BN → ReLU → Conv3×3 → BN
    Attention path: self-attention over spatial grid on input (optional)
    SE:             channel gating after conv+attention merge (optional)
    Spatial:        CBAM spatial attention after SE (optional)
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
        se_reduction: int = 16,
        num_attention_heads: int = 1,
        dilation: int = 1,
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

        out = out + residual
        return F.relu(out)


class PreActSEResidualBlock(nn.Module):
    """Pre-activation (ResNet-v2) residual block with optional SE, spatial,
    and self-attention.

    BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 → (+ Attn) → (SE) → (Spatial) → + skip

    Pre-activation places BN → ReLU before convolutions rather than after,
    improving gradient flow for deeper networks (He et al., 2016).
    """

    def __init__(
        self,
        channels: int,
        use_se: bool = True,
        use_attention: bool = True,
        use_spatial: bool = True,
        se_reduction: int = 16,
        num_attention_heads: int = 1,
        dilation: int = 1,
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

        return out + residual  # no final ReLU — absorbed by next block's BN → ReLU


class ResidualBlock(nn.Module):
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


class GomokuNet(nn.Module):
    """Dual-headed CNN for 15×15 Gomoku.

    Policy head  → log-softmax over 225 cells.
    Value head   → tanh  scalar in [-1, 1].

    Architecture highlights:

    * **Multi-head self-attention** (default 2 heads) complements local
      convolutions with global pairwise position interactions, helping
      detect disjoint threats across the board.
    * **Dilated convolutions** in later residual blocks expand the
      receptive field without extra parameters, critical for spotting
      five-in-a-row patterns that span 5+ cells.
    * **CBAM spatial attention** after SE channel gating gives each
      block a lightweight "where" signal alongside the "what" signal.
    * **Deeper policy head** with a 3×3 convolution stage before the
      1×1 projection gives the policy sufficient capacity to transform
      shared features into move-specific evidence.
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
        value_global_pool: bool = True,
        se_reduction: int = 16,
        num_attention_heads: int = 2,
        use_spatial: bool = True,
        dilations: list[int] | None = None,
        policy_hidden_channels: int = 32,
    ):
        super().__init__()
        self.board_size = board_size
        action_space = board_size * board_size  # 225

        # --- dilation schedule ---
        if dilations is None:
            # First ~70 % of blocks at dilation 1 (local pattern
            # refinement), remaining blocks at dilation 2 (long-range
            # line detection).  For the default 10 blocks this gives
            # [1,1,1,1,1,1,1, 2,2,2].
            n_late = max(0, num_res_blocks - 7)
            dilations = [1] * (num_res_blocks - n_late) + [2] * n_late

        if len(dilations) != num_res_blocks:
            raise ValueError(
                f"dilations length ({len(dilations)}) must match "
                f"num_res_blocks ({num_res_blocks})"
            )

        block_cls = (
            PreActSEResidualBlock if use_pre_activation else SEResidualBlock
        )

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
                )
                for i in range(num_res_blocks)
            ]
        )

        # --- Policy head (deeper) ---
        # 3×3 conv → BN → ReLU  extracts local move-relevant features
        # 1×1 conv → BN → ReLU  compresses to 2 channels for the FC layer
        self.policy_conv1 = nn.Conv2d(
            num_hidden_channels, policy_hidden_channels, kernel_size=3,
            padding=1, bias=False,
        )
        self.policy_bn1 = nn.BatchNorm2d(policy_hidden_channels)
        self.policy_conv2 = nn.Conv2d(
            policy_hidden_channels, 2, kernel_size=1, bias=False,
        )
        self.policy_bn2 = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_space)

        # --- Value head ---
        self.value_global_pool = value_global_pool
        self.value_conv = nn.Conv2d(num_hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_hidden_channels)
        if value_global_pool:
            self.value_avg_fc = nn.Linear(num_hidden_channels, num_hidden_channels)
            self.value_max_fc = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.value_fc2 = nn.Linear(num_hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy: 3×3 → BN → ReLU → 1×1 → BN → ReLU → FC → log-softmax
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = F.relu(self.policy_bn2(self.policy_conv2(p)))
        p = p.view(p.size(0), -1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        # Value: tanh → [-1, 1]
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))

        if self.value_global_pool:
            # Global context branches operating on the full feature maps.
            avg_pool = x.mean(dim=[2, 3])        # (B, C)
            max_pool = x.max(dim=3)[0].max(dim=2)[0]  # (B, C)
            global_feat = F.relu(self.value_avg_fc(avg_pool)) + \
                          F.relu(self.value_max_fc(max_pool))
            v = v + global_feat

        value = torch.tanh(self.value_fc2(v))

        return log_policy, value
