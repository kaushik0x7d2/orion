"""
FHE-compatible Transformer layers for Orion.

Implements polynomial alternatives to standard transformer operations,
enabling encrypted inference on transformer-based models.

Key components:
    PolySoftmax: x^p / sum(x^p) replaces exp-based softmax
    FHELayerNorm: Layer normalization (cleartext exact, FHE pre-computed)
    FHEMultiHeadAttention: Attention with polynomial softmax
    FHETransformerEncoderLayer: Complete Pre-LN encoder block

These layers can be used directly for training FHE-compatible transformers.
Standard PyTorch transformers need architectural changes (PolySoftmax) that
require finetuning — the converter handles LayerNorm automatically.

References:
    PowerSoftmax (IBM, ICLR 2025): arxiv.org/abs/2410.09457
    Zimerman et al. (2023): arxiv.org/abs/2311.08610
    THOR (ACM CCS 2025): eprint.iacr.org/2024/1881

Usage:
    from orion.nn.transformer import (
        FHETransformerEncoderLayer, PolySoftmax, FHELayerNorm)

    # Build an FHE-compatible transformer
    encoder = FHETransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, softmax_power=4)

    x = torch.randn(8, 16, 64)   # [batch, seq, d_model]
    out = encoder(x)              # same shape
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module, timer
from .linear import Linear


class PolySoftmax(Module):
    """
    Polynomial softmax: x^p / sum(x^p).

    Replaces standard exp-based softmax with a polynomial alternative
    that is FHE-compatible. Uses even powers so outputs are non-negative.

    PowerSoftmax (IBM, ICLR 2025) proves this works at 1.4B parameter
    scale with only 6% of total FHE latency from the softmax itself.

    Args:
        power: Even integer in {2, 4, 8, 16}. Higher = sharper attention.
        dim: Dimension to normalize along. Default: -1.
    """
    VALID_POWERS = {2, 4, 8, 16}

    def __init__(self, power=4, dim=-1):
        super().__init__()
        if power not in self.VALID_POWERS:
            raise ValueError(
                f"power must be in {self.VALID_POWERS}, got {power}")
        self.power = power
        self.dim = dim
        # Depth: log2(p) for x^p via repeated squaring
        # + ~10 levels for Goldschmidt inverse (1/sum)
        self.set_depth(int(math.log2(power)) + 10)

    def extra_repr(self):
        return f"power={self.power}, dim={self.dim}"

    @timer
    def forward(self, x):
        # x^p via repeated squaring (matches FHE operation pattern)
        result = x * x  # x^2
        p = 2
        while p < self.power:
            result = result * result
            p *= 2

        # Normalize: result / sum(result)
        denom = result.sum(dim=self.dim, keepdim=True).clamp(min=1e-12)
        return result / denom


class FHELayerNorm(Module):
    """
    FHE-compatible Layer Normalization.

    Cleartext mode: standard PyTorch LayerNorm (exact).
    FHE mode: uses pre-computed normalization statistics from the
    calibration (fit) step, reducing to an affine transform that
    avoids the expensive 1/sqrt(var) on encrypted data.

    This follows THE-X (2022) and is consistent with how Orion's
    BatchNorm uses frozen running statistics during FHE inference.

    Args:
        normalized_shape: Input shape from expected input.
        eps: Value for numerical stability.
        elementwise_affine: Whether to use learnable affine parameters.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.set_depth(2 if elementwise_affine else 1)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def extra_repr(self):
        base = super().extra_repr()
        return (f"normalized_shape={self.normalized_shape}, "
                f"eps={self.eps}, affine={self.elementwise_affine}, {base}")

    @timer
    def forward(self, x):
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)


class FHEMultiHeadAttention(Module):
    """
    FHE-compatible Multi-Head Self-Attention.

    Replaces standard softmax with PolySoftmax. Uses separate Q, K, V
    Linear projections (orion.nn.Linear, FHE-compatible).

    In cleartext: standard scaled dot-product attention with PolySoftmax.
    In FHE: Q/K/V projections are plaintext-ciphertext, Q*K^T and attn*V
    are ciphertext-ciphertext matmuls.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        softmax_power: Power for PolySoftmax (2, 4, 8, or 16).
        bias: Whether to add bias to projection layers.
    """
    def __init__(self, embed_dim, num_heads, softmax_power=4, bias=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Separate Q, K, V projections (FHE-compatible)
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.poly_softmax = PolySoftmax(power=softmax_power, dim=-1)

        # Depth: max(Q,K,V) + QK^T + PolySoftmax + attn*V + out_proj
        self.set_depth(4 + self.poly_softmax.depth)

    def extra_repr(self):
        base = super().extra_repr()
        return (f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim}, power={self.poly_softmax.power}, "
                f"{base}")

    @timer
    def forward(self, x, key=None, value=None):
        """
        Args:
            x: Query tensor [batch, seq_len, embed_dim].
            key: Key tensor (default: x for self-attention).
            value: Value tensor (default: x for self-attention).
        Returns:
            Output tensor [batch, seq_len, embed_dim].
        """
        if key is None:
            key = x
        if value is None:
            value = x

        B, S, E = x.shape
        S_kv = key.shape[1]

        # orion.nn.Linear requires 2D input, so reshape [B, S, E] -> [B*S, E]
        q = self.q_proj(x.reshape(B * S, E)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key.reshape(B * S_kv, E)).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value.reshape(B * S_kv, E)).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with polynomial softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.poly_softmax(scores)
        out = torch.matmul(attn, v)

        # Reshape back and project (2D for orion.nn.Linear)
        out = out.transpose(1, 2).contiguous().view(B * S, self.embed_dim)
        out = self.out_proj(out)
        return out.view(B, S, self.embed_dim)


class FHETransformerEncoderLayer(Module):
    """
    FHE-compatible Transformer Encoder Layer (Pre-LN architecture).

    Components:
    - PolySoftmax multi-head self-attention
    - Polynomial GELU activation (Chebyshev approximation)
    - FHELayerNorm (pre-computed statistics for FHE)
    - Residual connections (addition only, depth 0)

    Pre-LN is preferred over Post-LN for training stability and because
    normalizing before the sublayer constrains input ranges for FHE.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension. Default: 256.
        activation_degree: Chebyshev degree for GELU. Default: 7.
        softmax_power: Power for PolySoftmax. Default: 4.
    """
    def __init__(self, d_model, nhead, dim_feedforward=256,
                 activation_degree=7, softmax_power=4):
        super().__init__()

        self.self_attn = FHEMultiHeadAttention(
            d_model, nhead, softmax_power=softmax_power)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FHELayerNorm(d_model)
        self.norm2 = FHELayerNorm(d_model)

        from .activation import GELU as OrionGELU
        self.activation = OrionGELU(degree=activation_degree)

        # Total depth: norm1 + attention + norm2 + FFN
        act_depth = int(math.ceil(math.log2(activation_degree + 1))) + 1
        total = (self.norm1.depth + self.self_attn.depth +
                 self.norm2.depth + 1 + act_depth + 1)  # linear1 + act + linear2
        self.set_depth(total)

    def extra_repr(self):
        return (f"d_model={self.norm1.normalized_shape[0]}, "
                f"nhead={self.self_attn.num_heads}, "
                f"dim_ff={self.linear1.out_features}, "
                f"depth={self.depth}")

    @timer
    def forward(self, x):
        # Pre-LN: normalize before each sublayer
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        # FFN: reshape for orion.nn.Linear (requires 2D input)
        B, S, D = x.shape
        x = x.reshape(B * S, D)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = x.view(B, S, D) + residual

        return x
