"""
Binary linear layer with per-group learned scales.

The weight is stored as a normal FP16 tensor: weight = sign * scale_per_group.
This gives a standard matmul that MLX handles natively via optimized BLAS.

The binary constraint is structural:
  - Each weight is constrained to be +scale or -scale (sign * group_scale)
  - ES mutation = negate individual weight values (flip sign)
  - Serialization = pack signs to 1-bit + save scales separately

During training, the weight is just FP16 — no custom kernels needed.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class BinaryLinear(nn.Module):
    """Linear layer with binary-constrained weights (sign * per-group scale).

    The weight tensor is FP16 but every value within a group shares the same
    absolute magnitude (the group scale). Signs are {-1, +1}.

    This is mathematically equivalent to binary_weight * scale_broadcast,
    but stored as a single FP tensor for native MLX matmul performance.
    """

    def __init__(self, in_dim: int, out_dim: int, group_size: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.group_size = group_size

        # Pad in_dim to multiple of group_size
        self.padded_in = in_dim + (group_size - in_dim % group_size) % group_size
        self.num_groups = self.padded_in // group_size

        # Initialize: random signs * uniform scale ≈ Kaiming
        init_scale = 1.0 / math.sqrt(in_dim)
        signs = mx.where(mx.random.normal((out_dim, self.padded_in)) >= 0,
                         mx.array(1.0, dtype=mx.float16),
                         mx.array(-1.0, dtype=mx.float16))
        # Weight = sign * scale (FP16, native matmul)
        self.weight = (signs * init_scale).astype(mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight[:, :self.in_dim].astype(x.dtype)
        return x @ w.T

    # ------------------------------------------------------------------
    # Binary structure accessors (for ES and serialization)
    # ------------------------------------------------------------------

    def get_signs(self) -> mx.array:
        """Extract binary signs as int8 {-1, +1}."""
        return mx.where(self.weight >= 0,
                        mx.array(1, dtype=mx.int8),
                        mx.array(-1, dtype=mx.int8))

    def get_scales(self) -> mx.array:
        """Extract per-group scales as FP16 (out_dim, num_groups)."""
        w_abs = mx.abs(self.weight.astype(mx.float32))
        w_grouped = w_abs.reshape(self.out_dim, self.num_groups, self.group_size)
        return mx.mean(w_grouped, axis=2).astype(mx.float16)

    def set_signs(self, signs: mx.array) -> None:
        """Set binary signs while preserving per-group scale magnitudes.

        Args:
            signs: (out_dim, padded_in) int8 or float with values in {-1, +1}
        """
        magnitudes = mx.abs(self.weight)
        self.weight = (signs.astype(mx.float16) * magnitudes).astype(mx.float16)

    def flip_signs(self, mask: mx.array) -> None:
        """Flip signs where mask is True (in-place negate).

        Args:
            mask: boolean array, same shape as weight. True = negate.
        """
        flip = mx.where(mask, mx.array(-1.0, dtype=mx.float16),
                         mx.array(1.0, dtype=mx.float16))
        self.weight = (self.weight * flip).astype(mx.float16)


def init_binary_from_fp(layer: BinaryLinear, fp_weight: mx.array) -> None:
    """Initialize from a pretrained FP weight matrix.

    Takes sign(W) * mean(|W|) per group — the optimal 1-bit approximation.
    """
    w = fp_weight.astype(mx.float32)
    if w.shape[1] < layer.padded_in:
        pad = mx.zeros((w.shape[0], layer.padded_in - w.shape[1]), dtype=mx.float32)
        w = mx.concatenate([w, pad], axis=1)

    signs = mx.where(w >= 0, 1.0, -1.0)
    w_grouped = mx.abs(w).reshape(layer.out_dim, layer.num_groups, layer.group_size)
    scales = mx.mean(w_grouped, axis=2)  # (out_dim, num_groups)
    scales_broadcast = mx.repeat(scales, layer.group_size, axis=1)

    layer.weight = (signs * scales_broadcast).astype(mx.float16)


def param_size_report(layer: BinaryLinear) -> dict:
    """Report serialized size for this layer."""
    # Binary: 1 bit per weight for signs
    sign_bits = layer.weight.size
    # Scales: FP16 per group
    scale_bytes = layer.num_groups * layer.out_dim * 2
    return {
        'sign_bits': sign_bits,
        'sign_bytes': sign_bits // 8,
        'scale_bytes': scale_bytes,
        'total_bytes': sign_bits // 8 + scale_bytes,
        'fp16_bytes': layer.weight.size * 2,  # training size (FP16)
    }
