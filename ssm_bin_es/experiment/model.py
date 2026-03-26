"""
Binary Mamba SSM model.

Adapts the Mamba architecture from train_mamba_mlx.py with BinaryLinear layers
for all large projections. Small/stability-critical params stay FP.
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn

from ssm_bin_es.experiment.binary_linear import BinaryLinear

COMPUTE_DTYPE = mx.bfloat16


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class BinaryMambaBlock(nn.Module):
    """
    Mamba SSM block with binary projection weights.

    Binary: in_proj, x_proj, out_proj (large matmuls)
    FP: A_log, D, dt_proj, conv (small, stability-critical)
    """

    def __init__(self, dim: int, inner_dim: int, state_dim: int,
                 conv_width: int, dt_rank: int, group_size: int = 64):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.conv_width = conv_width
        self.dt_rank = dt_rank

        # Binary projections
        self.in_proj = BinaryLinear(dim, 2 * inner_dim, group_size)
        self.x_proj = BinaryLinear(inner_dim, dt_rank + 2 * state_dim, group_size)
        self.out_proj = BinaryLinear(inner_dim, dim, group_size)

        # FP: Causal 1D convolution (depthwise) - small
        self.conv_weight = mx.random.normal((inner_dim, conv_width)) * 0.1
        self.conv_bias = mx.zeros((inner_dim,))

        # FP: dt projection - small (inner x dt_rank)
        self.dt_proj_weight = mx.random.normal((inner_dim, dt_rank)) * (1.0 / math.sqrt(dt_rank))
        self.dt_proj_bias = mx.zeros((inner_dim,))

        # FP: A parameter (log-space)
        A = mx.broadcast_to(
            mx.log(mx.arange(1, state_dim + 1, dtype=mx.float32))[None, :],
            (inner_dim, state_dim)
        )
        self.A_log = mx.array(A)

        # FP: D skip connection
        self.D = mx.ones((inner_dim,))

        # Layer scale
        self.layer_scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape

        # 1. Project to 2*inner_dim
        xz = self.in_proj(x)
        x_path = xz[..., :self.inner_dim]
        z = xz[..., self.inner_dim:]

        # 2. Causal conv1d
        x_path = self._causal_conv1d(x_path)

        # 3. SiLU
        x_path = x_path * mx.sigmoid(x_path)

        # 4. SSM params
        x_dbl = self.x_proj(x_path)
        dt = x_dbl[..., :self.dt_rank]
        B = x_dbl[..., self.dt_rank:self.dt_rank + self.state_dim]
        C = x_dbl[..., self.dt_rank + self.state_dim:]

        # Project dt and softplus
        dt = dt @ self.dt_proj_weight.astype(dt.dtype).T + self.dt_proj_bias.astype(dt.dtype)
        dt = nn.softplus(dt)

        # 5. Selective scan (sequential - compatible with mx.compile)
        y = self._selective_scan(x_path, dt, B, C)

        # 6. Skip connection
        y = y + x_path * self.D.astype(y.dtype)

        # 7. Gate
        y = y * (z * mx.sigmoid(z))

        # 8. Output projection
        return self.out_proj(y)

    def _causal_conv1d(self, x: mx.array) -> mx.array:
        batch, seq_len, dim = x.shape
        pad = mx.zeros((batch, self.conv_width - 1, dim), dtype=x.dtype)
        x_padded = mx.concatenate([pad, x], axis=1)
        w = self.conv_weight.astype(x.dtype)
        out = mx.zeros((batch, seq_len, dim), dtype=x.dtype)
        for k in range(self.conv_width):
            out = out + x_padded[:, k:k + seq_len, :] * w[:, k]
        out = out + self.conv_bias.astype(x.dtype)
        return out

    def _selective_scan(self, x: mx.array, dt: mx.array,
                        B: mx.array, C: mx.array) -> mx.array:
        batch, seq_len, inner = x.shape
        A = -mx.exp(self.A_log.astype(mx.float32))

        h = mx.zeros((batch, inner, self.state_dim), dtype=mx.float32)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :].astype(mx.float32)
            dt_t = dt[:, t, :].astype(mx.float32)
            B_t = B[:, t, :].astype(mx.float32)
            C_t = C[:, t, :].astype(mx.float32)

            A_bar = mx.exp(A[None, :, :] * dt_t[:, :, None])
            B_bar = dt_t[:, :, None] * B_t[:, None, :]

            h = A_bar * h + B_bar * x_t[:, :, None]
            y_t = mx.sum(h * C_t[:, None, :], axis=-1)
            outputs.append(y_t)

        return mx.stack(outputs, axis=1).astype(x.dtype)


class BinaryGatedFFN(nn.Module):
    """SwiGLU FFN with binary projection weights."""

    def __init__(self, dim: int, mlp_mult: int, group_size: int = 64):
        super().__init__()
        hidden = dim * mlp_mult
        self.gate_proj = BinaryLinear(dim, hidden, group_size)
        self.up_proj = BinaryLinear(dim, hidden, group_size)
        self.down_proj = BinaryLinear(hidden, dim, group_size)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        gate = gate * mx.sigmoid(gate)  # SiLU
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class BinaryMambaLayer(nn.Module):
    """One binary Mamba block + binary FFN with residual connections."""

    def __init__(self, dim: int, inner_dim: int, state_dim: int,
                 conv_width: int, dt_rank: int, mlp_mult: int,
                 group_size: int = 64):
        super().__init__()
        self.mamba_norm = RMSNormNoWeight()
        self.mamba = BinaryMambaBlock(dim, inner_dim, state_dim, conv_width,
                                      dt_rank, group_size)
        self.ffn_norm = RMSNormNoWeight()
        self.ffn = BinaryGatedFFN(dim, mlp_mult, group_size)
        self.mamba_scale = mx.ones((dim,), dtype=mx.float32)
        self.ffn_scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.mamba_scale.astype(x.dtype) * self.mamba(self.mamba_norm(x))
        x = x + self.ffn_scale.astype(x.dtype) * self.ffn(self.ffn_norm(x))
        return x


class BinaryMambaLM(nn.Module):
    """
    Binary Mamba language model.

    All large projections use BinaryLinear. Embeddings, SSM dynamics (A_log, D),
    convolutions, and dt projections stay in FP for stability.
    """

    def __init__(self, vocab_size: int, num_layers: int, dim: int,
                 inner_dim: int, state_dim: int, conv_width: int,
                 dt_rank: int, mlp_mult: int, logit_softcap: float,
                 tied_embed_init_std: float, weight_tie_layers: int = 0,
                 group_size: int = 64):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.weight_tie_layers = weight_tie_layers

        # Embedding stays FP (lookup table, not a matmul)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32)
            * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

        if weight_tie_layers > 0:
            num_unique = max(num_layers // weight_tie_layers, 1)
            self.layers = [
                BinaryMambaLayer(dim, inner_dim, state_dim, conv_width,
                                 dt_rank, mlp_mult, group_size)
                for _ in range(num_unique)
            ]
            self._layer_indices = [i % num_unique for i in range(num_layers)]
        else:
            self.layers = [
                BinaryMambaLayer(dim, inner_dim, state_dim, conv_width,
                                 dt_rank, mlp_mult, group_size)
                for _ in range(num_layers)
            ]
            self._layer_indices = list(range(num_layers))

        self.final_norm = RMSNormNoWeight()

        # Zero-init output projections for stable training start
        for layer in self.layers:
            layer.mamba.out_proj.weight = mx.zeros_like(layer.mamba.out_proj.weight)
            layer.ffn.down_proj.weight = mx.zeros_like(layer.ffn.down_proj.weight)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        for layer_idx in self._layer_indices:
            x = self.layers[layer_idx](x)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# =========================================================================
# Utility: enumerate binary vs continuous parameters
# =========================================================================

# Patterns identifying BinaryLinear weight params (ES-evolved, sign-flipped).
# These are the .weight params inside in_proj, x_proj, out_proj, gate_proj, etc.
BINARY_PROJ_NAMES = ("in_proj.weight", "x_proj.weight", "out_proj.weight",
                     "gate_proj.weight", "up_proj.weight", "down_proj.weight")

# Patterns identifying small control tensors that should use scalar optimizer
CONTROL_PATTERNS = (
    "layer_scale", "mamba_scale", "ffn_scale", "A_log", "D",
    "dt_proj_bias", "conv_bias",
)


def _is_binary_weight(name: str) -> bool:
    """Check if a parameter name corresponds to a BinaryLinear weight."""
    return any(name.endswith(suffix) for suffix in BINARY_PROJ_NAMES)


def split_params(model: BinaryMambaLM) -> dict:
    """Split model parameters into categories for the hybrid optimizer.

    Returns dict with keys:
        'binary': list of (name, param) — BinaryLinear weights (ES-evolved)
        'embed': list of (name, param) — embedding (gradient-trained)
        'continuous_matrix': list of (name, param) — FP 2D params like conv, dt_proj
        'continuous_scalar': list of (name, param) — FP 1D/control params (Adam)
    """
    from mlx.utils import tree_flatten

    result = {
        'binary': [],
        'embed': [],
        'continuous_matrix': [],
        'continuous_scalar': [],
    }

    for name, param in tree_flatten(model.parameters()):
        if name == "tok_emb.weight":
            result['embed'].append((name, param))
        elif _is_binary_weight(name):
            result['binary'].append((name, param))
        elif param.ndim == 2 and not any(p in name for p in CONTROL_PATTERNS):
            result['continuous_matrix'].append((name, param))
        else:
            result['continuous_scalar'].append((name, param))

    return result


def param_budget_report(model: BinaryMambaLM) -> dict:
    """Report parameter counts and estimated serialized size.

    Binary weights serialize as 1-bit signs + FP16 per-group scales.
    """
    from mlx.utils import tree_flatten

    split = split_params(model)

    # Binary: 1 bit per sign + 16 bits per group for scale
    binary_weight_count = sum(p.size for _, p in split['binary'])
    # Estimate groups: each BinaryLinear has num_groups = padded_in / group_size
    # For budget, approximate as weight_size / group_size groups per layer
    group_size = 64  # default
    binary_sign_bytes = binary_weight_count // 8
    binary_scale_count = binary_weight_count // group_size
    binary_scale_bytes = binary_scale_count * 2  # FP16

    embed_bytes = sum(p.size * 2 for _, p in split['embed'])
    cont_matrix_bytes = sum(p.size * 2 for _, p in split['continuous_matrix'])
    cont_scalar_bytes = sum(p.size * 4 for _, p in split['continuous_scalar'])

    total_bytes = binary_sign_bytes + binary_scale_bytes + embed_bytes + cont_matrix_bytes + cont_scalar_bytes

    return {
        'binary_params': binary_weight_count,
        'binary_sign_bytes': binary_sign_bytes,
        'binary_scale_bytes': binary_scale_bytes,
        'binary_total_bytes': binary_sign_bytes + binary_scale_bytes,
        'embed_params': sum(p.size for _, p in split['embed']),
        'embed_bytes': embed_bytes,
        'continuous_matrix_params': sum(p.size for _, p in split['continuous_matrix']),
        'continuous_matrix_bytes': cont_matrix_bytes,
        'continuous_scalar_params': sum(p.size for _, p in split['continuous_scalar']),
        'continuous_scalar_bytes': cont_scalar_bytes,
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 * 1024),
    }
