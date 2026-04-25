"""
Tiny GPT-style transformer in MLX, sized for the spectral line-search
characterization experiments.

Default config: vocab=1024, dim=128, n_layers=3, n_heads=4, ffn_mult=2,
seq_len=128 -> ~520k parameters, ~ a couple MB of state. Comfortably runs
many forward passes per second on Apple Silicon.

Design choices kept deliberately plain:
  - Tied input/output embeddings (no separate LM head).
  - Pre-norm RMSNorm (no learnable weight).
  - Standard scaled-dot-product attention (no RoPE, no softcap).
  - GELU MLP (FFN_mult=2 keeps params low).

The point of the experiments is to characterize loss-landscape structure,
not to win benchmarks; absolute loss values are uninteresting.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

COMPUTE_DTYPE = mx.float32  # keep deterministic for spectral fits


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, H, L, Hd)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # Causal mask
        mask = mx.tril(mx.ones((L, L), dtype=x.dtype))
        scores = mx.where(mask[None, None] == 0, mx.array(-1e9, dtype=scores.dtype), scores)
        attn = mx.softmax(scores, axis=-1)
        y = attn @ v                                 # (B, H, L, Hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_heads)
        self.mlp = MLP(dim, ffn_mult)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 1024, dim: int = 128, n_layers: int = 3,
                 n_heads: int = 4, ffn_mult: int = 2, max_seq_len: int = 128,
                 init_std: float = 0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.tok_emb.weight = mx.random.normal(self.tok_emb.weight.shape) * init_std
        self.pos_emb.weight = mx.random.normal(self.pos_emb.weight.shape) * init_std

        self.blocks = [Block(dim, n_heads, ffn_mult) for _ in range(n_layers)]

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, L = input_ids.shape
        pos = mx.arange(L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x)
        # tied weights: logits = x @ tok_emb.weight.T
        logits = x @ self.tok_emb.weight.T
        return logits

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        logits = self(input_ids)
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            reduction="mean",
        )


def param_count(model: nn.Module) -> int:
    from mlx.utils import tree_flatten
    return sum(int(mx.prod(mx.array(p.shape)).item()) for _, p in tree_flatten(model.parameters()))


if __name__ == "__main__":
    mx.random.seed(0)
    m = TinyGPT()
    n = param_count(m)
    print(f"Param count: {n:,}")
    x = mx.random.randint(0, 1024, (2, 128))
    y = mx.random.randint(0, 1024, (2, 128))
    loss = m.loss(x, y)
    mx.eval(loss)
    print(f"Initial loss: {float(loss):.4f}  (random baseline ~ {math.log(1024):.4f})")
