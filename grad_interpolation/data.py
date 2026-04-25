"""
Minimal FineWeb token loader for the spectral line-search experiments.

Reads .bin shards in the same format used by the parent project (uint16 tokens
prefixed by a 256-int32 header with magic 20240520, version 1). Yields
(input_ids, target_ids) MLX arrays of shape (batch, seq_len).

Defaults match the parent repo:
  ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
  ./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import mlx.core as mx


HEADER_INT32_COUNT = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1


def load_data_shard(path: Path) -> np.ndarray:
    """Load a single .bin shard, validating the header. Returns int32 token array."""
    header_bytes = HEADER_INT32_COUNT * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=HEADER_INT32_COUNT)
    if header.size != HEADER_INT32_COUNT or int(header[0]) != HEADER_MAGIC or int(header[1]) != HEADER_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class TokenLoader:
    """Cycle through training shards yielding (input_ids, target_ids) batches.

    Reads shards lazily and concatenates as needed. Batches are non-overlapping
    contiguous chunks of seq_len tokens from the current shard.
    """
    def __init__(self, pattern: str, seq_len: int = 128, batch_size: int = 32, seed: int = 0):
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No shards matched {pattern}")
        self.files = [Path(f) for f in files]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.shard_idx = 0
        self.tokens: np.ndarray | None = None
        self.cursor = 0
        self._load_next_shard()

    def _load_next_shard(self) -> None:
        path = self.files[self.shard_idx % len(self.files)]
        self.tokens = load_data_shard(path)
        self.cursor = 0
        self.shard_idx += 1

    def next_batch(self) -> tuple[mx.array, mx.array]:
        chunk = self.batch_size * self.seq_len + 1
        if self.tokens is None or self.cursor + chunk > len(self.tokens):
            self._load_next_shard()
        block = self.tokens[self.cursor:self.cursor + chunk]
        self.cursor += chunk - 1
        x = block[:-1].reshape(self.batch_size, self.seq_len)
        y = block[1:].reshape(self.batch_size, self.seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def load_validation_tokens(pattern: str) -> np.ndarray:
    """Concatenate all validation shards into one int32 array."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val shards matched {pattern}")
    arrs = [load_data_shard(Path(f)) for f in files]
    return np.concatenate(arrs)


def make_fixed_batch(tokens: np.ndarray, batch_size: int, seq_len: int,
                     seed: int = 0) -> tuple[mx.array, mx.array]:
    """Construct a deterministic (x, y) batch sampled from `tokens`.

    Used by the static probe so that L(w) and L(w - eta*d) are evaluated on
    exactly the same data — the comparison only makes sense if the batch is
    fixed across all evaluations.
    """
    rng = np.random.default_rng(seed)
    max_start = len(tokens) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([tokens[s:s + seq_len] for s in starts])
    y = np.stack([tokens[s + 1:s + 1 + seq_len] for s in starts])
    return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


DEFAULT_TRAIN_PATTERN = "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
DEFAULT_VAL_PATTERN = "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"


if __name__ == "__main__":
    loader = TokenLoader(DEFAULT_TRAIN_PATTERN, seq_len=128, batch_size=4)
    x, y = loader.next_batch()
    print(f"x.shape: {x.shape}, dtype: {x.dtype}, max: {int(x.max())}, min: {int(x.min())}")
    print(f"y.shape: {y.shape}")
    val_tokens = load_validation_tokens(DEFAULT_VAL_PATTERN)
    print(f"val tokens: {val_tokens.size:,}")
    xv, yv = make_fixed_batch(val_tokens, batch_size=8, seq_len=128, seed=42)
    print(f"fixed val batch: {xv.shape}")
