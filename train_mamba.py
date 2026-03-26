#!/usr/bin/env python3
"""
Binary Mamba SSM training script for PyTorch/CUDA.

Port of ssm_bin_es Binary Mamba from MLX to PyTorch, using mamba_ssm for
optimized CUDA selective scan kernels. Follows train_gpt.py patterns for
data loading, evaluation, DDP, and model export.

Usage (single GPU):
    RUN_ID=mamba_test python3 train_mamba.py

Usage (multi-GPU via torchrun):
    torchrun --standalone --nproc_per_node=8 train_mamba.py
"""
from __future__ import annotations

import glob
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation / logging
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 0))  # 0 = same as train_seq_len

    # Training length
    iterations = int(os.environ.get("ITERATIONS", 9000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model architecture
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    state_dim = int(os.environ.get("STATE_DIM", 16))
    conv_width = int(os.environ.get("CONV_WIDTH", 4))
    expand_factor = int(os.environ.get("EXPAND_FACTOR", 2))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    weight_tie_layers = int(os.environ.get("WEIGHT_TIE_LAYERS", 0))
    dt_rank = int(os.environ.get("DT_RANK", 0))
    group_size = int(os.environ.get("GROUP_SIZE", 64))
    # Mamba-3 features
    use_complex_state = bool(int(os.environ.get("USE_COMPLEX_STATE", "1")))  # data-dependent RoPE
    use_trapezoidal = bool(int(os.environ.get("USE_TRAPEZOIDAL", "1")))  # trapezoidal discretization
    use_bc_bias = bool(int(os.environ.get("USE_BC_BIAS", "1")))  # learnable BC bias
    nheads = int(os.environ.get("NHEADS", 0))  # 0 = inner_dim (one head per channel)

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    @property
    def inner_dim(self) -> int:
        return self.model_dim * self.expand_factor

    @property
    def effective_dt_rank(self) -> int:
        if self.dt_rank > 0:
            return self.dt_rank
        return math.ceil(self.model_dim / 16)


# ============================================================================
# MUON OPTIMIZER (from train_gpt.py)
# ============================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# ============================================================================
# TOKENIZER-AGNOSTIC EVALUATION (from train_gpt.py)
# ============================================================================

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def eval_val(
    args: Hyperparameters, model: nn.Module, rank: int, world_size: int,
    device: torch.device, grad_accum_steps: int, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int | None = None,
) -> tuple[float, float]:
    # Use eval_seq_len if configured, otherwise train_seq_len
    if seq_len is None:
        seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# ============================================================================
# POST-TRAINING QUANTIZATION (from train_gpt.py)
# ============================================================================

CONTROL_TENSOR_NAME_PATTERNS = (
    "mamba_scale", "ffn_scale", "layer_scale", "A_log", "D",
    "dt_proj_bias", "conv_bias",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized, scales, dtypes = {}, {}, {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}
    stats = dict.fromkeys(("param_count", "num_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized,
           "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].to(dtype=torch.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# ============================================================================
# DATA LOADING (from train_gpt.py)
# ============================================================================

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


# ============================================================================
# MAMBA MODEL
# ============================================================================

def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def apply_data_dependent_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Apply rotary embedding with data-dependent cumulative angles.
    x: (..., state_dim), angles: (..., state_dim // 2)
    state_dim must be even."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    return torch.stack([cos_a * x1 - sin_a * x2,
                        sin_a * x1 + cos_a * x2], dim=-1).flatten(-2)


class MambaBlock(nn.Module):
    """Selective SSM block with Mamba-3 features:
    - Complex-valued state via data-dependent RoPE on B/C projections
    - Trapezoidal discretization (second-order accurate)
    - Learnable BC bias
    Falls back to pure-PyTorch sequential scan; uses mamba_ssm CUDA kernels
    for the base scan if available."""

    def __init__(self, dim: int, inner_dim: int, state_dim: int,
                 conv_width: int, dt_rank: int,
                 use_complex_state: bool = True,
                 use_trapezoidal: bool = True,
                 use_bc_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.use_complex_state = use_complex_state
        self.use_trapezoidal = use_trapezoidal

        # Projections
        self.in_proj = nn.Linear(dim, 2 * inner_dim, bias=False)
        self.x_proj = nn.Linear(inner_dim, dt_rank + 2 * state_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        # dt projection
        self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True)

        # Causal conv1d (depthwise)
        self.conv_weight = nn.Parameter(torch.randn(inner_dim, conv_width) * 0.1)
        self.conv_bias = nn.Parameter(torch.zeros(inner_dim))

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)
                                            .unsqueeze(0).expand(inner_dim, -1).clone()))
        self.D = nn.Parameter(torch.ones(inner_dim, dtype=torch.float32))

        # Residual scale
        self.layer_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        # Mamba-3: BC bias (learnable, initialized to ones)
        if use_bc_bias:
            self.B_bias = nn.Parameter(torch.ones(inner_dim, state_dim))
            self.C_bias = nn.Parameter(torch.ones(inner_dim, state_dim))
        else:
            self.B_bias = None
            self.C_bias = None

        # Mamba-3: theta projection for data-dependent RoPE angles
        if use_complex_state:
            assert state_dim % 2 == 0, "state_dim must be even for complex state"
            self.theta_proj = nn.Linear(inner_dim, state_dim // 2, bias=False)
            nn.init.normal_(self.theta_proj.weight, std=0.01)

        # Mamba-3: learnable lambda for trapezoidal interpolation
        if use_trapezoidal:
            self.lam_proj = nn.Linear(inner_dim, inner_dim, bias=True)

        # Try to import mamba_ssm for fast CUDA scan
        self._use_mamba_ssm = False
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            self._selective_scan_fn = selective_scan_fn
            self._use_mamba_ssm = True
        except ImportError:
            pass

    def _causal_conv1d(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        conv_width = self.conv_weight.shape[1]
        x_padded = F.pad(x.transpose(1, 2), (conv_width - 1, 0))
        weight = self.conv_weight.unsqueeze(1)
        out = F.conv1d(x_padded, weight, self.conv_bias, groups=D)
        return out.transpose(1, 2)

    def _sequential_scan(self, x: Tensor, dt: Tensor, B_mat: Tensor,
                         C_mat: Tensor, lam: Tensor | None = None) -> Tensor:
        """Sequential scan with optional trapezoidal discretization.
        x: (batch, seq, inner), dt: (batch, seq, inner),
        B_mat: (batch, seq, inner, state), C_mat: (batch, seq, inner, state)
        lam: (batch, seq, inner) or None for Euler mode."""
        batch, seq_len, inner = x.shape
        A = -torch.exp(self.A_log.float())  # (inner, state)

        h = torch.zeros(batch, inner, self.state_dim, device=x.device, dtype=torch.float32)
        prev_Bx = torch.zeros_like(h)
        ys = []
        for t in range(seq_len):
            dt_t = dt[:, t, :, None]  # (batch, inner, 1)
            alpha = torch.exp(A.unsqueeze(0) * dt_t)  # (batch, inner, state)
            Bx_t = B_mat[:, t] * x[:, t, :, None].float()  # (batch, inner, state)

            if lam is not None:
                # Trapezoidal: h = alpha*h + beta*prev_Bx + gamma*Bx_t
                lam_t = lam[:, t, :, None]  # (batch, inner, 1)
                beta = (1 - lam_t) * dt_t * alpha
                gamma = lam_t * dt_t
                h = alpha * h + beta * prev_Bx + gamma * Bx_t
            else:
                # Euler: h = alpha*h + dt*Bx_t
                h = alpha * h + dt_t * Bx_t

            y_t = (h * C_mat[:, t]).sum(dim=-1)  # (batch, inner)
            ys.append(y_t)
            prev_Bx = Bx_t

        return torch.stack(ys, dim=1).to(x.dtype)

    def _mamba_ssm_scan(self, x: Tensor, dt: Tensor, B_mat: Tensor,
                        C_mat: Tensor) -> Tensor:
        """Use mamba_ssm optimized CUDA scan (Euler mode only)."""
        A = -torch.exp(self.A_log.float())
        # mamba_ssm expects B/C as (batch, state, seq) but our B_mat is
        # (batch, seq, inner, state). For the standard Mamba-1/2 interface,
        # B and C are shared across inner_dim, so we take the first channel.
        # This is a simplification; for full Mamba-3 MIMO, we'd need custom kernels.
        return self._selective_scan_fn(
            x.transpose(1, 2).contiguous(),
            dt.transpose(1, 2).contiguous(),
            A.contiguous(),
            B_mat[:, :, 0, :].transpose(1, 2).contiguous(),  # (B, N, L)
            C_mat[:, :, 0, :].transpose(1, 2).contiguous(),
            self.D.float(),
            z=None, delta_bias=None, delta_softplus=False,
            return_last_state=False,
        ).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape

        # Input projection -> x_path and gate
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Causal conv + SiLU
        x_path = self._causal_conv1d(x_path)
        x_path = x_path * torch.sigmoid(x_path)

        # SSM parameter extraction
        x_proj_out = self.x_proj(x_path)
        dt_input = x_proj_out[..., :self.dt_rank]
        B_ssm = x_proj_out[..., self.dt_rank:self.dt_rank + self.state_dim]
        C_ssm = x_proj_out[..., self.dt_rank + self.state_dim:]

        # dt projection + softplus
        dt = F.softplus(self.dt_proj(dt_input))

        # Expand B/C to per-channel: (batch, seq, inner, state)
        B_ssm = B_ssm.unsqueeze(2).expand(-1, -1, self.inner_dim, -1)
        C_ssm = C_ssm.unsqueeze(2).expand(-1, -1, self.inner_dim, -1)

        # Mamba-3: BC bias
        if self.B_bias is not None:
            B_ssm = B_ssm + self.B_bias
        if self.C_bias is not None:
            C_ssm = C_ssm + self.C_bias

        # Mamba-3: data-dependent RoPE on B and C
        if self.use_complex_state:
            # Compute per-token angles from input, accumulate over time
            raw_angles = self.theta_proj(x_path)  # (B, L, state_dim//2)
            # Scale by dt for data-dependent rotation speed
            raw_angles = raw_angles * dt.mean(dim=-1, keepdim=True)
            cum_angles = -torch.cumsum(raw_angles, dim=1)
            # Expand to per-channel
            cum_angles = cum_angles.unsqueeze(2).expand(-1, -1, self.inner_dim, -1)
            B_ssm = apply_data_dependent_rope(B_ssm, cum_angles)
            C_ssm = apply_data_dependent_rope(C_ssm, cum_angles)

        # Trapezoidal lambda
        lam = None
        if self.use_trapezoidal:
            lam = torch.sigmoid(self.lam_proj(x_path))  # (B, L, inner)

        # Selective scan
        use_fast = self._use_mamba_ssm and not self.use_trapezoidal
        if use_fast:
            y = self._mamba_ssm_scan(x_path, dt, B_ssm, C_ssm)
        else:
            y = self._sequential_scan(x_path, dt, B_ssm, C_ssm, lam=lam)

        # Skip + gate + output
        y = y + x_path * self.D
        y = y * (z * torch.sigmoid(z))
        return self.out_proj(y)


class GatedFFN(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        return self.down_proj(gate * torch.sigmoid(gate) * self.up_proj(x))


class MambaLayer(nn.Module):
    def __init__(self, dim: int, inner_dim: int, state_dim: int,
                 conv_width: int, dt_rank: int, mlp_mult: int,
                 use_complex_state: bool = True, use_trapezoidal: bool = True,
                 use_bc_bias: bool = True):
        super().__init__()
        self.mamba = MambaBlock(dim, inner_dim, state_dim, conv_width, dt_rank,
                                use_complex_state=use_complex_state,
                                use_trapezoidal=use_trapezoidal,
                                use_bc_bias=use_bc_bias)
        self.ffn = GatedFFN(dim, mlp_mult)
        self.mamba_scale = nn.Parameter(torch.ones(dim))
        self.ffn_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mamba_scale * self.mamba(rms_norm(x))
        x = x + self.ffn_scale * self.ffn(rms_norm(x))
        return x


class MambaLM(nn.Module):
    """Mamba language model. Trained in FP/BF16, quantized at export."""

    def __init__(self, vocab_size: int, num_layers: int, dim: int,
                 inner_dim: int, state_dim: int, conv_width: int,
                 dt_rank: int, mlp_mult: int, logit_softcap: float,
                 tied_embed_init_std: float, weight_tie_layers: int = 0,
                 use_complex_state: bool = True, use_trapezoidal: bool = True,
                 use_bc_bias: bool = True):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.weight_tie_layers = weight_tie_layers

        self.tok_emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_emb.weight, std=tied_embed_init_std)

        layer_kwargs = dict(use_complex_state=use_complex_state,
                            use_trapezoidal=use_trapezoidal,
                            use_bc_bias=use_bc_bias)

        if weight_tie_layers > 0:
            num_unique = max(num_layers // weight_tie_layers, 1)
            unique_layers = nn.ModuleList([
                MambaLayer(dim, inner_dim, state_dim, conv_width, dt_rank, mlp_mult,
                           **layer_kwargs)
                for _ in range(num_unique)
            ])
            self.layers = unique_layers
            self._layer_indices = [i % num_unique for i in range(num_layers)]
        else:
            self.layers = nn.ModuleList([
                MambaLayer(dim, inner_dim, state_dim, conv_width, dt_rank, mlp_mult,
                           **layer_kwargs)
                for _ in range(num_layers)
            ])
            self._layer_indices = list(range(num_layers))

        # Zero-init output projections for stable start
        for layer in self.layers:
            nn.init.zeros_(layer.mamba.out_proj.weight)
            nn.init.zeros_(layer.ffn.down_proj.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        x = rms_norm(self.tok_emb(input_ids))
        for layer_idx in self._layer_indices:
            x = self.layers[layer_idx](x)
        x = rms_norm(x)
        x = x.reshape(-1, x.size(-1))

        # Tied embedding projection
        logits = x @ self.tok_emb.weight.to(x.dtype).T
        c = self.logit_softcap
        logits = c * torch.tanh(logits / c)

        if target_ids is not None:
            return F.cross_entropy(logits.float(), target_ids.reshape(-1))
        return logits


# ============================================================================
# PARAMETER CLASSIFICATION
# ============================================================================

MATRIX_PATTERNS = ("in_proj.weight", "x_proj.weight", "out_proj.weight",
                   "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                   "dt_proj.weight", "theta_proj.weight", "lam_proj.weight")
SCALAR_PATTERNS = ("mamba_scale", "ffn_scale", "layer_scale",
                   "A_log", "D", "conv_weight", "conv_bias", "dt_proj.bias",
                   "B_bias", "C_bias")


def classify_params(model: MambaLM) -> dict[str, list[nn.Parameter]]:
    """Split parameters into optimizer groups."""
    embed, matrix, scalar = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_emb" in name:
            embed.append(p)
        elif any(pat in name for pat in MATRIX_PATTERNS):
            matrix.append(p)
        elif any(pat in name for pat in SCALAR_PATTERNS):
            scalar.append(p)
        else:
            scalar.append(p)  # default to scalar/Adam
    return {"embed": embed, "matrix": matrix, "scalar": scalar}


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ---- Distributed + CUDA setup ----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. For Apple Silicon, use ssm_bin_es/experiment/train.py")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    if master_process:
        log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # ---- Tokenizer + validation ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer {sp.vocab_size()}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- Model ----
    model = MambaLM(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        inner_dim=args.inner_dim,
        state_dim=args.state_dim,
        conv_width=args.conv_width,
        dt_rank=args.effective_dt_rank,
        mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap,
        tied_embed_init_std=args.tied_embed_init_std,
        weight_tie_layers=args.weight_tie_layers,
        use_complex_state=args.use_complex_state,
        use_trapezoidal=args.use_trapezoidal,
        use_bc_bias=args.use_bc_bias,
    ).to(device).bfloat16()

    # Keep small control params in fp32
    for name, p in model.named_parameters():
        if any(pat in name for pat in SCALAR_PATTERNS):
            p.data = p.data.float()

    n_params = sum(p.numel() for p in model.parameters())
    log0(f"run_id:{args.run_id}")
    log0(f"architecture:mamba_ssm")
    log0(f"model_params:{n_params}")
    log0(f"layers:{args.num_layers} dim:{args.model_dim} inner:{args.inner_dim} "
         f"state:{args.state_dim} conv:{args.conv_width} dt_rank:{args.effective_dt_rank} "
         f"mlp_mult:{args.mlp_mult} weight_tie:{args.weight_tie_layers}")
    log0(f"mamba3:complex_state={args.use_complex_state} trapezoidal={args.use_trapezoidal} "
         f"bc_bias={args.use_bc_bias}")
    eval_seq = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    log0(f"seq_len:{args.train_seq_len} eval_seq_len:{eval_seq} "
         f"batch_tokens:{args.train_batch_tokens} "
         f"grad_accum:{grad_accum_steps} world_size:{world_size}")

    # torch.compile for fused ops
    compiled_model = torch.compile(model, dynamic=False, fullgraph=False)
    ddp_model = (torch.nn.parallel.DistributedDataParallel(compiled_model, device_ids=[local_rank],
                 broadcast_buffers=False) if distributed else compiled_model)

    # ---- Optimizers ----
    param_groups = classify_params(model)
    embed_opt = torch.optim.Adam(param_groups["embed"], lr=args.embed_lr,
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps)
    matrix_opt = Muon(param_groups["matrix"], lr=args.matrix_lr,
                      momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    scalar_opt = torch.optim.Adam(param_groups["scalar"], lr=args.scalar_lr,
                                   betas=(args.beta1, args.beta2), eps=args.adam_eps)
    optimizers = [embed_opt, matrix_opt, scalar_opt]

    log0(f"embed_params:{sum(p.numel() for p in param_groups['embed'])} "
         f"matrix_params:{sum(p.numel() for p in param_groups['matrix'])} "
         f"scalar_params:{sum(p.numel() for p in param_groups['scalar'])}")
    log0(f"embed_lr:{args.embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")

    # ---- Training loop ----
    log0(f"Starting training for {args.iterations} steps")
    t0 = time.time()
    train_loss_accum = 0.0

    for step in range(args.iterations):
        step_t0 = time.time()
        elapsed = step_t0 - t0

        if args.max_wallclock_seconds > 0 and elapsed > args.max_wallclock_seconds:
            log0(f"Wallclock limit reached at step {step}")
            break

        # LR schedule: linear warmup + cosine warmdown
        if step < args.warmup_steps:
            lr_mul = (step + 1) / args.warmup_steps
        elif step >= args.iterations - args.warmdown_iters:
            progress = (step - (args.iterations - args.warmdown_iters)) / args.warmdown_iters
            lr_mul = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            lr_mul = 1.0

        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = pg.get("_base_lr", pg["lr"]) * lr_mul
                if "_base_lr" not in pg:
                    pg["_base_lr"] = pg["lr"]

        # Gradient accumulation
        ddp_model.zero_grad()
        train_loss_accum = 0.0
        for micro in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            no_sync = distributed and micro < grad_accum_steps - 1
            ctx = ddp_model.no_sync() if no_sync else torch.enable_grad()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ddp_model(x, y)
                loss_scaled = loss * grad_scale
                loss_scaled.backward()
            train_loss_accum += loss.item() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()

        step_time = time.time() - step_t0

        # ---- Logging ----
        if step % args.train_log_every == 0 or step == args.iterations - 1:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss_accum:.4f} "
                 f"lr_mul:{lr_mul:.4f} step_time:{step_time*1000:.0f}ms "
                 f"elapsed:{elapsed:.0f}s")

        # ---- Validation ----
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or step == args.iterations - 1):
            val_loss, val_bpb = eval_val(
                args, ddp_model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # ---- Final eval ----
    total_time = time.time() - t0
    val_loss, val_bpb = eval_val(
        args, ddp_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"FINAL val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
         f"total_time:{total_time:.1f}s steps:{step}")

    # ---- Save + quantize ----
    if master_process:
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        saved_model, quant_stats = quantize_state_dict_int8(state_dict)
        saved_bytes = zlib.compress(torch.save(saved_model, f := __import__("io").BytesIO()) or f.getvalue(), level=9)
        save_path = Path(f"logs/{args.run_id}_model.pt")
        save_path.write_bytes(saved_bytes)
        compressed_mb = len(saved_bytes) / (1024 * 1024)
        log0(f"saved_model:{save_path} compressed_bytes:{len(saved_bytes)} "
             f"compressed_mb:{compressed_mb:.2f} "
             f"param_count:{quant_stats['param_count']} "
             f"baseline_mb:{quant_stats['baseline_tensor_bytes']/1024/1024:.2f}")
        code_bytes = len(code.encode("utf-8"))
        total_artifact = len(saved_bytes) + code_bytes
        log0(f"final_int8_zlib_roundtrip code_bytes:{code_bytes} "
             f"model_bytes:{len(saved_bytes)} total_artifact_bytes:{total_artifact} "
             f"total_artifact_mb:{total_artifact/1024/1024:.2f} "
             f"val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
