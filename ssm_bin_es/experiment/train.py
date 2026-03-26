#!/usr/bin/env python3
"""
Hybrid ES + Gradient training for Binary Mamba SSM.

Training loop:
  Phase 1 (optional): Gradient warmup with STE on binary weights
  Phase 2: Hybrid ES (binary weights) + gradient descent (continuous params)
  Phase 3: Refinement with annealed flip rate

Reuses data loading, eval/BPB, and serialization from the parent project.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from ssm_bin_es.experiment.binary_linear import BinaryLinear
from ssm_bin_es.experiment.model import (
    BinaryMambaLM, COMPUTE_DTYPE, split_params, param_budget_report,
    rms_norm, _is_binary_weight, CONTROL_PATTERNS,
)
from ssm_bin_es.experiment.es import BinaryES, ESConfig
from ssm_bin_es.experiment.batched_eval import BatchedPopulationEvaluator
from ssm_bin_es.kernels.scan_integration import patch_model_for_metal_scan

# Metal scan mode: 0=Python, 1=Metal fwd only, 2=Metal fwd+bwd
USE_METAL_SCAN = int(os.environ.get("USE_METAL_SCAN", "0"))
EVAL_GROUP_SIZE = int(os.environ.get("EVAL_GROUP_SIZE", "4"))

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop
    es_generations: int = int(os.environ.get("ES_GENERATIONS", 500))
    warmup_generations: int = int(os.environ.get("WARMUP_GENERATIONS", 0))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 25))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 5))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # ES config
    pop_size: int = int(os.environ.get("POP_SIZE", 32))
    elite_frac: float = float(os.environ.get("ELITE_FRAC", 0.25))
    init_flip_rate: float = float(os.environ.get("INIT_FLIP_RATE", 0.01))
    min_flip_rate: float = float(os.environ.get("MIN_FLIP_RATE", 0.0001))
    flip_rate_decay: float = float(os.environ.get("FLIP_RATE_DECAY", 0.999))
    crossover_rate: float = float(os.environ.get("CROSSOVER_RATE", 0.3))
    es_eval_tokens: int = int(os.environ.get("ES_EVAL_TOKENS", 32768))
    gradient_steps_per_gen: int = int(os.environ.get("GRADIENT_STEPS_PER_GEN", 4))
    gradient_batch_tokens: int = int(os.environ.get("GRADIENT_BATCH_TOKENS", 65536))

    # Model architecture
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 12))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    state_dim: int = int(os.environ.get("STATE_DIM", 16))
    conv_width: int = int(os.environ.get("CONV_WIDTH", 4))
    expand_factor: int = int(os.environ.get("EXPAND_FACTOR", 2))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    weight_tie_layers: int = int(os.environ.get("WEIGHT_TIE_LAYERS", 0))
    dt_rank: int = int(os.environ.get("DT_RANK", 0))
    group_size: int = int(os.environ.get("GROUP_SIZE", 64))

    # Continuous param optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    embed_lr: float = float(os.environ.get("EMBED_LR", 0.05))
    scale_lr: float = float(os.environ.get("SCALE_LR", 0.01))
    continuous_lr: float = float(os.environ.get("CONTINUOUS_LR", 0.01))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def inner_dim(self) -> int:
        return self.model_dim * self.expand_factor

    @property
    def effective_dt_rank(self) -> int:
        if self.dt_rank > 0:
            return self.dt_rank
        return math.ceil(self.model_dim / 16)


# ==============================================================================
# DATA LOADING (adapted from train_mamba_mlx.py)
# ==============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class TokenLoader:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None):
        self.stream = TokenStream(pattern, log_fn=log_fn)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# ==============================================================================
# EVAL (BPB)
# ==============================================================================

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def eval_val(
    model: BinaryMambaLM,
    val_tokens: np.ndarray,
    seq_len: int,
    batch_seqs: int,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    """Evaluate validation loss and BPB."""
    total_seqs = (val_tokens.size - 1) // seq_len
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0

    for batch_start in range(0, total_seqs, batch_seqs):
        batch_end = min(batch_start + batch_seqs, total_seqs)
        raw_start = batch_start * seq_len
        raw_end = batch_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = model.loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())

    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


# ==============================================================================
# CONTINUOUS PARAMETER OPTIMIZER
# ==============================================================================

class ContinuousOptimizer:
    """Adam optimizer for all non-binary parameters (scales, embeddings, SSM params)."""

    def __init__(self, model: BinaryMambaLM, args: Hyperparameters):
        self.args = args
        split = split_params(model)

        self.embed_keys = [n for n, _ in split['embed']]
        self.binary_keys = [n for n, _ in split['binary']]  # gradient tunes magnitudes
        self.matrix_keys = [n for n, _ in split['continuous_matrix']]
        self.scalar_keys = [n for n, _ in split['continuous_scalar']]

        # All continuous params get Adam (simpler than Muon for now,
        # since the continuous params are small)
        self.adam = optim.Adam(
            learning_rate=args.continuous_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: BinaryMambaLM, grads_tree: dict, lr_mul: float = 1.0) -> None:
        """Apply gradient updates to all continuous parameters."""
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        # Collect all continuous params and their gradients
        # Cast everything to float32 for optimizer stability (float16 scales
        # cause NaN in Adam's variance estimates due to eps underflow)
        cont_keys = self.binary_keys + self.embed_keys + self.matrix_keys + self.scalar_keys
        cont_grads = {}
        cont_params = {}
        orig_dtypes = {}
        for k in cont_keys:
            if k in grads:
                orig_dtypes[k] = params[k].dtype
                cont_grads[k] = grads[k].astype(mx.float32)
                cont_params[k] = params[k].astype(mx.float32)

        if cont_grads:
            self.adam.learning_rate = self.args.continuous_lr * lr_mul
            adam_out = self.adam.apply_gradients(cont_grads, cont_params)
            # Cast back to original dtypes
            for k, v in adam_out.items():
                updated[k] = v.astype(orig_dtypes[k])

        model.update(tree_unflatten(list(updated.items())))


# ==============================================================================
# MODEL <-> ES INTERFACE
# ==============================================================================

def extract_binary_weights(model: BinaryMambaLM) -> dict[str, mx.array]:
    """Extract all BinaryLinear weight tensors from the model."""
    return {name: param for name, param in tree_flatten(model.parameters())
            if _is_binary_weight(name)}


def inject_binary_weights(model: BinaryMambaLM,
                          binary_weights: dict[str, mx.array]) -> None:
    """Inject binary weight tensors into the model."""
    params = dict(tree_flatten(model.parameters()))
    params.update(binary_weights)
    model.update(tree_unflatten(list(params.items())))


# ==============================================================================
# GRADIENT COMPUTATION FOR CONTINUOUS PARAMS
# ==============================================================================

def _make_loss_fn(model: BinaryMambaLM):
    """Create a loss function for nn.value_and_grad.

    Binary weights are int8 and non-differentiable by construction.
    MLX's autograd will skip them automatically.
    """
    def loss_fn(x: mx.array, y: mx.array) -> mx.array:
        return model.loss(x, y)
    return loss_fn


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    # Log source code
    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={sp.vocab_size()}")

    # Data
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)
    train_loader = TokenLoader(args.train_files, log_fn=log)

    mx.random.seed(args.seed)

    # Build model
    model = BinaryMambaLM(
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
        group_size=args.group_size,
    )

    # Report parameter budget
    budget = param_budget_report(model)
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"architecture:binary_mamba_es")
    log(f"model_params:{n_params}")
    log(f"binary_params:{budget['binary_params']} "
        f"(signs:{budget['binary_sign_bytes']//1024}KB + scales:{budget['binary_scale_bytes']//1024}KB "
        f"= {budget['binary_total_bytes'] / 1024 / 1024:.2f} MB serialized)")
    log(f"embed_params:{budget['embed_params']} ({budget['embed_bytes'] / 1024 / 1024:.2f} MB)")
    log(f"continuous_params:{budget['continuous_matrix_params'] + budget['continuous_scalar_params']}")
    log(f"estimated_serialized:{budget['total_mb']:.2f} MB")
    log(f"layers:{args.num_layers} dim:{args.model_dim} inner:{args.inner_dim} "
        f"state:{args.state_dim} conv:{args.conv_width} dt_rank:{args.effective_dt_rank} "
        f"mlp_mult:{args.mlp_mult} group_size:{args.group_size}")
    log(f"es_pop:{args.pop_size} elite_frac:{args.elite_frac} "
        f"init_flip_rate:{args.init_flip_rate} crossover:{args.crossover_rate}")
    log(f"gradient_steps_per_gen:{args.gradient_steps_per_gen} "
        f"continuous_lr:{args.continuous_lr} embed_lr:{args.embed_lr}")

    # Metal scan: always use mode=1 (forward-only) for ES eval,
    # mode=2 (forward+backward) for gradient steps.
    # Mode switching is handled in the training loop.
    use_metal_scan = USE_METAL_SCAN > 0
    if use_metal_scan:
        patch_model_for_metal_scan(model, mode=1)  # start in eval mode

    # Initialize ES
    binary_weights = extract_binary_weights(model)
    es_config = ESConfig(
        pop_size=args.pop_size,
        elite_frac=args.elite_frac,
        init_flip_rate=args.init_flip_rate,
        min_flip_rate=args.min_flip_rate,
        flip_rate_decay=args.flip_rate_decay,
        crossover_rate=args.crossover_rate,
        eval_tokens=args.es_eval_tokens,
        gradient_steps_per_gen=args.gradient_steps_per_gen,
    )
    es = BinaryES(es_config, binary_weights)

    # Batched population evaluator
    evaluator = BatchedPopulationEvaluator(model, group_size=EVAL_GROUP_SIZE)

    # Continuous optimizer
    cont_opt = ContinuousOptimizer(model, args)

    # Loss and grad for continuous params.
    # When using Metal scan (mode=2 for grad), we skip mx.compile because
    # custom Metal kernels with mx.custom_function aren't compatible with it.
    # The Metal kernel itself is fast enough to not need compile tracing.
    loss_and_grad = nn.value_and_grad(model, _make_loss_fn(model))

    # Val eval config
    val_batch_seqs = args.val_batch_size // args.train_seq_len

    scan_mode = {0: "python", 1: "metal_fwd_only", 2: "metal_fwd_bwd"}[USE_METAL_SCAN]
    log(f"scan_mode:{scan_mode} eval_group_size:{EVAL_GROUP_SIZE}")
    log(f"Starting hybrid ES + gradient training for {args.es_generations} generations")
    t0 = time.time()

    for gen in range(args.es_generations):
        gen_t0 = time.time()
        elapsed = gen_t0 - t0

        # Check wallclock limit
        if args.max_wallclock_seconds > 0 and elapsed > args.max_wallclock_seconds:
            log(f"Wallclock limit reached at generation {gen}")
            break

        # ---- Phase A: Evaluate all candidates (batched) ----
        # Metal scan mode=1 (forward-only, fast, no grad needed)
        if use_metal_scan:
            patch_model_for_metal_scan(model, mode=1)

        eval_x, eval_y = train_loader.next_batch(args.es_eval_tokens, args.train_seq_len)

        candidates = [es.get_candidate(i) for i in range(args.pop_size)]
        fitnesses = evaluator.evaluate_population(candidates, eval_x, eval_y)

        es.set_fitnesses(fitnesses)

        # ---- Phase B: ES selection + new population ----
        best_binary = es.step()

        # Install best candidate into model
        inject_binary_weights(model, best_binary)

        # ---- Phase C: Gradient steps on continuous params ----
        # Metal scan mode=2 (forward+backward) for gradients through A_log/dt.
        if use_metal_scan and args.gradient_steps_per_gen > 0:
            patch_model_for_metal_scan(model, mode=2)

        for gstep in range(args.gradient_steps_per_gen):
            grad_x, grad_y = train_loader.next_batch(
                args.gradient_batch_tokens, args.train_seq_len)
            loss, grads = loss_and_grad(grad_x, grad_y)
            mx.eval(loss, grads)
            cont_opt.step(model, grads)

        gen_time = time.time() - gen_t0

        # ---- Logging ----
        if gen % args.train_log_every == 0 or gen == args.es_generations - 1:
            log(f"gen:{gen} {es.log_state()} "
                f"best_loss:{min(fitnesses):.4f} "
                f"mean_loss:{sum(fitnesses)/len(fitnesses):.4f} "
                f"gen_time:{gen_time:.1f}s "
                f"elapsed:{elapsed:.0f}s")

        # ---- Validation ----
        if args.val_loss_every > 0 and (gen % args.val_loss_every == 0 or gen == args.es_generations - 1):
            if use_metal_scan:
                patch_model_for_metal_scan(model, mode=1)
            val_loss, val_bpb = eval_val(
                model, val_tokens, args.train_seq_len, val_batch_seqs,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log(f"val gen:{gen} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # ---- Final eval ----
    if use_metal_scan:
        patch_model_for_metal_scan(model, mode=1)
    total_time = time.time() - t0
    val_loss, val_bpb = eval_val(
        model, val_tokens, args.train_seq_len, val_batch_seqs,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log(f"FINAL val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
        f"total_time:{total_time:.1f}s generations:{es.state.generation}")

    # ---- Save model ----
    save_path = out_dir / f"{args.run_id}_model.pkl"
    state = dict(tree_flatten(model.parameters()))
    # Convert binary weights to packed bits for space efficiency
    save_dict = {}
    for name, param in state.items():
        if _is_binary_weight(name):
            # Pack to 1-bit signs + per-group FP16 scales
            w = param.astype(mx.float32)
            signs = mx.where(w >= 0, mx.array(1, dtype=mx.int8),
                             mx.array(-1, dtype=mx.int8))
            flat_signs = ((signs.reshape(-1).astype(mx.int32) + 1) // 2)
            # Per-group scales
            group_size = 64
            out_dim = w.shape[0]
            padded_in = w.shape[1]
            num_groups = padded_in // group_size
            w_grouped = mx.abs(w).reshape(out_dim, num_groups, group_size)
            scales = mx.mean(w_grouped, axis=2)
            save_dict[name] = {
                'type': 'binary_packed',
                'bits': np.packbits(np.array(flat_signs, dtype=np.uint8)),
                'scales': np.array(scales.astype(mx.float16)),
                'shape': list(param.shape),
                'numel': param.size,
            }
        else:
            save_dict[name] = {
                'type': 'fp',
                'data': np.array(param),
            }

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    file_size = save_path.stat().st_size
    log(f"Model saved to {save_path} ({file_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
