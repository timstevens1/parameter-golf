# Binary Mamba with Evolutionary Search

## Overview

Train a Mamba SSM language model with **binary weights** ({-1, +1} with per-group FP16 scales) using a **hybrid evolutionary strategy + gradient descent** training loop. The binary projections are evolved via bit-flip mutations (no STE needed), while stability-critical continuous parameters (A_log, scales, embeddings) are trained with standard gradient methods.

This eliminates the fundamental tension in gradient-based binary training: non-differentiable quantization requiring straight-through estimators. ES operates natively in the discrete {-1, +1} space.

---

## Architecture

### Binary vs Continuous Parameter Split

| Component | Type | Rationale |
|---|---|---|
| in_proj weight | Binary + scale | Large 2D matmul, bulk of params |
| x_proj weight | Binary + scale | Large 2D matmul |
| out_proj weight | Binary + scale | Large 2D matmul |
| gate/up/down FFN weights | Binary + scale | Large 2D matmuls |
| A_log | FP32 | Controls SSM stability (eigenvalues) |
| D (skip) | FP32 | Small, stability-critical |
| dt_proj_weight | FP32 | Small (inner x dt_rank), controls discretization |
| dt_proj_bias | FP32 | Small |
| conv_weight, conv_bias | FP32 | Small (inner x conv_width) |
| layer_scale, ffn_scale | FP32 | Scalars, control residual weighting |
| tok_emb | FP16 | Embedding lookup, needs precision |
| per-group scales | FP16 | Learned amplitude per group of 64 binary weights |

### BinaryLinear Layer

Forward pass: `y = x @ (binary_weight * scale_per_group).T`

- `binary_weight`: shape (out, in), values in {-1, +1} stored as int8
- `scale`: shape (out, in // group_size), FP16
- At forward time, scales are broadcast across groups to produce effective FP weights
- No latent FP shadow weights needed since ES doesn't compute gradients through these

### SSM Scan

Unchanged from `train_mamba_mlx.py`. The scan operates on the FP outputs of the projections. Metal kernels work as-is.

---

## Training Strategy

### Phase 1: Gradient Warmup (optional, ~200 steps)

Standard gradient training with STE on binary weights to reach a reasonable basin. This gives ES a better starting point than random initialization. Can be skipped if initialization is good enough.

### Phase 2: Hybrid ES + Gradient Descent

Each generation:

1. **Mutate binary weights** across a population of candidates
   - Mutation = stochastic bit flips with adaptive flip rate
   - Elite selection: top-k by fitness (loss on eval batch)
   - Optional crossover: uniform crossover between elites

2. **Train continuous params** on the best candidate
   - Adam on: tok_emb, A_log, D, dt_proj, conv, scales, layer_scale
   - Muon on: any remaining 2D continuous matrices (if any)
   - Run N gradient steps per ES generation

3. **Evaluate fitness**
   - Loss on a small held-out batch (fast proxy for full eval)
   - Periodically run full BPB eval for logging

### Phase 3: Refinement

- Anneal flip rate toward zero
- Increase eval batch size for more accurate fitness signal
- Optional: local search (flip one bit at a time, greedy accept)

### Adaptive Flip Rate

Start with ~1% flip rate (flip 1 in 100 binary weights per mutation). Decay based on fitness improvement:
- If best fitness improves: maintain or slightly reduce rate
- If plateau: temporarily increase rate (exploration)
- Late training: decay to 0.01% for fine-grained search

---

## Parameter Budget (16MB target)

Binary weights at 1 bit + scale overhead:
- 1 bit per weight + 16 bits per group of 64 = 1.25 bits/param effective
- ~100M binary parameters in ~15MB
- ~500K continuous parameters in ~1MB (A_log, D, conv, dt, embeddings)
- **Total: ~100M binary + 500K continuous in 16MB**

Compare to INT8: ~16M params. Binary gives **~6x more parameters** in the same budget.

---

## Population Parallelism

Each candidate in the population shares the same continuous params (only binary weights differ). This means:

- Memory per candidate = binary weights only (~12.5MB for 100M binary params)
- Forward pass can be batched: swap binary weights, run same input batch
- On Apple Silicon with 64GB+ unified memory: population of 32-64 fits easily

---

## SSM Advantage for ES

The SSM recurrence `h(t) = A*h(t-1) + B*x(t)` makes ES more tractable:

1. **Bounded impact**: A is designed to have eigenvalues < 1, so perturbations in projection weights are damped as they propagate through time
2. **Smooth fitness landscape**: small bit-flip mutations produce small loss changes (unlike attention where one flip can restructure the entire pattern)
3. **No backward pass needed for binary weights**: the expensive Metal backward kernel is only needed for the small continuous parameter set

---

## Implementation Plan

1. `binary_linear.py` - BinaryLinear layer with per-group scales
2. `model.py` - BinaryMamba model (adapts MambaBlock/GatedFFN/MambaLM)
3. `es.py` - Evolutionary strategy: population, mutation, selection, crossover
4. `train.py` - Hybrid training loop integrating ES + gradient descent
5. Reuse from parent: data loading, Metal kernels, eval/BPB, tokenizer, quantization/serialization

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ES too slow to converge at 100M params | Hybrid approach: ES only on binary, gradient on continuous. Gradient warmup phase. |
| Binary weights too restrictive for SSM quality | Per-group scales restore amplitude expressivity. SSMs are empirically more robust to quantization than transformers. |
| Population evaluation too expensive | Share continuous params across population. Use small eval batches for fitness. |
| Flip rate tuning sensitive | Adaptive schedule based on fitness improvement rate. |
| Random init too far from good basin | Optional STE warmup phase, or init from a pretrained FP model's sign(W). |
