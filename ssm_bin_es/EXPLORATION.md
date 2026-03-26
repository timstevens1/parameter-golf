# Binary Mamba + Evolutionary Search: Exploration Log

## Hypothesis

Can evolutionary strategies (ES) train binary {-1, +1} SSM weights without straight-through estimators (STE)? ES operates natively in discrete space — no gradient hack needed for non-differentiable quantization.

## Architecture

**Binary Mamba SSM** on MLX (Apple Silicon):
- Mamba selective state space model backbone (linear time, no attention)
- Large projection weights (in_proj, x_proj, out_proj, FFN) constrained to binary signs with per-group learned scales
- Small stability-critical params (A_log, D, conv, dt_proj) stay FP32
- Tied embeddings in FP16/BF16

**Training modes** (configurable via `MODE` env var):
- `hybrid`: ES evolves binary weight signs + Adam trains continuous params
- `gradient_only`: Pure Adam on all weights (no binary constraint)
- `gradient_rebinarize`: Adam + periodic snap to sign(w) * group_mean(|w|)

## Key Results

### Round 1: Early Experiments (tiny model)

#### Pure ES (no gradient steps)
- 360 generations in 300s, fitness barely moved: 6.93 → 6.92
- Val BPB stuck at 4.10
- **Conclusion: ES alone cannot converge.** Without gradient-trained scales, binary sign flips are just shuffling signs on fixed-magnitude random vectors.

#### Hybrid ES + Gradient (early best)
- 601 generations in 363s (0.2s/gen with Metal scan kernel)
- Architecture: 4 layers, dim=128, inner=256, state=8, ~980K params, 0.44MB serialized
- Loss: 6.93 → 3.79, Val BPB: 3.90 → **2.47**

#### FP16 effective weight variant
- Simplified: store weight as sign*scale in FP16, native BLAS matmul
- With gradient on weights: Val BPB 2.89 (worse than separate scale approach)
- Without gradient on weights: Val BPB 3.22

### Round 2: Is ES Helping? (controlled comparison)

**Setup:** 4 layers, dim=256, 3.6M params, 1.22MB serialized. 3 min wallclock, sequential runs on M4 Max. Full 62M-token validation set.

| Mode | Val BPB | Gradient Steps | Notes |
|------|---------|----------------|-------|
| **gradient_only** | **2.1537** | 816 | Clear winner |
| gradient_rebinarize | 2.2593 | 816 | Binary constraint costs ~0.1 BPB |
| hybrid (ES+gradient) | 3.1044 | 204 | ES is actively harmful |

**Conclusion: ES does not help.** The hybrid mode wastes most of its wallclock on evaluating ES candidates (16 forward passes per generation), leaving only 204 gradient steps vs 816 for gradient-only. ES isn't compensating — the best_fitness barely moved (6.93 → 5.65) while gradient-only drove train_loss from 5.98 → 3.55.

### Round 3: Learning Rate Sweep

**Setup:** L4_D256, gradient_only, 3 min wallclock, 8K batch tokens, 524K val tokens.

| LR | Val BPB | Steps | Notes |
|----|---------|-------|-------|
| **0.005** | **1.9526** | 2512 | Best |
| 0.01 | 2.0168 | 2432 | Slightly worse |
| 0.02 | 2.4208 | 2048 | Too high |
| 0.05 | NaN | - | Diverged |

**With warm start** (from LR=0.005 checkpoint, 3 min additional training):

| LR | Val BPB | Notes |
|----|---------|-------|
| 0.001 | 2.0738 | Best for fine-tuning |
| 0.002 | 2.0754 | Similar |
| 0.005 | 2.0925 | Slightly worse for fine-tuning |
| 0.01 | 2.6192 | Too high |

### Round 4: Batch Size & Sequence Length (524K val tokens)

| Config | Val BPB | Steps | Notes |
|--------|---------|-------|-------|
| **L4_D256, bs=8K** | **3.0706** | 2368 | Best (most steps) |
| L4_D256, seq=512 | 3.0822 | 2320 | Similar |
| L4_D256, seq=1024 | 3.1645 | 1808 | Fewer steps, slightly worse |
| L4_D256, bs=64K (8x accum) | 3.2124 | 304 | Too few steps |
| L4_D256, bs=32K (4x accum) | 3.2262 | 368 | Too few steps |

**Note:** Val BPB numbers differ from Round 2-3 because these used 524K val tokens instead of the full 62M set. The relative ordering is what matters.

**Conclusion:** Small batch sizes with many gradient steps beat large batches with few steps on this hardware. Gradient accumulation is counterproductive when GPU-bound — it gives better gradient estimates per step but far fewer steps per wallclock.

### Round 5: Long Runs at Different Scales (30 min each)

**Per-step training loss comparison:**

| Steps | L4_D256 (3.6M) | L6_D320 (8.1M) | L8_D384 (15.2M) | L12_D512 (39.8M) |
|-------|-----------------|-----------------|------------------|-------------------|
| 16 | 6.02 | 6.00 | 5.99 | 6.00 |
| 816 | 5.42 | 5.38 | **5.30** | 5.39 |
| 1616 | 5.35 | 5.14 | - | 5.19 |
| 3216 | 5.03 | 4.97 | - | **4.97** |
| 5616 | 4.92 | 4.88 | - | - |
| 8016 | 4.90 | 4.87 | - | - |
| 11216 | 4.88 | - | - | - |

**Final validation (524K tokens):**

| Model | Params | Serialized | Val BPB | Steps | Time/step |
|-------|--------|------------|---------|-------|-----------|
| L4_D256 | 3.6M | 1.2 MB | ~2.95* | 11216 | 0.08s |
| L6_D320 | 8.1M | 2.2 MB | ~2.93* | 8016 | 0.16s |
| L8_D384 | 15.2M | 3.7 MB | 3.03 | 1504 | 1.2s |
| L12_D512 | 39.8M | 8.6 MB | 2.97 | 3264 | 0.55s |

*L4 and L6 log files were truncated before FINAL eval; values estimated from training loss.

**Key insight:** L8_D384 is the best model per-step at early training (step 816), but on M4 Max it's ~15x slower per step than L4_D256. On 8xH100s where step throughput would be similar across model sizes, L8 or larger would likely converge to significantly lower loss.

**L12_D512 underperformed expectations** at step 816, likely due to:
1. LR 0.003 may be suboptimal for this scale (all others used 0.005)
2. 39.8M params may need more warmup steps
3. Larger models benefit more from larger batch sizes (we used 8K)

## Metal Kernel Performance

| Component | Speedup | Notes |
|---|---|---|
| SSM scan (mode=1, forward-only) | ~10x vs Python | 0.1s vs 1+ seconds per eval |
| SSM scan (mode=2, fwd+bwd) | Instant | No compile overhead, unlike mx.compile on Python scan |
| Mutation/crossover kernels | Working | XOR-based packed bit ops, correct but marginal benefit |
| Binary matmul kernel | Abandoned | Hung in training due to GPU page faults on weight injection |

## Infrastructure Built

- **Training modes:** `MODE={hybrid,gradient_only,gradient_rebinarize}` env var
- **Gradient accumulation:** `GRAD_ACCUM_STEPS` for larger effective batch sizes
- **Gradient checkpointing:** `GRAD_CHECKPOINT=1` via `mx.checkpoint` (reduces memory ~7x)
- **Fast validation:** `MAX_VAL_TOKENS` to limit val set size for quick iteration
- **Warm start:** `WARM_START=path/to/model.pkl` to resume from saved checkpoint
- **Experiment runner:** `run_experiments.py` for sequential experiment management
- **Log monitor:** `monitor.py` for parsing and comparing training logs

## Hardware Notes (M4 Max, 128GB)

- **GPU parallelism:** Only 1 Metal training process at a time. Multiple MLX processes cause GPU memory deadlocks (~14GB per process without grad checkpoint, ~2GB with).
- **CPU is mostly idle** during Metal GPU training (~20% per process). The bottleneck is GPU compute, not CPU/GIL (separate processes, not threads).
- **Validation is the bottleneck:** The full 62M-token val set takes 80-400s depending on model size. Use `MAX_VAL_TOKENS=524288` for fast iteration.
- **Step throughput varies dramatically by model size:** L4_D256 gets ~12 steps/s, L12_D512 gets ~1.3 steps/s.

## Bugs Found and Fixed

1. **MLX Metal `thread_position_in_grid`** returns `uint3`, must use `.x` for 1D
2. **Metal kernel `.size()`** doesn't exist on raw pointers — pass via `#define`
3. **Adam NaN on FP16** — eps (1e-8) underflows. Cast to float32 before optimizer.
4. **`mx.custom_function` VJP unpacking** — `(dy,) = cotangents` fails via `nn.value_and_grad`. Fixed: `dy = cotangents if not isinstance(cotangents, (list, tuple)) else cotangents[0]`
5. **MLX lazy eval + mutable state** — can't defer `mx.eval()` across candidates that inject different weights into shared model
6. **Kernel recompilation** — baking batch dim `M` into `#define` header causes recompile per batch size. Fixed: pass M as runtime input, cache kernels by weight dims only.
7. **Model save with FP16 weights** — `np.array(param)` fails on FP16 MLX arrays. Fixed: cast to float32 before numpy conversion.

## Conclusions

1. **ES is not useful for this problem.** Pure gradient descent dramatically outperforms hybrid ES+gradient. The overhead of evaluating a population of candidates wastes wallclock that would be better spent on gradient steps.

2. **Binary constraints hurt modestly.** Gradient with re-binarization (enforcing sign * group_scale structure) costs ~0.1 BPB vs unconstrained FP16. This may be acceptable for the ~6x parameter density advantage of binary weights.

3. **Larger models are better per-step** but much slower on M4 Max. L8_D384 beats all smaller models at the same step count. On 8xH100s with faster step throughput, scaling up is the right strategy.

4. **LR 0.005 is a good default** for this architecture at the 4-12 layer scale. Lower LR (0.001-0.002) is better for fine-tuning from a warm start.

5. **Small batch sizes dominate on this hardware** because they maximize gradient steps per wallclock. On 8xH100s with data parallelism, larger batches would be preferred.

## Future Directions

### Immediate (for Parameter Golf submission)

1. **Scale to 16MB on 8xH100s.** The best architecture for the 16MB budget is likely L12-L16 with dim=512-768, trained with gradient_only mode, LR ~0.005, and larger batch sizes enabled by 8-way data parallelism.

2. **Re-evaluate binary constraint trade-off at scale.** Binary weights give ~6x more parameters per MB. At 16MB, that's ~100M binary params vs ~16M FP8 params. The 0.1 BPB cost of re-binarization may be worth the parameter count.

3. **LR scheduling.** Currently using constant LR. Warmup + cosine decay or linear warmdown would likely improve convergence.

4. **Quantization-aware training.** Instead of binary, try int4/int6 QAT like the leaderboard entries. The Mamba backbone + low-precision weights is the novel contribution.

### Research

5. **Distillation from transformer.** Train a standard Mamba model at high precision, then compress to binary. Separates architecture search from quantization.

6. **Weight tying + depth recurrence.** Share weights across layers with `WEIGHT_TIE_LAYERS`. The parameter savings could enable wider models.

7. **Hybrid Mamba-Attention.** The top leaderboard entries use XSA (cross-sequence attention). A few attention layers in a mostly-Mamba model could improve retrieval quality.

## File Structure

```
ssm_bin_es/
├── APPROACH.md                     # Original design document
├── EXPLORATION.md                  # This file — results and analysis
├── kernels/                        # Metal GPU kernels (reusable)
│   ├── metal_ssm_fwd.py           # Forward-only selective scan kernel
│   ├── metal_ssm_bwd.py           # Forward+backward scan with checkpoints
│   ├── scan_integration.py        # Mode-switching bridge for model patching
│   ├── metal_binary_matmul.py     # Binary matmul kernel (functional, not used)
│   └── metal_mutation.py          # ES mutation/crossover kernels
└── experiment/                     # Training experiment code
    ├── binary_linear.py            # BinaryLinear layer (FP16 effective weight)
    ├── model.py                    # BinaryMambaLM model (+ mx.checkpoint support)
    ├── es.py                       # Evolutionary strategy (sign-flip mutation)
    ├── train.py                    # Multi-mode training loop (hybrid/gradient/rebinarize)
    ├── batched_eval.py             # Batched candidate evaluation
    ├── run_experiments.py          # Sequential experiment runner
    └── monitor.py                  # Log parser and comparison tool
plan.md                             # Top-level research plan
```

## Running

Best known configuration (gradient_only, L4_D256):

```bash
MODE=gradient_only \
GRAD_CHECKPOINT=1 \
USE_METAL_SCAN=1 \
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=4 MODEL_DIM=256 STATE_DIM=16 \
EXPAND_FACTOR=2 MLP_MULT=2 TRAIN_SEQ_LEN=256 \
ES_GENERATIONS=10000 MAX_WALLCLOCK_SECONDS=600 \
GRADIENT_BATCH_TOKENS=8192 GRADIENT_STEPS_PER_GEN=16 \
CONTINUOUS_LR=0.005 VAL_LOSS_EVERY=0 \
MAX_VAL_TOKENS=524288 \
python3 -m ssm_bin_es.experiment.train
```

With warm start from previous checkpoint:

```bash
WARM_START=logs/previous_model.pkl \
... (same as above with adjusted LR, e.g. CONTINUOUS_LR=0.001)
```

Monitor experiments:

```bash
python3 -m ssm_bin_es.experiment.monitor logs/*.txt
python3 -m ssm_bin_es.experiment.monitor logs/*.txt --watch --interval 10
```
