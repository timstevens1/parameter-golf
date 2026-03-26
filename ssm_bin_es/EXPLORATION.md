# Binary Mamba + Evolutionary Search: Exploration Log

## Hypothesis

Can evolutionary strategies (ES) train binary {-1, +1} SSM weights without straight-through estimators (STE)? ES operates natively in discrete space — no gradient hack needed for non-differentiable quantization.

## Architecture

**Binary Mamba SSM** on MLX (Apple Silicon):
- Mamba selective state space model backbone (linear time, no attention)
- Large projection weights (in_proj, x_proj, out_proj, FFN) constrained to binary signs with per-group learned scales
- Small stability-critical params (A_log, D, conv, dt_proj) stay FP32
- Tied embeddings in FP16/BF16

**Hybrid training loop:**
- ES evolves binary weight signs (bit-flip mutation, uniform crossover, elite selection)
- Adam gradient descent trains continuous params (scales, embeddings, SSM dynamics)
- Metal GPU kernels for SSM scan (forward-only mode=1 for ES eval, forward+backward mode=2 for gradient steps)

## Key Results

### Pure ES (no gradient steps)
- 360 generations in 300s, fitness barely moved: 6.93 → 6.92
- Val BPB stuck at 4.10
- **Conclusion: ES alone cannot converge.** Without gradient-trained scales, binary sign flips are just shuffling signs on fixed-magnitude random vectors.

### Hybrid ES + Gradient (best result)
- 601 generations in 363s (0.2s/gen with Metal scan kernel)
- Architecture: 4 layers, dim=128, inner=256, state=8, ~980K params, 0.44MB serialized
- Loss: 6.93 → 3.79, Val BPB: 3.90 → **2.47**
- Used separate int8 binary_weight + FP16 scale parameters with gradient on scales

### FP16 effective weight variant
- Simplified: store weight as sign*scale in FP16, native BLAS matmul
- With gradient on weights: Val BPB 2.89 (worse than separate scale approach)
- Without gradient on weights: Val BPB 3.22

## Metal Kernel Performance

| Component | Speedup | Notes |
|---|---|---|
| SSM scan (mode=1, forward-only) | ~10x vs Python | 0.1s vs 1+ seconds per eval |
| SSM scan (mode=2, fwd+bwd) | Instant | No compile overhead, unlike mx.compile on Python scan |
| Mutation/crossover kernels | Working | XOR-based packed bit ops, correct but marginal benefit |
| Binary matmul kernel | Abandoned | Hung in training due to GPU page faults on weight injection |

## Bugs Found and Fixed

1. **MLX Metal `thread_position_in_grid`** returns `uint3`, must use `.x` for 1D
2. **Metal kernel `.size()`** doesn't exist on raw pointers — pass via `#define`
3. **Adam NaN on FP16** — eps (1e-8) underflows. Cast to float32 before optimizer.
4. **`mx.custom_function` VJP unpacking** — `(dy,) = cotangents` fails via `nn.value_and_grad`. Fixed: `dy = cotangents if not isinstance(cotangents, (list, tuple)) else cotangents[0]`
5. **MLX lazy eval + mutable state** — can't defer `mx.eval()` across candidates that inject different weights into shared model
6. **Kernel recompilation** — baking batch dim `M` into `#define` header causes recompile per batch size. Fixed: pass M as runtime input, cache kernels by weight dims only.

## Open Questions

### Is ES contributing anything beyond gradient descent?
The gradient steps on scales/magnitudes drive all the learning. ES provides sign exploration, but we haven't demonstrated it outperforms pure STE. Need a fair wall-time-bounded comparison:
- Pure gradient (no ES)
- Pure ES (no gradient) — already shown to fail
- Hybrid (current approach)

### FP16 effective weight vs separate binary + scale
The separate representation (int8 signs + FP16 scales) outperformed the unified FP16 weight (2.47 vs 2.89 BPB). Likely because:
- Scales have their own gradient pathway independent of sign
- Group-wise scale sharing acts as regularization
- Adam state for scales is cleaner (no sign noise)

### Re-binarization after gradient steps
When gradients update the FP16 effective weight, magnitudes within a group diverge and signs can flip from gradient rather than ES. A post-gradient re-binarization step (snap magnitudes to per-group mean, preserve signs) might help. Not yet tested.

## Potential Directions

### 1. Fair wall-time comparison (immediate)
Run pure-gradient vs hybrid with same wallclock budget on same hardware. Settle whether ES adds value.

### 2. STE Binary Mamba (pragmatic)
Drop ES, use STE for binary weights on the Mamba backbone. Ciprian's transformer binary/ternary scripts show this works. The Mamba architecture is the novel contribution, not the training method.

### 3. Scale up (next)
Current results are on a tiny model (0.44MB). The 16MB budget allows ~100M binary params. Need to test at competitive scale with proper hyperparameter tuning.

### 4. Circuit fabric distillation (research)
Train a normal Mamba model, then distill into binary/circuit representation. Separates "finding good weights" from "compressing to binary" — both individually well-understood.

### 5. Hybrid with re-binarization
After each gradient step, snap weights back to binary constraint (per-group uniform magnitude). This preserves the binary structure while allowing gradient to tune scales. Could close the gap between the two representations.

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
    ├── model.py                    # BinaryMambaLM model
    ├── es.py                       # Evolutionary strategy (sign-flip mutation)
    ├── train.py                    # Hybrid ES + gradient training loop
    └── batched_eval.py             # Batched candidate evaluation
plan.md                             # Top-level research plan
```

## Running

From the project root:

```bash
USE_METAL_SCAN=1 \
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
ES_GENERATIONS=5000 POP_SIZE=16 NUM_LAYERS=4 MODEL_DIM=128 \
STATE_DIM=8 EXPAND_FACTOR=2 MLP_MULT=2 TRAIN_SEQ_LEN=128 \
ES_EVAL_TOKENS=4096 GRADIENT_BATCH_TOKENS=4096 \
GRADIENT_STEPS_PER_GEN=2 MAX_WALLCLOCK_SECONDS=300 \
python3 -m ssm_bin_es.experiment.train
```
