# Spectral Hermite Trust Region: Experimental Log

Companion to `spectral_hermite_optimizer_proposal.md` and `prior_work_synthesis.md`.

## Headline result

**Phase 1 (1D Adam-direction line search) and Phase 2 (2D + SVD-extracted off-axis direction)
produce static-probe signal but do not translate to wallclock or NFE wins
in-loop on a 540k-parameter transformer.** Adam (small-batch, no line search) dominates
every comparison we ran, often by 10–50× in NFE. We characterize the failure modes
cleanly and identify Phase 3 (Krylov / rolling-subspace / multi-direction search) as the
natural next step.

## Setup

- **Model:** TinyGPT, 540,672 parameters. 3 layers, dim=128, heads=4, FFN_mult=2,
  vocab=1024, seq=128, tied embeddings, RMSNorm, GELU MLP. (`model.py`)
- **Data:** FineWeb 10B SP-1024 (parent project's tokenizer + shards). (`data.py`)
- **Framework:** MLX on M4 Max.
- **Optimizer baseline:** plain Adam (`lr=3e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`).

## Files

```
chebyshev.py          1D Lobatto/Gauss nodes, plain + Hermite fit, decay, minimization
chebyshev_2d.py       2D tensor-product Chebyshev fit, evaluate, minimize on rectangle
model.py              TinyGPT
data.py               FineWeb shard reader, fixed-batch helper, validation tokens
perturb.py            tree arithmetic, direction extraction, ray evaluation, macro-batch
perturb_2d.py         orthogonal projection, d2 extraction (SVD, atmin, krylov, random)
probe_static.py       Experiment 1A: 1D fit-then-query at multiple checkpoints
probe_wide.py         1D wide-range fit accuracy across 100/1000/10000 Adam-step distances
probe_static_2d.py    Experiment 2A: 2D fit, multiple d1 + d2 sources
probe_inloop.py       In-loop training: adam, spectral, spectral2d, spectral2d_cheap, armijo
plot_*.py             Plotting helpers
results/              JSON + PNG outputs from all probe runs
```

## Phase 1: 1D Spectral Line Search

### 1A: Fit-then-query characterization

For each (training-stage, direction) pair, sample a fixed probe batch's loss + dphi
at N=8 Lobatto nodes on `[0, eta_max]`, then evaluate predicted-vs-truth at K=25
random query points on `[-0.5·eta_max, 1.5·eta_max]`.

**Findings (step=1000, Adam-direction):**

- Hermite-Chebyshev fit captures loss across `[0, eta_max]` to ~6×10⁻⁴ RMSE
  inside the fit interval — essentially perfect.
- Outside the fit interval, both plain and Hermite fits diverge rapidly (Runge
  phenomenon). Hermite (degree 2N−1) is strictly worse on extrapolation than
  plain (degree N−1).
- The eta of the loss minimum is bracketed within ~1 Adam-step of natural-LR-Adam-step distance.

### 1B: Wide-range probe (single direction, varying interval width)

Same checkpoint, Adam direction, fit interval scaled to {100, 1000, 10000} Adam-step
equivalents:

| eta_far | loss range | N=8 plain RMSE | N=32 plain RMSE |
|---|---|---|---|
| 100 Adam-steps | 4.55 → 9.52 | **6×10⁻⁴** | 5×10⁻⁵ |
| 1000 Adam-steps | 4.56 → 79.5 | 0.124 | 0.007 |
| 10000 Adam-steps | 4.60 → 740 | 0.329 | 0.017 |

**Implication:** Loss along the Adam direction is **monotone-near-quadratic across
thousands of Adam-step distances** at this checkpoint. A degree-7 polynomial
captures it to 4–5 decimal places of relative accuracy. Whatever spectral methods
might find on the Adam direction, it is not "interesting structure beyond a quadratic
local model" — Adam's diagonal preconditioner is already approximately the right
local model.

### 1C: In-loop comparison (probe_inloop.py)

We tested four batch policies for the line search:

| variant | target val | NFE multiplier vs Adam | failure mode |
|---|---|---|---|
| Same batch as gradient (cap=50 nat-steps) | 6.19 | failed | over-fits, pegs cap each step |
| Same batch + cap=10 | 5.62 | ~10× | over-fits, pegs cap each step |
| Separate batch=128 (4× larger) | 4.82 | ~21× | underfits — eta collapses to 0 |
| Separate batch + e1 floor at natural step | ~5.4 | ~10× | partial fix; eta1 still pegs floor |

**Mechanism:** with the same batch, the polynomial finds a per-batch minimum at
unrealistically large eta (mini-batch overfit). With a separate batch, the gradient
batch's d1 is not a descent direction on the search batch's loss landscape at
meaningful magnitudes (cross-batch noise), and eta collapses to 0.

## Phase 2: 2D Spectral Trust Region

### 2A: Static probe at step=1000

For each d2 candidate (SVD across the d1 ray, gradient at 1D min, random
orthogonal, Krylov gradient-difference), fit a 2D Chebyshev tensor polynomial on
a small (~2 × 2 Adam-step) rectangle and find the 2D minimum.

**Adam-direction d1, narrow plane:**

| d2 source | 1D min L | 2D min L | extra ΔL |
|---|---|---|---|
| SVD orthogonal-grads | 4.584 | **4.514** | **−0.070** |
| Random orthogonal | 4.584 | 4.584 | 0.000 |
| Krylov | 4.584 | 4.565 | −0.019 |

**Raw-gradient d1, narrow plane (better!):**

| d2 source | 1D min L | 2D min L | total ΔL from L0=4.60 |
|---|---|---|---|
| SVD | 4.558 | 4.514 | −0.089 |
| **Gradient at 1D min** ("atmin") | 4.558 | **4.508** | **−0.094** |

The cheap variant (raw-grad d1 + grad-at-min d2 + plain Chebyshev) actually
**outperforms** the expensive variant (Adam-direction d1 + SVD d2 + Hermite) and
saves 5 backward passes per step.

### 2B: In-loop spectral2d / spectral2d_cheap

Five batch policies tested, all from cold-start (random init):

| variant | val at 30 outer | NFE | NFE multiplier vs Adam-128 |
|---|---|---|---|
| Adam-128 baseline | ~4.5 (extrapolated) | 4011 (at step=2000) | 1× |
| spec2d separate batch + e1 floor | ~5.4 | 20011 | 10× |
| spec2d same-batch=128 | 5.30 (at step=500) | 20011 | 50× |
| spec2d-macro inner=1 (524k tokens, 4× std) | 5.99 (stuck) | 34571 | failed |
| spec2d-macro inner=3 | 5.98 (stuck) | 103691 | failed |
| spec2d_cheap-macro | 6.10 (stuck) | 31691 | failed |

**The "random-init plateau" issue:** all macro-batch variants got stuck at val~6.0,
which is the unigram-entropy plateau (`H(p_token) ≈ 6.0`). Large batches lose the
noise-driven plateau escape mechanism that small-batch SGD relies on, so any
macro-batch optimizer (including Adam-macro) gets stuck on this shelf for many
outer iterations. The random-init plateau is a property of training small LMs from
scratch with large batches; it confounds the comparison of in-loop spec2d variants.

### 2C: Warm-start comparison

Train Adam-128 for 200 steps to escape the plateau (val=5.52), then run continued
methods from that checkpoint:

| method | val at end | total NFE | Δval from warmup |
|---|---|---|---|
| Continued Adam-128 (1000 more steps) | **4.49** | 2,414 | **−1.03** |
| Continued Adam-macro (30 outer) | 5.30 | 2,334 | −0.22 |
| spec2d_cheap-macro (30 outer) | ~5.50 | 29,982 | **−0.02** |

**spec2d_cheap-macro from the warmup checkpoint barely improved val at all.**
Continued Adam-128 reaches val=4.49 (12× the loss reduction at 12% the NFE).

Two diagnostic numbers from the warm-start spec2d_cheap log:
- **`d2_norm` averaged 0.11–0.32** in-loop, vs ~0.514 in the static probe at the
  same checkpoint — about 2× weaker. The off-axis direction has much less energy
  in-loop than the fixed-val-batch static probe suggested.
- Val loss **oscillates** rather than monotone-decreases: 5.45, 5.47, 5.52, 5.45,
  5.58, 5.48, 5.69. The optimal 2D move on macro-batch t lands at a different
  point on macro-batch t+1.

### 2D: Wide-plane diagnostic

After concluding spec2d's narrow-plane gain (~0.05–0.09 ΔL/step) is too small to
amortize cost, ran a 50×25 Adam-step plane probe to check whether wider planes
contain deeper basins (the proposal's "long-horizon spectral access").

**Result:** The polynomial fit hallucinated a deep basin (predicted L=3.24) that
**does not exist in ground truth**:

| d2 | predicted L_min | actual L at predicted location | actual ground-truth min |
|---|---|---|---|
| svd | 3.375 | **4.755** | 4.586 |
| **atmin** | **3.242** | **4.737** | 4.539 |
| rand | 4.563 | 4.585 | 4.585 |
| krylov | 4.500 | 4.620 | 4.586 |

With M=8, K=7 nodes covering 22.4×11.2 absolute units, node spacing was ~3
Adam-steps — too coarse for the polynomial to faithfully model the rapidly-varying
wide-range 2D landscape. The fit's claim of ΔL=−1.32 was an interpolation
artifact, not a discovered basin. **True wide-plane ΔL beyond the local 1D min
is only ~0.05** — same order as the narrow probe's clean signal.

**Implication:** With locally-extracted d1, d2 directions, the search plane is
bounded by polynomial-fit fidelity, which scales unfavorably with plane size in 2D.
Going wider needs proportionally more nodes (cost ~M·K), which destroys the cost
advantage. **The structure that does exist locally is small (~0.05 ΔL beyond 1D)
and does not amortize the per-step cost overhead.**

## Negative result diagnosis

Across all variants tested, Adam dominates spec2d on NFE-to-target. The reasons
are well-characterized:

1. **1D Adam-direction loss is too smooth.** Wide-range probe shows polynomial
   fits work perfectly across thousands of Adam-step distances, but the loss is
   monotone-near-quadratic — there is nothing for spectral methods to discover
   beyond what Adam's diagonal preconditioner already captures.
2. **2D off-axis structure exists statically but fails to transfer in-loop.** Each
   macro-batch has its own 2D landscape; the d2 direction extracted on macro-batch
   t doesn't generalize to macro-batch t+1. Static probe's clean signal becomes
   per-batch noise in-loop, and `d2_norm` is ~2× smaller in-loop than static.
3. **Locally-anchored directions can't see far-off structure.** Both d1 and d2
   are extracted at current params. Far from origin in the (d1, d2) plane,
   those directions don't necessarily point at meaningful features. Wide-plane
   probe confirmed no deep far-off basins exist in this locally-constructed plane.
4. **Polynomial fit fidelity vs plane size.** Wider search planes need more nodes
   for accurate fits; cost scales as M·K (or worse with k>2). The product of
   "structure that exists" × "polynomial accuracy across the plane" maxes out at
   modest plane sizes with modest ΔL.
5. **Random-init plateau confounds in-loop comparison.** Large-batch Adam (and
   any large-batch optimizer) gets stuck on the unigram-entropy shelf around
   val=6.0 for many outer iterations because it loses noise-driven plateau escape.
   Small-batch Adam burns through the plateau quickly via gradient noise, before
   spec2d's structural edge could even apply.

## Phase 3: Krylov / rolling-subspace as next step

The natural extension addresses (3) directly. Rather than locally-anchored
directions, maintain a deque of recent gradients and search in the subspace they
span:

```
maintain: deque of last K gradients
at each step:
  1. g_t ← gradient at current params
  2. push g_t to deque
  3. directions = orthogonalize(deque)  # k orthonormal vectors
  4. sample loss on Smolyak grid in span(directions)
  5. fit k-D Chebyshev polynomial
  6. find minimum on hyperrectangle, step
```

**Why this addresses our failure modes:**

- Gradients at *successive trajectory points*, when aligned, capture **direction
  along the loss-landscape level set** — the tangent space of the manifold the
  optimizer has been traversing, rather than just curvature perpendicular to it.
- Per-batch noise averages over the deque rather than corrupting a single
  direction.
- If the loss landscape has wedge/tunnel structure (Fort & Scherlis 2019), this
  subspace approximates the tunnel's tangent space, allowing the optimizer to
  step *along* the tunnel rather than *across* it.

**Connection to existing methods:** L-BFGS uses K=10–20 historical gradient
differences to approximate the inverse Hessian and take a quasi-Newton step.
Phase 3 spec3 would fit a higher-order polynomial in the same Krylov subspace.
The L-BFGS comparison is the natural baseline.

**Costs and effort:**

- Implementation: generalize chebyshev_2d → chebyshev_nd (1 day for k≤4 with
  full tensor; +500 lines for proper Smolyak sparse grids if k≥4 needed).
- Per-step cost: ~30–50 NFE for k=3, ~50–80 for k=4 (Smolyak).
- Realistic effort to a publishable comparison: 2–3 days for prototype, several
  weeks to make it actually competitive.

**Why we'd suggest pivoting to a larger model first:** the wedge/tunnel structure
that Phase 3 is supposed to exploit is more pronounced in larger overparameterized
networks. A 540k-parameter model on FineWeb might simply not have rich enough
manifold structure for Phase 3 to win, regardless of implementation quality.
Repeating Phase 1 + 2 + 3 on a 5–10M-parameter model would be a more informative
test.

## Detailed reproduction commands

```bash
# 1A: 1D static probe at multiple stages
python -m grad_interpolation.probe_static
python -m grad_interpolation.plot_static

# 1B: wide-range 1D probes
python -m grad_interpolation.probe_wide --eta-far-mult 100
python -m grad_interpolation.probe_wide --eta-far-mult 1000
python -m grad_interpolation.probe_wide --eta-far-mult 10000
python -m grad_interpolation.plot_wide

# 2A: narrow-plane 2D probe (Adam d1, all four d2 candidates)
python -m grad_interpolation.probe_static_2d
python -m grad_interpolation.plot_static_2d

# 2A': narrow-plane 2D probe (raw-grad d1)
python -m grad_interpolation.probe_static_2d --d1-source grad

# 2D wide-plane diagnostic
python -m grad_interpolation.probe_static_2d --d1-source grad \
    --eta1-k 50 --eta2-k 25 --m-nodes 8 --k-nodes 7

# 1B Adam baseline
python -m grad_interpolation.probe_inloop --mode adam --steps 2000

# 2B in-loop spec2d_cheap on 524k macro-batch (warm)
python -m grad_interpolation.probe_inloop --mode spectral2d_cheap \
    --train-batch 128 --macro-batch-tokens 524288 \
    --warmup-adam-steps 200 --steps 30
```

## Open questions / loose ends

1. Does the Phase 2 narrow-plane gain (~0.05 ΔL/step) actually translate to in-loop
   on a different problem scale? We saw it fail at 540k params; might work at 5M+.
2. Does Phase 3 (Krylov subspace) capture far-off tunnel structure on this model,
   or is the model too small for tunnels to exist?
3. How does L-BFGS at the same compute budget compare? It's the obvious baseline
   for Phase 3 and we never measured it.
4. The cheap variant's `d2_norm < 1e-3` fallback to 1D rarely fired, meaning the
   off-axis signal was always present at *some* level. But its decline from
   static (~0.5) to in-loop (~0.2) was unexplained — is it batch noise, or is it
   real degradation as the optimizer explores parameter space?
