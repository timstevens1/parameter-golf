# Spectral Hermite Trust Region: A Prototype Proposal

## Summary

We propose prototyping a new optimizer that combines Chebyshev spectral interpolation with Hermite (function-and-gradient) sampling along descent rays to construct a global polynomial model of the loss within a low-dimensional adaptive subspace. The optimizer replaces the local step-size heuristics of Adam with an empirically-measured optimal step in a subspace whose dimension grows progressively during training. The second and higher search dimensions are bootstrapped for free from gradients computed along the initial 1D ray, making the method competitive in per-step cost while accessing landscape structure local methods cannot see.

The primary baseline is ADAM-SLS (Kenneweg et al., 2024), which already established that Armijo line search along Adam's direction beats tuned Adam with cosine schedule on GPT-2 pretraining and BERT fine-tuning. Our method extends that direction with high-order spectral fitting, Hermite data, and progressive kD subspace extension, targeting regimes where local Armijo sufficiency breaks down.

The method is most likely to outperform both Adam and ADAM-SLS in the regime of small-to-medium models, large batches, and non-convex landscapes with medium-scale structure — which matches the "model golf" context of minimal specialist models. The prototype aims to validate the approach on tractable benchmarks before any scaling considerations.

## Motivation

Modern neural network optimizers (Adam, SGD+momentum) are structurally local: they take one gradient per step and scale it by running statistics of past gradients. This makes them cheap per step but fundamentally unable to access non-local information about the loss landscape — they cannot see ridges to jump over, basins beyond the current one, or large-scale structure along their descent direction.

Recent work (Kenneweg et al., 2024) established that line search along the Adam update direction — rather than the raw gradient — beats tuned Adam on GPT-2 pretraining and BERT fine-tuning. They identify the key failure of naive line search as using the wrong direction: "the Armijo criterion exclusively conducting the line search in the direction of the gradient... when this direction significantly diverges from the actual update direction, as is often the case in large-scale transformer training," the resulting step size estimates are unreliable. Fixing the direction while keeping classical Armijo backtracking gets meaningful gains. Their method, ADAM-SLS, is the correct baseline for this proposal.

Classical numerical analysis offers stronger tools than backtracking for the step-size subproblem: Chebyshev spectral interpolation can build accurate polynomial models of smooth functions from a handful of well-placed samples, and for the specific case of 1D line search along the Adam direction, the problem is genuinely low-dimensional and well-suited to spectral methods. Prior work on polynomial line search exists but uses low-order local fits (quadratic or cubic), not high-order spectral models with long horizons.

Three observations motivate this proposal:

1. **Line search is genuinely 1D.** Updates of the form w − η·d for a fixed direction d reduce loss evaluation to a scalar function of η. Chebyshev interpolation achieves exponential convergence on smooth scalar functions.

2. **Gradients along a ray carry second-order information for free.** The difference g(w + η·d) − g(w) ≈ η·H·d is a finite-difference Hessian-vector product. Evaluating gradients at the loss-sampling nodes yields Hessian-vector info without any dedicated curvature computation.

3. **Orthogonal gradient components give a principled second direction.** The component of g(w + η·d) orthogonal to d tells you "having moved this far along d, what direction should you go next?" Collecting these across Chebyshev nodes and taking the top singular vector yields a data-driven second direction that is adaptive to where the 1D step is actually heading. Recent work (Song et al., 2024; Wen et al., 2024) shows that while gradients concentrate in the top Hessian eigenspace (Gur-Ari et al., 2018), the actual learning signal lies in the orthogonal "bulk" subspace — so the method's Phase 2 direction extraction needs to verify it is finding bulk-space content, not just top-space magnitude.

Together, these observations enable a progressive-dimension optimizer that starts as a 1D spectral line search and grows into a kD trust-region method as training progresses, with the higher dimensions bootstrapped from the 1D ray's own gradient evaluations.

## Method

### Phase 1 (Early Training): 1D Spectral Hermite Line Search

At each training step:

1. Compute a descent direction d₁ (e.g., Adam's update direction) on a large batch B.
2. At N Chebyshev nodes η⁽ⁱ⁾ on an adaptive interval [0, η_max], evaluate:
   - Loss: φ(η⁽ⁱ⁾) = L_B(w − η⁽ⁱ⁾·d₁) via forward pass.
   - Gradient: g⁽ⁱ⁾ = ∇L_B at that point, via backward pass.
3. Compute φ'(η⁽ⁱ⁾) = −g⁽ⁱ⁾ · d₁ (directional derivative along ray).
4. Fit a Hermite-Chebyshev polynomial to the 2N data points (N function values + N derivatives). This yields a degree-(2N−1) polynomial from N nodes.
5. Analytically minimize the polynomial on the interval. Take the step.
6. Adapt interval [0, η_max] and node count N for next step based on Chebyshev coefficient decay.

Initial parameter choices: N = 8 nodes, interval adapted with η_max starting at ~4× Adam's default learning rate. Large batches (≥4× standard) to control noise floor.

### Phase 2 (Mid Training): 2D Spectral Trust Region with Bootstrapped Direction

Triggered when Phase 1's polynomial coefficient tail indicates the 1D landscape is well-resolved (high-order coefficients small relative to quadratic).

1. Perform Phase 1's sampling (loss + gradient at N Chebyshev nodes on the d₁ ray).
2. Extract orthogonal gradient components: g_⊥⁽ⁱ⁾ = g⁽ⁱ⁾ − (g⁽ⁱ⁾ · d₁/‖d₁‖²)·d₁.
3. Compute the top singular vector of the matrix [g_⊥⁽¹⁾, ..., g_⊥⁽ᴺ⁾]. Take this as d₂.
4. **Bulk-space verification.** Compute a few Hessian-vector products Hv via autograd at w_n and check that d₂ has substantial projection onto the complement of the top-k Hessian eigenspace (where k ≈ number of classes, per Gur-Ari et al. 2018). Song et al. (2024) and related work show the actual learning signal concentrates in this "bulk" subspace; if d₂ is dominated by top-eigenspace content, it will contribute magnitude but little loss reduction. If the projection fails this check, fall back to d₂ = orthogonalized raw gradient or skip the 2D extension for this step.
5. Sample the 2D plane spanned by (d₁, d₂) on a small additional Chebyshev grid (e.g., 4×3 supplementary points off the ray), reusing the ray samples as one edge of the grid.
6. Fit a 2D Chebyshev polynomial, find the minimum of the model, step.

The second direction d₂ costs essentially nothing beyond Phase 1 (a single SVD of small matrices, plus a few Hessian-vector products for the verification step). Additional sample cost is roughly 12 extra forward passes per step.

### Phase 3 (Late Training): kD Subspace with Rolling Basis

Triggered when Phase 2's cross-term coefficients indicate meaningful 2D structure is being resolved and higher-order directions would help.

Maintain a rolling subspace of dimension k = 3–5, with basis derived from:
- The current Adam direction (d₁).
- Top singular vectors of orthogonal gradient components from recent steps.
- Optionally: gradient differences across the last few iterations (Krylov-like).

Sample the kD rectangle using a sparse grid (Smolyak construction) rather than full tensor product, keeping sample count at O(N · log^(k−1) N) rather than N^k. For k=4 and N=6, this is ~50 samples instead of 1296.

Fit a kD polynomial via sparse-grid Chebyshev interpolation, minimize, step.

### Adaptive Phase Transitions

Transitions between phases are driven by spectral coefficient analysis:

- **Phase 1 → Phase 2:** When the 1D fit's coefficients beyond degree 3 fall below a threshold (signal well-resolved), indicating headroom to add dimensions.
- **Phase 2 → Phase 3:** When 2D cross-term coefficients become significant relative to marginal coefficients (genuine 2D structure present).
- **Phase retreat:** If coefficient decay is slow or batch-overfitting is detected (held-out batch loss diverges from fitted loss), revert to lower-dimensional phase.

## Evaluation Plan

### Experiment 1: Empirical Structure of Loss Along Gradient Rays

Before implementing the optimizer proper, characterize the structure being exploited:

- Train a small transformer (~1M params) and a small MLP on standard tasks.
- At fixed intervals during standard Adam training, densely sample φ(η) = L(w − η·d) along both the raw gradient direction and the Adam direction using 200+ points and large batches.
- Measure:
  - Number of local minima on the ray.
  - Smoothness, quantified by Chebyshev coefficient decay rate (this directly tells us what polynomial order spectral methods will need).
  - Long-range vs. short-range structure (is the best η near the local quadratic minimum, or far past it?).
  - How all the above change over training (supports or refutes the progressive-dimension schedule premise).
- Also measure the "chaotic vs. smooth" character of the landscape in the style of Li et al. (2018), using filter-normalized 2D visualizations, to assess whether high-order polynomial fits are viable for the architectures under test. Deep non-smooth landscapes would defeat the method's core premise.

Output: characterize what N and interval width are needed for faithful spectral representation, and confirm there is non-trivial medium-range structure worth exploiting beyond what ADAM-SLS already captures.

This experiment is cheap (one forward pass per sample, no additional training) and decisive: if φ is nearly quadratic everywhere and ADAM-SLS's Armijo criterion already finds close-to-optimal step sizes, the method's Phase 1 reduces to an expensive reimplementation of an existing method.

### Experiment 2: Phase 1 Isolated

Implement and benchmark the 1D Spectral Hermite Line Search alone:

- Task 1: MNIST with small MLP, well-understood landscape.
- Task 2: Tiny transformer (~5M params) on a language modeling task.
- Task 3: Small TRM-style model on algorithmic task (matches the model-golf context).

Baselines, in order of importance:

1. **ADAM-SLS (Kenneweg et al., 2024)** — primary baseline. Armijo line search along the Adam direction. This is what we need to beat to justify the spectral/Hermite complexity.
2. Adam with cosine schedule (well-tuned).
3. SGD+momentum (well-tuned).
4. Classical cubic line search along gradient direction.
5. (Optional) Adam with simple quadratic line search, to isolate the benefit of high-order vs. low-order fitting.

Metrics: final loss, convergence rate, step-size adaptation behavior, per-step wall clock, total wall clock to target loss. Also track the distribution of chosen step sizes over training — if our method picks η values consistently different from what Armijo would accept, that's diagnostic evidence the spectral model is finding something Armijo can't see.

Success criterion: match or beat ADAM-SLS in training loss at comparable wall-clock, demonstrating that high-order spectral fitting provides useful step sizes beyond what Armijo-based line search achieves.

### Experiment 3: Phase 2 Isolated

Add the 2D extension with bootstrapped d₂. Same tasks, same baselines plus Phase 1.

Additional diagnostic: measure the projection of the discovered d₂ onto (a) the top-k Hessian eigenspace and (b) its complement (the bulk subspace), using power iteration or Lanczos for the top eigenvectors. If d₂ is dominated by top-eigenspace content, the method is finding magnitude but not learning signal, and we should expect no loss improvement over Phase 1. If d₂ carries substantial bulk content, we should see improvement — and this gives a mechanistic explanation regardless of outcome.

Success criterion: meaningful loss improvement over Phase 1 alone on at least one task, validating that the 2D extension accesses useful structure. If Phase 2 matches Phase 1 exactly *and* the d₂ diagnostic shows the extension is mostly in the dom subspace, we've learned something specific and the method should stay 1D.

### Experiment 4: Full Progressive Schedule

Implement automatic phase transitions. Test on Tasks 2 and 3.

Success criterion: matches or exceeds the best isolated phase, without manual phase scheduling. Demonstrates the coefficient-decay heuristics correctly identify when to change dimension.

### Experiment 5: Sensitivity and Ablations

- Effect of batch size on optimal N (noise floor analysis).
- Effect of interval width on convergence (trust region size).
- Hermite vs. non-Hermite (does the derivative info help?).
- SVD-derived d₂ vs. plain orthogonalized gradient (does the bootstrap matter?).
- Full tensor grid vs. sparse grid in Phase 3.

## Implementation Plan

Estimated effort: 2–3 weekends to Phase 1 working, 2 additional weekends to full progressive schedule, 1–2 weekends for experiments.

### Week 1: Phase 1

- Implement 1D Chebyshev node generation, Hermite-Chebyshev fitting via DCT variant.
- Implement polynomial root-finding for minimizer extraction (companion matrix eigenvalues or explicit formulas for low degree).
- Write a PyTorch optimizer subclass that wraps an Adam step, runs spectral line search along the Adam direction, and applies the optimal step size.
- Validate on MNIST: does it converge? Does it match Adam with tuned LR?

### Week 2: Characterization

- Run Experiment 1 (landscape structure characterization).
- Based on results, tune Phase 1 defaults (N, interval, batch size scaling).
- Run Experiment 2 (Phase 1 benchmarks).

### Week 3: Phase 2

- Implement orthogonal gradient extraction, SVD for d₂.
- Implement 2D Chebyshev interpolation (tensor product, then sparse if needed).
- Implement grid sampling with shared-computation optimization (cache activations from layers before first adapted layer when possible).
- Run Experiment 3.

### Week 4: Phase 3 and Progressive Schedule

- Implement kD subspace management with rolling basis.
- Implement sparse grid sampling (Smolyak construction).
- Implement adaptive phase transitions based on coefficient-tail analysis.
- Run Experiments 4 and 5.

### Deliverables

- PyTorch optimizer implementation, released as a self-contained module.
- Jupyter notebooks reproducing each experiment.
- Write-up of empirical findings, with explicit failure modes documented.

## Risks and Mitigations

**Risk 1: φ is nearly quadratic in practice, and ADAM-SLS already captures what's there.** If the loss along Adam's direction is always well-approximated by a parabola in the relevant interval, Phase 1 reduces to an expensive reimplementation of Kenneweg's ADAM-SLS and the method's theoretical motivation evaporates. Experiment 1 is designed to detect this before committing to full implementation.

*Mitigation:* If Experiment 1 shows nearly-quadratic structure, pivot to investigating long-horizon regimes (large η_max) where non-quadratic structure must eventually appear as the ray leaves the local basin. Kenneweg's Armijo criterion is fundamentally local; if non-quadratic structure exists at moderate or long range, spectral methods will find it even if backtracking won't. The open question is whether that structure translates to meaningful loss reduction.

**Risk 2: Noise floor dominates.** Even large batches have finite noise, and high-order polynomial fits are notoriously noise-sensitive. The method may overfit to batch-specific structure.

*Mitigation:* Held-out validation batch at each step to detect overfitting. Adaptive N with conservative defaults. The Hermite formulation helps: using gradients as well as losses roughly halves the effective noise per piece of information.

**Risk 3: Per-step cost exceeds benefit.** If N=8 with gradients at each node costs 8× (forward+backward) = ~24× standard training cost per step, the method needs to reach target loss in fewer than 1/24th the steps to be wall-clock competitive.

*Mitigation:* Shared computation for micro-batched evaluation (compute all N losses with total compute of 1–2 full batches). Target the large-batch, small-model regime explicitly where per-step cost is amortizable. Evaluate on wall-clock time to target loss, not just step count.

**Risk 4: The method works but provides only marginal improvement over ADAM-SLS.** The strongest realistic risk: it works, it's novel, but ADAM-SLS is close to optimal for the 1D line-search subproblem, and the bulk-space extension is the only real source of novel improvement.

*Mitigation:* Frame the research question honestly — is there a regime where this meaningfully outperforms ADAM-SLS, even if it's a narrow one? Small-model specialist training (the model-golf context) is the primary target; failure to win elsewhere is acceptable. If only Phase 2 beats ADAM-SLS, the paper becomes "spectral modeling of gradient-orthogonal directions accesses bulk-space structure that line search cannot," which is still a valid contribution.

**Risk 5: The bootstrapped second direction is noise, or lies entirely in the dom subspace.** The orthogonal gradient components may not have meaningful structure (noise around zero), or may be dominated by top-Hessian-eigenspace content that contributes magnitude but not learning signal.

*Mitigation:* Experiment 3's bulk-space diagnostic directly tests this. If the projection of d₂ onto the bulk subspace is small, the method should skip Phase 2 — this becomes part of the adaptive phase logic.

**Risk 6: Chaotic loss landscapes defeat spectral fitting.** Li et al. (2018) showed that deep networks can have highly non-smooth loss landscapes once depth exceeds a certain threshold, and that this chaotic behavior coincides with poor generalization. High-order polynomial fits work poorly on chaotic functions — Chebyshev coefficients don't decay, and the fit overfits noise-like landscape roughness.

*Mitigation:* Experiment 1's coefficient-decay measurement directly diagnoses this. If coefficients don't decay on the architectures under test, the method should either (a) restrict to smoother architectures (shallow networks, residual networks, LoRA-style low-rank updates where smoothness is inherited from the base model), (b) reduce N and interval width to fit locally where smoothness holds, or (c) conclude the method is inapplicable to the tested setting and pivot to a smoother regime.

## Relationship to Existing Work

The proposal draws on several established threads without being subsumed by any of them. The most directly relevant prior work is reviewed here; a more comprehensive review is maintained in a companion document.

### Closest neighbors

**ADAM-SLS (Kenneweg et al., 2024)** — the primary baseline. Established that Armijo line search along Adam's direction beats tuned Adam with cosine schedule on GPT-2 and BERT. Our method extends this by replacing Armijo backtracking with Chebyshev spectral fitting on Hermite data, enabling long-horizon probing and kD extensions. If their direction choice is right (it empirically is), our method inherits that correctness and adds orthogonal-direction capability.

**Probabilistic line search (Mahsereci & Hennig, 2015)** — uses Gaussian processes to model φ(η) under stochastic noise. Similar goal, different modeling family. Our method uses polynomial models appropriate for the smooth-plus-large-batch regime; theirs is more robust to small batches but more expensive per step.

**EvoGrad (Feb 2025)** — combines CMA-ES sampling with gradient-based updates and Hessian correction. The closest recent hybrid gradient + black-box method in spirit. Different specific approach (covariance-matrix sampling, Taylor residual correction) but similar ingredients. Worth adopting evaluation methodology from.

### Subspace and curvature structure

**Gradient Descent Happens in a Tiny Subspace (Gur-Ari et al., 2018)** — showed that gradients concentrate in the top-k Hessian eigenspace where k ≈ number of classes. Supports the premise that a small subspace captures useful search directions.

**Dom vs. Bulk (Song et al., 2024; Wen et al., 2024; BSFA Oct 2025)** — refined Gur-Ari's picture: the top eigenspace captures magnitude but the bulk subspace carries learning signal. Our Phase 2 d₂ extraction is trying to reach bulk directions; the bulk-space verification step is a direct response to this literature.

**Intrinsic dimension (Li et al. 2018; Aghajanyan et al. 2021)** — tasks have much lower intrinsic dimension than model parameter count. RoBERTa fine-tuning ≈ 200 dimensions; MNIST ≈ 750. Our per-step k=3-5 is much smaller than these, but that's fine: we're capturing per-step useful directions, not the task's full subspace. The method works cumulatively across many steps, each using a different small subspace.

**Large-scale loss landscape structure (Fort & Scherlis, 2019)** — wedge/tunnel structure in high-dim loss landscapes. Supports the intuition that rays can carry past local structure into connected low-loss regions.

### Classical optimization antecedents

- **Polynomial line search**: low-order local fits. Our method uses high-order global spectral fits with Hermite data.
- **Trust region methods (Conn, Gould, Toint)**: local quadratic models in a trust region. Our method uses spectral polynomial models with subspace-constrained trust region.
- **L-BFGS (Nocedal)**: low-rank Hessian approximation from gradient differences. We use similar gradient-difference ideas but for direction discovery rather than Hessian approximation, and fit polynomial loss models rather than quadratic ones.
- **Subspace descent methods (Gratton et al.)**: optimize within a randomly or heuristically chosen low-dim subspace. Our method derives the subspace from the 1D line search's own gradient evaluations.
- **CMA-ES (Hansen)**: evolution strategy that adapts a covariance matrix to align with the inverse Hessian. Closest classical analog to the full progressive-dimension scheme, but ranking-based rather than spectral and requires no gradients.
- **Chebfun / spectral methods (Trefethen)**: Chebyshev interpolation for approximation and rootfinding. We apply these to neural network optimization, which has not been standard.
- **Hypergradient methods (Baydin et al.)**: adapt learning rate via gradient of loss w.r.t. learning rate. Uses gradient info rather than polynomial interpolation.

### Evolutionary approaches

**Evolution Strategies (Salimans et al., 2017)** — showed ES is competitive with SGD on RL tasks via random-direction sampling. Mathematical bridge: Whitelam et al. (Nature Comms 2021) proved neuroevolution equals SGD with noise in the small-perturbation limit. Our method sits between them: uses gradient-chosen directions (like SGD) with non-local probing along those directions (like ES).

**Recent ES at scale (Sep 2025)** — ES with population of 30 can fine-tune billion-parameter LLMs. Hints at strong low-dim structure in the LLM fine-tuning problem, consistent with Aghajanyan's intrinsic-dim results.

### Lottery ticket and connectivity

**Lottery Ticket Hypothesis (Frankle & Carbin 2019)**, **Linear Mode Connectivity (Frankle et al. 2020)**, **Permutation-mode connectivity (Entezari et al. 2021)** — all suggest that the effective optimization landscape is much simpler than the ambient parameter count suggests, up to initialization lottery and network symmetries. Supports the general premise that low-dim methods should work.

### Novel combination

The specific combination — spectral Hermite interpolation + Adam-direction as primary axis + bulk-verified bootstrapped subspace from ray gradients + progressive-dimension schedule — is, to the best of our knowledge, not present in the literature. Nearest relatives (ADAM-SLS, EvoGrad, CMA-ES hybrids) each share one or two of these ingredients but not the combination.

## Open Questions to Be Resolved by the Prototype

1. What's the right N for each phase in practice? Theory says 8–16 for smooth 1D functions; does this hold for realistic loss landscapes? Does Chebyshev coefficient decay actually occur on deep architectures?
2. How often do Phase 2 and Phase 3 actually trigger? Or is Phase 1 sufficient most of the time?
3. Does the bootstrapped d₂ carry useful information? Specifically, does it have substantial bulk-space content (not just dom-space magnitude)?
4. Is Hermite interpolation's extra complexity worth the accuracy gain, or is non-Hermite Chebyshev enough?
5. What's the wall-clock crossover point at which this method beats ADAM-SLS? Model size? Batch size? Task?
6. Where in the training trajectory does the method add the most value — early (establishing good step sizes), middle (navigating medium-range non-convex structure), or late (fine balancing of multiple directions near convergence)?

## Success Definition

The prototype is a success if it produces one of these three outcomes:

1. **Unambiguous win:** On at least one realistic task, the full progressive-dimension method converges to lower final loss, or reaches a target loss faster in wall-clock, than ADAM-SLS (the primary baseline). This would be a publishable result.

2. **Clear characterization:** The method matches ADAM-SLS's best performance but doesn't exceed it, with a clear understanding of why (e.g., "the loss along gradient rays is nearly quadratic in all regimes we tested, so Phase 1 offers no advantage over Armijo, and Phase 2's bulk-space projection reveals d₂ is dominated by dom-space content"). Negative results on a principled method with clean diagnostics constrain future work.

3. **Isolated regime win:** The method wins only in the narrow model-golf regime (small, specialist, large-batch, cheap-per-step) but not broadly. This is aligned with the proposing motivation and still useful.

The method fails if it produces consistent losses on all benchmarks with no clear explanation — suggesting an implementation or methodology flaw rather than a clean result. Diagnostic measurements from Experiment 1 and Experiment 3 (Chebyshev coefficient decay, bulk-space projection of d₂) should always explain outcomes, success or failure.
