# Prior Work Landscape: Spectral Hermite Trust Region

A synthesis of research relevant to the proposed optimizer, organized around your three questions.

## 1. Long-range gradient-space exploration approaches

### Armijo + ADAM line search (Kenneweg et al., 2024)

The most directly competing work. They identify exactly the failure mode that motivates long-range methods: "the Armijo criterion exclusively conducting the line search in the direction of the gradient. When this direction significantly diverges from the actual update direction, as is often the case in large-scale transformer training, setting the momentum term β₁=0 becomes unreliable for estimating the optimal step size." Their fix is to line-search along Adam's momentum-adjusted direction rather than raw gradient — this is the direction your Phase 1 should also use, and their paper validates that choice against tuned Adam on GPT-2 and BERT fine-tuning. They show "Our optimization approach outperforms both the previous Armijo implementation and tuned learning rate schedules for Adam" and release it as a Python package.

What they *don't* do: high-order spectral fitting, Hermite interpolation using gradients as derivative data, or multi-dimensional extension. Their line search is local (Armijo-style backtracking). This is a good sign for the proposal — it means the "search along Adam direction" idea has empirical backing, and the spectral/Hermite/kD extensions are the novel contribution.

### Probabilistic line search (Mahsereci & Hennig, 2015)

Uses Gaussian processes to model φ(η) under stochastic noise. Solves the noise problem differently than the proposal — GP posterior gives you probabilistic confidence intervals rather than a deterministic polynomial fit. In principle more robust to small batches but more expensive (GP inference, kernel choices). Conceptually similar to the proposal but with a different modeling family.

### Hypergradient descent (Baydin et al., 2017)

Rather than line searching, computes ∂L/∂η (gradient of loss with respect to learning rate) and takes a gradient step on η itself. Purely local — doesn't probe the landscape, just follows the derivative. Computationally cheaper than line search but can't see non-local structure.

### Two-way backtracking (Truong & Nguyen, 2020)

"One observes that if the sequence converges... one should allow to increase learning rate (and not just decrease as in the section Algorithm)." Extends standard Armijo backtracking to also increase the step size when recent steps succeeded. Gets at "non-local" only in the sense of probing wider intervals; still local fitting.

### Edge of stability and landscape probing

Cohen et al.'s edge-of-stability work (2021) and follow-ups document that "gradient descent trajectories tend to enter higher positive curvature regions of the loss landscape before eventually finding the desired flatter regions" — i.e., that real training often operates at effective step sizes *above* what classical stability analysis permits. This is exactly the regime where your long-range method has a conceptual edge: it empirically measures where loss goes, rather than relying on Taylor bounds that are known to be loose.

### Subspace line search / trust region methods (classical)

Conn, Gould & Toint's *Trust-Region Methods* (2000) is the canonical reference. Subspace trust region variants restrict the quadratic model to a Krylov-style subspace built from gradients and Hessian-vector products. Typically use quadratic models within the subspace, not polynomial models; typically small subspace dimensions (2-5). The proposal's Phase 3 is a spectral-model relative of this family.

### Recent subspace-aware optimizers

**BSFA — Bulk-Space-Filtration-Accelerator (Oct 2025).** "updates along the top eigendirections of the loss Hessian (Dom-space) capture most of the update magnitude, they often contribute minimally to loss reduction. In contrast, updates in the orthogonal component (Bulk-space) have smaller magnitudes but drive most learning progress... BSFA accelerates training by differentially scaling update components projected onto these distinct subspaces." This is very relevant: it shows that explicitly reasoning about dominant vs. bulk Hessian subspaces during training is an active, productive line. Your Phase 2/3 directions (gradient vs. orthogonal component from probe gradients) are reaching toward the same Dom/Bulk structure empirically.

### Evolution strategies (long-range by construction)

"ES only requires workers to communicate a few scalars between each other... Intuitively, this is because we control the random seeds on each worker, so each worker can locally reconstruct the perturbations of the other workers." Salimans et al.'s ES (2017) probes the landscape by sampling loss at many perturbed parameter settings — essentially random directions, not chosen ones. Inherently long-range but wasteful: most random directions aren't useful. Modern ES variants like CMA-ES (see below) do much better direction selection.

**Recent: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (Sep 2025).** "the ES implementation only needs a population of 30 to effectively optimize billions of parameters. In contrast, previous work used..." much larger populations. Shows that well-tuned ES with small populations can fine-tune LLMs competitively, which hints at the small-subspace structure of the task.

## 2. Intrinsic dimensionality estimates

This is directly relevant to the proposal's "small k is enough" argument. The empirical results are striking.

### Li et al. (2018) — original intrinsic dimension measurement

"We introduce the intrinsic dimension of an objective landscape with an illustrative toy problem... we measure intrinsic dimension over a variety of network types and datasets, including MNIST, CIFAR-10, ImageNet, and several RL tasks."

Their method: project the full D-dimensional weight update onto a random d-dimensional subspace, train in that subspace, find the smallest d at which the loss still reaches a target. Key findings:

- "Many problems have smaller intrinsic dimensions than one might suspect, and the intrinsic dimension for a given dataset varies little across a family of models with vastly different sizes."
- "solving the inverted pendulum problem is 100 times easier than classifying digits from MNIST, and playing Atari Pong from pixels is about as hard as classifying CIFAR-10."
- MNIST intrinsic dim ≈ 750
- CIFAR-10 intrinsic dim ≈ 2900 (for fully-connected), ≈1000 (for better architectures)
- Inverted pendulum ≈ 4
- Atari Pong ≈ 6000

### Aghajanyan et al. (2021) — intrinsic dimension of LM fine-tuning

"common pre-trained models have a very low intrinsic dimension; in other words, there exists a low dimension reparameterization that is as effective for fine-tuning as the full parameter space. For example, by optimizing only 200 trainable parameters randomly projected back into the full space, we can tune a RoBERTa model to achieve 90% of the full parameter performance levels on MRPC."

Key findings:
- "pre-training implicitly minimizes intrinsic dimension and, perhaps surprisingly, larger models tend to have lower intrinsic dimension after a fixed number of pre-training updates, at least in part explaining their extreme effectiveness."
- Fine-tuning RoBERTa on MRPC: d ≈ 200
- Fine-tuning BERT-large on various GLUE tasks: d ≈ 1000-10000

This is the foundational evidence for LoRA and PEFT methods. Directly supports the "small-dimensional subspace captures most of what matters" premise.

### Gur-Ari et al. (2018) — gradient lives in Hessian-top subspace

"We show that in a variety of large-scale deep learning scenarios the gradient dynamically converges to a very small subspace after a short period of training. The subspace is spanned by a few top eigenvectors of the Hessian (equal to the number of classes in the dataset), and is mostly preserved over long periods of training."

- CIFAR-10: top-10 Hessian eigenspace captures most gradient energy
- ImageNet: top-1000 Hessian eigenspace
- The number scales roughly as the number of output classes

### Song et al. (2024), Wen et al. (2024) — Dom vs Bulk refinement

Recent work challenges Gur-Ari's "learning happens in the top subspace" interpretation: "Gur-Ari et al., (2018) observed that the gradient vectors during training tend to align with the top-k eigenspace of the Hessian for k-class classification tasks. They hypothesized that learning predominantly occurs within this dominant subspace. However, this hypothesis was challenged in Song..." 2025. The correction: the top subspace captures magnitude but the *bulk* (orthogonal complement) drives actual learning progress. Your method should probably search primarily in directions with substantial bulk-space components.

### Fort & Scherlis (2019) — wedge structure

"In this section we will gradually build a phenomenological toy model of the landscape in an informal manner... we predict and demonstrate the existence of higher dimensional generalizations of low loss tunnels that we call m-tunnels."

Their model: low-loss regions form wedge-shaped structures that meet along lower-dimensional tunnels. Optimization moves radially along wedges. Matches your intuition that "the ray might encounter structure beyond the local basin" — the wedges mean that moving in gradient direction can carry you past local structure into wedge-shared regions.

### Li et al. (2018) — Visualizing the Loss Landscape

"when networks become sufficiently deep, neural loss landscapes quickly transition from being nearly convex to being highly chaotic. This transition from convex to chaotic behavior coincides with a dramatic drop in generalization error"

Important caveat for your method: deep networks can have highly non-smooth loss landscapes, which would make high-order Chebyshev fits less accurate. Matters especially for Phase 1 node-count choices. Their filter-normalization technique for visualization is probably worth adopting for your Experiment 1 (landscape characterization) to get comparable plots across runs.

### Specific domain numbers

Rough estimates for tasks you'd plausibly prototype on:

| Task | Model | Intrinsic dim estimate | Source |
|---|---|---|---|
| MNIST | small MLP | ~750 | Li 2018 |
| CIFAR-10 | small CNN | ~1000-3000 | Li 2018 |
| RoBERTa fine-tune (MRPC) | 125M params | ~200 | Aghajanyan 2021 |
| BERT fine-tune (GLUE) | 340M params | ~1000-10000 | Aghajanyan 2021 |
| Inverted pendulum | MLP | ~4 | Li 2018 |
| Atari Pong | CNN | ~6000 | Li 2018 |
| Gradient-Hessian top subspace (classification) | any | ~# of classes | Gur-Ari 2018 |

**Implication for the proposal.** The intrinsic dimension numbers are *much* higher than the proposed method's k=3-5 subspace. But the Gur-Ari observation matters more for Phase 3: the gradient-spanned Krylov subspace *at a given iteration* is roughly k-dimensional, even if the full task's intrinsic dimension is much higher. The method doesn't need to span the full intrinsic dimension in a single step — it just needs the rolling subspace of recent update directions to capture the current step's important variation. Still, the gap between "what we probe per step" and "what the task actually needs" is worth characterizing explicitly in your Experiment 1.

## 3. Weight initialization and evolutionary search tie-ins

### Lottery ticket hypothesis (Frankle & Carbin, 2019)

"Dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective."

The connection to your proposal is subtle but real. LTH says training implicitly selects a low-dimensional effective structure (the winning subnetwork) from the full parameter space. Your method explicitly constructs a low-dimensional subspace (from gradient probes) at each step. The claim that "winning tickets" exist means that at initialization, there's already a latent low-d structure your method might be able to find earlier than standard training, by probing the right directions.

### Linear mode connectivity (Frankle et al., 2020)

"We study whether a neural network optimizes to the same, linearly connected minimum under different samples of SGD noise"

Key finding: after a brief initial period of training (~1000-2000 steps for ResNet/VGG), models become *stable* to SGD noise — i.e., the optimization trajectory enters a region where different data orderings all converge to linearly-connected minima. This is empirical evidence for your Phase 3 regime: late in training, there's a well-defined low-d manifold of minima, and spectral modeling over a small subspace is enough to navigate it.

### Entezari et al. (2021) — permutation-mode connectivity

"Neural networks trained with stochastic gradient descent (SGD) starting from different random initialisations typically find functionally very similar solutions... any two solutions found by SGD can be permuted such that the linear interpolation between their parameters forms a path without significant increase in loss. Here, we use a simple but powerful algorithm to find such permutations... we find that two networks already live in the same loss valley at the time of initialisation and averaging their random, but suitably permuted initialisation performs significantly above chance."

Powerful result: up to permutation, all SGD solutions live in one big connected basin. The intrinsic structure of the loss landscape is much simpler than it appears when you account for network symmetries. Implication: your method might benefit from working in a symmetry-quotient space, though actually doing this is nontrivial.

### Salimans et al. (2017) — OpenAI Evolution Strategies

"Mathematically, you'll notice that this is also equivalent to estimating the gradient of the expected reward in the parameter space using finite differences, except we only do it along 100 random directions."

The key insight — that ES is essentially a randomized finite-difference gradient estimator — makes it a direct relative of your method. Your method *also* uses finite-difference-like probing of the loss, but:

1. Along a *chosen* direction (Adam's) rather than random ones — much more sample-efficient
2. With spectral polynomial fitting rather than linear averaging — captures non-linear structure
3. Using actual backprop gradients rather than purely sample-based estimates — exploits differentiability

The relationship: ES is what happens when you have only zeroth-order access to loss. Your method is what happens when you have first-order access (gradients) plus the compute to probe along that direction.

### CMA-ES (Hansen)

Probably the closest "classical" analog to your method's spirit. "Adaptation of the covariance matrix amounts to learning a second order model of the underlying objective function similar to the approximation of the inverse Hessian matrix in the quasi-Newton method in classical optimization." "In the vicinity of a quadratic optimum, the distribution of sampling points approaches... Hence, CMA-ES learns an empirical approximation to the inverse Hessian through covariance adaptation."

CMA-ES adapts a sampling distribution (Gaussian with covariance matrix C) based on which samples had low loss. Over time C aligns with the inverse Hessian. Your method does something structurally similar — probing directions derived from past gradients — but deterministically and with spectral polynomial fitting rather than ranking-based updates.

Key CMA-ES limitation directly relevant to your proposal: "Assuming a black-box optimization scenario, where gradients are not available... the CMA-ES method is likely to be outperformed by other methods in... on (nearly) convex-quadratic functions with low or moderate condition number of the Hessian matrix, where BFGS or NEWUOA or SLSQP are typically at least ten times faster". So in the regimes where your Phase 3 operates (near convergence, nearly quadratic), gradient-based quasi-Newton methods already dominate CMA-ES. Your method's potential win vs. CMA-ES is in the non-quadratic regime where gradient + spectral model beats ranking-based covariance adaptation.

### Hybrid approaches: EGGROLL, LS-CMA-ES, EvoGrad

Recent papers integrate gradient information into ES:

- **EGGROLL** ("Evolution Guided General Optimization via Low-rank Learning"): ES with low-rank structure for large-network compatibility.
- **LS-CMA-ES**: Uses "an approximation of the Hessian matrix of the target function... This approximation will be deterministically built using the available sample points". Builds a quadratic model from the ES population's samples — structurally similar to your "build a polynomial model from Chebyshev samples" but within the ES framework.
- **EvoGrad** (Feb 2025): "integrating global statistical insights from the evolutionary algorithm CMA-ES into the gradient learning framework, effectively biasing gradient estimates towards regions with higher optimization potential. Moreover, we enhance the gradient learning process by estimating the Hessian matrix, allowing us to correct the second-order residual of the Taylor series approximation."

EvoGrad is the closest in spirit to your proposal — gradient + Hessian + landscape-aware sampling. Your method differs in using polynomial (not just quadratic) models and spectral (not just random) sampling. Worth reading their paper for the shape of the evaluation they run.

### Weight-agnostic neural networks (Gaier & Ha, 2019)

Showed you can find architectures that perform well with *random* weights. Shifts the optimization problem from "find good weights" to "find good structure." Tangentially relevant: suggests there's redundancy between what weight optimization and what structural choices accomplish.

### Neuroevolution-GD equivalence (Whitelam et al., Nature Comms 2021)

"We show analytically that training a neural network by conditioned stochastic mutation or neuroevolution of its weights is equivalent, in the limit of small mutations, to gradient descent on the loss function in the presence of Gaussian white noise. Averaged over independent realizations of the learning process, neuroevolution is equivalent to gradient descent on the loss function."

Theoretical bridge: in the small-perturbation limit, ES and SGD are the same thing. What differs is the *sampling strategy* and the *non-local information* ES gathers. Your method is trying to get the best of both: SGD's gradient precision + ES's non-local probing, specifically on directions chosen to be maximally informative.

## Summary — how the proposal fits in

The proposed method sits at a three-way intersection that's surprisingly underexplored as a combined approach:

1. **Long-range line search** (where Kenneweg et al. 2024 is the closest competitor, but uses Armijo, not spectral)
2. **Low-dimensional subspace methods** (where LTH, intrinsic dim, and Gur-Ari subspace give strong empirical backing for the premise)
3. **Hybrid gradient + black-box probing** (where CMA-ES, LS-CMA-ES, EvoGrad are relatives, but use Gaussian sampling and ranking, not spectral polynomial fitting with Hermite data)

The combination of (a) Chebyshev/Hermite spectral models, (b) Adam-direction as the primary axis, (c) gradient-derived subspace extension via SVD of orthogonal components, and (d) progressive-dimension schedule from 1D to kD, is the specific configuration I can't find in the literature.

### Refinements to the proposal suggested by this review

1. **Use the Kenneweg Adam-direction result as a baseline, not just standard Adam.** They've already shown that Adam-direction line search beats tuned Adam. Your method needs to beat *that*, not just plain Adam. Include ADAM-SLS (their method) in Experiments 2-4.

2. **Characterize the relevant subspace more carefully.** Gur-Ari tells us the gradient lives in the top-k Hessian eigenspace. Song 2024 tells us the *learning signal* is in the bulk. Your SVD of orthogonal gradient components is trying to extract useful bulk-space directions, which is good — but you should explicitly check whether the discovered d₂ has substantial bulk-space content or just top-space content.

3. **Align with intrinsic dimension numbers.** Your k=3-5 subspace is much smaller than any task's intrinsic dimension. That's fine for per-step optimization but worth stating: you're not claiming the subspace captures the task, only that it captures the useful local search directions for the current step.

4. **Acknowledge the "chaotic landscape" regime as a risk.** Li 2018's visualization work shows deep networks have highly non-smooth landscapes. This could defeat high-order spectral fits. Experiment 1 should measure the Chebyshev coefficient decay rate on realistic architectures to know whether spectral methods are appropriate.

5. **Consider a CMA-ES-inspired variant as an alternative Phase 3.** Instead of SVD of orthogonal components, maintain a running covariance matrix of recent gradient directions and sample from it. This is closer to CMA-ES in spirit but with gradient-derived updates rather than ranking-based updates. Might be simpler to implement and tune.

6. **Read EvoGrad carefully.** Feb 2025 paper, closest in spirit to your proposal. Their evaluation methodology, benchmarks, and ablations are worth adopting or explicitly diverging from.

### Publication opportunity assessment

The proposal is likely to be publishable if it works, for these reasons:

- It combines tools from three mature literatures (classical numerical analysis, subspace optimization theory, and ES/gradient hybrids) in a way none of them individually do.
- The empirical question "can spectral modeling of the loss along Adam-direction usefully extend Kenneweg-style line search" is cleanly posed and the answer (yes/no/partially) is useful either way.
- If the progressive-dimension schedule works, it provides a new framing for "what kind of optimizer you want at each training stage" which is a concept with broader implications.

Risks to publishability:

- If Phase 1 doesn't clearly beat Kenneweg's ADAM-SLS, the method has no story.
- If it does beat ADAM-SLS but only on toy tasks, it's a small-scope contribution.
- The specific combination might be viewed as "more engineering than science" if the theoretical contribution is unclear. Framing the paper around "spectral models of the loss landscape in gradient-spanned subspaces" rather than "a new optimizer" probably lands better.
