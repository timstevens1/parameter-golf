# Input-Dependent Computation: A Unified Research Agenda for Efficient Language Models

## From Attention Residuals to Circuit Fabrics, MoE Compression, and Program Synthesis

---

## 1. The Unifying Principle

A single principle connects every idea in this document: **making previously fixed computation input-dependent yields more expressivity per parameter.** This principle has been validated at every scale it has been tested:

- **Attention** (Vaswani et al., 2017): Input-dependent token mixing replaced fixed convolutions
- **Mixture-of-Experts** (Shazeer et al., 2017): Input-dependent expert selection replaced fixed FFN layers
- **Attention Residuals** (Moonshot AI, 2026): Input-dependent depth aggregation replaced fixed residual connections, matching 1.25× more compute
- **Mixture-of-Depths** (Raposo et al., 2024): Input-dependent compute allocation replaced uniform per-token processing
- **Selective SSMs / Mamba** (Gu & Dao, 2023): Input-dependent state dynamics replaced fixed state transitions, matching transformers 2× the size
- **CRIREL circuits** (White et al., 2025): Input-dependent bias currents switch a 4-neuron circuit between 24 logic functions

The mathematical reason is consistent across all cases: a fixed d×d weight matrix stores one function using d² parameters. An input-dependent mechanism parameterizes a *family* of effective matrices using O(d) learned parameters. Each input receives a different effective transformation. The expressivity-per-parameter ratio is dramatically better.

This document synthesizes a research agenda exploring how far this principle can be pushed — from bit-level logic gates to system-level program synthesis.

---

## 2. The Landscape of Input-Dependent Computation

### 2.1 Historical Lineage

The idea has deeper roots than modern transformers:

**Fast Weight Programmers (Schmidhuber, 1991–93):** A "slow" network generates weights for a "fast" network at runtime. The first formalization of input-dependent weight generation.

**Self-Referential Weight Matrices (Schmidhuber, 1993; Irie et al., 2022):** Weight matrices that modify themselves during runtime using outer products and delta update rules. Enables recursive self-improvement — meta-learning to learn.

**Linear Transformers as Fast Weight Programmers (Schlag, Irie & Schmidhuber, 2021):** Proved that linear attention is mathematically equivalent to fast weight programming. Attention *already is* dynamic weight generation — we just weren't framing it that way.

**HyperNetworks (Ha, Dai & Le, 2017):** Small networks generating weights for larger networks, achieving parameter efficiency through weight compression and soft weight sharing.

**LambdaNetworks (Bello, 2021):** Input-dependent linear projections replacing attention maps. Transforms contexts into fixed-size linear functions applied per input, bypassing quadratic attention costs while capturing content and position interactions.

### 2.2 The Neuron-Level View

A standard neuron computes y = σ(w·x + b) with fixed weights. Making it input-dependent:

- **Dynamic weights:** y = σ(f(x)·x + b), where f generates weights from input. This produces x^T P^T x — a quadratic form giving second-order interactions. With low-rank factorization P = AB^T, you parameterize a family of d-dimensional weight vectors using 2dr parameters instead of d².
- **Multiplicative gating (SwiGLU):** y = σ((w ⊙ g(x))·x + b), where g modulates fixed weights per input. Already standard in modern LLMs — SwiGLU's (swish(xW₁) ⊙ xV)W₂ is a form of input-dependent computation within the FFN.
- **Recursive gating:** If g(x) is itself gated, you get nested meta-control — circuits configuring circuits configuring circuits. This creates layers of control rather than layers of transformation, analogous to biological neuromodulation (dopamine, serotonin modulating how neurons respond to signals).

### 2.3 State Space Models and Mamba

SSMs provide a parallel path to the same destination from control theory rather than attention.

A standard SSM uses fixed state transitions: h_t = Ah_{t-1} + Bx_t. Mamba makes A, B, C functions of the input, so the state update becomes h_t = A(x_t)h_{t-1} + B(x_t)x_t. The input simultaneously serves as data and configuration — determining how the state evolves.

**Key results:**
- Mamba-3B outperforms same-size transformers and matches transformers 2× its size
- 5× higher inference throughput than transformers (no KV cache, linear scaling)
- Mamba-2 showed duality: selective SSMs can be recast as structured attention under certain restrictions
- Mamba-3 (March 2026, ICLR 2026): 7× faster than Llama-3.2-1B on long sequences, complex-valued states, MIMO formulation

**Production status:** NVIDIA and IBM ship hybrid Mamba-Transformer models. Kimi, Tencent Hunyuan, and Qwen incorporate Mamba-2/GDN layers in production hybrids. Pure Mamba excels at sustained generation; attention layers handle precise retrieval. Hybrids outperform pure transformers.

**Connection to the unifying principle:** Mamba's "selection mechanism" is exactly input-dependent gating. The discretization parameter Δ generalizes RNN gating. RNNs, attention, SSMs, and CRIREL circuits are all instances of the same principle — input determines dynamics.

---

## 3. Learned Logic Circuits

### 3.1 CRIREL Circuits

White et al. (2025) demonstrated a four-neuron circuit (Coupled Recurrent Inhibitory and Recurrent Excitatory Loops) implementing 24 distinct logic functions via bias current changes alone, with no weight modifications. The mechanism is a double-cusp bifurcation: small input changes push the system across bifurcation boundaries into qualitatively different dynamical regimes.

- **Input structure:** 2 data inputs + 4 bias currents (configuration) = 6 total input bits
- **Functions:** 8 nontrivial logic gates × 3 input interpretation modes (magnitude, temporal, phase) = 24 functions
- **Switching latency:** 20–40ms (~2 spikes)
- **Robustness:** Tolerates ~10% noise in bias currents and ~10% variation in synaptic weights
- **Neuron model independence:** Works with LIF, Izhikevich, clipped ReLU, sigmoid — flexibility is topological

### 3.2 Differentiable Logic Gate Networks (Petersen et al., 2022)

Each node learns a probability distribution over 16 possible two-input Boolean gates (AND, OR, XOR, etc.). Trained via continuous relaxation with gradient descent, then discretized to hard gates.

- Purely feedforward — no settling time
- Over 1M MNIST classifications per second on a single CPU core
- Extended to convolutional patterns (86.29% CIFAR-10 with 61M logic gates, 29× smaller than alternatives)
- Extended to ternary logic (true/false/unknown) with built-in uncertainty quantification

### 3.3 Weightless Neural Networks (WiSARD)

Neurons implemented as lookup tables (truth tables) rather than weighted connections. An n-input LUT implements any Boolean function of its inputs by direct memorization. LogicWiSARD converts trained LUTs to minimized logic circuits, achieving 80%+ energy reduction versus equivalent MLPs.

### 3.4 Key Distinction

Petersen's gates and WiSARD LUTs produce static circuits — fixed after training. CRIREL produces dynamic circuits — the function changes per input via bias currents. Making the gate selection input-dependent (extending Petersen) or the LUT index input-dependent (extending WiSARD) would bridge the gap.

---

## 4. Proposal A: Circuit Fabric for Layer Replacement

### 4.1 Core Architecture

Replace transformer layers with hierarchical CRIREL-based circuit fabrics operating on Q4-quantized bit representations.

**Input:** Q4-quantized hidden state → ~20,000 bits (16,384 data bits + 4,096 metadata bits for d=4096)

**Reshape:** Bitvector → 15-dimensional binary hypercube (size-2 dimensions). Factorization respects quantization structure. Varies per layer for global interaction in O(log n) layers.

**Three-level hierarchy:**
- **Level 1 — Atomic Gates (~160K):** CRIREL-like circuits along each hypercube dimension, taking 2 data + 4 config bits
- **Level 2 — Circuit Banks (~1.5M gates):** Multiple gates per position with different data/config dimension assignments. Each bank is a different "interpretation" of the same bits.
- **Level 3 — Bank Selection (~300K gates):** Routing circuits selecting which banks to evaluate per input. Built from the same gate primitives (self-similar).

**Output:** Same-shape bitvector with quantization structure preserved. Residual aggregation via AttnRes-style selective depth attention.

### 4.2 Gate Primitive Alternatives

The architecture is agnostic to the gate primitive:
- **CRIREL:** Maximum functional diversity (24 functions), but requires settling time (3–5 iterations)
- **Differentiable Logic Gates:** Existing PyTorch implementation, purely feedforward, well-understood training
- **Lookup Tables:** Maximally expressive (any Boolean function), fastest inference (single table read)
- **Threshold Logic Units:** Familiar neural network primitive, smooth training landscape
- **Ternary Logic Gates:** Built-in uncertainty for adaptive compute routing

### 4.3 Parameter Budget

~2M gates × ~20 bits each ≈ 5 MB per layer, versus ~134 MB (float16) or ~33 MB (Q4) for a standard transformer layer. **6–7× reduction** at minimum; potentially 10–100× if applied to MoE expert ensembles.

### 4.4 Training

Evolutionary search with layer-wise distillation from pretrained models. Capture input-output pairs at each layer boundary, evolve circuits to minimize reconstruction error. Progressive bootstrapping from small to large scales.

---

## 5. Proposal B: MoE Expert Replacement

### 5.1 The MoE Parameter Problem

In Mixtral 8x7B, each MoE layer has 8 experts × ~117M parameters = ~940M parameters per layer (~470 MB at Q4). Experts are often redundant, routing is crude (single linear projection), and the model can only select from exactly 8 discrete functions.

### 5.2 Existing Compression Approaches

The field is actively exploring MoE compression:

- **Expert pruning:** Remove least-important experts. Risk of discarding important knowledge; >20% MMLU degradation reported.
- **Expert merging:** Cluster similar experts (HC-SMoE, MC-SMoE, PuzzleMoE). Works when experts have high similarity; fails with diverse experts.
- **Low-rank decomposition:** D2-MoE extracts shared weights + SVD on residual deltas. MoE-SVD achieves 60% compression with 1.5× speedup. MoBE (Mixture-of-Basis-Experts) achieves 50%+ lower reconstruction error than alternatives.
- **LoRA-as-expert:** MoLA, X-LoRA, HELLoRA — use LoRA modules as lightweight experts with learned routing. HELLoRA uses 15.74% of parameters with 9.24% accuracy improvement.

**Gap:** All existing work compresses within the same computational paradigm (smaller matrices, still float matmuls). Nobody is replacing the computational primitive itself.

### 5.3 Proposed Alternatives

#### Option 1: Circuit Fabric MoE Replacement

Replace the entire MoE layer (router + all experts) with a single circuit fabric. Configuration bits from the input implicitly determine which effective transformation is applied. Eliminates expert boundaries, load balancing, and separate routing.

**Compression:** ~5–50 MB versus ~470 MB per MoE layer (10–100× reduction).

#### Option 2: Masked Wide FFN

One FFN that's 2× wider than a single expert, with routing producing a binary mask over half the output neurons. Different tokens activate different neuron subsets, providing functional diversity without separate expert weight matrices.

**Parameter math:** 16d² versus 64d² for 8-expert MoE (4× reduction). Existing research validates this direction:
- **MoEfication:** Dense FFN layers already exhibit sparse activation; converting to MoE by grouping neurons works.
- **MoME (Mixture-of-Masked-Experts):** Learns binary masks over a shared base network to generate diverse expert functions.
- **MoM (Experts as Masks):** Trains multiple masks instead of multiple expert copies; discovers shared, independent, and redundant expert patterns.

#### Option 3: Latent-Space Expert Retrieval

Store (input hidden state, output hidden state) pairs from expert forward passes. At inference, retrieve nearest-neighbor input and return stored output. Replaces expert computation with vector similarity search.

**Foundation:** kNN-LM (Khandelwal et al., 2019) at the expert level rather than the output token level. RETRO showed a model with retrieval outperforms a 25× larger parametric model.

### 5.4 Separating Knowledge from Reasoning

A deeper question: what do MoE experts actually store?

- **Factual memory** ("Paris is the capital of France"): Pure lookup, replaceable by retrieval/RAG
- **Computational patterns** ("how to compose a subordinate clause"): Procedural knowledge in weight matrices, not directly retrievable

**LmLm (Limited Memory Language Models):** A 382M-parameter model matches LLaMA2-7B's factual precision by externalizing knowledge to a database. 18× parameter reduction with instant updates and unlearning via database operations. Validates that knowledge and reasoning are separable.

**FFN layers as key-value memory:** Mechanistic research confirms that transformer FFN layers can be interpreted as key-value memory for localizing and editing knowledge at the parameter level.

**Scaling crossover:** Compute-optimal curves show that beyond ~10²³ FLOPs, investing in datastore size and retrieval improves accuracy more steeply than increasing model parameters.

---

## 6. Proposal C: Functional Program Architecture

### 6.1 Core Idea

Replace the monolithic model with a small planning model (1–3B) that generates typed functional programs (DAGs of operations), executed by domain-specific executors coordinated by a deterministic scheduler.

### 6.2 Components

- **Planning Model (1–3B):** Decomposes queries into operation DAGs. Needs meta-cognitive skill, not factual knowledge.
- **Reasoning Executor (1–3B):** Chain-of-thought over provided context. Minimal parametric knowledge — reasons about what it's given.
- **Research Executor:** Embedding model + vector store (text-level or latent-space retrieval)
- **Calculate Executor:** Code generation + sandboxed interpreter
- **Verify Executor:** Entailment model for self-consistency checking
- **Scheduler:** Deterministic runtime — validates programs, topological sort, parallel dispatch, failure handling, result caching
- **Latent Knowledge Store:** Vector database of pre-encoded latent representations for direct hidden-state injection

### 6.3 Key Properties

- **Explicit plans as first-class objects:** Inspectable, modifiable, cacheable, reusable
- **Failure isolation:** Errors localized to individual DAG nodes with principled recovery
- **Clean knowledge separation:** No hallucination of facts; instant updates via database operations; verifiable provenance
- **Natural parallelism:** Independent DAG branches execute simultaneously
- **Recursive composition:** The "reason" executor can invoke the planning model for sub-programs

### 6.4 Size Implications

5–10B total neural parameters replacing 200B+ monolithic models. Knowledge store on disk (cheapest storage), neural models on GPU (small models). Total system cost dramatically lower.

### 6.5 Structural Isomorphism with Circuit Fabric

The program architecture is the circuit fabric at a different scale:

| Circuit Fabric | Program Architecture |
|---|---|
| CRIREL gate | Domain executor |
| Data bits | Content being processed |
| Config bits | Operation type and parameters |
| Circuit bank | Parallel executor group |
| Level 3 routing | Planning model |
| Hypercube reshape | Program DAG structure |

Same principle: input-dependent configuration selects computation applied to data, organized as composable circuits/programs.

---

## 7. State Space Models as the Base Architecture

### 7.1 Why Mamba Matters for This Agenda

Mamba already eliminates both attention and MLP blocks, achieving transformer-level performance with input-dependent state dynamics alone. This makes it a potentially better starting point for compression than transformers:

- **Already smaller:** Mamba-3B ≈ Transformer-6B performance
- **No KV cache:** Memory scales with fixed state size, not sequence length
- **Linear inference:** O(L) versus O(L²) for attention
- **Production-validated:** Hybrid Mamba-Transformer models shipping from NVIDIA, IBM, Kimi, Tencent, Qwen

### 7.2 Mamba + MoE

MoE-Mamba integrates Mamba with mixture of experts, achieving comparable performance with 2.2× fewer training steps. The expert compression proposals (Sections 5.2–5.3) apply directly to MoE-Mamba models.

### 7.3 Mamba as Reasoning Core

In the functional program architecture (Section 6), the reasoning executor could be a small Mamba model rather than a transformer. Mamba's efficient sequential processing is well-suited to chain-of-thought reasoning over provided context, and its linear scaling makes it cheap for long reasoning chains.

### 7.4 SSM-Circuit Fabric Connection

Mamba's selective state dynamics (input-dependent A, B, C matrices) are functionally equivalent to CRIREL's input-dependent bias currents. Both achieve input-dependent computation through dynamical systems — the difference is continuous-valued state evolution versus binary bifurcation dynamics. A circuit fabric could potentially implement an approximation of SSM dynamics through binary gate operations over quantized state representations.

---

## 8. Practical Starting Point: Parameter Golf

### 8.1 The Challenge

OpenAI's Parameter Golf (March 18 – April 30, 2026): Train the best language model in a 16MB artifact (weights + code), within 10 minutes on 8×H100s, evaluated by compression on FineWeb (bits per byte). MLX training script provided for Apple Silicon local iteration.

### 8.2 Why It's the Right Testbed

16MB forces radical parameter efficiency — exactly the regime where input-dependent computation should shine:
- float16: ~8M parameters
- Q4: ~32M parameters
- 1-bit: ~128M binary values

The constraint favors architectures that maximize expressivity per parameter over architectures that maximize raw parameter count.

### 8.3 Recommended Architecture

A Mamba-style SSM backbone (no attention, linear scaling) with:
- **Aggressive weight tying:** Depth recurrence — one set of shared weights applied repeatedly, with input-dependent routing determining iteration count per token (Mixture-of-Recursions)
- **Low-precision training:** 2–4 bit quantization-aware training, or BitNet-style 1.58-bit weights
- **Input-dependent gated FFN:** SwiGLU-style masking providing functional diversity within a single parameter-shared FFN
- **Varying reshapes between recurrence steps:** Different interaction patterns per iteration, enabling global feature mixing through local operations

### 8.4 Why This Combination

- Mamba eliminates attention's quadratic cost and KV cache memory
- Weight tying multiplies effective depth without multiplying parameters
- Low precision maximizes neuron count within 16MB
- Input-dependent gating provides the expressivity that static low-precision weights would otherwise lack

---

## 9. Incremental Research Path

### Step 1: Parameter Golf (Immediate — weeks)

Validate core ideas at extreme parameter constraint. Test Mamba + weight tying + low precision + gated FFN. Iterate locally on MacBook with MLX. Produces concrete benchmark results.

### Step 2: MoE Redundancy Analysis (Near-term — weeks)

Take OLMoE-1B-7B or similar small MoE model. Characterize expert redundancy: weight similarity, routing concentration, output rank, ablation sensitivity. Establish how much functional diversity actually exists versus how many parameters are spent. Produces foundational data for all compression work.

### Step 3: Masked FFN MoE Replacement (Near-term — months)

Average expert weights into a single 2× wide FFN. Train router to produce binary masks. Measure reconstruction error and downstream perplexity. Iterate: soft masking, low-rank deltas, variable width. Produces a practical MoE compression method.

### Step 4: Latent-Space Expert Retrieval (Medium-term — months)

Store (input, output) hidden state pairs from expert forward passes. Replace expert computation with kNN retrieval. Evaluate reconstruction quality versus datastore size. Test knowledge editability. Produces evidence on knowledge/reasoning separation.

### Step 5: Circuit Fabric Prototype (Medium-term — months)

Implement circuit fabric on a single small MoE layer. Test multiple gate primitives (LUT, Petersen, threshold). Evolve circuits to reproduce layer behavior. Measure compression ratio versus reconstruction quality. Produces proof of concept for the computational primitive change.

### Step 6: Functional Program Architecture (Long-term — quarters)

Build planning model via distillation from frontier model CoT. Implement core executors. Integrate with scheduler. Evaluate end-to-end. Produces the system-level separation of reasoning from knowledge.

Each step produces independently useful artifacts and validates or invalidates assumptions needed for subsequent steps.

---

## 10. Related Work (Comprehensive)

### Foundational
- **Fast Weight Programmers** (Schmidhuber, 1991–93): Networks generating weights for other networks
- **Self-Referential Weight Matrices** (Schmidhuber, 1993; Irie et al., 2022): Weights that modify themselves
- **HyperNetworks** (Ha, Dai & Le, 2017): Small networks generating parameters for larger networks
- **Linear Transformers as Fast Weight Programmers** (Schlag, Irie & Schmidhuber, 2021)

### Input-Dependent Architecture Components
- **Attention Residuals** (Moonshot AI, 2026): Input-dependent residual connections
- **LambdaNetworks** (Bello, 2021): Input-dependent linear projections
- **Mixture-of-Depths** (Raposo et al., 2024): Input-dependent compute allocation per token
- **Mixture-of-Recursions** (2025): Adaptive recursion depth in weight-tied transformers

### State Space Models
- **S4** (Gu et al., 2021): Structured State Spaces for sequence modeling
- **Mamba / Selective SSMs** (Gu & Dao, 2023): Input-dependent state dynamics
- **Mamba-2 / SSD** (Dao & Gu, 2024): Structured state space duality with attention
- **Mamba-3** (Lahoti et al., 2026): Inference-first SSM with complex dynamics and MIMO

### Learned Logic Circuits
- **CRIREL / Hyper-Flexible Neural Networks** (White et al., 2025): 4-neuron circuits with 24 switchable functions
- **Deep Differentiable Logic Gate Networks** (Petersen et al., 2022): Gradient-trained binary gate selection
- **Convolutional Differentiable Logic Gate Networks** (Petersen et al., 2024): Extending to convolutional patterns
- **Polynomial Surrogate Training for Ternary Logic** (2025): Three-valued logic with uncertainty
- **Weightless Neural Networks / WiSARD** (Aleksander et al., 1984): LUT-based neural computation
- **Differentiable Weightless Neural Networks** (2024): Backpropagation through LUTs

### Input-Driven Reconfiguration
- **Input-Driven Circuit Reconfiguration** (Magnasco, 2024): Signal pathway reconfiguration via inputs only
- **Hyper-Flexible Neural Networks** (White et al., 2025): Rapid function switching without weight changes

### MoE Compression
- **Expert pruning:** MoE-Pruner, NAEE, STUN
- **Expert merging:** HC-SMoE, MC-SMoE, PuzzleMoE, MergeMoE, Sub-MoE
- **Low-rank decomposition:** D2-MoE, MoE-SVD, MoLAE, MoBE, MoE-I²
- **LoRA-as-expert:** MoLA, X-LoRA, HELLoRA, MoELoRA, MixLoRA, TeamLoRA
- **Masked experts:** MoME, MoM, MoEfication, DS-MoE, Finedeep

### Knowledge Externalization
- **kNN-LM** (Khandelwal et al., 2019): Nearest-neighbor interpolation at the token level
- **RETRO** (Borgeaud et al., 2022): Retrieval-augmented pretraining; outperforms 25× larger parametric models
- **LmLm** (2025): 382M model matching LLaMA2-7B factual precision via knowledge externalization
- **Large Knowledge Models:** Latent-space retrieval of pre-encoded embeddings
- **Coconut** (2024): Continuous chain-of-thought reasoning in latent space

### Mixture of Experts
- **Sparse MoE** (Shazeer et al., 2017): Original sparsely-gated MoE
- **Switch Transformer** (Fedus et al., 2022): Top-1 routing at trillion-parameter scale
- **MoE-Mamba** (2024): Mamba + MoE integration with 2.2× fewer training steps
- **DeepSeek-V3** (2024): 256 fine-grained experts, auxiliary-loss-free load balancing
- **ReMoE** (ICLR 2025): ReLU routing as differentiable drop-in for top-k

---

## 11. Summary

This document synthesizes a research agenda built on a single principle: input-dependent computation is more parameter-efficient than fixed computation at every scale it has been tested. We trace this principle from Schmidhuber's fast weight programmers (1991) through attention, MoE, selective SSMs, and CRIREL circuits, showing that each is an instance of the same mathematical structure — inputs configuring which function is applied to other inputs.

We propose three concrete architectures at different scales of ambition:

1. **Circuit Fabric:** Replace transformer/MoE layers with hierarchical logic circuits operating on quantized bit representations. Maximum compression, fundamentally different computational primitive.

2. **MoE Expert Replacement:** Replace the router and expert ensemble with masked wide FFNs, latent retrieval, or circuit fabrics. Targets the most parameter-expensive model component.

3. **Functional Program Architecture:** Replace monolithic models with planning models coordinating domain executors through explicit computation graphs. Separates reasoning from knowledge at the system level.

These are connected by the same principle but differ in implementation risk and timeline. The practical entry point is OpenAI's Parameter Golf challenge, which tests the core ideas (weight tying, low precision, input-dependent gating, SSM backbone) under extreme parameter constraints. Success there validates the theoretical framework and sets up each subsequent step.

The field is converging on these ideas from multiple directions simultaneously — Mamba from control theory, AttnRes from depth aggregation, MoE compression from deployment economics, and CRIREL from neuroscience. The opportunity is to synthesize these convergent threads into a coherent research program.
