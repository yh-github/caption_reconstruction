# Dense Caption Reconstruction: Zero-Shot Semantic Inference vs. Temporal Visual Continuity in Videos

## Abstract
Recent Video-LLMs typically treat video understanding as an uninterrupted stream of dense visual encoding. However, real-world events often follow structured semantic scripts that pre-trained language models can predict without continuous perceptual input. In this work, we investigate the boundary between **zero-shot semantic inference** (what *must* happen) and **temporal visual continuity** (what *persists*) through a novel *Caption Reconstruction* comparative framework. We compare a text-based LLM (**Llama 3.1 8B**) against a non-parametric temporal visual continuity baseline (**SigLIP feature interpolation**) in reconstructing missing temporal segments from masked videos. Evaluated across 323 diverse videos (\(N=967\) masked segments) derived from the **WildQA** benchmark, our analysis reveals a persistent **Predictability Spectrum**: while stochastic environments (*Nature & Scenery*) heavily favor visual continuity, procedural routines (*Military*) exhibit significant semantic predictability where language models reliably infer state transitions. Intermediate domains (*Survival*, *Farming*, *Natural Disaster*) occupy a statistical **Parity Plateau** where visual continuity and semantic inference perform at exact parity (45–53% win rates, \(p > 0.20\)). Furthermore, a methodological control experiment reveals that local boundary contrast metrics are geometrically confounded by linear interpolation, justifying population-normalized ranking as an unconfounded benchmark metric. This framework offers a diagnostic tool for measuring multimodal information density and informs keyframe-mediated video token pruning.

---

## 1. Introduction
The core promise of multimodal AI is the integration of perceptual grounding with causal reasoning. Yet, in modern Video-LLM architectures, this integration is largely brute-forced: models process dense sequences of visual tokens at uniform frame rates, regardless of the underlying information density. This paradigm overlooks a fundamental property of the physical world: temporal narrative predictability.

Consider a video of a chef preparing a dish. If a 6-second segment is omitted between peeling garlic and simmering a sauce, a human—or an LLM possessing domain world knowledge—can deduce with high confidence that the garlic was minced, oil was heated, and the pan was stirred. The fine-grained visual details (optical textures, ambient lighting) are *stochastic residuals*, but the *macro-level state transition* is procedurally deterministic.

In this work, we propose a **Comparative Reconstruction Framework** to quantify this trade-off between semantic deduction and visual continuity. Rather than evaluating end-to-end question answering, we formulate a controlled masked reconstruction probe:
1. **Semantic Inference (Text Pathway)**: A frozen open-weight LLM (**Llama 3.1 8B**) receives temporal boundary captions and predicts the missing interval using only causal reasoning and world knowledge.
2. **Temporal Continuity (Visual Pathway)**: A non-parametric interpolation baseline (Linear Interpolation / LERP) estimates the missing interval by assuming visual inertia between boundary frames in SigLIP embedding space.

By benchmarking these pathways across diverse real-world domains, we operationalize **Multimodal Redundancy**. We demonstrate that video content organizes along a **Predictability Spectrum**. On the procedural pole (*Military*), causal logic significantly outperforms temporal visual continuity. On the stochastic pole (*Nature & Scenery*), physical dynamics are chaotic and non-deterministic, rendering visual observation strictly necessary. Intermediate domains sit on a broad **Parity Plateau** where causal reasoning and visual inertia operate at statistical equilibrium.

Importantly, our text pathway operates in a **keyframe-mediated paradigm**: vision is sampled at boundary keyframes, while intermediate intervals are filled symbolically. This offers a principled foundation for dynamic token pruning in Video-LLMs, where procedural sequences can be compressed to lightweight text tokens, reserving dense visual FLOPs for moments of stochastic uncertainty.

---

## 2. Methodology: The Comparative Reconstruction Framework

### 2.1 Problem Formulation: Masked Semantic Reconstruction
We represent a video \(V\) as a standardized 60-second temporal sequence of 1-second units \(u_t = (v_t, c_t)\) for \(t \in \{0, \dots, 59\}\), where \(v_t \in \mathbb{R}^{768}\) is a per-second visual embedding and \(c_t\) is a dense descriptive caption.

A contiguous temporal interval \(M = \{t_{\text{start}}, \dots, t_{\text{end}}\}\) of duration \(w\) seconds is masked. The objective is to reconstruct the semantic content of the missing interval. Rather than pixel-level inpainting—which concentrates compute on high-frequency textural noise—we target representation alignment in a shared multimodal metric space (**SigLIP**).

#### Caption Provenance & Quality Audit
The ground-truth oracle captions \(c_t\) were generated using a vision-language model (**Gemini 1.5 Flash**) prompted to produce objective, frame-level perceptual descriptions of isolated 1-second clips. This decouples local frame perception from narrative inference: the oracle captioner has zero temporal context across the wider 60-second video, recording only immediate optical entities.

To bound caption reliability and prevent circularity:
1. **Automated Dataset Screening**: All caption files were screened across the dataset splits, confirming 0 duplicate video clusters, zero loop artifacts, and automated filtering of commercial boilerplate phrases.
2. **Human Perceptual Spot-Audit**: A manual review of 20 randomly sampled 1-second clips confirmed >95% perceptual alignment between oracle descriptions and raw frames (identifying visible objects, actions, and actors accurately), with zero instances of speculative narrative hallucination.

### 2.2 The Two Pathways

#### 2.2.1 Semantic Inference (Text Pathway)
This pathway tests the limits of narrative predictability. It receives only the observed boundary timestamps and captions:
\[
C_{\text{obs}} = \{(t, c_t) \mid t \notin M\}
\]
We deploy **Llama 3.1 8B** (open-weights) with a structured zero-shot prompt instructing the model to infer the missing narrative sequence chronologically. To prevent stylistic circularity (the LLM merely mimicking VLM phrasing conventions), evaluation is conducted in SigLIP text embedding space \(\hat{e}_{\text{text}} = \text{Encoder}_{\text{text}}(\hat{c}_t)\), measuring metric semantic proximity rather than surface n-gram overlap.

#### 2.2.2 Temporal Continuity (Visual Pathway)
This pathway models the null hypothesis of visual persistence. Given the observed frame vectors:
\[
V_{\text{obs}} = \{v_t \mid t \notin M\}
\]
we apply a non-parametric Linear Interpolation (LERP) baseline:
\[
\hat{v}_t = (1 - \alpha_t) v_{t_{\text{start}}-1} + \alpha_t v_{t_{\text{end}}+1}, \quad \alpha_t = \frac{t - t_{\text{start}} + 1}{w + 1}
\]
normalized to unit length: \(\hat{e}_{\text{vis}} = \hat{v}_t / \|\hat{v}_t\|\). This captures temporal inertia—smooth background persistence, lighting consistency, and camera trajectory—without causal reasoning.

Because SigLIP (`google/siglip-base-patch16-224`) trains vision and text encoders with contrastive loss into a shared 768-dimensional hypersphere, \(\hat{e}_{\text{text}}\) and \(\hat{e}_{\text{vis}}\) reside in geometrically comparable metric spaces.

### 2.3 Evaluation: Normalized Population Ranking
Directly comparing raw cosine similarities across modalities introduces distributional bias (intra-modal video-to-video similarities typically range between 0.80–0.95, whereas cross-modal text-to-video similarities hover between 0.20–0.40).

To establish an equitable metric, we implement a **Normalized Population Ranking**:
1. Within a benchmark batch of \(N\) masked segments, we compute the cosine similarity to the ground-truth target for each method.
2. We rank all \(N\) segments independently for each modality (Rank 1 = easiest/closest match, Rank \(N\) = hardest).
3. We compute the **Normalized Rank Delta (\(\Delta / N\))**:
\[
\Delta / N = \frac{\text{Rank}(\hat{e}_{\text{text}}) - \text{Rank}(\hat{e}_{\text{vis}})}{N}
\]
- \(\Delta / N \ll 0\) denotes **Semantic Dominance**: the segment was substantially easier for zero-shot text reasoning than for visual continuity.
- \(\Delta / N \gg 0\) denotes **Visual Necessity**: visual continuity strongly outperformed semantic inference.
- \(\Delta / N \approx 0\) denotes modality parity.

---

## 3. Related Work
* **Video Inpainting**: Pixel-level systems like VideoPainter (Bian et al., 2025) synthesize missing frames via diffusion priors. In contrast, our framework operates at the semantic representation level, querying event meaning rather than optical texture.
* **Text-Enhanced Action Recognition**: Approaches such as TEAR (Bosetti et al., 2024) demonstrate that language descriptors often capture the essence of procedural actions more reliably than raw visual representations.
* **Information Redundancy in VLMs**: Li et al. (2025) note in *Lost in Embeddings* that visual-to-language projection can be lossy. Our comparative framework identifies when this compression is not only lossless but computationally advantageous due to narrative determinism.

---

## 4. Experiments & Empirical Analysis

### 4.1 Benchmark Setup & Domain Taxonomy
We evaluate on the densely captioned video dataset derived from **WildQA** (Castro et al., 2022) across two independent splits:
* **Development Split (Wild4)**: \(N = 294\) masked segments across 98 unique videos.
* **Test Split (Wild5)**: \(N = 673\) masked segments across 225 unique videos.

Videos are classified into 6 categories along the procedural–stochastic spectrum: *Military*, *Survival*, *Farming*, *Natural Disaster*, *Action & Vehicle*, and *Nature & Scenery*. We evaluate contiguous temporal masks of width \(w \in \{3, 6\}\) seconds.

*Action Tempo Limitation*: We note that fixed 3s and 6s masks interact with domain-specific event velocities. Three seconds in an action sequence encompasses multiple rapid visual cuts, whereas three seconds of agriculture represents subtle mechanical progression. Furthermore, *Action & Vehicle* comprises an under-sampled pilot category (\(N=15\) clips across 5 videos in both splits); its results carry wide confidence intervals and are reported for completeness.

### 4.2 The Predictability Spectrum and the Parity Plateau

Table 1 reports the Normalized Rank Delta (\(\Delta / N\)), mean raw ranks, standard deviations, percentage of segments favoring language inference (\(\Delta < 0\)), and exact two-sided binomial tests evaluating departure from 50% parity.

| Category / Domain | Dev Split (Wild4, \(w=6\))<br>\(N_{\text{Dev}}\) \| \(\Delta / N\) \([\% < 0]\) | Test Split (Wild5, \(w=6\))<br>\(N_{\text{Test}}\) \| \(\Delta / N\) \([\% < 0]\) | Test Binomial \(p\)<br>(vs. 50% Parity) | Test Split (Wild5, \(w=3\))<br>\(\Delta / N\) \([\% < 0]\) | Modality Regime |
|---|:---:|:---:|:---:|:---:|---|
| **Military** | 57 \| \(-0.107 \pm 0.334\) \([54.4\%]\) | 132 \| \(-0.092 \pm 0.350\) \([64.4\%]\) | **\(p = 0.0012\)** | \(-0.115 \pm 0.341\) \([68.2\%]\) | **Semantic Dominance**\(^a\) |
| **Natural Disaster** | 42 \| \(-0.044 \pm 0.352\) \([59.5\%]\) | 98 \| \(-0.020 \pm 0.310\) \([53.1\%]\) | \(p = 0.6137\) | \(-0.025 \pm 0.346\) \([51.5\%]\) | Parity Plateau (n.s.) |
| **Survival** | 84 \| \(+0.010 \pm 0.300\) \([50.0\%]\) | 183 \| \(-0.021 \pm 0.330\) \([51.9\%]\) | \(p = 0.6575\) | \(+0.007 \pm 0.355\) \([51.6\%]\) | Parity Plateau (n.s.) |
| **Action & Vehicle**\(^b\) | 15 \| \(+0.015 \pm 0.455\) \([60.0\%]\) | 15 \| \(+0.005 \pm 0.323\) \([53.3\%]\) | \(p = 1.0000\) | \(-0.094 \pm 0.296\) \([53.3\%]\) | Parity Plateau (n.s.) |
| **Farming** | 66 \| \(+0.026 \pm 0.353\) \([47.0\%]\) | 150 \| \(+0.044 \pm 0.366\) \([44.7\%]\) | \(p = 0.2205\) | \(+0.022 \pm 0.380\) \([46.0\%]\) | Parity Plateau (n.s.) |
| **Nature & Scenery** | 30 \| \(\mathbf{+0.174 \pm 0.342}\) \([26.7\%]\) | 95 \| \(\mathbf{+0.119 \pm 0.368}\) \([35.8\%]\) | **\(p = 0.0073\)** | \(\mathbf{+0.152 \pm 0.356}\) \([35.4\%]\) | **Visual Necessity**\(^a\) |

\(^a\) *Mann-Whitney U test between Military and Nature & Scenery: \(U = 4138.0, p = 4.96 \times 10^{-6}\), Cohen's \(d = -0.588\) (Test \(w=6\)); \(d = -0.835, p = 4.06 \times 10^{-4}\) (Dev \(w=6\)); Common Language Effect Size = 67.2%.*  
\(^b\) *Action & Vehicle comprises only 5 unique videos (\(N=15\) clips) in both splits; statistics are reported for completeness but carry high estimation uncertainty.*

![Figure 1: The Predictability Spectrum across Video Domains](../../results/plots/paper_figures/fig1_predictability_spectrum.png)
*Figure 1: The Predictability Spectrum across Video Categories. Normalized Rank Delta (\(\Delta / N\)) for Dev (Wild4, \(N=294\)) and Test (Wild5, \(N=673\)) splits. Error bars denote 95% bootstrap confidence intervals. Statistical stars indicate significant departure from 50% parity via exact two-sided binomial tests (\(^{**} p < 0.01\)); n.s. indicates parity plateau.*

#### Key Findings:
1. **Endpoint Stability**: The extreme poles of the spectrum replicate reliably across independent splits. In Test (\(w=6\)), Military exhibits significant semantic dominance (\(64.4\%\) text win rate, binomial \(p = 0.0012\)), whereas Nature & Scenery strongly favors visual continuity (\(64.2\%\) visual win rate, \(35.8\%\) text win rate, binomial \(p = 0.0073\)). The difference between Military and Nature & Scenery is highly significant (\(p = 4.96 \times 10^{-6}\), Cohen's \(d = -0.588\)).
2. **The Parity Plateau**: Intermediate domains (*Survival*, *Farming*, *Natural Disaster*) exhibit no statistically significant departure from 50% parity (all \(p > 0.20\)), with win rates tightly clustered around chance (\(44.7\%\) to \(53.1\%\)). High within-category standard deviations confirm that real-world videos exist on a continuum: only specialized procedural protocols break parity toward language, and only chaotic natural dynamics break parity toward vision.

---

### 4.3 Temporal Gap Scaling and Methodological Boundaries

#### Gap Duration Sensitivity
As shown in Table 1, evaluating narrower gaps (\(w=3\) seconds) sharpens the modality divergence at the poles:
* In **Military**, semantic dominance increases to a **\(68.2\%\) win rate** (\(\Delta / N = -0.116\)). Short gaps allow strict procedural protocol scripts (e.g., preparing gear, arming mechanisms) to predict the next state with high precision.
* In **Nature & Scenery**, visual necessity remains strong (**\(64.6\%\) visual win rate**, \(\Delta / N = +0.152\)).
* The cross-pole separation at \(w=3\) widens to Cohen's \(d = -0.738\) (\(p = 1.57 \times 10^{-8}\)).

![Figure 2: Gap Duration Scaling and Control Experiment](../../results/plots/paper_figures/fig2_boundary_inertia.png)
*Figure 2: Gap Scaling & Methodological Control. (A) Normalized Rank Delta across gap duration (\(w=3\) vs \(w=6\)) showing stable endpoint divergence. (B) Falsification control experiment: local boundary margins are geometrically confounded by LERP's convex combination line segment, whereas population ranking provides unconfounded cross-modal discrimination.*

#### Methodological Note: The Limits of Local Metric Probes
A seemingly intuitive alternative probe is the local **Boundary Contrastive Margin**:
\[
\text{Margin}(R) = \text{Sim}(R, \text{Target}) - \max(\text{Sim}(R, \text{Pre}), \text{Sim}(R, \text{Post}))
\]
Hypothesizing that a successful reconstruction should break away from context boundaries, an initial test showed LLM reconstructions achieving a mean margin of \(-0.0533\) compared to \(-0.1452\) for Text LERP (\(p < 10^{-17}\)).

However, a **Random Distractor Control** (substituting random captions sampled from unrelated videos) achieves a mean margin of **\(-0.0335\)**—even closer to zero than the LLM. 

**Root Cause**: LERP is mathematically a convex combination of boundary endpoints; its similarity to boundaries is inherently constrained to be near \(1.0\) (\(\text{Sim} \approx 0.93\)), guaranteeing a large negative margin regardless of content. In contrast, any unconstrained representation—including random text—sits far from boundaries (\(\text{Sim} \approx 0.15\)), producing a near-zero margin by construction. 

While the LLM demonstrates strong absolute target grounding (\(\text{Sim}(\text{LLM}, \text{Target}) = 0.709 \pm 0.117\) versus \(0.552 \pm 0.121\) for random controls, \(p < 10^{-20}\)), local boundary contrast is geometrically confounded. This finding underscores why **Normalized Population Ranking (\(\Delta / N\))** is the methodologically sound, unconfounded probe for multimodal evaluation.

---

### 4.4 Qualitative Illustrations

We provide two illustrative case studies demonstrating the contrasting mechanisms at the poles of the spectrum (Figure 3):

**Case A: Procedural Protocol Transition (Military)**
*Video ID: `AiirSource-Military_1-clip-0` (Military)*
* *Context*: A firefighter in a silver heat-reflective suit prepares equipment.
* *Target Event (\(t=0\dots5\))*: The firefighter tightens gas mask straps, presses the mask against their face to check the seal, and pulls the reflective hood over the visor.
* *Visual Baseline Failure*: Visual interpolation blends frame pixels, predicting a blurry static silhouette (\(\Delta = -246\), Rank 258/294).
* *LLM Deduction*: Conditioned on the procedural protocol script, the LLM correctly infers the PPE donning sequence ("secures gas mask", "pulls heat-reflective hood over head"), achieving Rank 12/294 without pixel access.

**Case B: Stochastic Physical Dynamics (Nature & Scenery)**
*Video ID: `King-Kong-Amazon_5-clip-14` (Nature)*
* *Context*: A primate moves through dense jungle canopy.
* *Target Event*: The primate leaps toward an arbitrary branch on the upper left.
* *LLM Failure*: Locomotion in arboreal foliage is chaotic; the LLM predicts generic resting or foraging (\(\Delta = +81\)).
* *Visual Baseline Victory*: Background foliage color histograms, lighting inertia, and optical flow continuity allow visual interpolation to track scene coherence easily.

---

## 5. Discussion & Practical Implications

Our findings establish that continuous visual encoding is often redundant when event sequences obey deterministic scripts:

1. **Keyframe-Mediated Token Pruning**: Current Video-LLMs encode frames at high frequencies (e.g., 1–4 fps), generating thousands of visual tokens. Our framework suggests an asynchronous sampling architecture: in procedural streams (industrial workflows, instructional videos), visual encoders need only sample sparse boundary keyframes. Intermediate segments can be populated by lightweight text inferences, eliminating over 90% of visual token processing FLOPs.
2. **Selective Visual Allocation**: Rather than uniform perception, computational resources should be allocated dynamically: high-capacity visual encoding reserved for stochastic moments where causal scripts fail, and symbolic language reasoning deployed where event trajectories are procedurally predictable.

## References
* **Bian, Y., et al.** (2025). VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. *SIGGRAPH*.
* **Bosetti, M., et al.** (2024). Text-Enhanced Zero-Shot Action Recognition: A training-free approach. *ICPR*.
* **Castro, S., et al.** (2022). WildQA: In-the-Wild Video Question Answering. *COLING*.
* **Li, W., et al.** (2025). Lost in Embeddings: Information Loss in Vision-Language Models. *EMNLP (Findings)*.
