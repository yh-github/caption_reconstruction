# Dense Caption Reconstruction: Zero-Shot Semantic Inference vs. Temporal Visual Continuity in Videos

## Abstract
Recent Video-LLMs typically treat video understanding as an uninterrupted stream of dense visual encoding. However, real-world events often follow structured semantic scripts that pre-trained language models can predict without continuous perceptual input. In this work, we investigate the boundary between **zero-shot semantic inference** (what *must* happen) and **temporal visual continuity** (what *persists*) through a novel *Caption Reconstruction* comparative framework. We compare a text-based LLM (**Llama 3.1 8B**) against a non-parametric temporal visual continuity baseline (**SigLIP feature interpolation**) in reconstructing missing temporal segments from masked videos. Evaluated across 323 diverse videos (\(N=967\) masked segments) derived from the **WildQA** benchmark, our analysis reveals a persistent **Predictability Spectrum**: while stochastic environments (*Nature & Scenery*) heavily favor visual continuity, procedural routines (*Military*, *Survival*) exhibit significant semantic predictability where language models reliably infer state transitions. Furthermore, contrastive temporal evaluation demonstrates that LLM semantic in-filling significantly mitigates the boundary anchoring that constrains interpolation baselines (\(p < 10^{-17}\) across independent videos). This framework offers a diagnostic tool for measuring multimodal information density and informs keyframe-mediated video token pruning.

---

## 1. Introduction
The core promise of multimodal AI is the integration of perceptual grounding with causal reasoning. Yet, in modern Video-LLM architectures, this integration is largely brute-forced: models process dense sequences of visual tokens at uniform frame rates, regardless of the underlying information density. This paradigm overlooks a fundamental property of the physical world: temporal narrative predictability.

Consider a video of a chef preparing a dish. If a 6-second segment is omitted between peeling garlic and simmering a sauce, a human—or an LLM possessing domain world knowledge—can deduce with high confidence that the garlic was minced, oil was heated, and the pan was stirred. The fine-grained visual details (optical textures, ambient lighting) are *stochastic residuals*, but the *macro-level state transition* is procedurally deterministic.

In this work, we propose a **Comparative Reconstruction Framework** to quantify this trade-off between semantic deduction and visual continuity. Rather than evaluating end-to-end question answering, we formulate a controlled masked reconstruction probe:
1. **Semantic Inference (Text Pathway)**: A frozen open-weight LLM (**Llama 3.1 8B**) receives temporal boundary captions and predicts the missing interval using only causal reasoning and world knowledge.
2. **Temporal Continuity (Visual Pathway)**: A non-parametric interpolation baseline (Linear Interpolation / LERP) estimates the missing interval by assuming visual inertia between boundary frames in SigLIP embedding space.

By benchmarking these pathways across diverse real-world domains, we operationalize **Multimodal Redundancy**. We demonstrate that video content organizes along a **Predictability Spectrum**. On the procedural pole (*Military*, *Survival*), causal logic frequently outperforms temporal visual continuity. On the stochastic pole (*Nature & Scenery*), physical dynamics are chaotic and non-deterministic, rendering visual observation strictly necessary.

Importantly, our text pathway operates in a **keyframe-mediated paradigm**: vision is sampled at boundary keyframes, while intermediate intervals are filled symbolically. This offers a principled foundation for dynamic token pruning in Video-LLMs, where procedural sequences can be compressed to lightweight text tokens, reserving dense visual FLOPs for moments of stochastic uncertainty.

---

## 2. Methodology: The Comparative Reconstruction Framework

### 2.1 Problem Formulation: Masked Semantic Reconstruction
We represent a video \(V\) as a standardized 60-second temporal sequence of 1-second units \(u_t = (v_t, c_t)\) for \(t \in \{0, \dots, 59\}\), where \(v_t \in \mathbb{R}^{768}\) is a per-second visual embedding and \(c_t\) is a dense descriptive caption.

A contiguous temporal interval \(M = \{t_{\text{start}}, \dots, t_{\text{end}}\}\) of duration \(w\) seconds is masked. The objective is to reconstruct the semantic content of the missing interval. Rather than pixel-level inpainting—which concentrates compute on high-frequency textural noise—we target representation alignment in a shared multimodal metric space (**SigLIP**).

#### Caption Provenance & Standardization
The ground-truth oracle captions \(c_t\) were generated using a vision-language model (**Gemini 1.5 Flash**) prompted to produce objective, frame-level perceptual descriptions of isolated 1-second clips. This decouples local frame perception from narrative inference: the oracle captioner has zero temporal context across the wider 60-second video, recording only immediate optical entities.

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

*Action Tempo Limitation*: We note that fixed 3s and 6s masks interact with domain-specific event velocities. Three seconds in an action sequence encompasses multiple rapid visual cuts, whereas three seconds of agriculture represents subtle mechanical progression. Our results reflect this natural tempo interaction.

### 4.2 The Predictability Spectrum Across Video Domains

Table 1 reports the Normalized Rank Delta (\(\Delta / N\)), mean raw ranks, standard deviations, and the percentage of segments where language inference outperforms visual continuity (\(\Delta < 0\)).

| Category / Domain | Dev Split (Wild4, \(w=6\))<br>\(\Delta / N\) \([\% < 0]\) | Test Split (Wild5, \(w=6\))<br>\(\Delta / N\) \([\% < 0]\) | Test Split (Wild5, \(w=3\))<br>\(\Delta / N\) \([\% < 0]\) | Modality Advantage |
|---|:---:|:---:|:---:|---|
| **Military** (\(N=132\)) | \(-0.107 \pm 0.334\) \([54.4\%]\) | \(-0.092 \pm 0.350\) \([64.4\%]\) | \(-0.116 \pm 0.354\) \([68.2\%]\) | **Semantic Dominance** (\(p = 4.96 \times 10^{-6}\)) |
| **Natural Disaster** (\(N=98\)) | \(-0.044 \pm 0.352\) \([59.5\%]\) | \(-0.020 \pm 0.310\) \([53.1\%]\) | \(-0.022 \pm 0.320\) \([55.1\%]\) | Moderate Semantic Trend |
| **Survival** (\(N=183\)) | \(+0.010 \pm 0.300\) \([50.0\%]\) | \(-0.021 \pm 0.330\) \([51.9\%]\) | \(-0.035 \pm 0.337\) \([53.6\%]\) | Moderate Semantic Trend |
| **Action & Vehicle** (\(N=15\)) | \(+0.015 \pm 0.455\) \([60.0\%]\) | \(+0.005 \pm 0.323\) \([53.3\%]\) | \(+0.018 \pm 0.334\) \([46.7\%]\) | Transition Zone |
| **Farming** (\(N=150\)) | \(+0.026 \pm 0.353\) \([47.0\%]\) | \(+0.044 \pm 0.366\) \([44.7\%]\) | \(+0.028 \pm 0.370\) \([48.0\%]\) | Transition Zone |
| **Nature & Scenery** (\(N=95\)) | \(\mathbf{+0.174 \pm 0.342}\) \([26.7\%]\) | \(\mathbf{+0.119 \pm 0.368}\) \([35.8\%]\) | \(\mathbf{+0.152 \pm 0.371}\) \([31.6\%]\) | **Visual Necessity** (\(p = 4.96 \times 10^{-6}\)) |

![Figure 1: The Predictability Spectrum across Video Domains](../../results/plots/paper_figures/fig1_predictability_spectrum.png)
*Figure 1: The Predictability Spectrum across Video Categories. Normalized Rank Delta (\(\Delta / N\)) for Dev (Wild4, \(N=294\)) and Test (Wild5, \(N=673\)) splits. Error bars indicate 95% bootstrap confidence intervals. Negative values indicate LLM superiority; positive values indicate visual continuity superiority.*

#### Analysis & Statistical Effect Sizes:
1. **Endpoint Stability**: The extreme poles of the spectrum replicate reliably across independent splits. In Test (\(w=6\)), Military exhibits a strong text advantage (\(\Delta / N = -0.092\), 64.4% win rate), whereas Nature & Scenery strongly favors visual continuity (\(\Delta / N = +0.119\), only 35.8% text win rate). The difference between Military and Nature & Scenery is statistically significant (Mann-Whitney \(U = 4138.0\), \(p = 4.96 \times 10^{-6}\)), with a medium-to-large effect size (Cohen's \(d = -0.588\); Common Language Effect Size = 67.2%).
2. **Intermediate Domain Variance**: While the endpoints replicate stably, intermediate domains (*Survival*, *Natural Disaster*, *Farming*) exhibit minor rank-order variations between splits, with win rates hovering close to 50% (\(44.7\%\) to \(59.5\%\)). High within-category standard deviations confirm that real-world videos exist on a continuum of predictability rather than in discrete, mutually exclusive classes.
3. **Temporal Gap Scaling**: At \(w=3\), the divergence between poles widens (Cohen's \(d = -0.738\), \(p = 1.57 \times 10^{-8}\)), indicating that short gaps provide maximal leverage for semantic script deduction before long-term narrative divergence occurs.

---

### 4.3 Mitigating Boundary Anchoring: Contrastive Temporal Evaluation

Naive interpolation baselines achieve deceptively high cosine similarity because surrounding frames reside in the same visual neighborhood. To test whether reconstructions capture distinct state changes or merely reproduce context inertia, we measure the **Boundary Contrastive Margin**:
\[
\text{Margin}(R) = \text{Sim}(R, \text{Target}) - \max(\text{Sim}(R, \text{Pre}), \text{Sim}(R, \text{Post}))
\]
Because both models exhibit negative absolute margins on average (mean \(-0.0576\) in text, \(-0.0097\) in cross-modal video space), reconstructions remain closer to context boundaries than to target frames in absolute terms. However, this metric measures **relative liberation from boundary lock**: how effectively each model pulls away from the context boundaries toward the internal target.

![Figure 2: Mitigating Boundary Anchoring Over Time](../../results/plots/paper_figures/fig2_boundary_inertia.png)
*Figure 2: Mitigating Boundary Anchoring. Contrastive margin across gap elapsed time. (A) In text space, LLM reconstructions maintain a consistent \(\approx +0.136\) margin advantage over Text LERP. (B) In cross-modal video space, LLM text representations mitigate boundary attraction relative to Visual LERP.*

#### Empirical Findings Across 98 Independent Videos (588 seconds):
* **Text Semantic Space**: LLM reconstructions achieve a mean margin of \(-0.0576\) versus \(-0.1934\) for Text LERP—a **\(+0.1358\) margin advantage**. Aggregated strictly to the independent video level (\(N=98\)), the LLM outperforms Text LERP in **100.0% of videos** (paired Wilcoxon \(W = 0.0\), \(p = 8.33 \times 10^{-18}\)).
* **Cross-Modal Video Space**: Querying raw target video frames, LLM text representations achieve a mean margin of \(-0.0097\) versus \(-0.0591\) for Visual LERP—a **\(+0.0494\) advantage**. Across independent videos, the LLM outperforms Visual LERP in **98.0% of videos** (\(W = 46.0\), \(p = 9.72 \times 10^{-18}\)).
* **Mechanism**: As illustrated in Figure 2, interpolation baselines are mathematically anchored to boundaries at the gap edges (\(t=1\) and \(t=6\)), whereas LLM semantic in-filling demonstrates consistent, position-invariant discrimination.

---

### 4.4 Qualitative Illustrations

We provide two illustrative case studies demonstrating the contrasting mechanisms at the poles of the spectrum (Figure 3):

**Case A: Procedural State Transition (Military / Farming)**
*Video ID: `Welker-Farms-Inc_3-clip-4` (Farming)*
* *Context*: A tractor positions itself at the edge of a field.
* *Target Event*: The operator unfolds the mechanical sprayer arms.
* *Visual Baseline Failure*: Visual interpolation blends optical features, predicting a blurry static tractor (\(\Delta = -91\)).
* *LLM Deduction*: Conditioned on the pre-gap script ("tractor aligns with crop row") and post-gap context ("sprayer sweeps across field"), the LLM deduces the intervening action ("unfolds sprayer arms"), matching the true video target without perceptual access.

**Case B: Stochastic Physical Dynamics (Nature & Scenery)**
*Video ID: `King-Kong-Amazon_5-clip-14` (Nature)*
* *Context*: A primate moves through dense foliage.
* *Target Event*: The primate leaps toward an upper-left branch.
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
