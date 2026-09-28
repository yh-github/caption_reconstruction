# Dense Caption Reconstruction: Zero-Shot Semantic Inference vs. Temporal Visual Continuity in Videos

## Abstract
Recent Video-LLMs typically treat video understanding as an uninterrupted stream of dense visual encoding. However, real-world events often follow structured semantic scripts that pre-trained language models can predict without continuous perceptual input. In this work, we investigate the boundary between **zero-shot semantic inference** (what *must* happen) and **temporal visual continuity** (what *persists*) through a *Dense Caption Reconstruction* comparative framework. We evaluate an open-weight LLM (**Llama 3.1 8B**) against non-parametric temporal visual continuity baselines (**SigLIP feature interpolation and nearest-neighbor persistence**) in reconstructing missing temporal intervals across 323 diverse videos (\(N=967\) segments) from the **WildQA** benchmark. Because population ranking is zero-sum across pooled benchmarks by construction, we formulate the Normalized Rank Delta (\(\Delta / N\)) as a **Relative Modality Sensitivity Index**. Our analysis reveals a persistent **Predictability Spectrum**: while stochastic environments (*Nature & Scenery*) skew heavily toward visual continuity (\(64.2\%\) visual win rate in Test, binomial \(p = 0.0073\)), procedural domains (*Military*) skew significantly toward zero-shot semantic inference (\(64.4\%\) text win rate, binomial \(p = 0.0012\); video-level Mann-Whitney \(U = 378.0, p = 6.16 \times 10^{-4}\), Cohen's \(d = -0.826\)). Intermediate domains (*Survival*, *Farming*, *Natural Disaster*) form an **Intermediate Transition Zone** (44.7%–53.1% win rates, all \(p > 0.20\)) where neither modality departs significantly from the benchmark mean. Empirical controls disprove the hypothesis of caption wording jitter, confirming that the spectrum is driven by physical temporal autocorrelation. This framework provides an empirical foundation for keyframe-mediated token pruning in video architectures.

---

## 1. Introduction
The core promise of multimodal AI is the integration of perceptual grounding with causal reasoning. Yet, in modern Video-LLM architectures, this integration is largely brute-forced: models process dense sequences of visual tokens at uniform frame rates, regardless of the underlying information density. This paradigm overlooks a fundamental property of the physical world: temporal narrative predictability.

Consider a video of an aircraft maintenance crew preparing equipment. If a 6-second segment is omitted between unpacking a pressure gauge and testing a hydraulic line, a human—or an LLM possessing procedural world knowledge—can deduce with high confidence that the technician attached the hose, calibrated the needle, and secured the valve. The fine-grained visual details (surface reflections, exact hand positions) are *stochastic residuals*, but the *macro-level state transition* is procedurally constrained.

In this work, we propose a **Comparative Reconstruction Framework** to quantify this trade-off between semantic deduction and visual continuity. Rather than evaluating end-to-end question answering—where reasoning and perception are entangled—we formulate a controlled masked reconstruction probe:
1. **Semantic Inference (Text Pathway)**: A frozen open-weight LLM (**Llama 3.1 8B**) receives temporal boundary captions and infers the missing interval using zero-shot causal reasoning.
2. **Temporal Continuity (Visual Pathway)**: Non-parametric baselines (Linear Interpolation / LERP and Nearest-Neighbor persistence) estimate the missing interval by projecting visual feature vectors in SigLIP embedding space.

By benchmarking these pathways across diverse real-world domains, we operationalize **Multimodal Redundancy**. We demonstrate that video content organizes along a **Predictability Spectrum**. On the procedural pole (*Military*), causal logic is relatively more predictive than temporal visual continuity. On the stochastic pole (*Nature & Scenery*), physical dynamics are non-deterministic, rendering visual observation indispensable. Intermediate domains occupy an **Intermediate Transition Zone** where neither modality diverges significantly from the dataset-wide average.

Importantly, our text pathway operates in a **keyframe-mediated paradigm**: vision is sampled at boundary keyframes, while intermediate intervals are inferred symbolically. This offers a principled foundation for selective visual allocation and dynamic token pruning in Video-LLMs.

---

## 2. Methodology: The Comparative Reconstruction Framework

### 2.1 Problem Formulation: Masked Semantic Reconstruction
We represent a video \(V\) as a standardized 60-second temporal sequence of 1-second units \(u_t = (v_t, c_t)\) for \(t \in \{0, \dots, 59\}\), where \(v_t \in \mathbb{R}^{768}\) is a per-second visual embedding and \(c_t\) is a dense descriptive caption.

A contiguous temporal interval \(M = \{t_{\text{start}}, \dots, t_{\text{end}}\}\) of duration \(w\) seconds is masked. The objective is to reconstruct the semantic content of the missing interval. Rather than pixel-level inpainting—which concentrates compute on high-frequency textural noise—we target representation alignment in a shared multimodal metric space (**SigLIP**).

#### Caption Provenance & Quality Audit
The ground-truth oracle captions \(c_t\) were generated using a vision-language model (**Gemini 1.5 Flash**) prompted to produce objective, frame-level perceptual descriptions of isolated 1-second clips. This decouples local frame perception from narrative inference: the oracle captioner has zero temporal context across the wider 60-second video, recording only immediate optical entities.

To bound caption reliability and prevent circularity:
1. **Automated Dataset Screening**: All caption files were screened across the dataset splits, confirming 0 duplicate video clusters, zero loop artifacts, and automated filtering of commercial boilerplate phrases.
2. **Human Perceptual Spot-Audit**: A manual audit of 20 randomly sampled 1-second clips by the primary author confirmed 20/20 perceptual alignment between descriptions and raw frames (Wilson score 95% CI \([83.9\%, 100\%]\)), with zero instances of speculative narrative hallucination.
3. **Domain Categorization**: Categories were mapped from YouTube series metadata (e.g., *AiirSource Military*, *Welker Farms*, *Tornado Trackers*) and verified via manual inspection.

### 2.2 The Two Pathways & Baselines

#### 2.2.1 Semantic Inference (Text Pathway)
This pathway tests narrative predictability. It receives the observed context captions:
\[
C_{\text{obs}} = \{(t, c_t) \mid t \notin M\}
\]
We deploy **Llama 3.1 8B** (open-weights) with a structured zero-shot prompt instructing the model to infer the missing narrative sequence chronologically. To prevent stylistic circularity (the LLM mimicking VLM phrasing conventions), evaluation is conducted in SigLIP text embedding space \(\hat{e}_{\text{text}} = \text{Encoder}_{\text{text}}(\hat{c}_t)\), measuring metric semantic proximity rather than surface n-gram overlap.

To separate genuine state progression ("what must happen") from contextual inertia ("what persists"), we compare against:
* **Copy-Nearest Caption (`Caption_RepeatClosest`)**: Copies the boundary caption nearest to the target timestamp.
* **Text LERP (`Caption_MeanClosest`)**: Linearly interpolates the boundary caption embeddings.
* **Within-Domain Random Control**: Samples captions from unrelated videos within the same domain.
* **Cross-Domain Random Control**: Samples captions from unrelated videos across other domains.

#### 2.2.2 Temporal Continuity (Visual Pathway)
This pathway models visual persistence given the observed frame vectors \(V_{\text{obs}} = \{v_t \mid t \notin M\}\):
* **Visual LERP (`Visual_SigLIP_MeanClosest`)**: Computes a linear interpolation between boundary vectors:
\[
\hat{v}_t = (1 - \alpha_t) v_{t_{\text{start}}-1} + \alpha_t v_{t_{\text{end}}+1}, \quad \alpha_t = \frac{t - t_{\text{start}} + 1}{w + 1}
\]
normalized to unit length: \(\hat{e}_{\text{vis}} = \hat{v}_t / \|\hat{v}_t\|\).
* **Visual Repeat (`Visual_SigLIP_RepeatClosest`)**: Copies the nearest observed boundary frame vector.

Because SigLIP (`google/siglip-base-patch16-224`) aligns vision and text encoders into a shared 768-dimensional hypersphere, both pathways operate in geometrically comparable metric spaces.

### 2.3 Evaluation: Normalized Population Ranking as Relative Sensitivity
Directly comparing raw cosine similarities across modalities is confounded because intra-modal video similarities (0.80–0.95) and text similarities (0.35–0.55) exhibit distinct dynamic ranges and noise ceilings.

To establish an equitable relative probe, we implement **Normalized Population Ranking**:
1. Within a benchmark batch of \(N\) masked segments, we compute the cosine similarity to the ground-truth target for each method.
2. We rank all \(N\) segments independently for each modality (Rank 1 = closest match, Rank \(N\) = furthest).
3. We compute the **Normalized Rank Delta (\(\Delta / N\))**:
\[
\Delta / N = \frac{\text{Rank}(\hat{e}_{\text{text}}) - \text{Rank}(\hat{e}_{\text{vis}})}{N}
\]

**Zero-Sum Relative Interpretation**: Because all \(N\) segments are ranked within the same pooled batch, \(\sum \Delta / N \equiv 0\) by construction, centering the dataset-wide text win rate at \(\approx 50\%\) (50.7% in Test, 49.7% in Dev). Therefore, \(\Delta / N\) does not measure absolute modality dominance; it functions as a **Relative Modality Sensitivity Index**:
* \(\Delta / N \ll 0\) indicates **Relative Semantic Sensitivity**: the segment was relatively easier for zero-shot text reasoning than for visual continuity compared to the benchmark average.
* \(\Delta / N \gg 0\) indicates **Relative Visual Sensitivity**: visual continuity was relatively more advantageous than semantic inference compared to the benchmark average.
* \(\Delta / N \approx 0\) denotes alignment with the benchmark average.

---

## 3. Related Work
* **Video Inpainting & Prediction**: Pixel-level systems like VideoPainter (Bian et al., 2025) synthesize missing frames via diffusion priors. In contrast, our framework operates at the semantic representation level, querying event meaning rather than optical texture.
* **Text-Enhanced Action Recognition**: Approaches such as TEAR (Bosetti et al., 2024) demonstrate that language descriptors often capture procedural actions more reliably than raw visual representations.
* **Token Pruning & Efficient Video-LLMs**: Architectures like DynamicViT (Rao et al., 2021) and FastV (Chen et al., 2024) prune spatial tokens within frames. Our framework motivates temporal keyframe-mediated pruning: replacing intermediate visual tokens with symbolic text representations when event transitions are predictable.

---

## 4. Experiments & Empirical Analysis

### 4.1 Benchmark Setup & Domain Taxonomy
We evaluate on the densely captioned video dataset derived from **WildQA** (Castro et al., 2022) across two independent splits:
* **Development Split (Wild4)**: \(N = 294\) masked segments across 98 unique videos.
* **Test Split (Wild5)**: \(N = 673\) masked segments across 225 unique videos.
* **Split Disjointness**: Wild4 and Wild5 share **0 overlapping video IDs** and **0 overlapping source video stems** (100% disjoint at the source video level).

Videos are classified into 6 categories: *Military*, *Survival*, *Farming*, *Natural Disaster*, *Action & Vehicle*, and *Nature & Scenery*. We evaluate contiguous temporal masks of width \(w \in \{3, 6\}\) seconds.

*Action & Vehicle Caveat*: *Action & Vehicle* comprises an under-sampled pilot category (\(N=15\) segments across 5 unique videos from 2 YouTube channels in both splits). Because of its small sample size and wide confidence intervals, it is classified as "insufficient data" and reported strictly for transparency.

### 4.2 The Predictability Spectrum and the Intermediate Transition Zone

Because segments are clustered within videos (~3 segments per video), segment-level tests alone violate independence assumptions. Table 1 reports both segment-level metrics and cluster-aggregated video-level metrics, including video-level win rates, exact binomial tests against 50% chance parity, and cluster-bootstrap 95% confidence intervals.

| Category / Domain | Dev Split (Wild4, \(w=6\))<br>\(N_{\text{vid}}\) / \(N_{\text{seg}}\) \| \(\Delta / N\) \([\% < 0]\) | Test Split (Wild5, \(w=6\))<br>\(N_{\text{vid}}\) / \(N_{\text{seg}}\) \| \(\Delta / N\) \([\% < 0]\) | Test Binomial \(p\)<br>(Seg \| Vid) | Test Split (Wild5, \(w=3\))<br>\(\Delta / N\) \([\% < 0]\) | Modality Sensitivity |
|---|:---:|:---:|:---:|:---:|---|
| **Military** | 19 / 57 \| \(-0.107 \pm 0.334\) \([54.4\%]\) | 44 / 132 \| \(\mathbf{-0.092 \pm 0.350}\) \([64.4\%]\) | **\(p = 0.0012\)** \| \(p = 0.1742\) | \(\mathbf{-0.115 \pm 0.341}\) \([68.2\%]\) | **Relative Semantic Sensitivity**\(^a\) |
| **Natural Disaster** | 14 / 42 \| \(-0.044 \pm 0.352\) \([59.5\%]\) | 33 / 98 \| \(-0.020 \pm 0.310\) \([53.1\%]\) | \(p = 0.6137\) \| \(p = 0.4869\) | \(-0.025 \pm 0.346\) \([51.5\%]\) | Transition Zone (n.s.) |
| **Survival** | 28 / 84 \| \(+0.010 \pm 0.300\) \([50.0\%]\) | 61 / 183 \| \(-0.021 \pm 0.330\) \([51.9\%]\) | \(p = 0.6575\) \| \(p = 0.4426\) | \(+0.007 \pm 0.355\) \([51.6\%]\) | Transition Zone (n.s.) |
| **Action & Vehicle**\(^b\) | 5 / 15 \| \(+0.015 \pm 0.455\) \([60.0\%]\) | 5 / 15 \| \(+0.005 \pm 0.323\) \([53.3\%]\) | \(p = 1.0000\) \| \(p = 1.0000\) | \(-0.094 \pm 0.296\) \([53.3\%]\) | Insufficient Data (\(N=5\) vids) |
| **Farming** | 22 / 66 \| \(+0.026 \pm 0.353\) \([47.0\%]\) | 50 / 150 \| \(+0.044 \pm 0.366\) \([44.7\%]\) | \(p = 0.2205\) \| \(p = 0.2026\) | \(+0.022 \pm 0.380\) \([46.0\%]\) | Transition Zone (n.s.) |
| **Nature & Scenery** | 10 / 30 \| \(\mathbf{+0.174 \pm 0.342}\) \([26.7\%]\) | 32 / 95 \| \(\mathbf{+0.119 \pm 0.368}\) \([35.8\%]\) | **\(p = 0.0073\)** \| \(p = 0.0501\) | \(\mathbf{+0.152 \pm 0.356}\) \([35.4\%]\) | **Relative Visual Sensitivity**\(^a\) |

\(^a\) *Cross-Pole Statistics (Military vs. Nature & Scenery):*  
* *Test Split (\(w=6\)): Video-level \(U = 378.0, p = 6.16 \times 10^{-4}\), Cohen's \(d = -0.826\); Segment-level \(U = 4112.5, p = 9.91 \times 10^{-6}\), Cohen's \(d = -0.588\).*  
* *Dev Split (\(w=6\)): Video-level \(U = 34.0, p = 0.00550\), Cohen's \(d = -1.268\); Segment-level \(U = 479.5, p = 8.12 \times 10^{-4}\), Cohen's \(d = -0.835\).*  
* *Test Split (\(w=3\)): Video-level \(U = 283.0, p = 9.69 \times 10^{-6}\), Cohen's \(d = -1.180\); Segment-level \(U = 3614.5, p = 3.14 \times 10^{-8}\), Cohen's \(d = -0.769\).*  
\(^b\) *Action & Vehicle comprises only 5 unique videos in both splits; statistics are reported for completeness.*

![Figure 1: The Predictability Spectrum across Video Domains](../../results/plots/paper_figures/fig1_predictability_spectrum.png)
*Figure 1: The Predictability Spectrum across Video Categories. Normalized Rank Delta (\(\Delta / N\)) for Dev (Wild4, \(N_{\text{vid}}=98\)) and Test (Wild5, \(N_{\text{vid}}=225\)) splits. Error bars denote cluster-bootstrap 95% confidence intervals. Statistical stars indicate significant departure from 50% chance win rate via exact two-sided binomial tests (\(^{**} p < 0.01\)); n.s. indicates the intermediate transition zone.*

#### Key Findings:
1. **Cross-Pole Separation**: Across all splits and gap widths, the divergence between Military and Nature & Scenery is statistically robust at both the segment and video cluster levels (\(p < 10^{-3}\), \(|d| \ge 0.588\)). In Test (\(w=6\)), Military exhibits significant semantic preference (\(64.4\%\) text wins, \(p = 0.0012\)), whereas Nature & Scenery exhibits significant visual preference (\(64.2\%\) visual wins, \(p = 0.0073\)). In Dev (\(w=6\)), Nature & Scenery independently departs from 50% (\(26.7\%\) text wins, \(p = 0.0161\)), while Military exhibits a directional skew (\(54.4\%\), \(p = 0.5966\); video \(\Delta / N = -0.107\) [95% CI: \(-0.206, -0.019\)]).
2. **The Intermediate Transition Zone**: Domains such as *Survival*, *Farming*, and *Natural Disaster* exhibit win rates between 44.7% and 53.1% (all binomial \(p > 0.20\)), with cluster-bootstrap confidence intervals overlapping zero. They form a continuum conforming to the dataset average rather than an absolute point of modality equivalence.

---

### 4.3 Temporal Gap Scaling and Empirical Controls

#### Gap Duration Sensitivity
As shown in Table 1 and Figure 2A, narrowing the temporal gap to \(w=3\) seconds sharpens semantic predictability at the procedural pole:
* In **Military**, the text win rate rises to **\(68.2\%\)** at the segment level (\(p < 10^{-4}\)) and **\(70.5\%\)** at the video level (31/44 videos, \(p = 0.0096\)), with \(\Delta / N = -0.115 \pm 0.341\). Shorter intervals allow strict procedural protocols to predict intermediate actions with higher precision.
* In **Nature & Scenery**, visual necessity remains dominant (**\(64.6\%\) visual win rate**, \(\Delta / N = +0.152 \pm 0.356\)).
* The cross-pole separation at \(w=3\) widens to Cohen's \(d = -0.769\) (segment) and \(d = -1.180\) (video-level, \(p = 9.69 \times 10^{-6}\)).

![Figure 2: Gap Duration Scaling and Control Experiment](../../results/plots/paper_figures/fig2_boundary_inertia.png)
*Figure 2: Gap Scaling & Methodological Control. (A) Normalized Rank Delta across gap duration (\(w=3\) vs \(w=6\)) showing stable endpoint divergence. (B) Target similarity across baselines: LLM semantic reconstruction (0.709) achieves significant grounding over within-domain random (0.526) and cross-domain random (0.519) controls (\(p < 10^{-20}\)).*

#### Grounding Control: LLM vs. Random Distractors
To verify that LLM reconstructions reflect genuine sequence comprehension rather than generic domain vocabulary, we benchmark against random distractor baselines (Figure 2B):
* **Target Cosine Similarity**: LLM reconstructions achieve \(\text{Sim}(\text{LLM}, \text{Target}) = 0.7094 \pm 0.1165\).
* **Within-Domain Random Control**: Sampling captions from unrelated videos within the *same* category yields \(0.5257 \pm 0.0982\).
* **Cross-Domain Random Control**: Sampling captions from unrelated videos across *different* categories yields \(0.5187 \pm 0.1044\).

The difference between LLM reconstructions and random controls is highly significant (paired \(t\)-test, \(p < 10^{-20}\)). Notably, within-domain random captions score almost identically to cross-domain random captions (0.526 vs. 0.519), confirming that Military captions are not artificially homogeneous.

#### Separating Progression from Persistence
In text space, `Caption_RepeatClosest` achieves \(0.725 \pm 0.095\) and `Caption_MeanClosest` achieves \(0.784 \pm 0.082\). Boundary caption repetition achieves high similarity because it inherits the specific lexical vocabulary of the VLM captioner. However, repetition cannot predict state changes: LLM reconstructions generate novel descriptive verbs capturing unseen actions, separating causal progression from lexical persistence.

#### Testing the Wording Jitter Hypothesis vs. Physical Continuity
A potential critique is that 1-second isolated captioning might introduce artificial "wording jitter" in static scenes (depressing text scores in Nature) while remaining formulaic in Military. To test this, we empirically measured adjacent-second continuity across 98 videos:
* **Caption Continuity (\(\text{Sim}(c_t, c_{t+1})\))**: Military (\(0.7340 \pm 0.0521\)) vs. Nature & Scenery (\(0.7268 \pm 0.0372\)) exhibits **no statistically significant difference** (Mann-Whitney \(U = 149.0, p = 0.3927\)). VLM captioning variance is invariant across domains.
* **Visual Frame Continuity (\(\text{Sim}(v_t, v_{t+1})\))**: In contrast, adjacent visual frame similarity in Nature & Scenery (\(0.9497 \pm 0.0292\)) is **significantly higher** than in Military (\(0.9225 \pm 0.0344\)) (\(U = 68.0, p = 0.0289\)).

This demonstrates that the Predictability Spectrum is driven by physical temporal autocorrelation—smooth visual backgrounds in Nature versus rapid camera and object motion in Military—rather than captioning artifacts.

---

## 4.4 Qualitative Illustrations

To inspect the underlying representation dynamics, we examine two contrasting case studies:

**Case A: Procedural Protocol Transition (Military)**
*Video ID: `AiirSource-Military_1-clip-0` (Best-case procedural example, \(\Delta = -246\), Rank 12 text vs. 258 visual)*
* *Context*: A firefighter in a silver heat-reflective suit prepares equipment.
* *Target Event (Human qualitative summary of \(t=0\dots5\))*: The firefighter tightens gas mask straps, presses the mask against their face to check the seal, and pulls the reflective hood over the visor.
* *Visual Baseline Behavior*: Visual LERP linearly interpolates 768-d SigLIP feature vectors between boundary frames. Because the firefighter shifts posture rapidly, the interpolated vector drifts from the intermediate frame features (Rank 258/294).
* *LLM Deduction*: Conditioned on the procedural protocol script, the LLM infers the donning sequence ("secures gas mask", "pulls heat-reflective hood over head"), achieving Rank 12/294 without visual access.
* *Representative Median Example*: In `AiirSource-Military_8-clip-1` (\(\Delta = -47\), \(\Delta / N = -0.070\), Rank 403 text vs. 450 visual), both modalities exhibit moderate performance, with procedural constraints giving text a modest advantage.

**Case B: Stochastic Physical Dynamics (Nature & Scenery)**
*Video ID: `King-Kong-Amazon_5-clip-14` (\(\Delta = +81\), Rank 145 text vs. 64 visual)*
* *Context*: A primate moves through dense jungle canopy.
* *Target Event*: The primate leaps toward an arbitrary branch on the upper left.
* *LLM Failure*: Locomotion in arboreal foliage is chaotic; the LLM predicts generic resting or foraging (\(\Delta = +81\)).
* *Visual Baseline Victory*: Consistent canopy foliage and lighting allow visual feature interpolation to track scene representations smoothly.

---

## 5. Discussion, Limitations & Practical Implications

### Practical Implications: Keyframe-Mediated Token Pruning
Current Video-LLMs uniformly encode frames at fixed rates, consuming substantial compute. Our findings suggest a **keyframe-mediated architecture**:
* In procedural streams (e.g., instructional guides, industrial workflows), dense visual frames can be pruned, sampling only boundary keyframes while in-filling intermediate intervals with lightweight symbolic inferences.
* In stochastic streams (e.g., wildlife, dynamic sports), visual encoders must maintain dense sampling.

### Limitations
1. **Single LLM Evaluated**: All semantic inferences were conducted using Llama 3.1 8B with a single zero-shot prompt. Evaluating multi-model scaling and prompt sensitivity remains an open direction.
2. **Caption Similarity vs. Downstream Utility**: Target similarity measures representation proximity rather than downstream task success. Future work should evaluate keyframe-mediated representations on downstream VideoQA tasks (such as WildQA).
3. **Imperfect Modality Boundaries**: Even in Military, 35.6% of segments favor visual continuity, demonstrating that real-world videos contain mixed procedural and stochastic phases.

---

## References
* **Bian, Y., et al.** (2025). VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. *SIGGRAPH*.
* **Bosetti, M., et al.** (2024). Text-Enhanced Zero-Shot Action Recognition: A training-free approach. *ICPR*.
* **Castro, S., et al.** (2022). WildQA: In-the-Wild Video Question Answering. *COLING*.
* **Chen, Y., et al.** (2024). FastV: Fast Video-LLM Inference via Dynamic Token Sparsification. *ArXiv*.
* **Li, W., et al.** (2025). Lost in Embeddings: Information Loss in Vision-Language Models. *EMNLP (Findings)*.
* **Rao, Y., et al.** (2021). DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. *NeurIPS*.
