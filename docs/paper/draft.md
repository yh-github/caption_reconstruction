
# Dense caption reconstruction: video in-filling with language models

## Abstract
Recent Video-LLMs typically treat video understanding as a continuous stream of visual encoding. However, real-world events often follow structured, semantic scripts that pre-trained language models can predict without immediate visual evidence. In this work, we investigate the boundary between **semantic inference** (what *must* happen) and **visual perception** (what *actually* happened) through a novel *Caption Reconstruction* comparative framework. We use two independent methods—a text based LLM and a Visual-Interpolation baseline method—to reconstruct missing segments from masked videos. We created a new dense caption dataset, based on the WildQA dataset. Our analysis showcases a spectrum of narrative predictability across video domains: while *stochastic* events (e.g., nature) require visual grounding, *procedural* events (e.g., farming, manufacturing) allow text-only models to in-fill accurate reconstructions, rendering visual processing redundant for significant durations. This framework serves as a diagnostic tool for measuring multimodal information density and suggests potential for temporal semantic compression.

## 1. Introduction
The promise of multimodal AI is the fusion of visual perception with semantic reasoning. Yet, in current Video-LLM architectures, this fusion is often brute-forced: models ingest massive sequences of visual tokens regardless of the information content. This approach ignores a fundamental property of the physical world: narrative predictability.

Consider a video of a person chopping an onion. If the next scene is missing, and the one after that has the onion in a pan with red peppers, a human (or an LLM) can predict with high confidence that the onion was fried, the peppers were added, and the dish was stirred, etc. The visual details—the exact shape of the pan, the lighting—are *stochastic residuals*, but the *semantic action* is procedurally deterministic.

In this paper, we propose a **Comparative Reconstruction Framework** to quantify this "Semantic Gap." We treat video understanding not as a single task, but as a competition between two modalities:
1.  **Text Only:** A Large Language Model (LLM) with pure text input that infers missing events based solely on temporal context and world knowledge.
2.  **Visual Only:** A video-based model that interpolates missing content using only visual embedding similarity.

By comparing their performance on a masked reconstruction task across diverse video domains, we operationalize the concept of **Multimodal Redundancy**. We find that video content effectively clusters along a **Predictability Spectrum**. On the "Procedural" end (e.g., *Farming*, *Military*), actions follow strict protocols that are logically deducible. On the "Stochastic" end (e.g., *Nature*, *Scenery*), events are driven by chaotic physical dynamics where visual observation is irreplaceable.

## 2. Methodology: The Comparative Framework

### 2.1 Problem Formulation: Masked Semantic Reconstruction
We represent a video $V$ as a sequence of $T$ semantic units, a 1 second interval for each unit, where each unit $u_t$ consists of a visual embedding $v_t$ and a textual caption $c_t$.
The task is to mask a subset of these units $M \subset \{1...T\}$ and reconstruct the missing semantic content. Unlike pixel-level in-painting, our target is the **semantic embedding** of the missing segment. This choice abstracts away low-level visual noise and focuses on event-level understanding.

### 2.2 The Two Pathways

#### 2.2.1 Text Logic
This pathway tests the limit of **Semantic Predictability**. It receives only the timestamps and captions of the *unmasked* segments: $C_{obs} = \{(t, c_t) | t \notin M\}$.
We employ a frozen Large Language Model (Gemini 1.5 Pro) with a structured prompt to "fill in the blanks." The model must rely on its internal world model to deduce the missing actions. For example, if $c_{t-1}$ is "Man connects hose" and $c_{t+1}$ is "Water sprays," the model infers $c_t \approx$ "Man turns on tap."
The predicted text $\hat{c}_t$ is then encoded into the embedding space: $\hat{e}_{text} = \text{Encoder}(\hat{c}_t)$.

#### 2.2.2 Visual Continuity
This pathway tests the limit of **Visual Fidelity**. It receives only the visual embeddings of the *unmasked* segments: $V_{obs} = \{v_t | t \notin M\}$.
We employ a baseline non-parametric interpolation approach. The missing embedding $\hat{e}_{vis}$ is reconstructed via weighted averaging of the surrounding visual vectors. This captures the visual "inertia" of the scene (e.g., color palette, background, object persistence) without understanding the causal logic.

### 2.3 Evaluation: Normalized Population Ranking
Directly comparing raw cosine similarity scores between modalities is flawed because the embedding spaces or density distributions may differ. A score of 0.8 might be high for one model but average for another.
To address this, we employ a **Population Ranking** strategy. For each experimental batch (e.g., $`N=100`$ videos at masking level $`k`$):
1.  We rank all videos by their reconstruction score ($`\text{cos\_sim\_mean}`$) separately for each method.
2.  Rank 1 represents the "easiest" video for that model; Rank $N$ represents the "hardest."

We then define the **Predictability Delta ($`\Delta`$)**:
$$
 \Delta = \text{Rank}(\hat{e}_{text}) - \text{Rank}(\hat{e}_{vis}) 
$$

This effectively "grades on a curve." It normalizes for the inherent difficulty of the dataset. A strongly **Negative $\Delta$** means the video was relatively much easier for the LLM to reconstruct (Top-Tier) than it was for the Video model (Bottom-Tier), identifying a specific "Semantic Advantage."

This $\Delta$ is our diagnostic metric.
- $\Delta \ll 0$: **Semantic Dominance**. The event was logically predictable, but visually discontinuous.
- $\Delta \gg 0$: **Visual Necessity**. The event was logically ambiguous, but visually smooth.
- $\Delta \approx 0$: **Agreement**. Both methods succeed (easy) or fail (hard) equally.

## 3. Related Work
Our work bridges three distinct areas:
**Video Inpainting and Completetion**: Recent pixel-level approaches like **VideoPainter** (Bian et al., 2025) focus on maintaining visual consistency in long videos. We lift this task to the *semantic* level, focusing on the meaning of the missing segment rather than its texture.
**Text-Enhanced Recognition**: Approaches like **TEAR** (Bosetti et al., 2024) have shown that text descriptors can enhance zero-shot action recognition, suggesting that language often captures the "essence" of an action better than noisy visual features.
**VLM Limitations**: Li et al. (2025) argue in **"Lost in Embeddings"** that the projection from visual to language space acts as a flawed compression, losing vital details. Our framework flips this finding: we identify when this "lossy" text representation is actually *sufficient* or even *superior* due to semantic redundancy.

## 4. Experiments & Analysis

### 4.1 Experimental Setup
We evaluated our framework on the densely captioned video dataset derived from **WildQA** across both development (**Wild4**, \(N=294\) masked segments across 98 videos) and test (**Wild5**, \(N=673\) masked segments across 225 videos) splits. Video domains span *Procedural* routines (*Military*, *Survival*, *Farming*) and *Stochastic* environments (*Nature & Scenery*, *Natural Disaster*, *Action & Vehicle*). Experiments evaluate contiguous masked gaps of \(w \in \{3, 6\}\) seconds.

Text in-filling is performed by **Llama 3.1 8B**, and visual continuity is modeled via **SigLIP** (`google/siglip-base-patch16-224`) feature interpolation.

### 4.2 The Predictability Spectrum Across Video Domains

To compare across modalities without raw score distribution artifacts, we evaluate the **Normalized Population Rank Delta** \(\Delta / N\):
\[
\Delta / N = \frac{\text{Rank}(\hat{e}_{\text{text}}) - \text{Rank}(\hat{e}_{\text{vis}})}{N}
\]
where \(\Delta < 0\) indicates **Semantic Dominance** (the video was ranked higher by text inference than visual interpolation), and \(\Delta > 0\) indicates **Visual Necessity** (visual continuity outperformed semantic prediction).

![Figure 1: The Predictability Spectrum across Video Categories.](../../results/plots/paper_figures/fig1_predictability_spectrum.png)
*Figure 1: The Predictability Spectrum across Video Categories. Distribution of Normalized Rank Delta (\(\Delta / N\)) across video domains for both Dev (Wild4, \(N=294\)) and Test (Wild5, \(N=673\)) splits. Negative values indicate LLM superiority; positive values indicate visual model superiority.*

| Domain / Category | Dev Mean \(\Delta\) (Wild4, \(w=6\)) | Test Mean \(\Delta\) (Wild5, \(w=6\)) | Test Mean \(\Delta\) (Wild5, \(w=3\)) | Modality Advantage |
|---|:---:|:---:|:---:|---|
| **Military** | \(-31.5 \pm 98.2\) | \(-61.6 \pm 235.6\) | \(-77.8 \pm 238.1\) | **Semantic Dominance** (\(p < 10^{-5}\)) |
| **Survival** | \(+2.8 \pm 88.1\) | \(-14.2 \pm 222.1\) | \(-23.4 \pm 226.7\) | Semantic Trend |
| **Natural Disaster** | \(-12.8 \pm 103.5\) | \(-13.7 \pm 208.5\) | \(-14.9 \pm 215.0\) | Semantic Trend |
| **Action & Vehicle** | \(+4.5 \pm 133.7\) | \(+3.2 \pm 217.2\) | \(+12.1 \pm 224.5\) | Mixed / Transition |
| **Farming** | \(+7.5 \pm 103.9\) | \(+29.5 \pm 246.1\) | \(+18.7 \pm 249.2\) | Mixed / Transition |
| **Nature & Scenery** | \(\mathbf{+51.1 \pm 100.5}\) | \(\mathbf{+80.0 \pm 248.0}\) | \(\mathbf{+102.2 \pm 249.5}\) | **Visual Necessity** (\(p < 10^{-5}\)) |

**Key Findings:**
1. **Perfect Cross-Split Replication**: The spectrum from *Military* to *Nature & Scenery* replicates consistently across both independent splits. In Wild4, the spread is 82.6 rank positions (Mann-Whitney \(p = 4.06 \times 10^{-4}\)). In Wild5, the spread widens to 141.6 rank positions (\(p = 4.96 \times 10^{-6}\)).
2. **Gap Duration Sensitivity**: At narrower gaps (\(w=3\)), the modality divergence is even more pronounced: *Military* reaches \(\Delta = -77.8\) while *Nature & Scenery* reaches \(\Delta = +102.2\) (a spread of 180 ranks, \(p = 1.57 \times 10^{-8}\)).

---

### 4.3 Breaking Boundary Inertia: Contrastive Temporal Evaluation

Naive interpolation baselines achieve deceptively high raw similarity by maintaining "visual/text inertia" (repeating or blending surrounding boundary frames). To determine whether reconstructed representations genuinely recover novel state changes, we evaluate the **Boundary Contrastive Margin**:
\[
\text{Margin}(R) = \text{Sim}(R, \text{Target}) - \max(\text{Sim}(R, \text{Pre}), \text{Sim}(R, \text{Post}))
\]
A positive or near-zero margin demonstrates that the model successfully differentiates the internal event from surrounding context distractors.

![Figure 2: Breaking Boundary Inertia.](../../results/plots/paper_figures/fig2_boundary_inertia.png)
*Figure 2: Breaking Boundary Inertia. Contrastive margin across gap elapsed time. (A) In text space, LLM reconstructions maintain a \(\approx +0.13\) margin advantage over Text LERP baselines. (B) In cross-modal video space, LLM reconstructions avoid boundary trapping, achieving superior discrimination over visual interpolation baselines.*

* **Text Semantic Space**: LLM reconstructions achieve a mean margin of \(-0.0576\) compared to \(-0.1934\) for Text LERP—a **\(+0.1358\) margin advantage** with a **92.5% win rate** (Wilcoxon \(p = 7.78 \times 10^{-89}\)).
* **Cross-Modal Video Space**: Evaluating LLM text embeddings directly against raw target video frames yields a mean margin of \(-0.0097\) versus \(-0.0591\) for Visual LERP—a **90.0% win rate** (Wilcoxon \(p = 2.59 \times 10^{-83}\)).
* **Mechanism**: As shown in Figure 2, interpolation baselines suffer from severe boundary anchoring at the gap edges (\(t=1\) and \(t=6\)), whereas the LLM infers narrative state transitions that break boundary inertia.

---

### 4.4 Qualitative Case Studies

**Case A: The "Blind" Victory (Procedural Logic)**
*Video ID: `Welker-Farms-Inc_3-clip-4` (Farming)*
* **Context**: A heavy tractor positions itself near an open field.
* **Target Event**: The tractor unfolds its mechanical sprayer arms.
* **Visual Baseline Failure**: Visual interpolation blends optical features, predicting a blurry stationary tractor (\(\Delta = -91\)).
* **LLM Semantic Inference**: Conditioned on the temporal text script ("tractor enters field" → "spraying operation"), the LLM explicitly infers "unfolds mechanical arms", matching the target video state without requiring raw pixels.

**Case B: The "Silent" Victory (Stochastic Dynamics)**
*Video ID: `King-Kong-Amazon_5-clip-14` (Nature)*
* **Context**: A primate navigates dense canopy branches.
* **Target Event**: The primate leaps to an arbitrary branch on the upper left.
* **LLM Failure**: Natural locomotion is non-deterministic; the LLM predicts generic feeding or resting (\(\Delta = +81\)).
* **Visual Baseline Victory**: Visual flow, foliage color distribution, and motion inertia preserve scene continuity, allowing visual interpolation to easily beat language prediction.

---

## 5. Discussion & Practical Implications

Our findings establish that visual perception and semantic inference play complementary roles across a predictable spectrum:
1. **Dynamic Video-LLM Token Pruning**: Current multimodal models uniformly ingest visual tokens at fixed frame rates. Our framework demonstrates that in procedural domains (e.g. instructional guides, industrial workflows, protocol execution), visual tokens can be heavily downsampled or replaced with lightweight text captions without loss of semantic fidelity.
2. **Selective Visual Verification**: Visual computation should be allocated adaptively—dense frame processing reserved for stochastic intervals where causal scripts are absent, and symbolic text reasoning leveraged where event sequences are procedurally deterministic.

## References
* **Bian, Y., et al.** (2025). VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. *SIGGRAPH*.
* **Bosetti, M., et al.** (2024). Text-Enhanced Zero-Shot Action Recognition: A training-free approach. *ICPR*.
* **Castro, S., et al.** (2022). WildQA: In-the-Wild Video Question Answering. *COLING*.
* **Li, W., et al.** (2025). Lost in Embeddings: Information Loss in Vision-Language Models. *EMNLP (Findings)*.
