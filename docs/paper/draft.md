<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$']]
    }
  };
</script>

<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


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
We evaluated our framework on the densly captioned video dataset, based on the **WildQA** dataset, selecting 100 diverse videos from the development set. Domains include *Survival*, *Farming*, *Nature/Documentary*, *Military*, and *Action/Vehicle*.
We performed experiments at increasing masking levels: removing $k=\{6, 9, 12, 15\}$ segments from each video.

### 4.2 Rank difference by Category

We showcase the per catergory analysis. Each method is ranked internally.

![Figure 1: Faceted Violin Plot showing the distribution of Rank Delta across categories.](../../results/plots/conditional_analysis/rank_delta_distribution_faceted_cos_sim_mean.png)
*Figure 1: The Predictability Spectrum. Distribution of Rank Delta ($\Delta$) across video categories. Negative values (left) indicate LLM superiority (Procedural); positive values (right) indicate Video Model superiority (Stochastic).*

**Figure 1** illustrates the distribution of Rank $\Delta$ across categories.
- **Procedural Resilience (The Left Tail)**: Categories like *Farming* and *Military* exhibit instances of extreme "Text high score, Video low score" ($\Delta < -50$). In these videos—often instructional or routine-based—the LLM successfully reconstructs the correct intermediate step (e.g., "loading the gun", "planting the seed") even when the visual jump is large.
- **Stochastic Dependence (The Right Tail)**: Categories like *Nature/Documentary* and *Scenery* show a heavy tail towards "Video high score, Text low score" ($\Delta > 50$). Low level weather or scenery changes are harder ro describe and reconstruct in short texts.
**Robustness Check**: To confirm these are not random noisy fluctuations, we performed a consistency analysis across all four masking levels. We identified **13 videos (13% of the dataset)** that remained in the top quintile of $\Delta$ across *every* condition ($k=6,9,12,15$). This stability confirms that for certain content types, the "modality advantage" is a persistent intrinsic property, not an artifact of specific sampling.

![Figure 2: Consistency Analysis showing Persistent Signal vs Random Noise.](../../results/plots/null/null_hypothesis_consistency.png)
*Figure 2: Robustness Check. Comparison of the **Average Rank Delta** across all four masking levels ($k=6,9,12,15$). The "Real Data" (Red) distribution is significantly wider ($\sigma=28.8$) than the "Random Chance" baseline ($\sigma=19.0$), proving that the observed modality advantages are persistent intrinsic properties of the videos, not random fluctuations which would average out to zero.*

### 4.3 Qualitative Case Studies ("The When")
We highlight two examples that define the extremes of the spectrum:

**Case A: The "Blind" Victory (Procedural)**
*Video ID: Welker-Farms-Inc_3-clip-4 (Farming)*
- **Context**: A large tractor is positioning itself near a field.
- **Masked Content**: The tractor unfolds its mechanical arms.
- **Result**: The Video model fails ($\Delta = -91$) because the visual shape of the tractor chances drastically (low visual continuity). The LLM, however, infers from the context of "positioning" and "field work" that "deployment" must follow.

**Case B: The "Silent" Victory (Stochastic)**
*Video ID: King-Kong-Amazon_5-clip-14 (Nature)*
- **Context**: A monkey is moving through dense branches.
- **Masked Content**: The monkey leaps to a specific branch on the left.
- **Result**: The LLM fails ($\Delta = +81$) because "leaping" is generic; it guesses "eating" or "climbing up." The Video model succeeds because the visual flow (optical flow, color histogram) is preserved across the short gap.

## 5. Discussion & Conclusion
We presented a diagnostic framework to measure the necessity (or lack thereof) of visual data in video understanding. Our results challenge the assumption that "more vision is always better," supporting the view that VLMs often suffer from information loss or redundancy (Li et al., 2025). For a significant portion of real-world "procedural" content, the visual stream is semantically redundant—a sufficiently powerful Language Model can mirror the event stream without seeing it.

**Future Work:**
- **The Event Horizon**: Our experiments ($k \le 15$) showed low but stable correlation between methods. Future work should increase masking duration to find the "breaking point" where even procedural scripts become chaotic and both models regress to random chance. 
- **Efficient Video RAG**: Our findings imply that massive video datasets can be efficiently indexed for NLI/QA. Segments with high LLM-predictability ($\Delta \ll 0$) can be stored and retrieved purely as text, saving orders of magnitude in VLM token costs. Video processing is then reserved only for the stochastic moments ($\Delta \gg 0$) where visual verification is strictly necessary.

## References
**Bian, Y., et al.** (2025). VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. *SIGGRAPH*.

**Bosetti, M., et al.** (2024). Text-Enhanced Zero-Shot Action Recognition: A training-free approach. *ICPR*.

**Li, W., et al.** (2025). Lost in Embeddings: Information Loss in Vision-Language Models. *EMNLP (Findings)*.
