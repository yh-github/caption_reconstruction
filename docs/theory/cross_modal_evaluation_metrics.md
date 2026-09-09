# Cross-Modal Evaluation, Metrics, and Comparative Criteria

This document provides a comprehensive theoretical and methodological reference for comparing **text-based semantic reconstruction** (LLM/SLM in-filling) against **video-based visual reconstruction** (visual vector interpolation). It details the supported metrics, their trade-offs, the fundamental challenge of representation dependency, and the role of shared multimodal contrastive spaces.

---

## 1. Overview & The Comparative Challenge

The core objective of this project is to explore the boundary between **semantic inference** (what *must* happen logically) and **visual perception** (what *actually* happened visually) across a spectrum of video domains:
* **Procedural domains** (e.g., *Farming*, *Military*, *Instructional*): Driven by strict causal scripts where actions are logically deducible without continuous visual observation.
* **Stochastic domains** (e.g., *Nature*, *Scenery*): Governed by chaotic or physical dynamics where visual observation is irreplaceable.

To operationalize this, the framework masks a subset of video segments and tasks two distinct pathways with reconstruction:
1. **Text Logic Pathway**: Observes unmasked caption timestamps and text \( C_{\text{obs}} = \{(t, c_t) \mid t \notin M\} \), prompts an LLM/SLM to infer missing events, and encodes the predicted text:
   \[
   \hat{e}_{\text{text}} = \text{Encoder}_{\text{text}}(\hat{c}_t)
   \]
2. **Visual Continuity Pathway**: Observes unmasked visual feature vectors \( V_{\text{obs}} = \{v_t \mid t \notin M\} \) and interpolates missing segments via neighbor weighting (e.g., [`RepeatClosestVector`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/reconstruction/vector_reconstruction.py#L23-L52) or [`MeanClosestVectors`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/reconstruction/vector_reconstruction.py#L53-L84)):
   \[
   \hat{e}_{\text{vis}} = \text{Interpolate}(V_{\text{obs}})
   \]

### The Fundamental Asymmetry
Directly evaluating and comparing these pathways is challenging because the models operate on entirely different modalities:
* Text generation produces **discrete language tokens** mapped into a text embedding space.
* Video interpolation produces **continuous numerical vectors** mapped in a visual feature space.

The codebase implements a multi-tier hierarchy of metrics and comparative criteria to tackle this asymmetry.

---

## 2. Supported Metrics & Scoring Criteria

The evaluation framework is structured across four primary layers:

```
┌─────────────────────────────────────────────────────────────┐
│ 4. Cross-Modal Criteria: Population Rank Delta (Δ), Bias    │
├─────────────────────────────────────────────────────────────┤
│ 3. Temporal Alignment: Temporal NDCG, Windowed Recall@1     │
├─────────────────────────────────────────────────────────────┤
│ 2. Retrieval & Ranking: MRR, Recall@1, Recall@5, Mean Rank  │
├─────────────────────────────────────────────────────────────┤
│ 1. Vector Similarity: Cosine Sim, Residual Sim, Z-Score     │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Vector-Level Representation Metrics

#### **A. Raw Cosine Similarity (`cos_sim` / `cos_sim_mean`)**
* **Implementation**: [`calculate_elementwise_cosine`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/eval_vectors.py#L43-L111) and [`VectorReconstructionEvaluator`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/evaluation.py#L118-L135).
* **Formula**:
  \[
  \text{cos\_sim}(\hat{v}, v^*) = \frac{\hat{v} \cdot v^*}{\|\hat{v}\| \|v^*\|}
  \]
* **PROs**: Bounded in \([-1, 1]\), computationally efficient, and the standard measure of orientation in neural embedding spaces.
* **CONs**:
  * **Incomparable across modalities**: Text embedding spaces (e.g., MPNet) and visual embedding spaces (e.g., ViT-S/16) have distinct geometric dimensions, clustering densities, and norms. A similarity of `0.80` in visual space cannot be equated to `0.80` in text space.
  * **Static scene / Inactivity bias**: When a camera remains fixed on an unchanging background, visual interpolation trivially achieves near `1.0` cosine similarity without predicting any actual event.

#### **B. Context Residual Cosine Similarity (`cos_sim_residual`)**
* **Implementation**: [`context_projection`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/eval_vectors.py#L114-L150) and [`ReconstructionEvaluator_EmbSimilarity`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/evaluation.py#L238-L280).
* **Formula**: Projects out the mean context vector of observed frames \( \bar{v}_{\text{context}} \) before computing similarity:
  \[
  \hat{v}_{\text{res}} = \hat{v} - \frac{\hat{v} \cdot \bar{v}_{\text{context}}}{\|\bar{v}_{\text{context}}\|^2} \bar{v}_{\text{context}}
  \]
* **PROs**: Strips away static scene background features, penalizing models that merely copy stationary visual context and measuring only the *novel*, dynamic event information.
* **CONs**: Assumes context variation lies on a simple 1D subspace spanned by the mean vector; susceptible to distortion if context has high internal variance.

#### **C. Z-Score Normalized Similarity (`stats_z_score`)**
* **Implementation**: [`MetricsRecordRaw.stats_z_score`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/metrics.py#L81-L100).
* **Formula**:
  \[
  z = \frac{x - \mu_{\text{corpus}}}{\sigma_{\text{corpus}}}
  \]
* **PROs**: Calibrates scores against the global corpus distribution of that specific model/modality.
* **CONs**: Does not adjust for higher-order moments (skewness, multimodality) across distinct embedding distributions.

#### **D. Euclidean Geometric Distance (`euclidean_dist`, `video_avg_dist`, `video_var_dist`)**
* **Implementation**: [`merge_and_correlate.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/merge_and_correlate.py#L69-L73) and `results/euclidean_metrics.csv`.
* **PROs**: Provides an unnormalized geometric measure of visual motion and feature spread across the video.
* **CONs**: Heavily affected by the curse of dimensionality and cannot be meaningfully compared across models with different output dimensions.

---

### Layer 2: Retrieval & Discriminative Ranking Metrics

Used when configured as `evaluation.type: "emb_retrieval"` (implemented in [`calculate_retrieval_metrics`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/eval_vectors.py#L277-L314) and [`ReconstructionEvaluator_Retrieval`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/evaluation.py#L281-L320)):

* **Mean Reciprocal Rank (`mrr`)**:
  \[
  \text{MRR} = \frac{1}{M} \sum_{i=1}^{M} \frac{1}{\text{rank}_i}
  \]
* **Recall@1 (`recall_at_1`) & Recall@5 (`recall_at_5`)**: Fraction of masked positions where the true clip is ranked 1st or within the top 5 among candidate distractors.
* **Mean Rank (`mean_rank`)**: Average rank assigned to the true target.

* **PROs**:
  * **Scale-invariant**: Evaluates relative discriminative power against candidate distractors rather than raw dot-product magnitudes.
  * Measures whether the reconstructed representation is fine-grained enough to identify the specific target event among adjacent events.
* **CONs**:
  * **Pool size sensitivity**: The metric depends heavily on the distractor pool size (e.g., retrieving from a pool of 3 masked clips vs. 15 masked clips).
  * Discrete and non-smooth: Small shifts in vector space can cause sharp ranking drops.

---

### Layer 3: Temporal Alignment & Sequence Metrics

Implemented in [`recalculate_metrics.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/recalculate_metrics.py#L25-L69) and [`calc_baseline_full.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/calc_baseline_full.py#L103-L159):

#### **A. Temporal NDCG (`temporal_ndcg`)**
Ranks all candidate clips by predicted similarity and measures whether high similarity corresponds to temporal proximity. The relevance of candidate \( j \) for query \( i \) decays exponentially with temporal distance:
\[
\text{Relevance}(i, j) = \exp\left(-\frac{|t_i - t_j|}{2.0}\right)
\]
\[
\text{DCG} = \sum_{r} \frac{\text{Relevance}(r)}{\log_2(r + 2)}, \quad \text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}
\]

#### **B. Windowed Temporal Recall@1 (`temporal_recall_at_1_w1`, `temporal_recall_at_1_w2`)**
Considers a retrieval successful if the top predicted clip falls within a temporal window of \(\pm 1\) or \(\pm 2\) time steps of the true time index.

* **PROs**:
  * **Distinguishes sequence order from semantic content**: If a model predicts the correct action but places it 2 seconds too early, strict MRR awards zero credit, whereas Temporal NDCG awards graded credit for temporal near-misses.
* **CONs**:
  * Assumes narrative time is strictly monotonic; struggles with cyclic or repetitive actions where identical actions recur at different timestamps.
  * Computationally heavier (requires parsing and caching full similarity matrices).

---

### Layer 4: Cross-Modal Comparative Criteria

#### **A. Population Relative Rank Delta (\(\Delta_{\text{rank}}\))**
Implemented in [`calculate_rank_differences_by_num_masked`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/analysis/compare_ranks.py#L14-L45) and detailed in the [paper draft](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/draft.md#L35-L52):
\[
\Delta_{\text{rank}} = \text{Rank}(\hat{e}_{\text{text}}) - \text{Rank}(\hat{e}_{\text{vis}})
\]
For a given batch of \( N \) videos (e.g., \( N=100 \) at masking width \( w \)), each method independently ranks all videos from 1 (easiest/best) to \( N \) (hardest/worst):
* **\( \Delta_{\text{rank}} \ll 0 \) (Negative Delta)**: **Semantic Dominance / Procedural**. The LLM easily deduced the missing step, whereas the video interpolator struggled (e.g., due to a large visual camera cut).
* **\( \Delta_{\text{rank}} \gg 0 \) (Positive Delta)**: **Visual Necessity / Stochastic**. The video interpolator succeeded (smooth visual continuity), while the LLM failed (stochastic or ambiguous actions).
* **\( \Delta_{\text{rank}} \approx 0 \)**: **Modal Agreement**. Both methods found the video equally easy or difficult.

* **PROs**:
  * **"Grades on a curve"**: Completely eliminates absolute scale and density disparities between visual and language embedding spaces.
  * Operationalizes the **Predictability Spectrum**, reliably separating procedural from stochastic content.
* **CONs**:
  * **Population-dependent**: A video's score depends on the composition of the other videos in the test batch.
  * Discards absolute quality: if both models perform poorly on a video, \( \Delta_{\text{rank}} \approx 0 \).

#### **B. Direct Performance Deltas (`mrr_delta`, `t_ndcg_delta`, `rank_delta`)**
Implemented in [`deep_analysis.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/deep_analysis.py#L43-L45) and [`merge_and_correlate.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/merge_and_correlate.py#L62-L63):
\[
\Delta_{\text{MRR}} = \text{MRR}_{\text{LLM}} - \text{MRR}_{\text{Video}}, \quad \Delta_{\text{T-NDCG}} = \text{NDCG}_{\text{LLM}} - \text{NDCG}_{\text{Video}}
\]
* **PROs**: Instance-level metric independent of other videos in the dataset.
* **CONs**: Assumes MRR on text and MRR on video have equal variance and baseline difficulty.

#### **C. Mask Width Consistency & Quintile Persistence**
Implemented in [`check_consistency.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/analysis/check_consistency.py#L29-L56) and [`null_hypothesis_consistency.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/analysis/null_hypothesis_consistency.py). Tracks whether a video consistently remains in the extreme top quintile (top 20%) of \( \Delta \) across increasing gap widths (\( w \in \{3, 6, 9, 12, 15\} \)).
* **PROs**: Validates that observed modality advantages are persistent intrinsic characteristics of the video content, ruling out random sampling noise.
* **CONs**: Requires executing broad parameter sweeps across multiple masking levels.

#### **D. Category Bias & Wilcoxon Signed-Rank Tests**
Implemented in [`check_category_bias.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/analysis/check_category_bias.py#L53-L77). Evaluates whether the median of \( \Delta \) within a domain category significantly departs from zero (\( p < 0.05 \)).
* **PROs**: Provides formal hypothesis testing for domain-level predictability differences.
* **CONs**: Requires adequate sample sizes per category (statistical power degrades if \( n < 6 \)).

#### **E. Information-Theoretic Surprisal Grounding**
Implemented in [`merge_and_correlate.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/merge_and_correlate.py#L88-L120). Correlates \( \Delta_{\text{MRR}} \) against:
* **Text Surprisal & Perplexity** (`text_surprisal_nll`, `text_perplexity` via prior language model sequence likelihoods).
* **Video Variance** (`video_var_dist`, `video_avg_dist`).
* **PROs**: Provides information-theoretic explanations for *why* an LLM or video model wins.
* **CONs**: Requires separate compute passes to estimate language sequence probabilities.

---

### Layer 5: Text-Only Metrics (Reference)

#### **BERTScore (`bs_p`, `bs_r`, `bs_f1`)**
* **Implementation**: [`ReconstructionEvaluator_BertScore`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/evaluation.py#L166-L237) using `microsoft/deberta-large-mnli`.
* **PROs**: Measures token-level semantic overlap and is robust to paraphrasing.
* **CONs**: Text-only. Cannot be applied to visual vector baselines unless the video representations are decoded into natural language text.

---

## 3. Comprehensive Summary Table of Metrics

| Metric / Criterion | Level | What It Measures | Key Advantage (PRO) | Key Limitation (CON) |
| :--- | :--- | :--- | :--- | :--- |
| **`cos_sim`** | Vector | Raw vector alignment with ground truth | Fast, standard vector metric | Incomparable across modalities; static scene bias |
| **`cos_sim_residual`** | Vector | Vector alignment after subtracting context | Cancels static background inertia | Assumes linear 1D subspace projection |
| **`stats_z_score`** | Vector | Standard deviations away from corpus mean | Calibrates scale within modality | Ignores higher-order distribution differences |
| **`euclidean_dist`** | Vector | L2 distance in embedding space | Unnormalized geometric spread | Sensitive to dimension count; cross-modal mismatch |
| **`mrr` / Recall@K** | Retrieval | Discriminative retrieval against distractors | Scale-invariant rank formulation | Dependent on distractor pool size |
| **`temporal_ndcg`** | Temporal | Sequence order & temporal proximity | Awards partial credit for near-misses | Assumes monotonic narrative progression |
| **Windowed Recall** | Temporal | Top-1 hit within \(\pm 1\) or \(\pm 2\) time steps | Tolerates minor timing jitter | Binary thresholding |
| **Population \( \Delta_{\text{rank}} \)** | Comparative | Relative percentile rank gap between modalities | Eliminates encoder scale mismatch | Population-dependent; ignores absolute performance |
| **Direct \( \Delta_{\text{MRR}} \)** | Comparative | Direct difference in retrieval score | Simple instance-level delta | Assumes equal baseline difficulty |
| **Width Consistency** | Robustness | Multi-width stability of modality advantage | Rejects random noise hypothesis | Requires multi-configuration sweeps |
| **Wilcoxon Category Bias**| Statistical | Non-parametric test for domain asymmetry | Statistically rigorous | Requires sufficient video count per category |
| **Surprisal Correlation** | Theoretical | Grounding performance in entropy/variance | Explains causal mechanisms | Requires separate prior computation pipelines |
| **`BERTScore`** | Text | DeBERTa token-level semantic match | Matches human linguistic judgment | Cannot evaluate visual vector baselines |

---

## 4. The Representation Dependency Critique

### Is Retrieval Truly "Modality-Independent"?
In evaluation discussions, ranking metrics like MRR or Recall@K are often described as "modality-independent." While the mathematical ranking function (\(1/\text{rank}\)) is generic, **the entire ranking outcome and discriminative difficulty are fundamentally hostage to the feature representation models chosen for each modality.**

```
Text Pipeline:   Unmasked Captions → LLM/SLM → Candidate Text → [Text Encoder]  → Text Vectors
                                                                          ↑
                                                        (e.g., Gemini-001, MPNet)

Video Pipeline:  Unmasked Frames   → [Vision Encoder] → Interpolation            → Video Vectors
                                            ↑
                                (e.g., ViT-S/16, VideoMAE)
```

### Why Representation Models Confound the Comparison

1. **Semantic Granularity vs. Low-Level Visual Inertia**:
   * **Text Encoders** (e.g., MPNet, Gemini Embedder): Human-annotated captions are pre-filtered through human perception. They discard lighting, camera shake, and background clutter, retaining only high-level conceptual actions (*"Man hooks hose"* → *"Man turns valve"*). Candidate captions in a video are usually lexically distinct, providing sharp decision boundaries.
   * **Vision Encoders** (e.g., ViT-S/16): Frame encoders retain low-level visual features (color palettes, dominant background, lighting). In a video with stationary background, adjacent frame embeddings may have cosine similarities of `0.98`, clustering tightly.
   * **Confound**: A higher MRR in text might not mean the language model reconstructed the event better than visual interpolation; it may simply mean **the text encoder's space has wider semantic margins between candidate clips than the vision encoder's space does.**

2. **Geometric Anisotropy**:
   * Pretrained text models frequently exhibit the "cone effect" (anisotropy), where embeddings cluster within a narrow cone.
   * Vision transformers often suffer from dominant background patch tokens or dimensional collapse.
   * Consequently, retrieval "difficulty" is fundamentally non-uniform across the two spaces.

3. **The "Human Bottleneck" Advantage**:
   * The text model receives input that has already been abstracted into language by humans.
   * The visual interpolation model operates directly on raw sensor features without human curation.

### Codebase Mitigations
The repository was explicitly designed around this limitation:
* **Population Ranking Delta (\( \Delta_{\text{rank}} \))**: Rather than comparing raw scores across modalities, it ranks videos *within* the text distribution and *within* the visual distribution, comparing only their relative percentiles.
* **Residual Projection**: Cancels stationary background features before visual vector comparison.
* **Consistency Sweeps**: Validates that signals persist across diverse mask widths rather than reflecting local embedding quirks.

---

## 5. Shared Multimodal Contrastive Space (CLIP / SigLIP)

### Disambiguation: Did This Codebase Use CLIP?
**No. OpenAI's CLIP was not used in this codebase.** 
* The term **"clip"** throughout the codebase refers strictly to a **1-second video segment** (e.g., `CaptionedVideo.clips`, `clip_size=1`, `Welker-Farms-Inc_3-clip-4`).
* The visual embeddings were extracted using an ImageNet-pretrained vision transformer via `timm`: [`vit_small_patch16_224`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/data/video_embeddings.py#L35) (384-dimensional features).
* The text embeddings were extracted using [`GeminiEmbedder`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/llm/embedder.py) (`gemini-embedding-001`, 512 dimensions) or [`LocalEmbedder`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/llm/local_embedder.py) (`all-mpnet-base-v2`, 768 dimensions).
* These encoders produce **disjoint, incompatible vector spaces**.

---

### How a Shared Multimodal Contrastive Space Works
In models like **CLIP** (Radford et al., 2021) or **SigLIP** (Zhai et al., 2023), an image/video tower \( f_v \) and a text tower \( f_t \) are trained jointly on massive collections of paired images/videos and text descriptions using a contrastive loss:
\[
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\cos(f_v(v_i), f_t(c_i)) / \tau)}{\sum_j \exp(\cos(f_v(v_i), f_t(c_j)) / \tau)}
\]
This objective forces matching visual and textual representations to project onto **the exact same unit hypersphere \( \mathbb{S}^{D-1} \)**.

```
                  ┌──────────────────────┐
Text Reconstruction:  │ LLM In-filled Text   │ ──► CLIP Text Tower   ──┐
                  └──────────────────────┘                         │
                                                                   ▼
                                                            Shared Metric Space
                                                            (Direct Cosine / L2)
                                                                   ▲
                  ┌──────────────────────┐                         │
Video Reconstruction: │ Visual Interpolation │ ──► CLIP Vision Tower ──┘
                  └──────────────────────┘
```

### Why a Shared Space Would Transform Cross-Modal Comparison

1. **A Common Geometric Currency**:
   * With both predictions mapped to the same space, you can directly measure:
     \[
     \text{Error}_{\text{text}} = 1 - \cos(\hat{e}_{\text{text}}, v^*), \quad \text{Error}_{\text{vis}} = 1 - \cos(\hat{e}_{\text{vis}}, v^*)
     \]
     where \( v^* \) is the ground-truth visual frame vector. This directly evaluates whether the LLM's inferred caption is closer to the physical scene than visual interpolation is.
2. **True Cross-Modal Retrieval**:
   * Reconstructed text can be evaluated on its ability to directly retrieve ground-truth *video frames*.
   * Interpolated video vectors can be evaluated on their ability to retrieve ground-truth *text captions*.
3. **Eliminating the Need for Population Ranking Workarounds**:
   * Because distances are directly comparable, the need to "grade on a curve" via \( \Delta_{\text{rank}} \) is mitigated.

### Remaining Caveats in Contrastive Spaces
Even with CLIP/SigLIP, two theoretical biases remain:
1. **The Modality Gap**: Empirical research (Liang et al., 2022) demonstrated that image and text embeddings in contrastive models still segregate into two distinct cones on the hypersphere due to temperature tuning and initialization bias.
2. **"Lost in Embeddings" Information Bottleneck**: As documented in the paper's bibliography ([`docs/paper/sources.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/sources.md#L52-L74), citing Li et al., EMNLP 2025), contrastive alignment achieves cross-modal pairing by discarding fine-grained spatial and temporal details in favor of coarse semantic concepts. While this benefits text alignment, it penalizes fine-grained visual continuity.

---

## 6. Summary & Recommendations

When selecting metrics and criteria for comparative experiments:
* **For diagnosing domain predictability (Procedural vs. Stochastic)**: Use **Population Ranking Delta (\( \Delta_{\text{rank}} \))** combined with **Quintile Consistency** across masking widths. This remains the most robust diagnostic tool when operating across disjoint embedding spaces.
* **For evaluating event identification within a modality**: Use **MRR** and **Temporal NDCG** rather than raw cosine similarity to avoid static background bias.
* **For penalizing static visual continuity baselines**: Use **Context Residual Cosine Similarity (`cos_sim_residual`)**.
* **For next-generation architecture benchmarks**: Transitioning to a **Shared Multimodal Contrastive Space (SigLIP / VideoCLIP)** or an extrinsic downstream task (**VideoQA on masked intervals**) provides the cleanest path toward representation-neutral cross-modal evaluation.
