# Experiment Protocol & State of the Art (October 2026)

This document establishes the authoritative ground truth for datasets, experiment configurations, apriori scores, and downstream evaluations to ensure full reproducibility and prevent confusion between current setups and legacy artifacts.

---

## 1. Datasets & Embedding Cohorts

| Cohort | Captions Path | Total Videos | SigLIP Embeddings (`768-dim`) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`wild4`** | `datasets/wildQA/captions__wild4/` | **100** | **100 / 100** | Complete | Sourced from WildQA dev split. (Note: filenames use `'` in captions but `_` on disk for Olly's Farm). |
| **`wild5`** | `datasets/wildQA/captions__wild5/` | **235** | **235 / 235** | Complete | Sourced from WildQA test split. Nested raw videos extracted with `--match_existing ""` in `.venv`. |
| **Combined** | `wild4` + `wild5` | **335** | **335 / 335** | Complete | Full benchmark suite for all apriori and downstream reconstruction tasks. |

> **Filename Normalization Rule**: Filesystem paths and video embedding files use underscores for special characters (e.g. `Olly_s-Farm_*.npy`), whereas original JSON caption metadata often has apostrophes (e.g. `Olly's-Farm_*.json`). Always normalize (`video_id.replace("Olly_s-Farm", "Olly's-Farm")`) when joining datasets.

> **Channel Clustering & Hierarchical Nesting**: The 335 clips originate from exactly **40 distinct YouTube channels** (`movie_id`), strictly nested inside semantic categories. Because intra-channel correlation is significant (e.g. \(\text{ICC} = 0.31\) for visual dynamism), statistical analyses must cluster standard errors by channel to prevent pseudoreplication. See [`docs/data/movie_id_channel_clustering_analysis.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/data/movie_id_channel_clustering_analysis.md) for full variance decomposition.

---

## 2. Current Setup vs. Legacy Experiments

### What is Current (Active Standard)
1. **Generative Text Model**: **Llama-3.1-8B** (`llama-3.1-8b__whole_window__t=0.6`, repetition penalty 1.05).
   - Prompt directory: `prompts/dense_window/`
   - Evaluation embedding model: `all-mpnet-base-v2` (`768-dim`, `pool_scope: "video"`)
   - Completed Benchmark: 335 videos (100 `wild4` + 235 `wild5`) across gap widths \(W \in [1, 2, 3, 4, 6, 8, 12, 16]\) at \(i=29\).
   - Gap widths \(W > 16\) (\(W=24\) with \(N=4\), \(W=30\) with \(N=17\)) are excluded from the official benchmark due to prompt context window limits causing widespread caption truncation and video skipping.
2. **Visual Representation Baseline**: **SigLIP 2** (`768-dim`; timm `vit_base_patch16_siglip_224.v2_webli`, text tower `google/siglip2-base-patch16-224`). Correction (2026-10-07): earlier docs and `metadata.yaml` say `google/siglip-base-patch16-224` (SigLIP 1), but timm's untagged name resolved to SigLIP 2 weights. Frame-only and text-only results are unaffected; any analysis that scored SigLIP 1 text against these frames is invalid.
   - Strategy: `Visual_SigLIP_MeanClosest` (boundary midpoint linear interpolation LERP at \(\alpha = 0.5\): \(\frac{v_{i-1} + v_{i+W}}{2}\)) and `Visual_SigLIP_RepeatClosest`.
   - Distractor pool: `pool_scope: "video"` (queries ranked against all 60 timestamps in the video).
3. **Harmonized Distractor Pool**:
   - `pool_scope: "video"` (queries evaluated against all other 59 timestamps in the 60-second video; Recall@1 chance = \(1/60 \approx 0.017\); MRR chance \(\approx 0.078\); Mean Rank chance \(= 30.5\); Calibrated AUC chance \(= 0.0\%\)).

### What is Legacy (DO NOT USE for New Hypotheses)
1. **Phi-3 / SLM text evaluations**: Early exploratory runs using Phi-3 Mini with varying temperatures (`wild_dev_sim_text`). Replaced by Llama-3.1-8B.
2. **Old 384-dim video embeddings**: Located in `local/wild_videos_embs/` (101 files). Superseded by 768-dim SigLIP.
3. **`phi_vs_video_integration_summary.csv` & `temporal_metrics_final.csv`**: Legacy summaries built by joining Phi-3 text outputs with 384-dim video embeddings. Contaminated and superseded.
4. **`wild4_llama_w3_window` anomaly**: An early run of \(W=3\) on `wild4` (`wild4_llama_w3_window_v3`) reported an anomalous MRR of \(\approx 0.558\) and Recall@5 of \(100\%\).
   - **Root Cause Identified**: The evaluation config erroneously set `pool_scope: "window"` (\(N=3\) masked timestamps only) instead of `pool_scope: "video"` (\(N=60\)).
   - For \(N=3\), random chance MRR is \(H_3/3 = (1 + 1/2 + 1/3)/3 \approx 0.611\), Mean Rank chance is \(2.0\), and Recall@5 is mathematically \(1.0\) (100%). Llama's score of \(0.558\) was actually below chance.
   - Clean Wild5 \(W=3\) under `pool_scope: "video"` (\(N=60\)) yields the true score: MRR \(\approx 0.123\), Mean Rank \(= 24.56\).
   - The interactive explorer app provides an explicit sidebar toggle (`exclude_w3_anomaly`) to automatically filter out this contaminated slice.

---

## 3. A-Priori Metrics Suite

All 335 videos have pre-computed visual, textual, and linguistic scores stored in **`results/apriori_full_scores.csv`**:

1. **Visual Dynamism (`SigLIP 768-dim`)**:
   - `average_dynamism`: Mean consecutive frame distance \(1 - \cos(v_t, v_{t+1})\).
   - `peak_dynamism`: 95th percentile frame distance.
   - `combined_dynamism`: \(100 \times (0.5 \times \text{avg} + 0.5 \times \text{p95})\).
   - `APCS_V` (Gail's Visual Metric): Average Pairwise Cosine Similarity across all frame pairs in the video.
2. **Textual Dynamism (`all-mpnet-base-v2` 768-dim)**:
   - `text_average_dynamism`, `text_peak_dynamism`, `text_combined_dynamism` computed sequentially on ground-truth captions.
   - `APCS_T`: Average Pairwise Cosine Similarity across all ground-truth caption pairs in the video.
3. **Linguistic Surprisal**:
   - `apcs_nll` (Negative Log Likelihood). `caption_perplexity` is deprecated/omitted in unified exports (`None` for all videos).
4. **Omission of `num_captions`**:
   - In 334 out of 335 benchmark videos, exactly 60 1-second clips exist (1 video has 61 due to duration edge cases). Because it has near-zero variance, `num_captions` was removed from the active a-priori feature set and app filters.

---

## 4. Key Scientific Findings So Far

### A. Metric Confounders vs. Robust Ranking
From correlation analysis on the benchmark (`results/analysis_correlations_w6_i29.csv`):

| Evaluation Metric | Apriori Predictor | Spearman \(\rho\) | Significance (\(p\)) | Scientific Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **MRR** | Visual `combined_dynamism` | -0.078 | \(p = 0.158\) (n.s.) | MRR is **unbiased** by visual dynamism. |
| **MRR** | Visual `APCS_V` | +0.023 | \(p = 0.683\) (n.s.) | Video frame similarity does not dictate LLM temporal ranking. |
| **MRR** | Textual `APCS_T` | -0.004 | \(p = 0.948\) (n.s.) | Ground truth caption repetition does not distort MRR. |
| **`cos_sim_mean`** | Visual `combined_dynamism` | -0.281 | \(p = 2.5 \times 10^{-7}\) | Weak-to-moderate correlation with visual motion. |
| **`cos_sim_mean`** | **Textual `text_combined_dynamism`** | **-0.519** | \(p = 7.6 \times 10^{-24}\) | Captions that vary heavily yield lower cosine similarity. |
| **`cos_sim_mean`** | **Textual `APCS_T`** | **+0.606** | \(p = 6.2 \times 10^{-34}\) | **Major Confounder**: videos with repetitive captions give artificially high cosine similarity. |

> [!WARNING]
> **Caveat on "Unbiased MRR" & Noise Attenuation**: Near-zero correlations between MRR and apriori features must be interpreted with caution. Because Llama's MRR sits near the chance floor (\(\approx 0.08\) vs \(0.078\)), measurement noise from the reciprocal cliff (\(1/r\)) dominates the signal, which heavily attenuates correlations toward zero. To detect true relationships without noise attenuation, evaluations should rely on **Calibrated AUC** (\(c\)) and **Paired Rank Differences** (\(\Delta \text{Rank}\)), which scale linearly with rank.

### B. Horizon Scaling & Modality Convergence
- As gap width \(W\) increases from 3 to 12, Llama's win rate rises from \(19.6\%\) to \(39.1\%\), and the margin closes from \(-0.091\) to \(-0.009\).
- **Critical Insight**: This convergence is **not** due to Llama improving at long gaps (Llama's MRR stays near the floor, \(0.119 \to 0.083\)). Instead, **visual continuity collapses toward chance** (SigLIP MRR falls from \(0.175 \to 0.092\)) because visual frames decorrelate over 12–16 second intervals.

---

## 5. Completed Infrastructure & Interactive Tools

1. **Unified Master Benchmark Dataset**:
   - Compiled in [`results/unified_benchmark_master.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/unified_benchmark_master.csv) (36,152 evaluation rows across 335 videos, including `cos_sim_min` and `cos_sim_residual`).
2. **Interactive Streamlit Explorer (`scripts/evaluation_explorer_app.py`)**:
   - **🏆 Llama Winners Explorer**: Conditional analysis tab isolating instances where Llama outperforms visual baselines based on prior sliders (`combined_dynamism`, `APCS_V`, `cos_sim_min`). Includes qualitative deep-dive panels (Context Before → Predictions vs Ground Truth → Context After).
   - **⚔️ Method Comparison & Rank Diffs**: Calculates intra-cohort ranking deltas (\(\Delta \text{Rank} = \text{Rank}_A - \text{Rank}_B\)) and Spearman rank correlations against prior features.
   - **📈 Macro View (W-Curves)**: Plots performance degradation across \(W \in [1, 2, 3, 4, 6, 8, 12, 16]\) for all 5 methods simultaneously.
   - **🔬 Micro View**: Per-method, per-instance scatter plots with OLS trendlines.
   - **📊 Stratified Cohorts**: Dynamic percentile splitting (top vs bottom quantile) conditioned on any a-priori feature.
   - **💾 Data & Codebook Export**: Export raw or paired benchmark slices with auto-generated Markdown codebooks matching the metadata companion standard.
3. **Qualitative Victory Documentation**:
   - Detailed qualitative case studies and failure archetypes documented in [`docs/experiments/slm_qualitative_victories.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/experiments/slm_qualitative_victories.md).
