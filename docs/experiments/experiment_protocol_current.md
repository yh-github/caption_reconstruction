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

---

## 2. Current Setup vs. Legacy Experiments

### What is Current (Active Standard)
1. **Generative Text Model**: **Llama-3.1-8B** (`llama-3.1-8b__whole_window__t=0.6`, repetition penalty 1.05).
   - Prompt directory: `prompts/dense_window/`
   - Evaluation embedding model: `all-mpnet-base-v2` (`768-dim`, `pool_scope: "video"`)
   - Configs:
     - `config/embs_vs_slms/wild4_llama_multi_width.yaml` (w ∈ [1, 2, 4, 8, 12, 16, 24, 30] at i=29)
     - `config/embs_vs_slms/wild4_llama_w6.yaml` (w=6, i ∈ [0, 29, 59])
     - `config/embs_vs_slms/wild5_llama_w3_w6.yaml` (w ∈ [3, 6], i ∈ [0, 29, 59])
     - `config/embs_vs_slms/wild5_llama_multi_width.yaml` (currently running on Kaggle)
2. **Visual Representation Baseline**: **Google SigLIP** (`google/siglip-base-patch16-224`, `768-dim`).
   - Strategy: `MeanClosestVectors` / `RepeatClosestVector`
   - Configs: `config/embs_vs_slms/wild4_siglip_sim_vec_vid.yaml`, `config/embs_vs_slms/wild5_siglip_sim_vec_vid.yaml`
3. **Harmonized Distractor Pool**:
   - `pool_scope: "video"` (queries evaluated against all other 59 timestamps in the 60-second video; Recall@1 chance = $1/60 \approx 0.017$; MRR chance $\approx 0.078$).

### What is Legacy (DO NOT USE for New Hypotheses)
1. **Phi-3 / SLM text evaluations**: Early exploratory runs using Phi-3 Mini with varying temperatures (`wild_dev_sim_text`). Replaced by Llama-3.1-8B.
2. **Old 384-dim video embeddings**: Located in `local/wild_videos_embs/` (101 files). Superseded by 768-dim SigLIP.
3. **`phi_vs_video_integration_summary.csv` & `temporal_metrics_final.csv`**: Legacy summaries built by joining Phi-3 text outputs with 384-dim video embeddings. Contaminated and superseded.
4. **`wild4_llama_w3_window` anomaly**: An early run of w=3 on `wild4` using `prompts/dense_window_v1.txt` had an anomalous MRR of ~0.55. On `wild5` with standard prompts, w=3 yields an expected MRR of ~0.10 - 0.14.

---

## 3. A-Priori Metrics Suite

All 335 videos have pre-computed visual, textual, and linguistic scores stored in **`results/apriori_full_scores.csv`**:

1. **Visual Dynamism (`SigLIP 768-dim`)**:
   - `average_dynamism`: Mean consecutive frame distance 1 - sim(v_t, v_{t+1}).
   - `peak_dynamism`: 95th percentile frame distance.
   - `combined_dynamism`: 100 * (0.5 * avg + 0.5 * p95).
   - `APCS_V` (Gail's Visual Metric): Average Pairwise Cosine Similarity across all frame pairs in the video.
2. **Textual Dynamism (`all-mpnet-base-v2`)**:
   - `text_average_dynamism`, `text_peak_dynamism`, `text_combined_dynamism` computed sequentially on ground-truth captions.
   - `APCS_T`: Average Pairwise Cosine Similarity across all ground-truth caption pairs in the video.
3. **Linguistic Surprisal**:
   - `apcs_nll`, `caption_perplexity` (from language model prior scoring where available).

---

## 4. Key Scientific Findings So Far

From our correlation analysis on 325 videos (`results/analysis_correlations_w6_i29.csv`):

| Evaluation Metric | Apriori Predictor | Spearman ρ | Significance (p) | Scientific Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **MRR** | Visual `combined_dynamism` | -0.078 | p = 0.158 (n.s.) | MRR is **unbiased** by visual dynamism. |
| **MRR** | Visual `APCS_V` | +0.023 | p = 0.683 (n.s.) | Video frame similarity does not dictate LLM temporal ranking. |
| **MRR** | Textual `APCS_T` | -0.004 | p = 0.948 (n.s.) | Ground truth caption repetition does not distort MRR. |
| **`cos_sim_mean`** | Visual `combined_dynamism` | -0.281 | p = 2.5e-07 | Weak-to-moderate correlation with visual motion. |
| **`cos_sim_mean`** | **Textual `text_combined_dynamism`** | **-0.519** | p = 7.6e-24 | Captions that vary heavily yield lower cosine similarity. |
| **`cos_sim_mean`** | **Textual `APCS_T`** | **+0.606** | p = 6.2e-34 | **Major Confounder**: videos with repetitive captions give artificially high cosine similarity. |

**Crucial Methodological Conclusion**: 
- `cos_sim_mean` is heavily confounded by the inherent redundancy of the ground-truth captions.
- **MRR is our most robust, unbiased metric** for measuring true temporal cloze reasoning.

---

## 5. Next Steps

1. **Complete Kaggle Multi-Width Run (`wild5_llama_multi_width`)**:
   - Sweep `w ∈ [1, 2, 4, 8, 12, 16, 24, 30]` at `i=29` across all 235 videos in `wild5`.
   - Results will automatically sync to Hugging Face (`Y3/dense_video_captions`).
2. **Download & Integrate `wild5` Multi-Width Results**:
   - Run `scripts/download_hf_results.py` / `huggingface_hub` snapshot.
   - Run `scripts/aggregate_llama_results.py` to compile the complete 335-video multi-width dataset.
3. **Cross-Cohort Analysis & Final Curves**:
   - Plot performance degradation vs. gap width W across the full 335 videos.
   - Compare `Llama-3.1-8B` vs. `SigLIP Video Baseline` vs. `Caption Vector Baseline` across dynamism bins.
