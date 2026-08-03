# Outputs and Results Guide

This document describes the directory structures, output file formats, column definitions, and master aggregated result files produced by experiment runs.

---

## 1. Directory Structure for Results

Experiment runs automatically produce structured output files saved across three primary locations:

```
caption_reconstruction/
├── results/
│   ├── recon/                       # Timestamped raw experiment output folders
│   │   └── <run_name>__<timestamp>/
│   │       ├── <run_name>.csv
│   │       └── <run_name>_z_score.csv
│   ├── for_analysis/                # Central repository of CSV copies for analysis scripts
│   │   ├── wild_dev_sim_vec_vid.csv
│   │   ├── wild_dev_sim_vec_vid_z_score.csv
│   │   ├── wild_dev_sim_one_shot_t=1.csv
│   │   └── wild_dev_sim_vec.csv
│   ├── final_correlations_master.csv # Master aggregated correlation dataset
│   ├── combined_analysis_data.csv    # Master summary table across methods & widths
│   └── baseline_full_metrics.csv    # Baseline clip-level retrieval metrics
└── mlruns/                          # MLflow tracking directory (parameters, metrics, logs)
```

---

## 2. Per-Experiment Output Files

Each experiment run generates two CSV files in `results/recon/<run_name>__<timestamp>/` (and copies them to `results/for_analysis/`):

1. **`<run_name>.csv`**: Contains raw metric statistics computed per video instance.
2. **`<run_name>_z_score.csv`**: Contains z-score normalized metric statistics calculated relative to the global corpus-wide distribution across all videos.

### CSV Column Definitions

#### **Metadata Fields**
* **`video_id`** (`str`): Unique identifier of the video or vector matrix.
* **`data_type`** (`str`): Data loader type (`CaptionedVideo`, `video_embeddings`, `text_embeddings(CaptionedVideo)`).
* **`recon_strategy`** (`str`): Reconstruction strategy used (e.g. `pro_d_one_shot_v1__t=1`, `RepeatClosestVector`, `BaselineRepeatStrategy`).
* **`size`** (`int`): Total number of clips/vectors in the video.
* **`masked`** (`list[int]`): List of clip indices that were masked during the run (e.g., `"[3, 4, 5, 6, 7, 8]"`).

#### **Cosine Similarity Metrics**
* **`cos_sim_mean`**: Mean cosine similarity score between reconstructed vectors and ground truth vectors across masked positions in this video.
* **`cos_sim_std`**: Standard deviation of cosine similarity scores across masked positions.
* **`cos_sim_min`**: Minimum cosine similarity score across masked positions.
* **`cos_sim_max`**: Maximum cosine similarity score across masked positions.

#### **Residual Cosine Similarity Metrics**
* **`cos_sim_residual_mean`**: Mean residual cosine similarity after projecting out unmasked context vectors. Measures new, un-shared semantic information.
* **`cos_sim_residual_std`**: Standard deviation of residual cosine similarity.
* **`cos_sim_residual_min`**: Minimum residual cosine similarity.
* **`cos_sim_residual_max`**: Maximum residual cosine similarity.

#### **Retrieval & Ranking Metrics** (Present when `evaluation.type` is `emb_retrieval` / `retrieval`)
* **`mean_rank_mean`**: Average rank of the ground-truth vector when retrieved against the distractor pool.
* **`mrr_mean`**: Mean Reciprocal Rank (\(1 / \text{rank}\)) across masked positions.
* **`recall_at_1_mean`**: Fraction of queries where the true vector was ranked #1.
* **`recall_at_5_mean`**: Fraction of queries where the true vector was in the top 5.
* **`retrieval_count_at_1_mean`**: Total number of top-1 retrieval hits.
* **`retrieval_total_queries_mean`**: Total number of evaluated retrieval queries.

---

## 3. Master Aggregated Result Files

Pre-computed master result files in the [`results/`](file:///home/yoavh/code/antigravity/caption_reconstruction/results) directory aggregate data across multiple experiments for paper analysis and plotting:

* **[`results/final_correlations_master.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/final_correlations_master.csv)**:
  Master correlation dataset combining model performance (`phi_mrr`, `temporal_ndcg`, `base_mrr`), metric deltas (`mrr_delta`, `t_ndcg_delta`), vector geometric distances (`euclidean_dist`, `video_avg_dist`, `video_max_dist`), video metadata (`video_length`, `category`), and language surprisal (`text_surprisal_nll`, `text_perplexity`).
* **[`results/combined_analysis_data.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/combined_analysis_data.csv)**:
  Aggregated metrics grouped by strategy `method` (e.g. `phi-3`) and mask `width`.
* **[`results/baseline_full_metrics.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/baseline_full_metrics.csv)**:
  Detailed retrieval metrics across 2,341 baseline test cases.
* **[`results/deep_analysis_final.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/deep_analysis_final.csv)** & **`deep_analysis_config_comparison.csv`**:
  Comprehensive comparative evaluation between visual and text-based reconstruction models.
* **[`results/temporal_metrics_final.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/temporal_metrics_final.csv)**:
  Temporal alignment and sequence ordering metrics across video domains.
* **[`results/euclidean_metrics.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/euclidean_metrics.csv)**:
  L2 distance analysis between video embedding spaces.
* **[`results/video_surprisal_scores.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/video_surprisal_scores.csv)**:
  Information-theoretic surprisal scores per video.

---

## 4. MLflow Experiment Tracking

Experiments are logged to MLflow under `mlruns/`.
To view logged parameters, metrics, run graphs, and artifact outputs in an interactive web UI:

```bash
mlflow ui
```

Or view run hierarchies via command line:

```bash
python scripts/mlflow_runs.py ./mlruns
```
