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

## 3. Master Aggregated Result Files (Current Benchmark Standard)

The authoritative, harmonized benchmark suite is compiled across **335 videos** (100 `wild4` + 235 `wild5`) using Llama-3.1-8B and 768-dim SigLIP/MPNet embeddings:

* **[`results/unified_benchmark_master.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/unified_benchmark_master.csv)**:
  The master evaluation table containing **36,152 rows** covering all 5 methods (`Llama-3.1-8B`, `Visual_SigLIP_MeanClosest`, `Visual_SigLIP_RepeatClosest`, `Caption_MeanClosest`, `Caption_RepeatClosest`) across gap widths \(W \in [1, 2, 3, 4, 6, 8, 12, 16]\) and query indices \(i \in [0, 29, 59]\). Includes `mrr`, `mean_rank`, `cos_sim_mean`, `cos_sim_min`, and `cos_sim_residual`.
* **[`results/apriori_full_scores.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/apriori_full_scores.csv)**:
  Pre-computed visual and textual video properties across all 335 videos: `APCS_V`, `average_dynamism`, `peak_dynamism`, `combined_dynamism`, `APCS_T`, `text_average_dynamism`, `text_peak_dynamism`, and `text_combined_dynamism`.
* **[`results/method_rank_differences_per_video.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/method_rank_differences_per_video.csv)**:
  Paired instance evaluations comparing Llama-3.1-8B vs. `Visual_SigLIP_MeanClosest`, containing computed ranks, rank differences (\(\Delta \text{Rank}\)), percentile advantages, and winner indicators.

### Archived Legacy Result Files (DO NOT USE for New Hypotheses)
The following files represent early exploratory runs (e.g. Phi-3 Mini, 384-dim embeddings, or contaminated window pools) and are preserved for historical provenance only:
* `results/final_correlations_master.csv` & `results/combined_analysis_data.csv` (Early Phi-3 runs).
* `results/temporal_metrics_final.csv` & `results/video_surprisal_scores.csv` (Early 384-dim exploratory runs).
* `results/baseline_full_metrics.csv` (Legacy baseline test cases).

---

## 4. Metadata Companion & Traceability Standard (`.md` next to `.csv`)

To ensure complete reproducibility and prevent confusion between experiment iterations, every primary CSV dataset and user-exported slice is accompanied by a Markdown metadata companion file with the **same base name and `.md` extension**:

```
results/
├── unified_benchmark_master.csv
├── unified_benchmark_master.md              # Auto-generated metadata companion
├── method_rank_differences_per_video.csv
└── method_rank_differences_per_video.md     # Auto-generated metadata companion
```

### Companion File Requirements:
1. **Provenance & Generation**: Records the exact script, configuration file, commit hash, date, and source cohort (`wild4`, `wild5`, or combined).
2. **Experiment Parameters**: Explicitly specifies the model name, gap widths \(W\), query positions \(i\), and distractor pool scope (`pool_scope: "video"` vs. legacy `"window"`).
3. **Column Data Dictionary**: Documents every column, data type, and mathematical definition, cross-referencing the master dictionary at [`docs/experiments/field_dictionary.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/experiments/field_dictionary.md).
4. **Automated Generation**:
   - CLI companion builder: `scripts/generate_csv_metadata_companion.py`
   - In-App companion export: In `scripts/evaluation_explorer_app.py` (Tab 5), clicking "Download Codebook & Column Explanation (.md)" automatically exports the corresponding companion markdown alongside the downloaded CSV.

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
