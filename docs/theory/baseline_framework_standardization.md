# Baseline Framework Standardization & Cross-Modal Alignment

## 1. Overview & Architectural Motivation

In the dense video caption reconstruction task, our goal is to compare **semantic generative inference** (LLMs/SLMs deducing missing events from surrounding text) against **continuity-based representation baselines** across both language and vision:
1. **Caption Vector Baseline (`Vector_Caption`)**: Linearly interpolates dense text embeddings (`all-mpnet-base-v2`) across the masked gap.
2. **Video Vector Baseline (`Vector_Video`)**: Linearly interpolates dense visual frame features (`SigLIP`) across the masked gap.
3. **Generative Language Model (`SLM_Text`)**: Generates natural language cloze captions using Llama-3.1-8B (or Phi-3), then embeds the generated text into the same text representation space.

To maintain scientific rigor and prevent pipeline divergence, all baseline evaluations are executed through the core `ExperimentPipeline` framework (`RECON_VECTORS`), preserving identical data loading, masking schemes, metric definitions, and distractor pool structures.

---

## 2. Key Methodological Decisions

### Decision 1: Distractor Pool Scope Harmonization (`pool_scope: "video"`)

#### The Historical Discrepancy
* In legacy runs (e.g. `wild4_sim_vec.csv` from Sept 8, 2026), `VectorReconstructionEvaluator_Retrieval` defaulted to:
  \[
  \text{distractor\_pool} = \text{true\_vecs}
  \]
  where \(\text{true\_vecs}\) was restricted solely to the \(W\) clips inside the masked gap (\(W \in \{3, 6\}\)). As a result, candidate queries were ranked against only \(2\) or \(5\) distractors.
* Conversely, generative LLM evaluations (`wild4_llama_multi_width.yaml`, `wild5_llama_w3_w6.yaml`) were evaluated with `pool_scope: "video"`, ranking each reconstructed caption against **all 60 original clips in the video** (\(59\) distractors).
* Comparing an MRR or Recall@1 calculated against \(3\) candidates with one calculated against \(60\) candidates introduces severe scale distortion.

#### The Standardized Solution
`VectorReconstructionEvaluator_Retrieval` and `VectorRunner` have been upgraded to support `pool_scope: "video"`.
* When `pool_scope: "video"`, `VectorRunner` injects the complete video embedding matrix \(\mathbf{M} \in \mathbb{R}^{60 \times D}\) as `full_vecs`, alongside the masked indices \(I_{\text{mask}} \subset \{0, \dots, 59\}\):
  \[
  \text{distractor\_pool} = \mathbf{M}
  \]
  \[
  \text{gt\_indices\_in\_pool} = I_{\text{mask}}
  \]
* In `src/evaluations/eval_vectors.py`, the query's own ground truth index in \(\mathbf{M}\) is masked out by setting its similarity to \(-\infty\).
* The remaining 59 clips in the video serve as background distractors, ensuring **identical rank distribution and chance-level probability** (\(1/60 \approx 0.0167\)) across both LLMs and vector baselines.

---

## 3. Visual Representation Space (SigLIP)

* **Previous Visual Features**: Early experiments used older visual vectors under `local/wild_videos_embs/`.
* **Standardized Visual Features**: All current and future visual vector baselines use Google's **SigLIP** (`google/siglip-base-patch16-224`, 768-dimensional), stored under:
  ```
  local/wild_videos_embs_siglip/
  ```
* **Dataset Cohort Filtering**: Because `local/wild_videos_embs_siglip/` will house embeddings for multiple cohorts, `VectorFileLoader` supports `filter_dir`, filtering numpy files to match the exact video stem set present in `datasets/wildQA/captions__wild4/` or `datasets/wildQA/captions__wild5/`.

---

## 4. Configuration Inventory

The unified baseline suite consists of four primary configuration files under `config/embs_vs_slms/`:

| Config File | Benchmark Cohort | Modality Evaluated | Embedding Model / Path | Distractor Scope |
| :--- | :--- | :--- | :--- | :--- |
| `wild4_sim_vec.yaml` | Wild4 (100 videos) | Text Captions | `local:all-mpnet-base-v2` | `video` (60 clips) |
| `wild4_siglip_sim_vec_vid.yaml` | Wild4 (100 videos) | Video Frames | `local/wild_videos_embs_siglip/` | `video` (60 clips) |
| `wild5_sim_vec.yaml` | Wild5 (235 videos) | Text Captions | `local:all-mpnet-base-v2` | `video` (60 clips) |
| `wild5_siglip_sim_vec_vid.yaml` | Wild5 (235 videos) | Video Frames | `local/wild_videos_embs_siglip/` | `video` (60 clips) |

Each config evaluates two classical heuristic strategies across `width: [3, 6, 9, 12]` and `start_ind: [0, 29, 59]`:
* `MeanClosestVectors`: \(\hat{v}_i = \frac{v_{\text{before}} + v_{\text{after}}}{2}\)
* `RepeatClosestVector`: \(\hat{v}_i = v_{\text{closest\_boundary}}\)

---

## 5. Execution Workflow

To run any baseline through the native pipeline:

```bash
# 1. Wild4 Caption Vector Baseline
.venv/bin/python src/main.py config/embs_vs_slms/wild4_sim_vec.yaml --ignore-unsafe

# 2. Wild4 Video Vector Baseline (SigLIP)
.venv/bin/python src/main.py config/embs_vs_slms/wild4_siglip_sim_vec_vid.yaml --ignore-unsafe

# 3. Wild5 Caption Vector Baseline
.venv/bin/python src/main.py config/embs_vs_slms/wild5_sim_vec.yaml --ignore-unsafe

# 4. Wild5 Video Vector Baseline (SigLIP)
.venv/bin/python src/main.py config/embs_vs_slms/wild5_siglip_sim_vec_vid.yaml --ignore-unsafe
```

### Result Artifacts
Each run automatically generates:
1. `results/recon/{config_name}/{config_name}.csv`: Per-video aggregated statistics.
2. `results/recon/{config_name}/{config_name}_z_score.csv`: Z-score normalized statistics.
3. Automatically mirrored copies in `results/for_analysis/`.
4. Merged downstream via `scripts/build_unified_wild4_dataset.py` (and equivalent Wild5 script) into master comparison datasets.
