# Master Field Dictionary for Caption Reconstruction Benchmarks

This dictionary defines standard column names, types, value ranges, and mathematical definitions used across all benchmark evaluation CSVs and analysis summaries.

---

## 1. Identifiers & Metadata

| Field Name | Type | Description | Example / Range |
| :--- | :--- | :--- | :--- |
| `video_id` | String | Unique identifier of the video clip (standard format: `Channel_N-clip-M` or `Channel_N-manual`). | `"BC-Bushcraft_10-clip-8"` |
| `movie_id` | String | Channel / source entity identity grouping multiple clips (~40 channels). | `"BC-Bushcraft"` |
| `dataset` | String | Benchmark cohort source split (`wild4` = dev split, 100 clips; `wild5` = test split, 235 clips). | `"wild4"`, `"wild5"` |
| `experiment` | String | Full configuration name of the reconstruction strategy and parameter set. | `"llama-3.1-8b__whole_window__t=0.6__fixed_fill(w=6, i=29)"` |
| `exp_category` | String | Experiment folder / cohort category grouping. | `"wild4_llama_multi_width"` |
| `w` | Integer | Gap width $W$ (number of consecutive seconds/captions masked and reconstructed). | `1, 2, 3, 4, 6, 8, 12, 16, 24, 30` |
| `i` | Integer | Start index $i$ of the masked gap within the 60-second window (0-indexed). | `0, 29, 59` |
| `video_length` | Integer | Total duration of the clip in seconds (typically 60-65s). | `60` |

---

## 2. A-Priori Visual & Textual Dynamism Metrics

All a-priori scores are computed **prior to masking and reconstruction**.

| Field Name | Modality & Model | Mathematical Formula | Description |
| :--- | :--- | :--- | :--- |
| `average_dynamism` | Vision (`SigLIP 768d`) | $\frac{1}{T-1} \sum_{t=1}^{T-1} (1 - \cos(v_t, v_{t+1}))$ | Mean consecutive visual frame distance. Measures overall motion speed. |
| `peak_dynamism` | Vision (`SigLIP 768d`) | $\text{Percentile}_{95} \{ 1 - \cos(v_t, v_{t+1}) \}$ | 95th percentile consecutive frame distance. Measures sudden visual scene cuts/actions. |
| `combined_dynamism` | Vision (`SigLIP 768d`) | $100 \times [0.5 \times \text{avg\_dyn} + 0.5 \times \text{peak\_dyn}]$ | Scaled composite score of visual dynamism. Higher = more volatile video. |
| `APCS_V` | Vision (`SigLIP 768d`) | $\frac{2}{T(T-1)} \sum_{j < k} \cos(v_j, v_k)$ | Gail's **Average Pairwise Cosine Similarity** across all video frame pairs. Higher = static/monotonous visuals. |
| `text_average_dynamism` | Text (`mpnet 384d`) | $\frac{1}{T-1} \sum_{t=1}^{T-1} (1 - \cos(c_t, c_{t+1}))$ | Mean consecutive cosine distance between ground-truth captions. Measures narrative pace. |
| `text_peak_dynamism` | Text (`mpnet 384d`) | $\text{Percentile}_{95} \{ 1 - \cos(c_t, c_{t+1}) \}$ | 95th percentile caption step distance. Measures abrupt narrative topic shifts. |
| `text_combined_dynamism` | Text (`mpnet 384d`) | $100 \times [0.5 \times \text{text\_avg} + 0.5 \times \text{text\_peak}]$ | Composite score of textual dynamism. Higher = rapidly changing caption content. |
| `APCS_T` | Text (`mpnet 384d`) | $\frac{2}{T(T-1)} \sum_{j < k} \cos(c_j, c_k)$ | **Average Pairwise Cosine Similarity** across all ground-truth captions in the clip. Higher = repetitive captions. |
| `apcs_nll` | Language Model | Negative Log Likelihood (NLL) of the caption sequence. | Measures linguistic rarity/surprisal of the caption transcript. |
| `caption_perplexity` | Language Model | $\exp(\text{NLL})$ | Perplexity of the ground-truth captions. |

---

## 3. Evaluated Posteriori Reconstruction Metrics

Reconstruction metrics measure the quality of reconstructed captions/vectors against ground truth.

| Field Name | Range | Evaluation Space | Description |
| :--- | :--- | :--- | :--- |
| `mrr` | $[0.0, 1.0]$ | Distractor pool (`pool_scope: "video"`) | **Mean Reciprocal Rank**. Evaluated against all 59 background timestamps in the video ($1/\text{rank}$). Random chance $\approx 1/60 = 0.0167$. **Unbiased metric**. |
| `mean_rank` | $[1.0, 60.0]$ | Distractor pool (`pool_scope: "video"`) | Average rank of the ground-truth timestamp among all 60 video candidates. Lower is better (1 = perfect match). |
| `recall_at_1` | $[0.0, 1.0]$ | Distractor pool (`pool_scope: "video"`) | Fraction of reconstructed timestamps ranked #1 against all distractors. |
| `recall_at_5` | $[0.0, 1.0]$ | Distractor pool (`pool_scope: "video"`) | Fraction of reconstructed timestamps ranked in top 5 against all distractors. |
| `cos_sim_mean` | $[-1.0, 1.0]$ | Embedding space (`all-mpnet-base-v2` for text; `SigLIP` for video) | Mean elementwise cosine similarity between reconstructed and ground-truth vectors. Note: Confounded by `APCS_T` in text space. |
| `cos_sim_min` | $[-1.0, 1.0]$ | Embedding space | Minimum cosine similarity across the reconstructed timestamps in the gap. |
| `cos_sim_max` | $[-1.0, 1.0]$ | Embedding space | Maximum cosine similarity across the reconstructed timestamps in the gap. |
| `cos_sim_residual_mean`| $[-1.0, 1.0]$| Embedding space | Cosine similarity after subtracting the video-level centroid/mean vector to penalize static bias. |

---

## 4. Analysis & Comparison Metrics

| Field Name | Description |
| :--- | :--- |
| `mrr_text` / `llama_mrr` | MRR attained by the generative language model (Llama-3.1-8B). |
| `mrr_video` / `siglip_mrr` | MRR attained by the visual continuity baseline (`MeanClosestVectors` on SigLIP). |
| `mrr_lift` / `delta_mrr` | Performance advantage of Llama over visual continuity: $\text{MRR}_{\text{LLM}} - \text{MRR}_{\text{Video}}$. |
| `cos_sim_lift` | Cosine similarity difference between LLM and baseline (valid only within identical embedding spaces). |
| `spearman` / `pearson` | Rank-order (Spearman $\rho$) and linear (Pearson $r$) correlation coefficients between apriori and posteriori metrics. |
