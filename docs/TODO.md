# TODO

## Dataset Quality Audits & Captions Status

- [x] **wild4 Captions Generation & Audit**: **100% Complete** (100/100 clips in `datasets/wildQA/captions__wild4/`).
- [x] **wild5 Captions Generation & Audit**: **100% Complete** (235/235 clips in `datasets/wildQA/captions__wild5/`).
- [x] **SigLIP Visual Embeddings Extraction**: **100% Complete** (335/335 clips in `local/wild_videos_embs_siglip/`).
- [x] **A-Priori Scores Generation**: **100% Complete** (Visual Dynamism, APCS_V, Textual Dynamism, APCS_T across 335 videos in `results/apriori_full_scores.csv`).

## Downstream Reconstruction Experiments (Llama-3.1-8B)

- [x] **wild4 Llama sweeps** (Multi-width w ∈ [1..30], w=6 window, w=3 window): Complete and archived.
- [x] **wild5 Llama w3/w6 benchmarks** (`wild5_llama_w3_w6.yaml`): Complete and synced from HF.
- [/] **wild5 Llama multi-width sweep** (`wild5_llama_multi_width.yaml`): **In Progress on Kaggle**.
- [ ] **Cross-Cohort Analysis & Final Curves**:
  - [ ] Download wild5 multi-width outputs once Kaggle completes.
  - [ ] Run `scripts/aggregate_llama_results.py` on full 335-video set.
  - [ ] Generate crossover and degradation curves across gap width W and dynamism strata.

