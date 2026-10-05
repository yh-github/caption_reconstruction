# TODO

## Dataset Quality Audits & Captions Status

- [x] **wild4 Captions Generation & Audit**: **100% Complete** (100/100 clips in `datasets/wildQA/captions__wild4/`).
- [x] **wild5 Captions Generation & Audit**: **100% Complete** (235/235 clips in `datasets/wildQA/captions__wild5/`).
- [x] **SigLIP Visual Embeddings Extraction**: **100% Complete** (335/335 clips in `local/wild_videos_embs_siglip/`).
- [x] **A-Priori Scores Generation**: **100% Complete** (Visual Dynamism, APCS_V, Textual Dynamism, APCS_T across 335 videos in `results/apriori_full_scores.csv`).

## Downstream Reconstruction Experiments (Llama-3.1-8B)

- [x] **wild4 Llama sweeps** (Multi-width w ∈ [1..16], w=6 window, w=3 window): Complete and archived.
- [x] **wild5 Llama w3/w6 benchmarks** (`wild5_llama_w3_w6.yaml`): Complete and synced from HF.
- [x] **wild5 Llama multi-width sweep** (`wild5_llama_multi_width.yaml`): **100% Complete** across all 235 videos.
- [x] **Baselines Benchmark Sweep**: **100% Complete** for all 4 baseline methods (`Visual_SigLIP_MeanClosest`, `Visual_SigLIP_RepeatClosest`, `Caption_MeanClosest`, `Caption_RepeatClosest`) across all active gap widths \(W \in [1, 2, 3, 4, 6, 8, 12, 16]\).
- [x] **Unified Master Dataset Assembly**: **100% Complete** (36,152 evaluation rows across 335 videos compiled in `results/unified_benchmark_master.csv`).
- [x] **Cross-Cohort Analysis & Final Curves**: **100% Complete** (Live interactive analysis suite in `scripts/evaluation_explorer_app.py` featuring Llama Winners Explorer, Method Comparisons with Rank Diffs, Macro W-curves, Micro per-method scatter, Stratified Cohorts, and Data/Codebook Export).

