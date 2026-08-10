# TODO

## Dataset Quality Audits & Captions Status

- [x] **wild4 Captions Generation & Audit**: **100% Complete** (100/100 clips generated in `datasets/wildQA/captions__wild4/`).
  - Quality Audit: [`docs/data/audit_captions/audit_wild4_results.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/data/audit_captions/audit_wild4_results.md) (0 duplicate clusters found, 79% domain accuracy).
- [x] **wild5 Captions Generation & Audit**: **100% Complete** (109/109 clips generated in `datasets/wildQA/captions__wild5/`).
  - Quality Audit: [`docs/data/audit_captions/audit_wild5_results.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/data/audit_captions/audit_wild5_results.md) (0 duplicate clusters found, 77.1% domain accuracy).

## Downstream Reconstruction Experiments


- [ ] **Run reconstruction experiments on wild4 dataset** (`datasets/wildQA/captions__wild4/`)
  - [ ] Create experiment configs for SLM (Phi-3) and LLM (Gemini Pro) on `wild4`.
  - [ ] Run benchmark evaluation sweeps and compile CSV metrics into `results/for_analysis/`.
  - [ ] Sync result artifacts to HuggingFace dataset `Y3/dense_video_captions`.

- [ ] **Run reconstruction experiments on wild5 dataset** (`datasets/wildQA/captions__wild5/`)
  - [ ] Create experiment configs for SLM (Phi-3) and LLM (Gemini Pro) on `wild5`.
  - [ ] Run benchmark evaluation sweeps and compile CSV metrics into `results/for_analysis/`.
  - [ ] Sync result artifacts to HuggingFace dataset `Y3/dense_video_captions`.

