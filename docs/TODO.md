# TODO

## Wild4 Dataset Audit

- [ ] **Re-run duplication & mislabeling check once wild4 captioning is complete.**
  - Currently only 34/95 files have been generated in `datasets/wildQA/captions__wild4/`.
  - Initial audit (2026-02-25) found **no duplication or mislabeling** in the partial data — all 34 files have unique content that matches their channel names.
  - However, the channels most responsible for duplicates in the old dataset (`Survival-Instinct`, `Weathershot`, `Primal-Earth-Sounds`, `King-Kong-Amazon`) have **zero files** in wild4 so far. These need to be verified once generated.
  - See `docs/paper/dataset_taxonomy.md` for the original duplication clusters found in the old data.
