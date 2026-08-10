# Dataset Caption Audit Reports

This directory stores comprehensive quality audit reports for generated video caption datasets (`captions__wild1` through `captions__wild5` under `datasets/wildQA/`).

## 📊 Purpose

Before running downstream reconstruction or embedding experiments, each caption dataset undergoes automated quality screening to verify dataset integrity:

1. **Deduplication & Pairwise Similarity**: Cosine similarity & n-gram Jaccard overlap checks to detect duplicate video clips or mislabeled channel files.
2. **Ad & Commercial Phrase Screening**: Regex pattern matching for sponsor disclaimers, channel subscription requests, and commercial boilerplate.
3. **Domain Alignment**: Keyword profile verification comparing predicted video domains (Military, Agriculture, Human Survival, Geography, Natural Disaster) against ground-truth annotations in `dev.json` / `test.json`.
4. **Caption Quality Issues**: Screening for loop/repetitive captions or empty segment fields.

## 📄 Included Reports

* [`audit_wild1_results.md`](audit_wild1_results.md): Initial audit report for `wild1`.
* [`audit_wild2_results.md`](audit_wild2_results.md): Quality audit for `wild2` captions.
* [`audit_wild3_results.md`](audit_wild3_results.md): Quality audit for `wild3` captions.
* [`audit_wild4_results.md`](audit_wild4_results.md): Quality audit report for `wild4` captions (100 clips, 0 duplicate clusters).
* [`audit_wild5_results.md`](audit_wild5_results.md): Quality audit report for `wild5` captions (109 clips, 0 duplicate clusters).

## 🛠️ Generating Audit Reports

Audit reports are generated using `scripts/audit_caption_quality.py`:

```bash
python scripts/audit_caption_quality.py --dataset wild4
python scripts/audit_caption_quality.py --dataset wild5
```

By default, the script saves output markdown reports directly to `docs/data/audit_captions/audit_{dataset}_results.md`.
