# Downstream Evidence Retrieval Experiment

## 1. Overview & Purpose
This experiment tests whether temporal semantic reconstruction restores functional downstream utility to videos with missing intervals, evaluated on the **WildQA benchmark** ([Castro et al., COLING 2022](https://aclanthology.org/2022.coling-1.496)).

In standard intrinsic evaluations, naive interpolation baselines (repeating adjacent frames or visual inertia) achieve deceptively high cosine similarity because surrounding frames reside in the same semantic neighborhood. 

The downstream retrieval task introduces a **discriminant probe (the Question)** that was unknown at reconstruction time. For each question \(Q\) with ground-truth evidence interval \([t_{\text{start}}, t_{\text{end}}]\), we evaluate how effectively \(Q\) retrieves the evidence segment from a 60-second video index.

```
                           [Question Q: "What did the farmer adjust?"]
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
     [Condition 1: Oracle]                                            [Condition 2: Masked]
   Full 60s dense captions                                           Evidence [ts, te] blanked out
   Expected: High MRR / Recall@1                                     Expected: Catastrophic drop (near 0)
               │                                                               │
               └───────────────────────────────┬───────────────────────────────┘
                                               ▼
                                  [Condition 3: Baseline Repeat]
                                 Nearest neighbor boundary caption
                                 Expected: Partial match, low specificity
```

---

## 2. Experimental Conditions

| Condition | Index Construction | Role in Scientific Evaluation |
|---|---|---|
| **Text Oracle (Ceiling)** | All 60 unmasked 1-second dense captions | Maximum attainable retrievability given full captions |
| **Text Masked (Floor)** | Evidence interval \([t_{\text{start}}, t_{\text{end}}]\) zeroed out | Measures catastrophic loss when evidence is missing |
| **Text Baseline Repeat** | Gap filled by repeating the nearest known boundary caption | Visual inertia / naive continuity baseline |
| **Visual Oracle** *(SigLIP)* | Unmasked pre-computed video frame vectors | Ground-truth visual retrieval ceiling |
| **Visual Interpolated** *(SigLIP)* | Gap filled by repeating boundary video frame vectors | Non-parametric visual interpolation baseline |

---

## 3. How to Run

### Run on Dev Set (`wild4` captions, default: SigLIP embedder)
```bash
PYTHONPATH=src python3.9 scripts/run_downstream_retrieval.py --split dev
```

### Run on Test Set (`wild5` captions)
```bash
PYTHONPATH=src python3.9 scripts/run_downstream_retrieval.py --split test
```

### Run with Sentence-Transformers Embedder
```bash
PYTHONPATH=src python3.9 scripts/run_downstream_retrieval.py --split dev --embedder all-MiniLM-L6-v2
```

### Quick Dry-Run (First 10 questions)
```bash
PYTHONPATH=src python3.9 scripts/run_downstream_retrieval.py --split dev --max-items 10 --verbose
```

---

## 4. Output Artifacts

Results are automatically saved to `results/downstream_retrieval/`:
- `retrieval_metrics_{split}_{embedder}.csv`: Per-question metrics including `oracle_rank`, `oracle_mrr`, `oracle_r1`, `oracle_r5`, `masked_mrr`, `baseline_repeat_mrr`, `contrast_margin`, and visual metrics.
- `retrieval_summary_{split}_{embedder}.md`: Auto-generated summary report aggregating results by **Domain** (Procedural vs. Stochastic) and **Question Category** (Reasoning vs. Perceptual).
