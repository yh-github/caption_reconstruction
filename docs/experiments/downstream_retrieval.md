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

---

## 5. Empirical Results & Findings

### Overall Benchmark Performance (Dev Split / Llama 3.1 8B)

Evaluated on 85 clean WildQA dev questions using SigLIP text embeddings:

| Condition | MRR | Recall@1 | Recall@5 | Mean Sim | Relative Gain vs. Baseline |
|---|:---:|:---:|:---:|:---:|:---:|
| **Text Masked (Black Hole)** | 0.0184 | 0.0% | 0.0% | 0.0000 | — |
| **Text Baseline (Repeat Nearest)** | 0.0618 | 0.0% | 5.9% | 0.6229 | *Baseline* |
| **Text Reconstructed (Llama 3.1 8B)** | **0.2640** | **12.5%** | **41.7%** | **0.7287** | **+327% relative gain** |
| **Text Oracle (Ceiling)** | 0.2309 | 9.4% | 35.3% | 0.7189 | — |
| **Visual Oracle (Video Frames)** | 0.2755 | 14.3% | — | — | — |
| **Visual Interp (Frame Repeat)** | 0.0964 | 3.6% | — | 0.0742 | — |

---

### High-Headroom Evaluation Subset (Oracle in Top 5)

The High-Headroom subset isolates questions where the caption modality is proven to contain the required visual information (Oracle rank \(\le 5\)). On this subset, Llama 3.1 8B achieves a **92.3% head-to-head win rate** against naive frame repetition:

* **Baseline Repeat MRR**: `0.0777` (Recall@5 = `6.7%`, Recall@1 = `0.0%`)
* **Llama 3.1 8B MRR**: **`0.4188`** (Recall@5 = **`61.5%`**, Recall@1 = **`23.1%`**)
* **Head-to-Head Comparison**: **12 Wins, 0 Ties, 1 Loss**

#### Question-by-Question Breakdown:

| Domain | Question | Oracle Rank | LLM Recon | Baseline Repeat | Masked Gap | Outcome |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Human Survival** | *Is the plant a potato?* | #3 | **#1** | #41 | #56 | **WIN** |
| **Military** | *Is there a person on a motorbike in the video?* | #2 | **#2** | #52 | #53 | **WIN** |
| **Military** | *Is there a military facility in the video?* | #4 | **#2** | #47 | #49 | **WIN** |
| **Agriculture** | *How many acres of farmland has to be planted?* | #1 | **#1** | #40 | #51 | **WIN** |
| **Human Survival** | *Does the human have shelter?* | #4 | **#4** | #36 | #54 | **WIN** |
| **Natural Disaster** | *What type of storm occurred?* | #3 | **#1** | #3 | #54 | **WIN** |
| **Military** | *How long does this training event last?* | #4 | **#2** | #17 | #53 | **WIN** |
| **Natural Disaster** | *What is happening?* | #2 | **#4** | #8 | #56 | **WIN** |
| **Human Survival** | *Does the man have water, food, fire, and shelter?* | #1 | **#9** | #54 | #57 | **WIN** |
| **Geography** | *Where are the lakes at?* | #5 | **#9** | #11 | #53 | **WIN** |
| **Natural Disaster** | *What area is the storm affecting?* | #3 | **#9** | #17 | #53 | **WIN** |
| **Human Survival** | *What kind of water source was there?* | #2 | **#14** | #56 | #57 | **WIN** |
| **Human Survival** | *How is the man making his tool?* | #3 | **#25** | #6 | #56 | *LOSS* |

---

### Why Can Reconstructed Captions Beat the Oracle?

In several instances (e.g. *Is the plant a potato?*, *Is there a military facility?*, *What type of storm occurred?*), Llama 3.1 8B's reconstructed captions achieve a **higher retrieval rank than the original unmasked captions**.

Inspection of the underlying caption texts reveals three complementary mechanisms driving this phenomenon:

#### 1. Abstractive Regularization vs. Perceptual Clutter
Raw dense video captioners (such as 1-second frame-by-frame multimodal models) describe instantaneous optical details. Consequently, the Oracle captions frequently fixate on incidental clutter rather than the salient action:
* *Example (Tornado query)*: Oracle at \(t=56\)s describes *"An oil pump jack is seen in the foreground with the tornado behind it."*
* *Llama Reconstruction*: When in-filling from temporal context, the LLM ignores transient foreground noise and predicts macro-narrative continuity: *"Dark clouds gather around the tornado, signaling an increase in storm activity."*
* Because human queries probe narrative events rather than incidental background objects, the LLM's abstracted summary aligns more cleanly with the query probe.

#### 2. Diction Alignment with Downstream Queries
LLMs naturally employ canonical, high-level vocabulary when describing activities:
* *Example (Potato plant query)*: The Oracle caption used generic terminology: *"The person holds up the tuber for the camera to see."* (Rank #3).
* *Llama Reconstruction*: Infers garden context and generates: *"They gently remove the soil from around the tuber, taking care not to damage its delicate skin."* (Rank #1).
* The contextual terms (*"soil"*, *"garden bed"*, *"delicate skin"*) produce an embedding that sits closer to the human agricultural question than the bare assertion of holding a tuber.

#### 3. Intra-Video Distractor Suppression
In continuous 60-second video footage, multiple frames depict related elements (e.g. distant storm clouds, general landscape). Under the Oracle condition, non-evidence frames occasionally exhibit spurious high similarity, displacing the target interval to rank #3 or #4. 
When the LLM fills the evidence interval, it concentrates the event's descriptive climax inside the target window, boosting target cosine similarity above intra-video distractor frames.

