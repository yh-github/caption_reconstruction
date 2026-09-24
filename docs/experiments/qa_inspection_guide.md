# QA Inspection & Qualitative Evaluation Guide

## 1. Overview & Scientific Motivation

In our downstream evidence retrieval experiments on **WildQA** ([Castro et al., COLING 2022](https://aclanthology.org/2022.coling-1.496)), LLaMA 3.1 8B's temporal caption reconstructions achieved an **MRR of 0.2795** and **Recall@5 of 44.9%**, decisively outperforming the naive frame-repeat baseline (`0.0840`, \(p < 10^{-10}\)) and matching the visual oracle ceiling (`0.2755`).

However, embedding-based retrieval metrics leave two critical questions unaddressed:

1. **Semantic Type vs. Factual Accuracy**:
   Dense embedding models (SigLIP, MPNet) measure semantic and topical proximity. A question like *"What tool is the farmer using?"* (Answer: *"Shovel"*) might achieve high cosine similarity if the reconstruction mentions *"gardening tools"* (semantic type) or *"a trowel"* (plausible hallucination) without getting the exact ground-truth fact right.
2. **Temporal Redundancy & Information Leakage**:
   In continuous 60-second video footage, answer facts might appear outside the annotated evidence interval \([t_{\text{start}}, t_{\text{end}}]\) (e.g., at second 5 or 50). In such cases, a downstream Q&A system might answer the question even under the masked condition, meaning the reconstruction was not strictly necessary.

To rigorously investigate both questions, we provide a unified inspection pipeline to prepare, audit, and benchmark samples for **manual review** and **automated LLM-as-a-judge evaluation**.

---

## 2. The Two Evaluation Dimensions

```
                                  [ Sample Question Q ]
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          [ Task A: Intrinsic Quality ]                 [ Task B: Downstream RAG ]
          Does the in-filled window text               Can a reader answer Q from
          contain the ground-truth fact?               Top-K retrieved candidates?
                     │                                             │
      ┌──────────────┼──────────────┐                 ┌────────────┴────────────┐
      ▼              ▼              ▼                 ▼                         ▼
 [Exact Fact]  [Semantic Type] [Hallucination]   [Condition: Masked]   [Condition: Recon]
                                                      (Acc = 0?)            (Acc = 1?)
                                                              │                 │
                                                              └────────┬────────┘
                                                                       ▼
                                                             Δ Accuracy (Necessity)
```

### Task A: Intrinsic Window Factual Recovery
Evaluates the text generated inside the masked interval \([t_{\text{start}}, t_{\text{end}}]\) compared against unmasked boundary context and human reference answers:
* **EXACT_FACT**: The reconstruction explicitly names the target entity, action, or attribute (e.g., *"blueberries"*).
* **SEMANTIC_TYPE**: The reconstruction identifies the correct general category or activity (e.g., *"picking fruit"*, *"berries"*).
* **PLAUSIBLE_HALLUCINATION**: The reconstruction invents a coherent but factually incorrect detail (e.g., *"strawberries"*).
* **MISSED**: The reconstruction talks about unrelated elements or misses the event entirely.
* **BOUNDARY_LEAKAGE**: Flags whether the answer was already explicitly stated in pre-gap (\(t-3\) to \(t-1\)) or post-gap (\(t+1\) to \(t+3\)) captions.

### Task B: Downstream RAG Answerability & Information Necessity
Evaluates whether a reader LLM can answer Question \(Q\) using only the Top-\(K\) candidates retrieved from the 60-second video index:
* **\(\text{Accuracy}(\text{Masked})\)**: If the reader answers correctly when the evidence interval is zeroed out, the question had outside temporal leakage or was guessable via parametric memory.
* **\(\text{Accuracy}(\text{Reconstructed})\)**: Measures whether the retriever + reconstructed captions succeed.
* **Information Necessity (\(\Delta \text{Accuracy}\))**:
  \[
  \Delta \text{Accuracy} = \text{Accuracy}(\text{Reconstructed}) - \text{Accuracy}(\text{Masked})
  \]
  Isolates the clean causal gain of the temporal reconstruction.

---

## 3. How to Run the Preparation Script

The script [`scripts/prepare_qa_inspection_dataset.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/prepare_qa_inspection_dataset.py) compiles all necessary context (metadata, answers, boundary context, gap conditions, and top-k retrieval candidates) into ready-to-inspect formats.

### Quick Dry-Run (No GPU / No Embedder, first 5 questions):
```bash
PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split dev --embedder none --max-items 5
```

### Full Run with SigLIP Embeddings on Dev Split (85 Clean Questions):
```bash
PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split dev --embedder siglip
```

### Full Run on Test Split (216 Clean Questions):
```bash
PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split test --embedder siglip
```

### Run with MPNet Embedder:
```bash
PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split dev --embedder all-mpnet-base-v2
```

### Command-Line Arguments

| Argument | Choices / Default | Description |
|---|---|---|
| `--split` | `dev`, `test`, `both` (default: `dev`) | WildQA split to process |
| `--embedder` | `siglip`, `all-mpnet-base-v2`, `none` (default: `siglip`) | Embedder for temporal retrieval and top-k candidates |
| `--recon-source` | Path / None (default: auto-detected) | Path to reconstructed JSON files; auto-detects from local results or HF snapshot cache |
| `--top-k` | int (default: `3`) | Number of top retrieved candidate frames to extract |
| `--boundary-window` | int (default: `3`) | Seconds of unmasked pre/post context before and after the gap |
| `--clean-only` | bool (default: `True`) | Restricts to single-interval evidence spans \(< 15\) seconds |
| `--max-items` | int / None | Limits number of items for fast debugging |
| `--output-dir` | Path (default: `results/qa_inspection`) | Output directory |
| `--format` | `all`, `jsonl`, `html`, `markdown` (default: `all`) | Output formats to generate |

---

## 4. Generated Artifacts

Files are saved directly to `results/qa_inspection/`:

1. **`qa_inspection_{split}_{embedder}.jsonl`**:
   - Machine-readable dataset containing full metadata, boundary captions, gap in-fills, retrieval ranks, and Top-K candidates.
   - Ideal for automated batch LLM evaluations.
2. **`qa_inspection_{split}_{embedder}.html`**:
   - Interactive, styled web dashboard viewable in any browser.
   - Displays sample cards with domain tags, highlighted answers, side-by-side context comparisons, collapsible Oracle captions, Top-K candidate tables, and rating radio buttons for human auditors.
3. **`qa_inspection_{split}_{embedder}.md`**:
   - GitHub-flavored markdown summary for terminal viewing or quick commits.

---

## 5. Automated LLM-as-a-Judge Evaluation

Standardized evaluation prompts are maintained in [`prompts/qa_judge_rubric.txt`](file:///home/yoavh/code/antigravity/caption_reconstruction/prompts/qa_judge_rubric.txt).

### Running Automated Audit via Python:
```python
import json
from openai import OpenAI  # or google.genai / local vLLM

client = OpenAI()

with open("results/qa_inspection/qa_inspection_dev_siglip.jsonl", "r") as f:
    records = [json.loads(line) for line in f]

for item in records:
    # Task A: Evaluate Intrinsic Factual Recovery
    prompt = f"""
    Question: {item['question']}
    Ground-Truth Answer: {item['ground_truth_answer']}
    Pre-gap: {[c['caption'] for c in item['boundary_context']['pre_gap']]}
    Post-gap: {[c['caption'] for c in item['boundary_context']['post_gap']]}
    Reconstructed: {[c['caption'] for c in item['gap_conditions']['reconstructed']]}
    
    Categorize as: EXACT_FACT, SEMANTIC_TYPE, PLAUSIBLE_HALLUCINATION, or MISSED.
    Check for boundary leakage (true/false).
    """
    # Send to model and log structured JSON response
```

---

## 6. Unit Testing & Verification

The inspection pipeline is covered by unit tests in [`tests/test_qa_inspection.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/tests/test_qa_inspection.py):
```bash
pytest tests/test_qa_inspection.py tests/test_evidence_retrieval.py
```
This tests:
- Lexical leakage detection logic and stopword filtering.
- Correct top-K extraction and rank indexing.
- Multi-condition gap construction (Oracle, Masked, Baseline, Recon).
- HTML and Markdown export generation.
