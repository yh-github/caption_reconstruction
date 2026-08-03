# Configuration Reference Guide

Experiments in this project are configured using YAML files located in the [`config/`](file:///home/yoavh/code/antigravity/caption_reconstruction/config) directory. This document provides a complete reference for all supported configuration options.

---

## 1. Top-Level Structure

An experiment configuration YAML file consists of five main sections:

```yaml
base_params:
  master_seed: 2025
  experiment_type: "recon" # or "RECON_VECTORS"

evaluation:
  type: "emb_sim"
  embedding_model: "gemini" # or "local:all-mpnet-base-v2"

data_config:
  name: "wild_captions"
  path: "datasets/wildQA/captions__wild2/"
  limit: 100

recon_strategy:
  - name: "pro_d_one_shot_v1__t=1"
    type: "llm"
    llm:
      model_name: "gemini-2.5-pro"
      thought_budget: -1
      temperature: 1
      prompt_template: "prompts/dense_one_shot_v1.txt"
  - name: "baseline_repeat_last"
    type: "baseline_repeat_last"

IMPORT masking_configs: masking_cont.yaml
```

---

## 2. Base Parameters (`base_params`)

* **`master_seed`** (`int`): Global master seed for random number generation to ensure experiment reproducibility across masking and LLM sampling.
* **`experiment_type`** (`str`): Pipeline execution mode.
  * `"recon"`: Indirect text reconstruction pipeline (text captions -> LLM -> embedding evaluation).
  * `"RECON_VECTORS"`: Direct vector reconstruction pipeline (vector matrices -> vector interpolation).

---

## 3. Data Loader Configuration (`data_config`)

Specifies the dataset source and loading behavior:

* **`name`** (`str`): Type of data loader to instantiate.
  * `"wild_captions"`: Loads caption datasets (JSON format with video clips and timestamps).
  * `"np_files"`: Loads pre-computed NumPy embedding matrices (`.npy` files).
  * `"toy_vectors"`: Generates synthetic vector matrices for testing and debugging.
* **`path`** (`str`): Path to the dataset directory (e.g. `datasets/wildQA/captions__wild2/` or `local/wild_videos_embs/`).
* **`limit`** (`int`, optional): Maximum number of video instances/matrices to process.
* **`embedding_model`** (`str`, optional): Embedding model used when converting text to vectors during loading. Defaults to `"gemini"`. Supports local models via `local:<hf_model_id>` (e.g., `local:all-mpnet-base-v2`, `local:all-MiniLM-L6-v2`).

---

## 4. Masking Configurations (`masking_configs`)

Masking strategies define which clips or vector rows are hidden/masked during experiment execution. Multiple configurations can be specified as a list, or imported from an external YAML file using `IMPORT masking_configs: <filename.yaml>`.

### Available Masking Schemes:

#### **A. `contiguous`**
Masks a single contiguous block of clips per video.
```yaml
masking_configs:
  - scheme: "contiguous"
    seed: [11, 22] # Seed offset(s) for start index selection
    width: [6, 9, 12, 15, 18, 21, 24] # Block width(s)
```

#### **B. `fixed_fill`**
Symmetrically expands a mask centered around a fixed starting clip index.
```yaml
masking_configs:
  - scheme: "fixed_fill"
    start_ind: [0, 5]
    width: [3, 5, 10]
```

#### **C. `random`**
Masks a random fraction of clips based on a ratio.
```yaml
masking_configs:
  - scheme: "random"
    ratio: [0.1, 0.25, 0.5]
    seed: 42
```

#### **D. `partition`**
Divides the video sequence into equal partitions and masks a block of partitions.
```yaml
masking_configs:
  - scheme: "partition"
    num_partitions: 10
    start_partition: 2
    num_parts_to_mask: 3
```

---

## 5. Reconstruction Strategies (`recon_strategy`)

Defines the algorithm or model used to fill the masked segments.

### **Text/Caption Reconstruction Strategies (Indirect Pipeline)**

#### **`llm` (Remote LLM API)**
Calls remote LLM APIs (e.g. Google Gemini) to reconstruct missing text captions.
```yaml
recon_strategy:
  - name: "gemini_pro_one_shot"
    type: "llm"
    llm:
      model_name: "gemini-2.5-pro" # or "gemini-2.5-flash"
      thought_budget: -1 # Thinking token budget (-1 for default)
      temperature: 1.0
      prompt_template: "prompts/dense_one_shot_v1.txt"
```

#### **`local_llm` (Local SLM / LLM)**
Runs local small language models (e.g., Microsoft Phi-3) using iterative cloze prompting.
```yaml
recon_strategy:
  - name: "phi-3__t=0.1_rp=1.2"
    type: "local_llm"
    model_key: "phi-3"
    prompt_dir: "iterative_cloze"
    temperature: 0.1
    repetition_penalty: 1.2
    max_new_tokens: 60
```

#### **`baseline_repeat_last` (Heuristic Baseline)**
Repeats the last valid unmasked caption to fill missing slots.
```yaml
recon_strategy:
  - name: "baseline_repeat_last"
    type: "baseline_repeat_last"
```

---

### **Vector/Embedding Reconstruction Strategies (Direct Pipeline)**

#### **`mean_closest`**
Reconstructs missing vectors by taking the component-wise mean of the nearest available vectors before and after the gap.
```yaml
recon_strategy:
  - type: "mean_closest"
```

#### **`repeat_closest`**
Reconstructs missing vectors by copying the nearest available vector in time index.
```yaml
recon_strategy:
  - type: "repeat_closest"
```

---

## 6. Evaluation Configuration (`evaluation`)

Specifies how reconstructed content is evaluated against ground truth.

```yaml
evaluation:
  type: "emb_sim" # Evaluation metric type
  embedding_model: "gemini" # Embedding model for text-to-vector conversion
```

### Supported Evaluation Types:

1. **`emb_sim` (Embedding Cosine Similarity)**:
   * Computes elementwise cosine similarity (`cos_sim`) between reconstructed vectors and ground truth vectors.
   * Computes context-projected residual cosine similarity (`cos_sim_residual`) by projecting out unmasked context vectors to isolate new semantic information.
2. **`emb_retrieval` / `retrieval` (Ranking & Retrieval Metrics)**:
   * Evaluates reconstruction as a retrieval task against all ground-truth clips in the video pool.
   * Computes `mean_rank`, `mrr` (Mean Reciprocal Rank), `recall_at_1`, and `recall_at_5`.
3. **`bert_score` (Text Semantic Metric)**:
   * Uses BERTScore (`microsoft/deberta-large-mnli`) to evaluate textual similarity, returning precision (`bs_p`), recall (`bs_r`), and F1 (`bs_f1`).
4. **`nop` (No Operation)**:
   * Bypasses evaluation steps.
