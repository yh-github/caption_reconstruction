# Prior-to-Masking (A-Priori) Scores: Theoretical & Empirical Reference

This document provides a comprehensive reference for all **prior-to-masking (a-priori)** metrics implemented in this project. 

These scores quantify the intrinsic properties of a video clip—both visually and textually—**before any segment is masked, reconstructed, or evaluated**. They serve as independent predictors to test the core research question (originating from Gail et al.): *"When does video need language, and can we predict reconstruction difficulty a-priori?"*

---

## 1. Motivation & Conceptual Framework

When evaluating video caption cloze reconstruction, posterior evaluation metrics (such as MRR or Cosine Similarity) measure how well a model inferred the missing content. However, to understand *why* certain videos are easier for visual interpolation while others demand language modeling, we require **a-priori descriptors** that characterize the clip beforehand.

```
                           [Raw Video + Dense Captions (60s)]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Visual Stream (SigLIP 768d)]                   [Text Stream (MPNet 384d)]
                  │                                               │
      ┌───────────┴───────────┐                       ┌───────────┴───────────┐
      ▼                       ▼                       ▼                       ▼
[Sequential Dynamism]    [All-Pairs APCS_V]      [Sequential Dynamism]   [All-Pairs APCS_T]
(Local frame changes)    (Global visual monotony)(Local narrative pace)  (Global text redundancy)
```

We categorize these scores along two axes:
1. **Modality**: **Visual** (computed from frame embeddings) vs. **Textual** (computed from ground-truth caption embeddings).
2. **Temporal Scope**: **Sequential / Step-by-Step** (consecutive transitions $t \leftrightarrow t+1$) vs. **Global / All-Pairs** (all pairs across the 60-second clip).

---

## 2. Mathematical Definitions & Formulas

Let $T$ be the total duration of the clip in seconds (typically $T=60$).

### A. Visual A-Priori Metrics (Embedding Space: Google SigLIP 768-dim)
Extracted at 1 FPS using `google/siglip-base-patch16-224`, yielding frame vectors $v_1, v_2, \dots, v_T \in \mathbb{R}^{768}$.

#### 1. Gail's Visual APCS (`APCS_V`)
* **Concept**: Average Pairwise Cosine Similarity across all frame pairs in the video.
* **Formula**:
  \[
  \text{APCS}_V = \frac{2}{T(T-1)} \sum_{1 \le j < k \le T} \cos(v_j, v_k) = \frac{2}{T(T-1)} \sum_{j < k} \frac{v_j \cdot v_k}{\|v_j\| \|v_k\|}
  \]
* **Interpretation**: Bounded in $[-1, 1]$ (empirically $[0.59, 0.99]$). Higher values indicate **static, visually monotonous scenes** (e.g. stationary camera, minimal movement). Lower values indicate significant visual scene variation across the clip.

#### 2. Average Visual Dynamism (`average_dynamism`)
* **Concept**: Mean cosine distance between consecutive 1-second video frames.
* **Formula**:
  \[
  \text{average\_dynamism} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( 1 - \cos(v_t, v_{t+1}) \right)
  \]
* **Interpretation**: Measures the average velocity of visual change / camera movement per second.

#### 3. Peak Visual Dynamism (`peak_dynamism`)
* **Concept**: 95th percentile of consecutive frame cosine distances.
* **Formula**:
  \[
  \text{peak\_dynamism} = \text{Percentile}_{95} \left\{ 1 - \cos(v_t, v_{t+1}) \mid t \in \{1, \dots, T-1\} \right\}
  \]
* **Interpretation**: Filters out continuous camera jitter to capture **abrupt scene cuts or sudden drastic actions**.

#### 4. Combined Visual Dynamism (`combined_dynamism`)
* **Concept**: Composite visual dynamism score weighting continuous motion and peak transitions equally.
* **Formula**:
  \[
  \text{combined\_dynamism} = 100 \times \left[ 0.5 \times \text{average\_dynamism} + 0.5 \times \text{peak\_dynamism} \right]
  \]
* **Interpretation**: Standardized scale (empirically $1.6$ to $32.3$). Higher = more visually dynamic / volatile video.

---

### B. Textual A-Priori Metrics (Embedding Space: `all-mpnet-base-v2` 384-dim)
Computed by embedding the ground-truth dense captions $c_1, c_2, \dots, c_T$ using `sentence-transformers/all-mpnet-base-v2`, yielding text vectors $e_1, e_2, \dots, e_T \in \mathbb{R}^{384}$.

#### 1. Textual APCS (`APCS_T`)
* **Concept**: Average Pairwise Cosine Similarity across all ground-truth caption pairs in the video.
* **Formula**:
  \[
  \text{APCS}_T = \frac{2}{T(T-1)} \sum_{1 \le j < k \le T} \cos(e_j, e_k)
  \]
* **Interpretation**: Bounded in $[-1, 1]$ (empirically $[0.20, 0.74]$). Measures **lexical and semantic redundancy in the narrative**. High `APCS_T` means the caption transcript repeatedly describes the same state (*"Man sits by fire"*, *"Man watches fire"*).

#### 2. Average Textual Dynamism (`text_average_dynamism`)
* **Concept**: Mean cosine distance between consecutive 1-second caption embeddings.
* **Formula**:
  \[
  \text{text\_average\_dynamism} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( 1 - \cos(e_t, e_{t+1}) \right)
  \]

#### 3. Peak Textual Dynamism (`text_peak_dynamism`)
* **Concept**: 95th percentile of consecutive caption cosine distances.
* **Formula**:
  \[
  \text{text\_peak\_dynamism} = \text{Percentile}_{95} \left\{ 1 - \cos(e_t, e_{t+1}) \mid t \in \{1, \dots, T-1\} \right\}
  \]

#### 4. Combined Textual Dynamism (`text_combined_dynamism`)
* **Concept**: Composite score of textual transition intensity.
* **Formula**:
  \[
  \text{text\_combined\_dynamism} = 100 \times \left[ 0.5 \times \text{text\_average\_dynamism} + 0.5 \times \text{text\_peak\_dynamism} \right]
  \]

---

### C. Linguistic Surprisal Metrics (Language Model Prior)
Estimated sequence log-likelihood from a causal language model:
* **`apcs_nll`**: Negative Log-Likelihood of the caption sequence. Measures information-theoretic rarity.
* **`caption_perplexity`**: $\exp(\text{NLL})$.

---

## 3. Empirical Distribution Across the Full Benchmark (N=335)

Summary statistics computed across all 335 videos in `results/apriori_full_scores.csv`:

| Metric | Modality | Mean | Std Dev | Min | 25% | Median | 75% | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`APCS_V`** | Vision (SigLIP) | **0.821** | 0.065 | 0.589 | 0.778 | 0.822 | 0.867 | 0.985 |
| **`average_dynamism`** | Vision (SigLIP) | 0.063 | 0.029 | 0.010 | 0.042 | 0.062 | 0.078 | 0.204 |
| **`peak_dynamism`** | Vision (SigLIP) | 0.179 | 0.090 | 0.022 | 0.109 | 0.168 | 0.233 | 0.491 |
| **`combined_dynamism`** | Vision (SigLIP) | **12.10** | 5.73 | 1.62 | 7.49 | 11.61 | 15.53 | 32.35 |
| **`APCS_T`** | Text (MPNet) | **0.397** | 0.097 | 0.195 | 0.325 | 0.394 | 0.462 | 0.737 |
| **`text_average_dynamism`**| Text (MPNet) | 0.509 | 0.099 | 0.052 | 0.458 | 0.511 | 0.568 | 0.768 |
| **`text_peak_dynamism`** | Text (MPNet) | 0.763 | 0.106 | 0.355 | 0.696 | 0.774 | 0.838 | 1.009 |
| **`text_combined_dynamism`**| Text (MPNet)| **63.59** | 9.69 | 23.83 | 57.73 | 63.94 | 70.86 | 88.26 |

---

## 4. Cross-Metric Correlation Matrix (How Priors Relate to Each Other)

Spearman rank correlation matrix ($\rho$) across all 335 videos:

| Metric | `combined_dyn` | `APCS_V` | `text_combined_dyn` | `APCS_T` | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`combined_dynamism`** | **1.000** | **-0.824** | +0.087 | **-0.410** | Strong inverse relationship with visual monotony; weak correlation with text dynamism. |
| **`APCS_V`** | **-0.824** | **1.000** | -0.111 | **+0.522** | **Moderate cross-modal alignment**: Visually static videos tend to have repetitive captions ($\rho = +0.522$). |
| **`text_combined_dynamism`**| +0.087 | -0.111 | **1.000** | **-0.718** | Strong inverse relationship with caption redundancy; essentially decoupled from visual step motion ($\rho = 0.087$). |
| **`APCS_T`** | **-0.410** | **+0.522** | **-0.718** | **1.000** | Key confounder for raw text cosine similarity. |

### Critical Observations:
1. **$\text{APCS}_V$ vs. $\text{APCS}_T$ ($\rho = +0.522$)**: Visual monotony and textual monotony are moderately aligned. When visual scenes don't change, annotators use similar words. However, the correlation is not 1.0—there are many videos where visual frames shift constantly while the high-level action remains static, and vice-versa.
2. **Visual Dynamism vs. Textual Dynamism ($\rho = +0.087$)**: Step-by-step consecutive frame motion is **completely decoupled** from consecutive caption distance. A fast-moving camera does not imply narrative topic changes.

---

## 5. Provenance & Code Implementations

* **Calculation Script for Visual Scores**: [`scripts/generate_apriori_scores.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/generate_apriori_scores.py)
* **Calculation Script for Textual Scores**: [`scripts/calc_text_apriori.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/calc_text_apriori.py)
* **Merging Script**: [`scripts/merge_apriori_scores.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/merge_apriori_scores.py)
* **Core Data Model & Logic**: [`src/data/video_surprisal.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/data/video_surprisal.py)
* **Authoritative Pre-computed CSV**: [`results/apriori_full_scores.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/apriori_full_scores.csv) (335 rows, 13 columns)
