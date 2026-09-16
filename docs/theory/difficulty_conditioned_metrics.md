# Difficulty-Conditioned Evaluation Metrics for Caption Reconstruction

This document establishes the theoretical foundation, mathematical formulation, and empirical protocol for **Difficulty-Conditioned Evaluation** in dense video caption reconstruction. 

---

## 1. Motivation: The Failure of Unconditioned Similarity

In dense video captioning and reconstruction tasks across masked temporal windows \(M \subset \{1, \dots, T\}\), standard evaluation protocols compute average cosine similarity or lexical overlap between model predictions \(\hat{\mathbf{e}}_t\) and ground truth \(\mathbf{e}_t\):

\[
\overline{\text{cos\_sim}} = \frac{1}{|M|} \sum_{t \in M} \cos(\hat{\mathbf{e}}_t, \mathbf{e}_t)
\]

While superficially straightforward, unconditioned averaging introduces severe statistical distortions:

1. **Inactivity & Static Scene Bias**: In many real-world video segments (quiescent camera, steady shot, unchanging background), adjacent 1-second ground truth captions are near-identical. A trivial non-parametric baseline that simply repeats the nearest unmasked boundary caption (\(\mathbf{e}_b\)) achieves an artificially high cosine similarity (often \(> 0.85\)) without performing any reasoning.
2. **Penalization of Generative Exploration**: When an LLM/SLM generates natural language descriptions, lexical richness and synonym substitution inherently introduce slight semantic variance, depressing raw cosine similarity relative to literal verbatim copying.
3. **Masking of True Semantic Competence**: Aggregating across static and dynamic scenes equally dilutes the signal. A model could score poorly on static scenes (by rephrasing an already known state) while excelling at predicting complex, unobserved causal actions (where the boundary heuristic fails completely).

To resolve this, we require a **difficulty-conditioned framework** that explicitly normalizes or stratifies performance based on how much narrative change actually occurred within the masked gap.

---

## 2. Mathematical Formulation

### 2.1 Nearest Boundary Reference (\(b^*(t)\))

For each masked interval \(t \in M\) within a video of \(T\) clips, the observable context is \(O = \{1, \dots, T\} \setminus M\). 
We define the **Nearest Observable Boundary** \(b^*(t)\) as:

\[
b^*(t) = \arg\min_{b \in O} |t - b|
\]

**Tie-Breaking Rule**: When a masked interval is equidistant from an earlier and a later boundary (e.g., the center of a 3-second gap where \(t - b_{\text{left}} = b_{\text{right}} - t\)), ties are strictly broken in favor of the **earlier (left) boundary** (\(b_{\text{left}}\)). This models causal forward-prediction in time.

---

### 2.2 Narrative Gap Difficulty (\(D_{\text{text}}\))

The difficulty of reconstructing interval \(t\) is defined as the semantic distance between the ground truth state \(\mathbf{e}_t\) and what is already trivially knowable from the nearest boundary \(\mathbf{e}_{b^*(t)}\):

\[
D_{\text{text}}(t) = 1 - \cos(\mathbf{e}_{b^*(t)}, \mathbf{e}_t)
\]

where \(\mathbf{e} \in \mathbb{R}^d\) denotes the normalized sentence embedding (e.g., extracted via `all-mpnet-base-v2`).

- **\(D_{\text{text}} \approx 0\) (Quiescent / Easy)**: The ground truth caption is semantically identical to the boundary. Repeating the boundary is near-optimal.
- **\(D_{\text{text}} \to 1\) (Dynamic / Hard)**: A major state change, action, or scene transition occurred. The boundary provides little to no information about interval \(t\).

---

### 2.3 Model Reconstruction Advantage (\(\Delta\))

For any reconstruction model \(m\) predicting representation \(\hat{\mathbf{e}}_t^{(m)}\), its **advantage over the boundary baseline** at interval \(t\) is:

\[
\Delta^{(m)}(t) = \cos(\hat{\mathbf{e}}_t^{(m)}, \mathbf{e}_t) - \cos(\mathbf{e}_{b^*(t)}, \mathbf{e}_t)
\]

- \(\Delta^{(m)}(t) > 0\): The model successfully recovered information beyond what the boundary contained.
- \(\Delta^{(m)}(t) \le 0\): The model performed worse than simply copying the nearest unmasked clip.

---

### 2.4 Reconstruction Gain over Boundary (RGB)

Analogous to the **Brier Skill Score** in forecasting or \(R^2\) in regression, the **Reconstruction Gain over Boundary (RGB)** normalizes advantage by available headroom:

\[
\text{RGB}^{(m)}(t) = \frac{\cos(\hat{\mathbf{e}}_t^{(m)}, \mathbf{e}_t) - \cos(\mathbf{e}_{b^*(t)}, \mathbf{e}_t)}{1 - \cos(\mathbf{e}_{b^*(t)}, \mathbf{e}_t)} = \frac{\Delta^{(m)}(t)}{D_{\text{text}}(t)}
\]

#### Properties of RGB:
* **Ground Truth Ceiling**: \(\text{RGB}_{\text{GT}} = \frac{1.0 - \cos_b}{1.0 - \cos_b} \equiv 1.0\).
* **Reference Baseline**: \(\text{RGB}_{\text{Boundary}} = \frac{\cos_b - \cos_b}{1 - \cos_b} \equiv 0.0\).
* **Relative Scale**: Positive values represent the percentage of the missing semantic gap recovered by the model.

> [!WARNING]
> **Division Instability**: As \(D_{\text{text}}(t) \to 0\) (clips where boundary is near-identical to ground truth), the denominator vanishes, causing extreme sensitivity or numerical explosion. For this reason, RGB should be thresholded (\(D_{\text{text}} > \epsilon\)), or better, aggregated using the two robust methods below.

---

## 3. Robust Aggregation Methods

To avoid division by near-zero while rigorously characterizing model behavior, two primary formulations are recommended:

### Method A: Difficulty Quartile Stratification (Recommended)

Partition all evaluation clips into four quartiles based on \(D_{\text{text}}\):
* **\(Q_1\) (Easy / Static)**: Lower 25% of \(D_{\text{text}}\) (scenes with minimal narrative movement).
* **\(Q_2\) (Mild Progression)**: 25th to 50th percentile.
* **\(Q_3\) (Moderate Progression)**: 50th to 75th percentile.
* **\(Q_4\) (Hard / Dynamic Transitions)**: Upper 25% of \(D_{\text{text}}\) (abrupt state changes, complex actions).

For each quartile, compute:
1. Mean boundary cosine similarity.
2. Mean model cosine similarity.
3. Win rate (\(\% \text{ of clips where } \Delta > 0\)).
4. Paired Student's \(t\)-test and Cohen's \(d\).

### Method B: Difficulty-Weighted Mean Advantage

Weight each clip's advantage proportionally to its difficulty, guaranteeing that quiescent clips carry minimal weight while high-surprisal events dominate the aggregate score:

\[
\bar{\Delta}_{\text{weighted}}^{(m)} = \frac{\sum_{t} D_{\text{text}}(t) \cdot \Delta^{(m)}(t)}{\sum_{t} D_{\text{text}}(t)}
\]

This eliminates all division singularities while penalizing models that fail on genuine state changes.

### Method C: Advantage Scaling Correlation

Compute the Pearson correlation between task difficulty and model advantage:

\[
r_{\text{scaling}} = \text{Corr}(D_{\text{text}}(t), \Delta^{(m)}(t))
\]

A strong positive correlation demonstrates that the model's value proposition scales with task complexity.

---

## 4. The Modality Decoupling Law: Why Text Vectors for Difficulty?

A critical methodological question is whether difficulty should be measured in **visual vector space** (video frame embeddings) or **textual vector space** (caption embeddings).

Empirical analysis across 879 clips from the `wild4` benchmark confirms that the two modalities measure fundamentally orthogonal phenomena:

### 4.1 Empirical Orthogonality
\[
\text{Corr}(D_{\text{text}}, D_{\text{video}}) = r = 0.057 \quad (p = 0.086, \text{statistically orthogonal})
\]

- **High Video Difficulty, Low Text Difficulty**: Shaky, moving, or panning camera across a static landscape. Optical motion is high, but the narrative state is invariant.
- **Low Video Difficulty, High Text Difficulty**: A fixed tripod recording a subtle, pivotal action (e.g., an instructor silently cutting a wire or placing a key object). Optical motion is near zero, but narrative state change is profound.

### 4.2 Cross-Modal Predictive Power

| Reconstructed Pathway | Correlation with \(D_{\text{text}}\) (Narrative Change) | Correlation with \(D_{\text{video}}\) (Optical Dynamics) |
|---|---|---|
| **Llama-3.1-8B Advantage** (\(\Delta_{\text{LLM}}\)) | **\(r = +0.588\) (\(p = 5.2 \times 10^{-83}\))** | \(r = -0.039\) (\(p = 0.245\), not significant) |
| **Visual Frame Interpolation** (\(\Delta_{\text{vis}}\)) | \(r = +0.058\) (\(p = 0.086\), not significant) | **\(r = +0.339\) (\(p = 4.7 \times 10^{-25}\))** |

### 4.3 Theoretical Principle: "To Each Their Own"
- **Language Models** reason over **symbolic narrative causality**. Their evaluation difficulty **must be indexed by text vectors** (\(D_{\text{text}}\)).
- **Visual Interpolation Models** smooth continuous optical representations. Their evaluation difficulty **must be indexed by video vectors** (\(D_{\text{video}}\)).

Using video vectors to assess language model difficulty conflates camera shake with narrative unpredictability.

---

## 5. Benchmark Reference Values (`wild4`, \(W=3\), \(N=879\) clips)

The table below provides reference metrics on the 100-video `wild4` benchmark across all 3 positions (Start, Mid, End) at gap width \(W=3\):

| Quartile | Mean \(D_{\text{text}}\) | Nearest Boundary | Visual Baseline (Mean-Closest) | Llama-3.1-8B (Whole-Window) | Dominant Method |
|---|---|---|---|---|---|
| **Overall** | 0.492 | 0.503 | **0.528** | 0.424 | Visual Baseline |
| **\(Q_1\) (Easy / Static)** | 0.262 | **0.738** | 0.736 | 0.511 | Nearest Boundary |
| **\(Q_2\) (Mild Change)** | 0.419 | 0.581 | **0.596** | 0.453 | Visual Baseline |
| **\(Q_3\) (Moderate)** | 0.557 | 0.443 | **0.479** | 0.404 | Visual Baseline |
| **\(Q_4\) (Hard / Dynamic)** | 0.754 | 0.246 | 0.300 | **0.330** | **Llama-3.1-8B** (\(p < 10^{-6}\)) |

### Key Scientific Insights
1. **Quiescent Dominance of Non-Parametric Heuristics**: In \(Q_1\), simple boundary repetition achieves a cosine similarity of `0.738`, whereas the LLM achieves only `0.511`. For static scenes, LLM inference is wasteful and introduces unnecessary variation.
2. **Crossover into Symbolic Advantage**: In \(Q_4\) (scenes where the boundary score plummets to `0.246`), Llama-3.1-8B achieves the highest score (`0.330`), beating both the boundary heuristic (win rate = 65.0%) and visual interpolation.
3. **Linear Advantage Scaling**: The LLM advantage \(\Delta\) scales linearly with difficulty at \(r = 0.588\), establishing that the utility of generative models in multimodal architectures is strictly conditional on narrative surprisal.

---

## 6. Python Implementation Recipe

The following standalone function implements the complete difficulty-conditioned evaluation protocol:

```python
import numpy as np
import pandas as pd
from scipy import stats

def compute_difficulty_conditioned_metrics(
    predictions: dict[int, np.ndarray],       # {clip_idx: embedding_vector}
    ground_truth: dict[int, np.ndarray],      # {clip_idx: embedding_vector}
    all_video_indices: list[int],             # All valid clip indices in video (e.g. 0..59)
    gap_indices: list[int],                   # Masked indices (e.g. [29, 30, 31])
) -> list[dict]:
    \"\"\"
    Computes difficulty, advantage, and RGB for each clip in the masked gap.
    Tie-breaker for nearest boundary strictly chooses the earlier (left) clip.
    \"\"\"
    unmasked = sorted(list(set(all_video_indices) - set(gap_indices)))
    records = []
    
    for idx in sorted(gap_indices):
        if idx not in ground_truth or idx not in predictions:
            continue
            
        gt_vec = ground_truth[idx]
        pred_vec = predictions[idx]
        
        # Ensure L2-normalized
        gt_vec = gt_vec / (np.linalg.norm(gt_vec) + 1e-9)
        pred_vec = pred_vec / (np.linalg.norm(pred_vec) + 1e-9)
        
        # Locate nearest unmasked boundary (ties broken by left/earlier)
        left_candidates = [b for b in unmasked if b < idx]
        right_candidates = [b for b in unmasked if b > idx]
        
        b_left = max(left_candidates) if left_candidates else None
        b_right = min(right_candidates) if right_candidates else None
        
        if b_left is not None and b_right is not None:
            dist_l = idx - b_left
            dist_r = b_right - idx
            b_idx = b_left if dist_l <= dist_r else b_right
        elif b_left is not None:
            b_idx = b_left
        elif b_right is not None:
            b_idx = b_right
        else:
            continue
            
        b_vec = ground_truth[b_idx]
        b_vec = b_vec / (np.linalg.norm(b_vec) + 1e-9)
        
        # Similarities
        b_cos = float(np.dot(b_vec, gt_vec))
        pred_cos = float(np.dot(pred_vec, gt_vec))
        
        difficulty = 1.0 - b_cos
        advantage = pred_cos - b_cos
        rgb = advantage / difficulty if difficulty > 0.01 else 0.0
        
        records.append({
            'clip_idx': idx,
            'boundary_idx': b_idx,
            'difficulty': difficulty,
            'boundary_cos': b_cos,
            'model_cos': pred_cos,
            'advantage': advantage,
            'rgb': rgb,
            'win': advantage > 0
        })
        
    return records
```
