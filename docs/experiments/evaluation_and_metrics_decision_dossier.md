# Cross-Modal Evaluation & Metric Redesign: Decision Dossier for AI Agent Review

**Purpose of this Document:**  
This dossier provides a complete, self-contained review of all evaluation metrics, empirical results, statistical tests, and known methodological confounders across the 335-video benchmark (`wild4` + `wild5`). It is written specifically for an incoming peer or subagent tasked with reviewing our current metrics and deciding whether to introduce alternative or enhanced evaluation frameworks.

---

## 1. Executive Summary of the Core Dilemma

In our research into video caption cloze reconstruction, we evaluate two competing paradigms:
1. **Generative Language Model (`Llama-3.1-8B`)**: Observes surrounding unmasked captions $C_{\text{obs}} = \{(t, c_t) \mid t \notin [i, i+W-1]\}$, infers the missing cloze text via zero-shot reasoning, and embeds the output into sentence space via `all-mpnet-base-v2` (768-dim).
2. **Visual Feature Continuity (`MeanClosestVectors` on SigLIP 768-dim)**: Observes unmasked video frame vectors $V_{\text{obs}} = \{v_t \mid t \notin [i, i+W-1]\}$, and linearly interpolates the missing segment between boundary frames.

### The Fundamental Evaluation Conflict:
* **Raw Cosine Similarity is Non-Comparable Across Modalities**: Llama's score lives in the 768-dim MPNet text space; SigLIP's score lives in the 768-dim SigLIP vision space. Equal dimensionality does not make the spaces comparable: they are unrelated geometries with different similarity distributions, so comparing raw dot products or cosine values across them is meaningless.
* **Raw Cosine Similarity is Massively Confounded by Caption Redundancy**: We discovered that Llama's `cos_sim_mean` correlates at Spearman $\rho = +0.606$ ($p = 6.2 \times 10^{-34}$) with textual caption redundancy (`APCS_T`). When a video's ground truth captions are naturally repetitive, *any* generated text receives an artificially inflated cosine similarity.
* **MRR (Mean Reciprocal Rank) is Scale-Free and Directly Comparable**, but has subtle representation-dependent boundaries and step-function discreteness.

---

## 2. Definitive Benchmark Results & Statistical Findings

All findings are computed over the full, clean combined benchmark of **335 videos** (100 in `wild4`, 235 in `wild5`) across gap widths $W \in [1, 2, 3, 4, 6, 8, 12, 16]$ at center cloze position $i=29$, with `pool_scope: "video"` (queries ranked against all other 59 timestamps in the video, pool size $N = 60$). **Chance levels for $N=60$**: MRR $= H_{60}/60 \approx 0.078$, Mean Rank $= 30.5$, Recall@1 $= 1/60 \approx 0.017$, Recall@5 $= 5/60 \approx 0.083$ (see Section 2.1).

### Table 1: Llama-3.1-8B Reconstruction Performance by Gap Width $W$
(Source: [`results/benchmark_wild4_wild5_summary_by_width.csv`](../../results/benchmark_wild4_wild5_summary_by_width.csv))

| Gap Width ($W$) | $N$ Videos | Mean MRR | Median MRR | Mean Rank (of 60) | Recall@1 | Recall@5 | Mean `cos_sim` | Mean `cos_sim_min` | Mean `cos_sim_residual` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$W=1$** | 320 | 0.119 | 0.045 | 23.0 | 3.4% | 16.6% | 0.494 | 0.494 | 0.129 |
| **$W=2$** | 335 | 0.102 | 0.053 | 26.3 | 2.5% | 12.4% | 0.457 | 0.378 | 0.088 |
| **$W=3$** | 235 | 0.088 | 0.054 | 27.6 | 2.0% | 10.2% | 0.435 | 0.314 | 0.060 |
| **$W=4$** | 335 | 0.088 | 0.055 | 27.4 | 1.7% | 10.1% | 0.430 | 0.287 | 0.056 |
| **$W=6$** | 335 | 0.090 | 0.062 | 27.6 | 1.8% | 10.8% | 0.422 | 0.246 | 0.066 |
| **$W=8$** | 335 | 0.088 | 0.065 | 28.0 | 1.9% | 9.7% | 0.411 | 0.211 | 0.062 |
| **$W=12$** | 334 | 0.083 | 0.067 | 29.5 | 1.9% | 8.8% | 0.392 | 0.165 | 0.049 |
| **$W=16$** | 306 | 0.076 | 0.063 | 30.2 | 1.6% | 8.0% | 0.380 | 0.146 | 0.038 |

### Table 2: Paired Direct Comparison on MRR (Llama vs. SigLIP Visual Continuity)
(Source: [`results/method_rank_differences_per_video.csv`](../../results/method_rank_differences_per_video.csv))

| Gap Width ($W$) | $N$ Shared | Mean Llama MRR | Mean SigLIP MRR | Margin ($\text{MRR}_{\text{LLM}} - \text{MRR}_{\text{Vid}}$) | Llama Wins | Video Wins | Llama Win Rate | Paired Wilcoxon $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$W=3$** | 225 | 0.085 | **0.175** | **-0.091** | 44 | 181 | **19.6%** | $p = 4.3 \times 10^{-20}$ |
| **$W=6$** | 323 | 0.090 | **0.129** | **-0.038** | 96 | 227 | **29.7%** | $p = 2.2 \times 10^{-14}$ |
| **$W=12$** | 322 | 0.083 | **0.092** | **-0.009** (median $-0.016$) | 126 | 196 | **39.1%** | $p = 2.1 \times 10^{-5}$ (paired $t$: $p = 0.052$) |

> **Correction (2026-10-04)**: All $p$-values recomputed from the source CSV (`scipy.stats.wilcoxon`, paired). The earlier $W=12$ row (margin $-0.001$, $p = 0.480$, "Statistical Parity") could not be reproduced. The margin $-0.0009$ comes from the 224-video subset that has both $W=3$ and $W=12$ (see H2); on that subset the paired $t$-test gives $p = 0.88$, but Wilcoxon still gives $p = 0.022$ with video winning 131/224. **The means converge, but video still wins on most videos.** Also, at $W=12$ Llama's MRR (0.083) is statistically indistinguishable from chance (0.078) and SigLIP (0.092) is close to it, so the convergence is mainly both methods approaching the chance floor, not Llama catching up (Section 2.1).

### Table 3: Rigorous Statistical Significance Tests on Key Hypotheses

| Hypothesis Tested | Test Applied | Sample Size | Test Statistic | $p$-value | Statistical Conclusion |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **H1: Does widening gap from $W=3$ to $W=12$ increase Llama's win rate?** | Two-proportion $z$-test | $N_3=225, N_{12}=322$ | $z = 4.868$ | **$1.13 \times 10^{-6}$** | **Overwhelmingly Significant**. 19.6% $\to$ 39.1% (+19.6% gain, 95% CI: $[+11.9\%, +26.8\%]$). |
| **H2: Does the paired deficit $\Delta \text{MRR}$ close across time?** | Paired $t$-test / Wilcoxon | $N=224$ paired | $t = 8.404, W = 4452$ | **$5.10 \times 10^{-15}$** | **Overwhelmingly Significant**. Deficit shrinks from $-0.0906$ to $-0.0009$ (on the 224-video subset; $-0.0086$ on all 322 videos at $W=12$). **Caveat**: shrinkage is largely driven by SigLIP falling toward chance, not by Llama improving (Llama's MRR is flat at ≈ chance, Section 2.1). |
| **H3: At $W=12$, does visual dynamism increase Llama's win rate (37% vs 44%)?** | Fisher's Exact & $z$-test | $N_{\text{Q1}}=81, N_{\text{Q4}}=81$ | $z = 0.959, \text{OR}=1.36$ | **$p = 0.337$ (n.s.)** | **NOT Significant**. 95% CI: $[-7.7\%, +22.2\%]$ crosses zero. |
| **H4: At $W=12$, does visual dynamism correlate with continuous $\Delta \text{MRR}$?** | Spearman Rank Corr | $N=322$ | $\rho = -0.063$ | **$p = 0.260$ (n.s.)** | **NOT Significant**. Visual motion does not alter MRR margin. |
| **H5: Does text redundancy (`APCS_T`) correlate with raw `cos_sim_mean`?** | Spearman Rank Corr | $N=335$ ($W=6$) | $\rho = +0.613$ | **$6.9 \times 10^{-36}$** | **Massive Confounder**. `cos_sim_mean` mostly measures caption repetition. |

### 2.1 Chance-Level Reference & Metric Formulations (Updated 2026-10-05)

Under a uniformly random ranking of the target among \(N\) candidates, the expected value of each metric is:

| Metric | Chance formula | \(N = 60\) | \(N = 31\) | \(N = 3\) (Window Pool) |
| :--- | :--- | :---: | :---: | :---: |
| MRR | \(H_N / N = \frac{1}{N}\sum_{r=1}^{N} \frac{1}{r}\) | **0.078** | 0.130 | **0.611** |
| Mean Rank | \((N+1)/2\) | 30.5 | 16.0 | **2.00** |
| Recall@\(k\) | \(k/N\) | R@1 = 0.017, R@5 = 0.083 | R@1 = 0.032, R@5 = 0.161 | R@1 = 0.333, R@5 = 1.000 |
| ROC-AUC \(= (N-r)/(N-1)\) | 0.5 (any \(N\)) | 0.5 | 0.5 | 0.5 |
| Calibrated \(c = 2\,\text{AUC} - 1\) | 0 (any \(N\)) | 0 | 0 | 0 |

> [!WARNING]
> Earlier versions of this dossier and related docs stated MRR chance as \(1/60 \approx 0.0167\). That is the chance level of **Recall@1**, not MRR. MRR chance for \(N=60\) is \(\approx 0.078\), i.e. 4.7× higher.

#### Complete Mathematical Derivation of ROC-AUC and Calibrated AUC:
When evaluating candidate retrieval for a missing slot, we have 1 positive (ground-truth caption) and \(K = N - 1\) negatives (distractor captions in the video pool).
1. **The ROC Curve (Receiver Operating Characteristic)**:
   - Varying the decision threshold \(\tau\) from \(+\infty\) down to \(-\infty\), True Positive Rate (TPR) steps from 0 to 1 when \(\tau\) crosses the target's score.
   - False Positive Rate (FPR) increments by \(\frac{1}{N - 1}\) for every distractor scored higher than the target.
   - For target rank \(r \in [1, N]\), exactly \(r - 1\) distractors are ranked above the target, and \(N - r\) distractors are ranked below it:
     \[ \text{ROC-AUC} = \frac{\text{distractors ranked below target}}{\text{total distractors}} = \frac{N - r}{N - 1} \]
   - This is identically the **Mann-Whitney \(U\) / Wilcoxon rank-sum statistic**: the probability that the ground-truth target is scored higher than a randomly drawn distractor:
     \[ \text{ROC-AUC} = P(\text{Score}_{\text{target}} > \text{Score}_{\text{distractor}}) \]
2. **The Cumulative Recall@\(k\) Step Curve**:
   - If we plot \(\text{Recall}@k\) on the y-axis (which steps from 0 to 1 at \(k = r\)) against normalized candidate list depth \(x = \frac{k - 1}{N - 1} \in [0, 1]\) on the x-axis, the integral of this curve from \(x = 0\) to \(x = 1\) is:
     \[ \int_{0}^{1} \text{Recall}(x) \, dx = 1 - \frac{r - 1}{N - 1} = \frac{N - r}{N - 1} \]
3. **Calibrated AUC (\(c\)) via Somers' \(D\) / Gini**:
   - Raw ROC-AUC gives \(0.5\) for random guessing and \(1.0\) for perfect retrieval.
   - To provide an intuitive, pool-size-invariant scale centered at zero:
     \[ c = 2 \cdot \text{ROC-AUC} - 1 = 1 - \frac{2(r - 1)}{N - 1} \]
     where \(c = +1.0\) (+100%) represents rank 1, \(c = 0.0\) (0%) represents random chance, and \(c = -1.0\) (-100%) represents inverted ranking (rank \(N\)).

#### Why RBP (Rank-Biased Precision) and Full Recall Curves Were Considered:
- **Full Recall@\(k\) Curves**: Plotting \(\text{Recall}@k\) for \(k \in [1, N]\) produces the complete Empirical Cumulative Distribution Function (CDF) of retrieval. AUC summarizes this entire curve without setting arbitrary top-\(k\) thresholds.
- **Rank-Biased Precision (RBP)** (Moffat & Zobel, 2008): In a 1-target retrieval setup, RBP simplifies to:
  \[ \text{RBP}(p) = (1 - p) \cdot p^{r - 1} \]
  where \(p \in (0, 1)\) is the persistence parameter. RBP was evaluated because MRR's harmonic decay (\(1/r\)) has an extreme drop-off between rank 1 (1.0) and rank 2 (0.50), becoming completely flat at ranks \(\ge 10\). RBP allows configuring user tolerance (e.g., \(p=0.8\) vs \(p=0.95\)).
- **Why Calibrated AUC was favored as primary macro metric**: Both MRR and RBP have non-zero, variable chance baselines that depend on \(N\) and \(p\). In contrast, Calibrated AUC maintains an invariant \(0.0\%\) chance baseline across any candidate pool size \(N\).

---

### 2.2 Root Cause of the Wild4 \(W=3\) Anomaly
In early benchmark summaries, an apparent anomaly appeared where Llama's performance jumped dramatically at \(W=3\) (MRR \(\approx 0.558\), Recall@5 \(= 100\%\), Mean Rank \(= 2.235\)).
- **Root Cause**: The early run `wild4_llama_w3_window_v3` was accidentally evaluated under `pool_scope: "window"` (\(N = 3\) masked candidates only) rather than `pool_scope: "video"` (\(N = 60\)).
- **Expected Values under \(N=3\)**:
  - Random chance MRR is \(H_3/3 = \frac{1 + 1/2 + 1/3}{3} = \frac{1.833}{3} \approx 0.611\). Llama's score of \(0.558\) was actually **below random chance**.
  - Random chance Mean Rank is \((3+1)/2 = 2.0\). Llama's mean rank of \(2.235\) was below chance.
  - Recall@5 is mathematically \(1.0\) (\(100\%\)) because there are only 3 candidates in total.
- **Resolution**: Clean evaluation on Wild5 with `pool_scope: "video"` (\(N=60\)) yields the true performance: Mean MRR \(\approx 0.123\), Mean Rank \(= 24.56\), perfectly consistent with the degradation curves.

---

### 2.3 Clarification on `Visual_SigLIP_MeanClosest` (Midpoint LERP)
- The visual baseline `Visual_SigLIP_MeanClosest` is a **boundary midpoint linear interpolation (fixed LERP at \(\alpha = 0.5\))**:
  \[ \hat{v}_t = \frac{v_{\text{before}} + v_{\text{after}}}{2} \]
  where \(v_{\text{before}} = v_{i-1}\) and \(v_{\text{after}} = v_{i+W}\) are the unmasked frame vectors immediately adjacent to the gap boundaries.
- It is **not** a rolling moving average or dynamic time-weighted LERP.
- `Visual_SigLIP_RepeatClosest` simply copies the nearest boundary frame:
  \[ \hat{v}_t = \begin{cases} v_{i-1} & \text{if } t - (i - 1) \le (i + W) - t \\ v_{i+W} & \text{otherwise} \end{cases} \]

---

### 2.4 Horizon Scaling Dynamics: Why Llama's Win Rate Rises at Large \(W\)
In paired comparisons between Llama-3.1-8B and `Visual_SigLIP_MeanClosest`, Llama's win rate rises from \(19.6\%\) at \(W=3\) to \(39.1\%\) at \(W=12\), and the MRR gap shrinks from \(-0.091\) to \(-0.009\).
- **The True Mechanism**: This convergence is **not** because Llama performs better over longer horizons. Llama's MRR stays relatively flat near the chance floor (\(0.119 \to 0.088 \to 0.083 \to 0.076\)).
- Instead, **the visual baseline collapses toward chance** as the gap widens (SigLIP MRR decays from \(0.175 \to 0.129 \to 0.092\)) because visual continuity and frame correlation degrade sharply over 12–16 second intervals.
- The shrinking margin reflects the collapse of visual continuity rather than language model mastery.

---

**Llama-3.1-8B vs. chance** (source: `results/benchmark_wild4_wild5_combined_per_video.csv`, one-sample \(t\)-tests against the chance value, per-video means):

| \(W\) | Mean MRR | \(p\) (MRR vs 0.078) | Mean Rank | \(p\) (Rank vs 30.5) | R@1 | R@5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.119 | \(2 \times 10^{-4}\) | 23.0 | \(5 \times 10^{-16}\) | 3.4% | 16.6% |
| 3 | 0.088 | 0.12 (n.s.) | 27.6 | \(9 \times 10^{-5}\) | 2.0% | 10.2% |
| 6 | 0.090 | 0.002 | 27.6 | \(3 \times 10^{-8}\) | 1.8% | 10.8% |
| 12 | 0.083 | 0.23 (n.s.) | 29.5 | 0.007 | 1.9% | 8.8% |
| 16 | 0.076 | 0.38 (n.s.) | 30.2 | 0.39 (n.s.) | 1.6% | 8.0% |

**Takeaways**:
1. Llama is only marginally above chance for \(W \ge 3\) and **at chance for \(W = 16\)** on every metric.
2. **Mean rank (linear in rank, equivalent to AUC) detects above-chance performance far more reliably than MRR** (e.g. \(W=3\): \(p = 9 \times 10^{-5}\) vs \(p = 0.12\)). MRR's \(1/r\) weighting compresses all non-top-3 ranks into a narrow band near the floor, where most Llama queries live.
3. Near-zero correlations between MRR and the a-priori scores (`experiment_protocol_current.md` §4) must be read with this in mind: when the outcome is mostly noise around chance, correlations are attenuated toward zero regardless of the true relationship.

---

## 3. Comprehensive Critique of Current Metrics

### 1. Mean Reciprocal Rank (`mrr`)
* **How it works**: For each query timestamp $t$ in the gap, rank the true item among all 60 video timestamps based on cosine similarity to the prediction. $\text{MRR} = \frac{1}{|M|} \sum_{t} \frac{1}{\text{rank}_t}$.
* **Why it's our current best**:
  - Scale-free: independent of raw embedding norms.
  - Aligns both modalities into an identical game with the same chance level ($H_{60}/60 \approx 0.078$ for MRR; $1/60 \approx 0.017$ for Recall@1).
  - Empirically near-zero correlation with video motion and caption redundancy (|ρ| ≤ 0.15 across widths). **Caveat**: this is partly because Llama's per-video MRR is close to chance and therefore noisy; low outcome reliability attenuates every correlation, so "uncorrelated" is not yet evidence of "unbiased".
* **Why it falls short / Weaknesses**:
  - **Reciprocal cliff**: Ranks 1, 2, 3 receive weights $1.0, 0.5, 0.333$, but ranks 10 through 60 receive tiny weights ($0.10 \to 0.016$). If a model deduces the action roughly and lands at rank 4, it loses 75% of credit compared to rank 1.
  - **Raw score is not chance-corrected**: an MRR of 0.08 looks like "something" but is chance for $N = 60$, and the chance value changes with pool size. Use $\text{MRR}_{\text{norm}} = (\text{MRR} - H_N/N)/(1 - H_N/N)$ or the calibrated score $c$ for interpretable reporting.
  - **Low power for detecting above-chance or tail differences**: see Section 2.1 and the metric sensitivity analysis (AUC / mean-rank detects V_Repeat vs V_Mean differences at $p = 10^{-8}$ that MRR rates as identical, $p = 0.69$).
  - **Zero temporal tolerance**: If the LLM generates the exact correct action, but the true action occurred at $t=30$ and the candidate pool has a near-identical frame at $t=29$, placing it at $t=29$ is penalized heavily even though it was off by only 1 second.
  - **Asymmetric Distractor Geometry**: As detailed in [`docs/theory/cross_modal_evaluation_metrics.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/theory/cross_modal_evaluation_metrics.md), human caption embeddings have distinct, discrete semantic clusters, whereas SigLIP video frame embeddings form a continuous, smooth visual trajectory. Ranking in a smooth space vs. a discrete cluster space may have unequal difficulty.

### 2. Context Residual Cosine Similarity (`cos_sim_residual_mean`)
* **How it works**: Computes the centroid of all observed frames/captions in the 60s video, projects it out from both prediction and ground truth, and measures the remaining directional cosine similarity:
  $$\hat{v}_{\text{res}} = \hat{v} - \frac{\hat{v} \cdot \bar{v}}{\|\bar{v}\|^2} \bar{v}$$
* **Strengths**: Successfully strips out visual and textual monotony. Unlike raw `cos_sim`, its correlation with `APCS_V` is statistically zero ($\rho = -0.079, p = 0.15$).
* **Weaknesses**: Still cannot be directly subtracted across modalities ($\Delta_{\text{res}} = \text{cos\_sim\_res}_{\text{text}} - \text{cos\_sim\_res}_{\text{vid}}$ is still comparing distinct vector spaces).

### 3. Raw Cosine Similarity (`cos_sim_mean` and `cos_sim_min`)
* **Status**: **Disqualified as a primary cross-modal benchmark metric**.
* **Reason**: Confounded by ground-truth caption repetition (`APCS_T`, $\rho = +0.61$). Highly misleading when comparing text to vision.

---

## 4. Promising Alternatives to MRR & Where to Integrate Them

If an agent or researcher wishes to implement superior or complementary evaluation metrics, here are the 4 strongest candidates:

### Candidate A: Temporal NDCG (Normalized Discounted Cumulative Gain)
* **Mathematical Definition**: Instead of a binary hit at exact timestamp $t^*$, grant graded relevance that decays exponentially with temporal distance $|\Delta t|$:
  $$\text{Relevance}(t, t^*) = \exp\left(-\frac{|t - t^*|}{\tau}\right), \quad \tau \in [1.0, 2.0]$$
  $$\text{DCG} = \sum_{r=1}^{60} \frac{\text{Relevance}(r)}{\log_2(r + 1)}, \quad \text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$
* **Why it's better than MRR**:
  - Smooths out the "reciprocal cliff".
  - Forgives 1-second or 2-second near-misses when a model deduces the sequence correctly but misjudges the exact boundary.
* **Where to Integrate**:
  - Implemented in legacy code in [`scripts/recalculate_metrics.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/recalculate_metrics.py#L25-L69).
  - Can be added directly into [`src/evaluations/eval_vectors.py:calculate_retrieval_metrics`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/eval_vectors.py) alongside MRR and Recall.

### Candidate B: Downstream Evidence Retrieval (Task-Grounded Discriminant Probe)
* **Concept**: Instead of asking whether the predicted representation matches the ground truth in the dark, test whether the reconstructed representation **answers a downstream user query or WildQA Question** ($Q$) that was unknown during reconstruction!
* **How it works**:
  - WildQA contains question-evidence pairs: Question $Q$ has ground-truth evidence interval $[t_{\text{start}}, t_{\text{end}}]$.
  - The interval is masked out and reconstructed by Llama or by Video Baseline.
  - We use Question $Q$ to retrieve from the reconstructed 60s index.
  - Evaluation metric: Does $Q$ retrieve the reconstructed interval with high MRR?
* **Why it's superior to all intrinsic metrics**:
  - Eliminates the representation dependency critique entirely! Both methods are tested on a functional, goal-oriented downstream task: **information recovery**.
* **Where to Integrate**:
  - Fully scaffolded in [`scripts/run_downstream_retrieval.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/run_downstream_retrieval.py) and documented in [`docs/experiments/downstream_retrieval.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/experiments/downstream_retrieval.md).

### Candidate C: Shared Cross-Modal Space Projection (SigLIP Cross-Modal Retrieval)
* **Concept**: Instead of encoding Llama's predicted captions with `all-mpnet-base-v2` (which is text-only), encode Llama's predicted captions using the **SigLIP Text Encoder** (`google/siglip-base-patch16-224`).
* **Why this is a breakthrough**:
  - SigLIP is a dual-encoder contrastive model trained to align visual frames and text descriptions into the **exact same 768-dimensional space**.
  - If Llama's captions are encoded into SigLIP text space, Llama's predictions can be **directly compared against ground-truth video vectors**!
  - We could calculate true cross-modal cosine similarity, cross-modal retrieval, and mutual embedding distance without separate representations.
* **Where to Integrate**:
  - The embedder is already written: [`src/llm/local_embedder.py:SiglipTextEmbedder`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/llm/local_embedder.py#L89-L135).
  - We only need to configure an evaluation block with `embedding_model: "local:siglip"` in an experiment config!

### Candidate D: Soft / Tolerance-Window Recall@K (Windowed MRR)
* **Concept**: A hit is recorded at rank 1 if the predicted representation ranks *any* timestamp in $[t^* - \delta, t^* + \delta]$ at position 1 (e.g., $\delta = 1$s or $2$s).
* **Why it's better**:
  - Accounts for human annotation boundary noise in WildQA (dense captions are often off by $\pm 1$s between crowd workers).
* **Where to Integrate**:
  - Modifying `calculate_retrieval_metrics` in [`src/evaluations/eval_vectors.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/eval_vectors.py).

---

## 5. Directory & File Pointers for Incoming Agents

1. **Master Field Dictionary**: [`docs/experiments/field_dictionary.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/experiments/field_dictionary.md) (All field definitions and formulas).
2. **Current Experiment Protocol & Status**: [`docs/experiments/experiment_protocol_current.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/experiments/experiment_protocol_current.md).
3. **Combined Benchmark Dataset**: [`results/benchmark_wild4_wild5_combined_per_video.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/benchmark_wild4_wild5_combined_per_video.csv) (2,535 rows across 335 videos).
4. **Summary Statistics by Width**: [`results/benchmark_wild4_wild5_summary_by_width.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/benchmark_wild4_wild5_summary_by_width.csv).
5. **Apriori-Posteriori Correlation Matrices**: [`results/benchmark_wild4_wild5_correlations_by_width.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/benchmark_wild4_wild5_correlations_by_width.csv).
6. **Paired Method Rank Differences**: [`results/method_rank_differences_per_video.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/method_rank_differences_per_video.csv).
7. **Downstream Retrieval Runner**: [`scripts/run_downstream_retrieval.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/run_downstream_retrieval.py).
8. **Evaluation Metric Implementations**: [`src/evaluations/eval_vectors.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/eval_vectors.py) and [`src/evaluations/evaluation.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/evaluations/evaluation.py).
