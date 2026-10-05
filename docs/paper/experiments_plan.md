# Experimental Plan & Protocol: Testing Temporal In-Filling

## 1. Central Research Question
> **When a temporal interval of video is omitted, does an LLM reading surrounding captions recover the missing content better than assuming persistence, and is that advantage larger in procedural domains?**

---

## 2. The Three Core Hypotheses & Experimental Designs

### Hypothesis 1: Beyond Persistence
* **Hypothesis**: An open-weight LLM (Llama 3.1 8B) reading boundary captions will infer missing state changes, outperforming trivial persistence baselines (`Caption_RepeatClosest` and `Caption_MeanClosest` / Text LERP) in shared SigLIP text embedding space.
* **Metric**: Direct target cosine similarity and candidate retrieval rank (out of 60 seconds).
* **Control**: Within-domain and cross-domain random caption controls.
* **Empirical Status**: **Tested & Rejected**. Persistence beats Llama 3.1 8B across all 6 domains (Military: \(0.477\) repeat vs. \(0.403\) LLM; retrieval rank \(24.1\) vs. \(28.5\)).

### Hypothesis 2: Procedural Advantage in Predictive Lift
* **Hypothesis**: Even if persistence is strong, the LLM's predictive lift over persistence (\(\text{Sim}_{\text{LLM}} - \text{Sim}_{\text{Repeat}}\)) will be significantly larger in procedural domains (*Military*) than in stochastic domains (*Nature & Scenery*).
* **Statistical Test**: Channel-clustered regression of lift on domain indicator (`is_nature`) across 105 videos (314 segments) from 15 YouTube channels at the poles.
* **Empirical Status**: **Tested & Rejected**. Lift over repeat is negative in both domains (\(-0.078\) in Military, \(-0.060\) in Nature) and the domain coefficient is statistically non-significant (\(\beta = +0.017, p = 0.304\)).

### Hypothesis 3: Rival Explanation — Scene-Change Rate & Physical Autocorrelation
* **Hypothesis**: Apparent domain differences in relative modality sensitivity (\(\Delta / N\)) are mediated by physical scene continuity (how static the video is) rather than semantic script determinism.
* **Statistical Test**: Channel-clustered multiple regression of \(\Delta / N\) on domain indicator and physical visual frame continuity (\(v_{\text{continuity}}\)).
* **Empirical Status**: **Tested & Confirmed**. Frame continuity is a massive, significant predictor (\(\beta = 3.427, p = 0.0065\)). Controlling for visual continuity cuts the domain coefficient nearly in half (from \(0.228\) to \(0.120\)) and reduces it to non-significance (\(p = 0.0863\)).

---

## 3. Benchmark Dataset Protocol

* **Development Cohort (Wild4)**: \(N = 294\) masked segments across 98 unique videos (100 total, 98 with complete paired embeddings).
* **Test Cohort (Wild5)**: \(N = 673\) masked segments across 225 unique videos (235 total).
* **Disjointness**: 0 overlapping video IDs and 0 overlapping source video stems between Wild4 and Wild5 (100% disjoint).
* **Gap Widths**: Primary evaluation at \(w = 6\)s; temporal gap scaling evaluated at \(w = 3\)s.
* **Unit Resolution**: Standardized 60-second video clips at 1-second resolution.

---

## 4. Execution Scripts & Pipelines

1. **Hypothesis Regression Suite**:
   [`scripts/run_hypothesis_regression_tests.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/run_hypothesis_regression_tests.py)
   * Runs the channel-clustered regression models for H1, H2, and H3 with cluster-robust standard errors.
2. **Cluster Statistics & Continuity Pipeline**:
   [`scripts/compute_cluster_and_continuity_stats.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/compute_cluster_and_continuity_stats.py)
   * Computes per-video aggregations, cluster bootstrap 95% CIs, exact binomial tests against 50% chance, and cross-pole Mann-Whitney / Cohen's \(d\).
3. **Publication Figures**:
   [`scripts/plot_paper_figures.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/plot_paper_figures.py)
   * Figure 1: Relative Modality Sensitivity across Categories.
   * Figure 2: (A) Gap Scaling (\(w=3\) vs. \(w=6\)); (B) Direct Text-Space Test of H1 & H2.
