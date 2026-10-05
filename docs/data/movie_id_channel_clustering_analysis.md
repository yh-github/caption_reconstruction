# Channel-Level (`movie_id`) Correlation & Clustering Analysis

**Scope & Context**: Analysis of whether video clips sharing the same `movie_id` (YouTube channel / creator series) exhibit significant statistical correlation after conditioning on semantic `category`, based on the WildQA benchmark cohort in [`docs/experiments/335/paired_comparison_Llama-3.1-8B_vs_Visual_SigLIP_MeanClosest_Averaged_over_W=[1,_2,_3,_4,_6,_8,_12,_16],_i=[29].csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/experiments/335/paired_comparison_Llama-3.1-8B_vs_Visual_SigLIP_MeanClosest_Averaged_over_W=%5B1,_2,_3,_4,_6,_8,_12,_16%5D,_i=%5B29%5D.csv).

---

## 1. Executive Summary

**Yes, there is substantial reason to be concerned.** Even after controlling for high-level semantic `category` (e.g., Military, Nature & Scenery, Farming), video clips sharing the same `movie_id` exhibit statistically significant correlation across visual, textual, and model evaluation metrics.

In this benchmark:
- `video_id` represents an isolated 60-second video segment (e.g., `AiirSource-Military_1-clip-0`).
- `stem` represents the source upload video (e.g., `AiirSource-Military_1`). All 335 clips in this evaluation set come from distinct upload stems (\(N = 335\) unique stems).
- `movie_id` represents the **source YouTube creator / channel identity** (e.g., *AiirSource-Military*, *4k-Relaxation*, *Millennial-Farmer*, *Primitive-Technology*).
- The 335 clips are clustered into only **40 unique channels** (mean \(\approx 8.38\) clips per channel, range \(1\) to \(13\)).
- Every channel is strictly nested inside a single category (hierarchical nesting: \(\text{Category} \to \text{Channel / movie\_id} \to \text{Clip / video\_id}\)).

Because visual production styles, camera settings, and recurring environments are consistent within a channel, treating the 335 clips as independent and identically distributed (\(i.i.d.\)) observations leads to **pseudoreplication**, deflated standard errors, and artificially inflated statistical significance.

---

## 2. Empirical Statistical Findings

### 2.1 Nested Variance Decomposition

To test whether `movie_id` explains variance beyond `category`, we compare nested OLS models:
\[
\text{Model 1: } y = \beta_0 + \beta_{\text{category}} + \epsilon
\]
\[
\text{Model 2: } y = \beta_0 + \beta_{\text{category}} + \gamma_{\text{movie\_id}} + \epsilon
\]

| Metric | \(R^2\) (`category`) | \(R^2\) (`category` + `movie_id`) | \(\Delta R^2\) | Partial \(F\) | \(p\)-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `combined_dynamism` | \(0.125\) | \(0.421\) | **\(+0.296\)** | \(4.43\) | \(8.15 \times 10^{-13}\) |
| `APCS_V` (Visual Homogeneity) | \(0.080\) | \(0.338\) | **\(+0.258\)** | \(3.38\) | \(9.66 \times 10^{-9}\) |
| `cos_sim_mean_B` (Visual LERP) | \(0.083\) | \(0.259\) | **\(+0.176\)** | \(2.07\) | \(7.40 \times 10^{-4}\) |
| `text_combined_dynamism` | \(0.089\) | \(0.245\) | **\(+0.155\)** | \(1.79\) | \(6.09 \times 10^{-3}\) |
| `cos_sim_mean_A` (Llama-3.1-8B) | \(0.089\) | \(0.221\) | **\(+0.132\)** | \(1.47\) | \(4.85 \times 10^{-2}\) |
| `APCS_T` (Caption Repetition) | \(0.088\) | \(0.214\) | **\(+0.126\)** | \(1.39\) | \(8.10 \times 10^{-2}\) |
| `rank_diff` (\(\text{Rank}_A - \text{Rank}_B\)) | \(0.091\) | \(0.192\) | \(+0.101\) | \(1.08\) | \(0.350\) |
| `metric_diff` (\(\text{Score}_A - \text{Score}_B\)) | \(0.075\) | \(0.177\) | \(+0.102\) | \(1.07\) | \(0.363\) |

Adding `movie_id` increases explained variance for physical visual dynamism from \(12.5\%\) to \(42.1\%\) (\(\Delta R^2 \approx 30\%\), \(p < 10^{-12}\)) and visual homogeneity from \(8.0\%\) to \(33.8\%\) (\(p < 10^{-8}\)).

### 2.2 Linear Mixed-Effects Model & Intraclass Correlation (ICC)

Fitting a random-intercept model \(y_{ij} = \mu + \alpha_{\text{category}(j)} + u_j + \epsilon_{ij}\) with \(u_j \sim \mathcal{N}(0, \sigma_{\text{channel}}^2)\):
\[
\text{ICC} = \frac{\sigma_{\text{channel}}^2}{\sigma_{\text{channel}}^2 + \sigma_{\text{residual}}^2}
\]

- **Physical Visual Dynamism (`combined_dynamism`)**: \(\text{ICC} = 0.310\) (\(p = 0.0032\)). Nearly one-third of the residual variation within categories is attributable to channel identity.
- **Visual Homogeneity (`APCS_V`)**: \(\text{ICC} = 0.238\) (\(p = 0.0082\)).
- **Text Dynamism (`text_combined_dynamism`)**: \(\text{ICC} = 0.160\) (\(p = 0.0061\)).
- **Visual Baseline Performance (`cos_sim_mean_B`)**: \(\text{ICC} = 0.138\) (\(p = 0.0498\)).
- **Generative LLM Performance (`cos_sim_mean_A`)**: \(\text{ICC} = 0.054\) (\(p = 0.202\)).
- **Paired Score Difference (`metric_diff`)**: \(\text{ICC} = 0.014\) (\(p = 0.661\)).

### 2.3 Within-Category Channel Variance

Running one-way ANOVA within individual categories reveals large differences between channels:

| Category | Channels | Clips | `combined_dynamism` \(F\) | \(p\)-value | Channel Dynamism Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Military** | \(8\) | \(63\) | \(F = 6.72\) | \(9.31 \times 10^{-6}\) | \([7.8, 23.8]\) |
| **Natural Disaster** | \(7\) | \(47\) | \(F = 7.75\) | \(1.46 \times 10^{-5}\) | \([7.6, 22.3]\) |
| **Survival** | \(8\) | \(89\) | \(F = 3.95\) | \(9.36 \times 10^{-4}\) | \([9.3, 16.7]\) |
| **Nature & Scenery** | \(7\) | \(42\) | \(F = 2.63\) | \(3.26 \times 10^{-2}\) | \([3.6, 13.1]\) |
| **Farming** | \(8\) | \(84\) | \(F = 1.85\) | \(0.091\) | \([10.1, 16.8]\) |
| **Action & Vehicle** | \(2\) | \(10\) | \(F = 0.05\) | \(0.829\) | \([11.2, 11.9]\) |

Visual homogeneity (`APCS_V`) also shows statistically significant between-channel variation in 5 out of 6 categories (*Action & Vehicle*: \(p = 0.029\); *Military*: \(p = 0.0048\); *Natural Disaster*: \(p = 0.00088\); *Nature & Scenery*: \(p = 0.00032\); *Survival*: \(p = 0.0061\)).

---

## 3. Root Mechanisms of Same-Channel Correlation

1. **Cinematography & Production Signatures**:
   - Camera mounting and motion: Drone sweeps and tripod shots (e.g., *4k-Relaxation*) vs. body-worn shaky cams (e.g., *WarLeaks*) vs. in-vehicle dash/tractor mounts (e.g., *Millennial-Farmer*).
   - Cutting pace: Some channels employ rapid montage cuts (triggering sharp drops in `APCS_V` and spikes in `peak_dynamism`), whereas others favor uncut continuous takes.
   - Sensor, resolution, and color palette: Consistent color grading and compression artifacts create tight clustering in SigLIP embedding space.

2. **Environmental & Setting Constancy**:
   - Multiple videos from the same creator often share the same recurring physical backdrop (e.g., the same workshop, the same homestead, or the same forest clearing).

3. **Narrator & Caption VLM Bias**:
   - The oracle captions were generated by Gemini 1.5 Flash on 1-second frames.
   - Recurring channel topics, recurring actors/narrators, and channel-specific machinery produce repetitive vocabulary and syntactic structures, driving higher caption pairwise similarity (`APCS_T`).

---

## 4. Methodological Implications & Risks

### A. Pseudoreplication & Inflated Statistical Significance (Type I Error)
When observations within a cluster are correlated, the effective sample size is lower than the nominal sample size \(N\). Using the Design Effect formula:
\[
\text{DEFF} = 1 + (\bar{m} - 1) \times \text{ICC}
\]
where \(\bar{m} \approx 8.38\) is the average cluster size:
- For `combined_dynamism` (\(\text{ICC} \approx 0.31\)):
  \[
  \text{DEFF} \approx 1 + (7.38 \times 0.31) \approx 3.29 \implies N_{\text{eff}} = \frac{335}{3.29} \approx 102
  \]
- For `APCS_V` (\(\text{ICC} \approx 0.24\)):
  \[
  \text{DEFF} \approx 1 + (7.38 \times 0.24) \approx 2.77 \implies N_{\text{eff}} = \frac{335}{2.77} \approx 121
  \]

Standard errors estimated without clustering underestimate actual sampling uncertainty by \(\sqrt{\text{DEFF}} \approx 1.7\) to \(1.8\times\), resulting in artificially low \(p\)-values.

### B. Category Confounding by Dominant Channels
When a category contains very few channels, channel idiosyncrasies become indistinguishable from domain effects:
- *Action & Vehicle* contains only **10 clips** across **2 channels** (*Ultimate-Chase* with 6 clips, *TK-Hinshaw* with 4).
- Findings attributed to "Action & Vehicle" may reflect those two creators' editing conventions rather than the domain as a whole.

### C. Data Leakage in Train/Val/Test Splits
If video clips are randomly assigned to train and test sets at the `video_id` level, clips from the same channel appear on both sides of the split. Models can memorize channel-level stylistic tokens, color palettes, or background characteristics rather than learning general temporal progression.

### D. Distractor Retrieval Contamination
In retrieval benchmarks, same-channel distractors (`other_video_same_channel`) are substantially harder than cross-channel distractors (`other_video_other_channel`) because they share lighting, camera profiles, and vocabulary (as handled in [`src/shared_target/pools.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/src/shared_target/pools.py)).

---

## 5. Mitigations & When It Is Less Concerning

### 1. Paired Comparison Invariance (`metric_diff`)
Because Method A (Llama-3.1-8B) and Method B (Visual SigLIP LERP) are evaluated on the exact same clips, common channel-level baseline shifts affect both methods in parallel. The intraclass correlation for `metric_diff` is low (\(\text{ICC} \approx 0.014, p = 0.661\)). Paired head-to-head performance comparisons are therefore much less sensitive to channel clustering than absolute score evaluations.

### 2. Distinct Video Stems
All 335 clips in this evaluation set originate from separate YouTube video uploads (`stem` uniqueness = 335 / 335), ruling out temporal overlap or duplicate sub-clips from a single video file.

---

## 6. Recommended Best Practices

1. **Clustered Standard Errors**:
   Always cluster standard errors by `movie_id` when running regressions or domain comparisons:
   ```python
   import statsmodels.formula.api as smf
   model = smf.ols("metric_diff ~ C(category)", data=df).fit(
       cov_type="cluster", cov_kwds={"groups": df["movie_id"]}
   )
   ```
   *(This matches the methodology adopted in [`docs/paper/draft.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/draft.md#L100-L135)).*

2. **Linear Mixed-Effects Models**:
   Model channel identity as a random intercept:
   ```python
   mixed_model = smf.mixedlm("metric ~ C(category)", df, groups=df["movie_id"]).fit()
   ```

3. **Grouped Cross-Validation**:
   When training or fine-tuning models, split data using `GroupKFold(groups=df["movie_id"])` to prevent channel leakage between folds.
