# Qualitative and Quantitative Analysis of SLM Reconstruction Victories

This document provides a comprehensive analysis of when, why, and how generative language models (**Llama-3.1-8B**) outperform visual continuity baselines (**Google SigLIP `Visual_SigLIP_MeanClosest`** via midpoint LERP) in video caption cloze reconstruction across the 335-video benchmark (`wild4` + `wild5`).

---

## 1. Executive Summary & The "Shrinking Gap" Phenomenon

In paired evaluations on 335 videos across gap widths \(W \in [1, 2, 3, 4, 6, 8, 12, 16]\), visual feature continuity generally outperforms zero-shot language modeling. However, the performance gap between the two modalities changes dramatically with gap width:

| Gap Width \(W\) | Shared Videos | Mean Llama MRR | Mean SigLIP MRR | Margin (\(\text{MRR}_{\text{LLM}} - \text{MRR}_{\text{Vid}}\)) | Llama Win Rate (%) | Video Win Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **\(W=1\)** | 320 | 0.119 | **0.238** | -0.119 | **31.2%** | 68.8% |
| **\(W=3\)** | 225 | 0.085 | **0.175** | -0.090 | **19.6%** | 80.4% |
| **\(W=6\)** | 323 | 0.090 | **0.129** | -0.039 | **29.7%** | 70.3% |
| **\(W=12\)** | 322 | 0.083 | **0.092** | -0.009 | **39.1%** | 60.9% |
| **\(W=16\)** | 306 | 0.076 | **0.081** | -0.005 | **44.8%** | 55.2% |

### Deconstructing the Convergence:
- **What it looks like at first glance**: It appears that Llama becomes more competitive and capable as the missing gap widens to 12 or 16 seconds.
- **The Empirical Reality**: This is an artifact of **visual baseline collapse**, not language model improvement.
  - Llama's MRR stays essentially flat near the random chance floor across widths (\(0.119 \to 0.088 \to 0.083 \to 0.076\); chance for \(N=60\) is \(\approx 0.078\)).
  - SigLIP's visual continuity decays steeply from \(0.238 \to 0.175 \to 0.129 \to 0.092 \to 0.081\) because consecutive video frames decorrelate over 12–16 second intervals.
  - As the visual baseline drops towards chance, Llama wins on more instances by default (rising from \(19.6\%\) to \(44.8\%\)).

---

## 2. Quantitative Predictors of Llama Victories

Using the interactive **Llama Winners Explorer** in [`scripts/evaluation_explorer_app.py`](file:///home/yoavh/code/antigravity/caption_reconstruction/scripts/evaluation_explorer_app.py), we isolate the prior and posterior thresholds where Llama reliably beats visual interpolation:

```
                   ┌──────────────────────────────────────────────┐
                   │ Video Profile: When Does Language Win?       │
                   ├──────────────────────────────────────────────┤
                   │ 1. High Visual Dynamism: combined_dyn > 15.0 │
                   │ 2. Low Visual Monotony:  APCS_V < 0.78       │
                   │ 3. Boundary Disruption:  cos_sim_min < 0.20  │
                   │ 4. Strong Narrative:     APCS_T in [0.3, 0.5]│
                   └──────────────────────────────────────────────┘
```

1. **High Visual Dynamism (`combined_dynamism > 15.0`)**:
   - In videos with high camera velocity or rapid action, frame-to-frame visual similarity degrades rapidly. Midpoint LERP produces an indistinct, smeared visual vector that matches background noise. Language reasoning, operating on abstracted actions, is immune to physical motion blur.
2. **Low Visual Monotony (`APCS_V < 0.78`)**:
   - Videos with diverse scenes across their 60 seconds prevent visual baselines from exploiting static background inertia.
3. **Visual Boundary Disruption (`cos_sim_min < 0.20` or `cos_sim_residual < 0.05`)**:
   - When the frame immediately before the gap and the frame immediately after have low cosine similarity (e.g. an abrupt shot change or camera repositioning), visual linear interpolation fails completely. In contrast, text models navigate narrative scene changes smoothly.
4. **Moderate-to-High Narrative Structure (`text_combined_dynamism` \(\approx 55 - 70\))**:
   - When the caption transcript progresses at a steady narrative clip without extreme repetition (`APCS_T < 0.50`), Llama has sufficient context to deduce intermediate actions.

---

## 3. Qualitative Archetypes: Where Language Outperforms Vision

Analysis of individual video traces reveals distinct qualitative mechanisms driving language model victories:

### Archetype A: The Hard Shot Cut / Camera Angle Switch
* **Visual Baseline Failure**: At \(t=29\), the camera cuts from a wide establishing shot to a close-up of a subject's hands. `Visual_SigLIP_MeanClosest` calculates:
  \[ \hat{v}_t = 0.5 \cdot v_{\text{wide}} + 0.5 \cdot v_{\text{close\_up}} \]
  This hybrid vector resembles neither shot, landing far down the candidate ranking (Mean Rank \(> 40\)).
* **Llama Victory**: Surrounding captions provide unbroken semantic continuity:
  - Context Before: *"A man stands in the workshop holding a piece of timber."*
  - Ground Truth Cloze: *"The man places the timber onto the workbench."*
  - Context After: *"The man measures the board with a steel ruler."*
  - Llama Output: *"He sets the wood down on the table to prepare for cutting."*
  - **Outcome**: Llama's embedding aligns strongly with the ground-truth action (*"places timber onto workbench"*), achieving Rank 1–3.

### Archetype B: Causal Tool / Procedural Sequences (Survival & Farming)
* **Observed Channels**: `John-Suscovich`, `Bertram-Craft`, `Primitive-Technology`, `Welker-Farms-Inc`.
* **The Mechanism**: Procedural activities follow a strict causal syntax. If a person gathers dry grass and strikes a flint, the missing intermediate step *must* involve blowing on the tinder or nurturing a flame:
  - Context Before: *"He scrapes dry bark shavings into a small pile."*
  - Context After: *"Smoke rises as small flames catch on the dry twigs."*
  - Ground Truth: *"He strikes the ferro rod to create sparks onto the shavings."*
  - Llama Output: *"He uses a fire starter to ignite the tinder nest."*
* **Why Vision Struggles**: The visual appearance of a sparking rod is brief and transient (\(< 1\)s). Midpoint frame interpolation captures static hand positions rather than the dynamic spark action.

### Archetype C: Object Handoff / Target Interaction with Fixed Camera
* **Visual Baseline Limitation**: When the actor remains stationary in the center of the frame, `Visual_SigLIP_MeanClosest` achieves high cosine similarity to all frames in the video, but lacks the discriminative power to identify the *specific* second of interaction versus 10 seconds of preparatory idle standing.
* **Llama Victory**: Language models generate the exact predicate describing the action onset (*"reaches into bag"*, *"picks up wrench"*), easily retrieving the unique ground-truth timestamp from the candidate pool.

---

## 4. Qualitative Archetypes: Where Visual Continuity Dominates

Understanding Llama's weaknesses is equally important for identifying the boundaries of language modeling:

### Archetype D: Atmospheric & Chaotic Dynamics (Nature & Scenery)
* **Observed Channels**: `4k-Relaxation`, `Dan-Robinson`, `Tornado-Trackers`, `Climate-Change`.
* **Failure Mode**: In nature videos (e.g. ocean waves crashing, clouds drifting, storm formation), there is no causal human grammar. Dense captions are often arbitrary or poetically descriptive (*"Waves wash over the shore"*, *"The water glistens in the evening light"*).
* **Why Vision Wins**: SigLIP visual vectors track lighting, horizon lines, and color temperature continuously. Even over a 6-second gap, the interpolated vector easily identifies the correct temporal window among disparate distractor clips.

### Archetype E: Broad Narrative Branches at Wide Horizons (\(W \ge 12\))
* **Failure Mode**: When 12 to 16 seconds are masked, the number of plausible narrative paths multiplies. Llama frequently generates a plausible but completely ungrounded sequence (e.g., imagining dialogue or an unperformed task), causing its retrieval rank to drop to chance (\(r \approx 30.5\)).
* **Visual Robustness**: While visual interpolation at \(W=16\) degrades, it remains physically anchored to the scene geometry, frequently placing in the top 20 distractors and beating unanchored language hallucinations.

---

## 5. Tools for Ongoing Inspection

Researchers and subagents can explore these qualitative instances using our interactive suite:

1. **Streamlit Explorer Application**:
   - Run: `streamlit run scripts/evaluation_explorer_app.py`
   - Navigate to the **"🏆 Llama Winners Explorer"** tab.
   - Adjust prior sliders (`Min Visual Dynamism`, `Max APCS_V`, `Max Boundary Min Cosine`) to isolate subsets where Llama's win rate exceeds \(60\%\).
   - Use the **Qualitative Victory Deep-Dive** dropdown to inspect the exact context captions, predicted text, ground truth, and candidate rankings side by side.
2. **Pre-computed Paired Benchmark Dataset**:
   - Source: [`results/unified_benchmark_master.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/unified_benchmark_master.csv) and [`results/method_rank_differences_per_video.csv`](file:///home/yoavh/code/antigravity/caption_reconstruction/results/method_rank_differences_per_video.csv).

---

## 6. Empirical Distribution Audit: Why Individual Victories Do Not Equal Aggregate Superiority

While the qualitative archetypes in Section 3 illustrate authentic instances of semantic deduction, rigorous regression and hypothesis testing across 323 videos (\(N=967\) segments) establish crucial empirical boundaries:

1. **Persistence Outperforms LLM In-Filling Across All Domains (H1 Rejected)**:
   - When evaluated in shared SigLIP text space, non-parametric text persistence (`Caption_RepeatClosest`, \(\text{Sim} = 0.477 \pm 0.088\)) and Text LERP (\(0.500 \pm 0.082\)) consistently outperform Llama 3.1 8B (\(0.403 \pm 0.096\)).
   - In-filling does not achieve positive lift over persistence in any of the 6 benchmark categories.

2. **No Greater Lift in Procedural vs. Stochastic Domains (H2 Rejected)**:
   - Channel-clustered regression of text lift (\(\text{Lift} = \text{Sim}_{\text{LLM}} - \text{Sim}_{\text{Repeat}}\)) on domain yields \(\beta = +0.017, p = 0.304\). The LLM suffers a comparable penalty relative to persistence across both Military and Nature.

3. **Physical Continuity Drives the Cross-Modal Spectrum (H3 Confirmed)**:
   - The apparent cross-modal advantage of language in dynamic military videos (\(\Delta / N < 0\)) is heavily mediated by physical scene continuity (\(v_{\text{continuity}}\), \(\beta = 3.427, p = 0.0065\)).
   - In static scenes, visual persistence is virtually unbeatable (\(\text{Sim} \approx 0.95\)). In dynamic scenes, visual features decorrelate, making language in-filling *relatively* more competitive without outperforming text persistence.

4. **Connective Tissue Bias vs. Key Events**:
   - Randomly masking video intervals predominantly samples visually static connective tissue where persistence is optimal.
   - For detailed protocol and formal regression tables, consult [`docs/paper/draft.md`](../paper/draft.md) and [`docs/theory/cross_modal_evaluation_metrics.md`](../theory/cross_modal_evaluation_metrics.md).
