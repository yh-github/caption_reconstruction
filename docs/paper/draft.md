# Testing Temporal In-Filling: Zero-Shot Semantic Inference vs. Trivial Persistence in Video Caption Reconstruction

## Abstract
Recent Video-LLMs typically treat video understanding as an uninterrupted stream of dense visual encoding. However, real-world events often follow structured semantic scripts that pre-trained language models might predict without continuous perceptual input. In this work, we formulate a direct empirical test: **When a temporal interval of video is omitted, does an LLM reading surrounding captions recover missing content better than assuming persistence, and is that advantage larger in procedural domains?** We benchmark an open-weight LLM (**Llama 3.1 8B**) against non-parametric temporal persistence baselines (**copy-nearest caption, boundary feature interpolation**) across 323 diverse videos (\(N=967\) segments) from the **WildQA** benchmark. Our evaluation yields three decisive findings: (1) In embedding space, trivial persistence consistently outperforms zero-shot LLM in-filling across all domains (copy-nearest caption achieves cosine similarity \(0.477\) vs. \(0.403\) for the LLM in Military; retrieval rank \(24.1\) vs. \(28.5\)). (2) The LLM's predictive lift over text persistence does not differ between procedural and stochastic domains (channel-clustered regression \(p = 0.304\)). (3) While cross-modal relative sensitivity (\(\Delta / N\)) reliably separates Military from Nature & Scenery across independent splits (Test: video-level Mann-Whitney \(U = 378.0, p = 6.16 \times 10^{-4}, d = -0.826\); Dev: \(p = 0.0055\)), multiple regression reveals that this separation is heavily mediated by physical scene-change rate (\(\beta = 3.427, p = 0.007\)), which cuts the domain effect nearly in half and reduces it to non-significance (\(p = 0.086\)). Rather than demonstrating that language models make vision redundant in procedural workflows, our findings reveal that non-parametric visual continuity is an exceptionally strong baseline in static scenes, while LLM temporal reasoning remains constrained by the persistence prior and caption stylistic variance.

---

## 1. Introduction
The central promise of multimodal AI is the integration of perceptual grounding with causal reasoning. Yet modern Video-LLM architectures largely brute-force this integration: models process dense sequences of visual tokens at uniform frame rates, regardless of underlying information density. This paradigm overlooks a fundamental question: **When does a video actually require continuous visual perception, and when can missing content be inferred from context?**

Consider an aircraft maintenance crew preparing equipment. If a 6-second segment is omitted between unpacking a pressure gauge and testing a hydraulic line, a human—or an LLM possessing procedural world knowledge—might deduce that the technician attached the hose and calibrated the needle. Conversely, in a wildlife video of a primate moving through dense canopy, chaotic locomotion makes predicting the exact landing branch virtually impossible, yet visual background features persist smoothly across time.

To test whether multimodal models can leverage this distinction, we formulate a **Comparative Reconstruction Framework** structured around one question and three explicit, testable hypotheses:

> **Central Question**: When a span of video is hidden, does an LLM reading surrounding captions recover the missing content better than assuming persistence, and is that advantage larger in procedural domains?

* **Hypothesis 1 (Beyond Persistence)**: An LLM with world knowledge will infer what happens beyond trivial persistence, achieving higher target similarity and better retrieval ranks than copying boundary captions or linearly interpolating boundary embeddings.
* **Hypothesis 2 (Procedural Advantage)**: The LLM's predictive lift over persistence will be significantly larger in procedural domains (*Military*) than in stochastic domains (*Nature & Scenery*).
* **Hypothesis 3 (Rival Explanation: Scene-Change Rate)**: Apparent domain differences in cross-modal reconstruction are mediated by physical scene continuity (how static the video is) rather than semantic script determinism.

By testing these hypotheses across 323 videos from the WildQA benchmark, clustered across 15 YouTube channels at the domain poles, we provide an honest, empirical diagnostic of temporal in-filling capabilities.

---

## 2. Methodology: The Comparative Reconstruction Framework

### 2.1 Problem Formulation: Masked Temporal In-Filling
We represent a video \(V\) as a standardized 60-second temporal sequence of 1-second units \(u_t = (v_t, c_t)\) for \(t \in \{0, \dots, 59\}\), where \(v_t \in \mathbb{R}^{768}\) is a per-second visual embedding and \(c_t\) is a dense descriptive caption.

A contiguous temporal interval \(M = \{t_{\text{start}}, \dots, t_{\text{end}}\}\) of duration \(w \in \{3, 6\}\) seconds is masked. The task is to reconstruct the semantic content of the missing interval. Rather than pixel-level synthesis—which concentrates compute on high-frequency textural noise—we evaluate representation alignment in a shared multimodal metric space (**SigLIP**, `google/siglip-base-patch16-224`).

#### Caption Provenance, Audit & Channel Attribution
The ground-truth oracle captions \(c_t\) were generated using **Gemini 1.5 Flash** prompted to produce objective, frame-level perceptual descriptions of isolated 1-second clips without temporal context.
1. **Quality Audit**: A manual spot-audit of 20 randomly sampled 1-second clips by the primary author verified 20/20 perceptual accuracy (Wilson score 95% CI \([83.9\%, 100\%]\)) with zero narrative hallucination. Automated screening confirmed 0 duplicate clusters and zero boilerplate text.
2. **Channel Clustering**: To prevent confounding domain effects with individual channel production styles, videos were mapped to their source YouTube series channels (e.g., *AiirSource Military*, *WarLeaks*, *King Kong*, *TreadmillTV*). The two extreme poles (*Military* and *Nature & Scenery*) encompass 105 videos across 15 distinct channels.

### 2.2 The Reconstruction Pathways & Baselines

#### 2.2.1 Semantic Inference (Text Pathway)
Conditioned on observed context captions \(C_{\text{obs}} = \{(t, c_t) \mid t \notin M\}\), **Llama 3.1 8B** (open-weights) is prompted zero-shot to infer the missing narrative sequence chronologically. Reconstructed captions \(\hat{c}_t\) are projected into SigLIP text embedding space: \(\hat{e}_{\text{text}} = \text{Encoder}_{\text{text}}(\hat{c}_t)\).

To rigorously test whether the LLM predicts state progression ("what must happen") rather than context inertia ("what persists"), we evaluate against:
* **Copy-Nearest Caption (`Caption_RepeatClosest`)**: Copies the boundary caption nearest to target timestamp \(t\).
* **Text LERP (`Caption_MeanClosest`)**: Linearly interpolates the boundary caption embeddings.
* **Within-Domain Random Control**: Samples captions from unrelated videos within the same domain.
* **Cross-Domain Random Control**: Samples captions from unrelated videos across different domains.

#### 2.2.2 Temporal Continuity (Visual Pathway)
Given observed frame vectors \(V_{\text{obs}} = \{v_t \mid t \notin M\}\):
* **Visual LERP (`Visual_SigLIP_MeanClosest`)**: Linearly interpolates between boundary vectors:
\[
\hat{v}_t = (1 - \alpha_t) v_{t_{\text{start}}-1} + \alpha_t v_{t_{\text{end}}+1}, \quad \alpha_t = \frac{t - t_{\text{start}} + 1}{w + 1}
\]
normalized to unit length: \(\hat{e}_{\text{vis}} = \hat{v}_t / \|\hat{v}_t\|\).
* **Visual Repeat (`Visual_SigLIP_RepeatClosest`)**: Copies the nearest observed boundary frame vector.

### 2.3 Evaluation Metrics
1. **Target Cosine Similarity**: Evaluated directly against the ground-truth target. Text predictions are scored against the target caption in SigLIP text space (\(\text{Sim}(\hat{e}_{\text{text}}, e_{\text{target\_cap}})\)); visual baselines are scored against the target frame in SigLIP visual space (\(\text{Sim}(\hat{e}_{\text{vis}}, v_{\text{target\_frame}})\)).
2. **Candidate Retrieval Rank**: The rank of the true target among the 60 candidate units in the video (Rank 1 = best, chance expectation = 30.5).
3. **Normalized Rank Delta (\(\Delta / N\))**: To compare relative ease across modalities despite different scoring noise ceilings, we compute:
\[
\Delta / N = \frac{\text{Rank}(\hat{e}_{\text{text}}) - \text{Rank}(\hat{e}_{\text{vis}})}{N}
\]
Because all \(N\) segments are ranked in the same pool, \(\sum \Delta / N \equiv 0\) by construction. \(\Delta / N\) is therefore a **zero-sum relative modality sensitivity index**, measuring whether a domain is relatively more favorable to text (\(\Delta / N < 0\)) or vision (\(\Delta / N > 0\)) than the benchmark average.

---

## 3. Related Work
* **Token Pruning in Vision-Language Models**: DynamicViT (Rao et al., 2021) and FastV (Chen et al., ECCV 2024) prune visual tokens based on spatial redundancy or attention scores after early transformer layers. Our work explores the temporal dimension: identifying when entire multi-second spans of video contain redundant visual information.
* **Multimodal Representation Loss**: Li et al. (Findings of EMNLP 2025) demonstrate in *Lost in Embeddings* that visual-to-text projection can introduce information bottlenecks. Our framework investigates the inverse question: when can textual deduction adequately substitute for missing visual features?
* **Action Prediction & Inpainting**: VideoPainter (Bian et al., 2025) and TEAR (Bosetti et al., ICPR 2024) explore visual synthesis and text-enhanced action recognition. We focus on representation-level semantic reconstruction without generative pixel overhead.

---

## 4. Empirical Results: Answering the Three Hypotheses

### 4.1 Testing Hypothesis 1: Does the LLM Infer What Happens Beyond Persistence?

To test H1, Table 1 compares Llama 3.1 8B against trivial text persistence baselines (`Caption_RepeatClosest` and `Caption_MeanClosest`) across all 6 domains in the Wild5 test split (\(w=6\)s).

| Category / Domain | Llama 3.1 8B<br>Sim \| Rank | Copy-Nearest (`Repeat`)<br>Sim \| Rank | Text LERP (`Mean`)<br>Sim \| Rank | Lift over Repeat<br>\(\Delta \text{Sim}\) |
|---|:---:|:---:|:---:|:---:|
| **Military** | 0.403 \| 28.5 | **0.477 \| 24.1** | **0.500 \| 24.3** | \(-0.074\) |
| **Natural Disaster** | 0.415 \| 26.5 | **0.490 \| 22.9** | **0.510 \| 23.3** | \(-0.075\) |
| **Survival** | 0.403 \| 25.2 | **0.506 \| 20.3** | **0.520 \| 20.9** | \(-0.103\) |
| **Action & Vehicle** | 0.381 \| 28.2 | **0.483 \| 24.7** | **0.486 \| 25.9** | \(-0.102\) |
| **Farming** | 0.356 \| 28.3 | **0.436 \| 23.6** | **0.457 \| 24.3** | \(-0.080\) |
| **Nature & Scenery** | 0.429 \| 26.4 | **0.495 \| 24.8** | **0.519 \| 25.0** | \(-0.066\) |

**Finding for H1**: **Hypothesis 1 is not supported.** In embedding space, trivial persistence outperforms zero-shot LLM in-filling across all domains. Copying the nearest boundary caption achieves higher cosine similarity (\(0.436\)–\(0.506\) vs. \(0.356\)–\(0.429\)) and better retrieval ranks (\(20.3\)–\(24.8\) vs. \(25.2\)–\(28.5\)). Text LERP achieves even higher similarity (\(0.457\)–\(0.520\)).

While the LLM demonstrates genuine grounding over random distractors (LLM \(0.424\) vs. within-video random \(0.382\)), it fails to surpass trivial persistence. Because real-world video frames at 1-second resolution exhibit high temporal autocorrelation and adjacent captions share VLM stylistic conventions, persistence is a formidable baseline that zero-shot language models do not overcome.

---

### 4.2 Testing Hypothesis 2: Is There a Procedural Advantage in Text Lift?

Hypothesis 2 posits that even if persistence is strong, the LLM will exhibit a smaller deficit (or positive lift) in procedural domains (*Military*) where causal scripts constrain subsequent actions, compared to stochastic domains (*Nature & Scenery*).

We regress text-space lift (\(\text{Lift} = \text{Sim}_{\text{LLM}} - \text{Sim}_{\text{Repeat}}\)) on domain (`is_nature`), clustering standard errors by YouTube channel across 105 videos (314 segments) at the two poles:

\[
\text{Lift}_i = \beta_0 + \beta_1 \cdot \text{is\_nature}_i + \epsilon_i
\]

| Parameter | Coefficient (\(\beta\)) | Cluster SE | \(z\)-score | \(p\)-value | 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Intercept (\(\beta_0\), Military)** | \(-0.0777\) | 0.0121 | \(-6.410\) | \(< 0.0001\) | \([-0.1015, -0.0540]\) |
| **Domain (\(\beta_1\), `is_nature`)** | \(+0.0173\) | 0.0168 | \(+1.028\) | **\(0.3040\)** | \([-0.0157, +0.0503]\) |

*Standard errors clustered by 15 YouTube channels.*

**Finding for H2**: **Hypothesis 2 is not supported.** The domain term is statistically non-significant (\(p = 0.304\)). The LLM incurs a substantial penalty relative to persistence in both Military (\(-0.078\)) and Nature (\(-0.060\)). Zero-shot language models do not gain a measurable procedural advantage over persistence in text semantic space.

---

### 4.3 Testing Hypothesis 3: Does Scene-Change Rate Mediate the Cross-Modal Spectrum?

If language models exhibit no procedural advantage in text space, why does the cross-modal index (\(\Delta / N\)) reliably separate Military from Nature & Scenery? Hypothesis 3 proposes that the effect is driven by physical visual autocorrelation: static scenes favor visual baselines, while dynamic scenes degrade them.

We empirically measured adjacent visual frame continuity (\(v_{\text{continuity}} = \frac{1}{T-1} \sum_{t} \text{Sim}(v_t, v_{t+1})\)) across 323 videos. Nature & Scenery exhibits significantly higher visual continuity (\(0.9497 \pm 0.0292\)) than Military (\(0.9225 \pm 0.0344\), Mann-Whitney \(p < 0.001\)).

We fit two channel-clustered regression models predicting \(\Delta / N\) across 105 videos (314 segments) at the poles:
* **Model 3A (Unadjusted Domain Effect)**: \(\Delta / N \sim \text{is\_nature}\)
* **Model 3B (Adjusted for Scene Continuity)**: \(\Delta / N \sim \text{is\_nature} + v_{\text{continuity}}\)

| Model & Covariates | Coefficient (\(\beta\)) | Cluster SE | \(z\)-score | \(p\)-value | 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Model 3A (Unadjusted)** | | | | | |
| Intercept (Military) | \(-0.0963\) | 0.0357 | \(-2.694\) | \(0.0071\) | \([-0.1663, -0.0262]\) |
| `is_nature` | \(+0.2284\) | 0.0560 | \(+4.078\) | **\(0.000045\)** | \([+0.1186, +0.3382]\) |
| **Model 3B (Adjusted)** | | | | | |
| Intercept | \(-3.2843\) | 1.1704 | \(-2.806\) | \(0.0050\) | \([-5.5783, -0.9903]\) |
| `is_nature` | \(+0.1196\) | 0.0697 | \(+1.715\) | **\(0.0863\)** | \([-0.0171, +0.2562]\) |
| \(v_{\text{continuity}}\) | \(+3.4268\) | 1.2605 | \(+2.719\) | **\(0.0065\)** | \([+0.9563, +5.8973]\) |

**Finding for H3**: **Hypothesis 3 is confirmed.** Physical visual continuity is a massive, statistically significant predictor of relative modality sensitivity (\(\beta = 3.427, p = 0.0065\)). Controlling for visual continuity reduces the domain coefficient by nearly half (from \(0.228\) to \(0.120\); \(\Delta = -0.108\), 95\% bootstrap CI \([-0.205, -0.011]\)) and reduces the direct effect to non-significance (\(p = 0.0863\)).

The Predictability Spectrum is primarily driven by physical scene dynamics: in static nature scenes, visual persistence achieves near-perfect frame similarity (\(\approx 0.95\)), making visual baselines virtually unbeatable. In dynamic military scenes, rapid camera and actor motion degrade visual continuity, making language in-filling *relatively* more competitive.

---

### 4.4 The Full Spectrum & Gap Scaling across Benchmark Splits

Table 2 presents the full benchmark across all 6 domains and two independent splits (Wild4 Dev, \(N_{\text{vid}}=98\); Wild5 Test, \(N_{\text{vid}}=225\)), confirming 0 overlapping videos between splits.

| Category / Domain | Dev Split (Wild4, \(w=6\))<br>\(N_{\text{vid}}\) / \(N_{\text{seg}}\) \| \(\Delta / N\) | Test Split (Wild5, \(w=6\))<br>\(N_{\text{vid}}\) / \(N_{\text{seg}}\) \| \(\Delta / N\) | Test Win Rate<br>(Seg \| Vid) | Test Split (Wild5, \(w=3\))<br>\(\Delta / N\) \([\% < 0]\) | Modality Sensitivity |
|---|:---:|:---:|:---:|:---:|---|
| **Military** | 19 / 57 \| \(-0.107 \pm 0.334\) | 44 / 132 \| \(\mathbf{-0.092 \pm 0.350}\) | **\(64.4\%\)** \| \(61.4\%\) | \(\mathbf{-0.115 \pm 0.341}\) \([68.2\%]\) | **Relative Semantic Sensitivity**\(^a\) |
| **Natural Disaster** | 14 / 42 \| \(-0.044 \pm 0.352\) | 33 / 98 \| \(-0.020 \pm 0.310\) | \(53.1\%\) \| \(57.6\%\) | \(-0.025 \pm 0.346\) \([51.5\%]\) | Transition Zone (n.s.) |
| **Survival** | 28 / 84 \| \(+0.010 \pm 0.300\) | 61 / 183 \| \(-0.021 \pm 0.330\) | \(51.9\%\) \| \(55.7\%\) | \(+0.007 \pm 0.355\) \([51.6\%]\) | Transition Zone (n.s.) |
| **Action & Vehicle**\(^b\) | 5 / 15 \| \(+0.015 \pm 0.455\) | 5 / 15 \| \(+0.005 \pm 0.323\) | \(53.3\%\) \| \(60.0\%\) | \(-0.094 \pm 0.296\) \([53.3\%]\) | Insufficient Data (\(N=5\) vids) |
| **Farming** | 22 / 66 \| \(+0.026 \pm 0.353\) | 50 / 150 \| \(+0.044 \pm 0.366\) | \(44.7\%\) \| \(40.0\%\) | \(+0.022 \pm 0.380\) \([46.0\%]\) | Transition Zone (n.s.) |
| **Nature & Scenery** | 10 / 30 \| \(\mathbf{+0.174 \pm 0.342}\) | 32 / 95 \| \(\mathbf{+0.119 \pm 0.368}\) | \(35.8\%\) \| **\(31.2\%\)** | \(\mathbf{+0.152 \pm 0.356}\) \([35.4\%]\) | **Relative Visual Sensitivity**\(^a\) |

\(^a\) *Cross-Pole Statistics (Military vs. Nature & Scenery):*  
* *Test Split (\(w=6\)): Video-level \(U = 378.0, p = 6.16 \times 10^{-4}\), Cohen's \(d = -0.826\); Segment-level \(U = 4112.5, p = 9.91 \times 10^{-6}, d = -0.588\).*  
* *Dev Split (\(w=6\)): Video-level \(U = 34.0, p = 0.00550\), Cohen's \(d = -1.268\); Segment-level \(U = 479.5, p = 8.12 \times 10^{-4}, d = -0.835\).*  
* *Test Split (\(w=3\)): Video-level \(U = 283.0, p = 9.69 \times 10^{-6}\), Cohen's \(d = -1.180\); Segment-level \(U = 3614.5, p = 3.14 \times 10^{-8}, d = -0.769\).*  
\(^b\) *Action & Vehicle comprises only 5 unique videos in both splits; statistics are reported for completeness.*

![Figure 1: The Predictability Spectrum across Video Domains](../../results/plots/paper_figures/fig1_predictability_spectrum.png)
*Figure 1: Relative Modality Sensitivity across Categories. Normalized Rank Delta (\(\Delta / N\)) for Dev (Wild4, \(N_{\text{vid}}=98\)) and Test (Wild5, \(N_{\text{vid}}=225\)) splits with cluster-bootstrap 95% confidence intervals.*

![Figure 2: Gap Duration Scaling and Direct Hypothesis Testing](../../results/plots/paper_figures/fig2_boundary_inertia.png)
*Figure 2: (A) Gap Scaling (\(w=3\) vs. \(w=6\)) showing stable endpoint divergence. (B) Direct Text-Space Test of H1 and H2: Persistence beats Llama 3.1 8B in both Military and Nature, with no significant difference in lift between domains (\(p = 0.304\)).*

---

### 4.5 Qualitative Illustrations: Best-Case vs. Median Realities

To inspect how feature geometry behaves in practice:

**Case A: Procedural Protocol (Military)**
* *Best-Case Outlier (`AiirSource-Military_1-clip-0`, \(\Delta = -246\), Rank 12 text vs. 258 visual)*: A firefighter in heat-reflective gear prepares equipment. During the masked interval (\(t=0\dots5\)), the firefighter dons and tightens a gas mask. Because posture shifts rapidly, linear interpolation of 768-d SigLIP feature vectors drifts from intermediate frames. Conditioned on protocol context, the LLM correctly generates the donning sequence ("secures mask"), achieving Rank 12.
* *Representative Median Example (`AiirSource-Military_8-clip-1`, \(\Delta = -47\), \(\Delta / N = -0.070\))*: In a typical procedural segment, both modalities show modest performance (Rank 403 text vs. 450 visual); procedural cues give text a minor relative edge, but persistence remains strong.

**Case B: Stochastic Physical Dynamics (Nature & Scenery)**
* *Video ID: `King-Kong-Amazon_5-clip-14` (\(\Delta = +81\), Rank 145 text vs. 64 visual)*: A primate moves through dense foliage. Arboreal movement is unpredictable, leading the LLM to generate generic foraging descriptions. Meanwhile, canopy foliage color histograms and ambient lighting allow SigLIP visual feature interpolation to track scene representations smoothly.

---

## 5. Discussion, Limitations & Practical Implications

### Practical Implications
Rather than justifying a broad replacement of visual tokens with language models, our findings provide a more grounded, cautionary architecture lesson:
1. **Compress Static Scenes with Trivial Persistence**: In nature, surveillance, and scenery streams with high temporal autocorrelation (\(v_{\text{adj}} \approx 0.95\)), expensive visual token encoding can be aggressively pruned by simply repeating boundary keyframes, requiring zero language model compute.
2. **The High Bar for Semantic In-Filling**: In dynamic procedural streams, zero-shot language models do not automatically surpass persistence. True temporal in-filling requires models specifically trained to overcome the persistence prior, explicitly predicting state transitions rather than ambient descriptions.

### Limitations
1. **Single Model and Prompt**: We evaluated Llama 3.1 8B with a single zero-shot prompt. Larger frontier models or instruction-tuned chain-of-thought prompts may narrow the deficit against persistence.
2. **Metric vs. Downstream Task**: SigLIP embedding similarity evaluates representation proximity, which conflates semantic correctness with captioner lexical style. Future work should evaluate keyframe-mediated in-filling on downstream Question Answering (e.g. WildQA).

---

## References
* **Bian, Y., et al.** (2025). VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. *SIGGRAPH*.
* **Bosetti, M., et al.** (2024). Text-Enhanced Zero-Shot Action Recognition: A training-free approach. *ICPR*.
* **Castro, S., et al.** (2022). WildQA: In-the-Wild Video Question Answering. *COLING*.
* **Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B.** (2024). An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models. *ECCV*. (arXiv:2403.06764).
* **Li, W., et al.** (2025). Lost in Embeddings: Information Loss in Vision-Language Models. *Findings of EMNLP*.
* **Rao, Y., et al.** (2021). DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. *NeurIPS*.
