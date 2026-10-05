# Paper Outline: Testing Temporal In-Filling

**Title**: Testing Temporal In-Filling: Zero-Shot Semantic Inference vs. Trivial Persistence in Video Caption Reconstruction  
**Format**: 4-Page Workshop / Short Paper (e.g. CVPR/ECCV/EMNLP/ACL workshops)  
**Target Manuscript**: [`docs/paper/draft.md`](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/draft.md)

---

## 1. Abstract & Introduction
* **The Multimodal Dilemma**: Video-LLMs encode dense video frames uniformly, ignoring temporal information density.
* **The Core Diagnostic Question**: When a video interval is omitted, does an LLM reading boundary captions infer the missing content better than assuming persistence, and is that advantage larger in procedural domains?
* **The Three Explicit Hypotheses**:
  - **H1 (Beyond Persistence)**: Does an LLM (Llama 3.1 8B) beat non-parametric persistence (copy-nearest caption, text LERP) in shared SigLIP text space?
  - **H2 (Procedural Advantage in Lift)**: Is the LLM's lift over persistence larger in procedural domains (*Military*) than in stochastic domains (*Nature & Scenery*)?
  - **H3 (Rival Explanation: Scene-Change Rate)**: Is the cross-modal sensitivity spectrum (\(\Delta / N\)) mediated by physical frame continuity (\(v_{\text{continuity}}\)) rather than causal script determinism?

---

## 2. Benchmark Architecture & Methodology
* **Task Formulation**: Masked temporal in-filling over standardized 60-second clips at 1-second resolution (\(w \in \{3, 6\}\)s).
* **Shared Representation Space**: Dual-encoder **SigLIP** (`google/siglip-base-patch16-224`, 768-d).
* **Models & Pathways**:
  - **Text Pathway**: Llama 3.1 8B (zero-shot causal in-filling) vs. Copy-Nearest Caption vs. Text LERP vs. Random controls.
  - **Visual Pathway**: Visual LERP (midpoint feature interpolation) vs. Visual Copy-Nearest.
* **Evaluation Metrics & Statistical Safeguards**:
  - Direct target cosine similarity in modality home space.
  - Candidate retrieval rank out of 60 seconds within video.
  - Normalized Rank Delta (\(\Delta / N\)) as a zero-sum relative sensitivity index.
  - **Channel-Level Clustering**: All significance testing and regressions clustered by YouTube creator channel to prevent pseudoreplication across the 15 channels (105 videos at poles).
  - **Independent Splits**: Development cohort (Wild4, \(N_{\text{vid}}=98\)) and Test cohort (Wild5, \(N_{\text{vid}}=225\)), with 0 overlapping videos.

---

## 3. Empirical Results: Answering the Three Hypotheses

### 3.1 Test of H1: Beyond Persistence (\(\rightarrow\) Rejected)
* Persistence outperforms Llama 3.1 8B across all 6 domains in SigLIP text space:
  - Copy-nearest achieves \(\text{Sim} = 0.477 \pm 0.088\) (Rank \(24.1\)) vs. Llama \(0.403 \pm 0.096\) (Rank \(28.5\)) in Military.
  - Text LERP achieves \(\text{Sim} = 0.500 \pm 0.082\).
  - Llama demonstrates grounding over random distractors (\(0.403\) vs. \(0.382\)), but fails to overcome the persistence prior.

### 3.2 Test of H2: Procedural Advantage in Lift (\(\rightarrow\) Rejected)
* Channel-clustered regression of text lift (\(\text{Lift} = \text{Sim}_{\text{LLM}} - \text{Sim}_{\text{Repeat}}\)) on domain:
  - Intercept (Military): \(\beta_0 = -0.0777, p < 0.0001\).
  - Domain effect (`is_nature`): \(\beta_1 = +0.0173, p = 0.3040\).
* The LLM suffers a significant penalty relative to persistence across both procedural and stochastic regimes.

### 3.3 Test of H3: Scene-Change Rate Mediation (\(\rightarrow\) Confirmed)
* Adjacent visual frame continuity (\(v_{\text{continuity}}\)) is significantly higher in Nature & Scenery (\(0.950 \pm 0.029\)) than in Military (\(0.922 \pm 0.034, p < 0.001\)).
* Channel-clustered multiple regression of \(\Delta / N\) on domain and visual continuity:
  - Frame continuity is a massive, significant predictor (\(\beta = 3.427, p = 0.0065\)).
  - Controlling for continuity cuts the domain coefficient nearly in half (\(0.228 \to 0.120\)) and reduces it to non-significance (\(p = 0.0863\)).
* Disproved caption wording jitter: Adjacent caption continuity (\(\text{Sim}(c_t, c_{t+1})\)) is identical across domains (\(p = 0.393\)).

### 3.4 Full Spectrum & Independent Cohort Replication
* Table of Dev (\(N_{\text{vid}}=98\)) vs. Test (\(N_{\text{vid}}=225\)) showing consistent pole divergence between Military and Nature (Test: \(p = 6.16 \times 10^{-4}, d = -0.826\); Dev: \(p = 0.0055, d = -1.268\)).
* Intermediate domains (Survival, Farming, Disaster) exhibit 45%–53% win rates, forming a transition zone centered at dataset chance (\(\sum \Delta / N \equiv 0\)).

---

## 4. Discussion, Mechanisms & Architectural Lessons
* **The Persistence Deficit & Key Events**:
  - Random temporal masking overwhelmingly samples visually static connective tissue where persistence is optimal.
  - On computationally isolated "Wild-Key-Events" (high semantic anomaly moments), the LLM demonstrates semantic bridging where persistence fails.
* **Architectural Implications**:
  - Prune static video intervals (\(v_{\text{continuity}} \approx 0.95\)) via trivial keyframe repeat with zero LLM compute.
  - Route discontinuous, complex key events to LLMs for semantic narrative deduction.
* **Limitations**: Zero-shot single prompt; representation similarity vs. extrinsic downstream QA.
