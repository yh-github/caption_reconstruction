Here is a detailed peer review of [caption_index.pdf](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.pdf) (with line and source references to [caption_index.tex](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.tex)), structured around your four core questions plus standard ACL Rolling Review (ARR) criteria.

---

## 1. Executive Summary

This paper asks a crisp, practical question about "caption-once, ask-later" video QA pipelines: **when a reader answers better from pre-computed, question-blind captions than from video frames, why does that happen, and when does it actually pay off financially?** 

By holding the input information fixed—feeding the captioner the exact five 224 px frames per 5 s window that the reader sees—the paper decomposes the total effect of captioning into four distinct components: **source** (original video vs. extracted frames), **visual budget** (token count spent encoding frames), **captioner** (stronger vs. self-captioner), and **modality** (text vs. pixels from the same model). Across 297 TempCompass questions and two closed model families (Claude Haiku/Sonnet 5.5, Gemini 3.8 Flash) plus an open-weights check (Qwen2.5-VL-7B), the paper shows that:
1. Converting frames to text offers **no intrinsic modality advantage** (≤ +0.09 for Haiku, −0.09 for Sonnet, −0.12 for Qwen).
2. Accuracy gains come from **distilling a stronger captioner's perception** (+0.064 at identical tokens within Claude; +0.081 at matched tokens for Gemini; +0.125 for Qwen reading Sonnet) and/or a **larger visual token budget** (+0.067 for Gemini image vs. video path).
3. **Financial break-even comes from downgrading the reader**, not from saving input tokens on the same reader (since the fixed reader prompt dominates token counts).
4. Question-blind captions act as an **information bottleneck ("cap")** that systematically drops relational comparisons across simultaneous views (stacked speed videos) and temporal direction on reversed videos (narrating forward or omitting direction).

**Overall Assessment:** In terms of **experimental design and statistical hygiene**, this is one of the cleanest empirical studies you will read in an ACL submission. However, its **external validity** (short 10 s clips on a single synthetic-heavy benchmark to study "long-video indexing") and **extreme telegraphic compression in the writing/formatting** currently hold it back from its full impact.

---

## 2. Evaluation Across Your Criteria

### A. Is it clear? (**Mostly yes on the macro level; too compressed on the micro level**)
- **Macro-structure:** The core thesis is stated immediately in the Abstract and Introduction, and every section answers a single well-defined question (§4.1 Who writes the caption, §4.2 Modality & the accuracy cap, §4.3 Video types & minimal pairs, §4.4 Error audit, §4.5 Question-aware controls, §5 Cost).
- **Micro-level density & telegraphic prose:** The prose is written in an ultra-compressed, Hemingway-esque style that sometimes sacrifices readability for brevity. Readers are hit with dense streams of numbers, confidence intervals, and shorthand arm names before the definitions are fully digested.
- **CRITICAL Layout Bug — Empty Section 2.3:** In [caption_index.tex](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.tex#L107-L128), `\subsection{The four effects}` contains *only* the floating `table*` environment (`Table 2`) and zero body text. Because `table*` floats to the top of page 3, page 2 of [caption_index.pdf](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.pdf) (lines 154–155) renders as an empty heading:
  > **2.3 The four effects**
  > **2.4 Statistics and pre-registration**
  Furthermore, the four effects are never formally defined in prose in §2.3—the reader has to infer their definitions from Table 2 on the next page or wait until §4.1–4.2.
- **Section 3 (Theory) is a single dense paragraph:** In [caption_index.tex](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.tex#L133-L135) (PDF lines 171–192), Propositions 1, 2, and 3 and the definition of *Retention* are all packed inline into one 20-line paragraph without formal environments or equations. Calling them "Propositions" when they are stated as one-sentence observations feels slightly overstated; formatting the additive decomposition and the break-even equation \( Q^* = \frac{C(\tau)}{c_f - c_\tau} \) cleanly would make §3 much easier to reference.
- **Figure 3 legibility:** On page 7 of [caption_index.pdf](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.pdf), Figure 3 squeezes three panels horizontally into a single column (`\columnwidth`). The axis labels, tick marks, and text callouts inside the plot are microscopic (~4–5 pt) and nearly impossible to read without zooming in to 300%. It should be a two-column `figure*` or stacked vertically.

---

### B. Is it founded? Do claims have a strong basis? (**Yes — exceptionally well-founded within its scope**)
Every major claim in the paper is backed by a direct, unconfounded comparison:
- **Claim 1 (Captioner effect > 0):** Supported three ways—across families at high budget (+0.148), across families at matched visual tokens (+0.081), and *within* the Claude family on the exact same 5 JPEG files and prompt (+0.064, where token counts differ by literally 2 tokens per call), plus replicated on Qwen2.5-VL-7B (+0.125) and on minimal pairs (+0.104).
- **Claim 2 (Modality effect ≤ 0 for capable readers):** Supported by comparing each model reading its *own* isolated captions of the 5 frames against reading the 5 frames directly (Haiku +0.030, Sonnet −0.088, Qwen −0.121), plus a 448 px control ruling out frame starvation.
- **Claim 3 (Break-even via reader downgrade):** Supported by exact token accounting (input, output, thinking tokens) at list prices.
- **Claim 4 (Losses are omissions of comparisons and reversals caused by question-blindness):** Triangulated via three independent methods: (1) minimal pairs on sibling videos (Table 4), (2) an LLM-as-judge audit validated against 40 blind human annotations (\(\kappa = 0.81\), Table 5), and (3) a question-aware "describe, then answer" control (+0.22 recovery on stacked videos, §4.5).

The only place where a broader implication stretches beyond the direct evidence is **scaling to long videos** (see Weaknesses below): the paper motivates per-window indexing via long-video QA, but tests exclusively on ~10 s clips (58% have only 1–2 windows) where retrieval/routing across hundreds of windows is absent.

---

### C. Is it scientific and rigorous? (**Outstanding — top 1% of empirical NLP papers**)
This is the paper's greatest strength. It practices a level of scientific and statistical discipline rarely seen in ML/NLP benchmarks:
1. **Information-Theoretic Control (Garbling Control):** Most prior papers comparing captions to frames let the captioner sample denser frames or use a different model without isolating self-captioning. Feeding the captioner the *exact* JPEG frames the reader sees turns vague observations into a rigorous causal decomposition.
2. **Pre-registration & Prediction Logging:** Hypotheses P1–P3 were set on a 100-question dev set, pre-registered before confirmatory runs with Bonferroni-corrected 98.33% CIs, and every follow-up experiment (C2–C12) had point predictions committed to Git (and archived via Software Heritage) prior to execution.
3. **Intellectual Honesty on Nulls & Failed Predictions:** The paper transparently reports when pre-registered hypotheses failed—most notably **P1** (the aspect contrast predicting captions lose continuous quantities like speed/direction more than discrete ones, −0.101 vs. predicted −0.20, which failed because videomodels *also* fail direction/reversal on pixels) and the audit prediction on reversed misstatements (0.28–0.45 vs. predicted ≥ 0.50).
4. **Clustered Bootstrap & Minimal Pairs:** Recognizing that 297 questions come from 156 underlying source clips, the authors check paired bootstrap CIs against clip-clustered CIs (`cl.`) throughout, and use minimal pairs (where guessing yields only 0.093 pair accuracy) to separate true visual grounding from language priors (which score 0.41 from the question alone!).
5. **Variance / Resampling Check:** In Appendix E (PDF lines 1136–1143), the authors exploit the 48 single-window videos—where the per-window and whole-video prompts are identical—to measure pure caption-sampling variance, finding that re-captioning identical inputs changes the reader's answer on 10–17% of questions.

---

### D. Is it novel, impactful, and surprising? Any non-trivial findings? (**Yes — several sharp, counter-intuitive findings**)
At first glance, a skeptic might say: *"Isn't it obvious that a stronger captioner writes better captions and a cheaper reader is cheaper?"* What makes the paper genuinely novel and non-trivial are **five specific, counter-intuitive findings** that overturn common assumptions in multimodal QA:

1. **Debunking the "Text Is Easier to Reason Over" Myth (The Modality Null/Negative):** Works like IsoBench (Fu et al., 2024) showed 15–29 point gains when converting visual inputs to text, leading many video QA papers to claim that LLMs reason better over text captions than visual tokens. By introducing the **self-caption control** on the exact same frames, this paper shows that self-captioning does *not* help overall (+0.03 for Haiku, −0.09 for Sonnet, −0.12 for Qwen). Prior "captioning gains" were actually **hidden model distillation** (stronger captioner) or **hidden visual budget increases** (Gemini spends ~5,500 tokens on 5 images vs. 200 for Haiku).
2. **The Two-Clip Exception (Where Self-Captioning *Does* Help a Weak Reader):** While self-captioning doesn't help overall, on **two-clip videos** Haiku's pair accuracy jumps from near-chance on frames (0.18) to 0.53 on its *own* captions (+0.36 pair accuracy, PDF lines 327–338). Why? Because captioning each 5 s window in an isolated call prevents the weak reader from confusing ~10 small frames across two scenes, whereas describing all windows in a single call ("describe, then answer") only gains +0.09. That is a neat, non-trivial architectural insight about context isolation vs. modality.
3. **Why Token Savings Alone Don't Pay Off (`Q* = 202` vs. `Q* = 1.2–3.7`):** A common pitch for caption indexes is "text tokens are fewer than image tokens, so queries are cheaper." Section 5 shows this barely works for a fixed reader: because the fixed reader prompt/system overhead (~1,025 tokens) dominates the 10 s video's frame tokens, captions cut video tokens to 43% but total input tokens only to 81%. For Haiku reading Gemini captions, it takes **202 questions per video** to break even on cost! The index only pays off rapidly (1.2–3.7 questions) when used as a **knowledge distillation bridge to downgrade the reader** (Sonnet/Gemini captions → Haiku reader, cutting per-query cost by 23–26×).
4. **The "Caption Cap" (Reader Invariance on a Fixed Index):** On Sonnet's captions, Haiku (0.707) and Sonnet (0.710) achieve identical accuracy (difference −0.003, PDF lines 282–291), despite Sonnet costing 17× more on captions and scoring +0.185 higher on raw frames. The audit (§4.4) explains why: **61–73% of errors are caption omissions** (the fact isn't in the text) and only 5–17% are reader reasoning errors. Once the index is written, a stronger reader cannot extract facts that were omitted.
5. **Perceptual Failure vs. Pragmatic Omission on Stress Tests (Stacked vs. Reversed):** Using Table 6 and §4.5, the paper cleanly separates *why* captions fail on different temporal phenomena:
   - On **stacked videos** (split-screen speed comparison), the model *can* see which copy is faster when given the question (Sonnet 0.66 on frames; Haiku +0.22 in one-call "describe, then answer"), and even telling the captioner to "state the speed of movements" (aspect-aware, §4.5) fails to fix it. Why? Because without seeing the specific comparative question, a captioner describes the shared action ("a man jumps rope in a split-screen video") rather than contrasting the two panels.
   - On **reversed videos**, by contrast, the failure is partly perceptual (Sonnet fails 36% of the time even on frames) and partly world-prior attraction (wrong answers revert to the question-only forward prior 59–65% of the time; e.g., Gemini hallucinating salt pouring *onto* a shrinking pile in all 5 windows).

---

### E. Additional Review Criteria (Reproducibility, Formatting, Venue Readiness)
- **Reproducibility:** Exemplary. Exact prompts are in Appendix A, every deviation/amendment is in Appendix C, all 50 predicted terms vs. outcomes are in Table 7, and code/outputs/Software Heritage archive are prepared for release.
- **Formatting & ACL Style Readiness:** Currently, the paper reads partly like an internal technical report / pre-registration log ported directly from Markdown ([caption_index_draft.md](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/caption_index_draft.md)). For example:
  - The Limitations section (page 9, lines 634–672) is a raw bulleted list of telegraphic notes (`One benchmark; short clips; multiple choice only.`).
  - Internal experimental batch codes (`C1`, `C2`, `C2c`, `C3a`, `C12a, d, f`, `K1`, `K2`) leak into the main text (e.g., Limitations line 664 mentions `C1 and C2` before the reader has ever seen those codes defined in Appendix B/C!).
  - Appendix E includes a paragraph on "The reconstruction phase" (lines 1208–1222) describing an earlier abandoned/pivoted project on masking 1 s clips in 60 s videos—interesting context for why reference-similarity was rejected, but jarring unless framed clearly as a preliminary study on evaluation metrics.

---

## 3. Summary of Main Strengths & Weaknesses

### Main Strengths
1. **Causal Isolation via the Garbling Control:** Fixing the visual input between captioner and reader cleanly separates source, visual budget, captioner capability, and text modality.
2. **Gold-Standard Methodological Rigor:** Pre-registration, logged point predictions for exploratory arms (median absolute prediction error of just 0.021 across 50 terms!), clip-clustered bootstrap CIs, minimal pairs, and human-validated error audits (\(\kappa = 0.81\)).
3. **Practical Cost & Break-Even Analysis:** Moves beyond hand-waving about "efficiency" to compute exact token economics (including thinking tokens and CLI system prompt overhead) and break-even query counts \(Q^*\).
4. **Sharp Diagnostic Insights:** The contrast between two-clip gains (window isolation helps weak readers), stacked failures (pragmatic omission of relational comparisons), and reversed failures (world-prior attraction), backed by question-aware and task-aware controls.

### Main Weaknesses
1. **Mismatch Between Long-Video Motivation and Short-Video Evaluation (External Validity):**
   - The paper motivates per-window caption indexing via long-video QA systems (Cai et al., 2026; Liang et al., 2026; Chen et al., 2026), yet evaluates exclusively on TempCompass clips with a **median length of 9.7 s** (58% have 1 or 2 windows).
   - This affects **both accuracy and cost**:
     - *Cost:* On a 10 s video (2 windows), the reader's fixed prompt (~1,025 tokens) is ~2× the frame tokens (~500 tokens), which is why same-reader token savings look small (18–30%) and Haiku's self-break-even \(Q^*\) is 202. On a 10-minute video (120 windows = 24,000 frame tokens vs. ~5,000 caption tokens), the fixed 1,025-token prompt becomes negligible, and same-reader token savings would be much larger! This nuance is never discussed in §5.
     - *Accuracy:* Long-video QA requires multi-window retrieval/aggregation across dozens of irrelevant windows, whereas here the reader sees at most 1–7 windows.
2. **Single Benchmark with 54% Synthetic Stress-Test Manipulations:**
   - TempCompass is a diagnostic stress test where 54% of confirmatory videos are artificially reversed, split-screen stacked, or concatenated two-clip videos. On natural ("plain") videos (n=137), all five main arms cluster tightly between 0.80 and 0.89 (Figure 2, Table 6), and Sonnet's index loss is 0.01. While the paper is admirably transparent about this, a reviewer may question whether the headline 0–9 point accuracy trade-off holds on natural long-form benchmarks (e.g., Video-MME, EgoSchema, LongVideoBench) where split-screen speed comparisons and backwards-played videos do not exist.
3. **CLI Confounds on Closed Models (System Prompt & Uncontrolled Thinking):**
   - Running Claude Haiku and Sonnet through `claude -p` injects a ~1,025-token CLI system prompt (which inflates reader input costs in Table 3 and §5!) and leaves temperature and extended thinking uncontrolled (`effort medium`). Intriguingly, Appendix E (lines 1151–1158) notes that when the model chooses to think on Gemini captions, accuracy is *lower* (0.62–0.65 vs. 0.84–0.90 without thinking, likely because hard/ambiguous items trigger thinking). While Qwen2.5-VL-7B mitigates the accuracy concern, Qwen is not included in the cost accounting.
4. **Presentation & LaTeX Polish Issues:**
   - Empty Section 2.3 in the compiled PDF; unreadable Figure 3; telegraphic bullet-point Limitations section; undefined internal codes (`C1`, `C2`) in the main text; inline-squished Propositions in §3.

---

## 4. Concrete, Actionable Recommendations for Revision

1. **Fix Section 2.3 in [caption_index.tex](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.tex#L107-L128):**
   - Write a short paragraph (4–6 sentences) inside `\subsection{The four effects}` explicitly defining the four effects (*Source*, *Visual budget*, *Captioner*, *Modality*) and pointing to Table 2, so the heading is not orphaned at the bottom of page 2.
2. **Discuss How Video Length Scales the Cost Equation in §5:**
   - Explicitly note in §5 that on 10 s clips (mean ~2.5 windows), the ~1,025-token CLI/reader prompt overhead dominates the denominator \(c_f - c_\tau\). Show briefly (even in 2 sentences) how \(Q^*\) behaves as the number of windows \(W\) grows (since both \(C(\tau)\) and \(c_f - c_\tau\) scale linearly with \(W\) once \(W\) is large enough to dwarf the fixed prompt, \(Q^*\) converges to the per-window cost ratio!). Also report what \(c_f\) and \(c_\tau\) look like *without* the ~900-token `claude -p` CLI system prompt overhead (i.e., raw API token counts, matching Qwen's ~121-token prompt).
3. **Redesign Figure 3 ([caption_index.tex](file:///home/yoavh/code/antigravity/caption_reconstruction/docs/paper/acl2027/caption_index.tex#L314-L319)):**
   - Either span Figure 3 across both columns (`\begin{figure*}[t]`) or stack the \(Q=1\) and \(Q=10\) panels vertically with larger fonts so the labels are legible at 100% zoom.
4. **Format Section 3 (Theory) and Section 9 (Limitations) as Proper Prose:**
   - Unpack Propositions 1–3 and *Retention* in §3 into a clean bulleted list or theorem environments so they are visually distinct.
   - Convert the bulleted notes in `Limitations` into 2–3 cohesive paragraphs of prose, and replace internal log IDs (`C1 and C2` on line 664) with descriptive names ("the initial confirmatory and frame-clip runs").
5. **Clarify the "Natural vs. Stress-Test" Takeaway Early:**
   - In the Abstract/Introduction, highlight the contrast between **plain natural videos** (where a question-blind caption index retains 95–100% of frame headroom with ~0 index loss) and **synthetic temporal stress tests** (where stacked comparisons and reversals cause nearly all the index loss). That actually makes your story *stronger* for practitioners building indexes on natural videos, while keeping the diagnostic value of TempCompass crystal clear.
