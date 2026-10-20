# Draft v1 (skeleton): caption once, ask cheaply

Started 2026-10-20; v1 the same evening, after C3–C6. This is a skeleton, not prose. It shows which claims we make, what backs each one, and what is missing. Numbers come from `docs/new_direction/research_log.md` (entry names in brackets) and `results/index_qa/reports/*.txt`. **⟨PENDING: X⟩** marks a number that run X will fill. Terms follow the *Terms* block of `research_plan_caption_index.md`: source, visual budget, captioner and modality effects. CIs are 95% (98.33% for C1's P1–P3); "cl." marks a CI over clip clusters.

Target: ACL 2027 (ARR, January 2027), long paper (8 pages). **G5 is case 1** (C3a positive), so the plan says ACL main; long or short is decided at G9 (Nov 30).

---

## 0. Claims and evidence

| # | Claim (as the paper would state it) | Evidence | Status |
|---|---|---|---|
| 1 | **Who writes the caption matters.** With the same frames, a stronger model's question-blind captions make a cheap reader more accurate than the reader's own captions. | P3 +0.148 [+0.088, +0.209] (C1, pre-registered); C2a, Gemini at about matched visual tokens, +0.081 [+0.030, +0.131]; **C3a, Sonnet at identical input, +0.064 [+0.017, +0.111], cl. [+0.011, +0.119]**; on minimal pairs, C4b +0.104 [+0.017, +0.190] | **Backed**, in two families and at identical input, three ways. Smaller than predicted (C3a predicted +0.09). |
| 2 | **Text itself is not easier for the reader.** The reader's own captions of its frames are no better than the frames, and for a strong reader they are worse. | Haiku: P2 +0.030 [−0.034, +0.094]. Sonnet on its own captions: −0.088 [−0.135, −0.044], cl. [−0.140, −0.036] | **Backed.** Haiku gives a bound of ≤ +0.09; Sonnet gives a cost. |
| 2a | **The question-blind caption caps accuracy, whatever the reader.** On Sonnet's captions, Haiku and Sonnet score the same. | Haiku 0.707 vs. Sonnet 0.710 on Sonnet captions: −0.003 [−0.047, +0.040] (C3 result, unplanned) | **New, exploratory.** One captioner. On Gemini video captions Sonnet leads Haiku by 0.054, so the cap is not reached with every index. |
| 2b | …**except on two-clip videos, where captions help a weak reader** (+0.22 accuracy; +0.36 pair accuracy). | A3; C4a two-clip +0.36 [+0.16, +0.56]; C3b two-clip +0.33 [+0.20, +0.48] | **Backed.** The mechanism is open: describing in one call gives two-clip +0.09 [−0.06, +0.24] (C5a), below the self-captions by −0.13 [−0.30, +0.02] (C5b). This leans to "a separate call per window", without showing it. |
| 3 | **The index pays through the reader downgrade, not through fewer tokens.** A cheap reader on a strong captioner's captions costs ~1/25 per question and repays the captioning within 2–4 questions per video. Its accuracy cost depends on the captioner. | Q\* 1.8–3.7 (A2); the Claude chain, Sonnet captions once and Haiku answers vs. Sonnet answers: Q\* 2.3 [2.2, 2.4], accuracy **−0.091** [−0.145, −0.040]. Gemini frame captions: Q\* 3.7, accuracy −0.007 [−0.057, +0.044]. Same-reader savings: 18% (Haiku), 30% (Sonnet) | **Backed for cost.** "Close to the strong reader" holds only for Gemini frame captions (≈ 5,500 visual tokens per window). The Claude chain loses 9 points. **Reword: the index trades ≈ 0–9 points for a 20–25× lower query cost.** |
| 4 | **Question-blind captions drop comparisons between things shown at the same time** (stacked clips at two speeds), even from a strong captioner. The loss comes from not seeing the question, not from the text. | C4 stacked pairs: caption arms' pair accuracy 0.10–0.20 vs. Sonnet on frames 0.51; C4c −0.34 [−0.60, −0.08]; C4e −0.41 [−0.62, −0.23]; same-answer rate above Sonnet on frames' in 3 of 4 caption arms. C5b stacked: a description that sees the question beats question-blind self-captions by **+0.22 [+0.09, +0.37]**. | **Backed** (exploratory, written in advance in C4 and C5). Audit (C7: silent ≥ 0.70) **⟨PENDING: C7⟩** |
| 5 | **Reversed clips: question-blind captions narrate the forward event, even when the captioner can see the reversal.** | Sonnet on frames beats Haiku on Sonnet captions on reversed videos: C3c −0.19 [−0.32, −0.07]. Sonnet on its own captions −0.18 [−0.30, −0.05] vs. its frames. Sonnet captions do not report reversals more often than Haiku's: C3a reversed +0.01, C4b reversal +0.06 [−0.01, +0.13]. Gemini's do: C2a reversed +0.14. Wrong answers fall to the prior in both modalities (A3). | **Reworded and backed** (the old wording, "a stronger captioner fixes it", failed for Sonnet). Audit (C7: wrong ≥ 0.50) **⟨PENDING: C7⟩** |
| 6 | **Captions beat frames for Haiku even when the frames get 2× more tokens.** | C6: 448 px frames, 2,676 input tokens, 0.650. Captions minus 448 px: Gemini video +0.088 [+0.037, +0.138]; Sonnet +0.057 [+0.003, +0.111]; Gemini frame +0.141. C6a (448 − 224 px) +0.037 [−0.010, +0.084] | **Backed.** The gain is mostly on two-clip videos. |
| 7 | *(minor)* **Writing before answering helps a cheap reader** (+0.07), at 2× the query cost. It is not an index. | C5a +0.071, cl. [+0.029, +0.114]; $0.000415 vs. $0.000204 per question | Backed. A paragraph in §4.6. |
| — | *Dropped:* "captions lose continuous quantities (direction, speed)" | P1 −0.10 [−0.23, +0.03], not passed; aspect ≈ video type (A3) | Reported as a null in §4.5 |
| — | *Dropped:* "the captioner's knowledge" | No arm separates perceiving, recognizing and writing (D2) | We say "captioner effect" |
| — | *Dropped:* "a captioner that reports reversals more often fixes reversed clips" | C3a reversed +0.01 | Folded into claim 5 |

**Headline (case 1):** "Given exactly the reader's frames, a stronger captioner's question-blind captions make a cheap reader more accurate (+0.06 to +0.15). Turning the frames into text does not (≤ +0.09 for Haiku; −0.09 for Sonnet). A caption index lets a 20–25× cheaper reader answer at 0–9 points below the strong reader on frames, and repays itself within 2–4 questions per video. What it loses is what a question-blind captioner does not write down: the comparison between stacked copies, and reversals."

**Tension to resolve while writing:** claims 1 and 2 say the captioner adds usable information. Claims 2a, 4 and 5 say question-blindness caps it. Both hold: the captioner helps a reader below the cap (Haiku) and hurts one above it (Sonnet). Fig. 3 could show this as reader strength on frames vs. accuracy on captions.

---

## Abstract (bullets)

- Setting: captions as a one-time, question-independent index of short videos, read later by an LLM.
- Question: when and why does a reader answer better from the index than from frames, and what does it cost?
- Design: an information control. The captioner sees exactly the reader's frames, which splits the caption effect into source, visual budget, captioner and modality effects. Pre-registered on 297 TempCompass questions, with exploratory follow-ups each logged with predictions before running.
- Findings:
  - The captioner effect is +0.15 (P3), +0.08 at matched visual tokens (C2a), and +0.06 within one model family at identical input (C3a).
  - The modality effect is ≤ +0.09 for Haiku and −0.09 for Sonnet.
  - The index pays through the reader downgrade: Q\* ≈ 2–4 questions per video, at 0–9 points of accuracy.
- Where captions fail: minimal pairs show that question-blind captions lose the comparison in stacked clips (pair accuracy 0.10–0.20 vs. 0.51) and narrate reversed clips forward. A question-aware description recovers the stacked loss (+0.22). Audit ⟨PENDING: C7⟩.

---

## 1. Introduction (bullets)

- **Motivation.** Captioning a video is paid once; questions come later and repeatedly. Caption-then-reason pipelines are widely used (LLoVi, VideoTree, SiLVR), and "caption once, reuse" is now published for long video (CFD, CapMem, Sprout). Two questions are open:
  1. When the captions win, **why**? Did the captioner see more, spend more visual tokens, write differently, or is text easier to read than pixels?
  2. **Does it pay**, counting the captioning and the reader's full per-question cost?
- **Approach.** The information control (Prop. 1): give the captioner exactly the frames the reader sees. Then any caption advantage is usable information added by the captioner or by the modality, not information about the video.
- **Contributions:**
  1. A decomposition of the caption effect into source, visual budget, captioner and modality effects, with arms that change one thing each, in two model families (§2.3, Table 2).
  2. A pre-registered test plus a within-family replication at identical input: the captioner effect is positive, and the modality effect is ≤ +0.09 (and negative for a strong reader) (§4.1–4.2).
  3. A cost account per question (input, output and thinking tokens, list-price dollars) and the break-even Q\* (§5).
  4. What question-blind captions lose, shown with minimal pairs, a question-aware control and a caption audit (§4.3–4.4).
- **Scope:** short videos (≈ 10 s), multiple choice, one benchmark.

---

## 2. Setup

### 2.1 Task and data
- **TempCompass** multiple choice (CC BY-NC). Aspects: action, speed, direction, order, attribute change. Median video 10 s.
- **Development set:** 100 questions (K0–K2), used to set the hypotheses.
- **Confirmatory set:** 300 questions sampled (60 per aspect, one per video, on videos the development set does not use), 297 after Amendment 2; pre-registered (C1).
- **Video types** (A3): plain (137), reversed (74), stacked (41), two-clip (45). They nearly determine the aspect. There are 156 clusters of videos that share a filmed clip.
- **Minimal pairs** (C4): 193 confirmatory questions, each re-asked with the same wording on a sibling video whose correct option differs (107 reversal, 41 stacked, 45 two-clip). Chance pair accuracy is 0.093.
- **Windows:** 5 s; frames at 1 fps (t = 5p + k), 224 px long edge (40 Claude image tokens each).

### 2.2 Arms

| Arm | Reader sees | Visual tokens for the reader / captioner per 5 s window |
|---|---|---|
| Question only | question and options | 0 |
| 1 frame | the middle frame of each window | 40 |
| 5 frames | all 5 frames of each window | 200 |
| 5 frames, 448 px | the same frames at 448 px | 720 |
| Describe, then answer | 5 frames; writes one sentence per window, then answers | 200 |
| Self-captions | the reader's own isolated caption of each window's 5 frames | — / 200 |
| Sonnet captions | Sonnet's isolated caption of the same 5 frames | — / 200 |
| Gemini frame-clip captions | Gemini's caption of the same 5 frames, played as a 1 fps clip | — / ≈ 280–330 |
| Gemini frame captions | Gemini's caption of the same 5 frames, as images | — / ≈ 5,500 |
| Gemini video captions | Gemini's caption of the video window | — / ≈ 330 |

- **Readers:** Claude Haiku 5.5 and Sonnet 5.5 (pinned IDs, effort medium), through `claude -p` with no tools, isolated per call.
- **Captioners:** Haiku (self), Sonnet, Gemini 3.8 Flash. Every caption is question-blind and window-isolated, and all use one prompt.

### 2.3 The four effects (Haiku reads)

| Effect | Arms compared | Claude chain | Gemini chain |
|---|---|---|---|
| Source | video captions − frame-clip captions | 0 by design | C2c +0.013 [−0.030, +0.061] |
| Visual budget | frame captions − frame-clip captions | 0 by design | C2b +0.067 [+0.030, +0.108] |
| Captioner | other captioner − self-captions, same pixels | C3a +0.064 [+0.017, +0.111] | C2a +0.081 [+0.030, +0.131] |
| Modality | self-captions − 5 frames | P2 +0.030 [−0.034, +0.094] | (same) |
| *Total* | other captioner − 5 frames | C3b +0.094 [+0.044, +0.148] | P3 + P2; frame captions − 5 frames +0.178 |

- C3d (Sonnet − Gemini frame-clip captions) is −0.017 [−0.071, +0.037]. At about equal visual tokens the two families' captioners are equally useful to Haiku.

### 2.4 Statistics and pre-registration
- C1 was pre-registered: P1–P3 at 98.33% CIs (Bonferroni over three); secondary estimates at 95%.
- C2–C7 are exploratory. Each was logged with predictions before its first call, and each analysis was run once.
- Paired bootstrap over questions; clip-clustered bootstrap as a check (the CIs agree everywhere; report both or state it).
- **Many exploratory terms** (≈ 30 across C2–C6). The paper should say which were predicted directionally and report all of them in the appendix.
- **Missing:** a public, timestamped archive of the pre-registration commit (plan §5 step 9; G11).

---

## 3. Theory (short)

- **Prop. 1 (garbling bound).** A caption made from the reader's own frames cannot add information about the video, for an unbounded reader. For a fixed, bounded reader, it can add *usable* information (Xu et al. 2020), and that is what the captioner effect measures.
- **Corollary.** The total caption effect = source + captioner + modality, plus a visual budget term when the visual tokens differ.
- **Prop. 2 (break-even).** Q\* = C(τ) / (c_f − c_τ): the captioning cost over the per-question saving. With a reader downgrade, c_τ is the cheap reader's cost.
- **Prop. 4 (price of being question-blind)** is now used. C5b stacked (+0.22) and claim 4 are its evidence: a question-aware description recovers what the question-blind caption omits.
- **Possible addition:** the cap of claim 2a. The accuracy from a question-blind caption is bounded by what the caption holds, so readers above that bound lose and readers below it gain. This is informal; decide whether it needs a proposition.
- Prop. 3 (routing headroom) is not used.

---

## 4. Results

### 4.1 Table 1: accuracy and cost per question (confirmatory set, 297 questions)

| Reader, input | Acc. | Input tok. (median) | $ / question | Index $ / video |
|---|---|---|---|---|
| Haiku, question only | 0.411 | 1,025 | 0.000109 | — |
| Haiku, 1 frame | 0.582 | 1,189 | 0.000146 | — |
| Haiku, 5 frames | 0.613 | 1,520 | 0.000204 | — |
| Haiku, 5 frames 448 px | 0.650 | 2,676 | 0.000339 | — |
| Haiku, describe then answer | 0.684 | 1,579 | 0.000415 | — |
| Haiku, self-captions | 0.643 | 1,283 | 0.000189 | 0.00058 |
| Haiku, Sonnet captions | 0.707 | 1,280 | 0.000181 | 0.00910 |
| Haiku, Gemini frame-clip captions | 0.724 | 1,232 | 0.000167 | 0.00760 |
| Haiku, Gemini frame captions | 0.791 | 1,229 | 0.000167 | 0.01461 |
| Haiku, Gemini video captions | 0.737 | 1,236 | 0.000168 | 0.00726 |
| Sonnet, question only | ⟨gap: not run⟩ | | | — |
| Sonnet, 5 frames | 0.798 | 1,522 | 0.004112 | — |
| Sonnet, Sonnet captions | 0.710 | 1,282 | 0.003089 | 0.00910 |
| Sonnet, Gemini video captions | 0.791 | 1,238 | 0.002868 | 0.00726 |

### 4.2 Table 2: the decomposition
- Both chains are done (§2.3).
- **Text for the table:**
  - P3 passed at 2.5× the predicted size.
  - About half of it survives at matched visual tokens (C2a, Gemini), and a similar amount within Claude at identical input (C3a +0.064). The rest of P3 is visual budget (C2b). The source effect is ≈ 0.
  - Predictions on record: C3a +0.09, C3b +0.12, C3c −0.07. All results fall inside their CIs.
- **Sonnet on its own captions** (−0.088) is the strong-reader counterpart of P2: the modality effect changes sign with reader strength.

### 4.3 Where captions lose: by video type (A3) and minimal pairs (C4)
- **Figure 2:** accuracy by video type for 5 arms (`docs/paper/figures/caption_index_fig2_types.png`).
- **Table 3** (C4, pair accuracy; chance 0.093):

| Arm | All | Reversal (107) | Stacked (41) | Two-clip (45) | Same answer |
|---|---|---|---|---|---|
| Haiku, 5 frames | 0.233 | 0.28 | 0.17 | 0.18 | 0.54 |
| Haiku, self-captions | 0.326 | 0.33 | 0.10 | 0.53 | 0.42 |
| Haiku, Sonnet captions | 0.430 | 0.38 | 0.17 | 0.78 | 0.43 |
| Haiku, Gemini video captions | 0.482 | 0.47 | 0.20 | 0.78 | 0.37 |
| Sonnet, 5 frames | 0.606 | 0.56 | 0.51 | 0.80 | 0.28 |
| Sonnet, Sonnet captions | 0.430 | 0.40 | 0.10 | 0.80 | 0.39 |

- **Stacked:** every caption arm is near pair chance, and Sonnet on frames is at 0.51. This is claim 4.
- **Reversal:** caption arms reach 0.33–0.47; Sonnet on frames reaches 0.56. Sonnet loses 0.16 on its own captions. This is claim 5.
- **Two-clip:** Haiku on frames is near chance (0.18) and captions reach 0.53–0.80. This is claim 2b.
- Accuracy by video type for C3 (Amendment 1): §0 and the log's C3 result. Use them in the appendix.

### 4.4 Why caption arms fail: the audit (C7)
- **Table 4:** each reader arm's wrong answers, split into captions state the answer (reader error), state a wrong option (misstatement), or are silent (omission); by video type. ⟨PENDING: C7, 442 items, runs after the 5-hour quota reset at 22:40 UTC⟩
- Judge agreement with the user's 40 hand labels (κ). ⟨PENDING: user⟩
- Two quoted examples: the reversed salt pile (forward narration) and the stacked jump rope (no comparison) (A3).
- The first judge label checked by hand is right: a stacked sunrise, where the caption mentions one sun and no speeds, was labelled "silent".

### 4.5 Aspects (P1, null) and thinking tokens (A1)
- P1 is not passed. Aspect is nearly the same as video type, so it moves to the appendix as a table.
- **Thinking** marks wrong answers on captions (no thinking 0.84–0.90 vs. thinking 0.62–0.65), but not for Haiku on frames. Sonnet thinks 3.3× less on captions. One paragraph, exploratory.

### 4.6 Describing first (C5) and frame size (C6)
- **C5:** describing the windows before answering, in the same call, gives +0.071, cl. [+0.029, +0.114], mostly on reversed and stacked videos (+0.12 each). On stacked videos it beats the question-blind self-captions by +0.22: Prop. 4. It costs 2× the frames arm per question, and it is not an index.
- **C6:** 448 px frames give +0.037 [−0.010, +0.084] over 224 px at 1.76× the input tokens. Captions still beat them (claim 6).

---

## 5. Cost and break-even

- **Figure 1** (`docs/paper/figures/caption_index_fig1_cost.png`): accuracy against list-price $ per question (log x), one point per arm, with the index cost amortized at Q = 1, 10 and ∞.
- **Same reader:** captions save 18% (Haiku) and 30% (Sonnet) per question. The fixed prompt (≈ 1,025 tokens) and thinking dominate.
- **Reader downgrade** (Haiku on captions vs. Sonnet on frames):

| Index | Q\* (questions per video) | Accuracy difference |
|---|---|---|
| Gemini video captions | 1.8 [1.7, 2.0] | −0.061 [−0.111, −0.010] |
| Gemini frame-clip captions | 1.9 [1.8, 2.1] | −0.074 [−0.125, −0.024] |
| Sonnet captions | 2.3 [2.2, 2.4] | −0.091 [−0.145, −0.040] |
| Gemini frame captions | 3.7 [3.5, 4.0] | −0.007 [−0.057, +0.044] |

- **Within one reader:**
  - Sonnet on its own captions repays after Q\* 8.9 [7.8, 10.2] questions, but loses 0.088.
  - Haiku on Sonnet captions vs. Haiku on 448 px frames: Q\* 57.5, and it gains +0.057.
- Latency does not favour captions (+0.1–1.1 s per call; CLI wall clock, uncontrolled).
- Thinking is 2–5% of reader tokens.

---

## 6. Related work (bullets; only Q5 of the literature pass is done: log N2a)

- Caption-then-reason for video QA: LLoVi (the captioner matters: oracle captions +10.8), VideoTree, SiLVR, Vamos (text competitive with visual embeddings; verified).
- Caption indexes and memories: CFD, CapMem, Sprout, VidCtx (question-aware captions +1–2 points; compare with C5b stacked +0.22).
- **A captioner in front of a fixed reader (closest to claim 1):** Prism (images; a 2B captioner + GPT-3.5 ≈ 10× larger VLMs), MetaCaptioner, CapRL / CapRL++ (caption utility as the RL reward). **What none has:** a self-caption control, matched visual tokens, or the reader's own view of the frames (N2a). That is our gap; word claim 1 as the controlled decomposition.
- Text vs. pixels for the same content: IsoBench (text +15–29 points when the text is exact), TokenSwap. Our modality effect is ≤ +0.09, and negative for the strong reader.
- Visual token budgets: LoHi, VEI.
- Caption utility judged by QA: CaptionQA, CapQuiz, ViSIL.
- Temporal benchmarks: TempCompass, plus T3 (text-to-video temporal transfer; related at most, checked 2026-10-19). Reversed-video shortcuts and the forward prior (find the citation).
- **Still missing:** Q6 (break-even with a reader downgrade), Q7 (new close work since Oct 15), and the rest of G6.

---

## 7. Limitations (bullets)

- One benchmark; short clips; multiple choice only.
- Closed readers run through a CLI: its system prompt is included, and there is no temperature control. Mitigations: pinned IDs, isolation checked on every call, released outputs.
- Claude reading Claude (C3). Mitigation: the Gemini chain.
- Development → confirmatory drop of Haiku on frames (0.70 → 0.61); about half of it is the mix of video types.
- Questions share filmed clips. Mitigation: clustered CIs.
- Thinking is decided per call by the model (effort medium), so it is observed, not controlled.
- **Question-blind captions drop what the question needs and the captioner did not think to say** (stacked comparisons, reversals). This is a limit of question-blind indexes, not only of our captioners.
- Claims 2a, 2b, 4, 5 and 7 are exploratory, even though each was predicted in writing before its run.

---

## Appendix (list)

- A. Prompts: caption, reader, describe-then-answer, audit.
- B. The development phase (K0–K2) and how it set the hypotheses; predictions against results (C1–C7).
- C. Amendments, deviations, and the term mapping (D2).
- D. Tables by aspect and by video type; clustered CIs.
- E. The minimal-pair construction (C4), with pair results by group.
- F. Release: code, captions, raw outputs; TempCompass licence (CC BY-NC).

---

## Gaps found while outlining (to decide)

| Gap | Why it matters | Fix | Cost |
|---|---|---|---|
| **C7 audit** | Claims 4–5 need the per-error labels; Table 4 | `audit.py run`, queued after the 22:40 UTC reset; then `report`, `sheet` | ≈ 442 Sonnet calls |
| **40 hand labels** | The judge's validity (κ ≥ 0.6) | `results/index_qa/audit/hand_labels.csv` | user, 1–2 h |
| Sonnet question-only arm | Table 1 has no Sonnet floor; Sonnet's prior on reversed videos (claim 5) | 297 Sonnet calls, after a log amendment | ≈ 4 points of the 7-day quota |
| **Claim 3's accuracy depends on the captioner's visual budget** | Only Gemini frame captions (≈ 5,500 visual tokens per window) match Sonnet on frames; the Claude chain at 200 loses 9 points | Optional: Sonnet captions from 448 px frames (≈ 880 windows, plus 297 Haiku reads). It tests whether the Claude chain's gap is visual budget | ≈ 10–15 points of the 7-day quota; decide |
| Figure for the reader-strength tension (claim 2a) | Explains why one index helps Haiku and hurts Sonnet | plot: x = reader on frames, y = reader on each index | none |
| Colour consistency | Fig. 1 uses orange for Sonnet; Fig. 2 uses orange for Haiku self-captions | Fig. 2: a hue per arm that avoids the reader colours, or the reader colours with two shades | none |
| Literature: Q6, Q7 and the rest of G6 open (Q5 done: close, not scooped) | Novelty of claims 1 and 3; a citation for reversed-video priors | `final_search_brief.md` | this session's quota |
| Public pre-registration archive | Reviewers can't verify local commits | an anonymous repository or OSF, before submission | user |
| Open-weights reader | "All readers are closed" | G5 is case 1, so this is optional for ACL main; decide at G9 | Kaggle |
