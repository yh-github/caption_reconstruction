# Draft v1 (skeleton): caption once, ask cheaply

Started 2026-10-20; v1 the same evening, after C3–C6; v1.1 on 2026-10-21, after C7–C10. This is a skeleton, not prose. It shows which claims we make, what backs each one, and what is missing. Numbers come from `docs/new_direction/research_log.md` (entry names in brackets) and `results/index_qa/reports/*.txt`. **⟨PENDING: X⟩** marks a number that run X will fill. Terms follow the *Terms* block of `research_plan_caption_index.md`: source, visual budget, captioner and modality effects. CIs are 95% (98.33% for C1's P1–P3); "cl." marks a CI over clip clusters.

Target: ACL 2027 (ARR, January 2027), long paper (8 pages). **G5 is case 1** (C3a positive), so the plan says ACL main; long or short is decided at G9 (Nov 30).

---

## 0. Claims and evidence

| # | Claim (as the paper would state it) | Evidence | Status |
|---|---|---|---|
| 1 | **Who writes the caption matters.** With the same frames, a stronger model's question-blind captions make a cheap reader more accurate than the reader's own captions. | P3 +0.148 [+0.088, +0.209] (C1, pre-registered); C2a, Gemini at about matched visual tokens, +0.081 [+0.030, +0.131]; **C3a, Sonnet at identical input, +0.064 [+0.017, +0.111], cl. [+0.011, +0.119]**; on minimal pairs, C4b +0.104 [+0.017, +0.190] | **Backed**, in two families and at identical input, three ways. Smaller than predicted (C3a predicted +0.09). |
| 2 | **Text itself is not easier for the reader.** The reader's own captions of its frames are no better than the frames, and for a strong reader they are worse. | Haiku: P2 +0.030 [−0.034, +0.094]. Sonnet on its own captions: −0.088 [−0.135, −0.044], cl. [−0.140, −0.036] | **Backed.** Haiku gives a bound of ≤ +0.09; Sonnet gives a cost. |
| 2a | **The question-blind caption caps accuracy, whatever the reader.** On Sonnet's captions, Haiku and Sonnet score the same. | Haiku 0.707 vs. Sonnet 0.710 on Sonnet captions: −0.003 [−0.047, +0.040] (C3 result, unplanned) | **New, exploratory.** One captioner. On Gemini video captions Sonnet leads Haiku by 0.054, so the cap is not reached with every index. |
| 2b | …**except on two-clip videos, where captions help a weak reader** (+0.22 accuracy; +0.36 pair accuracy). | A3; C4a two-clip +0.36 [+0.16, +0.56]; C3b two-clip +0.33 [+0.20, +0.48] | **Backed.** The mechanism is open: describing in one call gives two-clip +0.09 [−0.06, +0.24] (C5a), below the self-captions by −0.13 [−0.30, +0.02] (C5b). This leans to "a separate call per window", without showing it. |
| 3 | **The index pays through the reader downgrade, not through fewer tokens.** A cheap reader on a strong captioner's captions costs ~1/25 per question and repays the captioning within 2–4 questions per video. Its accuracy cost depends on the captioner. | Q\* 1.8–3.7 (A2); the Claude chain, Sonnet captions once and Haiku answers vs. Sonnet answers: Q\* 2.3 [2.2, 2.4], accuracy **−0.091** [−0.145, −0.040]. Gemini frame captions: Q\* 3.7, accuracy −0.007 [−0.057, +0.044]. Same-reader savings: 18% (Haiku), 30% (Sonnet). **C9:** Sonnet captioning 448 px frames (720 visual tokens per window) does not close the Claude gap: C9a +0.003 [−0.037, +0.047]; Haiku on them vs. Sonnet on frames −0.088 [−0.138, −0.037], Q\* 3.0 | **Backed for cost.** "Close to the strong reader" holds only for Gemini frame captions (≈ 5,500 visual tokens per window). The Claude chain loses 9 points, and C9 shows the loss is not the captioner's 224 px frames: it sits on reversed (−0.18) and stacked (−0.24) videos, as in claims 4–5. **Reword: the index trades ≈ 0–9 points for a 20–25× lower query cost.** |
| 3a | *(robustness, cost)* **Splitting short videos into 5 s windows costs nothing; one caption per video halves the index cost.** | C10a +0.013 [−0.027, +0.057] (Haiku), C10b +0.010 [−0.030, +0.051] (Sonnet); one-window control ≈ 0. Whole-video index $0.0049 vs. $0.0091 per video; Q\* 1.2 [1.2, 1.3] vs. Sonnet on frames, at −0.077 | **Backed** (exploratory, predicted +0.00). By type, reversed +0.12 for Haiku, and stacked leans negative; descriptive. |
| 4 | **Question-blind captions drop comparisons between things shown at the same time** (stacked clips at two speeds), even from a strong captioner. The loss comes from not seeing the question, not from the text. | C4 stacked pairs: caption arms' pair accuracy 0.10–0.20 vs. Sonnet on frames 0.51; C4c −0.34 [−0.60, −0.08]; C4e −0.41 [−0.62, −0.23]; same-answer rate above Sonnet on frames' in 3 of 4 caption arms. C5b stacked: a description that sees the question beats question-blind self-captions by **+0.22 [+0.09, +0.37]**. | **Backed** (exploratory, written in advance in C4, C5 and C7). The audit (C7) labels 0.93–1.00 of stacked failures "silent" in every arm (predicted ≥ 0.70). Judge vs. hand labels κ = 0.81. |
| 5 | **Reversed clips: question-blind captions narrate the forward event, even when the captioner can see the reversal.** | Sonnet on frames beats Haiku on Sonnet captions on reversed videos: C3c −0.19 [−0.32, −0.07]. Sonnet on its own captions −0.18 [−0.30, −0.05] vs. its frames. Sonnet captions do not report reversals more often than Haiku's: C3a reversed +0.01, C4b reversal +0.06 [−0.01, +0.13]. Gemini's do: C2a reversed +0.14. Wrong answers fall to the prior in both modalities (A3, Haiku). **C8, Sonnet:** question-only accuracy on reversed videos 0.18 (Haiku 0.16); 0.59 of its wrong answers on frames and 0.65 on its own captions equal its question-only answer (uniform 0.43). | **Reworded; partly backed** (the old wording, "a stronger captioner fixes it", failed for Sonnet). The audit finds forward narration ("wrong") in 0.28–0.45 of reversed failures, below the predicted ≥ 0.50; the rest are "silent", where the captions do not commit to a direction. Say "narrate forward or don't commit". The fall-back to the forward prior now holds for both readers (C8). |
| 6 | **Captions beat frames for Haiku even when the frames get 2× more tokens.** | C6: 448 px frames, 2,676 input tokens, 0.650. Captions minus 448 px: Gemini video +0.088 [+0.037, +0.138]; Sonnet +0.057 [+0.003, +0.111]; Gemini frame +0.141. C6a (448 − 224 px) +0.037 [−0.010, +0.084] | **Backed.** The gain is mostly on two-clip videos. |
| 7 | **Caption arms fail mainly by omission.** Most wrong answers come from captions that don't decide the question; few come from the reader. | C7: silent 0.61–0.73 of each caption arm's wrong answers, misstatement 0.15–0.28, reader error 0.05–0.17 (predicted reader error 0.15–0.25) | **Backed by the judge**; κ = 0.81 against 40 hand labels, where every disagreement is a hand "silent", so omission is if anything undercounted. Supports claim 2a (the cap). |
| 8 | *(minor)* **Writing before answering helps a cheap reader** (+0.07), at 2× the query cost. It is not an index. | C5a +0.071, cl. [+0.029, +0.114]; $0.000415 vs. $0.000204 per question | Backed. A paragraph in §4.6. |
| — | *Dropped:* "captions lose continuous quantities (direction, speed)" | P1 −0.10 [−0.23, +0.03], not passed; aspect ≈ video type (A3) | Reported as a null in §4.5 |
| — | *Dropped:* "the captioner's knowledge" | No arm separates perceiving, recognizing and writing (D2) | We say "captioner effect" |
| — | *Dropped:* "a captioner that reports reversals more often fixes reversed clips" | C3a reversed +0.01 | Folded into claim 5 |

**Headline (case 1):** "Given exactly the reader's frames, a stronger captioner's question-blind captions make a cheap reader more accurate (+0.06 to +0.15). Turning the frames into text does not (≤ +0.09 for Haiku; −0.09 for Sonnet). A caption index lets a 20–25× cheaper reader answer at 0–9 points below the strong reader on frames, and repays itself within 2–4 questions per video. What it loses is what a question-blind captioner does not write down: the comparison between stacked copies, and reversals."

**Tension to resolve while writing:** claims 1 and 2 say the captioner adds usable information. Claims 2a, 4 and 5 say question-blindness caps it. Both hold: the captioner helps a reader below the cap (Haiku) and hurts one above it (Sonnet). **Fig. 3** (`docs/paper/figures/caption_index_fig3_readers.png`) shows this: from Haiku to Sonnet, frames rise 0.61 → 0.80 and cross both indexes; Sonnet captions stay at 0.71; Gemini video captions rise 0.74 → 0.79. The CIs in Fig. 3 are per arm; the paired differences (claim 2a, C3c) are tighter.

---

## Abstract (bullets)

- Setting: captions as a one-time, question-independent index of short videos, read later by an LLM.
- Question: when and why does a reader answer better from the index than from frames, and what does it cost?
- Design: an information control. The captioner sees exactly the reader's frames, which splits the caption effect into source, visual budget, captioner and modality effects. Pre-registered on 297 TempCompass questions, with exploratory follow-ups each logged with predictions before running.
- Findings:
  - The captioner effect is +0.15 (P3), +0.08 at matched visual tokens (C2a), and +0.06 within one model family at identical input (C3a).
  - The modality effect is ≤ +0.09 for Haiku and −0.09 for Sonnet.
  - The index pays through the reader downgrade: Q\* ≈ 2–4 questions per video, at 0–9 points of accuracy.
- Where captions fail: minimal pairs show that question-blind captions lose the comparison in stacked clips (pair accuracy 0.10–0.20 vs. 0.51) and narrate reversed clips forward. A question-aware description recovers the stacked loss (+0.22). An audit of the wrong answers finds that 61–73% come from captions that are silent on the question, and only 5–17% from the reader.

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
  - Videos are short: median 9.7 s, 1–7 windows (1: 16%, 2: 42%).
  - The window length was checked: one caption per video from the same frames does no better (C10, claim 3a).
  - The per-window design is kept because it is what scales to long video.

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
| Sonnet captions, 448 px | the same, from the 448 px frames (C9) | — / 720 |
| Sonnet whole-video captions | one Sonnet caption of all of a video's frames, 45 words per window (C10) | — / 200 |
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
- **Visual budget within Claude (C9a):** Sonnet's captions of 448 px frames − of 224 px frames, +0.003 [−0.037, +0.047]. Unlike Gemini's (C2b +0.067), Claude's captioner gains nothing from 3.6× the visual tokens. The two budget changes differ in size (200 → 720 vs. ≈ 300 → 5,500), so say "in this range".

### 2.4 Statistics and pre-registration
- C1 was pre-registered: P1–P3 at 98.33% CIs (Bonferroni over three); secondary estimates at 95%.
- C2–C7 are exploratory. Each was logged with predictions before its first call, and each analysis was run once.
- Paired bootstrap over questions; clip-clustered bootstrap as a check (the CIs agree everywhere; report both or state it).
- **Many exploratory terms** (≈ 30 across C2–C6). The paper should say which were predicted directionally and report all of them in the appendix.
- **Missing:** a public, timestamped archive of the pre-registration commit (plan §5 step 9; G11).

---

## 3. Theory (short)

- **Prop. 1 (garbling bound).** A caption made from the reader's own frames cannot add information about the video, for an unbounded reader. For a fixed, bounded reader, it can add *usable* information (Xu et al. 2020), and that is what the captioner effect measures.
  - **Retention**, (captions − question only) / (frames − question only), is the share of the frames' gain the index keeps. For Sonnet (C8): Gemini video captions 0.98 [0.87, 1.11]; its own captions 0.77 [0.66, 0.88]. For Haiku, every index's point estimate exceeds 1 because Haiku on frames is below what the captions hold (claim 2a). The CIs exclude 1 for every index but its own captions (D3, over clip clusters):
    - self-captions 1.15 [0.91, 1.46];
    - Sonnet captions 1.47 [1.18, 1.95];
    - Gemini video captions 1.62 [1.33, 2.10];
    - Gemini frame captions 1.88 [1.53, 2.51].

    The variants (appendix): Haiku retains 1.48 on Sonnet's 448 px captions, 1.53 on its whole-video captions and 1.55 on Gemini frame-clip captions. Sonnet retains 0.82 and 0.80 on its own 448 px and whole-video captions.
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
| Haiku, Sonnet captions 448 px | 0.710 | 1,279 | 0.000180 | 0.01167 |
| Haiku, Sonnet whole-video captions | 0.721 | 1,194 | 0.000156 | 0.00492 |
| Sonnet, question only | 0.414 | 1,027 | 0.002182 | — |
| Sonnet, 5 frames | 0.798 | 1,522 | 0.004112 | — |
| Sonnet, Sonnet captions | 0.710 | 1,282 | 0.003089 | 0.00910 |
| Sonnet, Sonnet captions 448 px | 0.731 | 1,281 | 0.003103 | 0.01167 |
| Sonnet, Sonnet whole-video captions | 0.721 | 1,196 | 0.002849 | 0.00492 |
| Sonnet, Gemini video captions | 0.791 | 1,238 | 0.002868 | 0.00726 |

### 4.2 Table 2: the decomposition
- Both chains are done (§2.3).
- **Text for the table:**
  - P3 passed at 2.5× the predicted size.
  - About half of it survives at matched visual tokens (C2a, Gemini), and a similar amount within Claude at identical input (C3a +0.064). The rest of P3 is visual budget (C2b). The source effect is ≈ 0.
  - Predictions on record: C3a +0.09, C3b +0.12, C3c −0.07. All results fall inside their CIs.
- **Sonnet on its own captions** (−0.088) is the strong-reader counterpart of P2: the modality effect changes sign with reader strength. From 448 px frames it is −0.067 [−0.114, −0.024] (C9c), with C9e +0.020 [−0.024, +0.064].
- **The two readers share the question-only floor** (C8a +0.003 [−0.061, +0.064]), so the gaps between them on frames (+0.185) and captions are all from reading the input.

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
- **Table 4** (C7; Sonnet judge, 442 items; share of each reader arm's wrong answers):

| Reader arm | Reader error | Misstatement | Omission | n wrong |
|---|---|---|---|---|
| Haiku, self-captions | 0.08 | 0.28 | 0.63 | 106 |
| Haiku, Sonnet captions | 0.13 | 0.26 | 0.61 | 87 |
| Haiku, Gemini frame-clip captions | 0.17 | 0.15 | 0.68 | 82 |
| Haiku, Gemini frame captions | 0.10 | 0.18 | 0.73 | 62 |
| Haiku, Gemini video captions | 0.10 | 0.21 | 0.69 | 78 |
| Sonnet, Gemini video captions | 0.05 | 0.24 | 0.71 | 62 |
| Sonnet, Sonnet captions | 0.08 | 0.26 | 0.66 | 86 |

- **By type** (appendix table):
  - stacked: omission 0.93–1.00;
  - reversed: misstatement 0.28–0.45, omission 0.45–0.62;
  - two-clip: few errors, mostly reader errors.
- **Judge agreement with the user's 40 hand labels: 0.88, κ = 0.81 [0.64, 0.96]** (bootstrap over items; rule set in C7: below 0.6, Table 4 would be replaced by the hand-labelled sample).
  - "Correct" and "wrong" agree on all 22 items.
  - All 5 disagreements are hand "silent" that the judge decided (3 "correct", 2 "wrong"), so Table 4's omission shares are, if anything, low.
  - Method note: the labels were blind to the judge, but made in a session where Claude (Opus) pointed out caption details on 6 items. With the user's first-instinct labels κ is also 0.81 (log C7, judge check).
- Two quoted examples: the reversed salt pile (forward narration) and the stacked jump rope (no comparison) (A3).

### 4.5 Aspects (P1, null) and thinking tokens (A1)
- P1 is not passed. Aspect is nearly the same as video type, so it moves to the appendix as a table.
- **Thinking** marks wrong answers on captions (no thinking 0.84–0.90 vs. thinking 0.62–0.65), but not for Haiku on frames. Sonnet thinks 3.3× less on captions. One paragraph, exploratory.

### 4.6 Describing first (C5) and frame size (C6)
- **C5:** describing the windows before answering, in the same call, gives +0.071, cl. [+0.029, +0.114], mostly on reversed and stacked videos (+0.12 each). On stacked videos it beats the question-blind self-captions by +0.22: Prop. 4. It costs 2× the frames arm per question, and it is not an index.
- **C6:** 448 px frames give +0.037 [−0.010, +0.084] over 224 px at 1.76× the input tokens. Captions still beat them (claim 6).

### 4.7 Window length and caption resampling (C10)
- **Splitting:** one caption per video vs. one per window, with the same frames and the summed length limit: +0.013 (Haiku), +0.010 (Sonnet), both CIs include 0 (claim 3a).
- **Resampling noise:** on one-window videos the captioning call is identical, yet none of the 48 captions repeats, and the reader's answer changes on 10% (Haiku) to 17% (Sonnet) of those questions.
  - That is as often as removing the split changes it (17% on ≥ 2-window videos).
  - Per question, a caption is one sample, so caption-level comparisons need many questions. This is also a point for the discussion of why caption utility is hard to measure.
- Figure 1 leaves out the 448 px and whole-video variants (Table 1 has them).

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
| Sonnet captions, 448 px | 3.0 [2.8, 3.1] | −0.088 [−0.138, −0.037] |
| Sonnet whole-video captions | 1.2 [1.2, 1.3] | −0.077 [−0.128, −0.027] |
| Gemini frame captions | 3.7 [3.5, 4.0] | −0.007 [−0.057, +0.044] |

- **Within one reader:**
  - Sonnet on its own captions repays after Q\* 8.9 [7.8, 10.2] questions, but loses 0.088.
  - Haiku on Sonnet captions vs. Haiku on 448 px frames: Q\* 57.5, and it gains +0.057.
- Latency does not favour captions (+0.1–1.1 s per call; CLI wall clock, uncontrolled).
- Thinking is 2–5% of reader tokens.

---

## 6. Related work (bullets; literature pass done, log N2 and `related_work_caption_index.md` §10: no scoop on Q1–Q7)

- Caption-then-reason for video QA: LLoVi (the captioner matters: oracle captions +10.8), VideoTree, SiLVR, Vamos (text competitive with visual embeddings; verified).
- Caption indexes and memories: CFD, CapMem, Sprout, VidCtx (question-aware captions +1–2 points; compare with C5b stacked +0.22).
- **A captioner in front of a fixed reader (closest to claim 1):** Prism (images; a 2B captioner + GPT-3.5 ≈ 10× larger VLMs), MetaCaptioner, CapRL / CapRL++ (caption utility as the RL reward). **What none has:** a self-caption control, matched visual tokens, or the reader's own view of the frames (N2a). That is our gap; word claim 1 as the controlled decomposition.
- Text vs. pixels for the same content: IsoBench (text +15–29 points when the text is exact), TokenSwap. Our modality effect is ≤ +0.09, and negative for the strong reader.
- **A model reading its own description (closest to claims 2 and 2a; images only):**
  - Caption This, Reason That (NeurIPS 2025): own captions instead of the image give a weak 7B model +22 to +29 on synthetic location tasks, and GPT-4o +8 (not significant).
  - GEASS: the model's own caption added to the image costs ≈ 10 points on HallusionBench.
  - Seeing and Solving: two-pass extract→solve beats direct answering on charts, but a one-call "state first" prompt does not (contrast with C5).
  - **Our framing:** the sign depends on where the reader stands relative to what the question-blind caption holds.
- Minimal pairs as a design: TempCompass's conflicting videos, MotionBlind, TimeBlind. We apply the pairs to representations, not to models.
- Motion and reversal from pixels: Which Way Did It Move?, MotionBlind, Seeing the Arrow of Time, and the reversal-drop paper. Video models often fail direction, speed and reversal from frames too, which fits P1's null.
- Cost: no paper solves for break-even with a reader downgrade (Q6 open). Related: SAFE-Cascade (per-question routing, −9.3% cost), Sprout (lazy memory), MIT³ (parametric memory), Industrial ReMEmbR (caches vision tokens instead of captioning at write time: the opposite design).
- Visual token budgets: LoHi, VEI.
- Caption utility judged by QA: CaptionQA, CapQuiz, ViSIL.
- Temporal benchmarks: TempCompass, plus T3 (text-to-video temporal transfer; related at most, checked 2026-10-19). Reversed-video shortcuts and the forward prior (find the citation).
- **Still to do:** repeat Q7 in the week before submission; Google Scholar and OpenReview were not searched.

---

## 7. Limitations (bullets)

- One benchmark; short clips; multiple choice only.
- Closed readers run through a CLI: its system prompt is included, and there is no temperature control. Mitigations: pinned IDs, isolation checked on every call, released outputs.
- Claude reading Claude (C3). Mitigation: the Gemini chain.
- Development → confirmatory drop of Haiku on frames (0.70 → 0.61); about half of it is the mix of video types.
- Questions share filmed clips. Mitigation: clustered CIs.
- Thinking is decided per call by the model (effort medium), so it is observed, not controlled.
- **Question-blind captions drop what the question needs and the captioner did not think to say** (stacked comparisons, reversals). This is a limit of question-blind indexes, not only of our captioners.
- Claims 2a, 2b, 3a, 4, 5, 7 and 8 are exploratory, even though each was predicted in writing before its run (2a was not predicted).
- The audit's judge (Sonnet) also wrote one of the caption sets it judges. Mitigation: the 40 hand labels and κ.
- **54% of the confirmatory videos are manipulated** (reversed, stacked, two-clip), so overall accuracies mix natural video with stress tests. Report plain videos separately (§8.1).

---

## 8. Discussion: why a caption index is hard to beat and hard to evaluate (skeleton)

**Core idea.** A caption is a question-blind summary shaped by the captioner's prior. Its value lies in where the video departs from that prior, and those departures are:
- what captioners write least;
- what similarity metrics detect least;
- the only places where baselines fail.

### 8.1 What the benchmark tests when the captioner is also misled
- **The concern:** TempCompass's manipulated videos mislead models. The captioner is a model, so a low score may only mean that the captioner was fooled.
- **Answer:** the system under test is the whole index: a model writes once without the question, and the reader answers later. A captioner that can be misled is part of that system, not a confound. The arms separate the two causes, because the same model also reads the same pixels with the question in view.
- **Table 5** (D3, descriptive; Sonnet as captioner and reader; split of headroom by type; 95% CIs over clip clusters):
  - perception shortfall = 1 − frames: the model fails even when it sees the question, so frames lose this too;
  - index loss = frames − own captions: it saw the answer when asked but did not write it down blind;
  - retention = (captions − question only) / (frames − question only).

| Type (n) | Question only | Frames | Own captions | Perception shortfall | Index loss | Retention |
|---|---|---|---|---|---|---|
| plain (137) | 0.61 | 0.89 | 0.88 | 0.11 [0.06, 0.16] | +0.01 [−0.03, +0.06] | 0.95 [0.79, 1.12] |
| two-clip (45) | 0.29 | 0.91 | 0.91 | 0.09 [0.02, 0.17] | +0.00 [−0.07, +0.06] | 1.00 [0.91, 1.12] |
| reversed (74) | 0.18 | 0.64 | 0.46 | 0.36 [0.25, 0.48] | **+0.18** [+0.05, +0.30] | 0.62 [0.35, 0.88] |
| stacked (41) | 0.32 | 0.66 | 0.39 | 0.34 [0.17, 0.51] | **+0.27** [+0.09, +0.47] | 0.21 [−0.27, 0.67] |
| all (297) | 0.41 | 0.80 | 0.71 | 0.20 [0.15, 0.26] | +0.09 [+0.04, +0.14] | 0.77 [0.64, 0.90] |

- **Reading:**
  - On natural video (plain), the index keeps almost everything (0.95). The index loss is ≈ 0 on plain and two-clip videos, and its CIs exclude 0 only on reversed and stacked.
  - On reversed videos, the model is misled even when it looks with the question in view (0.36 shortfall; wrong answers equal the prior answer 0.59 of the time, C8). Writing blind loses another 0.18.
  - On stacked videos, the model can see the comparison when asked (0.66), but blind captions keep little of it (0.21; the CI is wide). This is the price of being question-blind (Prop. 4, C5b +0.22).
- **Haiku version** (Haiku reads Sonnet's captions; appendix table): the index loss is negative overall, −0.09 [−0.15, −0.04], so the captions beat Haiku's own frames, mostly on two-clip videos (−0.33 [−0.48, −0.20]); plain −0.05 [−0.11, +0.00], reversed −0.07 and stacked −0.02 (CIs include 0). Haiku's perception shortfall is 0.39 overall and ≈ 0.6 on reversed and stacked. Its retention by type is unstable where its frames are barely above its question-only score (stacked: 0.41 vs. 0.34), so it is not reported by type.
- **Role of the hand labels:** they test whether the judge correctly sorts caption failures into three kinds:
  - misstatement: the captioner was misled and wrote it;
  - omission: the captioner did not commit;
  - reader error.

  Table 5 cannot show this split. The hand labels do not split perception from blindness; the arms do that.
- **The captioner prompt is not changed.**
  - The confirmatory prompt is generic by design.
  - A prompt that names the benchmark's tricks (reversal, stacking) tells the index the test, and any gain from it would not generalize.
  - **Optional C11 (not run):** a prompt that knows the question family but not the tricks ("note direction, speed, order of events and changes of attributes"). It would measure how much of the index loss closes when the captioner knows what kind of question will come, between question-blind (now) and question-aware (C5).
- The hand-label notes are in §8.3, and κ is in §4.4.

### 8.2 Why the baselines are hard to beat
- **The prior is strong:** the question alone scores 0.41 for both readers (C8a); 0.61 on plain videos.
- **Errors fall back to the prior:** on reversed videos, Sonnet's wrong answers equal its question-only answer 0.59–0.74 of the time, across frames and caption arms (uniform 0.43; C8). The caption helps only where it contradicts the prior, and blind captioners rarely write that.
- **Omission dominates:** 0.61–0.73 of wrong answers on captions come from captions that are silent on the question (C7). Misstatement is 0.15–0.28 and reader error 0.05–0.17.
- **More pixels do not help the Claude captioner:** C9a +0.003 at 3.6× the visual tokens. The loss is in what gets written, not in what is seen.
- **The value depends on the reader** (Fig. 3): the same index helps Haiku (Sonnet captions: retention 1.47 [1.18, 1.95]) and hurts Sonnet (0.77 [0.64, 0.90]). Reading frames gains 0.19 from Haiku to Sonnet; reading Sonnet's captions gains 0.00. Whether an index "beats the baseline" depends on which reader the baseline uses (claim 2a, the cap).
- **Reconstruction phase (`docs/paper/captions_vs_frames.md`, §4):** continuity baselines were as strong as the models.
  - An 8B LLM in-filling masked captions lost to copying the nearest caption.
  - On forced-choice recognition (160 items), Haiku from captions was level with caption copy (51.2% vs. 45.6%), and frame copy (87.5%) was level with Opus (85.0%).
  - Sonnet's generated captions only drew level with caption copy (text rank 22.3 vs. 23.6).

### 8.3 Why caption utility is hard to evaluate
- **No question-free reference:** a caption is right or wrong only relative to a question. The same caption is complete for "what happens" and silent for "which copy is faster".
- **Similarity metrics track style:** in the reconstruction phase, cos_sim correlated with caption repetitiveness (APCS_T, Spearman ρ = +0.61; `docs/experiments/evaluation_and_metrics_decision_dossier.md`). Per-second and order-free scoring gave opposite headlines:
  - per-second rank had no headroom (the true caption was level with caption copy, −0.2 ranks);
  - order-free rank showed Claude's fills clearly beating caption copy (`captions_vs_frames.md` §4).
- **"Silent" and "wrong" overlap:**
  - The judge found forward narration in only 0.28–0.45 of reversed failures, against a prediction of ≥ 0.50. A caption that never states a direction is consistent with both readings.
  - The hand labels show the same overlap: all 5 judge/hand disagreements are captions the user left "silent" and the judge decided. The hard items in the user's notes fall into four kinds:
    - a static description that fits a wrong option ("nothing visibly moves" read as "unchanged"; "the upper one holds at 100%");
    - a near-synonym of an option ("places it onto" vs. "dropped");
    - a 1 s last window that contradicts the long first one ("the clip shifts from … to …" in a split-screen video);
    - an answer that needs knowledge of the task: debris "tumbling upward" means "rebuilding" only to a reader who knows clips may be reversed.
- **One caption is one sample:** re-captioning with identical input never gave the same caption (0 of 48), and answers changed on 10–17% of those questions (C10). Only averages over many questions are stable.
- **The judge shares a family with the captioner:** Sonnet judges Sonnet's captions. κ against the hand labels is the check: 0.81.

### 8.4 What made it measurable
- **Utility through a fixed reader on multiple-choice questions**, not similarity to a reference caption.
- **Controls:**
  - the self-caption control;
  - the reader's own view of the frames;
  - question only.

  Without them, a "caption effect" mixes captioner, modality and prior.
- **Minimal pairs:** chance pair accuracy is 0.093, and a reader that falls back to the prior scores 0 on a pair (C4).
- **Reporting by video type:** the overall number hides that the loss is concentrated in two types (Table 5).
- **Many questions per comparison**, because of resampling noise (§4.7).

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
| ~~40 hand labels~~ | Done (2026-10-22): agreement 0.88, κ = 0.81 ≥ 0.6, so Table 4 keeps the judge's labels | — | — |
| ~~Sonnet question-only arm~~ | Done (C8): 0.414; claim 5 holds for Sonnet | — | 7-day quota 12% → 16% for C8 + C9 |
| ~~Claim 3 and the captioner's visual budget~~ | Done (C9): 448 px captions do not close the 9 points (C9a +0.003) | — | (above) |
| ~~Window length on short videos~~ | Done (C10): whole-video captions are no better (+0.01), and cost half as much per video | — | 7-day quota 17% → 19% |
| ~~Discussion §8 numbers~~ | Done (D3): Table 5 with clustered CIs, the Haiku version, and the reconstruction-phase numbers cited | — | — |
| ~~Haiku retention CIs~~ | Done (D3, §3): CIs exclude 1 for every index but self-captions | — | — |
| C11 (planned for the week of 2026-10-26): captioner prompt that knows the question family | Splits the index loss into not knowing the kind of question vs. not knowing the question (between now and C5) | Sonnet captions with an aspect-aware prompt (no benchmark tricks named); Haiku and Sonnet read | ≈ C10's quota (7-day +2–3%) |
| ~~Figure for the reader-strength tension (claim 2a)~~ | Done: Fig. 3, a slope chart by reader | — | — |
| ~~Colour consistency~~ | Done: Fig. 2 uses palette slots 3–7, which Fig. 1's reader colours don't use; Fig. 3 reuses Fig. 2's caption hues | — | — |
| Literature: done (N2, no scoop). Repeat Q7 before submission | New work weekly | rerun the §10.3 queries | this session's quota |
| Public pre-registration archive | Reviewers can't verify local commits | an anonymous repository or OSF, before submission | user |
| Open-weights reader | "All readers are closed" | G5 is case 1, so this is optional for ACL main; decide at G9 | Kaggle |
