# Draft v0 (skeleton): caption once, ask cheaply

Started 2026-10-20. This is a skeleton, not prose. It shows which claims we make, what backs each one, and what is missing. Numbers come from `docs/new_direction/research_log.md` (entry names in brackets). **⟨PENDING: X⟩** marks a number that run X will fill. Terms follow the *Terms* block of `research_plan_caption_index.md`: source, visual budget, captioner and modality effects.

Target: ACL 2027 (ARR, January 2027), long paper (8 pages); long or short decided at G9 (Nov 30).

---

## 0. Claims and evidence

| # | Claim (as the paper would state it) | Evidence | Status |
|---|---|---|---|
| 1 | **Who writes the caption matters.** With the same frames, a stronger model's question-blind captions make a cheap reader more accurate than the reader's own captions. | P3 +0.148 [98.33%: +0.088, +0.209] (C1, pre-registered); C2a at about matched visual tokens +0.081 [+0.030, +0.131]; C3a, the Claude captioner at identical visual tokens | **Backed across families** (Gemini). The within-family test at exact token parity is **⟨PENDING: C3⟩** |
| 2 | **Text itself is not easier for the reader.** The reader's own captions of its frames are no better than the frames. | P2 +0.030 [98.33%: −0.034, +0.094] (C1); upper bound +0.09, against IsoBench's 15–29 points | **Backed as a bound** (one reader). Sonnet's version (C4e on its own captions) **⟨PENDING: Sonnet reads of C3 captions⟩** |
| 2b | …except where the video has two clips: there captions help (+0.22). Is it from verbalizing, or from captioning each window on its own? | A3 by video type (exploratory) | **Weak: post hoc.** Mechanism **⟨PENDING: C5, C4a⟩** |
| 3 | **The index pays through the reader downgrade, not through fewer tokens.** A cheap reader on a strong model's captions comes close to the strong model on frames, at about 1/25 the cost per question, and repays the captioning after a few questions. | A2: Haiku on Gemini frame captions vs. Sonnet on frames −0.007 [−0.057, +0.044], Q\* 3.7 [3.5, 4.0]; Haiku on Gemini video captions Q\* 1.8, −0.061; same-reader savings only 18% (Haiku), 30% (Sonnet) | **Backed for Gemini captions.** The Claude chain (C3c, Sonnet captions once and Haiku answers vs. Sonnet answers each question) is **⟨PENDING: C3⟩** |
| 4 | **Question-blind captions drop comparisons between things shown at the same time** (stacked clips at two speeds), even from a strong captioner. | A3: on stacked videos Sonnet on frames beats every caption arm, in both sets | **Weak: post hoc, small n (41).** Tested by the minimal pairs (C4 H1) and the audit (C7: silent ≥ 0.70) **⟨PENDING⟩** |
| 5 | **Reversed clips: both modalities fall back to the forward reading;** a captioner that reports the reversal helps. | A3: the share of wrong answers that equal the question-only answer is 0.5–0.67 in both modalities; C2a on reversed videos +0.14 | **Weak.** C4 (reversal pairs) and C7 (wrong ≥ 0.50) **⟨PENDING⟩** |
| 6 | **Captions beat frames for Haiku even when the frames get 2× more tokens than the captions.** | C6 (448 px) | **⟨PENDING: C6⟩**. If C6c's CI includes 0, claim 6 is dropped and the "starved frames" risk is answered by cost (claim 3) only |
| — | *Dropped:* "captions lose continuous quantities (direction, speed)" | P1 −0.10 [98.33%: −0.23, +0.03], not passed; aspect ≈ video type (A3) | Reported as a null in §4.5 |
| — | *Dropped:* "the captioner's knowledge" | No arm separates perceiving, recognizing and writing (D2) | We say "captioner effect" |

**Headline, if C3 is positive (go/no-go case 1):** "A one-time caption index written by a strong model lets a cheap model answer temporal questions about short videos almost as well as the strong model reading the frames, at 1/25 of the query cost. The gain comes from the captioner, not from the text modality."

**If C3 is not positive (case 2):** claim 1 names Gemini, and claim 3 stays; the length is decided at G9.

---

## Abstract (bullets)

- Setting: captions as a one-time, question-independent index of short videos, read later by an LLM.
- Question: when and why does a reader answer better from the index than from frames, and what does it cost?
- Design: an information control. The captioner sees exactly the reader's frames, which splits the caption effect into source, visual budget, captioner and modality effects. Pre-registered on 297 TempCompass questions.
- Findings: captioner effect +0.15 (P3), still +0.08 at matched visual tokens (C2a), and ⟨PENDING: C3a⟩ within one model family at identical tokens. Modality effect ≤ +0.09. The index pays through the reader downgrade: Q\* ≈ 2–4 questions per video.
- Where captions fail: ⟨PENDING: C4, C7⟩.

---

## 1. Introduction (bullets)

- **Motivation.** Captioning a video is paid once; questions come later and repeatedly. Caption-then-reason pipelines are widely used (LLoVi, VideoTree, SiLVR), and "caption once, reuse" is now published for long video (CFD, CapMem, Sprout). Two questions are open:
  1. When the captions win, **why**? Did the captioner see more, spend more visual tokens, write differently, or is text easier to read than pixels?
  2. **Does it pay**, counting the captioning and the reader's full per-question cost?
- **Approach.** The information control (Prop. 1): give the captioner exactly the frames the reader sees. Then any caption advantage is usable information added by the captioner or by the modality, not information about the video.
- **Contributions:**
  1. A decomposition of the caption effect into source, visual budget, captioner and modality effects, with arms that change one thing each, in two model families (§2.3, Table 2).
  2. A pre-registered test with its result: captioner effect positive, modality effect bounded at ≤ +0.09 (§4.1–4.2).
  3. A cost account per question (input, output and thinking tokens, list-price dollars) and the break-even Q\* (§5).
  4. An analysis of what captions lose, by how the video was built, with minimal pairs and a caption audit (§4.3–4.4).
- **Scope:** short videos (≈ 10 s), multiple choice, one benchmark.

---

## 2. Setup

### 2.1 Task and data
- **TempCompass** multiple choice (CC BY-NC). Aspects: action, speed, direction, order, attribute change. Median video 10 s.
- **Development set:** 100 questions (K0–K2), used to set the hypotheses.
- **Confirmatory set:** 300 questions sampled (60 per aspect, one per video, on videos the development set does not use), 297 after Amendment 2; pre-registered (C1).
- **Video types** (A3): plain (137), reversed (74), stacked (41), two-clip (45). They nearly determine the aspect.
- **Windows:** 5 s; frames at 1 fps (t = 5p + k), 224 px long edge (40 Claude image tokens each).

### 2.2 Arms

| Arm | Reader sees | Visual tokens for the reader / captioner per 5 s window |
|---|---|---|
| Question only | question and options | 0 |
| 1 frame | the middle frame of each window | 40 |
| 5 frames | all 5 frames of each window | 200 |
| 5 frames, 448 px | the same frames at 448 px | 720 ⟨C6⟩ |
| Describe, then answer | 5 frames; writes one sentence per window, then answers | 200 ⟨C5⟩ |
| Self-captions | the reader's own isolated caption of each window's 5 frames | — / 200 |
| Sonnet captions | Sonnet's isolated caption of the same 5 frames | — / 200 ⟨C3⟩ |
| Gemini frame-clip captions | Gemini's caption of the same 5 frames, played as a 1 fps clip | — / ≈ 280–330 |
| Gemini frame captions | Gemini's caption of the same 5 frames, as images | — / ≈ 5,500 |
| Gemini video captions | Gemini's caption of the video window | — / ≈ 330 |

- **Readers:** Claude Haiku 5.5 and Sonnet 5.5 (pinned IDs, effort medium), through `claude -p` with no tools, isolated per call.
- **Captioners:** Haiku (self), Sonnet (C3), Gemini 3.8 Flash. Every caption is question-blind and window-isolated, and all use one prompt.

### 2.3 The four effects (Haiku reads)

| Effect | Arms compared | Claude chain | Gemini chain |
|---|---|---|---|
| Source | video captions − frame-clip captions | 0 by design | C2c +0.013 [−0.030, +0.061] |
| Visual budget | frame captions − frame-clip captions | 0 by design | C2b +0.067 [+0.030, +0.108] |
| Captioner | other captioner − self-captions, same pixels | C3a ⟨PENDING⟩ | C2a +0.081 [+0.030, +0.131] |
| Modality | self-captions − 5 frames | P2 +0.030 [−0.034, +0.094] | (same) |

### 2.4 Statistics and pre-registration
- C1 was pre-registered: P1–P3 at 98.33% CIs (Bonferroni over three); secondary estimates at 95%.
- C2–C7 are exploratory, and each was logged with predictions before its first call.
- Paired bootstrap over questions; clip-clustered bootstrap as a check (A3: 156 clusters; the CIs agree).
- **Missing:** a public, timestamped archive of the pre-registration commit (plan §5 step 9; G11).

---

## 3. Theory (short)

- **Prop. 1 (garbling bound).** A caption made from the reader's own frames cannot add information about the video, for an unbounded reader. For a fixed, bounded reader, it can add *usable* information (Xu et al. 2020), and that is what the captioner effect measures.
- **Corollary.** The total caption effect = source + captioner + modality, plus a visual budget term when the visual tokens differ.
- **Prop. 2 (break-even).** Q\* = C(τ) / (c_f − c_τ): the captioning cost over the per-question saving. With a reader downgrade, c_τ is the cheap reader's cost.
- Props. 3–4 (routing headroom, price of task-agnosticism) are **not used** unless C5 is read as Prop. 4 evidence. Decide while writing.

---

## 4. Results

### 4.1 Table 1: accuracy and cost per question (confirmatory set, 297 questions)

| Reader, input | Acc. | Input tok. (median) | $ / question | Index $ / video |
|---|---|---|---|---|
| Haiku, question only | 0.411 | 1,025 | 0.000109 | — |
| Haiku, 1 frame | 0.582 | 1,189 | 0.000146 | — |
| Haiku, 5 frames | 0.613 | 1,520 | 0.000204 | — |
| Haiku, 5 frames 448 px | ⟨C6⟩ | ⟨C6⟩ | ⟨C6⟩ | — |
| Haiku, describe then answer | ⟨C5⟩ | ⟨C5⟩ | ⟨C5⟩ | — |
| Haiku, self-captions | 0.643 | 1,283 | 0.000189 | 0.00058 |
| Haiku, Sonnet captions | ⟨C3⟩ | ⟨C3⟩ | ⟨C3⟩ | ⟨C3⟩ |
| Haiku, Gemini frame-clip captions | 0.724 | ⟨cost rerun⟩ | ⟨cost rerun⟩ | ⟨cost rerun⟩ |
| Haiku, Gemini frame captions | 0.791 | 1,229 | 0.000167 | 0.01461 |
| Haiku, Gemini video captions | 0.737 | 1,236 | 0.000168 | 0.00726 |
| Sonnet, 5 frames | 0.798 | 1,522 | 0.004112 | — |
| Sonnet, Sonnet captions | ⟨Sonnet reads⟩ | | | ⟨C3⟩ |
| Sonnet, Gemini video captions | 0.791 | 1,238 | 0.002868 | 0.00726 |

### 4.2 Table 2: the decomposition
- Gemini chain: done (C2). Claude chain: ⟨C3⟩.
- **Text for the table:** P3 passed at 2.5× the predicted size. About half of it survives at matched visual tokens (C2a); the rest is visual budget (C2b). The source effect is ≈ 0.
- C3 predictions on record: C3a +0.09, C3b +0.12, C3c −0.07.

### 4.3 Where captions lose: by video type (A3) and minimal pairs (C4)
- **Figure 2:** accuracy by video type, all arms (A3's table).
- **Table 3:** pair accuracy and same-answer rate per arm, by group (reversal, stacked, two-clip). ⟨PENDING: C4⟩
- **Claims 4 and 5** stand or fall here.

### 4.4 Why caption arms fail: the audit (C7)
- **Table 4:** each reader arm's wrong answers, split into captions state the answer (reader error), state a wrong option (misstatement), or are silent (omission); by video type. ⟨PENDING: C7⟩
- Judge agreement with the user's 40 hand labels (κ). ⟨PENDING: user⟩
- Two quoted examples: the reversed salt pile (forward narration) and the stacked jump rope (no comparison) (A3).

### 4.5 Aspects (P1, null) and thinking tokens (A1)
- P1 is not passed. Aspect is nearly the same as video type, so it moves to the appendix as a table.
- **Thinking** marks wrong answers on captions (no thinking 0.84–0.90 vs. thinking 0.62–0.65), but not for Haiku on frames. Sonnet thinks 3.3× less on captions. One paragraph, exploratory.

### 4.6 Describing first (C5) and frame size (C6)
- C5: does verbalizing in the same call recover the two-clip gain? ⟨PENDING⟩
- C6: do 448 px frames close the gap to captions? ⟨PENDING⟩

---

## 5. Cost and break-even

- **Figure 1:** accuracy against list-price $ per question (log x), one point per arm, with the index cost amortized at Q = 1, 10 and 100. **Missing: a plotting script** (to write; data in `reader.py cost`).
- Same reader: captions save 18% (Haiku) and 30% (Sonnet) per question. The fixed prompt (≈ 1,025 tokens) and thinking dominate.
- **Reader downgrade:** Q\* 1.8–3.7 (A2); the Sonnet-caption chain ⟨C3⟩.
- Latency does not favour captions (+0.1–1.1 s per call; CLI wall clock, uncontrolled).
- Thinking is 2–5% of reader tokens.

---

## 6. Related work (bullets; only Q5 of the literature pass is done: log N2a)

- Caption-then-reason for video QA: LLoVi (the captioner matters: oracle captions +10.8), VideoTree, SiLVR, Vamos (text competitive with visual embeddings; verified).
- Caption indexes and memories: CFD, CapMem, Sprout, VidCtx (question-aware captions +1–2 points).
- **A captioner in front of a fixed reader (closest to claim 1):** Prism (images; a 2B captioner + GPT-3.5 ≈ 10× larger VLMs), MetaCaptioner, CapRL / CapRL++ (caption utility as the RL reward). **What none has:** a self-caption control, matched visual tokens, or the reader's own view of the frames (N2a). That is our gap; word claim 1 as the controlled decomposition.
- Text vs. pixels for the same content: IsoBench (text +15–29 points when the text is exact), TokenSwap.
- Visual token budgets: LoHi, VEI.
- Caption utility judged by QA: CaptionQA, CapQuiz, ViSIL.
- Temporal benchmarks: TempCompass, plus T3 (text-to-video temporal transfer; related at most, checked 2026-10-19).
- **Still missing:** Q6 (break-even with a reader downgrade), Q7 (new close work since Oct 15), and the rest of G6.

---

## 7. Limitations (bullets)

- One benchmark; short clips; multiple choice only.
- Closed readers run through a CLI: its system prompt is included, and there is no temperature control. Mitigations: pinned IDs, isolation checked on every call, released outputs.
- Claude reading Claude (C3). Mitigation: the Gemini chain.
- Development → confirmatory drop of Haiku on frames (0.70 → 0.61); about half of it is the mix of video types.
- Questions share filmed clips. Mitigation: clustered CIs.
- Thinking is decided per call by the model (effort medium), so it is observed, not controlled.

---

## Appendix (list)

- A. Prompts: caption, reader, describe-then-answer, audit.
- B. The development phase (K0–K2) and how it set the hypotheses; predictions against results.
- C. Amendments, deviations, and the term mapping (D2).
- D. Tables by aspect and by video type; clustered CIs.
- E. The minimal-pair construction (C4).
- F. Release: code, captions, raw outputs; TempCompass licence (CC BY-NC).

---

## Gaps found while outlining (to decide)

| Gap | Why it matters | Fix | Cost |
|---|---|---|---|
| Sonnet question-only arm | Sonnet's retention (S1) and its prior on reversed videos; Table 1 has no Sonnet floor | 297 Sonnet calls | ≈ 4 points of the 7-day quota |
| Figure 1 script | The cost result needs a figure | `scripts/index_qa/plots.py` | none |
| Literature pass: Q5 done (N2a: close, not scooped); Q6, Q7 and the rest open | Novelty of claims 1 and 3 | `final_search_brief.md` | this session's quota |
| Public pre-registration archive | Reviewers can't verify local commits | an anonymous repository or OSF, before submission | user |
| Open-weights reader | "All readers are closed" | decide after C3 (G5) | Kaggle |
| Cost rows for frame-clip, Sonnet captions, 448 px, describe | Table 1 completeness | rerun `reader.py cost` after the runs | none |
