# Research log: caption reconstruction → caption index

This is a record of how the research proceeded: what we asked, what we expected before each experiment, what we found, and why we changed direction. It is written for the team, and parts of it may go into the paper's method and appendix.

**Rules we keep:**
1. Before each experiment, write down the hypothesis, the prediction (with a number where possible), and what result would make us stop or change course. Commit the entry, or at least save it, *before* the run.
2. Small steps, each one able to kill the idea cheaply. Extend a step only if its result is ambiguous.
3. Report every result, including the ones against us. If a prediction was wrong, say so here, next to it.
4. Development data (used to decide what to do next) is kept apart from confirmatory data (used for the paper's claims). The 100 TempCompass questions below are development data. The confirmatory set will be drawn from the other 310 TempCompass videos.
5. Every term has one meaning (see *Terms* in `research_plan_caption_index.md`).

Where an entry was not written in advance, it says so.

---

## Part 1: how we got here (2026-09 to 2026-10-15), in brief

| Phase | Question | What happened | Why we moved on |
|---|---|---|---|
| Generation | Can an LLM fill a gap in a video's captions from the surrounding captions, better than copying a neighbour? | Llama-8B, Sonnet and Opus fills never clearly beat "copy the neighbouring caption" or "average the neighbours" in any evaluation we tried (embedding similarity, retrieval, lenient scoring, WildQA retrieval). | The baselines are strong because neighbouring captions are nearly the same. A metric that rewards matching the true caption rewards copying. We could not find an evaluation where reconstruction shows a benefit, so we treat this path as blocked. |
| Recognition | Can a model pick the true missing span among candidates? | Opus recognized well, about at the level of copying the neighbouring frame. | Positive, but it does not beat the visual baseline, and its captions had a leak (they were written with the whole minute in view). |
| Timing audit | Do the captions record *when* things happen? | They run 1–2 s early, nearly constantly. After correcting that, they are only slightly better than the neighbouring second's caption at picking out their own second. | A constant offset is a captioning convention, not lost time. What remains is a narrow audit of one captioner. Too weak to stand alone. |
| Unshuffling | Can captions vs. frames put shuffled 5 s pieces of a clip back in order? | Frames beat captions for every Claude reader, in direction too. The central hypothesis (captions carry direction better) failed. | Ordering shuffled pieces is not a real-world task, and the language-vs-images version is published (SPLICE, TempVS). The useful piece we keep is the *information control*: captions made from exactly what the frame arm sees. |
| Caption index (now) | Can a one-time caption index answer later questions as well as frames, at lower query cost? | See Part 2. | — |

The motivation for the current direction: captioning a video is a one-time cost. If many later questions can be answered from the captions, the video need not be read again. The literature has the amortized formula (CFD, CapMem, Sept. 2026), but not per-task accuracy at matched cost on short video, nor a split of *why* captions help. See `related_work_caption_index.md`.

---

## Part 2: experiment log

### K0 — TempCompass kill test (2026-10-18)

**Question.** Does a caption index come anywhere near frames on short-video QA at a similar token cost? If not, the index direction is dead.

**Stated before the run** (in conversation with the user, not committed):
- Continue if any caption arm comes close to frames at matched cost on any aspect.
- Stop if the question-only arm is about as good as frames (the benchmark would not separate the arms).
- No numeric predictions were written down. That was a gap; from K1 on, they are.

**Design.**
- **Questions:** TempCompass multiple-choice, 100 questions, 20 per aspect (action, speed, direction, order, attribute change), one per video, seeded sample (`results/index_qa/kill_test_items.json`).
- **Windows:** videos cut into 5 s windows (283 in total).
- **Arms:** question only; captions (Gemini 3.8 Flash, one isolated caption per window, the same prompt as our WildQA captions); 1 frame per window (224 px); 5 frames per window (224 px, 1 s apart).
- **Reader:** Claude Haiku 5.5, one isolated `claude -p` call per item.
- **Code:** `scripts/index_qa/`.

**Result.**

| Arm | Accuracy | Tokens for the video content |
|---|---|---|
| Question only | 0.36 (chance 0.30) | 0 |
| Captions | 0.71 | ≈ 245 |
| 1 frame / window | 0.65 | ≈ 180 |
| 5 frames / window | 0.70 | ≈ 560 |

- Captions − 5 frames: +0.01 [−0.07, +0.09]. Captions − 1 frame: +0.06 [−0.04, +0.16]. Every call also carries ≈ 1,024 tokens of fixed prompt.
- By aspect (n = 20 each, so ±0.2 per cell):

  | Aspect | Question only | Captions | 1 frame | 5 frames |
  |---|---|---|---|---|
  | action | 0.40 | 0.90 | 0.90 | 0.90 |
  | attribute change | 0.35 | 0.90 | 0.65 | 0.80 |
  | order | 0.30 | 0.80 | 0.65 | 0.65 |
  | speed | 0.45 | 0.60 | 0.65 | 0.70 |
  | direction | 0.30 | 0.35 | 0.40 | 0.45 |

- Cost: $0.81 of Gemini (283 calls); 400 Haiku calls, 5-hour quota 40% → 44%, 7-day figure unchanged at 71%.

**Reading.** The benchmark separates the arms (question only is near chance). Captions tie 5 frames at about 40% of the video tokens. The per-aspect leads are hints only. **Decision: continue.**

**Open confounds K0 cannot settle.**
1. **Weak reader:** Haiku may simply read images poorly, so captions from a stronger captioner help it. A stronger reader could close the gap. → K1.
2. **Captioner vs. format:** captions may help because Gemini sees better (captioner effect) or because text is easier for Haiku to reason over (format effect). → K2.
3. **Coverage and resolution:** Gemini saw each window as video at its own resolution; the frame arms see 224 px JPEGs. Not addressed yet.

---

### K1 — Does the tie survive a strong reader? (pre-registered 2026-10-18, before any Sonnet call)

**Items.** The first 10 questions of each aspect in the K0 sample: 50 questions, 138 windows. On these 50, Haiku's K0 accuracy was: captions 0.64, 1 frame 0.64, 5 frames 0.66, question only 0.34 (computed before this entry was written, from the K0 rows).

**Arms.** Claude Sonnet 5.5 on captions and on 5 frames: 100 calls. Same prompts, isolation and effort (medium) as K0.

**Hypotheses and predictions.**
- **K1a (strong reader reads frames better):** Sonnet-frames > Haiku-frames. Prediction: Sonnet-frames ≈ 0.76 (+0.10).
- **K1b (the tie survives):** Sonnet-captions − Sonnet-frames ≥ −0.05. Prediction: −0.04. Reasoning: captions cap what any reader can get (direction and speed are mostly lost), so Sonnet gains more from frames than from captions, but not much more on these short clips.
- **K1c (cheap reader on captions vs. strong reader on frames):** Haiku-captions − Sonnet-frames. Prediction: −0.12. This is the "see once with a strong model, answer cheaply" claim. A small gap at a fraction of the cost would support it.

**Decision rules (on the 50 questions).**
- If Sonnet-captions − Sonnet-frames ≤ −0.15, captions lose for a strong reader. The K0 tie was a weak-reader effect. **Stop** the index direction, and report.
- If it is ≥ −0.05, the tie survives. **Continue**: K2, then extend K1 to the other 50 questions.
- In between: ambiguous. Extend K1 to the other 50 questions (100 more calls) before deciding.
- K1c is recorded and does not decide anything at this stage.

**Precision.** With 50 paired questions and about a quarter of them discordant, a 95% CI on a paired difference is about ±0.14. K1 can detect a large gap, not a small one.

**Result, stage 1 (50 questions; 100 Sonnet calls; 5-hour quota 52% → 58%, 7-day 72% → 73%):**

| | Accuracy | Prediction |
|---|---|---|
| Sonnet, 5 frames | 0.82 | 0.76 |
| Sonnet, captions | 0.76 | — |
| Haiku, 5 frames (K0) | 0.66 | — |
| Haiku, captions (K0) | 0.64 | — |

| Difference | Observed [95% CI] | Prediction | Verdict |
|---|---|---|---|
| K1a: Sonnet frames − Haiku frames | +0.16 [+0.06, +0.26] | +0.10 | Holds; larger than predicted |
| K1b: Sonnet captions − Sonnet frames | −0.06 [−0.16, +0.04] | −0.04 | Ambiguous zone (between −0.15 and −0.05) |
| K1c: Haiku captions − Sonnet frames | −0.18 [−0.30, −0.08] | −0.12 | A cheap reader on captions does *not* match a strong reader on frames; the gap is larger than predicted |
| (not pre-registered) Sonnet captions − Haiku captions | +0.12 [+0.02, +0.24] | — | The reader matters even when the input is text |

By aspect (10 questions each, descriptive only), Sonnet frames vs. Sonnet captions: action 1.0 vs. 1.0, attribute change 0.8 vs. 0.9, order 0.9 vs. 0.9, speed 0.8 vs. 0.7, **direction 0.6 vs. 0.3**. The frames' lead for Sonnet comes mostly from direction, the aspect with the lowest codability.

**Decision (by the rule above):** K1b is in the ambiguous zone, so extend K1 to the other 50 questions (100 more Sonnet calls) before deciding.

**What changed in our thinking:**
- K0's tie was partly a weak-reader effect, as feared. Sonnet gets more from frames than Haiku does (+0.16), and more than it gets from captions.
- The "see once with a strong model, answer cheaply" claim (K1c) is weakened. With Gemini captions, Haiku stays 0.18 below Sonnet on frames.
- Captions still keep most of the value for Sonnet at less than half the video tokens. Whether that counts as a tie or a small loss is what stage 2 decides.

**Result, stage 2 (all 100 questions; 200 Sonnet calls in total; 7-day quota 72% → 73%):**

| Difference | 100 questions [95% CI] | Stage 1 (50) |
|---|---|---|
| K1a: Sonnet frames − Haiku frames | +0.14 [+0.06, +0.22] | +0.16 |
| K1b: Sonnet captions − Sonnet frames | **−0.07 [−0.15, 0.00]** | −0.06 |
| K1c: Haiku captions − Sonnet frames | −0.13 [−0.21, −0.06] | −0.18 |
| Sonnet captions − Haiku captions | +0.06 [0.00, +0.12] | +0.12 |

Accuracies: Sonnet frames 0.84, Sonnet captions 0.77, Haiku frames 0.70, Haiku captions 0.71, question only 0.36. **Retention** (the share of the frames' gain over question-only that captions keep): Sonnet 0.85, Haiku 1.03.

By aspect (20 each), Sonnet captions vs. frames: action 1.00 vs. 1.00, attribute change 0.95 vs. 0.85, order 0.95 vs. 0.95, **speed 0.60 vs. 0.80, direction 0.35 vs. 0.60**.

**Decision.** K1b is still in the ambiguous zone (−0.07; the CI just reaches 0). **The rule did not say what to do if stage 2 were also ambiguous. That was a gap in the pre-registration.** We do not stop the index direction on this. We do not claim a tie either. What the data support, as development findings:
1. For a strong reader, a question-blind caption index keeps about 85% of what frames add, with about 44% of the video tokens (≈ 245 vs. ≈ 560).
2. The loss is concentrated in direction and speed. On action, order and attribute change, captions lose nothing.
3. Caption benefit comes from the captioner, not the format (K2).
4. A cheap reader on captions does not reach a strong reader on frames (K1c).

**Caveat on point 2.** Direction and speed were named as low-codability families in the plan before K0 (2026-10-15, H3: "direction, speed, order of continuous motion, fine appearance"). But the split of TempCompass's five aspects into high and low groups has not been fixed independently of these results. For the confirmatory test it must be fixed by codability ratings made without seeing any results, or stated as the plan's a-priori grouping and nothing else.

### D1 — How the confirmatory aspect hypothesis is fixed (2026-10-18)

**Record keeping.** K1 and K2's predictions were in this file before their runs, but were committed only afterwards (commit a94116c), together with their results. From the confirmatory step on, each entry is committed before the run.

**Question raised by the user.** Why rate questions for codability at all, and how could we know the ratings are right?

**Answer and decision.**
- **The aspect claim needs no ratings.** The claim that captions lose on direction and speed, and not on action, order or attribute change, can be tested directly on fresh data. That is what the development/confirmatory split is for. The development data suggested the grouping; the 310 unused TempCompass videos test it. A grouping chosen from development data is not circular when the test is on held-out data.
- **Ratings would only serve the explanation** ("because these are hard to put into words"). Codability has no ground truth. In cognitive science it is *measured* as agreement between people's descriptions, and it is valid insofar as it predicts something. The user's suggestion (let a model predict, then check) is how such a measure would be validated. In this field it is also a known method: CFD's visual-need router predicts from the question whether frames are needed. That is a separate, optional experiment ("can a model tell in advance when the index will fail?"), not a prerequisite.
- **The original paper's grouping does not match ours.** TempCompass (Liu et al. 2024) treats action as largely solvable from static cues and the other four aspects as temporal. It reports no text-only or caption-input baselines. In our development data, captions keep two temporal aspects (order, attribute change) and lose two (direction, speed). So "temporal vs. static" is not what a caption loses.
- **The sharper hypothesis** (post hoc, from development data; to be tested on the confirmatory set): a caption keeps discrete content (what happened, in what order, what changed) and loses continuous quantities (how fast, which way).
- **Decision:** drop the rating step. Use TempCompass's own aspects, with the grouping {direction, speed} vs. {action, order, attribute change} fixed now, before any confirmatory data.

### K2 — Captioner vs. format: Haiku captions its own frames (pre-registered 2026-10-18)

**Items.** The same 50 questions as K1.

**Arm.** *Self-captions:*
- **Captioning:** Haiku writes one caption per window, from exactly the 5 frames (224 px) that the 5-frame arm shows. One isolated call per window, never seeing the question. The prompt is the Gemini caption prompt with "the attached 5-second video clip" replaced by a description of the frames (`prompts/index_qa/self_caption.txt`). 138 calls.
- **Reading:** Haiku answers from these captions, with the same reader prompt and modality text as the Gemini captions. 50 calls.

**What it separates (Haiku as the reader throughout):**
- **Format effect** = self-captions − 5 frames: the same model's own words vs. the pixels it saw.
- **Captioner effect** = Gemini captions − self-captions: what a different (stronger) captioner adds. It includes Gemini's better resolution and its video input, which we cannot separate yet (confound 3).

**Prediction.** Self-captions ≈ 0.60 on these 50: format effect ≈ −0.06, captioner effect ≈ +0.04. Reasoning: Haiku cannot write down what it does not notice, and verbalizing loses detail; most of the help comes from the captioner.

**Interpretation map** (no stop rule; this step measures the mechanism):
- **Self-captions ≈ Gemini captions:** the format effect drives the benefit. Text is easier to reason over, and any decent captioner will do.
- **Self-captions ≈ 5 frames or lower, Gemini captions higher:** the captioner effect drives the benefit. The index transfers a strong model's perception into text ("see once with a strong model").
- **Self-captions > 5 frames:** verbalization helps even within one model. This is the describe-then-answer effect, now measured with the description made blind to the question.

With n = 50, only differences of about 0.14 or more are resolvable. Extend to 100 if the result is near a boundary that matters for the paper.

**Result (138 self-caption calls + 50 reader calls, all Haiku; 5-hour quota 53% → 63% including K1 running alongside, 7-day 73%):**

| Haiku reads | Accuracy | Prediction |
|---|---|---|
| 5 frames | 0.66 | — |
| Gemini captions | 0.64 | — |
| Its own captions of those 5 frames | **0.56** | 0.60 |
| Question only | 0.34 | — |

| Effect | Observed [95% CI] | Prediction |
|---|---|---|
| Format = self-captions − 5 frames | −0.10 [−0.22, +0.02] | −0.06 |
| Captioner = Gemini captions − self-captions | +0.08 [0.00, +0.18] | +0.04 |
| Self-captions − question only | +0.22 [+0.06, +0.38] | — |

By aspect (descriptive), self-captions vs. 5 frames: speed 0.3 vs. 0.6, action 0.7 vs. 0.8, attribute change 0.8 vs. 0.7, order 0.7 vs. 0.8, direction 0.3 vs. 0.4.

**Deviation.** Haiku did not keep to the 45-word limit: its captions have a median of 46 words (78 of 138 over the limit), against 30 for Gemini. So the self-caption arm had *more* text and still did worse. Length does not explain the gap.

**Reading.** The result falls in the pre-mapped branch "self-captions ≈ 5 frames or lower, Gemini captions higher": **the benefit, where there is one, comes from the captioner, not from the text format.** Both effects are about the predicted size, slightly larger, and in the predicted direction. Neither is individually significant at n = 50; together they point the same way.

This contrasts with IsoBench (N1), where text beats images of the same problem by 15–29 points. A plausible explanation, not yet tested: IsoBench's text is an exact, complete rendering of the problem. A question-blind caption is lossy, and the loss (speed, fine motion) outweighs any advantage of reasoning over text. If this holds up with more data, it is a clean, citable contrast: *the text advantage holds for lossless text, and reverses for question-blind descriptions.*

### N1 — Novelty search (2026-10-18)

Targeted at the two candidate claims:
1. A strong captioner plus a cheap reader matches a strong reader on frames at lower cost, on short-video QA.
2. A split of caption benefit into captioner and format effects, with question-blind captions.

**Searches** (7 web searches, 2026-10-18): strong captioner with a weak reader; describe-then-answer and verbalization; captions vs. frames at matched tokens on short video and TempCompass; question-aware vs. generic captions; perception–reasoning decoupling; same problem as text vs. image.

**What is already known:**
- **The format effect, in general.** [IsoBench](https://arxiv.org/abs/2404.01266) (COLM 2024) gives the same math, science and game problems as text and as images. Models do much better on text (Claude 3 Opus −28.7 points on images, GPT-4 Turbo −18.7, Gemini Pro −14.9). So "text is easier for the reader than pixels" is established for content that has an exact text equivalent. It is not established for video, where the text is a lossy caption written without the question.
- **Splitting perception and reasoning.**
  - [Prism](https://arxiv.org/abs/2406.14544) (NeurIPS 2024, images): a VLM describes, an LLM reasons. A 2B captioner with GPT-3.5 matched VLMs about 10× larger. That is the *small captioner, strong reader* direction.
  - On long video: [LLoVi](https://arxiv.org/abs/2312.17235), [VideoTree](https://arxiv.org/abs/2405.19209) and [SiLVR](https://arxiv.org/abs/2505.24869) caption clips and reason over the text. LLoVi reports that the choice of captioner matters (oracle captions +10.8 points on EgoSchema), and that GPT-3.5 was the best accuracy-for-cost reader.
- **Question-aware captions beat generic ones** on NExT-QA ([VidCtx](https://arxiv.org/abs/2412.17415), Q-ViD), by about 1–2 points in VidCtx's ablation. This is the price of task-agnosticism (plan Prop. 4), and it is small there.
- **Caption-once on long video:** [CFD](https://arxiv.org/abs/2609.11899) and [CapMem](https://arxiv.org/abs/2609.17688), already in the review.

**What we did not find:**
1. A comparison at **matched query cost** of a cheap reader on a strong captioner's question-blind captions against strong and cheap readers on the frames themselves, on **short** video.
2. A split of the caption benefit into **captioner** and **format** effects with a **self-caption control** (the reader captions the same frames itself, without the question), per question aspect.
3. Break-even counts that include the reader downgrade.

**Verdict.** Claim 1 alone is incremental: Prism and LLoVi already treat captioner and reader as separable modules, and LLoVi already reports an accuracy-for-cost choice of reader. The open, defensible contribution is the combination: a decomposition on video into coverage, captioner and format effects, with question-blind captions, at matched cost, per aspect. Our results so far fit the IsoBench finding (format) and a captioner effect (K2 will say which dominates). The paper should cite IsoBench and Prism as the image-domain precedents and position itself as their video, amortized-index counterpart. This was a quick check, not a full review; a final pass is needed before submission.
