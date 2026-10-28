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
- **Captioning:** Haiku writes one caption per window, from exactly the 5 frames (224 px) that the 5-frame arm shows. One isolated call per window, never seeing the question. The prompt is the Gemini caption prompt with "the attached 5-second video clip" replaced by a description of the frames (built in code: `caption.frames_prompt`, formerly `self_caption.prompt_for`). 138 calls.
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

### C1 — Confirmatory study (pre-registered 2026-10-18, committed before any confirmatory call)

**Why new data.** Two reasons, both about the paper's claims, not about the questions themselves:
1. **The development questions chose the hypotheses.** The aspect grouping (D1) and the "captioner, not format" reading (K2) came from looking at the 100 development questions. Testing them on the same questions would be circular. New questions on new videos are an independent test.
2. **Precision.** On 50–100 questions, the key effects (format −0.10, captioner +0.08, caption loss −0.07) all have CIs that reach 0. 300 questions roughly halve the CI widths of K2.

The development results stay in the paper as the exploratory phase; the claims rest on C1.

**Items** (`results/index_qa/confirm_items.json`, committed with this entry). 60 questions per aspect, 300 in total, one per video, on 300 of the 310 TempCompass videos the development set does not use (no overlap); 772 windows, mean chance 0.297. Seed 1; aspects filled rarest first, and from videos that only one aspect asks about before videos several aspects ask about (`tempcompass.py`, `sample_confirm`). *Amendment, before any file was written:* the first rule (rarest first, without the single-aspect preference) reached only 59 action and 59 direction questions and exited without writing; the single-aspect preference was added and the sample drawn once.

**New arm: coverage control.** K2's captioner effect included a difference in input: Gemini watched the video window, Haiku saw 5 JPEGs of 224 px. C1 adds *frame captions*: Gemini captions exactly the 5 JPEGs that the 5-frame arm shows, with the same prompt as the self-captions (`caption.py --input frames`). With Haiku as the reader, the total caption benefit then splits exactly, as in the plan's corollary:

| Term | Difference (Haiku reads) | What differs |
|---|---|---|
| Coverage | Gemini video captions − Gemini frame captions | what the captioner saw (video at full resolution vs. the 5 small frames) |
| Captioner | Gemini frame captions − self-captions | who wrote the caption, from identical input |
| Format | self-captions − 5 frames | the same model's own words vs. the pixels it saw |
| Total | Gemini video captions − 5 frames | = coverage + captioner + format |

**Arms and cost.**
- **Captions:** Gemini 3.8 Flash on the video (772 calls) and on the frames (772 calls), about $4.4 together; cumulative cap `--max-usd 7.5` (spent so far $2.37). Haiku self-captions: 772 calls, the K2 prompt unchanged.
- **Haiku reads:** question only, Gemini video captions, 1 frame, 5 frames, self-captions, Gemini frame captions: 1,800 calls.
- **Sonnet reads:** primary arms 5 frames and Gemini video captions (600 calls); secondary arms question only and 1 frame (600 calls).
- Prompts, isolation, effort (medium) and parsing as in K0–K2. A failed call counts as wrong.

**Primary hypotheses** (Bonferroni over three: 98.33% CIs; pass if the CI excludes 0 in the predicted direction):

| | Hypothesis | Statistic | Prediction | Development value |
|---|---|---|---|---|
| P1 | Captions lose continuous quantities, keep discrete content | Sonnet, mean (captions − 5 frames) on {direction, speed} minus the same on {action, order, attribute change} | −0.20 | −0.26 (K1, 100 questions) |
| P2 | No format advantage: text is not easier than the pixels it came from | Haiku, self-captions − 5 frames | −0.08 | −0.10 (K2, 50) |
| P3 | A stronger captioner adds usable information | Haiku, Gemini frame captions − self-captions | +0.06 | (new arm; K2's +0.08 included coverage) |

**Secondary** (95% CIs, estimates rather than tests):
- S1. Sonnet captions − 5 frames: −0.07. Retention, with Sonnet's own question-only arm: 0.85.
- S2. Coverage, Haiku: Gemini video captions − frame captions: +0.02 (no directional hypothesis).
- S3. Total, Haiku: Gemini video captions − 5 frames: 0.00 (K0: +0.01).
- S4. P1's contrast with Haiku as the reader: −0.12 (K0: −0.18).
- S5. Matched cost, captions (≈ 245 tokens) − 1 frame (≈ 180): Haiku +0.05 (K0: +0.06), Sonnet +0.02.
- S6. Haiku on captions − Sonnet on 5 frames: −0.13 (K1c).

**Power, stated in advance.** A paired difference on 300 questions with about 30% discordant pairs has SE ≈ 0.032, so the 98.33% CI is about ±0.08. P2 at −0.08 then has about 50% power, and P3 at +0.06 about 30%. P1 (120 vs. 180 questions) has SE ≈ 0.065 and about 75% power at −0.20. **A P2 or P3 that does not pass is not evidence of no effect**; we would report its estimate and CI, and the claim becomes "no evidence of a format advantage (upper bound X)". There are no more unused TempCompass videos, so n cannot grow.

**What each outcome means for the paper.**
- P1 passes: the headline finding. A question-blind caption keeps *what* happened, in what order and what changed, and loses *how fast* and *which way*. TempCompass's own static/temporal split does not predict this.
- P2 at or below 0: the IsoBench contrast. The text advantage holds for lossless text and does not hold for question-blind video captions.
- P3 positive with S2 near 0: the caption index transfers the captioner's knowledge, not its view of the video. P3 near 0 with S2 positive: Gemini helps because it saw more (full-resolution video), not because it knows more.
- All three fail: a negative paper (captions are a lossy index whose loss is not predicted by aspect). We would decide then whether that is worth writing up.

**Run order and stop rules.**
1. Gemini video and frame captions (paid). The first 3 frame-caption outputs are read for sanity only (is it a caption of those frames?). Stop and investigate if more than 2% of windows come back invalid or blocked.
2. Haiku self-captions and all six Haiku reader arms; Sonnet's two primary arms. `--max-5h 0.8`; the isolation and model checks of K0–K2 apply.
3. Sonnet's secondary arms only if the 7-day quota is below 85% after step 2.
4. No decision depends on interim accuracy. `reader.py confirm` (committed with this entry) is run once all arms in a step are complete, and its output is reported here in full.

**Amendment 1 (2026-10-18, after the 3-call frame-caption check, before any reader call on Gemini captions).**
- **Cost.** Gemini encodes each attached image at its default (high) media resolution, about 1,085 tokens per 224-px image, against about 66 tokens per frame for video input. A frame-caption call costs about $0.0099, not $0.0029: the frames run is ≈ $7.6, and C1's Gemini total ≈ $9.8. With the user's approval, the cumulative cap is raised from $7.50 to $13. The default resolution is kept, so the captioner settings stay as tested. A 224-px image carries the same pixels at any encoding resolution.
- **Correction to the coverage row above.** Gemini's video input is not at full resolution either: it is sampled at 1 fps and encoded at about 66 tokens per frame, from the original-resolution file. So coverage compares *the source video, as Gemini's video encoder sees it* with *the reader's five 224-px JPEGs, as Gemini's image encoder sees them*. The term still isolates what the captioner was given; it is not a pure resolution effect.
- **The 3-call check:** all three captions described their frames (a forest time-lapse turning golden; a forward camera move along an autumn trail; two static circle outlines on black). The three windows count towards the 772.

**Amendment 2 (2026-10-19, after all Gemini captions, before any reader call on the Gemini-caption arms).**
- **Captioning finished.** 1,540 of 1,544 windows (772 video, 772 frames) got a caption. The C1 Gemini total is ≈ $6.60 (video $2.18, frames $4.42), and the logged total is $8.98, under the $13 cap. The first attempts hit 429 `RESOURCE_EXHAUSTED`: the project's monthly spend cap in AI Studio, not a rate limit. After the user raised that cap, the run resumed.
- **4 windows have no caption** (0.26%, under the 2% stop rule). A second attempt gave the same result:
  - **Video, `1022115337` windows 0–1:** blocked by Gemini (`PROHIBITED_CONTENT`).
  - **Frames, `1058907236_1021488277_1` and `1096238125_1098970151_1`, window 1:** Gemini answered `ERROR`, the prompt's option for "a description cannot be generated". Both are spliced order videos, and each window spans the cut (squirrel → pineapple; kettlebell swings → cat). Gemini's reasoning says it took the hard cut for a corrupted or test input. The video captioner described the same windows.
- **Rule: the 3 questions are dropped from every arm**, so the paired comparisons stay on the same questions. They are tcc132, tcc153 and tcc169, all order (`EXCLUDED` in `tempcompass.py`; `confirm_items.json` stays as sampled). They are not analysed further. C1 now has **297 questions**: 60 per aspect, and 57 for order.
- **Possible bias:** two of the three are spliced clips, where the cut is the content. So the exclusion slightly removes the kind of item that a frame captioner finds odd. At 3 questions it cannot move any estimate. We note it and do not correct for it.

**Result (2026-10-19; `reader.py confirm`, run once, after all arms of steps 1–2 were complete).**

*Runs.* Gemini: 1,544 caption calls (Amendment 2). Haiku: 772 self-caption calls and 1,782 reader calls (6 arms × 297 questions; one 5-frame answer needed its format retry). Sonnet: 594 reader calls (the 2 primary arms). No call failed. Quota (account-wide, other sessions included): 7-day 75% → 86%; the 5-hour figure peaked at 62%. **Stop rule 3 applied:** the 7-day figure was 86% after step 2, so Sonnet's secondary arms (question only, 1 frame) were not run, and S1 retention and Sonnet's S5 are missing.

*Accuracy.* Tokens are the median input tokens above the question-only prompt (≈ 1,025), i.e. the video content; Sonnet's are the same within 2 tokens.

| Arm | Haiku | Sonnet | Video-content tokens |
|---|---|---|---|
| Question only | 0.411 | — | 0 |
| 1 frame | 0.582 | — | 164 |
| 5 frames | 0.613 | 0.798 | 495 |
| Haiku self-captions | 0.643 | — | 258 |
| Gemini frame captions | 0.791 | — | 204 |
| Gemini video captions | 0.737 | 0.791 | 211 |

| By aspect (60 each; order 57) | action | attribute change | direction | order | speed |
|---|---|---|---|---|---|
| Haiku, question only | 0.63 | 0.37 | 0.28 | 0.37 | 0.40 |
| Haiku, 1 frame | 0.88 | 0.55 | 0.32 | 0.65 | 0.52 |
| Haiku, 5 frames | 0.93 | 0.67 | 0.28 | 0.65 | 0.53 |
| Haiku, self-captions | 0.90 | 0.70 | 0.38 | 0.82 | 0.42 |
| Haiku, Gemini frame captions | 1.00 | 0.90 | 0.55 | 1.00 | 0.52 |
| Haiku, Gemini video captions | 0.95 | 0.82 | 0.43 | 0.93 | 0.57 |
| Sonnet, 5 frames | 0.97 | 0.82 | 0.58 | 0.91 | 0.72 |
| Sonnet, Gemini video captions | 1.00 | 0.82 | 0.55 | 0.98 | 0.62 |

*Primary hypotheses* (98.33% CIs):

| | Observed [98.33% CI] | Prediction | Development value | Verdict |
|---|---|---|---|---|
| P1 aspect, Sonnet | −0.101 [−0.231, +0.028] | −0.20 | −0.26 | **Not passed.** Right sign, half the size. Captions − frames is −0.067 on {direction, speed} and +0.034 on the rest. The loss is in speed (−0.10); direction lost only −0.03. |
| P2 format, Haiku | +0.030 [−0.034, +0.094] | −0.08 | −0.10 | **Not passed, and the CI excludes the prediction.** No evidence of a format effect in either direction; it is at most +0.09. |
| P3 captioner, Haiku | **+0.148 [+0.088, +0.209]** | +0.06 | (+0.08, with coverage) | **Passed**, at 2.5× the predicted size. Positive in every aspect (+0.10 to +0.20). |

*Secondary* (95% CIs):

| | Observed [95% CI] | Prediction |
|---|---|---|
| S1 caption loss, Sonnet (captions − 5 frames) | −0.007 [−0.054, +0.040] | −0.07 |
| S1 retention, Sonnet | not run (stop rule 3) | 0.85 |
| S2 coverage, Haiku (video captions − frame captions) | −0.054 [−0.098, −0.013] | +0.02 |
| S3 total, Haiku (video captions − 5 frames) | **+0.125 [+0.071, +0.178]** | 0.00 |
| S4 aspect contrast, Haiku | −0.055 [−0.172, +0.060] | −0.12 |
| S5 matched cost, Haiku (captions − 1 frame) | **+0.155 [+0.098, +0.212]** | +0.05 |
| S5 matched cost, Sonnet | not run (stop rule 3) | +0.02 |
| S6 Haiku on captions − Sonnet on 5 frames | −0.061 [−0.111, −0.010] | −0.13 |

By aspect, paired differences (descriptive):
- Sonnet, captions − 5 frames: action +0.03, attribute change +0.00, direction −0.03, order +0.07, speed −0.10.
- Haiku, format (self-captions − 5 frames): action −0.03, attribute change +0.03, direction +0.10, order +0.18, speed −0.12.
- Haiku, captioner (frame captions − self-captions): action +0.10, attribute change +0.20, direction +0.17, order +0.18, speed +0.10.
- Haiku, coverage (video captions − frame captions): action −0.05, attribute change −0.08, direction −0.12, order −0.07, speed +0.05.

**What each comparison held constant, checked after the run.** The visual tokens per 5 s window differ between the arms (Gemini's from `results/piece_captions/calls.jsonl`; Claude's from the image rule, 224×126 → 40 per frame):
- Gemini video captions: ≈ 330 (5 frames at 1 fps, ≈ 66 tokens each).
- Gemini frame captions: ≈ 5,500 (≈ 1,100 per image at the default media resolution, Amendment 1).
- Haiku self-captions: 200.
- Neither Gemini arm had audio tokens (the audio track is removed). The frame-caption and self-caption prompts are the video prompt with only the first clause changed to describe frames (`caption.frames_prompt`), so the prompts are not a confound.

So **S2 compares 330 with 5,500 visual tokens**, besides the source video with the reader's frames, and cannot be read as "seeing less helps". And **P3 compares Gemini at 5,500 tokens with Haiku at 200**: it shows that Gemini's caption of the same pixels is more useful, but not whether that comes from what Gemini knows or from the 27× larger visual budget it spent on them. C2 tests this.

**Development vs. confirmatory.** Haiku on 5 frames fell from 0.70 (development) to 0.61; Haiku on Gemini captions rose from 0.71 to 0.74; Sonnet moved less (frames 0.84 → 0.80, captions 0.77 → 0.79). Part of S3's +0.125 (development +0.01) comes from Haiku's frame arm doing worse on these videos.

**Predictions against results.** P1 had the right sign and half the size. P2 had the wrong sign. P3 had the right sign and 2.5× the size. Every prediction based on development data was too pessimistic about captions: S1 (−0.07 → −0.01), S3 (0.00 → +0.13), S5 (+0.05 → +0.16), S6 (−0.13 → −0.06). The K2 estimates came from 50 questions and did not replicate.

**Reading, by the outcomes mapped in advance.**
- **P3 positive, S2 not positive:** the mapped reading is "the caption index transfers the captioner's knowledge, not its view of the video". It stands only if C2 shows the captioner effect at matched visual tokens.
- **P2:** no format advantage for question-blind video captions: at most +0.09 (98.33% upper bound), against IsoBench's 15–29 points for exact text renderings. This is the IsoBench contrast, stated as a bound.
- **P1:** not confirmed. Speed is the only aspect where captions lost for Sonnet. The development finding that captions lose direction did not replicate: Haiku even did better on direction from Gemini captions than from frames (0.43 vs. 0.28).

**Proposed claims for the paper (2026-10-19, pending C2):** a strong captioner's question-blind captions carry its knowledge to a cheap reader. Haiku on Gemini captions beats Haiku on 5 frames by 12.5 points with 43% of the video tokens, and comes within 6 points of Sonnet on frames. Sonnet loses nothing on captions (−0.007) with 43% of the video tokens. The text format adds at most a little. P1 is reported as a null, with speed as an exploratory loss.

### C2 — Does the captioner effect survive at matched visual tokens? (written 2026-10-19, committed before any C2 call)

**Status.** Exploratory robustness check, designed after seeing C1, on the same 297 questions. It cannot change C1's verdicts. Its job is to say whether P3 means "Gemini knows more" or "Gemini looked at the frames with more tokens". The user approved it on 2026-10-19.

**Why.** In C1, Gemini spent ≈ 5,500 visual tokens per window on the reader's frames (image path), while Haiku spent 200 on the same frames for its self-captions. Gemini's video path spends ≈ 330 (66 per frame). So P3 (+0.148) mixes the captioner with a 27× larger visual budget, and S2 (−0.054) mixes the source with a 17× larger budget.

**New arm: frame-clip captions.** Each window's frames (the same JPEGs as the 5-frame arm and the self-captions) are played as a 1 fps clip, one second per frame (`caption.frame_clip`: H.264 at the settings the video arm's cuts use, no audio). Gemini captions the clip through the same video path, with the same video prompt as the video captions (`caption.py --input frameclip`, captioner `gemini-3.8-flash-frameclip`). Gemini then sees exactly the reader's frames, at the video arm's ≈ 66 tokens per frame.
- In a short last window, the clip lasts one second per frame (e.g. 4 s for a 3.2 s window). The prompt states the window's own length, as in the video arm.
- Re-encoding changes the frames by about 3 grey levels out of 255 on average (checked on one video's two windows), the same kind of change the video arm's cuts make.
- Haiku reads the frame-clip captions with the same reader prompt as the other caption arms (`frameclipcaptions`).

**Terms** (all with Haiku as the reader; they extend the *Terms* block in `research_plan_caption_index.md`):

| Term | Difference | Same in both arms | What differs |
|---|---|---|---|
| **C2a, captioner at matched visual tokens** | frame-clip captions − self-captions | pixels (the reader's frames); visual tokens about equal (≈ 330 vs. 200) | the captioning model (and the first clause of the prompt) |
| **C2b, visual budget** | frame captions − frame-clip captions | pixels; captioner | Gemini's visual tokens (≈ 5,500 image path vs. ≈ 330 video path), and the prompt's first clause |
| **C2c, source** | video captions − frame-clip captions | captioner, prompt, video path, ≈ 330 tokens | the source: the original video vs. the reader's 224 px JPEGs |

With C1's format term, **C2c + C2a + P2 = S3** exactly, the total caption effect. This is the decomposition with the visual budget held about constant. C1's split (S2 + P3 + P2) remains as pre-registered.

**Predictions** (95% CIs, paired bootstrap over questions as in C1):
- Frame-clip captions: accuracy ≈ 0.74, about the video captions' 0.737. Reasoning: Gemini's video path reduces every frame to ≈ 66 tokens, so the source's higher resolution mostly does not reach the captioner, and the frames are the same 1 fps instants.
- **C2a ≈ +0.10.** This is the key quantity.
- C2b ≈ +0.05 (the rest of P3's +0.148).
- C2c ≈ 0.00.

**What each outcome means.**
- **C2a's CI excludes 0, positive:** the captioner effect holds at matched visual tokens. The claim "a strong captioner's knowledge carried to a cheap reader" stands. C2b is reported as the extra gain from the larger visual budget.
- **C2a's CI includes 0, and C2b is positive:** P3 came mostly from Gemini's larger visual budget, not from what it knows. The claim becomes "spend the visual budget once, at captioning time, and query cheaply". That is still an index result, but not a knowledge result. P3's wording in the paper changes accordingly.
- **C2a negative:** at matched visual tokens, Haiku's own captions are as good or better. The captioner effect is a budget effect.
- **C2c** has no directional hypothesis. If it is not ≈ 0, the source resolution matters even through the video path.

**Precision.** As in C1: SE ≈ 0.03 for a paired difference on 297 questions, so the 95% CI is about ±0.06.

**Run and stop rules.**
1. **3-call check** (`--limit-calls 3`): read the three captions (do they describe those frames?) and the logged video tokens. Stop and fix if a full window does not come out at ≈ 66 tokens per frame (≈ 330), because the token match would then fail.
2. **Captions:** 761 windows (the 297 questions' videos), ≈ $0.0027 each, ≈ $2.0. Cap `--max-usd 11.5` (logged total $8.98; the approved cap is $13). Stop and investigate if more than 2% of windows get no caption. Otherwise, a question with a window without a caption is dropped from every C2 comparison (`reader.py run --skip-missing`, and `c2` uses only questions all its arms answered), and listed here.
3. **Reader:** Haiku, 297 calls, `--max-5h 0.8`, the isolation and model checks of K0–C1.
4. **Analysis:** `reader.py c2` is run once, when both steps are complete, and its output is reported here in full.

**Amendment 1 (2026-10-19, after the 3-call check, before the rest of the run).**
- **The check:** the three captions describe their frames: two static circle outlines on black; a split-screen forest path turning from green to autumn gold; the same forward walk along the autumn path. They cost $0.011.
- **Tokens:** they came out at 280 per full window (56 per frame) and 168 for a 3-frame window, not ≈ 330. That breaks rule 1 as written. But the video arm got **exactly the same counts for the same windows** (280, 168, 280). Gemini's tokens per video frame depend on the video's frame size: these are split-screen videos (56 per frame), and most videos get 66. The ≈ 330 in rule 1 was the median over windows, not a per-window target.
- **What the rule meant:** the frame clip must cost Gemini the same visual tokens as the video for the same window. That holds. **Rule 1 is restated that way, and the run continues.**
- `reader.py c2` now also prints, per window, whether the two arms' tokens are equal.

**Run status (2026-10-19, evening).**
- Captioning ran in 4 parallel shards (`caption.py --shard K/4`; the spend cap is checked against the shared log before every call).
- It stopped at **450 of 761 windows**, with no window lacking a caption. All four shards got `429 RESOURCE_EXHAUSTED`: the AI Studio project's monthly spend cap, as in C1 Amendment 2, not our `--max-usd`.
- Logged spend is $10.29. The remaining 311 windows need ≈ $0.90.
- No reader call has been made on the frame-clip captions. The run resumes after the user raises the cap.

**Run completed (2026-10-20).**
- **Captions.** The user raised the AI Studio spend cap, and the remaining 311 windows were captioned in 4 shards. All 761 windows have a caption; the 4 failed calls were the 429s of 2026-10-19, retried.
- **Cost.** C2's Gemini calls cost $2.26 (765 calls). The logged total is $11.24, under the run's `--max-usd 11.5` and the approved $13.
- **Reader.** Haiku read the frame-clip captions in 297 calls, all valid at the first try. Quota (account-wide): 5-hour 23% → 24%, 7-day 92%.

**Result (2026-10-20; `reader.py c2`, run once, after both steps were complete; output saved in `logs/c2_result.txt`).** No question was dropped: all 297 count.

*Visual tokens.*
- The video arm and the frame-clip arm both have a median of 330 visual tokens per window. They are equal in 693 of the 761 windows.
- The other 68 are all a video's short last window. There the clip has one frame more (56 or 66 tokens): it plays each of the reader's frames for one second, while Gemini samples the original last window at 1 fps and gets one frame fewer. In total, the clip has 216,696 tokens against the video's 212,338 (+2%).
- The image path has a median of 5,500 visual tokens per window.

*Accuracy (Haiku reads):*

| Arm | All | Action | Attribute change | Direction | Order | Speed |
|---|---|---|---|---|---|---|
| 5 frames | 0.613 | 0.93 | 0.67 | 0.28 | 0.65 | 0.53 |
| Self-captions | 0.643 | 0.90 | 0.70 | 0.38 | 0.82 | 0.42 |
| **Gemini frame-clip captions** | **0.724** | 0.98 | 0.85 | 0.43 | 0.88 | 0.48 |
| Gemini frame captions (image path) | 0.791 | 1.00 | 0.90 | 0.55 | 1.00 | 0.52 |
| Gemini video captions | 0.737 | 0.95 | 0.82 | 0.43 | 0.93 | 0.57 |

*Effects (95% CIs):*

| Term | Observed | Prediction |
|---|---|---|
| Frame-clip captions, accuracy | 0.724 | ≈ 0.74 |
| **C2a, captioner at about matched visual tokens** (frame-clip captions − self-captions) | **+0.081 [+0.030, +0.131]** | +0.10 |
| C2b, visual budget (frame captions − frame-clip captions) | +0.067 [+0.030, +0.108] | +0.05 |
| C2c, source (video captions − frame-clip captions) | +0.013 [−0.030, +0.061] | 0.00 |

- **Two decompositions.**
  - At about matched visual tokens: S3 (+0.125) = C2c (+0.013) + C2a (+0.081) + P2 (+0.030).
  - C1's P3 (+0.148) = C2a + C2b, so about half captioner and half visual budget.
- **By aspect** (paired differences, descriptive):
  - C2a: action +0.08, attribute change +0.15, direction +0.05, order +0.05, speed +0.07.
  - C2b: action +0.02, attribute change +0.05, direction +0.12, order +0.12, speed +0.03.
  - C2c: action −0.03, attribute change −0.03, direction 0.00, order +0.05, speed +0.08.

**Predictions against results.** All four came within 0.02, and all three effects had the predicted sign.

**Reading,** by the mapped outcomes as reworded in D2:
- **C2a is positive: the captioner effect holds across model families at about matched visual tokens.** "About": Gemini's video path still spent ≈ 330 visual tokens per window on the frames, against Haiku's 200 (1.65×). C3 matches them exactly.
- **C2b: the visual budget accounts for the rest of P3.** Gemini's image path (≈ 5,500 tokens per window) adds ≈ +0.07 over its video path on the same pixels, most on direction and order (+0.12 each).
- **C2c ≈ 0, as predicted.** With the same captioner, prompt and tokens, captions of the original video and captions of the reader's 224 px frames are about equally useful. Gemini's video path reduces each frame to ≈ 66 tokens, so the source's higher resolution mostly does not reach the captioner.
- **Cost** (A2's method): Haiku on the frame-clip captions costs $0.000167 per question, and the captioning $0.0076 per video. Against Sonnet on frames, Q\* = 1.9 [1.8, 2.1], at −0.074 [−0.125, −0.024] accuracy.
- **Go/no-go (G5, plan §0):** case 2 at least holds, since C2a is positive. C3 decides between case 1 and case 2.

### D2 — Terms renamed; Claude captioner as the main chain (2026-10-20)

**Decided with the user** after a review of the ACL plan (`acl2027_plan.md`, revised the same day). No analysis changes, and no run.

**Terms.** The definitions are in the *Terms* block of `research_plan_caption_index.md`.

| Old word (K2, C1, C2) | New term | Why |
|---|---|---|
| format effect (C1, P2) | **modality effect** | "Format" suggests a change of encoding (JSON to YAML). What changes is the type of input: the reader's own text instead of the images. |
| coverage (C1, S2) | **source effect** | What changed is the captioner's input itself (the video file vs. the reader's 224 px JPEGs, the same 1 fps instants), not how much of the video it saw. C1's S2 also differs 17× in visual tokens, so it is source + visual budget. |
| visual budget (C2b) | **visual budget effect** | Used in C2, now defined. |
| "knowledge" (C1 claims; C2's outcome mapping) | **captioner effect** | No arm separates what a captioner perceives, recognizes or writes. |

- **The entries above keep their old words.** K2 and C1 were fixed before their runs, and rewording them would blur what was fixed in advance. Read "format" there as modality, and "coverage" as source.
- **C2's mapped outcome for C2a positive** ("the claim 'a strong captioner's knowledge carried to a cheap reader' stands") now reads: the captioner effect holds across model families, at about matched visual tokens.

**Design.**
- **Claude carries the main decomposition.** In C3 (entry to be written and committed before its run), Sonnet captions exactly the reader's 5 frames, with the self-caption prompt, and Haiku reads the captions.
  - The visual tokens are identical by construction: one image-token rule for all Claude models. Haiku's and Sonnet's 5-frame reader calls have median inputs of 1,525 and 1,527 tokens.
  - The source effect is 0 by design.
- **Gemini is kept only where it adds something Claude can't:** video input (the source effect), a second model family, and dollars instead of quota. No new Gemini arms beyond finishing C2.
- **P3 remains the pre-registered result.** C3 is exploratory and replicates P3 within one family.

**Cost framing** (from C1's logged calls; plan §3).
- For one reader, captions save little per question: Haiku 18%, Sonnet 30% of the list-price cost. The fixed prompt (≈ 1,025 tokens) and thinking tokens dominate, so captions cut video tokens to 43% but input tokens only to 81%.
- The economic result is the reader downgrade: Haiku on captions costs 1/24 of Sonnet on frames, and repays ≈ $0.0073 of captioning per video after ≈ 2 questions, at 6 points lower accuracy.

### C3 — Claude's captioner effect at identical input (written 2026-10-20, committed before any C3 call)

**Status.** Exploratory: designed after C1, on the same 297 questions. It cannot change C1's verdicts. It replicates P3 within one model family, with the captioner given exactly the self-captions' input. The run starts after the user approves this entry, and after the 7-day quota reset (Oct 20, 17:00 UTC).

**Why.** P3 (+0.148) compared Gemini at ≈ 5,500 visual tokens per window with Haiku at 200. C2 brings Gemini down to ≈ 280–330, about matched, and across model families. Every Claude model counts image tokens by the same rule (C1's 5-frame reader calls: median input 1,525 tokens for Haiku, 1,527 for Sonnet). So a Claude captioner given the same frames spends exactly Haiku's visual tokens.

**New arm: Sonnet captions (`sonnetcaptions`).**
- **Captioning:** Sonnet captions each window from the same five 224 px JPEGs as the self-captions, with the same prompt (`caption.frames_prompt`) and effort (medium). One isolated call per window, without the question (`self_caption.py --model sonnet`, folder `claude-sonnet-5-5-frames`).
- **Reading:** Haiku reads the captions with the reader prompt of every caption arm.

**Terms** (Haiku reads; names as in D2):

| Term | Difference | Same in both arms | What differs |
|---|---|---|---|
| **C3a, captioner** | Sonnet captions − Haiku self-captions | pixels, prompt, visual tokens (200 per full window), effort | the captioning model |
| **C3b, total within Claude** | Sonnet captions − 5 frames | pixels | C3a + modality (P2); source and visual budget are 0 by design |
| **C3c, index vs. direct** | Haiku on Sonnet captions − Sonnet on 5 frames | pixels; Sonnet sees them in both | Sonnet captions once and Haiku answers, vs. Sonnet answers each question |
| **C3d, family** | Sonnet captions − Gemini frame-clip captions (C2) | pixels | captioner and family; visual tokens about equal (200 vs. ≈ 280–330); prompt's first clause (frames vs. video) |

**Predictions** (95% CIs, paired bootstrap over questions, as in C2):
- **Haiku on Sonnet captions ≈ 0.73.**
  - Sonnet reads these frames far better than Haiku: 0.798 vs. 0.613 on 5 frames.
  - Sonnet loses almost nothing through a strong captioner's captions (S1: −0.007).
  - A question-blind caption should carry part of Sonnet's advantage, though less than Gemini's frame captions did (0.791, with 27× the visual tokens).
- **C3a ≈ +0.09.** The key quantity.
- **C3b ≈ +0.12.**
- **C3c ≈ −0.07.**
- **C3d ≈ 0.00.** No directional hypothesis.

**What each outcome means.** This is the go/no-go rule of G5 (plan §0), stated with C2:
1. **C3a's CI excludes 0, positive:** the captioner effect holds with identical input, so it is not a visual budget effect. It is the paper's headline; target ACL main.
2. **C3a not positive, C2a positive:** the effect is shown only across families (Gemini as captioner). The claim names the captioner, and the length and venue are decided at G9.
3. **C3a and C2a not positive, C2b positive:** the claim becomes "spend the visual budget once, at captioning time".
4. **None positive:** the paper is the cost result plus the bound on the modality effect; target Findings.
- **C3a negative:** Haiku's own captions are as useful as Sonnet's, which would be reported as such.

**Precision.** As in C2: SE ≈ 0.03 for a paired difference on 297 questions, so the 95% CI is about ±0.06. If the effect is +0.09, the power is about 85%.

**Run and stop rules.**
1. **3-call check** (`--limit 3`). Read the three captions: do they describe those frames? Compare each call's input tokens with Haiku's self-caption call for the same window. Stop and investigate if any differs by more than 10 tokens: the image tokens must be equal, and the text may differ by a few tokens between models.
2. **Captions:** the remaining 758 windows, with `--max-5h 0.7 --max-7d 0.5`.
   - Sonnet reader calls in C1 took ≈ 0.06–0.09 points of the 5-hour window each, so expect ≈ 0.1 per caption call. The run then needs two 5-hour windows; done windows are skipped when it resumes.
   - Expect ≈ 10–15 points of the 7-day figure.
   - Stop and investigate if more than 2% of windows get no caption. Otherwise, a question with a missing window is dropped from every C3 comparison (`--skip-missing`) and listed here.
3. **Reader:** Haiku, 297 calls, `--max-5h 0.8`, with the isolation and model checks of K0–C2.
4. **Analysis:** `reader.py c3` is run once, when steps 2 and 3 are complete. Its full output is reported here.

```bash
.venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --limit 3
.venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --max-5h 0.7 --max-7d 0.5
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms sonnetcaptions --skip-missing
.venv/bin/python scripts/index_qa/reader.py c3
```

**Amendment 1 (2026-10-20, before any C3 call): results by video type.** A3 (below; exploratory, on C1 and C2's data) found that the caption arms' results depend on how TempCompass built the video. This amendment adds secondary, descriptive predictions by video type, fixed before any C3 call. Nothing above changes: C3a over all questions stays the key quantity, and the outcome mapping stands.
- **Report:** `reader.py types`, run once after step 4. It adds the C3 rows once the `sonnetcaptions` arm exists, and gives:
  - accuracy by video type;
  - C3a, C3b and C3c by video type, with 95% CIs from resampling clusters of questions whose videos share a filmed clip;
  - every C3 term over all questions, with the clustered CI next to the question-level one.
- **Predictions:** Haiku on Sonnet's captions scores plain 0.87, two-clip 0.88, reversed 0.55, stacked 0.36. With C1's arms, that gives:

| Term | Plain (137) | Two-clip (45) | Reversed (74) | Stacked (41) |
|---|---|---|---|---|
| C3a, Sonnet captions − self-captions | +0.07 | +0.06 | +0.12 | +0.04 |
| C3b, Sonnet captions − 5 frames | +0.07 | +0.28 | +0.17 | −0.05 |
| C3c, Haiku on Sonnet captions − Sonnet on 5 frames | −0.02 | −0.03 | −0.09 | −0.30 |

  Weighted by the type counts, these give C3a ≈ +0.08, within the precision of the +0.09 predicted above. That prediction stands.
- **Directional hypotheses.** Each is read from its 95% clustered CI. With 41–137 questions per type, the CIs are about ±0.08 to ±0.15, so only large effects can show.
  1. **Stacked: C3c < 0.** A question-blind caption does not state how two copies of a clip differ in speed, even when a strong captioner writes it. Sonnet reading the frames sees the difference (0.66 vs. Haiku 0.41 on 5 frames).
  2. **Reversed: C3a > 0,** as C2a was (+0.14 [+0.04, +0.24]).
  3. **Two-clip: C3b > 0,** as S3 (+0.33) and P2 (+0.22) were.
- **Use:** descriptive, for the paper's analysis section. If hypothesis 1 holds, the limitations state that question-blind captions drop comparisons between things shown at the same time.

**Result (2026-10-20; `reader.py c3` and `reader.py types`, each run once after steps 2–3 were complete; output in `results/index_qa/reports/c3.txt`, `types.txt`).**
- **Run.**
  - All 876 windows were captioned and no question was dropped: 297 in every C3 arm.
  - Sonnet's caption calls took exactly 2 more input tokens than Haiku's for the same window, in all 876 (the median and range were both 2).
  - Quota for C3–C6 together: the 7-day figure went from 3% to 7%. The 5-hour window reached 80% at 19:21 UTC.
  - One run crashed while Claude Code auto-updated (`FileNotFoundError: 'claude'`). It was resumed, and done windows were skipped.
- **Accuracy:**
  - Haiku on 5 frames: 0.613.
  - Haiku on its self-captions: 0.643.
  - **Haiku on Sonnet captions: 0.707** (predicted 0.73).
  - Sonnet on 5 frames: 0.798.

| Term | Result, 95% CI over questions; clustered | Prediction |
|---|---|---|
| **C3a, captioner** (Sonnet − Haiku captions, identical input) | **+0.064 [+0.017, +0.111]; [+0.011, +0.119]** | +0.09 |
| C3b, total within Claude (Sonnet captions − 5 frames) | +0.094 [+0.044, +0.148]; [+0.042, +0.147] | +0.12 |
| C3c, index vs. direct (Haiku on Sonnet captions − Sonnet on frames) | −0.091 [−0.145, −0.040]; [−0.151, −0.032] | −0.07 |
| C3d, family (Sonnet − Gemini frame-clip captions) | −0.017 [−0.071, +0.037] | 0.00 |

- **Go/no-go (G5): case 1.** C3a's CI excludes 0 and is positive, so the captioner effect holds with identical pixels, prompt and visual tokens. It is not a visual budget effect. The effect is smaller than predicted (+0.064 vs. +0.09); the prediction lies inside the CI.
- **By aspect** (descriptive), C3a is:

  | Action | Attribute change | Direction | Order | Speed |
  |---|---|---|---|---|
  | +0.07 | +0.02 | +0.03 | +0.09 | +0.12 |

- **Amendment 1, by video type** (clustered CIs). Haiku on Sonnet captions scores plain 0.85, two-clip 0.93, reversed 0.45 and stacked 0.44 (predicted 0.87, 0.88, 0.55, 0.36).

| Term | Plain (137) | Two-clip (45) | Reversed (74) | Stacked (41) |
|---|---|---|---|---|
| C3a | +0.06 [−0.01, +0.12] | +0.11 [−0.03, +0.24] | +0.01 [−0.08, +0.10] | +0.12 [−0.03, +0.28] |
| C3b | +0.05 [+0.00, +0.11] | **+0.33 [+0.20, +0.48]** | +0.07 [−0.04, +0.18] | +0.02 [−0.16, +0.21] |
| C3c | −0.04 [−0.10, +0.02] | +0.02 [−0.10, +0.13] | **−0.19 [−0.32, −0.07]** | −0.22 [−0.44, +0.00] |

  1. **Stacked, C3c < 0: not shown at 95%.** The estimate is −0.22; the 95% CI's upper end is 0.00, and the 90% CI is [−0.41, −0.05]. C4 (stacked pairs) tests the same reading with more power.
  2. **Reversed, C3a > 0: not supported** (+0.01). Unlike Gemini (C2a +0.14 on reversed), Sonnet as a captioner does not report reversals more often than Haiku does. C3c on reversed is −0.19, so Sonnet sees the reversal when it reads the frames but does not write it into a question-blind caption.
  3. **Two-clip, C3b > 0: holds** (+0.33).
- **Sonnet reading its own captions** (plan §6; the 297 confirmatory reads of C4's run; not predicted in C3):
  - **Sonnet on Sonnet captions scores 0.710.** That is −0.088 below Sonnet on frames, [−0.135, −0.044]; clustered [−0.140, −0.036].
    - By type: plain −0.01, two-clip 0.00, **reversed −0.18 [−0.30, −0.05], stacked −0.27 [−0.47, −0.09]**.
  - **Haiku and Sonnet read Sonnet's captions equally well:** −0.003 [−0.047, +0.040]. Once the video is in these captions, a 20× cheaper reader loses nothing.
  - **Sonnet reads Gemini's video captions better than its own:** 0.791 vs. 0.710, +0.081 [+0.034, +0.128]; clustered [+0.031, +0.129]. The video captions come from a different captioner, which sees the video, at a much larger visual budget.
  - **Reading:** with question-blind captions, the reader's accuracy is capped by what the captions hold. That cap is the captioner's question-blind description, not the reader's strength.
    - Haiku is below the cap on frames (0.613), so captions help it.
    - Sonnet is above it (0.798), so its own captions cost it 0.09, almost all on reversed and stacked videos.
    - S1 (Sonnet on Gemini video captions ≈ Sonnet on frames) holds only for a captioner that writes those facts down.

### A1 — Thinking tokens in C1 (exploratory analysis, 2026-10-20; not written in advance)

**Why.** The user asked whether we quantify the readers' thinking tokens, and whether they correlate between frames and captions. The analysis was adopted as exploratory (G2, plan §6). It made no new calls; the effort sweep that could test whether thinking *helps* was declined. Reproduce with `reader.py thinking [--set dev]`.

**What thinking is here.** All reader calls run at effort medium, where the model decides per call whether to think, so thinking is observed, not set. In all 3,042 reader calls, the logged thinking tokens are above 0 exactly when the call has a thinking block.

**Results, confirmatory set (297 questions):**

| Arm | Calls that think | Mean thinking tokens | Accuracy, no thinking | Accuracy, thinking |
|---|---|---|---|---|
| Haiku, 5 frames | 0.25 | 79 | 0.61 | 0.61 |
| Haiku, 1 frame | 0.22 | 43 | 0.57 | 0.63 |
| Haiku, self-captions | 0.54 | 104 | 0.75 | 0.55 |
| Haiku, Gemini frame captions | 0.44 | 73 | 0.90 | 0.65 |
| Haiku, Gemini video captions | 0.45 | 74 | 0.84 | 0.62 |
| Sonnet, 5 frames | 0.44 | 80 | 0.93 | 0.63 |
| Sonnet, Gemini video captions | 0.24 | 24 | 0.84 | 0.64 |

- **Development set (100 questions, other videos):** the same pattern.
  - Haiku on captions: 0.85 without thinking vs. 0.51 with it.
  - Haiku on 5 frames: 0.67 vs. 0.77.
  - Sonnet on captions: 0.86 vs. 0.50.
  - Sonnet on 5 frames: 0.94 vs. 0.67.
- **Where thinking goes:** speed and direction draw the most thinking in every arm, and action almost none.
- **Correlation across arms.** The same questions draw thinking in different arms. Spearman over questions, then the mean within aspects:
  - Haiku, frames vs. captions: +0.38 and +0.25.
  - Sonnet, frames vs. captions: +0.47 and +0.31.
  - Haiku, Gemini video vs. frame captions: +0.60 and +0.44.
- **Cost.** Thinking is 2–5% of reader tokens. Sonnet thinks 3.3× less on captions than on frames (24 vs. 80 tokens on average), which is part of why captions save Sonnet 30% per question.

**Reading** (hypotheses, untested):
- Thinking marks hard questions in every arm.
- It marks *wrong answers* when the reader reads text, and for Sonnet on frames, but not for Haiku on 40-token frames. One explanation: from a caption, the reader can tell that the answer is not there; from small frames, Haiku does not know what it is missing.

**Escalation (illustration, post hoc).** Haiku answers from the Gemini captions; when it thinks, the question goes to Sonnet on 5 frames (45% of questions). That gives 0.774 at $0.0022 per question, against 0.798 at $0.0041 for Sonnet on frames every time, and 0.737 at $0.00017 for Haiku on captions alone. On the development set: 0.800 at $0.0020 vs. 0.840 at $0.0040. It is a middle point on the accuracy–cost curve, not a win. The rule (escalate on any thinking) was chosen after seeing the data, without tuning.

**Use in the paper:** thinking tokens are part of the cost account (G4), and this is an exploratory paragraph or appendix section. It is not a claim.

### A2 — Cost account for C1's arms (2026-10-20; G4, first part; no new calls)

**Inputs.**
- **Reader calls:** C1's, from `results/index_qa/claude/*.jsonl`.
- **Claude list prices:** from the claude-api skill's model table (cached 2026-10-06), in $ per million input / output tokens, with thinking billed as output: Haiku 5.5 $0.10 / $0.50, Sonnet 5.5 $2 / $10. Computed from the logged tokens, they equal the CLI's `costUSD` on all 2,376 calls (largest relative difference 4e-16).
- **Gemini captioning:** from `results/piece_captions/calls.jsonl`, by mode, as `caption.py` logs it ($0.50 / $3.00 per million input / output tokens, thinking included).
- **Haiku self-captions:** from their caption files, at the list prices.
- Reproduce with `reader.py cost`.

**Per arm** (297 questions). Tokens are per question; the dollar figures are list prices; seconds are the median wall clock of a CLI call, start-up included:

| Arm | Accuracy | Input tokens (median) | Output tokens incl. thinking (mean) | $ per question | Seconds | Captioning $ per video (once) |
|---|---|---|---|---|---|---|
| Haiku, question only | 0.411 | 1,025 | 12 | 0.000109 | 2.5 | — |
| Haiku, 1 frame | 0.582 | 1,189 | 54 | 0.000146 | 2.2 | — |
| Haiku, 5 frames | 0.613 | 1,520 | 90 | 0.000204 | 2.4 | — |
| Haiku, self-captions | 0.643 | 1,283 | 115 | 0.000189 | 2.9 | 0.00058 |
| Haiku, Gemini frame captions | 0.791 | 1,229 | 85 | 0.000167 | 3.1 | 0.01461 |
| Haiku, Gemini video captions | 0.737 | 1,236 | 85 | 0.000168 | 3.1 | 0.00726 |
| Sonnet, 5 frames | 0.798 | 1,522 | 93 | 0.004112 | 3.0 | — |
| Sonnet, Gemini video captions | 0.791 | 1,238 | 35 | 0.002868 | 4.1 | 0.00726 |

**Break-even Q\*** (Prop. 2 for one caption arm against one frame arm). Q\* is the number of questions per video after which captioning once costs less in total. The CIs are 95%, from resampling questions; with one question per video, that resamples videos too.

| Captions | Frames | Q\* | Accuracy, captions − frames |
|---|---|---|---|
| Haiku, Gemini video captions | Haiku, 5 frames | 202 [156, 279] | +0.125 [+0.071, +0.178] |
| Sonnet, Gemini video captions | Sonnet, 5 frames | 5.8 [5.2, 6.5] | −0.007 [−0.054, +0.040] |
| Haiku, Gemini video captions | Sonnet, 5 frames | 1.8 [1.7, 2.0] | −0.061 [−0.111, −0.010] |
| Haiku, Gemini frame captions | Sonnet, 5 frames | 3.7 [3.5, 4.0] | −0.007 [−0.057, +0.044] |

**Reading.**
- **For one reader, captions save little per question.** The fixed prompt (≈ 1,025 tokens) dominates the input. With Haiku, the index pays off only after ≈ 200 questions per video, though Haiku is also 12.5 points more accurate on the captions. With Sonnet, it pays off after ≈ 6 questions, at equal accuracy.
- **The reader downgrade is what pays.** Haiku on Gemini's video captions costs 1/24 of Sonnet on frames per question, and repays the captioning after 2 questions, at 6 points lower accuracy.
- **New, exploratory: Haiku on Gemini's frame captions matches Sonnet on frames** (−0.007 [−0.057, +0.044]) at 1/25 of the cost per question. It repays the captioning after ≈ 4 questions.
  - Those captions cost twice the video captions ($0.0146 per video), because Gemini's image path spends ≈ 5,500 visual tokens per window. That spend is the C2 question.
  - For cost, the comparison is fair: the index cost includes the spend.
- **Latency per call does not favour captions.** The caption arms' median call is 0.1–1.1 s slower than the frame arms' with the same reader. These are wall-clock times of a CLI call, start-up included, and were not controlled for time of day.
- **Still to add to G4:** the frame-clip captions (C2), Sonnet's captions (C3) and the 448 px arm, once they are run.

### A3 — Results by video type, and the questions where frames and captions disagree most (exploratory analysis, 2026-10-20; not written in advance)

**Why.** The user asked whether we had looked at the questions where frames and captions differ most. We had not. A first look showed that those questions are almost all on videos TempCompass built (reversed, stacked or two-clip), so the analysis was redone by video type. It made no new calls. Reproduce with `reader.py types [--set dev]`.

**Video types.** TempCompass builds videos so that one frame or the question's wording is not enough. `tempcompass.video_type` reads the type from the video ID; frames of one stacked and two two-clip videos were checked by eye.
- **Plain** (`<id>`): one clip as filmed.
- **Reversed** (`<id>_reverse`): a clip played backwards. TempCompass asks the same question about the forward clip, with the mirrored answer.
- **Stacked** (`<id>_concat_<k>`): the same clip twice, one above the other, at equal or different speeds.
- **Two-clip** (`<a>_<b>_<k>`): two different clips, one after the other or one above the other.

**Video type is nearly the same as aspect.** In the confirmatory set (297 questions):
- Every stacked question (41) is about speed, and every two-clip one (45) about order.
- Reversed videos carry 40 of the 60 direction questions and 34 of the 60 attribute-change ones.
- All 60 action questions are on plain videos, as are 137 questions in all.

So C1's aspect contrast (P1) mostly compared video types.

**Accuracy by video type** (confirmatory set):

| Arm | Plain (137) | Two-clip (45) | Reversed (74) | Stacked (41) | All (297) |
|---|---|---|---|---|---|
| Haiku, question only | 0.60 | 0.31 | 0.16 | 0.34 | 0.411 |
| Haiku, 1 frame | 0.77 | 0.62 | 0.27 | 0.49 | 0.582 |
| Haiku, 5 frames | 0.80 | 0.60 | 0.38 | 0.41 | 0.613 |
| Haiku, self-captions | 0.80 | 0.82 | 0.43 | 0.32 | 0.643 |
| Haiku, Gemini frame-clip captions | 0.88 | 0.87 | 0.57 | 0.34 | 0.724 |
| Haiku, Gemini frame captions | 0.92 | 1.00 | 0.64 | 0.41 | 0.791 |
| Haiku, Gemini video captions | 0.86 | 0.93 | 0.53 | 0.49 | 0.737 |
| Sonnet, 5 frames | 0.89 | 0.91 | 0.64 | 0.66 | 0.798 |
| Sonnet, Gemini video captions | 0.93 | 0.98 | 0.57 | 0.54 | 0.791 |

**Differences by video type.** The CIs are 95%, from resampling clusters of questions whose videos share a filmed clip (see *Clusters* below):

| Difference | Plain | Two-clip | Reversed | Stacked |
|---|---|---|---|---|
| P2, modality (Haiku self-captions − 5 frames) | −0.01 [−0.07, +0.06] | **+0.22 [+0.10, +0.35]** | +0.05 [−0.04, +0.15] | −0.10 [−0.26, +0.05] |
| P3, captioner (Gemini frame captions − self-captions) | +0.12 [+0.06, +0.19] | +0.18 [+0.09, +0.28] | +0.20 [+0.09, +0.32] | +0.10 [+0.02, +0.19] |
| C2a, captioner at about matched tokens | +0.08 [+0.01, +0.15] | +0.04 [−0.09, +0.17] | +0.14 [+0.04, +0.24] | +0.02 [−0.08, +0.14] |
| S1, Sonnet: captions − 5 frames | +0.04 [−0.01, +0.09] | +0.07 [−0.02, +0.16] | −0.07 [−0.19, +0.05] | −0.12 [−0.31, +0.05] |
| S3, Haiku: Gemini video captions − 5 frames | +0.06 [−0.01, +0.12] | +0.33 [+0.22, +0.47] | +0.15 [+0.04, +0.27] | +0.07 [−0.03, +0.17] |

**Development set** (100 questions: 60 plain, 10 two-clip, 17 reversed, 13 stacked):
- **Stacked videos are where captions lose,** for both readers: Sonnet captions − frames −0.38 [−0.69, −0.07], Haiku −0.31 [−0.58, −0.08].
- On plain videos the differences are small: Sonnet −0.03, Haiku +0.03.
- **Two-clip videos don't test the confirmatory pattern here.** Haiku on Gemini captions − frames is +0.30 [−0.10, +0.70] (10 questions). K2's self-captions cover only 5 two-clip questions (difference 0.00).

**Why Haiku's 5-frame accuracy fell from 0.70 (dev) to 0.61 (confirmatory).** Part of it is the mix of video types.
- On plain videos it is 0.80 in both sets.
- Built videos are 40% of the development set and 54% of the confirmatory set.
- Weighting the confirmatory per-type accuracies by the development mix gives 0.66, so about half of the drop.
- The rest is stacked videos: 0.69 on 13 development questions, 0.41 on 41 confirmatory ones.

**Wrong answers fall to the question's prior in both modalities.** Haiku with the question only scores 0.16 on reversed videos, below chance, so the wording favours the forward event. Among wrong answers on reversed videos, the share equal to Haiku's question-only answer is:

| Reader | Frames | Captions |
|---|---|---|
| Haiku | 5 frames 0.67 (46 wrong), 1 frame 0.63 (54) | self-captions 0.55 (42); Gemini captions 0.51–0.67 (27–35) |
| Sonnet | 5 frames 0.56 (27) | Gemini video captions 0.53 (32) |

Captions are not more prior-bound than frames. Both readers fall back to the forward reading when they miss the reversal. The examples below had suggested that captions carry the prior; the counts don't support "more than frames".

**The most discordant questions** (confirmatory set):
- **Sonnet on frames right, at most 1 of 5 caption arms right: 19 questions.** 8 are reversed, 7 stacked and 4 plain. Two kinds of example:
  - **Reversed video, captions narrate the forward event.**
    - Salt pile, answer "shrinking" (tcc195): all five of Gemini's video captions describe salt pouring onto the pile ("piling up"). Haiku's self-captions say the pile "grows taller"; only its last window notes that "the pile is lower".
    - Zebras, answer "leftward" (tcc265): both captioners write "toward the right" / "left to right".
    - Only Sonnet on frames answered either one correctly.
  - **Stacked video, captions don't compare the two copies.** Jump rope, answer "the man above is faster" (tcc005):
    - Gemini: "An athletic man … jumps rope in a split-screen video".
    - Haiku: one woman skipping rope.
    - No caption says how the two copies differ.
- **At least 4 of 5 caption arms right, all 3 frame arms wrong: 5 questions.** 2 are plain, 2 reversed and 1 two-clip. In each, the caption commits to the event:
  - Reversed virus animation, answer "it assembles" (tcc189): Gemini writes "a coronavirus particle … forms and binds with small, blue Y-shaped antibodies". Haiku's own caption says the sphere "grows larger in the foreground", and that arm was wrong.
  - Two-clip, answer "at the same time" (tcc136): Gemini writes "a red squirrel picks up a walnut … in the top frame, while below, a hand reaches …". Haiku's own caption makes it a sequence ("Then a person's hand reaches in"), and that arm was wrong.
  - Reversed camera pan, answer "panning to the left" (tcc296): Gemini writes "pans to the left"; Haiku's own caption says "drifts sideways".

**Clusters.** C1 sampled one question per video, but reversed, stacked and two-clip videos reuse filmed clips. The 297 questions fall into 156 clusters of videos that share a clip, directly or through another video (`reader.clip_clusters`; largest 12 questions). Resampling clusters instead of questions barely changes the CIs (95%):

| Term | Question-level | Clustered |
|---|---|---|
| P2 | +0.030 [−0.020, +0.081] | [−0.019, +0.079] |
| P3 | +0.148 [+0.098, +0.199] | [+0.100, +0.199] |
| C2a | +0.081 [+0.030, +0.131] | [+0.035, +0.125] |
| S1 | −0.007 [−0.054, +0.040] | [−0.060, +0.047] |
| S3 | +0.125 [+0.071, +0.178] | [+0.077, +0.172] |

C1's and C2's conclusions stand. The paper should report the clustered CIs, or state that they agree.

**Reading** (hypotheses, untested):
- **The captioner effect is the robust one.** P3 is positive in every video type, and C2a is clearest on reversed and plain videos.
- **The modality effect depends on the video type.** It is positive on two-clip videos, near 0 on plain ones, and negative but not significant on stacked ones. P2 (+0.03) averages these.
  - The two-clip gain may come from captioning each window on its own rather than from text as such. With frames, Haiku must keep about 10 small images from two clips apart; a caption per window states what each window shows.
- **Question-blind captions lose comparisons between things shown at the same time.** On stacked videos, no captioner states the difference between the copies, and Sonnet on frames beats every caption arm in both sets.
- **On reversed videos, both modalities fail toward the prior.** A captioner that does report the reversal helps (C2a +0.14).

**Use in the paper:** an exploratory analysis paragraph with a figure (accuracy by video type), and the clustered CIs. C3 Amendment 1 adds predictions by video type before C3 runs. Follow-up tests of these hypotheses are being decided with the user before C3.

**Decision (2026-10-20, after A3).** The user chose all three recommended follow-ups: minimal pairs (C4), describe-then-answer (C5) and the caption audit (C7). The 448 px arm (C6) was decided earlier (G2). Each entry below was written and committed before any of its calls. C3's caption run started first, under its own entry, after its 3-call check passed (Sonnet's caption calls took 2 more input tokens than Haiku's for the same windows: 1,499 vs. 1,497 and 1,365 vs. 1,363).

### C4 — Minimal pairs: does the temporal fact reach each representation? (written 2026-10-20, committed before any C4 call)

**Status.** Exploratory, designed after C1–C3's design and A3. On the confirmatory questions and their sibling videos.

**Why.** A3 read the video-type results as: captions narrate reversed clips forwards, and question-blind captions don't compare the two copies of a stacked clip. Accuracy on one video can't separate "the representation lost the fact" from "the reader guessed". A minimal pair can: the same question, with the same wording and option order, on two videos built from the same filmed clip whose correct options differ. An arm that doesn't receive the fact gives the same answer on both and gets at most one right.

**Pairs** (`tempcompass.py sample --set pairs`, `results/index_qa/pairs_items.json`).
- For each confirmatory question, the first sibling video (sorted by ID) with a TempCompass question of the same aspect, stem and option set whose correct option differs. Siblings: a plain clip and its reversed version; the other stacked versions of the clip (other speed pairs); the other two-clip videos of the same two clips (other order or layout).
- Only siblings already captioned by Gemini (on the video), so no Gemini call is needed. That leaves out 3 reversal pairs.
- **193 pairs:** 107 reversal (66 with the reversed video confirmatory, 41 with the plain one), 41 stacked and 45 two-clip. They are on 174 sibling videos (449 windows). 9 of the 41 stacked pairs, and some others, have siblings in the dev set; that is fine, because the sibling item is a new question–video combination.
- Chance pair accuracy (both right by guessing): 0.093.

**Arms** (each also read on the sibling, with the confirmatory question's wording):

| Arm | New calls |
|---|---|
| Haiku, 5 frames | 193 Haiku |
| Haiku, self-captions | 193 Haiku; self-captions for ≈ 50 sibling windows that lack them |
| Haiku, Sonnet captions | 193 Haiku; Sonnet captions for the ≈ 120 sibling windows outside the confirmatory set |
| Haiku, Gemini video captions | 193 Haiku |
| Sonnet, 5 frames | 193 Sonnet |
| Sonnet, Sonnet captions | 193 Sonnet, and the 297 confirmatory reads (plan §6: Sonnet on its own captions) |

**Outcomes** (`reader.py c4`): per arm, accuracy on each side, **pair accuracy** (right on both) and the **same-answer rate** (one letter for both videos). Differences in pair accuracy with 95% CIs from resampling clip clusters, overall and by group (reversal, stacked, two-clip).

**Predictions** (pair accuracy, overall / reversal / stacked / two-clip):

| Arm | All | Reversal (107) | Stacked (41) | Two-clip (45) |
|---|---|---|---|---|
| Haiku, 5 frames | 0.31 | 0.33 | 0.15 | 0.40 |
| Haiku, self-captions | 0.37 | 0.36 | 0.08 | 0.65 |
| Haiku, Sonnet captions | 0.46 | 0.47 | 0.12 | 0.75 |
| Haiku, Gemini video captions | 0.49 | 0.47 | 0.15 | 0.85 |
| Sonnet, 5 frames | 0.58 | 0.55 | 0.40 | 0.80 |
| Sonnet, Sonnet captions | 0.50 | 0.50 | 0.15 | 0.80 |

| Term (pair accuracy) | Prediction |
|---|---|
| C4a modality: Haiku self-captions − 5 frames | +0.06 |
| C4b captioner: Haiku on Sonnet captions − self-captions | +0.09 |
| C4c index vs. direct: Haiku on Sonnet captions − Sonnet on 5 frames | −0.12 |
| C4d Haiku: Gemini video captions − 5 frames | +0.18 |
| C4e Sonnet: own captions − 5 frames | −0.08 |

**Directional hypotheses** (each read from its 95% clustered CI):
1. **Stacked: captions lose the comparison.** On stacked pairs, every caption arm's same-answer rate is above Sonnet on 5 frames', and C4c < 0.
2. **Two-clip: C4a > 0.** The modality effect of A3 holds on pair accuracy.
3. **Reversal: C4b > 0.** A stronger captioner reports the reversal more often.

**Stop rules.** Isolation, model and format checks as in C1–C3; `--max-5h 0.8`. A pair with a failed call counts as wrong in that arm. The sibling caption runs follow C3's rules (stop at > 2% missing windows).

```bash
.venv/bin/python scripts/index_qa/self_caption.py --set pairs --model haiku
.venv/bin/python scripts/index_qa/self_caption.py --set pairs --model sonnet --max-5h 0.7 --max-7d 0.5
.venv/bin/python scripts/index_qa/reader.py run --set pairs --model haiku --arms frames5 selfcaptions sonnetcaptions captions
.venv/bin/python scripts/index_qa/reader.py run --set pairs --model sonnet --arms frames5 sonnetcaptions
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model sonnet --arms sonnetcaptions
.venv/bin/python scripts/index_qa/reader.py c4
```

**Result (2026-10-20; `reader.py c4`, run once after all reads were complete; output in `results/index_qa/reports/c4.txt`).**
- **Run.** All 6 arms read all 193 pairs. No call failed and every answer parsed.
- **Pair accuracy** (chance 0.093), with the prediction in brackets:

| Arm | All | Reversal (107) | Stacked (41) | Two-clip (45) | Same-answer rate |
|---|---|---|---|---|---|
| Haiku, 5 frames | 0.233 (0.31) | 0.28 (0.33) | 0.17 (0.15) | 0.18 (0.40) | 0.54 |
| Haiku, self-captions | 0.326 (0.37) | 0.33 (0.36) | 0.10 (0.08) | 0.53 (0.65) | 0.42 |
| Haiku, Sonnet captions | 0.430 (0.46) | 0.38 (0.47) | 0.17 (0.12) | 0.78 (0.75) | 0.43 |
| Haiku, Gemini video captions | 0.482 (0.49) | 0.47 (0.47) | 0.20 (0.15) | 0.78 (0.85) | 0.37 |
| Sonnet, 5 frames | 0.606 (0.58) | 0.56 (0.55) | 0.51 (0.40) | 0.80 (0.80) | 0.28 |
| Sonnet, Sonnet captions | 0.430 (0.50) | 0.40 (0.50) | 0.10 (0.15) | 0.80 (0.80) | 0.39 |

| Term (pair accuracy; 95% CI over clip clusters) | Result | Prediction |
|---|---|---|
| C4a modality: Haiku self-captions − 5 frames | +0.093 [+0.028, +0.161] | +0.06 |
| C4b captioner: Haiku on Sonnet − self-captions | **+0.104 [+0.017, +0.190]** | +0.09 |
| C4c index vs. direct: Haiku on Sonnet captions − Sonnet on frames | −0.176 [−0.276, −0.079] | −0.12 |
| C4d Haiku: Gemini video captions − 5 frames | +0.249 [+0.173, +0.319] | +0.18 |
| C4e Sonnet: own captions − 5 frames | −0.176 [−0.260, −0.093] | −0.08 |

- **The captioner effect replicates on pairs.** C4b is +0.104, so Sonnet's captions carry facts that distinguish the two videos which Haiku's own captions do not. This is the C3a result, measured without guessing.
- **Hypotheses:**
  1. **Stacked, captions lose the comparison: holds.**
     - C4c on stacked pairs is −0.34 [−0.60, −0.08]; C4e is −0.41 [−0.62, −0.23].
     - The same-answer rate minus Sonnet on frames' is above 0 for 3 of the 4 caption arms:
       - Haiku self-captions +0.32 [+0.13, +0.49];
       - Haiku on Sonnet captions +0.32 [+0.07, +0.55];
       - Sonnet on Sonnet captions +0.24 [+0.05, +0.43];
       - Haiku on Gemini video captions +0.22 [−0.02, +0.44], the exception.
     - Every caption arm's stacked pair accuracy is 0.10–0.20, about chance level for pairs, against 0.51 for Sonnet on frames.
     - This is also C3 Amendment 1's hypothesis 1 (stacked C3c < 0), which C3 alone could not show at 95%.
  2. **Two-clip, C4a > 0: holds.** +0.36 [+0.16, +0.56].
  3. **Reversal, C4b > 0: not shown.** +0.06 [−0.01, +0.13]. As in C3 Amendment 1, Sonnet's captions do not report reversals much more often than Haiku's.
- **Unpredicted:** Haiku on 5 frames gives the same letter on both videos of a pair 54% of the time, and is below its predicted pair accuracy, mostly on two-clip pairs (0.18 vs. 0.40). On two-clip pairs, with 10 small frames from two clips, Haiku's frame reading is close to a guess about order.

### C5 — Describe, then answer (written 2026-10-20, committed before any C5 call)

**Status.** Exploratory; the 297 confirmatory questions.

**Why.** A3 found a modality effect of +0.22 on two-clip videos (Haiku's self-captions vs. its 5 frames). Two things differ between those arms: the frames are put into words, and each window is captioned in its own call. C5 keeps the input and the single call of the frames arm, and asks Haiku to write one sentence per window before answering (`describe5`, `prompts/index_qa/mcq_describe.txt`; only the answer instruction differs from `mcq.txt`). Its descriptions see the question, unlike the captions.

**Readings.**
- On two-clip videos, **C5a = describe5 − frames5 > 0** means that writing the windows down helps, even in one call.
- **describe5 − self-captions < 0** there means that captioning each window on its own matters.
- On stacked videos, describe5 > self-captions would mean that the stacked loss is an omission that a question-aware description fixes (Prop. 4: the price of being question-blind).

**Predictions** (accuracy; 95% CIs over clip clusters for the differences):
- **describe5:** plain 0.80, two-clip 0.72, reversed 0.40, stacked 0.45, so **0.64** overall.
- **C5a ≈ +0.03** overall; two-clip +0.12.
- **C5b ≈ 0.00** overall: describe5 − self-captions is −0.10 on two-clip and +0.13 on stacked.

```bash
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms describe5
.venv/bin/python scripts/index_qa/reader.py c5
```

**Result (2026-10-20; `reader.py c5`, run once; output in `results/index_qa/reports/c5.txt`).** 297 calls, and every answer parsed.
- **describe5 scores 0.684** (predicted 0.64): plain 0.82, two-clip 0.69, reversed 0.50, stacked 0.54.

| Term (95% CI over clip clusters) | All | Plain | Two-clip | Reversed | Stacked | Prediction (all) |
|---|---|---|---|---|---|---|
| C5a, describe5 − frames5 | **+0.071 [+0.029, +0.114]** | +0.02 | +0.09 [−0.06, +0.24] | +0.12 [+0.01, +0.23] | +0.12 [+0.00, +0.23] | +0.03 |
| C5b, describe5 − self-captions | +0.040 [−0.011, +0.092] | +0.03 | −0.13 [−0.30, +0.02] | +0.07 | **+0.22 [+0.09, +0.37]** | 0.00 |

- **Reading:**
  - **Writing the windows down helps,** even in the same call with the same input: C5a > 0. The gain is on built videos (reversed and stacked), not on two-clip ones as predicted.
  - **Two-clip:** describe5 is below the self-captions (0.69 vs. 0.82), but the CI includes 0. The direction fits "captioning each window in its own call matters", without showing it.
  - **Stacked:** a description that sees the question beats the question-blind self-captions by +0.22. The stacked loss is an omission caused by not seeing the question (Prop. 4), not a limit of the text format.
- **Cost:** describe5 writes 501 output tokens per question (346 of them thinking) against 90 for frames5. At list price that is $0.000415 against $0.000204 per question: 2× the frames arm, and 2.3× Haiku on Sonnet captions, which is more accurate (0.707). describe5 sees the frames at every question, so it is not an index.

### C6 — Haiku on 448 px frames (written 2026-10-20, committed before any C6 call; decided at G2)

**Status.** Exploratory; the 297 confirmatory questions.

**Why.** It answers "you starved the frames": at 224 px a frame is 40 tokens, and Haiku gains little from 1 to 5 of them (0.58 → 0.61). At 448 px (448×252, 144 tokens) a full window is 720 tokens. That puts a frame arm at about 2.3× the input tokens of the caption arms on the accuracy–cost curve.

**Arm.** `frames5_448`: the frames5 prompt and instants, with frames extracted at long edge 448 (`tempcompass.py frames --long-edge 448`, `local/tempcompass/frames_448/`).

**Predictions:**
- **Accuracy 0.66:** plain 0.84, two-clip 0.65, reversed 0.42, stacked 0.48.
- **C6a (448 − 224 px) ≈ +0.04.**
- **C6b** (Haiku on Sonnet captions − 448 px) ≈ +0.07.
- **C6c** (Gemini video captions − 448 px) ≈ +0.08.
- **C6d** (Gemini frame captions − 448 px) ≈ +0.13.
- **Median input tokens ≈ 2,800** per question, against ≈ 1,520 at 224 px.
- **Reading:** if C6c's CI includes 0, the caption advantage for Haiku is a frame-size effect, and the cost comparison (A2) carries the argument. If it stays positive, captions beat a 2× larger visual budget.

```bash
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms frames5_448
.venv/bin/python scripts/index_qa/reader.py c6
```

**Result (2026-10-20; `reader.py c6`, run once; output in `results/index_qa/reports/c6.txt`).** 297 calls, and every answer parsed.
- **Accuracy 0.650** (predicted 0.66): plain 0.82, two-clip 0.62, reversed 0.43, stacked 0.49.
- **Median input: 2,676 tokens** per question (predicted ≈ 2,800), against 1,520 at 224 px and ≈ 1,280 for the caption arms. One extracted frame was portrait (398×448).

| Term (95% CI over questions; clustered) | Result | Prediction |
|---|---|---|
| C6a, 448 − 224 px | +0.037 [−0.010, +0.084]; [−0.007, +0.082] | +0.04 |
| C6b, Sonnet captions − 448 px | +0.057 [+0.003, +0.111]; [+0.004, +0.109] | +0.07 |
| C6c, Gemini video captions − 448 px | **+0.088 [+0.037, +0.138]; [+0.040, +0.132]** | +0.08 |
| C6d, Gemini frame captions − 448 px | +0.141 [+0.088, +0.195]; [+0.085, +0.195] | +0.13 |

- **Reading:** the caption advantage for Haiku is not a frame-size effect. C6c stays positive, so Haiku reading captions (≈ 1,280 input tokens) beats Haiku reading frames at 2.1× those input tokens, and so does C6b, barely.
  - By type, all of C6b–C6d's advantage is on two-clip videos (+0.31 to +0.38) and, for C6d, also on reversed and plain ones.
  - On stacked videos every caption arm is within ±0.07 of 448 px frames.
- Every result is within 0.03 of its prediction.

### C7 — Caption audit: why the caption arms fail (written 2026-10-20, committed before any C7 call)

**Status.** Exploratory; labels only, no new reader calls.

**Why.** It separates a caption arm's errors into three kinds:
- **misstatement:** the caption states a wrong option, e.g. forward narration of a reversed clip;
- **omission:** the caption doesn't decide the question, e.g. no speed comparison on a stacked clip;
- **reader error:** the caption states the answer, but the reader missed it.

This is the paper's qualitative analysis, and it tests A3's readings.

**Design** (`scripts/index_qa/audit.py`, prompt `prompts/index_qa/audit.txt`).
- **Items:** every (caption arm, confirmatory question) that some reader got wrong from those captions. The reader arms are Haiku on self, Sonnet, Gemini frame-clip, Gemini frame and Gemini video captions, and Sonnet on Gemini video and Sonnet captions. That is ≈ 336 items before C3, ≈ 420 after.
- **Judge:** Sonnet, given the captions, the question, the options and the correct answer, labels the captions "correct", "wrong" or "silent", with a quote as evidence. One isolated call per item.
- **Check:** `audit.py sheet` draws 40 labelled items, stratified by the judge's label (seed 0), for the user to label blind. `audit.py agree` reports agreement and Cohen's κ. **If κ < 0.6, the paper reports only the hand-labelled sample,** with the judge's labels as a secondary check.

**Predictions** (share of a reader arm's wrong answers by label: correct / wrong / silent):
- **Haiku, self-captions:** 0.20 / 0.30 / 0.50.
- **Haiku, Gemini video captions:** 0.25 / 0.30 / 0.45.
- **Sonnet, Gemini video captions:** 0.15 / 0.35 / 0.50. A stronger reader makes fewer reader errors.
- **By video type, every captioner:**
  - Stacked: silent ≥ 0.70 (omission).
  - Reversed: wrong ≥ 0.50 (forward narration).

```bash
.venv/bin/python scripts/index_qa/audit.py run --limit 3      # read the three labels, then the rest
.venv/bin/python scripts/index_qa/audit.py run
.venv/bin/python scripts/index_qa/audit.py report
.venv/bin/python scripts/index_qa/audit.py sheet              # for the user's 40 hand labels
```

**Result (2026-10-21; `audit.py report`, run once after all labels; output in `results/index_qa/reports/c7.txt`).**
- **Run.**
  - **3-call check:** the first label was checked by hand and is right. It was a stacked sunrise, where the caption names one sun and no speeds, and the judge said "silent".
  - **Labels:** 442 items, all labelled; 7-day quota 7% → 9%.
  - **Hand-label sheet:** `results/index_qa/audit/hand_labels.csv`, 40 items, blind (prompt only). Every number below is the judge's.
- **Judge check (2026-10-22; `audit.py agree`): agreement 0.88 (35/40), Cohen's κ = 0.81 ≥ 0.6,** so the judge's labels stand (rule above). Bootstrap over the 40 items (10,000 draws, seed 0): 95% CI [0.64, 0.96], so the lower end is also above 0.6.
  - **The judge is never confused between "correct" and "wrong":** it agrees on all 11 of the user's "correct" labels and all 11 of the user's "wrong" ones.
  - **All 5 disagreements are the user's "silent":** the judge said "correct" 3 times (items 3, 18, 19) and "wrong" 2 times (items 1, 20). The judge settles borderline captions where the user leaves them undecided, so its "silent" shares are, if anything, an undercount.
  - **Labelling was blind to the judge, but not to a Claude reading.** The user labelled in a session with Claude (Opus), which never showed the judge's labels. Claude did point out caption details before the user settled six items (1, 3, 11, 18, 20, 25). Three of those labels changed from the user's first instinct (1: correct → silent; 18: wrong → silent; 20: correct or silent → silent). With the first-instinct labels, κ is also 0.81: the changes move which items disagree, not how many.
  - **The user's notes** are in the sheet's `note` column. Item 25 (a reversed wall) is "correct" only given the task's knowledge that videos may be reversed; the judge's prompt doesn't say so, and the judge also said "correct" from "debris tumble upward".
- **Wrong answers by the judge's label of the captions** (share: reader error "correct" / misstatement "wrong" / omission "silent"; n wrong), with the prediction:

| Reader arm | Correct | Wrong | Silent | n | Prediction |
|---|---|---|---|---|---|
| Haiku, self-captions | 0.08 | 0.28 | **0.63** | 106 | 0.20 / 0.30 / 0.50 |
| Haiku, Sonnet captions | 0.13 | 0.26 | 0.61 | 87 | — |
| Haiku, Gemini frame-clip captions | 0.17 | 0.15 | 0.68 | 82 | — |
| Haiku, Gemini frame captions | 0.10 | 0.18 | 0.73 | 62 | — |
| Haiku, Gemini video captions | 0.10 | 0.21 | **0.69** | 78 | 0.25 / 0.30 / 0.45 |
| Sonnet, Gemini video captions | 0.05 | 0.24 | **0.71** | 62 | 0.15 / 0.35 / 0.50 |
| Sonnet, Sonnet captions | 0.08 | 0.26 | 0.66 | 86 | — |

- **Omission is the main failure:** 61–73% of every caption arm's wrong answers come from captions that don't decide the question. Reader errors are rarer than predicted (5–17% against 15–25%). Misstatements are 15–28%.
- **Sonnet makes fewer reader errors than Haiku on the same captions:** 0.05 vs. 0.10 on Gemini video captions, as predicted, but the counts are small (3 vs. 8).
- **By video type:**
  - **Stacked: silent ≥ 0.70 holds.** It is 0.93–1.00 in every arm. Almost no caption says how the two copies differ.
  - **Reversed: wrong ≥ 0.50 fails.** Wrong is 0.28–0.45 and silent 0.45–0.62. About two in five reversed failures are forward narration; the rest are captions that do not commit to a direction. Sonnet's captions misstate as often as Haiku's (0.41 vs. 0.45).
  - Plain: silent 0.60–0.80.
  - Two-clip: few errors (1–8 per arm), mostly reader errors, where the captions state the order but the reader still answered wrongly.
- **Reading:** a question-blind caption fails mainly by leaving out what the question asks, and less often by stating the wrong thing. This fits the cap of the C3 result: once the fact is missing, no reader can recover it. It also fits Prop. 4 and C5b on stacked videos (+0.22 for a description that sees the question).
- **Caveat:** Sonnet is both the judge and one of the captioners. If κ with the hand labels is < 0.6, only the hand-labelled sample is reported (pre-registered above). κ is 0.81 (judge check above), so the judge's labels are reported.

### C8 — Sonnet with the question only (written 2026-10-21, committed before any C8 call)

**Status.** Exploratory; the 297 confirmatory questions. The user asked for it on 2026-10-21 (draft v1, gap list).

**Why.**
- Table 1 has no question-only row for Sonnet.
- Prop. 1's retention for Sonnet needs that row: the share of the frames' gain over the question that captions keep.
- Claim 5 (reversed videos fall back to the forward reading) needs Sonnet's own prior. A3 and the C3 result used Haiku's.

**Arm.** `sonnet:blind`: the question and options only, with the reader prompt of every arm (`reader.py run --set confirm --model sonnet --arms blind`). 297 Sonnet calls.

**Predictions:**
- **Accuracy 0.43:** plain 0.62, two-clip 0.33, reversed 0.15, stacked 0.38. C8a (Sonnet − Haiku, question only) ≈ +0.02, and C8b (Sonnet frames − question only) ≈ +0.37.
- **Retention**, (captions − question only) / (frames − question only):
  - Sonnet on Gemini video captions: 0.98.
  - Sonnet on its own captions: 0.76.
- **Directional hypotheses:**
  1. **Reversed videos: Sonnet's question-only accuracy is below 0.25,** as Haiku's was (0.16). The wording favours the forward event for both readers.
  2. **Among Sonnet's wrong answers on reversed videos, the share equal to its question-only answer is above the uniform rate (≈ 0.43)** for both frames and its own captions (predicted 0.55 and 0.60). Descriptive; there are ≈ 25–40 wrong answers each.

```bash
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model sonnet --arms blind --limit 3
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model sonnet --arms blind --max-5h 0.8
.venv/bin/python scripts/index_qa/reader.py c8
```

**Result (2026-10-21; `reader.py c8`, run once after the reads; output in `results/index_qa/reports/c8.txt`).** 297 calls, and every answer parsed. Run alongside C9; the two together moved the 7-day quota from 12% to 16%.

| | Prediction | Result |
|---|---|---|
| Sonnet, question only | 0.43 (plain 0.62, two-clip 0.33, reversed 0.15, stacked 0.38) | **0.414** (0.61, 0.29, 0.18, 0.32) |
| C8a, Sonnet − Haiku, question only | +0.02 | **+0.003** [−0.061, +0.064]; clustered [−0.054, +0.061] |
| C8b, Sonnet: frames − question only | +0.37 | **+0.384** [+0.320, +0.451]; clustered [+0.319, +0.448] |
| Retention, Sonnet on Gemini video captions | 0.98 | **0.98** [0.87, 1.11] |
| Retention, Sonnet on its own captions | 0.76 | **0.77** [0.66, 0.88] |

- **H1 holds.** Sonnet's question-only accuracy on reversed videos is 0.18, below 0.25. Haiku's was 0.16. The two readers share the question-only prior: C8a is +0.003, and within ±0.02 for every type.
- **H2 holds.** These are the shares of Sonnet's wrong answers on reversed videos that equal its question-only answer, against 0.43 if the wrong answers were uniform:
  - frames: 0.59 (27 wrong; predicted 0.55);
  - its own captions: 0.65 (40; predicted 0.60);
  - Gemini video captions: 0.66 (32);
  - its own 448 px captions: 0.74 (38).

  This is descriptive.
- **Reading:**
  - Claim 5 now holds for both readers. On reversed videos, a reader that errs tends to fall back to the forward reading the question alone suggests, with frames and with captions alike.
  - On stacked videos, Sonnet's frame errors do not revert (0.14 of 14), but its caption errors do (0.48–0.53). This fits C7: a caption that omits the difference leaves the reader with only the prior.
- **Prop. 1 retention for Sonnet:**
  - Gemini video captions keep all of the frames' gain over the question.
  - Sonnet's own captions keep about three-quarters of it.

### C9 — Sonnet captions from 448 px frames (written 2026-10-21, committed before any C9 call)

**Status.** Exploratory; the 297 confirmatory questions. The user asked for it on 2026-10-21 (draft v1, gap list).

**Why.**
- In claim 3, Haiku on Sonnet's captions trails Sonnet on frames by 0.091 (C3c).
- Only Gemini's frame captions match Sonnet on frames, and their captioner used ≈ 5,500 visual tokens per window against Sonnet's 200.
- C2b put Gemini's visual budget effect at +0.067. C9 asks whether a larger captioning budget closes part of the Claude chain's gap.
- C7 points the other way. The failures are mostly omissions, of things a question-blind captioner does not think to write, so more pixels may not help.

**Arm.** `sonnetcaptions448`:
- **Captioning:** Sonnet captions each window from the frames5_448 frames, which C6 already extracted: the same instants as every arm, at 448 px, 144 tokens per landscape frame, so up to 720 per window. The prompt and effort are the same as `sonnetcaptions`. One isolated call per window, question-blind (`self_caption.py --model sonnet --long-edge 448`, folder `claude-sonnet-5-5-frames448`). 761 windows.
- **Reading:** Haiku and Sonnet each read the captions with the reader prompt of every caption arm. That is 297 calls each.
- **Only the captioner's frame size differs** from `sonnetcaptions`. The index cost per video rises with the captioner's input tokens.

**Terms and predictions** (95% CIs over questions and over clip clusters, and by video type):

| Term | Difference | Prediction |
|---|---|---|
| **C9a, captioner visual budget** | Haiku: Sonnet 448 px captions − Sonnet 224 px captions | **+0.03** |
| C9b, index vs. direct | Haiku on Sonnet 448 px captions − Sonnet on 5 frames | −0.06 |
| C9c, Sonnet on its own 448 px captions − its frames | | −0.07 |
| C9d, vs. Gemini frame captions | Haiku: Sonnet 448 px captions − Gemini frame captions | −0.06 |
| C9e, Sonnet's own captions, 448 − 224 px | | +0.02 |

- **Prediction for Haiku on Sonnet 448 px captions: 0.74.** By type: plain 0.88, two-clip 0.93, reversed 0.50, stacked 0.44.
- **Readings:**
  - **C9a's CI excludes 0, positive:** within Claude, the captioner's visual budget matters too. Part of claim 3's gap is budget, and the draft should say how much of it C9a closes.
  - **C9a's CI includes 0 and C9b < 0:** the gap to Sonnet on frames is not the captioner's budget. That supports C7's reading that it is question-blindness.
  - The by-type estimates are descriptive. With power ≈ 0.25 for +0.03 on 297 questions, a null C9a is weak evidence of no effect, and will be reported as such.

**Run and stop rules.**
1. **3-call check** (`--limit 3`): read the three captions. Each call's input tokens should exceed Sonnet's 224 px caption call for the same window by 104 per landscape frame (520 for 5 frames). Stop and investigate if any differs from that by more than 10 tokens beyond portrait frames.
2. **Captions:** the remaining windows, with `--max-5h 0.7 --max-7d 0.5`. Stop and investigate if more than 2% of windows get no caption.
3. **Reads:** Haiku and Sonnet, `--max-5h 0.8`, `--skip-missing`, with the isolation and model checks of K0–C3.
4. **Analysis:** `reader.py c9`, run once after steps 2 and 3. Then `reader.py cost`, for the index cost and Q\*.

```bash
.venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --long-edge 448 --limit 3
.venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --long-edge 448 --max-5h 0.7 --max-7d 0.5
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms sonnetcaptions448 --skip-missing
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model sonnet --arms sonnetcaptions448 --skip-missing
.venv/bin/python scripts/index_qa/reader.py c9
```

**Result (2026-10-21; `reader.py c9`, run once after steps 2–3; output in `results/index_qa/reports/c9.txt`).**
- **Calls:** 761 windows captioned (3 in the check, 758 in the run), with none missing. 297 reads each by Haiku and Sonnet, and every answer parsed.
- **Token check:** 448 − 224 px input tokens per caption call, same window: median 520 (5 landscape frames × 104), range 104–960. The 960 cases are windows with portrait frames.

| Term | Prediction | Result (95% CI over questions; clustered) |
|---|---|---|
| Haiku on Sonnet 448 px captions | 0.74 (0.88, 0.93, 0.50, 0.44) | **0.710** (plain 0.88, two-clip 0.87, reversed 0.46, stacked 0.41) |
| **C9a, captioner visual budget** | **+0.03** | **+0.003** [−0.037, +0.047]; [−0.034, +0.041] |
| C9b, index vs. direct | −0.06 | **−0.088** [−0.138, −0.037]; [−0.141, −0.035] |
| C9c, Sonnet: own 448 px captions − frames | −0.07 | **−0.067** [−0.114, −0.024]; [−0.114, −0.024] |
| C9d, vs. Gemini frame captions | −0.06 | **−0.081** [−0.131, −0.030]; [−0.126, −0.037] |
| C9e, Sonnet's own captions, 448 − 224 px | +0.02 | **+0.020** [−0.024, +0.064]; [−0.026, +0.064] |

- **The second reading holds: C9a's CI includes 0, and C9b < 0.**
  - Four times the captioner's visual tokens did not move Haiku's accuracy on Sonnet's captions: +0.003, with no type above ±0.07.
  - The index still trails Sonnet on frames by 0.088. The loss is on reversed (−0.18) and stacked (−0.24) videos; plain and two-clip are within 0.04.
  - With power ≈ 0.25 for +0.03, this is weak evidence against a small budget effect. The CI rules out more than about +0.045.
- **The captioner effect survives a larger Claude visual budget.** Sonnet's 448 px captions (720 visual tokens per window) still trail Gemini's frame captions (≈ 5,500) by 0.081 when Haiku reads them. Budget is not fully matched: C2b's +0.067 for Gemini came from a much larger change. But within Claude, quadrupling the budget moved nothing, so the budget is an unlikely account of the 0.08.
- **Reading for claim 3:** the Claude-only chain's gap is not the 224 px frames. Together with C7 (omissions dominate), the gap is what a question-blind captioner chooses to write.
- **Index cost:** the 448 px captions cost $0.0117 per video against $0.0091. Q\* for Haiku on them vs. Sonnet on frames is 3.0 [2.8, 3.1] questions per video.

**Quota estimate for C8 and C9:** ≈ 1,350 Sonnet calls and 297 Haiku calls. C3–C6 used 4 points of the 7-day figure for ≈ 1,700 Sonnet calls, so expect ≈ 4 points. The 5-hour window will need two or three windows; done items are skipped on resume.

**Order of the runs (C3–C7).** C3's Sonnet captions run first (started 2026-10-20 17:53 UTC). The Haiku-only arms run alongside it: C5, C6, and C4's Haiku reads that don't need Sonnet captions. The Sonnet reads and the audit follow once C3's captions are done. The 7-day quota guard for every Sonnet run is 0.5–0.6.

### C10 — One caption per video instead of one per 5 s window (written 2026-10-21, committed before any C10 call)

**Status.** Exploratory; the 297 confirmatory questions. The user asked for it on 2026-10-21. The question was whether 5 s windows still make sense when the videos are short.

**Why.**
- The 5 s windows come from the 60 s WildQA design (12 windows). TempCompass videos are short:
  - The median is 9.7 s, and the range is 1.5–35 s.
  - Windows per video: 48 videos have 1 (16%), 126 have 2 (42%), 64 have 3, and 59 have 4 or more.
  - 39 videos end in a window with a single frame.
- Per-window captions can't state facts that span windows, such as order, change, or a comparison across the cut. The reader has to piece those together.
- A reviewer will ask whether captioning the whole video in one call does better.
- **What exists so far is descriptive.** Sonnet on its own captions loses most on one-window videos: 0.54 vs. 0.75 on frames. At ≥ 3 windows it matches the frames. But window count is confounded with video type: 35 of the 41 stacked videos have 1–2 windows. And one-window videos are not split at all, so the split can't be what costs them.

**Arm.** `sonnetwhole`:
- **Captioning:** Sonnet captions each video once, from all its frames5 frames in time order. These are the same 224 px frames as every arm, 11.3 per video on average, at most 35. It is one isolated call, question-blind (`self_caption.py --model sonnet --whole`, folder `claude-sonnet-5-5-frames-whole`).
  - **The prompt** (`caption.whole_prompt`) is the frames prompt of `sonnetcaptions`, with "N frames from a D-second video clip". Its length clause, "in one or two sentences of at most 45 words in total", becomes "in at most 2W sentences of at most 45W words in total" for W windows.
  - **For one-window videos (W = 1) the call is identical to `sonnetcaptions`':** the same frames and the same prompt. Only sampling differs.
- **Reading:** Haiku and Sonnet each read the caption, with the reader prompt of every caption arm. Two things change:
  - **The label:** the caption is labelled with the whole span ("0–9.7 s:") instead of one label per window.
  - **The modality sentence:** it says one description was written by a model that watched the whole video (`parts.json`, `modality_wholecaption`). The window arms' sentence says each description came from a model that saw only its window. Telling the reader the truth requires this change.

  That is 297 calls per reader.
- **The only change from `sonnetcaptions` is the split:** one call over all frames vs. one call per window, with the length limit summed over the windows.

**Terms and predictions** (95% CIs over questions and over clip clusters; by video type, and by windows per video: 1, 2, 3+):

| Term | Difference | Prediction |
|---|---|---|
| **C10a, splitting, Haiku reads** | Haiku: whole-video − window captions (Sonnet) | **+0.00** (plain 0.00, two-clip −0.05, reversed +0.03, stacked +0.02) |
| C10b, splitting, Sonnet reads | Sonnet: whole-video − its window captions | +0.01 |
| C10c, index vs. direct | Haiku on whole-video captions − Sonnet on 5 frames | −0.09 |
| C10d, Sonnet: own whole-video captions − frames | | −0.08 |

- **Prediction for Haiku on whole-video captions: 0.71.**
- **Why near 0:**
  - The losses so far are within a window: the comparison between stacked copies, and the direction of a reversed clip (C4, C7). Seeing more frames at once may help reversed and stacked a little.
  - The two-clip gain came from captions that keep the clips apart, and the window labels give the order for free. A single caption has to narrate the order itself, so two-clip may lose a little.
- **By windows:**
  - **One window is a control.** C10a and C10b are predicted 0.00 there, with CIs that include 0: the captioner input is identical, and only sampling and the reader's modality sentence differ. If the one-window difference is not ≈ 0, the ≥ 2-window differences are read against it (descriptive).
  - **Two windows and 3+:** C10a +0.01 and −0.01.
- **Words**, as the median ratio of the whole-video caption to the window captions summed, per video: 1.0 at W = 1, 0.8 at W = 2, 0.7 at W ≥ 3.
- **Index cost:** one call per video instead of W pays the fixed prompt (≈ 1,180 tokens) once.
  - Captioning cost per video: about $0.005 against $0.0091 for the window captions (ratio ≈ 0.55).
  - Q\* for Haiku on them vs. Sonnet on frames: about 1.3, against 2.3.
- **Readings:**
  - **C10a and C10b CIs include 0, and the 2 and 3+ groups are near 0:** splitting into 5 s windows costs nothing measurable on these videos. The window design stands for the indexing reasons (§1), and the paper states the check.
  - **C10a > 0 (CI excludes 0), from the ≥ 2-window videos:** splitting costs accuracy. The paper reports the per-window index as a lower bound for short videos and says where the loss is, by type.
  - **C10a < 0:** per-window isolation helps. That supports claim 2b's "a separate call per window" mechanism.
  - With 297 questions, CIs are about ±0.045 (as C9a). A null is weak evidence against an effect of a few points.

**Not run:** minimal pairs (C4) for this arm. They follow only if C10a ≠ 0.

**Run and stop rules.**
1. **3-call check** (`--limit 3`). Read the three captions. Each call's input tokens should exceed the `sonnetcaptions` call for window 0 of the same video by 40 per extra landscape frame, or 64 per portrait frame, within ±15 for the prompt change. A one-window video should match it within ±2. Stop and investigate otherwise.
2. **Captions:** the remaining videos, with `--max-5h 0.7 --max-7d 0.5`. Stop and investigate if more than 2% of videos get no caption.
3. **Reads:** first a dry run of one item, to read the reader prompt. Then Haiku and Sonnet, `--max-5h 0.8 --skip-missing`, with the isolation and model checks of K0–C9.
4. **Analysis:** `reader.py c10`, run once after steps 2 and 3. Then `reader.py cost`, and `plots.py cost`.

```bash
.venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --whole --limit 3
.venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --whole --max-5h 0.7 --max-7d 0.5
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms sonnetwhole --dry-run --limit 1
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms sonnetwhole --skip-missing
.venv/bin/python scripts/index_qa/reader.py run --set confirm --model sonnet --arms sonnetwhole --skip-missing
.venv/bin/python scripts/index_qa/reader.py c10
```

**Quota estimate:** 297 Sonnet caption calls (larger inputs), 297 Sonnet reads and 297 Haiku reads. That is about 2 points of the 7-day figure; C8 + C9 used 4 for ≈ 1,350 Sonnet calls.

**3-call check (2026-10-21; the first call at 08:56 UTC, the other two after the 5-hour reset at 12:21; the quota rule had stopped the run at 72%).** All three videos have 198×224 frames, 64 tokens each.

| Video | Windows | Extra frames | Input-token difference vs. window 0 | Rule's expectation (64 × extra) |
|---|---|---|---|---|
| 1098153561_concat_0 | 2 | 3 | 203 | 192 |
| 1100319395_concat_0 | 3 | 10 | 672 | 640 |
| 15695290_concat_1 | 3 | 6 | 404 | 384 |

- **Two of three were outside ±15, so per the rule I investigated.** All three fit (64 + 3) × extra + 2 exactly.
  - Each image block costs 3 tokens beyond its pixels, which the rule left out.
  - The 2 tokens are the prompt change.
  - C9's check could not show the per-image cost, because it changed the frame size, not the number of frames.
- **Verdict:** the inputs are as intended, and the run proceeds. The rule should have been (tokens per frame + 3) × extra frames ± 15.
- **The one-window check (±2)** had no one-window video among the three. It is done after the run over all 48 one-window videos.
- **Captions:** 72, 88 and 99 words, against limits of 90, 135 and 135. Each tells events in order ("begins … then … next").
  - The stacked video's caption says nothing moves in any of the 15 frames, which is the omission pattern of C7.
  - One caption remarks that it leaves out a watermark.

**Result (2026-10-21; `reader.py c10`, run once after steps 2–3; output in `results/index_qa/reports/c10.txt`).**
- **Calls:** 297 videos captioned, none missing. 297 reads each by Haiku and Sonnet, and every answer parsed. The 7-day quota went from 17% to 19%.
- **One-window check:** the input tokens equal `sonnetcaptions`' exactly for all 48 one-window videos (difference 0).

| | Prediction | Result (95% CI over questions; clustered) |
|---|---|---|
| Haiku on whole-video captions | 0.71 | **0.721** (plain 0.85, two-clip 0.89, reversed 0.57, stacked 0.37) |
| **C10a, splitting, Haiku reads** | **+0.00** (0.00, −0.05, +0.03, +0.02) | **+0.013** [−0.027, +0.057]; [−0.031, +0.054]. By type: plain +0.00, two-clip −0.04, **reversed +0.12 [+0.03, +0.23]**, stacked −0.07 [−0.21, +0.06] |
| C10b, splitting, Sonnet reads | +0.01 | **+0.010** [−0.030, +0.051]; [−0.031, +0.052]. By type: +0.02, +0.02, +0.05, −0.12 [−0.29, +0.05] |
| C10c, index vs. direct | −0.09 | **−0.077** [−0.128, −0.027]; [−0.132, −0.024] |
| C10d, Sonnet: own whole-video captions − frames | −0.08 | **−0.077** [−0.125, −0.034]; [−0.128, −0.027]. Stacked −0.39 |
| C10a by windows: 1 / 2 / 3+ | 0.00 / +0.01 / −0.01 | +0.02 [−0.07, +0.11] / −0.01 / +0.03 |
| C10b by windows: 1 / 2 / 3+ | 0.00 / — / — | −0.02 [−0.09, +0.04] / +0.06 [−0.01, +0.12] / −0.02 |
| Words, whole ÷ windows summed (median) | 1.0 / 0.8 / 0.7 | 1.00 / 0.84 / 0.74 |
| Captioning $ per video | ≈ 0.005 (≈ 0.55 of windows) | **0.00492** vs. 0.00910 (mean; median ratio 0.61) |
| Q\*, Haiku on whole-video captions vs. Sonnet on frames | ≈ 1.3 | **1.2** [1.2, 1.3] |

- **The first reading holds: splitting into 5 s windows costs nothing measurable on these videos.**
  - Both readers' CIs include 0. They rule out a cost of splitting larger than ≈ 0.05–0.06 (the upper CI bounds), and a benefit from splitting larger than ≈ 0.03.
  - The one-window control is ≈ 0 for both readers, as predicted.
  - **One unpredicted pattern (descriptive; one of 8 by-type terms, uncorrected):**
    - Haiku gains +0.12 on reversed videos from whole-video captions.
    - Stacked videos lean the other way for both readers (−0.07, −0.12; CIs include 0).
    - A plausible account: seeing all frames at once helps notice that motion runs backwards, but a single caption compresses two stacked copies even more than window captions do.
- **Resampling noise (from the one-window control):**
  - None of the 48 one-window captions is identical to its `sonnetcaptions` twin, though the input is identical.
  - A reader's answer from the two captions agrees on 0.90 (Haiku) and 0.83 (Sonnet) of the 48 questions.
  - On ≥ 2-window videos, where the split also differs, agreement is 0.83 for both.
  - So drawing a new caption changes the answer about as often as removing the split does. Per question, the caption is a noisy measurement, and only the averages are stable.
- **Cost:** one call per video pays the fixed prompt once. The index costs 0.54× as much per video at the same accuracy, and Q\* against Sonnet on frames falls from 2.3 to 1.2.
- **For the paper:**
  - §2.1 can say that the window length was checked: whole-video captions are no better.
  - §5 can offer the whole-video index as the cheaper variant for short videos.
  - The per-window design remains the one that scales to long videos.

### D3 — Headroom split by video type, retention CIs, reader-strength figure (written 2026-10-22, before running)

No new calls. These analyses are descriptive: the point estimates of Table 5 (draft §8.1) and of Haiku's retention were already seen, so there are no predictions, only CIs for numbers in the draft.
- **Table 5 with CIs** (`reader.py headroom`). For each reader, by video type and overall, with 95% CIs over clip clusters:
  - question-only, frames and captions accuracy;
  - perception shortfall = 1 − frames;
  - index loss = frames − captions;
  - retention = (captions − question only) / (frames − question only).

  Two versions:
  - Sonnet reading its own captions, as in §8.1;
  - Haiku reading Sonnet's captions. Its frames are below its captions (claim 2a), so its index loss is expected to be negative.
- **Retention per index, overall**, with CIs over questions and over clip clusters:
  - every index Haiku reads, which fills the gap of the Haiku retention CIs in draft §3;
  - every index Sonnet reads.
- **Figure 3** (`plots.py readers`): a slope chart of accuracy by reader (Haiku, Sonnet) for question only, frames, and the two indexes both readers read (Sonnet captions, Gemini video captions). Each point has a 95% CI over clip clusters. It shows the tension of claim 2a: frames gain with the reader, and the indexes barely do.
- **Figure 2:** new colours, so that no hue means a reader in one figure and an arm in another.

**Result (2026-10-22; `reader.py headroom` and `plots.py readers`, run once; output in `results/index_qa/reports/headroom.txt` and `fig3.txt`).**
- **Table 5, Sonnet on its own captions** (95% CIs over clip clusters):
  - The index loss is ≈ 0 on plain (+0.01 [−0.03, +0.06]) and two-clip (+0.00) videos.
  - It is clearly positive on reversed (+0.18 [+0.05, +0.30]) and stacked (+0.27 [+0.09, +0.47]) videos.
  - The perception shortfall is 0.36 [0.25, 0.48] on reversed videos and 0.34 [0.17, 0.51] on stacked.
  - Retention: plain 0.95 [0.79, 1.12], reversed 0.62 [0.35, 0.88], stacked 0.21 [−0.27, 0.67], all 0.77 [0.64, 0.90].
- **Haiku on Sonnet captions:**
  - The index loss is −0.09 [−0.15, −0.04] overall, mostly two-clip (−0.33 [−0.48, −0.20]); plain −0.05 [−0.11, +0.00].
  - The perception shortfall is 0.39 overall, and 0.62 and 0.59 on reversed and stacked.
  - Retention by type is unstable where frames barely beat the question alone (stacked: CI [−4.0, undefined]), so it is not reported by type.
- **Retention per index, overall (clustered CIs):**
  - Haiku:
    - self-captions 1.15 [0.91, 1.46];
    - Sonnet captions 1.47 [1.18, 1.95];
    - Sonnet 448 px 1.48;
    - Sonnet whole-video 1.53;
    - Gemini frame-clip 1.55;
    - Gemini video 1.62 [1.33, 2.10];
    - Gemini frame 1.88 [1.53, 2.51].

    Every CI but the self-captions' excludes 1.
  - Sonnet: own captions 0.77 [0.64, 0.90], 448 px 0.82, whole-video 0.80, Gemini video 0.98 [0.85, 1.13].
- **Figure 3:** accuracy by reader, with 95% CIs over clip clusters:
  - frames 0.613 → 0.798;
  - Sonnet captions 0.707 → 0.710;
  - Gemini video captions 0.737 → 0.791;
  - question only 0.411 → 0.414.

  The frames line crosses both index lines.
- **Reading:** the index loss is concentrated where the video departs from the prior (reversed) or where a comparison must be chosen in advance (stacked). On natural video, a question-blind index loses nothing measurable for the strong reader.

### C11 — A captioner that knows the kinds of question (written 2026-10-22, committed before any C11 call)

**Why.** Draft §8.1 splits Sonnet's index loss (+0.09 overall; +0.18 on reversed and +0.27 on stacked videos, D3) from its perception shortfall. The captioner sees the frames but not the question. C11 asks how much of the index loss closes when the captioner knows the *kinds* of question that will come, but not the question itself. That places the index between question-blind (now) and question-aware (C5, describe then answer). The user asked whether the captioner prompt should change, because the captioner is also misled (2026-10-22).

**Arm `sonnetaspects`.**
- Sonnet captions each confirmatory video once, from all of its frames5 frames. This is exactly C10's `sonnetwhole` call: the same frames, the same length limit, effort medium.
- One bullet is added before the prompt's last bullet (`caption.ASPECT_BULLET`): "The description will be used later to answer questions about the actions, the speed of the movements, the direction of the movements, the order of the events, and changes in the attributes of things (such as size, colour, shape or state). State these whenever the frames show them."
  - It names TempCompass's five aspects in general terms.
  - It does not name how the benchmark's videos are made: no reversal, stacking or concatenation (test `test_with_aspects_adds_only_the_bullet`).
  - It does not ask for comparisons.
- Haiku and Sonnet read the captions with `sonnetwhole`'s reader prompt, so the reader does not know the captioner's prompt changed.
- The confirmatory captioner prompt is unchanged. C11 is exploratory.

**Predictions** (95% CIs over questions and over clip clusters; by type over clip clusters):

| Term | Prediction |
|---|---|
| C11a: Haiku, aspects − whole | **+0.03** (plain +0.01, two-clip 0.00, reversed +0.04, stacked +0.08) |
| C11b: Sonnet, aspects − whole | **+0.03** (same by type) |
| C11c: Haiku on aspects − Sonnet on frames | −0.05 |
| C11d: Sonnet on own aspects − its frames | −0.05 (C10d was −0.077: ≈ 40% of the index loss closes) |
| Captioner input tokens, aspects − whole, same video | a constant: the bullet alone (≈ 50–70) |
| Words per caption, median ratio aspects / whole | 1.05 (same limit) |
| Says the clip runs backwards ("revers", "backward", "rewind") | reversed videos +0.05 over whole; other videos +0.00 |
| Index $ per video; Q\* vs. Sonnet on frames | ≈ C10 (0.0049; 1.2–1.3) |

**Readings, set now.**
- **C11b ≥ +0.04 with a CI above 0:** knowing the kinds of question buys a measurable part of the index loss. If the gain sits on stacked videos, the loss there is about choosing what to write. If it sits on reversed videos, the captioner could see the direction when told to state it, so part of the "perception shortfall" was attention.
- **C11a and C11b ≈ 0:** naming the aspects does not help. Only the question itself (C5) recovers the loss, which strengthens Prop. 4: the price is of not knowing the question, not of not knowing the task.
- **Plain videos drop (≤ −0.03):** the bullet invites statements the frames do not support. Check by reading the captions of the lost questions.
- **Reversal mentions rise without accuracy:** the captioner states direction more, but the readers don't use it.

**Run (resumable; `scripts/index_qa/run_c11.sh`, committed with this entry).**
1. **3-call check** (`run_c11.sh check`): 3 captions. Inspect them, and check that the input tokens are `sonnetwhole`'s plus a constant. If the check fails, stop and investigate.
2. **Run** (`run_c11.sh run`): the captions, with Haiku and Sonnet reading each video as soon as its caption exists (to save wall-clock time), then `reader.py c11` and `cost`, run once.
   - Stop rules: captions `--max-5h 0.7 --max-7d 0.5`, reads `--max-5h 0.8`; after a stop, the script waits for the 5-hour reset.
   - `run_c11.sh stop` stops everything, and `run` again resumes: caption files are written atomically, and a partial reader row is skipped and its item rerun.
   - Expected: about 50 min of compute and a 7-day quota rise of about +2%.

**3-call check (2026-10-22, 07:55 UTC): passed.**
- All 3 calls are ok. Quota: 5h 0.20, 7d 0.22.
- Input tokens are `sonnetwhole`'s + 74 for all 3 videos: a constant, as predicted (≈ 50–70; the bullet is a little longer).
- Words: 80, 97 and 112, within the limits (90, 135, 135).
- The captions state direction and order more explicitly than C10's ("draws a wavy, dotted teal line that stretches rightward", "straight vertical lines downward"; the forest's colour sequence step by step).
- The stacked two-circle video is "nothing moves" under both prompts (1 fps frames). That is perception, not the prompt.

**Result (2026-10-22; `run_c11.sh run`, 07:55–08:26 UTC; `reader.py c11` and `cost` run once; output in `results/index_qa/reports/c11.txt` and `cost.txt`).**
- **Calls:** 297 captions and 297 reads each by Haiku and Sonnet. Every call was ok and every answer parsed.
- **Timing:** captions took 21 min. Reads followed the captions and ended 7 and 9 min after them, so the whole run took 31 min.
- **Quota:** 7-day 22% → 25%; 5-hour ≈ 48% at the end.

| | Prediction | Result (95% CI over questions; clustered) |
|---|---|---|
| Input tokens, aspects − whole | constant ≈ 50–70 | **74 for all 297 videos** |
| Words, median ratio | 1.05 | **1.15** (median 86 vs. 76) |
| Says the clip runs backwards: reversed / other videos | +0.05 / +0.00 | **0.05 vs. 0.08** (4 vs. 6 of 74) / 0.00 vs. 0.00 |
| **C11a, Haiku, aspects − whole** | **+0.03** | **+0.027** [−0.013, +0.067]; [−0.011, +0.066]. By type: plain +0.02, two-clip +0.07, reversed +0.01, stacked +0.02 |
| **C11b, Sonnet, aspects − whole** | **+0.03** | **+0.040** [+0.003, +0.081]; [+0.000, +0.081]. By type: plain +0.02, two-clip +0.02, reversed +0.04 [−0.05, +0.14], **stacked +0.12 [+0.00, +0.26]** |
| C11c, Haiku on aspects − Sonnet on frames | −0.05 | **−0.051** [−0.104, +0.000]; stacked −0.27 |
| C11d, Sonnet on own aspects − its frames | −0.05 | **−0.037** [−0.081, +0.007]; [−0.089, +0.014]. Plain +0.03, two-clip +0.04, reversed −0.08, stacked −0.27 [−0.49, −0.05] |
| Index $ per video; Q\* vs. Sonnet on frames | ≈ 0.0049; 1.2–1.3 | 0.00527; **1.3** [1.3, 1.4] |

- **Accuracy:**

  | Reader | Aspect captions | Whole-video captions (C10) | Per-window captions | Frames |
  |---|---|---|---|---|
  | Haiku | 0.747 | 0.721 | 0.707 | — |
  | Sonnet | 0.761 | 0.721 | 0.710 | 0.798 |
- **Reading (rules set in advance):**
  - C11b sits exactly at the threshold of the first reading (+0.040). Its question-level CI is above 0, and its clustered CI touches 0.
  - C11a matches its prediction, and its CI includes 0.
  - **Say:** knowing the kinds of question buys a small gain (+0.03 to +0.04), consistent in sign across readers and types. It closes about half of Sonnet's index loss against its own frames (C10d −0.077 → C11d −0.037).
- **Where the gain is, and two caveats (descriptive, not planned):**
  1. **Stacked:**
     - Sonnet's +0.12 on stacked videos recovers C10's whole-video loss there (C10b stacked −0.12): `sonnetwhole` 0.27 → `sonnetaspects` 0.39, which equals the per-window captions (0.39).
     - Against the per-window captions, the aspect captions gain on plain (+0.04), two-clip (+0.05) and reversed (+0.09) videos, and not on stacked (0.00 for Sonnet, −0.05 for Haiku).
     - So the bullet does not fix the stacked comparison: Sonnet on frames stays +0.27 ahead. That loss needs the question (C5b), as Prop. 4 says.
  2. **Length:**
     - The aspect captions are 15% longer at the same limit, so part of the gain may be more words, not the right words. C11 does not separate the two.
     - Reversal mentions did not rise (0.05 vs. 0.08). On reversed videos, the gain (+0.04 Sonnet, +0.01 Haiku; CIs include 0) comes from stating direction more explicitly, not from noticing the reversal.
- **Resampling noise (C10):** a fresh caption alone changes 10–17% of answers. So a gain of +0.03–0.04 averaged over 297 questions is the scale this design can detect, and the by-type terms (n = 41–137) cannot be told apart from noise except stacked for Sonnet, whose CI touches 0.
- **For the paper:**
  - §8.1 can report the middle point between question-blind and question-aware: knowing the kinds of question closes about half of the index loss for Sonnet, and none of the stacked loss.
  - The cheapest index that comes closest to Sonnet on frames is now Haiku on aspect captions: −0.051, Q\* 1.3.

### C12 — An open-weights reader: Qwen2.5-VL-7B on the main decomposition (written 2026-10-26, committed before any C12 call)

**Why.** Every reader so far is a closed model run through a CLI, with its system prompt and no temperature control (draft §7). That is the most likely reviewer objection. The plan kept an open-weights reader as optional (plan §6, G7). With G5 settled as case 1 it is not needed for the claim, but the user chose to run it (2026-10-26) because Kaggle's GPUs are free. C12 asks whether claims 1, 2 and 2a hold for an open reader that anyone can rerun.

**Model.** Qwen2.5-VL-7B-Instruct (`Qwen/Qwen2.5-VL-7B-Instruct`), float16, split over Kaggle's two T4s. It was chosen over Llama 3.2 11B Vision for three reasons:
- Qwen2.5-VL takes many images in one prompt. Llama 3.2 Vision was trained on one image per prompt, and our frame arms show up to 30.
- Its 7B weights fit two T4s in float16 without quantization. Llama's 11B would need 8-bit.
- It counts visual tokens almost as Claude does (28 px per token): over the C12 frames, Qwen gets 0.82 of Claude's visual tokens (224×126 → 32 vs. 40, because Qwen rounds the height and Claude rounds it up).

**Arms (all on the 297 confirmatory questions; 2,376 reads).**
- `blind`, `frames5`, and six caption arms: `qwencaptions`, `selfcaptions` (Haiku's), `sonnetcaptions`, `frameclipcaptions`, `framecaptions` and `captions` (Gemini's).
- **`qwencaptions` is new.** Qwen captions each of the 761 confirmatory windows from exactly the frames5 frames, with the self-captions' prompt (`caption.frames_prompt`), images before the prompt. These are the open reader's own captions, so `qwencaptions − frames5` is the modality effect for Qwen.
- Every reader prompt is `reader.build`'s, exported by `open_reader.py bundle`. `open_reader.py import` refuses any read whose logged prompt differs from `reader.build`'s.

**Deviations from the Claude arms (fixed now):**
- Decoding is greedy (with the shipped repetition penalty), so there is no retry: a greedy retry repeats itself. An unparsed answer scores as wrong.
- Qwen's default system prompt replaces the CLI's.
- Content parts are joined into one chat message, with a newline after each text block. Claude received separate content blocks.
- No thinking. Qwen2.5-VL-Instruct does not think.
- No cost account: the GPU is free, so Q\* is not defined for Qwen. Tokens are logged.

**Predictions** (95% CIs over questions and over clip clusters; by type over clip clusters). Expected accuracies: question only 0.40, frames 0.58, own captions 0.57, Sonnet captions 0.66, Gemini video captions 0.70.

| Term | Prediction |
|---|---|
| C12a modality: Qwen captions − frames (P2 for Qwen) | **−0.01** |
| C12b captioner at identical input: Sonnet captions − Qwen captions (C3a for Qwen) | **+0.09** |
| C12c captioner at identical input: Haiku captions − Qwen captions | +0.05 |
| C12d captioner at about matched tokens: Gemini frame-clip captions − Qwen captions (C2a) | +0.08 |
| C12e Gemini frame captions − Qwen captions (P3 for Qwen) | +0.14 |
| C12f index vs. frames: Gemini video captions − frames (S3 for Qwen) | +0.12 |
| C12g the cap: Qwen − Haiku, both reading Sonnet captions | −0.05 |
| C12h question-only floor: Qwen − Haiku | 0.00 |
| Retention (captions − question only) / (frames − question only): own, Sonnet, Gemini video captions | 0.95, 1.40, 1.60 |

**Readings, set now.**
- **Claim 1 holds for an open reader** if C12b's CI excludes 0 on the positive side. If C12b ≈ 0 but C12e is positive, only Gemini's larger visual budget helps Qwen, as in C2b.
- **Claim 2 holds** if C12a's upper bound is below +0.09. A modality effect of IsoBench's size (≥ +0.15) would mean that text is easier for a weak open reader. That would narrow claim 2 to the Claude readers.
- **Claim 2a (the cap)** holds if C12g's CI includes 0 and |C12g| ≤ 0.05. If Qwen is clearly below Haiku on the same captions (≤ −0.08 with a CI below 0), the cap is reached only by readers with at least Haiku's text skill. Then the paper says so.
- **If frames help Qwen little** (frames − question only < 0.08), Qwen cannot use up to 30 small images in one prompt. The terms against `frames5` (C12a, C12f) then measure that, not the modality, and the paper reports them as a limit of the reader.
- By-type terms are descriptive. Resampling noise (C10) does not apply to greedy decoding, but caption choice still does.

**Run (on Kaggle; the user runs it).**
1. `open_reader.py bundle` writes `local/kaggle_c12.zip` (29.5 MB: 761 caption jobs, 2,376 reader prompts, 3,345 frames, `qwen_kaggle.py`, and the exporting commit in `manifest.json`). It is uploaded as a private Kaggle dataset. Kaggle unzips an upload and allows at most 1,000 files, so the upload is `local/kaggle_c12_upload.zip`, which holds only `kaggle_c12.zip`. Kaggle then unzipped both levels, so the bundle's files sit directly in the dataset folder and `qwen_kaggle.py` runs from there with its default `--input /kaggle/input` (2026-10-26, before any call; packaging only, the bundle is unchanged).
2. **Smoke check** (`qwen_kaggle.py --stage smoke`): 3 captions and 12 reads (6 questions × question only and frames). Passes if the model loads on 2×T4, the captions are English sentences under the length limit, and at least 10 of the 12 answers parse. If float16 gives garbage, retry with `--fp32-vision`, then stop and investigate.
3. **Run** (`--stage all`, as a committed "Save & Run All" version): captions, then reads. It resumes if rerun. Expected: 2–5 h of GPU time, within one 12 h session.
4. Locally: `open_reader.py import <outputs>`, then `reader.py c12`, run once. The output goes to `results/index_qa/reports/c12.txt`.
- Timing (2026-10-26): this entry and its predictions were committed (7a65fb1) before any Qwen call. Their first public push (about 15:15 +0200) came after the smoke check (3 captions and 12 reads, which passed and were seen) and after the full run started, but before any output of the full run was seen.
- Added after this entry (2026-10-26, descriptive, while the run was going): the run's log showed Qwen captions over the prompt's 45-word limit (47 and 57 words), so `reader.py c12` also prints words per caption and the share over the limit for every captioner (`caption_lengths`). Before Qwen, on the 761 confirmatory windows: Haiku self-captions median 45 words, 0.47 over the limit; Sonnet 42, 0.16; Gemini video, frame-clip and frame captions 29–31, none over. The most useful captions are the shortest, so caption length does not explain the captioner effect.
- Already checked without a GPU (2026-10-26): the full round trip with fake outputs (every prompt equal to `reader.build`'s, then `c12` runs), and the runner's code path on CPU with a tiny random Qwen2.5-VL test model.

**Result (2026-10-27; `open_reader.py import`, then `reader.py c12` run once; output in `results/index_qa/reports/c12.txt`).**
- **Run:** Kaggle, 2×T4, float16 (`fp32_vision` not needed), transformers 5.14.1, model commit `cc59489`. Captions 14:19–15:01 UTC, reads 15:01–15:33 UTC (Kaggle's clock): 74 min in all, well under the 2–5 h expected. The GitHub activity API shows the first push of this entry at 13:11 UTC, before the run whose outputs were imported.
- **Import:** 761 captions, 2,376 reads, and every logged prompt equal to `reader.build`'s.
- **Unparsed: 49 of 2,376 (2%), scored as wrong.** None hit the token limit; they are refusals ("not enough information", `{"answer": "None"}`). By arm: own captions 15, Haiku captions 11, question only 7, Gemini video 5, Sonnet 4, Gemini frame 4, frame-clip 3, frames 0. So the rule slightly penalizes the caption arms.
- **Qwen's captions:** median 30 words, 0.10 over the 45-word limit (max 154), 2 at the 200-token limit, none empty or ERROR. Median 160 visual tokens per window.

| Term | Prediction | Result (95% CI over questions; clustered) |
|---|---|---|
| Accuracy: question only / frames / own captions / Sonnet captions / Gemini video captions | 0.40 / 0.58 / 0.57 / 0.66 / 0.70 | 0.397 / **0.657** / **0.535** / 0.660 / 0.697 |
| **C12a modality: Qwen captions − frames** | **−0.01** | **−0.121** [−0.172, −0.071]; [−0.170, −0.074]. Plain −0.11, two-clip +0.02, reversed −0.19, stacked −0.20 |
| **C12b captioner, Sonnet − Qwen captions** | **+0.09** | **+0.125** [+0.074, +0.175]; [+0.070, +0.182]. Plain +0.11, two-clip +0.04, reversed +0.26, stacked +0.02 |
| C12c captioner, Haiku − Qwen captions | +0.05 | +0.067 [+0.020, +0.114]; [+0.019, +0.118] |
| C12d Gemini frame-clip − Qwen captions | +0.08 | +0.152 [+0.098, +0.205]; [+0.097, +0.208] |
| C12e Gemini frame − Qwen captions | +0.14 | +0.178 [+0.125, +0.236]; [+0.122, +0.239] |
| C12f Gemini video captions − frames | +0.12 | +0.040 [−0.010, +0.091]; [−0.007, +0.085]. Two-clip +0.18, stacked −0.10 |
| C12g cap: Qwen − Haiku on Sonnet captions | −0.05 | **−0.047** [−0.094, +0.000]; [−0.094, +0.000]. Plain +0.01, **two-clip −0.29** [−0.40, −0.17], reversed +0.07, **stacked −0.17** [−0.28, −0.05] |
| C12h question only: Qwen − Haiku | 0.00 | −0.013 [−0.077, +0.051]; [−0.076, +0.047] |
| Retention (clustered): own / Sonnet / Gemini video captions | 0.95 / 1.40 / 1.60 | 0.53 [0.34, 0.70] / 1.01 [0.85, 1.20] / 1.16 [0.98, 1.37] |

- **Reading (rules set in advance):**
  - **Claim 1 holds for an open reader:** C12b's CI is above 0, and is larger than Claude's C3a (+0.064). Every other captioner also beats Qwen's own captions (C12c–e).
  - **Claim 2 holds:** C12a's upper bound (−0.071) is far below +0.09. The prediction (−0.01) missed by 0.11. The sign is the one Sonnet showed (−0.088), not Haiku's (+0.030).
  - **Claim 2a holds at the boundary.** |C12g| = 0.047 ≤ 0.05, and the CI's upper bound is exactly 0.000 (32 questions lost and 18 gained of 297), so it includes 0 only as its endpoint. The failure rule (≤ −0.08 with a CI below 0) is not met. **Say:** Qwen reaches the cap overall but not on two-clip and stacked videos, the types where the answer must be combined across windows or across the two copies. The cap is reached on the plain and reversed questions.
  - **Frames help Qwen** (+0.26 over question only, more than Haiku's +0.20), so the terms against `frames5` measure the modality, not a reader that cannot use 30 small images.
- **Where the predictions went wrong (descriptive):**
  - The predictions treated Qwen as a weaker Haiku. In fact, Qwen on frames beats Haiku on frames (0.657 vs. 0.613; plain 0.86 vs. 0.80), while Qwen as captioner is the weakest captioner in the study. Its own captions are below Haiku's self-captions even for Qwen as reader (C12c), and lose most on reversed (0.26) and stacked (0.24) videos.
  - So the modality effect's sign follows reader strength on frames relative to the index, not model size: Haiku +0.03, Qwen −0.12, Sonnet −0.09. This is the cap of claim 2a again: a reader's own captions help only a reader that sits below them on frames.
  - It runs against *Caption This, Reason That* (a 7B model gained +22 to +29 points from its own captions, on images). Our 7B reader loses 12 points on video.
  - C12f (+0.04) is a third of its prediction for the same reason: Qwen's frames are strong.
- **Sensitivity without the 49 refusals (added after the result; descriptive):** every reading holds, and the effects shrink by 0.01–0.02. Clustered: C12a −0.103 [−0.152, −0.055], C12b +0.106 [+0.052, +0.164], C12g −0.041 [−0.090, +0.010], C12h +0.000, own-caption retention 0.62 [0.43, 0.79].
- **For the paper:** the closed-reader limitation gets an answer. The captioner effect and the absence of a modality gain replicate on an open model run with greedy decoding that anyone can rerun. The cap is reached by a 7B reader overall, but cross-window reasoning on text needs at least Haiku. Qwen's rows go into Table 2 (accuracy only; no cost, since the GPU is free) and the generated appendix table.

### D4 — Paper assembly checks (2026-10-27; no calls, descriptive)

- **Predictions against results, over all 50 accuracy differences with a numeric prediction (P1–C12h; paper Appendix B, Table `tab:preds`):** the prediction lies inside the result's CI for 39 of 50, and for 37 of the 41 after C1. Of the 44 non-zero predictions, 42 have the result's sign (exceptions: P2 and S2, both set from the 50 development questions). The median |estimate − prediction| is 0.021. The misses after C1 are C4e and C12a, C12d, C12f.
- **Figure 3** (`plots.py readers`) now has Qwen as a third reader, ordered by accuracy on frames: Haiku 0.613, Qwen 0.657, Sonnet 0.798. On Sonnet's captions: 0.707, 0.660, 0.710; on Gemini video captions: 0.737, 0.697, 0.791; question only: 0.411, 0.397, 0.414 (95% CIs over clip clusters in `fig3` output).
- **Haiku's retention by video type stays unreported,** as D3 decided (the stacked CI is unbounded); the paper's Haiku headroom table gives retention overall only.
- **Number audit:** every decimal number in `caption_index.tex` appears in this log, in `results/index_qa/reports/*.txt`, or (for the reconstruction phase) in `docs/TODO.md`, except simple derivations (0.185 = 0.798 − 0.613; 1.76 = 2,676 / 1,520; 3.6 = 720 / 200; cost columns scaled by 10³ or 10⁴; 0.021 above).

### D5 — Q\* by video length, and fixes from an external review (2026-10-27; no calls, logged tokens only)

**Why.** An external review of the PDF (Gemini, `docs/new_direction/argon_review.md`) pointed out that on ~10 s clips the fixed reader prompt (≈ 1,025 tokens) dominates the input, so the same-reader savings and Q\* may not carry over to longer videos. The analysis was not predicted in advance; it is descriptive.

**Method** (`reader.py cost`, function `qstar_by_length`; output in `results/index_qa/reports/cost.txt`):
- An arm's cost per question at W windows is its measured mean plus its input tokens per window, at its input price, times (W − mean W).
- The input tokens per window are the slope of input tokens on W. That slope is exact by construction: frames 200; captions 63–87.
- Output and thinking tokens stay at their mean. This is an assumption. On these clips, output falls with W, but that is confounded with video type (stacked and two-clip videos are short).
- The index costs its mean per window times W, since each window is its own call. The reader reads every window.
- Check: at the mean W = 2.56 the projection reproduces the measured Q\* (202.7, 5.9, 2.3).

**Results** (Q\* at W = 2.56 / 12 / 120 / ∞; cost per question, captions ÷ frames, at 2.56 / 12 / 120):

| Index vs. frames | Q\* | cost ratio |
|---|---|---|
| Haiku on Gemini video captions vs. Haiku | 203 / 212 / 214 / 214 | 0.82 / 0.59 / 0.38 |
| Sonnet on Gemini video captions vs. Sonnet | 5.9 / 9.1 / 10.5 / 10.7 | 0.70 / 0.53 / 0.37 |
| Sonnet on own captions vs. Sonnet | 8.9 / 13.5 / 15.4 / 15.6 | 0.75 / 0.60 / 0.46 |
| Haiku on Gemini video captions vs. Sonnet | 1.8 / 4.5 / 6.8 / 7.2 | 0.04 / 0.03 / 0.02 |
| Haiku on Gemini frame-clip captions vs. Sonnet | 1.9 / 4.7 / 7.1 / 7.5 | 0.04 / 0.03 / 0.02 |
| Haiku on Sonnet captions vs. Sonnet | 2.3 / 5.6 / 8.5 / 9.1 | 0.04 / 0.03 / 0.02 |
| Haiku on Gemini frame captions vs. Sonnet | 3.7 / 8.9 / 13.6 / 14.5 | 0.04 / 0.03 / 0.02 |

**Reading.**
- The review was half right. The saving per question does grow with length. Q\* does not fall, because the captioning cost grows with every window too: Q\* → k / (s_f − s_τ).
- Q\* rises for Sonnet and for the reader downgrade. Their short-clip savings partly came from per-question costs that don't scale with W: the expensive reader's fixed prompt, and its thinking.
- Halving caption tokens would change s_f − s_τ by only ≈ 20–30%. The captioning cost k decides the long-video Q\*: thinking is ≈ 87% of a Gemini caption call.
- A reader that retrieves a few windows saves less per question, so for it these Q\* are lower bounds.

**Paper changes (same day):**
- §2.3 now defines the four effects in prose.
- §3 sets Propositions 1–3 apart, gives Q\* as a display equation, and adds its limit as W grows.
- §5 has a short "Longer videos" paragraph; the full projection is in Appendix E.
- The cost figure spans two columns with legible labels. The question-only points are dropped and given in the caption.
- The abstract and intro state the plain-video result early (retention 0.95; index loss ≈ 0.01).
- Limitations is now prose, without run codes, and says that every caption follows one generic prompt.
- The symbol Q\* is now Q_BE (macro `\qbe`), named "the break-even quantity" where it is defined. Q\* usually means an optimal quantity (e.g. EOQ), and this is not one. This log and `reader.py` keep "Q\*".
- Cuts to stay at 8 pages: Discussion merged into one section; the intro, open-reader, audit and Conclusion paragraphs shortened; the reference-similarity point is left to Appendix E.

### D6 — EgoSchema feasibility: partial download and a 180-image call (2026-10-27; 1 Haiku call, not data)

**Why.** These checks decide whether EgoSchema (3 min, 36 windows) can be the length test in `nextqa_plan.md`. The user approved one pilot call. It was a feasibility check, not an experiment, so there is no prediction. Its question is out of every future sample.

- **Data.** `lmms-eval/egoschema` (redirected from `lmms-lab`), MIT licence, ungated. The "Subset" split has 500 questions with answers, one per video, with 5 options.
- **Partial download.** HTTP range requests on the five zips (20–21 GB each, 5,031 mp4 in total).
  - Each zip's central directory reads in ≈ 2.5 s (4 requests, 0.1 MB).
  - All 500 subset videos are found: 101, 92, 91, 118 and 98 per chunk.
  - One video (14.2 MB, deflate) extracted in 3.2 s, fetching 14.7 MB.
  - The test script is in the session scratchpad; a repo version is to be written with the loader.
- **The 180-image call.** One Haiku call (`claude -p`, effort medium, the reader's system prompt), with the frames arm built like `reader.build`'s `frames5`. Video `8345d177-0ed4-4965-8455-bd17d2b42399`:
  - The video is 480×360, 30 fps, 180.0 s, so 36 windows and 180 frames of 224×168 (48 tokens each).
  - The message was 3.6 MB.
  - Return code 0; 10,897 input tokens; 265 output tokens (254 of them thinking); 5.0 s; CLI cost $0.00122.
  - The answer was E, which is correct.
- **Reading.** Both blockers are gone. Haiku on frames costs ≈ $0.0012 per question at W = 36. By D5's input slope, Sonnet would cost ≈ 20× that. **Video `8345d177-…` is excluded from every EgoSchema sample.**

### E1 — EgoSchema development run (written 2026-10-27, committed before any E1 call)

**Status.** A development run, as K0–K2 were for TempCompass. It fixes nothing in the prompts, which are TempCompass's, unchanged. It sets the numeric predictions of the confirmatory run (E2, a separate entry written after E1's result). The user chose EgoSchema first on 2026-10-27, with NExT-QA to be reassessed afterwards (`nextqa_plan.md`).

**Why.** The paper's videos are short (2.6 windows on average). E1–E2 test the claims at 36 windows, where the reader must keep 180 frames apart and where D5's cost projection can be checked.

**Data and sample** (`egoschema.py`, seed 0, drawn before any call on these videos):
- `ego_dev`: 40 questions on 40 videos. `ego_confirm`: 160 questions on 160 other videos. Both are in git.
- Every video is 180 s, so 36 windows of 5 frames: 1 per second, 224×168, 48 tokens each.
- Five options per question; chance is 0.20. One question per video, so a paired bootstrap over questions is also over videos.
- The D6 pilot video is excluded.

**Arms** (TempCompass's prompts and settings: `claude -p`, effort medium, isolated calls):

| Reader | Arms |
|---|---|
| Haiku | question only; 1 frame per window (36 images); 5 frames (180 images); self-captions; Sonnet captions; Sonnet whole-video caption |
| Sonnet | question only; 5 frames; Sonnet captions (its own) |

Captioning:
- Haiku and Sonnet each caption every window from its 5 frames (`self_caption.py`): 1,440 calls each.
- Sonnet also writes one whole-video caption per video from all 180 frames (`--whole`, 40 calls). C10's prompt allows up to 72 sentences and 1,620 words at W = 36.

**Volume:** ≈ 2,920 captioning calls and 360 reads; ≈ 1,600 of them Sonnet. Quota guards: 5-hour 0.7, 7-day 0.9.

**Predictions** (40 questions, so a paired 95% CI is about ±0.15; these are rough, from TempCompass and D5):

| Term | Prediction | Reasoning |
|---|---|---|
| Question only, Haiku and Sonnet | 0.40 each | Text-only readers are known to do well above chance on EgoSchema |
| Haiku, 5 frames | 0.60 | |
| Sonnet, 5 frames | 0.72 | |
| Captioner effect: Haiku on Sonnet captions − on own | +0.06 | TempCompass +0.064 |
| Modality: Haiku own captions − frames | +0.03 | 180 frames are hard to keep apart; per-window captions help, as on two-clip videos |
| Modality: Sonnet own captions − frames | −0.04 | Gist questions written from narrations should suit captions better than TempCompass's built videos (−0.088) |
| The cap: Haiku − Sonnet on Sonnet captions | 0.00 | TempCompass −0.003 |
| Whole-video − per-window Sonnet captions, Haiku reading | 0.00 | C10 +0.013 |
| 1 frame − 5 frames, Haiku | −0.03 | Gist questions; little gain from the extra frames |
| Q_BE, Haiku on Sonnet captions vs. Sonnet on frames | 7.5 (D5 at W = 36) | If outside [5, 10], D5's assumption of constant output and thinking tokens fails at length. The D6 pilot's 265 output tokens, against ≈ 90 on TempCompass, already hint at this |

**Stop rules.** Stop on any isolation failure. Stop if the first 5-frames call errors or exceeds the 900 s time limit. Stop if more than 5% of 180-image calls fail.

**Then.** Write E2 (`ego_confirm`, 160 questions): main tests, numeric predictions from E1, and their CIs. Commit, push and archive it before any `ego_confirm` call.

**Run notes** (added after the start; facts only, no results):
- **Pre-run evidence.** E1 was pushed to GitHub at 13:14 UTC on 2026-10-27 (`00cf257`). The user then submitted a Software Heritage "Save code now" request, before the first E1 call.
- **Check (13:19 UTC):** `run_e1.sh check` ran 3 Haiku and 3 Sonnet window captions and one 180-image Haiku read. All succeeded, with isolation confirmed. Quota was 5-hour 11% and 7-day 40%.
- **A CLI change since TempCompass.**
  - A caption call now takes 1,417 input tokens (Haiku) and 1,396 (Sonnet), with 240 of them visual.
  - On TempCompass (Oct 20) the two models took equal counts for equal input. Now Haiku's are 21 higher, and the non-visual part is ≈ 60–85 tokens larger.
  - So the CLI's built-in system prompt changed, and it now differs by model. Within E1 this changes nothing. In cost comparisons with TempCompass, it shifts the fixed prompt by ≈ 5–8%.
- **Caption length.** Haiku's EgoSchema window captions run longer than on TempCompass: median 56 words on the 3 checked windows, against 45. Sonnet's are 43, against 42. The prompt is unchanged.
- **Full run** started at 13:20 UTC: `setsid nohup scripts/index_qa/run_e1.sh run`, detached from the session.
- **Change to the run, not the design (13:35 UTC, user's request):**
  - The 7-day quota window was 2.4 h from its reset, with 58% unused, so unused quota would be lost.
  - E1 was stopped at ≈ 245 captions per model; nothing was lost, because caption files are atomic.
  - On restart, each captioner runs as 3 parallel processes (`self_caption.py --shard K/3`, a stable hash of video and window, so the shards never overlap), and the 5-hour guard is 0.95 instead of 0.7. The 7-day guard stays at 0.9.
  - The calls, prompts and settings are unchanged, and so is every caption's content definition. The order in which windows are captioned changed, which can't affect a caption.
  - At 13:47 UTC, at the user's request, the 5-hour guard was raised to 0.99, to use the window fully before the reset. 5-hour use was then 0.77.
- **Amendment 1 (13:51 UTC, before any reader call on this question's captions): question `egd017` is dropped from every arm.**
  - Its video (`083c5e8e-…`) is black for its first 95 s. That was checked by decoding the video directly, so it is not a frame-extraction error. 95 of its 180 frames, windows 0–18, are near-black.
  - Sonnet answered "ERROR" twice on those windows, which is what the caption prompt asks for when no description can be made. Haiku answered ERROR once, then described "a solid black screen".
  - This follows C1's amendment 2: a question with a window that has no caption is dropped from every arm (`tc.EXCLUDED`).
  - Its question-only and frame reads (Haiku: question only, 1 frame, 5 frames; Sonnet: question only, 5 frames) were already made. They are dropped as well.
  - E1 now has 39 questions.
- **Fix to `self_caption.py`.** Its "first call failed" stop now fires only on a CLI failure, not on an "ERROR" caption. Before the fix, this video's windows stopped 3 of the 6 shards. Restarted at 13:51 UTC.
- **Interruption.** The run reached the 0.99 guard at ≈ 13:58 UTC and waited for the reset. The machine rebooted at 17:29 UTC, during the wait, and that killed the run. It was resumed unchanged on 2026-10-28 at 06:39 UTC and finished at 06:56 UTC.

**Result** (`reader.py e1`, run once; `results/index_qa/reports/e1.txt`). 39 questions, 351 reads, all `ok`: no errors, timeouts or invalid answers, so no stop rule fired.

| Term | Predicted | Result [95% CI] |
|---|---|---|
| Question only, Haiku | 0.40 | 0.41 [0.26, 0.56] |
| Question only, Sonnet | 0.40 | **0.72 [0.56, 0.85]**, outside |
| Haiku, 5 frames | 0.60 | 0.67 [0.51, 0.82] |
| Sonnet, 5 frames | 0.72 | 0.69 [0.54, 0.82] |
| Captioner effect, Haiku | +0.06 | +0.13 [−0.03, +0.28] |
| Modality, Haiku | +0.03 | −0.10 [−0.28, +0.08] |
| Modality, Sonnet | −0.04 | 0.00 [−0.13, +0.13] |
| The cap | 0.00 | 0.00 [−0.13, +0.15] |
| Whole-video − per-window, Haiku | 0.00 | 0.00 [−0.13, +0.13] |
| 1 frame − 5 frames, Haiku | −0.03 | −0.05 [−0.18, +0.08] |

- **Sonnet gains nothing from the video here.** Question only, it scores 0.72, as high as on frames (0.69) or its captions (0.69). Its modality and cap terms are therefore 0 with no information about the representations. Haiku does need the video (question only 0.41, against 0.67 on frames).
- **Q_BE, Haiku on Sonnet captions vs. Sonnet on frames: 6.3 [6.1, 6.4]**, inside [5, 10] and near D5's 7.5. Captioning costs $0.131 per video, and a read costs $0.00049 vs. $0.0215. Haiku on Sonnet's whole-video caption gives the same accuracy (0.69) with a $0.027 index, so Q_BE = 1.2.
- **Output tokens did not stay constant, but they fell instead of rising.** Against TempCompass, mean output per read is 0.59–0.78× for Haiku's frame and caption arms and for Sonnet on frames. It is 1.57× for Sonnet on its captions (72 vs. 46). The D6 pilot's 265 tokens were not typical. Input tokens grow with length as expected: 10.6K for 5 frames, 4.5K for window captions.
- **Spend.** List price $7.9: captions $6.5, reads $1.4. Quota: 7-day +0.06 on 2026-10-27 and +0.03 on 2026-10-28, ≈ 0.09 in all, account-wide. E2 at 160 videos, ≈ 4×, is ≈ 0.36 of a 7-day window.

### E2 — EgoSchema confirmatory run (written 2026-10-28, committed before any E2 call)

**Status.** Confirmatory, after E1 (development). The user chose all 160 `ego_confirm` questions on 2026-10-28. Its result goes in the paper as the length test, next to C1.

**Why.** E1 had 39 questions, so every paired CI was about ±0.15. E2 tests, on 160 new questions on new videos, the claims E1 can only suggest at 36 windows: the captioner effect, Haiku's modality effect and the break-even.

**What E1 changes in the design.** Sonnet scored 0.72 from the question alone, the same as from frames (0.69). Its answers differ between those two arms on 44% of the questions, so it does use the video, but its net accuracy gains nothing. Sonnet's modality term and the cap therefore can't separate the representations here, and they move to the secondary terms. The primary tests are on Haiku as the reader, where the video adds +0.26 over the question alone, and on the cost model.

**Items.** `ego_confirm`: 160 questions on 160 videos, seed 0, drawn with `ego_dev` before any E1 call (`egoschema.py`), in git. No overlap with `ego_dev`. Every video is 180 s, so 36 windows, and 5 options per question, so chance is 0.20. The videos and frames are on disk (2026-10-27), and no `ego_confirm` question has been read.

**Arms and settings.** Exactly E1's: the same 9 arms, prompts, `claude -p` isolation, effort medium, captioners and `--whole` caption (`run_e2.sh` is `run_e1.sh` with `ego_dev` → `ego_confirm` and `e1` → `e2`). A failed or invalid read counts as wrong.

**Exclusion rule, fixed in advance.** As in C1's amendment 2 and E1's amendment 1, a question with a window that a captioner answered "ERROR" or left without a caption is dropped from every arm, and recorded as an amendment before any reader call on that question's captions.

**Primary hypotheses** (Bonferroni over three: 98.33% CIs, paired bootstrap over questions, one per video; a pass needs the CI to exclude 0 on the predicted side). The predictions take E1's value toward TempCompass's, since E1's CIs are wide.

| | Hypothesis | Statistic | Prediction | E1 | TempCompass |
|---|---|---|---|---|---|
| H1 | A stronger captioner adds usable information at length | Haiku: Sonnet captions − own captions | +0.08 | +0.13 | +0.064 |
| H2 | At length, Haiku's own captions lose against its frames | Haiku: own captions − 5 frames | −0.05 | −0.10 | +0.03 |
| H3 | D5's cost model holds at 36 windows | Q_BE, Haiku on Sonnet captions vs. Sonnet on 5 frames: 98.33% CI inside [5, 10] | 6.3 | 6.3 [6.1, 6.4] | 2.3 (W = 2.6) |

**Secondary** (95% CIs, estimates rather than tests):

| | Term | Prediction | E1 |
|---|---|---|---|
| S1 | Question only, Haiku; Sonnet | 0.41; 0.68 | 0.41; 0.72 |
| S2 | Video use, Sonnet: 5 frames − question only | 0.00 | −0.03 |
| S3 | Video use, Haiku: 5 frames − question only | +0.25 | +0.26 |
| S4 | Modality, Sonnet: own captions − 5 frames | −0.03 | 0.00 |
| S5 | The cap: Haiku − Sonnet on Sonnet captions | 0.00 | 0.00 |
| S6 | Whole-video − per-window Sonnet captions, Haiku reading | 0.00 | 0.00 |
| S7 | 1 frame − 5 frames, Haiku | −0.04 | −0.05 |
| S8 | Haiku on Sonnet captions − Sonnet on 5 frames | −0.02 | 0.00 |
| S9 | Q_BE, Haiku on Sonnet's whole-video caption vs. Sonnet on 5 frames | 1.2 | 1.2 |
| S10 | Output tokens per read against TempCompass | below 1× for Haiku's frame and window-caption arms and Sonnet's frames; above 1× for Sonnet on its captions | 0.59–0.78×; 1.57× |

**Exploratory, no predictions.** The video-needed subset: the questions Sonnet answers wrongly from the question alone (E1: 11 of 39). The non-Sonnet-blind terms above are reported on it, with 95% CIs. It conditions on an arm that never sees the video, so it can't be biased by the representations. But it's chosen by an outcome, so it's descriptive only.

**Power, stated in advance.** E1's discordant pairs were 26–31% of questions, so a paired difference on 160 questions has SE ≈ 0.043, and the 98.33% CI is ≈ ±0.10. H1 at +0.08 then has ≈ 30% power, and H2 at −0.05 ≈ 11%. **A H1 or H2 that does not pass is not evidence of no effect**; we would report the estimate and CI as a bound. H3's CI is narrow (±0.2 on E1's 39 questions), so it tests the cost model, not accuracy. There are no more `ego_confirm` questions in the sample, and n is not increased after seeing results.

**What each outcome means for the paper.**
- H1 passes: the captioner effect holds at 3 minutes, about twice as large as on TempCompass, if E1's size holds.
- H2 passes: at length, a weak reader's own captions lose against its frames. TempCompass's +0.03 then turns negative with length, so the modality effect's sign depends on the length as well as the reader.
- H3 passes: the paper's cost projection (D5) holds on real 3-minute videos.
- S2 near 0: on EgoSchema a frontier reader gains no net accuracy from the video, a property of the benchmark that we report next to the results, not a finding about captions.

**Volume and quota.** 11,520 window captions (5,760 per captioner), 160 whole-video captions and 1,440 reads, ≈ 13,100 calls. List price ≈ $32. Quota ≈ 0.35–0.40 of a 7-day window, which was 0.03 at the end of E1. About 4 five-hour windows, so ≈ 15–20 hours with the waits.

**Run order and stop rules.**
1. `run_e2.sh check`: 3 window captions per captioner and one 180-image read. Its calls count as data, as in E1.
2. `run_e2.sh run`, detached. 5-hour guard 0.99 (E1's final setting, at the user's request), 7-day guard 0.9. It resumes itself after each 5-hour reset.
3. Stop on any isolation failure. Stop if the first 5-frame call errors or exceeds the 900 s time limit. Stop if more than 5% of 180-image calls fail. If the 7-day guard stops the run, ask the user before resuming.
4. No decision depends on interim accuracy. `reader.py e2` (committed with this entry) is run once, when all arms are complete, and its output is reported here in full (`results/index_qa/reports/e2.txt`).

**Run notes** (added after the start; facts only, no results):
- **Pre-run evidence.** E2 was pushed to GitHub on 2026-10-28 (`a9f1fea`). The user then requested a Software Heritage "Save code now" visit, before the first E2 call.
- **Check (08:58 UTC):** `run_e2.sh check` ran 3 Haiku and 3 Sonnet window captions (video `c50c237d-…`, windows 0–2) and one 180-image Haiku read (`egc000`, 5 frames). All succeeded, with isolation confirmed (no tools, no MCP, 1 turn, the model as requested). The captions describe their frames, and input tokens match E1's (1,417 Haiku, 1,396 Sonnet; the read 10,587). Quota was 5-hour 43% and 7-day 4%.
- **Full run** started at 08:59 UTC: `setsid nohup scripts/index_qa/run_e2.sh run`, detached from the session.
- **Stopped at 09:52 UTC (the user's decision, during the first 5-hour wait), for a reason outside the data: contamination.**
  - The readers' training data runs to June 2026, and EgoSchema has been public since 2023.
  - KAIROS (arXiv 2609.08755, September 2026) measures a question-only frontier model at +33.6 points above chance on EgoSchema, and E1 measured +52 for Sonnet.
  - The quota is kept for a benchmark released after the readers' cutoff.
  - Saved at the stop: 822 Haiku and 777 Sonnet window captions of 5,760 each, no whole-video captions, and 247 Haiku and 168 Sonnet reader rows.
  - **No E2 result was computed or looked at:** `reader.py e2` was not run on `ego_confirm`, and no reader row was read. E2 stays resumable (`run_e2.sh run`) with its design unchanged. If it is resumed, this stop is reported with it.

### P1 — Reconstruction post-mortem: the prior-metric correlations, and do captions track frames? (2026-10-28; exploratory, not written in advance; no calls)

**Why.** To explain to the team why reconstruction is future work, not a quick salvage. The team's main result was that the a-priori metrics predict which modality reconstructs a video better: the rank difference rank(Llama cos_sim) − rank(SigLIP interpolation cos_sim) over the 335 videos correlates with text dynamism (ρ up to +0.44) and visual dynamism (−0.37), p ≈ 1e-17 (`scripts/analyze_method_rank_differences.py`). Not written in advance; a post-mortem, not a test of a new claim.

**Scripts and outputs.** `scripts/postmortem/prior_correlation_placebo.py` and `scripts/postmortem/change_tracking.py {wild,windows,nocut,ego}`, run from the repo root. Outputs: `results/postmortem/*.txt` (gitignored). The EgoSchema frame embeddings are cached in `local/egoschema/siglip2_windows/` (≈ 35 min on CPU to rebuild).

**1. Placebo: copying reproduces the correlations.** Replace Llama by caption copy (no model) in the same rank difference, i = 29:

| ρ at W = 6 | Llama (the team's result) | Caption copy | Caption mean |
|---|---|---|---|
| text_average_dynamism | +0.44 [+0.35, +0.53] | +0.52 [+0.44, +0.59] | +0.53 |
| average_dynamism (visual) | −0.37 [−0.47, −0.26] | −0.41 [−0.51, −0.30] | −0.38 |
| APCS_V | +0.25 | +0.38 | +0.35 |

- The same pattern holds at W = 3 and 12. CIs are channel bootstrap.
- Each prior predicts its own modality's copy score almost by definition: text dynamism vs. the caption copy score ρ = −0.67, visual dynamism vs. the SigLIP interpolation score ρ = −0.70. Both are cosine distances between neighbours in the same embedding space.
- Copy's rank difference explains R² = 0.55 of Llama's. The four priors add 0.06.
- Within-modality ranks are also zero-sum (`docs/theory/cross_modal_evaluation_metrics.md` §7.1), so "text wins" on about half the videos by construction.

**2. Residual: priors vs. Llama − copy**, the only target that is about the model. Llama trails copy on average at every W (ahead on 23–41% of videos). Correlations are |ρ| ≤ 0.26. 7–8 of 36 survive a channel-bootstrap Bonferroni (the count moves with the bootstrap seed). All say one thing: Llama loses less to copy on visually static videos (APCS_V +0.18 to +0.26), where every caption is near the same.

**3. Do caption changes track frame changes?** The 2026-10-07 audit found per-video Spearman ≈ 0 between the caption change curve and the frame change curve, at 1 s. At 5 s they track:

| Data (5 s windows) | Caption set | ρ(caption change, frame change) | Frame split-half ceiling |
|---|---|---|---|
| Wild pilot, 30 videos × 12 pieces | Gemini isolated | +0.43 [+0.33, +0.53] | +0.79 |
| | Gemini in-context | +0.22 [+0.09, +0.35] | |
| | Old 1 s captions, joined per piece | +0.34 [+0.22, +0.45] | |
| EgoSchema dev, 39 videos × 36 windows | Sonnet | +0.37 [+0.30, +0.44] | +0.66 |
| | Haiku (same 5 frames) | +0.38 [+0.33, +0.44] | |

- The 1 s result reproduces on the same 30 videos. The old captions' agreement rises with window size: 0.00, 0.12, 0.26, 0.33 at 1, 2, 3, 5 s, and 0.42 at 5 s with the +2 s lead corrected. At 1 s, wording changes swamp content changes.
- It is not shot cuts. Dropping each video's largest frame change leaves +0.39, and dropping the top 3 leaves +0.30.
- Two caption sets of the same windows agree with each other no more than with the frames (Gemini isolated vs. in-context 0.32, Sonnet vs. Haiku 0.35). The caption-side noise is wording.
- Captions written with the whole clip in view track the frames about half as well as isolated ones (0.22 vs. 0.43). This fits the leak (Part 1).
- SigLIP 2 text gives the same picture as MPNet, slightly lower. All-pairs distance agreement (RSA) is +0.38 to +0.56.

**4. The same checks on Claude's fills** (`scripts/postmortem/claude_recon_priors.py`, added the same day; partial correlations computed inline when the result was read, not yet in the script). Sonnet on 50 videos, Opus on 32, gaps at W = 1, 4, 8, 16. Gain = copy's rank − Claude's rank.
- **Raw:** Claude's gain over copy does correlate with the priors, more than Llama's. The largest values are under lenient text scoring: APCS_V +0.57 to +0.59, APCS_T +0.54 to +0.56, visual dynamism −0.43 to −0.50. Claude gains most on static, repetitive videos.
- **Why:** those are the videos where copy fails. When all captions look alike, copy's rank drops toward chance. The gain correlates with copy's own score at ρ = −0.32 to −0.66.
- **Copy's score held fixed** (rank partial correlation, channel bootstrap; Sonnet and Opus × text rank, text order-free, text segment c, frame rank × 4 priors = 32 tests):
  - the correlations shrink to −0.36 to +0.37, e.g. Opus text rank, APCS_V +0.40 → +0.02; Sonnet order-free, APCS_V +0.59 → +0.37 [−0.05, +0.67];
  - 1 of 32 CIs excludes 0 (Opus frame rank, text dynamism +0.37 [+0.00, +0.66]), about the chance rate.
  - With 32–50 videos the CIs are about ±0.4. This is an underpowered null, not evidence of no effect.
- **Stability:** a video's gain over copy barely repeats across W: mean ρ between widths +0.02 to +0.18 under strict scoring, up to +0.36 under lenient. Sonnet's and Opus's gains on the same 120 gaps correlate at +0.62 (text) and +0.74 (frame), but both contain the same copy term (each correlates +0.39 to +0.43 with copy's failure).

**Decision.**
- The prior-metric correlations are not a reconstruction result. They describe how continuous each video's captions and frames are, which the priors measure directly.
- For Claude, the priors' link to the gain over copy runs through copy's own score. What remains is not detectable with 32–50 videos. Testing it needs a frontier run on all 335 videos with the analysis fixed in advance.
- The "captions barely track frames" audit result is withdrawn as a finding. It belongs to the 1 s time scale.
- Reconstruction stays future work. A repair needs new captions, a task with distractors that defeat copying, a visual arm that reconstructs, and a metric both modalities share.

### A4 — Contamination check on TempCompass (written 2026-10-28, before running; no calls)

**Why.** TempCompass (March 2024) predates the readers' training data (June 2026). The question alone scores 0.11 above chance (C8). Do the headline effects rest on remembered answers? The user approved this check on 2026-10-28, instead of a new benchmark first (no clean post-cutoff short-video benchmark was found; see `docs/TODO.md`).

**Idea.** A remembered answer is a function of the question text. TempCompass's conflicting videos ask the same question, with the same options, of videos with different answers. On such a question, the text cannot tell which answer is right.
- **Text-ambiguous question:** TempCompass contains a question with the same stem and the same set of option texts whose correct option text differs. Computed from the data before this entry: **197 of the 297 confirmatory questions** (speed 42 of 60, direction 53, order 45, attribute change 57, action 0). All 193 C4 pairs are text-ambiguous. The other 100 are the **text-unique** questions (action 60, speed 18, order 12, direction 7, attribute change 3).
- If remembered answers drive an effect, the effect sits in the text-unique questions and shrinks on the text-ambiguous ones.
- Pair accuracy (C4) is immune by construction: an answer that depends only on the text is the same on both videos, so it gets at most one right. The pair-accuracy terms were computed in C4 and are restated here, not predicted. The cap's pair-accuracy CI is new (C4 printed only the two arms' means, both 0.430).

**Terms** (per question accuracy; the paper's headline differences):

| Term | Arms |
|---|---|
| T1 captioner effect | Haiku on Sonnet captions − Haiku on self-captions |
| T2 modality effect | Haiku on self-captions − Haiku on 5 frames |
| T3 cap | Haiku on Sonnet captions − Sonnet on Sonnet captions |
| T4 index vs. direct | Haiku on Sonnet captions − Sonnet on 5 frames |

**Outcomes** (`reader.py a4`, committed with this entry; output to `results/index_qa/reports/a4.txt`):
1. Question-only accuracy minus chance (1 / number of options), Haiku and Sonnet, on text-ambiguous and text-unique questions, and the difference (95% CI, questions resampled within each group).
2. T1–T4 on text-ambiguous questions, on text-unique ones, and the difference (95% CIs as in 1).
3. The same difference within aspect, on the four aspects that have both kinds (speed, direction, order, attribute change; 40 text-unique questions): the per-aspect differences weighted by the aspect's question count, CI from resampling within each aspect × kind cell. This removes action, which is all text-unique.
4. T1–T4 on pair accuracy, 95% CIs over clip clusters (C4's method).

**Predictions.**

| Outcome | Prediction |
|---|---|
| 1. Question-only excess over chance, text-unique (both models) | +0.20 |
| 1. Question-only excess over chance, text-ambiguous (both models) | +0.04 |
| 2. T1–T4, text-ambiguous − text-unique | each within ±0.06, CI includes 0 |
| 3. T1–T4, the same within aspect | each within ±0.06, CI includes 0 |

**Reading.**
- Contamination is ruled out as the source of a term if the term on text-ambiguous questions has the sign of its full-set estimate and the 2/3 differences' CIs include 0. With 100 (and 40) text-unique questions, a null difference is weak evidence alone. Outcome 4 is the strong evidence.
- If a term is clearly larger on text-unique questions, within aspect too, that term is reported as possibly inflated by remembered answers. The paper then cites its text-ambiguous estimate.
- The paper's Contamination paragraph is updated with the result either way.

**Result (2026-10-28; `reader.py a4`, run once after the entry was pushed in `b5064ad`; output in `results/index_qa/reports/a4.txt`).**

1. **The question alone is at chance on the text-ambiguous questions.** All of C8's leakage is on the text-unique ones.

| Question only, minus chance | Text-unique (100) | Text-ambiguous (197) | Ambiguous − unique | Within aspect |
|---|---|---|---|---|
| Haiku | +0.29 [+0.19, +0.39] | +0.02 [−0.04, +0.09] | −0.27 [−0.39, −0.16] | −0.16 [−0.37, +0.03] |
| Sonnet | +0.36 [+0.27, +0.46] | −0.01 [−0.07, +0.06] | −0.37 [−0.48, −0.25] | −0.39 [−0.53, −0.25] |
| Prediction | +0.20 | +0.04 | | |

2–3. **Every term keeps its sign on the text-ambiguous questions, and none is smaller there:**

| Term | All (297) | Text-ambiguous (197) | Text-unique (100) | 2. Ambiguous − unique | 3. Within aspect (237) |
|---|---|---|---|---|---|
| T1 captioner effect | +0.064 [+0.017, +0.111] | +0.081 [+0.020, +0.142] | +0.030 [−0.040, +0.100] | +0.05 [−0.04, +0.15] | **+0.17 [+0.05, +0.29]** |
| T2 modality effect | +0.030 [−0.020, +0.081] | +0.061 [−0.010, +0.127] | −0.030 [−0.090, +0.040] | +0.09 [−0.00, +0.18] | −0.02 [−0.20, +0.14] |
| T3 cap | −0.003 [−0.047, +0.040] | +0.015 [−0.041, +0.071] | −0.040 [−0.100, +0.010] | +0.06 [−0.03, +0.14] | +0.12 [−0.01, +0.25] |
| T4 index vs. direct | −0.091 [−0.145, −0.040] | −0.107 [−0.178, −0.036] | −0.060 [−0.130, +0.010] | −0.05 [−0.15, +0.05] | +0.05 [−0.10, +0.20] |

4. **Pair accuracy** (193 pairs, clip clusters): T1 +0.104 [+0.017, +0.190], T2 +0.093 [+0.028, +0.161], **T3 0.000 [−0.065, +0.058]** (new), T4 −0.176 [−0.276, −0.079].

- **Predictions.** Question only: text-ambiguous held (+0.02 and −0.01 against +0.04). Text-unique was larger than predicted (+0.29 and +0.36 against +0.20). The "within ±0.06" prediction failed for 4 of the 8 differences: T1 within aspect (CI excludes 0), T2 overall, T3 within aspect and T4 overall (the last on the edge, −0.047). **Every miss but T4's has the text-ambiguous estimate larger,** the opposite of what remembered answers would do. T4 is more negative on the text-ambiguous questions, which is also not inflation: caption arms do worse there than Sonnet on frames.
- **Reading (rules above).** No term is larger on the text-unique questions. Contamination is not the source of T1–T4. The text-unique questions are the ones the question alone half answers (Sonnet +0.36). Every arm scores high on them, which shrinks the differences between arms. T1's within-aspect difference rests on 40 text-unique questions.
- **For the paper:** 197 of 297 confirmatory questions have the same text with another answer in TempCompass itself. On them, the question alone is at chance, and the captioner effect (+0.08), the cap (+0.02) and index vs. direct (−0.11) hold.

### N2a — Novelty check for the captioner effect (Q5 of `final_search_brief.md`; partial, 2026-10-20)

**Scope.** Only Q5 (has anyone shown that a stronger captioner adds usable information from identical input?), done while C3 ran, because it decides the novelty of the draft's claim 1. Five web searches. It does not replace the full pass (G6, N2).

**New since N1:**
- [CapRL](https://arxiv.org/abs/2509.22647) (ICLR 2026) trains a captioner with a reward: how well a text-only LLM answers multiple-choice questions from the caption alone. Its follow-up [CapRL++](https://huggingface.co/papers/2606.09393) (June 2026) adds video. Caption utility for a fixed text-only reader is therefore an established *training signal*. There is no self-caption control, and the reader never sees the pixels.
- [MetaCaptioner](https://arxiv.org/abs/2510.12126) (images) ranks captioners by a text-only LLM's accuracy on visual tasks from their captions. It is a captioner comparison with a fixed reader, but with no self-caption arm and no frame view for the reader.
- [Vamos](https://arxiv.org/abs/2311.13627) (ECCV 2024; verified, it was a lead from memory) finds free-text descriptions competitive with visual embeddings for an LLM reasoner on five video benchmarks, where the visual embeddings add little or nothing. That is in line with our bound on the modality effect, but it is learned, on long and egocentric video, with no control of what the captioner saw.

**Verdict for Q5: close, not scooped.** Varying the captioner in front of a fixed reader is established (Prism, LLoVi, MetaCaptioner, CapRL). Not found, on images or video:
- a **self-caption control**: the reader captions the same pixels itself;
- **identical or matched visual tokens** for the two captioners;
- **the reader's own view of the frames** as the third arm, which turns the captioner comparison into the decomposition (captioner + modality).

Claim 1 must be worded as the controlled decomposition, not as "a better captioner helps". Prism and CapRL are cited as the closest work.

### N2 — Literature pass, fourth (G6; 2026-10-21)

**Scope.** All of `final_search_brief.md` (Q1–Q7), run after C1–C7 against draft v1. Details, the paper table and the search log are in §10 of `related_work_caption_index.md`.

**Verdicts:**
- **Q1, the decomposition: open.**
- **Q2, captions vs. frames at matched cost on short video: open.**
- **Q3, which aspects survive the caption: open.** It is no longer our claim (P1 failed).
- **Q4, no modality advantage for lossy text: open,** with close image work whose sign varies:
  - *Caption This, Reason That* (NeurIPS 2025): a weak 7B model gains +22 to +29 points from its own captions, and GPT-4o +8, not significant.
  - *GEASS*: appending the model's own caption costs ≈ 10 points.
- **Q5, the captioner effect: close, not scooped** (N2a).
- **Q6, break-even with a reader downgrade: open.**
- **Q7, new work since Oct 15: nothing above related** (MIT³, Industrial ReMEmbR).
- **No scoop on any question.**

**Changes to the paper:**
1. Frame the modality result with *Caption This* and *GEASS*. The modality effect's sign depends on the reader, which fits the cap in draft claim 2a.
2. Cite MotionBlind, TimeBlind and TempCompass's conflicting videos as precedent for minimal pairs. Ours applies the pairs to representations.
3. Contrast C5 with *Seeing and Solving*: there, a one-call "state first" prompt did not help untuned open models on charts, while ours helps a frontier reader on video (+0.07).
4. Repeat Q7 in the week before submission.
