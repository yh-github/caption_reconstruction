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
