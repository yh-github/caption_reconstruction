# Plan: a second benchmark (NExT-QA), a length test (EgoSchema) and query-ready captions

Written 2026-10-27, after the external review (`argon_review.md`) and log D5. **Status (2026-10-27 evening): EgoSchema is under way.** Stage D's development run is log E1, running; see the handover. NExT-QA (stages A–C) is reassessed after EgoSchema's confirmatory run (E2). The rest of this document is the plan as written. Each stage below needs its own research-log entry with numeric predictions, pushed and archived on Software Heritage before its first call (handover rule).

## 1. Why: what the paper rests on now

The paper's claims come from one benchmark. Three weaknesses follow:

1. **One benchmark, mostly built to fool.** 54% of the 297 TempCompass questions are on built videos (reversed, stacked, two-clip). The abstract's natural-video claim (retention 0.95; index loss ≈ 0.01) rests on 137 plain questions.
2. **Short clips.** The median is 9.7 s and the mean 2.6 windows. The cost results at other lengths are projections (D5), not measurements.
3. **One generic caption prompt.** The captions are question-blind and task-agnostic. They are not what someone building an index for retrieval or QA would write. The paper's "question-blind captions lose X" may only mean that *generic* captions lose X.

The CLI confound is already answered by the Qwen reader (C12), so it is not a goal here.

## 2. What NExT-QA can and cannot do

**NExT-QA** ([Xiao et al., CVPR 2021](https://arxiv.org/abs/2105.08276)) is multiple-choice QA with 5 options on natural videos (VidOR, from YFCC100M).
- Its questions are causal ("why", "how"), temporal (previous, current, next) and descriptive (count, location, other).
- The data is on Hugging Face as `lmms-eval/NExTQA` (redirected from `lmms-lab/NExTQA`), ungated. Checked 2026-10-27.
- The MC split has 8,564 questions with answers on 1,000 videos, 1–15 questions per video. Videos are 6.5 GB.
- Types: why 3,329; next 1,399; how 1,173; current 1,165; other 600; location 483; count 322; previous 93.
- Length, from frame counts at an assumed 30 fps: median ≈ 31 s, mean ≈ 38 s, 10th–90th percentile 14–75 s, at most 150 s.
- In 5 s windows: median 7, mean 8. 21% of videos have 12 or more windows (≥ 1 min).
- Most videos are 640 px wide, so 224 px frames are 224×168, or 48 Claude tokens each.

**What it gives us:**
- Natural video, from a different source and question style.
- About 3× the windows of TempCompass.
- Many questions that need inference ("why"), not only perception.
- Published numbers from caption pipelines (LLoVi, VideoTree at 75.6%) to place our absolute accuracies.

**What it does not give us:**
- **It is not long video.** ~40 s is moderate length. NExT-QA can't support a length claim beyond about one minute.
- No minimal pairs: there are no sibling videos, so claims 4 and 5 can't be retested the way C4 tested them.
- Weak temporal headroom. Many NExT-QA questions can be answered from one frame: the ATP study (Buch et al., CVPR 2022) found this and defined an "ATP-hard" subset. We would stratify on it (the file still has to be found) and run a 1-frame arm to measure the headroom.

**For length, the honest option is EgoSchema** (§5). It has 3-minute videos, 36 windows each, and is the standard long-form benchmark for caption pipelines (LLoVi, VideoTree).
- An earlier concern was wrong: the API takes up to 600 images per request for 1M-context models (Haiku 5.5, Sonnet 5.5). Above 20 images, each must be ≤ 2000 px per side, and ours are 224 px. So 180 frames fit in one call, and the design where the captioner sees exactly the reader's frames still holds. A pilot call through `claude -p` confirmed it (§5, log D6).
- Hour-long video (Video-MME long, LVBench) means thousands of frames. The frame arm would have to subsample, which changes the question to *coverage*: the index has seen frames the reader can't afford. That is a different paper, and the Limitations section should say so.

## 3. Which claims each result strengthens or weakens

Claim numbers follow the claims table in `docs/paper/caption_index_draft.md`. The predictions here are directions only. Numeric predictions come from a 100-question development run (§4), as for TempCompass.

| Claim | NExT-QA test | Strengthened if | Weakened or disproved if |
|---|---|---|---|
| **1. Captioner effect** (+0.06 at identical input, Claude chain) | Haiku on Sonnet's captions vs. Haiku on its own, same frames | > 0 on natural video. Then it isn't an artefact of built videos. | ≈ 0. Then the effect needs hard perception (built videos), and the paper must say "where perception is hard" |
| **2. No modality gain** (Haiku +0.03; Sonnet −0.09) | Each reader on its own captions vs. its frames | Haiku ≈ 0 and Sonnet ≤ 0 again | Haiku clearly > 0 at 8 windows. That would extend 2b (one separate call per window helps a weak reader that has to keep many frames apart) into a length effect: a new finding, and the claim would need rewording |
| **2a. The cap** (Haiku = Sonnet on Sonnet's captions) | Haiku and Sonnet on the same Sonnet captions | Equal again, on questions that need inference | Sonnet > Haiku on "why" and "how" questions. Then the cap holds for perception, and reasoning over the caption still matters. This is the sharpest test NExT-QA offers |
| **Abstract: natural video loses ≈ 0** (retention 0.95, 137 questions) | Sonnet on its own captions vs. its frames, all questions and ATP-hard | Index loss ≈ 0 overall and small on ATP-hard | A real loss on temporal ("previous", "next") questions. Then question-blind captions lose ordinary temporal relations, not only stress-test facts. The "losses are concentrated" framing weakens, and a finding is added |
| **3. Downgrade pays; Q_BE** | Measured Q_BE at about 8 windows against D5's projection: Haiku on Sonnet captions vs. Sonnet on frames, 4.7 at W = 8; Haiku on own captions vs. own frames, ≈ 210; Sonnet on own captions vs. own frames, 12.6 | Measured values close to the projection: the cost model is validated, and §5's long-video paragraph rests on a test, not an assumption | Far off, e.g. because output or thinking tokens grow with W. Then D5's assumption fails and §5 must be revised |
| **3a. Whole-video caption ≈ per-window** (C10) | One Sonnet caption per video vs. per window | Still ≈ 0 at 8 windows | Whole-video is better at 8 windows, when facts span windows. Then per-window splitting has a cost that grows with length |
| **4a. Task-aware captions help a little** (C11) | Query-ready captions (§4) | — | — |
| **7. Failures are omissions** (0.61–0.73) | The same audit (judge plus 40 hand labels) | Omission dominates again | Reader errors dominate on "why" questions: reasoning, not coverage, limits the index there |
| *New:* **cost of question-blindness** (§4, stage C) | Query-ready question-blind captions vs. generic vs. describe-then-answer | They close little of the gap: the loss is question-blindness itself, and claim 4's wording holds | They close most of it. Then the paper's losses describe *generic* captions, and the practical message becomes "write captions for the queries you expect" (also a useful result) |

Every row can come out against us. That is the point: the paper is stronger with a second benchmark whatever the outcome, as long as the predictions are committed first.

## 4. Design

**Stage A: development run** (sets the numeric predictions).
- 100 questions on 100 videos, stratified by type. All arms below, Haiku and Sonnet.
- Prompts are fixed before this run and unchanged after it: the TempCompass caption and reader prompts, with "N frames" adapted.

**Stage B: confirmatory run** (pre-registered).
- 300 questions on 300 other videos, one per video, stratified across the 8 types. The ATP-hard subset is a reported stratum, if the file is found.
- 300 is TempCompass's size, so effects of ±0.05 have CIs about as wide.
- Arms, with Haiku and Sonnet reading unless noted:

| Arm | Reader sees |
|---|---|
| Question only | question and 5 options |
| 1 frame (Haiku only) | the middle frame of each window: the temporal headroom check |
| 5 frames | all frames, 1 per second, 224 px |
| Self-captions | the reader's own isolated caption of each window |
| Sonnet captions | Sonnet's isolated caption of the same frames |
| Sonnet whole-video captions | one Sonnet caption of all of a video's frames (C10's prompt) |
| Describe, then answer (Haiku only) | the question-aware upper bound (C5) |

- Main tests, as on TempCompass: the captioner effect (Haiku), the modality effect (Haiku; Sonnet as a secondary test), and the cap (Haiku − Sonnet on Sonnet captions). 98.33% CIs (Bonferroni over three); everything else exploratory at 95%.
- With one question per video, the paired bootstrap over questions is also a bootstrap over videos.
- Then the audit (judge, plus 40 hand labels) and the cost account.

**Stage C: query-ready captions** (exploratory, on stage B's 300 questions).
- Two new caption designs, each one Sonnet call per video like the whole-video arm. They stay question-blind and never mention the benchmark:
  1. **Structured:** fixed fields that retrieval and QA need in general: entities and attributes; actions with manner, direction and speed; counts; spatial layout; what changed and in what order; why someone did something, where it is visible.
  2. **Anticipated questions:** the captioner writes the questions a user would most likely ask about the video, with their answers (the doc2query idea from retrieval), plus a one-line summary.
- Both prompts are written from a generic list of query needs before any NExT-QA call, frozen, and logged.
- Report accuracy and index tokens for generic, task-aware (C11's bullet), structured, anticipated and question-aware (describe-then-answer), with Haiku and Sonnet reading.
- The measure is the share of the gap between question-blind and question-aware that each design closes, and at what cost in tokens.
- D5 says caption length barely moves Q_BE. So these designs should aim for coverage of facts that answer questions, not brevity.
- The same two designs run on the 297 TempCompass questions too (≈ 600 Sonnet captions, ≈ 1,200 reads), to see whether they recover the stacked and reversed losses.

## 5. Stage D: EgoSchema, the length test

**Update (2026-10-27): the user chose EgoSchema first; NExT-QA is reassessed afterwards.**
- Samples: `ego_dev` 40 and `ego_confirm` 160 (`egoschema.py`, seed 0).
- Development run: log E1. Confirmatory: E2, written after E1's result.

- `lmms-eval/egoschema`, MIT licence, ungated. The 500-question "Subset" has public answers (checked 2026-10-27). Every video is 180 s, so 36 windows.
- The videos are in five zips of ≈ 21 GB each (106 GB). **Partial download works (tested 2026-10-27, log D6):** HTTP range requests read each zip's central directory in ≈ 2.5 s, all 500 subset videos are found (101/92/91/118/98 per chunk), and one video (14 MB) extracts in 3 s. So 200 videos is ≈ 3 GB and minutes, not 106 GB.
- **A 180-image call works (D6):** one Haiku call through `claude -p`, with every frame of a 180 s video built like `reader.build`'s 5-frames arm.
  - The video is 480×360, so the frames are 224×168, 48 tokens each.
  - 180 images, a 3.6 MB message, 10,897 input tokens, 265 output tokens (254 of them thinking), 5 s, $0.0012 at list price. The answer was correct.
  - That video (`8345d177-0ed4-4965-8455-bd17d2b42399`) is excluded from every sample.
- Sample: 150 questions on 150 videos (one question per video). Arms: question only; 5 frames (180 images per call); self-captions; Sonnet captions; Haiku and Sonnet reading.
- **What it tests:**
  - The cost model at W = 36. D5 projects Q_BE 7.5 for Haiku on Sonnet captions vs. Sonnet on frames, and a cost per question 1/30 of Sonnet's.
  - Whether a weak reader keeping 180 frames apart gains from per-window captions (2b at length).
  - Whether the cap holds when the questions are about the whole video.
- **Known issue:** EgoSchema has a strong language prior, which the question-only arm measures.

## 6. Volume and cost

These are call counts. Quota is a guess from C8 + C9, which took about 4% of the 7-day quota for ≈ 1,350 Sonnet calls. A 20-call pilot will calibrate it.

| Stage | Captioning calls | Reading calls | Sonnet calls (≈) |
|---|---|---|---|
| A: development (100 videos, ≈ 800 windows) | Haiku 800, Sonnet 800 + 100 whole | ≈ 1,100 | 1,500 |
| B: confirmatory (300 videos, ≈ 2,400 windows) | Haiku 2,400, Sonnet 2,400 + 300 whole | ≈ 3,000, plus ≈ 300 audit | 4,500 |
| C: query-ready (NExT-QA and TempCompass) | Sonnet ≈ 1,200 | ≈ 2,400 | 2,400 |
| D: EgoSchema (150 videos, 5,400 windows) | Haiku 5,400, Sonnet 5,400 | ≈ 900 | 5,900 |
| **Total** | | | **≈ 14,000** (A–C ≈ 8,400) |

- Gemini: none needed (memory rule: Gemini only when Claude can't cover it). A Gemini chain on NExT-QA would cost ≈ $10 if a reviewer wants the second family there.
- Qwen on Kaggle (free) can read stage B's main arms as the open-weights check.
- Disk: NExT-QA 6.5 GB; EgoSchema ≈ 150 videos (a few GB) if range requests work.

## 7. Timeline (ARR January 2027 cycle, deadline about Jan 4; to be confirmed)

| Week | Work |
|---|---|
| 1 (from Oct 28) | Download; a NExT-QA loader next to `tempcompass.py`; frames; sampling; tests; pilot (quota, 40-image calls) |
| 2 | Stage A; log entry with numeric predictions; push, archive; stage B captioning |
| 3 | Stage B reads, audit, analysis; the query-ready prompts frozen and logged |
| 4 | Stage C; stage D captioning if quota allows |
| 5 | Stage D reads; analysis |
| 6–7 | Paper: a second-benchmark section, the cost test, the query-ready result; the main text stays at 8 pages (TempCompass detail moves to the appendix) |
| 8–9 | Buffer, literature query Q7, final read |

## 8. Risks

- **Little temporal headroom on NExT-QA.** If 1 frame ≈ 5 frames, the comparison tests description, not time. Mitigation: the ATP-hard stratum and the temporal question types, reported separately.
- **Contamination.** NExT-QA and EgoSchema are old and public, so the readers may know questions or answers. The question-only arm measures the prior; every effect is a difference between arms with the same prior.
- **Page budget.** A second benchmark needs about half a page; it has to come from TempCompass detail.
- **The paper's framing.** If NExT-QA disagrees with TempCompass, the paper becomes "where captions help and where they don't". That is still a paper, but the abstract changes.
- **Reading time.** About 9 weeks for A–C plus writing is tight but feasible. D is the first thing to drop.

## 9. Decisions for the user

1. Stages A–C on NExT-QA: go or no go.
2. Stage D (EgoSchema): plan for it, or drop it now and say "short and moderate length" in the paper.
3. Stage C: the two designs above, or others (e.g. dense captions with no length limit; a two-level index of a summary plus per-window changes).
4. Whether to add the Gemini chain on NExT-QA (≈ $10).
