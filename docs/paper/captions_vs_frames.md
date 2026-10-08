# What Dense Video Captions Keep and Lose: Topic Without Timing

Working draft, started 2026-10-08. Supersedes `draft.md` (the reconstruction-first thesis, whose H1/H2 our own tests rejected). Numbers come from the scripts named in each section; anything not yet verified is marked **TODO**.

**Thesis.** Dense per-second captions written by a frontier video LLM share a video's *topic* with its frame embeddings, but barely track *what changes second to second*, and they are systematically 1–2 s early, in every channel. That gap caps any text-only method that tries to recover missing seconds: an 8B LLM in-filling captions loses to simply copying the nearest caption, and both lose badly to copying the nearest frame.

## Abstract

**TODO** (write last). Points to cover: dense captions as a stand-in for video; a no-masking audit of captions against SigLIP 2 frames (grounding, shared variation, change timing); the ~2 s lead; where grounding is weak, caption copy trails frame copy more; gap reconstruction from text (Llama-3.1-8B) loses to copy on every metric, including a forced-choice test.

## 1. Introduction

- Dense captions are increasingly used as a proxy for video: for retrieval, for QA, and as training targets. The implicit assumption is that a caption sequence is a time-aligned, lossy-but-faithful transcript of the frames.
- We test that assumption directly, on 335 one-minute clips from 40 YouTube channels (WildQA), captioned once per second by Gemini 3 Flash and embedded once per second with SigLIP 2.
- Findings:
  1. Captions are grounded at the level of the video but weakly at the level of the second (MRR 0.142 vs. 0.079 chance).
  2. They run about 2 s ahead of the frames they describe.
  3. Their second-to-second variation shares almost nothing with the frames' (R² ≈ 0.02), and caption changes do not coincide with visual changes.
  4. Videos whose captions are better grounded are the ones where caption-based gap filling gets closest to frame-based gap filling.
  5. Consequently, text-only reconstruction of masked seconds is capped: Llama-3.1-8B is below caption copy, which is below frame copy, on every metric we tried.

## 2. Data

- **Videos:** WildQA dev (wild4, 100 clips) and test (wild5, 235 clips); first 60 s of each. 40 channels (`movie_id`); every statistic clusters by channel.
- **Captions:** `gemini-3-flash-preview`, one call per video over the whole 60 s at 1 fps, temperature 0.8, thinking enabled (`config/gen_captions/wild{4,5}.yaml`, prompt `prompts/yt_video_processing/one_minute.txt`). The prompt asks for 60 self-contained captions, one per 1-s interval, each describing "what is visible during that 1-second interval". The model sees the whole clip at once, so nothing forces caption t to be about second t only.
  - Correction to `draft.md`: it said Gemini 1.5 Flash on isolated 1-s clips. That is wrong.
- **Frames:** one frame per second (row t = the frame at t s; checked against exact-time ffmpeg extraction, offset 0 in 16/16 checks, `scripts/check_frame_timestamps.py`), embedded with SigLIP 2 base (timm `vit_base_patch16_siglip_224.v2_webli`, 768-d). Text is put in the same space with the SigLIP 2 text tower (`google/siglip2-base-patch16-224`, lowercased).
- **Text-to-text space:** `all-mpnet-base-v2`.

## 3. Captions vs. frames, without masking

Source: `scripts/caption_vs_siglip_audit.py` → `results/caption_audit/`.

### 3.1 Grounding (Q1)

- Retrieve frame t among the video's 60 frames from caption t: MRR 0.142 (chance 0.079; frames circularly shifted ≥10 s: 0.063). Best frame within ±2 s for 22% of captions (chance 8%).
- **Lag.** Mean z-scored similarity of caption t to frame t+k peaks at k = +2 (z 0.58 at 0, 0.72 at +2, 0.35 at −2). Per video, the best lag is +1 or +2 in 222/335 videos and 0 in 12.

### 3.2 Robustness of the lag

Source: `scripts/caption_lag_robustness.py` → `results/caption_audit/lag_*.csv`. CIs are channel-bootstrap 95%.

- **Shape.** Mean z by lag: 0.58 at 0, 0.72 at +1, 0.72 at +2, 0.65 at +3. So the lead is about 1.5 s, and z(+2) − z(0) = 0.141 [0.122, 0.162].
- **Universal across sources.** The peak is at +1 or +2 in 39 of 40 channels and at +3 in the remaining one. No channel peaks at 0 or earlier, and z(+2) > z(0) in 39 of 40. The profile is the same for wild4 and wild5, and in every caption-length tertile.
- **Mostly a constant offset, with slight shrinkage.** The peak is +2 in each third of the clip. The lead is 0.164 [0.127, 0.203] early, 0.161 [0.130, 0.194] in the middle, and 0.117 [0.089, 0.146] late. The per-video slope of the per-second best lag is −0.33 s per minute [−0.61, −0.04].
- **Re-alignment helps the true captions only.** We score the fill at slot t against frame t+s, using rank among 60 frames (chance 30.5) over 2,535 gaps (W = 1–16):

  | Fill | s = 0 | s = +1 | s = +2 | gain at +2 |
  |---|---|---|---|---|
  | Oracle (true caption) | 22.9 | 20.7 | 20.2 | +2.67 [+2.10, +3.23] |
  | Caption copy | 23.9 | 23.7 | 23.5 | +0.42 [−0.00, +0.82] |
  | Llama-3.1-8B | 28.5 | 28.6 | 28.8 | −0.27 [−0.68, +0.10] |

  The lag sits in the true captions' timing. Gap fills carry no second-level timing to re-align, so fixing the lag cannot rescue them. Two more points stand out:
  - In frame space, caption copy is within one rank of the oracle caption (23.9 against 22.9).
  - Llama is close to chance (28.5 against 30.5).

### 3.3 Shared second-to-second variation (Q2)

- Ridge maps between within-video-centered embeddings, held out by channel: R² ≈ 0.02 in every direction (MPNet→frames, SigLIP 2 text→frames, and back).

### 3.4 Change timing (Q3)

- Per-video Spearman between caption change and frame change: −0.04. Top-10% change events coincide (±1 s) 29% of the time vs. 26% by chance.
- **With the lag corrected** (caption change at t against frame change at t+s): coincidence rises to 33% at s = +1 or +2, against 27% by chance (shuffled caption changes). Per-video Spearman stays at about 0.01. So correcting the lag recovers only a little event alignment, and the two change signals stay essentially unrelated.

### 3.5 Topic vs. timing, by distractor pool

Source: `scripts/eval_shared_target.py` (config `configs/eval_shared_target.yaml`, re-run 2026-10-08 with SigLIP 2 text) → `results/redesign_shared_target*.csv`. Setup:
- Each arm's prediction for a masked second is scored in frame space: SigLIP 2 text or frame against the true frame.
- The score is calibrated AUC c = 2·AUC − 1 (0 = chance, 1 = perfect) against distractor frames from different pools.
- Mid gaps (i = 29), W = 3 and 6, 335 videos; per-video means.
- All pre-registered sanity gates pass.

| Arm (frame space) | same video, ±10 s | same video, far | other video, same channel | other channel |
|---|---|---|---|---|
| True caption | 0.18 | 0.33 | 0.72 | 0.80 |
| Caption copy (nearest boundary) | 0.19 | 0.35 | 0.72 | 0.81 |
| Llama-3.1-8B | 0.05 | 0.14 | 0.58 | 0.70 |
| Frame copy | 0.52 | 0.76 | 0.93 | 0.94 |
| Random caption / frame | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

- **Topic survives.** Against other videos, the true caption reaches 0.80, not far below frame copy (0.94).
- **Timing does not.** Against nearby seconds of the same video, the true caption falls to 0.18, and it is no better than the boundary caption copied into the gap (0.19).
  - The caption written for second t carries no more information about which nearby frame is second t than a caption written for a neighboring second does.
  - Frames keep 0.52 there.
  - The far-minus-near drop is 0.16 [0.13, 0.19] for the true caption and 0.24 [0.22, 0.26] for frame copy (channel bootstrap).
  - This pool uses unshifted captions, so the 1–2 s lead (§3.2) accounts for part of the near-pool weakness.
- **Llama is below copy in every pool**, so it loses topic as well as timing. Pre-registered C1–C4 (Llama against caption copy and caption mean, W = 3 and 6): all INFERIOR, paired difference −0.19 to −0.23, Holm p < 0.001.

### 3.6 Caption correctness (Q4)

**TODO** (needs API budget or manual pass): about 50 sampled seconds, labeled correct / wrong time / hallucinated (e.g. the "runner" camera-holder).

## 4. Consequence: reconstructing masked seconds from text

Setup: mask W ∈ {1,2,3,4,6,8,12,16} seconds centered at t = 29. Text arm: Llama-3.1-8B, prompt `prompts/dense_window/`, T = 0.6. Baselines: caption copy / mean of boundary captions (text) and frame copy / mean of boundary frames (SigLIP 2).

### 4.1 Per-second retrieval

- **TODO:** numbers from `results/unified_benchmark_master.csv` (MPNet for text arms, SigLIP 2 for frame arms; rank among 60). Llama's within-gap ordering is at chance (calibrated c ≈ 0), so gap-level evaluation is the fair one.

### 4.2 Forced choice

Source: `scripts/forced_choice_gap.py`. The gap plus 3 same-length distractor spans are masked jointly; each method picks the gap's content (chance 25%).

| Method | Accuracy (180 items scored by Llama) |
|---|---|
| Llama-3.1-8B, PMI (pre-registered) | 35% (CI 28–42) |
| Caption copy | 52% |
| Caption copy, assignment-aware | 59% |
| Frame copy | 85% |

Over all 1,325 items: caption copy 47% (assignment-aware 54%), frame copy 86% (94%). Llama's errors are independent of caption copy's, but fusion gives no gain (51%, leave-one-channel-out weight). Continuity-matched distractors do not exist in enough numbers to remove the copy advantage (`scripts/forced_choice_matched_feasibility.py`).

### 4.3 Grounding predicts the gaps

Per video, Spearman with channel-bootstrap CIs; partial = controlling for the rates of visual and caption change.

| Agreement metric | caption copy's rank deficit to frame copy | frame copy's forced-choice advantage | Llama's deficit to caption copy |
|---|---|---|---|
| Q1 MRR | −0.51 (partial −0.39) | −0.30 (partial −0.18) | +0.33 (partial +0.26) |
| Q2 MPNet→frames | −0.48 (partial −0.36) | −0.27 (partial −0.16) | +0.32 (partial +0.24) |
| Q3 Spearman | ≈ 0 | ≈ 0 | ≈ 0 |

Where captions are better grounded, caption copy gets closer to frame copy, and Llama falls further behind caption copy.

## 5. Related work

**TODO.** Dense video captioning and temporal grounding; caption-as-proxy pipelines (caption-then-reason video QA); hallucination and temporal misalignment in video LLMs; vision-language embedding spaces (SigLIP 2). Candidates in `sources.md`.

## 6. Discussion and limitations

- **One captioner, one prompt.** The lag may come from whole-video captioning with timestamps. Control: re-caption a few videos per second (TODO, API).
- **SigLIP 2 as the reference for "what is visible".** It is an embedding, not ground truth; Q4 covers what it can't.
- **8B text model.** A stronger model is a scaling point, not a fix: the cap comes from what the captions contain.

## Figures

1. Caption-to-frame lag profile (Q1), overall and by thirds of the clip.
2. Forced-choice accuracy by modality and width.
3. Per-video grounding (Q1 MRR) vs. caption copy's deficit to frame copy.
