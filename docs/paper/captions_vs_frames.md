# What Dense Video Captions Keep and Lose: Topic Without Timing

Working draft, started 2026-10-08. Supersedes `draft.md` (the reconstruction-first thesis, whose H1/H2 our own tests rejected). Numbers come from the scripts named in each section; anything not yet verified is marked **TODO**.

**Thesis.** Dense per-second captions written by a frontier video LLM share a video's *topic* with its frame embeddings, but barely track *what changes second to second*, and they are systematically 1–2 s early, in every channel. That gap caps any text-only method that tries to recover missing seconds: an 8B LLM in-filling captions loses to simply copying the nearest caption, and both lose badly to copying the nearest frame.

## Abstract

Dense per-second captions from video LLMs are increasingly used as a stand-in for the video itself: for retrieval, for question answering, and as training data. We test how much of a video such captions actually carry. We caption 335 one-minute clips from 40 YouTube channels once per second with Gemini 3 Flash, then compare each caption with SigLIP 2 embeddings of the matching frames.

The captions keep a video's topic but not its timeline:
- Against frames from other videos, a caption identifies its own frame well (calibrated AUC 0.80; a copy of a neighboring frame reaches 0.94).
- Against frames from the surrounding ten seconds, it falls to 0.18, no better than the caption of a neighboring second, while frames keep 0.52.
- Captions run 1–2 s ahead of the frames in every channel. Correcting this lead recovers only a third of the gap.
- Caption change and visual change are uncorrelated.

These properties cap text-only recovery of missing seconds. An 8B LLM that in-fills masked captions loses to copying the nearest caption, and both lose badly to copying the nearest frame, on per-second retrieval and on a forced-choice test (35%, 52% and 85% accuracy; chance 25%). Videos with better-grounded captions show smaller caption-to-frame gaps.

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
  - **With the lead corrected** (`caption_lag_robustness.py` part 5: slot t scored against frame t+s, pool built around the shifted gap), the true caption rises from 0.17 at s = 0 to 0.25 at s = +1 and 0.28 at s = +2. The gain at +2 is +0.11 [+0.07, +0.15]. It then beats caption copy (0.20) but stays far below frame copy (0.52). Caption copy (+0.01) and Llama (−0.02) don't gain. The lead accounts for about a third of the caption-to-frame gap on nearby seconds; the rest is timing information the captions don't carry.
- **Llama is below copy in every pool**, so it loses topic as well as timing. Pre-registered C1–C4 (Llama against caption copy and caption mean, W = 3 and 6): all INFERIOR, paired difference −0.19 to −0.23, Holm p < 0.001.

### 3.6 Is it the captions or the encoder? A time-aligned ceiling

Source: `scripts/bridge_ceiling.py` → `results/bridge_ceiling/`. The near-pool weakness could come from SigLIP 2's text-to-image link rather than from the captions. To test that, we captioned each frame on its own with Florence-2 base (`<DETAILED_CAPTION>`), which gives captions that are time-aligned by construction. Setup:
- 30 videos from 30 channels, every second;
- the same SigLIP 2 text tower;
- near pool: frames 2–10 s from the target, t ∈ [10, 50);
- 95% CIs by video bootstrap (one video per channel).

| Captions | near c | other-video c | Q1 MRR | Lag-profile peak |
|---|---|---|---|---|
| Florence-2, per frame | **0.47** [0.39, 0.54] | 0.99 | **0.27** | **0** (28/30 videos) |
| Gemini, whole minute | 0.16 [0.12, 0.21] | 0.92 | 0.14 | +2 |
| Gemini, shifted +2 s | 0.21 [0.16, 0.27] | 0.93 | — | — |
| Frame copy (frame t−1) | 0.65 | 0.98 | — | — |

- **The text-to-image link can carry second-level information.** Per-frame captions beat Gemini's on nearby frames by +0.31 [0.24, 0.37], and still by +0.25 [0.19, 0.31] after correcting Gemini's lead.
- **The weak timing is a property of the Gemini captions, not of the encoder.**
- **The lag method is validated.** Per-frame captions peak at lag 0, so Gemini's +1 to +2 s lead is not an artifact of the measurement.
- **Text loses something even when aligned.** Per-frame captions still fall short of frame copy (0.47 vs. 0.65), which bounds what any caption can carry through this encoder.
- **Caveat:** Florence-2 is a different captioner, and its captions are more literal and visual, often near-duplicates across adjacent seconds. So this isolates "aligned, frame-specific text" rather than "the same captioner, aligned". Per-second re-captioning with the same kind of model is the remaining control (TODO).

### 3.7 Caption correctness (Q4)

**TODO** (needs API budget or manual pass): about 50 sampled seconds, labeled correct / wrong time / hallucinated (e.g. the "runner" camera-holder).

## 4. Consequence: reconstructing masked seconds from text

Setup: mask W ∈ {1,2,3,4,6,8,12,16} seconds centered at t = 29. Text arm: Llama-3.1-8B, prompt `prompts/dense_window/`, T = 0.6. Baselines: caption copy / mean of boundary captions (text) and frame copy / mean of boundary frames (SigLIP 2).

### 4.1 Per-second retrieval

Source: `results/unified_benchmark_master.csv`, i = 29, 335 videos. Each arm is scored in its own modality: text arms in MPNet against the true caption, frame arms in SigLIP 2 against the true frame. The metric is the mean rank of the true second among the video's 60 (chance 30.5; lower is better).

| Arm | W=1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 23.0 | 26.3 | 27.5 | 27.4 | 27.6 | 28.0 | 29.5 | 30.2 | 27.4 |
| Caption copy | 23.4 | 23.1 | 22.3 | 22.7 | 22.7 | 23.5 | 24.4 | 25.6 | 23.5 |
| Caption mean | 22.7 | 22.6 | 22.8 | 23.0 | 24.1 | 25.4 | 27.1 | 28.6 | 24.5 |
| Frame copy | 7.4 | 8.2 | 10.3 | 11.1 | 13.7 | 14.9 | 17.6 | 19.5 | 12.8 |
| Frame mean | 6.9 | 8.9 | 10.8 | 12.4 | 15.5 | 17.6 | 21.3 | 23.9 | 14.7 |

- **Llama trails caption copy** by 3.9 ranks [3.1, 4.8] (per video, averaged over widths, channel bootstrap). It wins in 32% of videos.
- **The only exception is W = 1.** There Llama roughly ties copy (23.0 vs. 23.4; MRR 0.119 vs. 0.107).
- **Caption copy trails frame copy** by 10.6 ranks [9.3, 12.0]. It wins in 9% of videos.
- **Gap-level evaluation is the fair one.** Llama's within-gap ordering is at chance (calibrated c ≈ 0), which is why the forced-choice test (§4.2) scores whole gaps.

### 4.2 Forced choice

Source: `scripts/forced_choice_gap.py`. The gap plus 3 same-length distractor spans are masked jointly; each method picks the gap's content (chance 25%).

| Method | Accuracy (180 items scored by Llama) |
|---|---|
| Llama-3.1-8B, PMI (pre-registered) | 35% (CI 28–42) |
| Caption copy | 52% |
| Caption copy, assignment-aware | 59% |
| Frame copy | 85% |

Over all 1,325 items: caption copy 47% (assignment-aware 54%), frame copy 86% (94%). Llama's errors are independent of caption copy's, but fusion gives no gain (51%, leave-one-channel-out weight). Continuity-matched distractors do not exist in enough numbers to remove the copy advantage (`scripts/forced_choice_matched_feasibility.py`).

**A frontier text model** (preliminary; `scripts/blind_llm_runner.py`):
- **Setup:** Claude Sonnet 5.5 answered each item directly, one isolated session per item, seeing only the captions (the same masked context Llama saw, with the 4 candidates in random order). The prompt didn't say that the other candidates fill the other gaps.
- **Sample:** 160 items, 36 channels; channel-bootstrap CIs.

| Method | Accuracy |
|---|---|
| **Sonnet 5.5** | **76.9% [69.6, 83.0]** |
| Caption copy | 45.6% |
| Frame copy | 87.5% |

- **Paired:** Sonnet beats caption copy by 31 points [21, 42] and trails frame copy by 11 [3, 18]. It's flat across W.
- **So the 8B result does not generalize.** A strong reader recovers much of what tells a gap apart from its distractors, which caption copy cannot, but frames still carry more.
- **Protocol caveat:** this is direct choice, while Llama's is PMI scoring. Getting Llama's direct choice would need a GPU run.
- **Generation is a different story so far.** On 10 videos, Sonnet's free-text reconstructions are only about level with caption copy (§4.1). That is consistent with captions carrying topic and event order, but little second-level detail to generate from. **TODO:** a larger reconstruction sample, and Opus as a scaling point.

### 4.3 Grounding predicts the gaps

Per video, Spearman with channel-bootstrap CIs; partial = controlling for the rates of visual and caption change.

| Agreement metric | caption copy's rank deficit to frame copy | frame copy's forced-choice advantage | Llama's deficit to caption copy |
|---|---|---|---|
| Q1 MRR | −0.51 (partial −0.39) | −0.30 (partial −0.18) | +0.33 (partial +0.26) |
| Q2 MPNet→frames | −0.48 (partial −0.36) | −0.27 (partial −0.16) | +0.32 (partial +0.24) |
| Q3 Spearman | ≈ 0 | ≈ 0 | ≈ 0 |

Where captions are better grounded, caption copy gets closer to frame copy, and Llama falls further behind caption copy.

## 5. Related work

**Dense captioning and temporal grounding.** Dense video captioning asks a model to localize events and describe them [Krishna et al. 2017]. Moment retrieval asks the reverse: find the span that matches a sentence [Gao et al. 2017; Anne Hendricks et al. 2017]. Both treat timing as part of the target and score it against human annotations. We ask instead whether timing survives when a frontier video LLM is asked for per-second captions with no localization objective. We score it against frame embeddings, so no human annotation is needed.

**Captions as a proxy for video.** A common recipe turns video into text and reasons over the text:
- Socratic Models compose pretrained models through language [Zeng et al. 2023].
- LLoVi captions short clips and lets an LLM answer long-video questions from the captions alone [Zhang et al. 2024].
- Large caption corpora are built the same way, to train video models: Panda-70M [Chen et al. 2024b] and ShareGPT4Video [Chen et al. 2024a]. The ShareGPT4Video authors note that naive multi-frame captioning gives temporally confused descriptions, and they design around it.

Our results put a number on what such pipelines lose: topic survives, the second-to-second timeline mostly does not.

**Temporal failures of video LLMs.** Benchmarks show that video LLMs perceive temporal properties poorly:
- speed and direction [TempCompass; Liu et al. 2024];
- actions, temporal order and scene transitions [VidHalluc; Li et al. 2025].

Timestamp-aware architectures target localization directly [TimeChat; Ren et al. 2024]. We find a related failure in a frontier model's *output* format: per-second timestamps that are consistently 1–2 s early, which a QA benchmark would not expose.

**Information loss across modalities.** Li et al. [2025] show that the connector that projects visual features into an LLM's embedding space loses information that predicts downstream errors. We measure a coarser bottleneck, the caption itself, and compare it with the visual embedding it was produced from.

**Encoders.** Frames and text are compared in SigLIP 2 [Tschannen et al. 2025]; text-to-text comparisons use MPNet sentence embeddings [Song et al. 2020; Reimers and Gurevych 2019]. The clips come from WildQA [Castro et al. 2022]. The text arm is Llama-3.1-8B [Grattafiori et al. 2024].

### References

Verified 2026-10-08 against the venue pages (search results); entries marked † were cited from memory and should be checked before submission.

- † Anne Hendricks, L., Wang, O., Shechtman, E., Sivic, J., Darrell, T., Russell, B. 2017. Localizing Moments in Video with Natural Language. ICCV.
- Castro, S., Deng, N., Huang, P., Burzo, M., Mihalcea, R. 2022. In-the-Wild Video Question Answering. COLING, 5613–5635.
- Chen, L., Wei, X., Li, J., et al. 2024a. ShareGPT4Video: Improving Video Understanding and Generation with Better Captions. NeurIPS Datasets and Benchmarks.
- Chen, T.-S., et al. 2024b. Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. CVPR, 13320–13331.
- † Gao, J., Sun, C., Yang, Z., Nevatia, R. 2017. TALL: Temporal Activity Localization via Language Query. ICCV.
- † Grattafiori, A., et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783.
- † Krishna, R., Hata, K., Ren, F., Fei-Fei, L., Niebles, J. C. 2017. Dense-Captioning Events in Videos. ICCV.
- Li, C., Im, E. W., Fazli, P. 2025. VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding. CVPR, 13723–13733.
- Li, W., Tang, R., Li, C., Zhang, C., Vulić, I., Søgaard, A. 2025. Lost in Embeddings: Information Loss in Vision-Language Models. Findings of EMNLP. (From `sources.md`.)
- Liu, Y., Li, S., Liu, Y., Wang, Y., Ren, S., Li, L., Chen, S., Sun, X., Hou, L. 2024. TempCompass: Do Video LLMs Really Understand Videos? Findings of ACL, 8731–8772.
- † Reimers, N., Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP.
- Ren, S., Yao, L., Li, S., Sun, X., Hou, L. 2024. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding. CVPR, 14313–14323.
- † Song, K., Tan, X., Qin, T., Lu, J., Liu, T.-Y. 2020. MPNet: Masked and Permuted Pre-training for Language Understanding. NeurIPS.
- Tschannen, M., et al. 2025. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv:2502.14786.
- Zeng, A., et al. 2023. Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language. ICLR.
- Zhang, C., Lu, T., et al. 2024. A Simple LLM Framework for Long-Range Video Question-Answering. EMNLP, 21715–21737.

## 6. Discussion and limitations

- **One captioner, one prompt.** The lag may come from whole-video captioning with timestamps. Control: re-caption a few videos per second (TODO, API).
- **SigLIP 2 as the reference for "what is visible".** It is an embedding, not ground truth; Q4 covers what it can't.
- **8B text model.** A stronger model is a scaling point, not a fix: the cap comes from what the captions contain.

## Figures

Made by `scripts/make_paper_figures.py` → `docs/paper/figures/fig{1..4}_*.{pdf,png}`. All CIs are 95% channel bootstrap. Each arm keeps one color across figures.

1. `fig1_lag`:
   - (a) the caption-to-frame lag profile (§3.1, §3.2);
   - (b) the near-pool score against caption shift (§3.5).
2. `fig2_topic_timing`: calibrated c by distractor pool, frame space (§3.5). Candidate main figure.
3. `fig3_forced_choice` (§4.2):
   - (a) the copy baselines by width (W ∈ {1, 2, 4, 8} in the item set);
   - (b) Llama against both copies on the 180 items Llama scored.
4. `fig4_grounding`: per-video grounding (Q1 MRR) against caption copy's rank deficit to frame copy, ρ = −0.51 [−0.60, −0.42], 323 videos with outcomes (§4.3).

A 4-page paper probably fits two of these. Fig 2 and Fig 1 carry the thesis; Fig 3 and Fig 4 could become a table and a sentence.
