# What Dense Video Captions Keep and Lose: Topic Without Timing

Working draft, started 2026-10-08. **Target: 8 pages** (decided 2026-10-08). The page budget is at the end of this file. Supersedes `draft.md` (the reconstruction-first thesis, whose H1/H2 our own tests rejected). Numbers come from the scripts named in each section; anything not yet verified is marked **TODO**.

**Thesis.** Dense per-second captions written by a frontier video LLM share a video's *topic* with its frame embeddings, but barely track *what changes second to second*, and they are systematically 1–2 s early, in every channel. That gap caps any text-only method that tries to recover missing seconds: an 8B LLM in-filling captions loses to simply copying the nearest caption, and both lose badly to copying the nearest frame.

## Abstract

Dense per-second captions from video LLMs are increasingly used as a stand-in for the video itself: for retrieval, for question answering, and as training data. We test how much of a video such captions actually carry. We caption 335 one-minute clips from 40 YouTube channels once per second with Gemini 3 Flash, then compare each caption with SigLIP 2 embeddings of the matching frames.

The captions keep a video's topic but not its timeline:
- Against frames from other videos, a caption identifies its own frame well (calibrated AUC 0.80; a copy of a neighboring frame reaches 0.94).
- Against frames from the surrounding ten seconds, it falls to 0.18, no better than the caption of a neighboring second, while frames keep 0.52.
- Captions run 1–2 s ahead of the frames in every channel. Correcting this lead recovers only a third of the gap.
- Caption change and visual change are uncorrelated.

Per-frame captions from a much smaller model, passed through the same text encoder, reach 0.47 on nearby frames and show no lead, so the lost timing belongs to the captions, not to the encoder.

These properties cap text-only recovery of missing seconds. An 8B LLM that in-fills masked captions loses to copying the nearest caption, and both lose badly to copying the nearest frame, on per-second retrieval and on a forced-choice test (35%, 52% and 85% accuracy; chance 25%). Stronger readers of the same captions recognize the missing content far more often: on 160 items, accuracy rises from 51% (Claude Haiku) to 77% (Sonnet) and 85% (Opus), against 46% for caption copy and 88% for frame copy. The captions' topic and event order suffice to recognize the gap, but not to regenerate its seconds. Videos with better-grounded captions show smaller caption-to-frame gaps.

## 1. Introduction

Dense captions are increasingly used as a stand-in for video. Pipelines caption short clips and let a language model answer questions from the text alone [Zhang et al. 2024]; large corpora of captioned clips train the next generation of video models [Chen et al. 2024a; Chen et al. 2024b]; and retrieval systems index the captions instead of the pixels. All of these rest on an assumption that is rarely tested: that a sequence of timestamped captions is a lossy but faithful, time-aligned transcript of the frames. If it is, a caption track can replace the video for any purpose that does not need pixel detail. If it is not, the loss is invisible to the pipelines that depend on it, because they never compare the text with the frames.

We test the assumption directly. We take 335 one-minute clips from 40 YouTube channels (WildQA; Castro et al. 2022), ask a frontier video LLM (Gemini 3 Flash) for one caption per second, and compare the caption track with SigLIP 2 embeddings of the same frames [Tschannen et al. 2025]. The comparison needs no human annotation. It asks whether caption t picks out frame t, whether the two tracks vary and change together, and whether a caption tells its own second apart from other videos (topic) and from nearby seconds (timing).

The answer is that captions keep the topic and lose the timeline. Against frames from other videos, a caption identifies its own frame almost as well as a copy of the neighboring frame does. Against frames from the surrounding ten seconds, it is no better than the caption of a neighboring second. Captions also run one to two seconds ahead of the frames they describe, in every one of the 40 channels, and correcting this lead recovers only a third of the timing gap. A control with per-frame captions from a much smaller model shows that the text encoder can carry second-level timing, so the loss lies in the captions themselves.

We then ask what this loss costs a text-only attempt to recover masked seconds. With a gap of W seconds masked, an 8B LLM that in-fills the missing captions from their context loses to copying the nearest caption, which in turn loses badly to copying the nearest frame, on per-second retrieval and on a forced-choice test. Recognition scales with the reader: on the forced-choice test, the largest frontier model we tried matches the frame-copy baseline from captions alone. Generation does not, on the evidence so far, which is what a loss of timing rather than of content predicts. Across videos, the better a video's captions are grounded, the closer caption-based gap filling gets to frame-based gap filling.

Our contributions are:
1. An annotation-free protocol that scores a caption track against frame embeddings, separating topic from timing by the choice of distractor pool.
2. The finding that frontier dense captions are grounded at the level of the video but barely at the level of the second (MRR 0.142 against 0.079 chance), that their second-to-second variation shares almost nothing with the frames' (R² ≈ 0.02), and that they lead the frames by 1–2 s in every channel.
3. A time-aligned ceiling showing that the loss is in the captions, not in the text-to-image link.
4. A model-scale comparison for masked seconds. An 8B in-filler loses to copy everywhere. Recognizing the missing content scales with the reader, up to the frame-copy baseline (51%, 77% and 85% for Claude Haiku, Sonnet and Opus), while generating it does not.

## 2. Data

<!-- Captions: config/gen_captions/wild{4,5}.yaml, prompt prompts/yt_video_processing/one_minute.txt. Frame timing check: scripts/check_frame_timestamps.py. Note: draft.md wrongly said Gemini 1.5 Flash on isolated 1-s clips. -->

**Videos.** We use the dev (100 clips) and test (235 clips) splits of WildQA [Castro et al. 2022], outdoor, in-the-wild YouTube videos from five domains (agriculture, geography, human survival, natural disasters and military), and keep the first 60 s of each. The 335 clips come from 40 channels. Clips from one channel share a filming style, a narrator and often a setting, so they are not independent: every confidence interval in this paper comes from a bootstrap that resamples channels, unless stated otherwise.

**Captions.** Each clip is captioned by `gemini-3-flash-preview` in a single call that sees the whole 60 s at one frame per second (temperature 0.8, thinking enabled). The prompt asks for 60 self-contained captions, one per one-second interval, each describing "what is visible during that 1-second interval". Because the model sees the whole minute at once, nothing forces caption t to describe second t alone. We study exactly this common setup, rather than captioning each second in isolation.

**Frames.** We take one frame per second (row t is the frame at t s, which we checked against exact-time extraction: offset 0 in 16 of 16 checks) and embed it with SigLIP 2 base (`vit_base_patch16_siglip_224.v2_webli`, 768 dimensions) [Tschannen et al. 2025]. Captions are mapped into the same space with the matching SigLIP 2 text tower (`google/siglip2-base-patch16-224`, lowercased input). Using the matching text tower matters: with the SigLIP 1 text tower, the similarity between captions and these frames is about zero.

**Text space.** Comparisons between pieces of text (for example, a reconstructed caption against the true one) use MPNet sentence embeddings (`all-mpnet-base-v2`) [Song et al. 2020; Reimers and Gurevych 2019].

**Metrics.** We report two families of scores, both computed per video and then averaged.
- *Rank*: the rank of the true second among the video's 60, in the arm's own space. Chance is 30.5, and lower is better; MRR is the mean reciprocal rank.
- *Calibrated AUC*: c = 2·AUC − 1 of the true target against a pool of distractors. A c of 0 is chance and 1 is perfect, and the choice of pool sets what is being tested (§3.5).

We prefer ranks and AUC to raw cosine similarity, because cosine similarity is strongly confounded by how repetitive a video's captions are.

## 3. Captions vs. frames, without masking

<!-- Source for 3.1–3.4: scripts/caption_vs_siglip_audit.py → results/caption_audit/; 3.2 also scripts/caption_lag_robustness.py → results/caption_audit/lag_*.csv. CIs are 95% channel bootstrap unless stated. -->

Before masking anything, we ask how closely the caption track follows the frame track it was written from. Each video gives us two aligned sequences of 60 vectors: the captions, embedded with the SigLIP 2 text tower (and with MPNet for text-to-text comparisons), and the frames, embedded with the SigLIP 2 image tower. We examine four properties in turn: whether a caption picks out its own second (Q1), whether the two sequences vary together within a video (Q2), whether they change at the same moments (Q3), and how these properties trade off between topic and timing.

### 3.1 Grounding (Q1)

For each second t, we rank the video's 60 frames by cosine similarity to caption t and record the rank of frame t. Averaged over videos, the mean reciprocal rank is 0.142. Uniform chance is 0.079, and an empirical floor, obtained by circularly shifting the frames by a random offset of at least 10 s, gives 0.063. The most similar frame lies within ±2 s of t for 22% of captions, against 8% by chance. Captions are therefore grounded in their own video's frames, but only loosely in their own second.

Part of that looseness is a systematic offset. To measure it, we z-score each caption's similarities across the 60 frames and average z(caption t, frame t+k) over t for lags k ∈ [−10, 10]. The profile peaks at k = +2, not k = 0: z is 0.58 at 0, 0.72 at +2 and 0.35 at −2 (Fig. 1a). Caption t describes what is on screen one to two seconds *later*. Per video, the best lag is +1 or +2 in 222 of 335 videos and 0 in only 12.

### 3.2 Robustness of the lag

The profile is flat between +1 and +2 (z = 0.72 at both, 0.65 at +3), so the lead is about 1.5 s, and the advantage of +2 over 0 is z = 0.141 [0.122, 0.162]. It is not driven by a few sources. The peak falls at +1 or +2 in 39 of the 40 channels and at +3 in the last one; no channel peaks at 0 or earlier, and z(+2) > z(0) in 39 of 40. The profile is the same in the dev and test splits and in every tertile of caption length.

The lead is close to a constant offset. The peak stays at +2 in each third of the clip, although the advantage of +2 over 0 shrinks slightly toward the end: 0.164 [0.127, 0.203] in the first third, 0.161 [0.130, 0.194] in the middle and 0.117 [0.089, 0.146] in the last. Fitting the per-second best lag against time within each video gives a slope of −0.33 s per minute [−0.61, −0.04]. A constant lead suggests a convention of the captioner, which sees the whole minute at once and appears to describe each second by what is about to happen, rather than drift that accumulates over the clip.

If the lead were the whole story, shifting the captions by two seconds would repair them. We test this on gap fills, anticipating §4: for each masked slot t we score the fill against frame t+s and report the rank of that frame among the video's 60 (chance 30.5; lower is better), over 2,535 gaps with W = 1–16.

  | Fill | s = 0 | s = +1 | s = +2 | gain at +2 |
  |---|---|---|---|---|
  | Oracle (true caption) | 22.9 | 20.7 | 20.2 | +2.67 [+2.10, +3.23] |
  | Caption copy | 23.9 | 23.7 | 23.5 | +0.42 [−0.00, +0.82] |
  | Llama-3.1-8B | 28.5 | 28.6 | 28.8 | −0.27 [−0.68, +0.10] |

Shifting improves the true captions by 2.7 ranks [2.1, 3.2] but does nothing for either gap fill. The lead lives in the captions' timing, and a fill that carries no second-level timing has nothing to re-align. The table also previews §4: in frame space, copying the caption at the gap boundary is within one rank of the true caption (23.9 against 22.9), and Llama is close to chance (28.5 against 30.5).

### 3.3 Shared second-to-second variation (Q2)

Grounding asks whether a caption is closer to its own frame than to other frames. A stronger requirement is that the caption track and the frame track vary *together*: when the scene changes, the caption embedding should move in a corresponding direction. We remove each video's mean from both sequences, so that only within-video variation remains, and fit ridge regressions (α = 10) from one modality to the other, with 5-fold cross-validation grouped by channel. Held-out R² is about 0.02 in every direction: MPNet to frames, SigLIP 2 text to frames, and both reverse maps. Once the video's overall topic is removed, almost none of the frames' second-to-second variation is linearly predictable from the captions, or the reverse.

### 3.4 Change timing (Q3)

Q2 is a linear test, and it could miss shared structure that is not linear. A coarser test is whether the two tracks *change* at the same moments. We define per-second change as the cosine distance between consecutive embeddings (MPNet for captions, SigLIP 2 for frames). The per-video Spearman correlation between the two change series is −0.04. Taking the top 10% of changes in each modality as events, a visual event has a caption event within ±1 s 29% of the time, against 26% when caption changes are shuffled within the video.

Correcting the lead helps only a little. Comparing caption change at t with frame change at t+s, coincidence rises to 33% at s = +1 or +2 (chance 27%), and the per-video Spearman stays near 0.01. Caption changes and visual changes are essentially unrelated, even after alignment.

### 3.5 Topic vs. timing, by distractor pool

<!-- Source: scripts/eval_shared_target.py (configs/eval_shared_target.yaml, re-run 2026-10-08 with SigLIP 2 text) → results/redesign_shared_target*.csv; lag-corrected near pool: caption_lag_robustness.py part 5. -->

Q1–Q3 mix two abilities: telling a video apart from other videos (topic) and telling a second apart from nearby seconds (timing). We separate them by changing where the distractors come from. An arm's prediction for a masked second is scored in frame space (SigLIP 2 text or image embedding against the true frame) by calibrated AUC, c = 2·AUC − 1, against distractor frames drawn from one of four pools: the same video within ±10 s, the same video farther away, other videos of the same channel, and other channels. A c of 0 is chance and 1 is perfect. We use mid-clip gaps (i = 29) with W = 3 and 6 over all 335 videos and average per video. All pre-registered sanity gates pass; for example, a random caption or frame scores about 0 in every pool.

| Arm (frame space) | same video, ±10 s | same video, far | other video, same channel | other channel |
|---|---|---|---|---|
| True caption | 0.18 | 0.33 | 0.72 | 0.80 |
| Caption copy (nearest boundary) | 0.19 | 0.35 | 0.72 | 0.81 |
| Llama-3.1-8B | 0.05 | 0.14 | 0.58 | 0.70 |
| Frame copy | 0.52 | 0.76 | 0.93 | 0.94 |
| Random caption / frame | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

Topic survives. Against frames from other channels, the true caption reaches c = 0.80, not far below a copy of the neighboring frame (0.94), and it loses little against other videos of the same channel (0.72). Timing does not survive. Against frames from the surrounding ten seconds, the true caption falls to 0.18 (Fig. 2). That is no better than the boundary caption copied into the gap (0.19): the caption written for second t says no more about which nearby frame is second t than a caption written for a neighboring second. Frame copy keeps 0.52 in the same pool. Moving from far to near distractors costs the true caption 0.16 [0.13, 0.19] and frame copy 0.24 [0.22, 0.26], but frame copy starts from a much higher level and stays far ahead.

The lead explains part of this. When slot t is scored against frame t+s, with the near pool rebuilt around the shifted gap, the true caption rises from 0.17 at s = 0 to 0.25 at s = +1 and 0.28 at s = +2, a gain of 0.11 [0.07, 0.15] (Fig. 1b). It then beats caption copy (0.20) but remains far below frame copy (0.52), while caption copy (+0.01) and Llama (−0.02) do not gain. The lead accounts for about a third of the caption-to-frame gap on nearby seconds. The rest is timing information that the captions do not carry.

Llama's in-filled captions fall below copy in every pool (0.05 near, 0.70 against other channels), so they lose topic as well as timing. All four pre-registered comparisons of Llama with caption copy and with the mean boundary caption, at W = 3 and 6, come out inferior (paired differences −0.19 to −0.23, Holm-adjusted p < 0.001). We return to this in §4.

### 3.6 Is it the captions or the encoder? A time-aligned ceiling

<!-- Source: scripts/bridge_ceiling.py (+ scripts/kaggle_bridge_ceiling.py for the GPU captioning) → results/bridge_ceiling/. -->

The near-pool result admits a second reading: perhaps the captions do carry timing, but SigLIP 2's text-to-image link cannot express it, because a single caption embedding is too coarse to separate frames a few seconds apart. To tell these readings apart, we need text that is time-aligned by construction and passes through the same link. We captioned every second of 30 videos (one per channel, 30 channels) frame by frame with Florence-2 base (`<DETAILED_CAPTION>`), embedded the captions with the same SigLIP 2 text tower, and repeated the near-pool and lag analyses. Here the near pool holds frames 2–10 s from the target, with t ∈ [10, 50) so that the pool is complete, and CIs are by video bootstrap. Gemini's near-pool score on this subset (0.16) is therefore close to, but not identical with, the 0.18 of §3.5.

| Captions | near c | other-video c | Q1 MRR | Lag-profile peak |
|---|---|---|---|---|
| Florence-2, per frame | **0.47** [0.39, 0.54] | 0.99 | **0.27** | **0** (28/30 videos) |
| Gemini, whole minute | 0.16 [0.12, 0.21] | 0.92 | 0.14 | +2 |
| Gemini, shifted +2 s | 0.21 [0.16, 0.27] | 0.93 | — | — |
| Frame copy (frame t−1) | 0.65 | 0.98 | — | — |

The text-to-image link can carry second-level information. Per-frame captions from a far smaller model reach c = 0.47 on nearby frames, beating Gemini's captions by 0.31 [0.24, 0.37], and still by 0.25 [0.19, 0.31] after Gemini's lead is corrected. Their retrieval MRR is nearly double Gemini's (0.27 against 0.14). The weak timing of §3.5 is therefore a property of the Gemini captions, not of the encoder.

The same experiment validates the lag measurement. Per-frame captions peak at lag 0 in 28 of 30 videos, so the procedure of §3.1 does not manufacture a lead, and Gemini's +1 to +2 s offset is real. Aligned text still loses something: per-frame captions stay below frame copy (0.47 against 0.65), which bounds what any caption can carry through this encoder.

One caveat limits the comparison. Florence-2 is a different captioner, and its captions are more literal and visual than Gemini's, often near-duplicates across adjacent seconds. The experiment therefore isolates "aligned, frame-specific text" rather than "the same captioner, aligned". Re-captioning with a frontier model one second at a time is the remaining control (§6).

### 3.7 Caption correctness (Q4)

**TODO** (manual pass, or an Opus judge through the blind runner's image mode with planted wrong captions as a check): about 50 sampled seconds, labeled correct / wrong time / hallucinated (e.g. the "runner" camera-holder). Until then, §3 measures agreement with the frame embedding, not correctness, and §6 says so.

## 4. Consequence: reconstructing masked seconds from text

§3 shows what the captions lack. Here we ask what that costs a method that has only the captions. We mask a contiguous gap of W ∈ {1, 2, 3, 4, 6, 8, 12, 16} seconds in the middle of each clip, starting at second 29, and ask each arm to fill it.
- **Text arm:** Llama-3.1-8B [Grattafiori et al. 2024] sees the surrounding captions with the gap marked and writes one caption per missing second (prompt in the appendix; temperature 0.6, repetition penalty 1.05).
- **Copy baselines:** *caption copy* fills each missing second with the nearest unmasked caption, and *caption mean* with the mean embedding of the two captions at the gap's edges.
- **Frame baselines:** *frame copy* and *frame mean* do the same with frame embeddings.

The frame baselines are not a competing method, since they see frames the text arm never sees. They show how much a video's own visual continuity gives away for free.

<!-- Text arm prompt: prompts/dense_window/. -->

### 4.1 Per-second retrieval

<!-- Source: results/unified_benchmark_master.csv, i = 29, 335 videos. -->

Each arm is scored in its own modality: text arms in MPNet against the true caption, frame arms in SigLIP 2 against the true frame. The score is the rank of the true second's embedding among the video's 60 (chance 30.5; lower is better).

| Arm | W=1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 23.0 | 26.3 | 27.5 | 27.4 | 27.6 | 28.0 | 29.5 | 30.2 | 27.4 |
| Caption copy | 23.4 | 23.1 | 22.3 | 22.7 | 22.7 | 23.5 | 24.4 | 25.6 | 23.5 |
| Caption mean | 22.7 | 22.6 | 22.8 | 23.0 | 24.1 | 25.4 | 27.1 | 28.6 | 24.5 |
| Frame copy | 7.4 | 8.2 | 10.3 | 11.1 | 13.7 | 14.9 | 17.6 | 19.5 | 12.8 |
| Frame mean | 6.9 | 8.9 | 10.8 | 12.4 | 15.5 | 17.6 | 21.3 | 23.9 | 14.7 |

Llama trails caption copy by 3.9 ranks [3.1, 4.8], averaged per video over widths, and beats it in only 32% of videos. The one exception is W = 1, where a single missing second sits between two known ones; there Llama roughly ties copy (23.0 against 23.4; MRR 0.119 against 0.107). As the gap widens, Llama drifts toward chance, while caption copy degrades slowly. Caption copy in turn trails frame copy by 10.6 ranks [9.3, 12.0] and beats it in 9% of videos.

Per-second scoring is also harsher on the LLM than it needs to be. Llama's ordering of seconds *within* a gap is at chance (calibrated c ≈ 0), so even a fill with the right content in the wrong order is scored as wrong. The forced-choice test below scores whole gaps instead.

### 4.2 Forced choice

<!-- Source: scripts/forced_choice_gap.py; Llama scoring on Kaggle via scripts/kaggle_forced_choice.py; matched distractors: scripts/forced_choice_matched_feasibility.py. -->

Each item masks the gap together with three distractor spans of the same length elsewhere in the clip. A method sees the masked context and the four candidate spans (the true caption sequences of the four masked spans) and must pick the one that belongs in the gap (chance 25%). Copy baselines pick the candidate most similar to the gap's boundary captions (MPNet) or boundary frames (SigLIP 2). Their *assignment-aware* variants subtract from each candidate's score its best similarity to the boundaries of any other masked span, so that a candidate which fits another slot better is penalized. Llama scores each candidate by pointwise mutual information, the log-probability of the candidate in the gap given the context minus its log-probability without context, a choice we fixed before running.

| Method | Accuracy (180 items scored by Llama) |
|---|---|
| Llama-3.1-8B, PMI (pre-registered) | 35% (CI 28–42) |
| Caption copy | 52% |
| Caption copy, assignment-aware | 59% |
| Frame copy | 85% |

Llama is scored on 180 of the 1,325 items, a GPU budget limit. On those items it reaches 35%, above chance but well below caption copy (52%), which is in turn far below frame copy (85%). Over all 1,325 items the baselines are similar: caption copy 47% (54% assignment-aware) and frame copy 86% (94%). Llama's errors are independent of caption copy's, yet combining the two with a weight fit by leaving one channel out gives no gain (51%).

A natural objection is that frame copy wins only because adjacent frames look alike. Distractors that match the gap's continuity would remove that advantage, but in these videos such distractors are too rare to build a fair test: with both modalities matched, frame copy still reaches 65% on the 286 items that remain.

**A frontier text model.**

<!-- Source: scripts/blind_llm_runner.py run --task choice_nohint --model sonnet --pool all --shuffle --limit 160; runner protocol in docs/handover.md 2026-10-08 §D. -->

Is the 8B model or the captions the bottleneck? We gave the same items to Claude Sonnet 5.5, which sees only text. Each item ran in a fresh, isolated session with no tools, the same masked context Llama saw and the four candidates in random order; the model answered with a letter. The prompt did not say that the other three candidates belong to the other masked spans, which would turn the task into a jigsaw. We used 160 items sampled across 36 channels, with channel-bootstrap CIs.

| Method | Accuracy |
|---|---|
| **Sonnet 5.5** | **76.9% [69.6, 83.0]** |
| Caption copy | 45.6% |
| Frame copy | 87.5% |

On paired items, Sonnet beats caption copy by 31 points [21, 42] and trails frame copy by 11 [3, 18], at every width (76–78% at W = 1, 2, 4 and 8). The 8B result therefore does not extend to recognition. A strong reader extracts much of what tells a gap apart from its distractors, which caption copy cannot, but the frames still carry more.

Recognition grows steeply with model scale. On the same 160 items and with the same protocol, the smaller Claude Haiku 5.5 scores 51.2% [44.3, 58.2], level with caption copy (+5.6 [−4.4, +16.6]). Claude Opus 5.5 scores 85.0% [78.7, 90.7], 8.1 points [4.1, 12.4] above Sonnet and statistically level with frame copy (−2.5 [−10.1, +5.6]). Its accuracy is flat across widths (83–88%). Frame copy is a crude reader of the frames, and its assignment-aware variant reaches 95.6%, so this is not parity with what the frames contain. It does show that, from captions alone, a frontier reader can tell which span belongs in the gap about as well as the nearest frame can. Topic and the order of events, which §3 found the captions keep, are enough for that.

> **Note for the authors (2026-10-09):** this weakens "captions cap text-only recovery" for *recognition*. The cap now holds for second-level timing (§3) and for *generation*: on 50 videos Sonnet's reconstructions are no better than caption copy, and worse in frame space (below). The abstract and §6 are worded accordingly, but the framing is a decision to make. Opus reconstruction is the remaining check that generation does not also scale.

<!-- Haiku/Sonnet/Opus comparison: scripts/blind_choice_compare.py haiku sonnet [opus]. -->

| Reader (captions only) | Accuracy, 160 items |
|---|---|
| Llama-3.1-8B, PMI | 35% (different 180 items; see above) |
| Claude Haiku 5.5 | 51.2% [44.3, 58.2] |
| Claude Sonnet 5.5 | 76.9% [70.1, 82.9] |
| Claude Opus 5.5 | 85.0% [78.7, 90.7] |
| *Caption copy* | 45.6% [37.6, 53.6] |
| *Frame copy* | 87.5% [81.8, 92.5] | Two caveats apply. Sonnet makes a direct choice while Llama is scored by PMI, so the two numbers come from different protocols. And the frontier model's training data are unknown, although it is unlikely to have seen these particular captions, which we generated.

Recognition is not generation. We asked Sonnet to write the missing captions with Llama's exact prompt, for 50 videos from 28 channels × W ∈ {1, 4, 8, 16}, and scored the result as in §4.1. Over the 191 gaps that all three arms filled (ranks averaged per video, channel-bootstrap CIs):
- Sonnet beats Llama in MPNet text rank (−4.8 [−6.7, −2.9]) and in SigLIP 2 frame rank (−3.1 [−5.4, −1.0]).
- It only draws level with caption copy in text rank (22.3 against 23.6; −1.4 [−3.0, +0.5]).
- It is *worse* than caption copy in frame rank (25.8 against 22.5; +3.3 [+1.5, +5.1]).

The same model that picks the right span 77% of the time cannot write the span better than the boundary caption does. This is what §3 predicts. The captions carry the topic and the order of events, which is enough to recognize the right span, but they carry little second-level detail to generate from. **TODO:** Opus reconstruction on the same 50 videos (about 200 calls, roughly 30 points of the 5-hour quota).

<!-- Source: blind_llm_runner.py run --task recon --model sonnet --limit 200 (video-major, 50 videos × 4 widths); score --model sonnet → results/blind_llm/recon__sonnet__scores.csv; paired CIs computed ad hoc (2026-10-09). 1 of 200 items invalid. -->

| Arm (50 videos, mean rank; chance 30.5) | MPNet text rank | SigLIP 2 frame rank |
|---|---|---|
| Llama-3.1-8B | 27.1 | 28.9 |
| Claude Sonnet 5.5 | 22.3 | 25.8 |
| Caption copy | 23.6 | 22.5 |

### 4.3 Grounding predicts the gaps

If the loss in §3 causes the reconstruction cap, videos whose captions are better grounded should show smaller caption-to-frame gaps. For each video we correlate the agreement measures of §3 with three outcomes (Spearman, channel-bootstrap CIs). The partial correlations control for how fast the video's frames and captions change, since fast-changing videos are hard for every method.

| Agreement metric | caption copy's rank deficit to frame copy | frame copy's forced-choice advantage | Llama's deficit to caption copy |
|---|---|---|---|
| Q1 MRR | −0.51 (partial −0.39) | −0.30 (partial −0.18) | +0.33 (partial +0.26) |
| Q2 MPNet→frames | −0.48 (partial −0.36) | −0.27 (partial −0.16) | +0.32 (partial +0.24) |
| Q3 Spearman | ≈ 0 | ≈ 0 | ≈ 0 |

Where captions are better grounded, caption copy gets closer to frame copy (ρ = −0.51 [−0.60, −0.42] for Q1, Fig. 4), and the relation survives the control for change rates (partial −0.39). Grounding also predicts Llama falling further behind caption copy (partial +0.26). Our reading is that well-grounded captions make the boundary caption a good proxy for the gap, which raises the bar Llama has to clear. Change timing (Q3) predicts none of the outcomes, which is consistent with its near-zero signal in §3.4.

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

Dense captions from a frontier video LLM are a good record of what a video is about and a poor record of when things happen in it. Topic survives almost intact. Second-level timing is mostly lost, and what remains is shifted one to two seconds early. For pipelines that use captions as a stand-in for video, this means that questions about content are safe and questions about timing, order at the scale of seconds, or the moment of a change are not, and that the failure will not show up in the text itself. For reconstruction, the picture splits by task. Recognizing which content belongs in a gap scales with the reader, and the largest model we tried matches the frame-copy baseline from captions alone. Generating the missing seconds does not scale, on the evidence so far. The 8B model loses to copying a neighbor, and a frontier model draws level with it in text space and falls behind it in frame space. What the captions lose is the timeline, and no reader can put back a timeline the text never recorded.

**Limitations.**
- **One captioner, one prompt.** All captions come from a single model, prompted once for the whole minute. The lead and the lost timing may be properties of this setup (whole-clip captioning with timestamps) rather than of video LLMs in general. The Florence-2 ceiling (§3.6) shows that aligned text can carry timing, but with a different captioner. Re-captioning a sample one second at a time with a frontier model is the direct control (**TODO**).
- **An embedding as the reference.** We use SigLIP 2 as the reference for what is visible. It is an embedding, not ground truth, and agreement with it is not correctness. The caption correctness audit (§3.7, **TODO**) covers part of what it cannot.
- **Short clips from one dataset.** The clips are 60 s from 40 YouTube channels in WildQA. Longer videos, other domains and scripted content may behave differently.
- **Frontier text models through a subscription.** The Claude runs used the Claude Code CLI, with tools, persistence and prompt caching disabled and the thinking effort pinned. The models are closed, so their training data and future versions are outside our control. Every prompt and raw response is saved, for release with the paper.
- **Different scoring protocols.** Llama's forced-choice scores use PMI and Claude's use direct choice. Running Llama with direct choice would make the comparison exact.
- **Channels, not videos, are the unit.** With 40 channels, CIs from the channel bootstrap are wide for subgroup analyses, and we report no per-domain results.

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

With 8 pages, all four fit. Updates to make:
- Fig 2: add the Florence-2 per-frame ceiling as a reference.
- Fig 3: add Sonnet (and later Opus and Haiku) as model-scale points.

## Page budget (8 pages + references)

As of 2026-10-09 the whole draft is prose: about 5,200 words (abstract to discussion, without references or HTML comments) plus 7 tables. 8 pages holds roughly 5,500–6,500 words with 4 figures and 2–3 tables, so the *tables*, not the words, are what's over budget: move 3–4 of them (re-alignment, bridge ceiling, grounding correlations) to the appendix or into figures. Section numbers below are the planned order (Related work as §2); the draft file still has Related work as §5.

| Section | Pages | Content | Status |
|---|---|---|---|
| Abstract + 1 Introduction | 1.0 | Motivation (captions as a stand-in for video), the questions, the findings list, contributions | Prose done 2026-10-09; abstract updated with the Florence-2 ceiling and Sonnet |
| 2 Related work | 0.6 | As drafted | Drafted; check the † references |
| 3 Data and setup | 0.8 | Clips, captioner, encoders, metrics (calibrated c, ranks), channel-clustered statistics | Prose done 2026-10-09 |
| 4 Captions vs. frames | 2.3 | Grounding and lag (Fig 1), lag robustness, Q2/Q3, topic vs. timing (Fig 2), bridge ceiling (table) | Prose done 2026-10-09 (draft §3, ~2,000 words + 3 tables ≈ 3 pages: over budget). Trim candidates: move the re-alignment table to §5 or the appendix, shorten Q2/Q3 to one paragraph |
| 5 Consequences for reconstruction | 2.0 | Per-second table, forced choice across model scale (Fig 3), grounding vs. deficit (Fig 4), recognize vs. generate | Prose done 2026-10-09; still needs the Haiku/Opus scale points and more Sonnet reconstruction |
| 6 Discussion and limitations | 0.8 | One captioner, encoder-based reference, 60-s clips, subscription models, PMI vs. direct choice | Prose done 2026-10-09 |
| Appendix (no page limit at most venues) | — | Prompts, isolation protocol of the blind runner, per-channel lag table, extra widths | To assemble |

**Experiments that would strengthen the 8-page version** (cheapest first):
1. Haiku on the same 160 forced-choice items. Cheap; it gives a three-point model-scale curve with Sonnet and Opus.
2. Opus on the same 160 items, after a 20-call cost probe.
3. Sonnet reconstruction on about 50 videos (200 calls), to back "recognizes but can't generate".
4. Q4 correctness audit, about 50 seconds: manual, or an Opus judge with planted wrong captions.
5. Gemini per-second re-captioning of a few videos (the same-captioner control for §3.6). This needs an image mode in the runner, or Gemini CLI.
