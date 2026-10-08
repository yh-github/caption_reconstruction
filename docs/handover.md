# Handover

The newest session is at the top. Older sessions stay below because other docs link to their sections (for example "§1" means the 2026-10-07 section 1).

# 2026-10-08 → next session

**Goal:** firm up the caption lag, settle the shared-target evaluation, and start the paper in a new file.

## A. Housekeeping

- Committed the 2026-10-07 work in three commits (SigLIP 2 fixes, new scripts, docs). `.gitignore` now ignores `.test_lock*`.
- **New paper file: `docs/paper/captions_vs_frames.md`.** It is the working draft around the captions-vs-frames thesis. `draft.md` stays untouched as the record of the old thesis.
- **Caption provenance corrected.** `draft.md` §2.1 says Gemini 1.5 Flash on isolated 1-s clips. That is wrong. The configs (`config/gen_captions/wild{4,5}.yaml`) show `gemini-3-flash-preview`: one call per 60-s video, 1 fps, temperature 0.8, thinking on, prompt `prompts/yt_video_processing/one_minute.txt`. Seeing the whole clip in one call is a plausible cause of the lead.

## B. Caption lag robustness: the lead is real and universal

Source: `scripts/caption_lag_robustness.py` → `results/caption_audit/lag_{profiles,shift_ranks,q3}.csv`. Full numbers are in `captions_vs_frames.md` §3.2.

- **Size:** the lag profile is flat between +1 and +2, so the lead is about 1.5 s.
- **Universal:** the peak is at +1 or +2 in 39 of 40 channels (+3 in one); no channel peaks at 0 or earlier. The profile is the same for wild4 and wild5 and in every caption-length tertile.
- **Mostly a constant offset.** The peak is +2 in each third of the clip. The lead shrinks somewhat late in the clip (slope −0.33 s per minute, CI [−0.61, −0.04]).
- **Re-alignment** (fill at slot t scored against frame t+s, SigLIP 2, rank among 60):

  | Fill | s = 0 | s = +2 |
  |---|---|---|
  | Oracle caption | 22.9 | 20.2 |
  | Caption copy | 23.9 | 23.5 (CI touches 0) |
  | Llama | 28.5 | 28.8 (about chance) |

  Gap fills carry no second-level timing to re-align. In frame space, caption copy is within one rank of the oracle.
- **Q3 with the lag corrected:** event coincidence rises to 33% against 27% chance; Spearman stays ≈ 0.

## C. Shared-target evaluation: re-run with SigLIP 2 text

- **Old outputs moved.** The outputs containing frame-home text-arm numbers (frame-home `T_Oracle` c ≈ 0) were moved to `results/invalid_siglip1_text/`, with a README:
  - the whole old `redesign_*` run;
  - `wild4_sweep_shared_target.csv` and `wild4_sweep_summary.csv`.
- **Outputs left in place.** Caption-home-only and visual-only outputs stay where they were, and so do all figures in `docs/paper/figures/`. None of the plots uses frame-home text arms.
- **Config fixes.** `configs/eval_shared_target.yaml` now uses `google/siglip2-base-patch16-224`, and `analyze_wild4_sweep.py` takes the model from that config instead of the SigLIP 1 default.
- **Results** (335 videos, mid gaps, W = 3 and 6; full table in `captions_vs_frames.md` §3.5). All gates pass; frame-home `T_Oracle` is now 0.32, where it was about 0. In frame space, by distractor pool:

  | Arm | ±10 s, same video | other channel |
  |---|---|---|
  | True caption | 0.18 | 0.80 |
  | Caption copy | 0.19 | 0.81 |
  | Llama | 0.05 | 0.70 |
  | Frame copy | 0.52 | 0.94 |

  - **Topic is kept, timing is lost.** Against nearby seconds, the true caption is no better than the boundary caption copied into the gap.
  - C1–C4: Llama is INFERIOR to both caption baselines at W = 3 and 6 (Holm p < 0.001).
  - The "headroom" sweep printout is uninformative (every CI spans [0, 1.9]); ignore it.
- **Paper value.** This table is the cleanest single statement of the thesis and a candidate main table.

## D. Next

1. Done: the near pool with lag-corrected captions (`caption_lag_robustness.py` part 5). The true caption goes from 0.17 to 0.28 at +2 s, against 0.20 for caption copy and 0.52 for frame copy, so the lead explains about a third of the near-second gap.
2. §4.1 of `captions_vs_frames.md` is filled. Llama trails caption copy by 3.9 ranks [3.1, 4.8], and only at W = 1 does it tie. Caption copy trails frame copy by 10.6 ranks [9.3, 12.0].
3. Still to write: related work and the abstract. Then make Figures 1 to 3 (and maybe the §3.5 table as a figure).
4. Doc reconciliation (TODO §1): now mostly moot, since `draft.md` is frozen. Just make sure the new draft uses the right counts.
5. Q4 caption correctness and the re-captioning control: waiting on the API budget decision.

---

# 2026-10-07 → next session

**Today's goal:** a GO/NO-GO on reconstruction as the paper's core, and a test of the "captions vs. SigLIP" direction against criteria set before running it.

**Outcome:**
- **Reconstruction as the core: NO-GO** at 8B.
- **Captions vs. frames: GO.** It meets both criteria: a structured difference between the two views, and a link to the reconstruction outcomes that survives controls.

## 1. Pipeline bug: the frame embeddings are SigLIP 2, not SigLIP 1

- timm's untagged `vit_base_patch16_siglip_224` resolves to `v2_webli`, so `local/wild_videos_embs_siglip/` holds **SigLIP 2** embeddings. `metadata.yaml` said `google/siglip-base-patch16-224` (SigLIP 1); it is now corrected.
- `SiglipTextEmbedder()` defaults to SigLIP 1 text, which is **not in the frames' space**:
  - the stored rows have cosine ≈ 0 with HF SigLIP 1 image features;
  - timm `v2_webli` reproduces the stored rows exactly (cosine 1.0);
  - SigLIP 2 text (`google/siglip2-base-patch16-224`, lowercased) aligns with them: it retrieves the right video among 120 with MRR 0.76 (chance 0.008).
- **Still valid**, because they compare frames with frames or text with text:
  - the visual copy and mean-closest baselines;
  - forced-choice `vis_copy`;
  - all MPNet results;
  - text-only comparisons in SigLIP-text space.
- **Invalid if run with SigLIP 1 text**, because they compare text with frames:
  - the shared-target evaluation (`scripts/eval_shared_target.py`);
  - `run_downstream_retrieval.py --embedder siglip` `vis_*`;
  - any `evaluation.py` path that scores SigLIP text against video embeddings.

  E1 and E1 symmetric were re-run today with SigLIP 2 text (section 4).
- **Code fixes (uncommitted at time of writing):**
  - `SiglipTextEmbedder` supports SigLIP 2;
  - `VideoEmbedder` is pinned to `.v2_webli`;
  - on CPU, `SiglipTextEmbedder` used to load float16 weights, because a `torch.device` never equals `"cpu"`. That was about 50× slower but numerically fine, and is fixed;
  - `set_num_interop_threads` is now guarded;
  - CLAUDE.md and `experiment_protocol_current.md` carry the correction.

## 2. Decision 1: reconstruction as the paper's core: NO-GO

**Forced choice** (Kaggle, 180 of 1,325 items scored; chance 25%), all on the same items:

| Method | Accuracy |
|---|---|
| Llama, PMI score (the primary score fixed in advance) | 35% (CI 28–42) |
| Caption copy | 52% |
| Caption copy, assignment-aware | 59% |
| Frame copy | 85% |

- Llama is right 35% of the time whether caption copy is right or wrong, and it stays flat across gap novelty.
- Combining it with caption copy gives no gain: 51% with the weight chosen by leave-one-channel-out.
- **The Kaggle run should be stopped.** More items won't change the conclusion.

**The exception (matched distractors) failed** (`scripts/forced_choice_matched_feasibility.py`). The criterion was ≥300 items with both copy baselines ≤35%:

| Distractors matched on | Items | Caption copy | Frame copy |
|---|---|---|---|
| Captions | 675 | 16% | 87% |
| Frames, tight | 183 | 29% | 35% |
| Both, loose | 286 | 19% | 65% |

The true gap is almost always the span most visually similar to its neighbors, so fully matched distractors hardly exist. Reconstruction stays only as a section of the paper (the LLM loses to copy on every metric).

## 3. Decision 2: captions vs. frames: GO

`scripts/caption_vs_siglip_audit.py` → `results/caption_audit/`; 335 videos, 40 channels, no masking.

**Q1: does caption t match frame t?** (SigLIP 2 text against frames)
- **Accuracy:** MRR 0.142, against 0.079 chance and 0.063 for a control with the frames shifted in time. The best frame falls within ±2 s for 22% of captions (chance 8%).
- **Captions are systematically early.** The lag profile peaks at **+2 s** (mean z by lag: 0 → 0.58, +2 → 0.72, −2 → 0.35). Per video, the best lag is +1 or +2 for 222 of 335 videos and 0 for only 12.
- **The frame timing is correct.** `scripts/check_frame_timestamps.py`: for 4 random videos, exact-time ffmpeg frames match stored row t at offset 0 in all 16 checks.
- So Gemini's caption for second t describes what is on screen about 1–2 s later. The likely cause is its single-call, whole-video captioning and its timestamping.

**Q2: how much do the two views share within a video?** (ridge regression, held out by channel)
- R² is about **0.02** in every direction: MPNet↔frames, and SigLIP 2 text→frames.
- Captions share the video's topic with the frames, but barely track the second-to-second variation.

**Q3: do they change at the same moments?**
- Per-video Spearman between caption change and frame change is −0.04.
- Top-10% change events coincide (±1 s) 29% of the time, against 26% by chance.
- One caveat: with the +2 s lag, ±1 s may be too tight a window. Re-check with the lag corrected.

**Link to reconstruction outcomes** (per video, Spearman, channel-bootstrap CIs). Partial correlations control for the rate of visual change and the rate of caption change, which are correlated with Q1 and Q2 at about 0.4.

| Agreement metric | vs. caption copy's deficit to frame copy (per-second rank) | vs. frame copy's forced-choice advantage | vs. Llama's deficit to caption copy |
|---|---|---|---|
| Q1 MRR | −0.51 (partial −0.39) | −0.30 (partial −0.18) | +0.33 (partial +0.26) |
| Q2 MPNet→frames | −0.48 (partial −0.36) | −0.27 (partial −0.16) | +0.32 (partial +0.24) |
| Q3 Spearman | ≈ 0 | ≈ 0 | ≈ 0 |

Where captions are better grounded in the frames, caption copy gets closer to frame copy, and Llama falls *further* behind caption copy.

## 4. E1 re-run with SigLIP 2 text

- **Frame arms (SigLIP 2):**
  - oracle 0.333, repeat 0.217, lerp 0.198, other-video 0.219;
  - "other-video frames beat the frame oracle" was an artifact of the space mismatch;
  - other-video frames still equal repeat, so frame fills aren't gap-specific under E1 either.
- **SigLIP 2 text arms:** recon 0.322 > oracle 0.272, and recon from another video 0.268. The "recon beats oracle" style artifact persists in SigLIP 2 text, so MPNet stays primary.
- **MPNet arms:** unchanged (recon 0.237 = same-video other gap 0.238).
- **Symmetric design (K=3, chance 0.611):** vis oracle 0.707, MPNet oracle 0.671, SigLIP 2 text oracle 0.630. The frame oracle now has some headroom, but it is small.
- **E1 stays demoted** as an anchor.

## 5. Next steps

1. **Stop the Kaggle forced-choice run** (user).
2. **Commit today's changes** (fixes, three new scripts, doc corrections).
3. **Firm up the caption timing finding**, which may be the paper's headline:
   - Is it robust across channels and caption styles?
   - Does the lag depend on position in the video (drift) or is it constant?
   - Does re-aligning captions by +1 to +2 s improve caption copy and LLM reconstruction against frames? This is a cheap test: shift and re-score.
   - Re-check Q3 after correcting for the lag.
4. **Caption correctness audit (Q4)**, starting with the hallucination class (the "runner" example). Use about 50 sampled seconds, manually or with a VLM judge once the API budget is decided.
5. **Re-run or retire the shared-target evaluation** with SigLIP 2 text, and mark its old results invalid.
6. **Paper reframe:** "what dense LLM captions keep and lose relative to visual embeddings (topic yes; timing and second-level change no), and why that caps text-based gap reconstruction."
