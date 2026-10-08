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

## D. Frontier text-only arm through Claude Code (no API key)

**Why.** Is the reconstruction cap in the captions or in the 8B model? A frontier model that sees only captions answers that. The model being multimodal doesn't matter, because it never sees frames. Running Claude through `claude -p` uses the user's Claude Code subscription (usage limits), not the API, so it needs no budget.

**Driver:** `scripts/blind_llm_runner.py`, one fresh isolated `claude -p` session per item:
- no built-in tools (`--tools ""`) and no MCP servers (`--strict-mcp-config` with an empty config);
- `--system-prompt` replaces the agent prompt;
- `--no-session-persistence`;
- a fixed, empty working directory outside the repo, checked before every call;
- refuses to run if `ANTHROPIC_API_KEY` is set (that would bill the API);
- `--effort` pinned (default `medium`) and recorded, because the models think adaptively;
- prompt caching off (`DISABLE_PROMPT_CACHING=1`). In the pilot, Claude Code's own cache breakpoints meant nothing was ever read back, so every call just wrote a 1-hour cache entry. With caching off, each call costs about 5k input tokens.
- `--bare` can't be used: it accepts only an API key and ignores the subscription login.

**Tasks:**
- `recon`: the exact Llama prompt (`prompts/dense_window` via `JSONPromptBuilder`, `FixedFillMasking(w, i=29)`). Items: a seeded sample of 100 videos × W ∈ {1, 4, 8, 16}, 400 items in video-major order, so `--limit 200` = 50 videos × 4 widths.
- `choice`: the 180 forced-choice items Llama scored. The context has all 4 spans masked, as in the Llama run, and the 4 candidates appear in seeded random order. The correct letters are balanced (A 43, B 48, C 39, D 50). The prompt was checked: the true captions don't appear outside the candidate list.
  - **Protocol caveat:** Llama was scored by log-prob (PMI). Claude makes a direct choice. The paper must say so. Getting Llama's direct choice would need a GPU run.
- `score`: MPNet rank among the video's 60 captions (as in the master CSV) and SigLIP 2 frame rank, for Claude, Llama and caption copy on the same gaps. Verified against the master CSV on 61 Llama gaps: mean |diff| 0.01 ranks.

**Pilot (Haiku 5.5, 10 calls in total):**
- `probe`: tools `[]`, mcp_servers `[]`, 1 turn. ISOLATION OK.
- 9 items: all valid JSON, 1 turn each, 2–6 s per call, sensible captions, choice 3/3. The 3-gap scores mean nothing yet.
- Pilot outputs are in `results/blind_llm/pilot/` (before effort pinning). The current `recon__haiku.jsonl` holds 3 items.

**Saving and re-running rules** (the user asked that no paid-for output is ever lost or re-bought):
- Every call is saved in full to `results/blind_llm/<task>__<model>__events.jsonl`, before any parsing. That includes the prompt, every stream-json event (thinking blocks included, when the model emits any), stderr and the return code.
- The compact row in `<task>__<model>.jsonl` carries a status: ok, invalid, error or timeout.
- Items with any saved row are never re-run. `--retry invalid|error|timeout` re-runs only the chosen statuses.
- `reparse` re-derives rows from the saved calls after a parser fix, with no calls. It already recovered one Sonnet answer where the model wrote a draft array, then "Correction, here is the full valid array". The parser now takes the last JSON block that is a valid answer.
- An earlier parser crash lost one call's output, before the raw call was saved first. That's fixed.

**Sampling options:**
- `--shuffle` takes a seeded sample. Without it, items run in video order, so the first 40 forced-choice items came from 10 videos and only 4 channels.
- `--pool llama` (default) uses only the 180 items Llama scored, which cover 14 channels. `--pool all` uses all 1,325 items, which cover 40 channels and have copy baselines but no Llama score.
- Task `choice_nohint` drops the sentence "the other candidates belong to the other missing intervals". That sentence turns the task into a jigsaw (match each candidate to its slot), which Llama's PMI scoring never used.

**Small Sonnet run (2026-10-08; 80 calls, ~0.38M input tokens, effort medium):**
- **Forced choice, first 40 Llama-scored items (10 videos, 4 channels), all on the same items:**

  | Method | Accuracy |
  |---|---|
  | Sonnet | **92.5%** |
  | Frame copy | 92.5% |
  | Caption copy | 45% |
  | Caption copy, assignment-aware | 40% |
  | Llama PMI | 42.5% |

  If this holds, the forced-choice cap was the 8B model, not the captions. It is **not yet trustworthy**: 4 channels, the jigsaw hint, and direct choice against PMI.
- **Reconstruction, 10 videos × W ∈ {1, 4, 8, 16}** (rank of the true second among 60; lower is better):
  - Sonnet beats Llama at every width.
  - Against caption copy it is roughly level: better in MPNet text rank at W = 1 and 4 (15.2 vs. 32.4; 19.4 vs. 21.5), worse in SigLIP 2 frame rank at every width.
- **Tentative reading:** a strong model can *recognize* which content fits a gap, but cannot *generate* second-level content much better than copying a neighbor. That fits "captions keep topic, lose timing" better than "the LLM is the cap".

**Two 40-call checks (run 2026-10-08 with quota tracking):**
- **Broad sample** (`--pool all --shuffle --limit 40`: 38 videos, 23 channels):

  | Method | Accuracy |
  |---|---|
  | Sonnet | **75%** |
  | Caption copy | 47.5% |
  | Caption copy, assignment-aware | 52.5% |
  | Frame copy | 87.5% |

  By width, Sonnet scores 62%, 75%, 100% and 71% at W = 1, 2, 4, 8 (only 7–10 items per width). The first run's 92.5% came from 4 channels and was optimistic.
- **No hint** (`choice_nohint`, the first 40 items): 85% (34/40), against 92.5% (37/40) with the hint. The jigsaw hint adds a little; most of Sonnet's skill doesn't depend on it.
- **Reading so far:**
  - From captions alone, a strong model recognizes the right gap content far better than caption copy or Llama (≈75–85% vs. ≈45%), but below frame copy (≈88–92%).
  - So for *recognition*, the 8B model was a large part of the cap; the captions carry more than copy or Llama extract.
  - *Generation* (reconstruction) is still only about level with caption copy (10 videos).
  - The thesis survives in a revised form: the remaining gap to frames is real, but "the LLM loses to copy everywhere" is an 8B finding.
- **Thinking:** blocks are emitted (at medium effort on some calls), but their text is empty in the CLI output. Only the token count is visible, so the reasoning can't be saved.

**Main Sonnet estimate (run 2026-10-08):** `run --task choice_nohint --model sonnet --pool all --shuffle --limit 160`. No hint; 160 items, 136 videos, 36 channels; 95% CIs by channel bootstrap.

| Method | Accuracy |
|---|---|
| **Sonnet** | **76.9% [69.6, 83.0]** |
| Caption copy | 45.6% [37.7, 53.8] |
| Caption copy, assignment-aware | 53.7% [47.3, 60.2] |
| Frame copy | 87.5% [81.6, 92.3] |
| Frame copy, assignment-aware | 95.6% |

- **Paired differences:** Sonnet − caption copy = **+31.2 points [+20.7, +41.6]**; Sonnet − frame copy = **−10.6 [−18.3, −2.9]**.
- **Flat across width:** 78%, 77%, 77% and 76% at W = 1, 2, 4, 8 (about 40 items each). Caption copy is 33–51% and frame copy 78–95%.
- **Errors partly complement frames:** Sonnet is right and frame copy wrong on 11 items, the reverse on 28, both wrong on 9.
- **Quota:** 160 calls moved the 5-hour figure from 49% to 73% and the 7-day figure from 22% to 23%.
- **The paper's §4.2 now carries this result.** The "LLM loses to copy everywhere" claim is now limited to the 8B model, at least for recognition.

**Quota cost, measured.** Each row records the account's 5-hour and 7-day usage reported with the call; `run` prints them at the start and end of a batch, and `--max-5h` (default 0.8) stops a batch at that share. These are account-wide figures, so run batches with nothing else active.
- Each 40-call Sonnet batch (~195k input tokens) cost about **6–8 points of the 5-hour limit** and **~0.5 point of the 7-day limit**.
- That's roughly 500–600 Sonnet calls per 5-hour window, and several thousand per week.
- Opus will cost more per call. Measure it on a small batch before planning.

**Original full-run plan** (on hold until the small runs show the effect is real):

| Model | Run | Calls |
|---|---|---|
| Sonnet 5.5 | `run --task choice --model sonnet` | 180 |
| Sonnet 5.5 | `run --task recon --model sonnet` | 400 |
| Opus 5.5 | `run --task choice --model opus` | 180 |
| Opus 5.5 | `run --task recon --model opus --limit 200` | 200 |

After each run: `check --task <t> --model <m>` (validity, turns, tokens, choice accuracy), then `score --model <m>` for recon. Runs resume: valid items are skipped. A long run can go in the background, with a `--limit` per sitting to spread quota.

**Later, same driver, image mode** (to add): per-second captioning versus whole-minute captioning of about 30 videos with Sonnet. It tests whether whole-video captioning causes the lead and the lost timing, and whether the 0.18 is the captions or SigLIP's text-to-image link. It's also the Q4 judge (Opus, with planted wrong captions as a check).

## E. Next

1. Done: the near pool with lag-corrected captions (`caption_lag_robustness.py` part 5). The true caption goes from 0.17 to 0.28 at +2 s, against 0.20 for caption copy and 0.52 for frame copy, so the lead explains about a third of the near-second gap.
2. §4.1 of `captions_vs_frames.md` is filled. Llama trails caption copy by 3.9 ranks [3.1, 4.8], and only at W = 1 does it tie. Caption copy trails frame copy by 10.6 ranks [9.3, 12.0].
3. Done: figures 1 to 4 (`scripts/make_paper_figures.py`; list in `captions_vs_frames.md` "Figures").
4. Done: the abstract and related work. References marked † were cited from memory; the rest were checked. **The target is now 8 pages** (user, 2026-10-08), so all four figures stay. Next for the paper: notes to prose, using the page budget at the end of `captions_vs_frames.md`, then the LaTeX template.
5. **Main open threat to the thesis:** is the near-pool 0.18 a property of the captions, or of SigLIP's text-to-image link? `scripts/bridge_ceiling.py` tests it with Florence-2 per-frame captions, which are time-aligned by construction, on 30 videos from 30 channels.
   - **Done locally:** `extract` wrote 1,800 frames to `results/bridge_ceiling/frames/` (60 MB).
   - **`caption` needs the GPU.** On this CPU, Florence-2 takes about 30 s per frame, so about 15 hours.
   - **Frames are uploaded** to the private HF repo (`Y3/dense_video_captions/bridge_ceiling/frames.zip`).
   - **Kaggle:** paste `scripts/kaggle_bridge_ceiling.py` into a notebook (GPU T4, Internet on, `HF_TOKEN` secret). It downloads the frames, captions them, and uploads `captions.jsonl`; a re-run resumes.
   - **Then locally:** `bridge_ceiling.py pull-captions`, then `eval`.
   - **Risk:** the runner installs `transformers==5.14.1` (verified locally for Florence-2) on top of Kaggle's torch. If that combination fails, the error shows up at install or model load.
   - **Reading:** if per-frame captions also score about 0.18 near, soften the timing claim. If they score well above it, Gemini's captions lack timing.
6. Claude text-only runs (section D): Sonnet forced choice is done (77%). Next: an Opus cost probe (about 20 calls), Opus on the same 160 items, and Sonnet reconstruction on more videos. All of these spend quota, so confirm with the user first.
7. **Done: the bridge ceiling** (Kaggle captioning, then local `eval`; details in `captions_vs_frames.md` §3.6). It resolves item 5.
   - Florence-2 per-frame captions score near c = 0.47 [0.39, 0.54] through the same SigLIP 2 link, against 0.16 for Gemini (0.21 shifted +2 s) and 0.65 for frame copy.
   - Per-frame captions peak at lag 0 in 28 of 30 videos.
   - So the lost timing is in the Gemini captions, not the encoder, and the lead isn't a measurement artifact.
8. Doc reconciliation (TODO §1): now mostly moot, since `draft.md` is frozen. Just make sure the new draft uses the right counts.
9. Q4 caption correctness and the re-captioning control: no longer blocked on an API budget. They can use the blind runner's image mode (section D) on the subscription.

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
