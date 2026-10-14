# Handover

The newest session is at the top. Older sessions stay below because other docs link to their sections (for example "§1" means the 2026-10-07 section 1).

# 2026-10-14 (afternoon) → next session: Gate 1

**Start here:** `docs/new_direction/unshuffle_pilot_results.md` (the Gate 1 report), then `unshuffle_pilot_preregistration.md` (what was fixed in advance, and its Deviations section).

**Decided (user, 2026-10-14):** Gemini only writes captions; Claude through `claude -p` is the reader to evaluate (Sonnet main, Haiku everywhere, Opus on a subset, confirmed by the calibration's rule). Recorded in `paper_plan.md` §0, E1–E10, budget and §7. Gemini spend falls to ≈ $25.

**Built and committed** (`scripts/unshuffle/`, 50 tests in `tests/test_unshuffle.py`):
- `core.py`: puzzle sets P12, C6, G6, K4 d = 1–3 with counterbalanced shuffles (shuffle 2 is shuffle 1 reversed); τ, adjacency, local direction, chance values; exact Hamiltonian-path and linear-ordering DPs (and a batched path DP); channel-cluster bootstrap.
- `cues.py`: rule-based order-cue removal for the leak sets (20 real captions in the tests).
- `arms.py`, `run_free.py`: chaining, edge, and leave-one-channel-out probe and pairwise arms; writes `results/unshuffle/{puzzles.json, free/}`.
- `frames.py`: the reader frames, at exactly t = 5p + k, long edge 448 px, in `local/unshuffle_frames/` (aligned with the SigLIP rows: checked).
- `claude_reader.py`: listwise reader with images through stream-json; blind-runner isolation, pinned model IDs (`claude-{haiku,sonnet,opus}-5-5`), effort medium, one retry on format failure, quota recorded per call. Item sets `smoke` and `calibration`; add new sets in `item_set()`.
- `analyze.py`: every table in the results doc, plus the Prop. 3 Monte Carlo.
- Prompts: `prompts/unshuffle/listwise.txt`, `parts.json`.

**Two bugs found and fixed** (logged as deviations): the DPs got pieces in true order, and the index tie-break leaked direction into symmetric arms; the first frame extraction (`fps=1`) was 0.5–1 s late.

**Results in one paragraph:** see the results doc. Frames beat captions for every Claude model, in direction as well as structure (Sonnet P12 τ +0.67 vs +0.18; Opus +0.76 vs +0.26). H2 fails at 0 s and 5 s gaps, and scale helps frames, not captions (H6 reversed). A continuity-only edge arm gets most of the frames' direction on contiguous puzzles. The no-reader structure advantage of frames disappears with 5 s gaps (H1 holds). The leak shows in learned probes and survives cue removal. Prop. 3 is calibrated for captions and gapped frames; contiguous frames need a shot-aware model.

**Quota measured:** Sonnet 0.175 and Opus 0.18–0.19 points of the 5-hour window per call (a 12-piece frame call is 10k input tokens); the 7-day figure moves about 1 point per 80 Sonnet calls and 2 per 80 Opus calls.

**Next (needs the user's go; `unshuffle_pilot_results.md` §5):**
1. The reframing (continuity has a range; language can only compete beyond it), for the user and partners.
2. Main reader: Opus recommended (same 5-hour cost per call as Sonnet, stronger on frames, about 2× the 7-day cost).
3. Gap-knob reader pilot: add K3 d = 5 (20 s gaps) to `core.SETS`, write it into the pre-registration's Deviations section before running, then 660 Opus calls (30 videos × K4 d = 1–3 and K3 d = 5 × 1 shuffle × frames and captions).
4. Redesign E2 (reversal) around the knob, with the edge arm as the continuity-only reference.
5. Kaggle pairwise arms (Llama-8B, Qwen2.5-VL-7B) and the theory write-up are still open from Phase 1.

# 2026-10-14 → next session

**Goal:** plan the paper for a top-tier venue after a literature review, now that paid Gemini and Claude Code runs have a go.

**The user's directives (2026-10-14):**
1. Literature review for novelty and contribution; adjust or pivot to get them.
2. Go for more experiments with Claude Code and the paid Gemini API; the credits work.
3. A rigorous paper, theory and experiments, with top-tier experiments.
4. Nice to have: insight into the power of language vs. visual perception.

**Start here:** `docs/new_direction/paper_plan.md` (the master plan; it wins over `new_plan.md` and the kickoff plan where they conflict), then `docs/new_direction/related_work.md`.

## A. Literature review (first pass; `related_work.md`)

- **No direct threat** (no paper meets all three kill criteria). About 40 papers were checked at their abstracts or full text.
- **Closest:** SPLICE (arXiv 2509.24640, 2025). COIN clips are reordered from video, text or both. Its text is human step labels, and it has no continuity control and no matched non-reader arms.
- **Also close:** Wu et al. ACL 2022 (instruction manuals: text τ 0.82 vs images 0.16 for models), Sort Story (EMNLP 2016), TempVS (ACL Findings 2025: sentence ordering 86% vs image ordering 54%).
- **Consequence:** "language orders better than vision" is already known for models, so it cannot be the headline. What no one has: captions made in isolation from exactly the frames the vision arms see (an information control), the gap as a graded knob across modalities, matched arms, and a split into structure vs. direction.
- **The gap device is known:** VCOP (CVPR 2019) uses a fixed 8-frame interval between clips "to avoid trivial solutions". The novelty is varying it and comparing modalities.
- **Second pass still to do** (`related_work.md` §6): VIST's descriptions in sequence (DIS), step-order verification in video, current LLM sentence ordering, and the cognitive science of language vs. perception.

## B. The plan (`paper_plan.md`)

- **Reframing:** from *how much* order each modality carries to *why*.
  - **Prop. 1:** isolated captions are a garbling of the frames, so I(π; captions) ≤ I(π; frames). Any caption advantage is usable information (Xu et al., ICLR 2020), and the in-context leak is formally a break of this chain.
  - **Prop. 2:** symmetric similarity cannot know direction (E[τ] = 0), so order is measured as structure (undirected adjacency) and direction (a reversal test).
  - **Prop. 3:** a continuity model predicts how gaps hurt similarity chaining per modality, from measured similarity-decay profiles.
- **Hypotheses H1–H6:** pixels carry structure at least as well; captions carry direction better for the same reader; verbalization inside one model helps; the advantage grows with agents, goals and scripts; the leak; scale (exploratory).
- **Experiments E1–E10:**
  - E1: main comparison, matched no-reader, learned and reader arms.
  - E2: reversal test.
  - E3: K4 gap knob (4 pieces, gaps 0, 5 or 10 s) with continuity profiles.
  - E4: verbalization inside Gemini and Claude.
  - E5: leak.
  - E6: where language wins.
  - E7: scale.
  - E8: COIN as a second, script-rich dataset.
  - E9: human baseline.
  - E10: robustness.
- **Confirmatory design:** the 30 pilot videos are for development. The confirmatory tests are fixed in a pre-registration and committed before any arm runs on the 305 held-out videos; up to 4 primary tests, Holm-corrected.
- **Budget:** Gemini ≈ $80–120 in total (≈ $14 captions, ≈ $40–75 readers, ≈ $25 COIN), against about $12 of credit now. Claude runs on subscription quota: Haiku on everything, Sonnet on most, Opus on 100 videos. Kaggle is free.

## C. Facts established this session

- **Claude can read frames:** `claude -p --model haiku --tools "" --input-format stream-json --output-format stream-json --verbose` accepted a base64 PNG in a user message and answered correctly, in one turn with an empty tool list. So `scripts/blind_llm_runner.py` can be given an image mode for frame readers.
- **The captioner saw low-resolution frames:** in `calls.jsonl`, an isolated call has 330 video tokens for 5 frames (≈ 66 per frame) plus 187 text tokens. Frame readers can get the same frames cheaply: a 12-piece Gemini puzzle is ≈ 4k video tokens, so the cost is mostly thinking output. Giving readers frames at no lower resolution than this keeps Prop. 1 valid.
- **Billing:** the user confirmed the Cloud credits apply. `calls.jsonl` logs $1.56 in total. A comparison of the billed cost with that estimate is still worth doing once, before the budget decision.

## D. Next (Phase 1 in `paper_plan.md` §5)

1. Write `docs/new_direction/unshuffle_pilot_preregistration.md` (kickoff §3.5, plus E2–E4 on the pilot) and commit it before any arm runs.
2. Build `scripts/unshuffle/` with tests: pieces, puzzles (P12, C6, G6, K4), no-reader and learned arms, pairwise aggregation (exact subset DP), score, continuity profiles. Run on the 30 pilot videos.
3. Reader runners:
   - Gemini listwise, with a cost log and a cumulative cap;
   - Claude listwise with frames through stream-json.

   Pilot both with `--limit`, then on the 30 videos (≈ $3 + quota). Measure per-call cost, format failures and position bias.
4. Kaggle: Llama-8B and open-VLM pairwise scores on the pilot pieces.
5. Theory write-up: Props. 1–3 with proofs, and the Monte Carlo for Prop. 3.
6. **Gate 1:** report to the user with the pilot table, measured costs, the confirmatory pre-registration draft and the budget ask, plus the partner decisions in `paper_plan.md` §7.

The live plan doc (https://claude.ai/code/artifact/a0d1b4ac-8b80-42f6-8935-7183326d43a3) still matches `new_plan.md` at rev 54. It has neither the pilot results nor this plan. Sync it if the partners will read the plan there.

---

# 2026-10-13 → next session

**Goal:** move the main study from generation to unshuffling, make the re-captioning leak-free, pay for it, and freeze the captioning setup on 2 videos before the 30-video pilot.

**Start here:**
- `docs/new_direction/new_plan.md`: why generation is demoted, the unshuffling design, risks. It is the export of the live plan doc (https://claude.ai/code/artifact/a0d1b4ac-8b80-42f6-8935-7183326d43a3); edit both together.
- `docs/new_direction/experiment_kickoff_plan.md`: the steps, gates, commands and stop rules. The next step is in section E below.
- `docs/new_direction/literature_review_plan.md`: a brief for a research agent; not yet run.
- Checklist: `docs/TODO.md` section U.

## A. Review of the plans (applied and committed)

The three docs were reviewed and fixed before any spending:
- **Audio:** `caption_pieces.py` kept the audio track, so the captioner heard narration despite "describe what is seen". Pieces are now cut with `-an`; outputs record `audio: false`. The in-context input fell from 5,816 to 4,339 tokens, confirming the audio had been processed.
- **On-screen text:** clocks, timestamps, dates, numbering and subtitles are not transcribed; they would give the text arm an order the frame arms cannot use.
- **Matched arms:** the original main contrast (Llama vs. SigLIP edge chaining) was reader vs. non-reader, so "pixels lose on gapped puzzles" would follow from the design. Both modalities now get similarity chaining and a learned order probe; a vision reader is optional.
- **Metrics:** pairwise order accuracy equals (τ+1)/2, so it is dropped. Kendall τ (global) and adjacent-pair accuracy (local) are both reported, with the primary one fixed in a pre-registration. Symmetric arms get a seeded coin-flip direction, never |τ|.
- **Puzzles:** a contiguous-6 control (pieces 0–5, 6–11) separates "fewer pieces" from "gaps". Gaps weaken edge matching (edges 6 s apart forward, 14 s back); they don't remove it.
- **Leak:** in-context − isolated is a "context advantage" (leak plus accuracy). It is split with an "in-context, cue words removed" set; the original captions are descriptive only (different model, length and the 1–2 s lead).
- **Orderability:** per-video τ is noisy (SD ≈ 0.22 with 12 pieces at chance). Reliability comes from the two gapped puzzles per video; orderability is one pre-specified arm, not the best over arms.
- **Literature brief:** Sort Story, multimodal instruction-manual sequencing and RecipeQA are now in area A (likely close prior work). VIST's descriptions in isolation vs. in sequence is the precedent for the leak design. Gaps against edge matching are a known device (jigsaw tiles), so the novelty is the graded knob across modalities.
- **Claims corrected in `new_plan.md`:** generation stops scaling after Sonnet (Llama → Claude does improve; Opus beats Sonnet on order-free rank −1.8 [−3.2, −0.4]). Frame copy (87.5%) is level with Opus on recognition.

## B. Billing and the Gemini API

- **Key:** `GEMINI_PAID_API_KEY` in `~/.secrets`. Never `GEMINI_API_KEY` (free) for this work.
- **History:** free tier on 2026-10-12 (429, 20 requests per day). On 2026-10-13: 402 "prepayment credits are depleted" (a Prepay billing account with a zero balance). After the user added ₪15 of prepay credit, 21 test calls in a row passed: paid tier confirmed.
- **Cloud credits:** under Prepay, Google Cloud promotional credits apply only after funds are added, and not all of them cover the Gemini API ([billing docs](https://ai.google.dev/gemini-api/docs/billing)). Whether the ₪30.71 of credits are used is still unknown.
- **Latency on the paid tier:** median about 11 s per isolated call and about 30 s per in-context call, against 1–7 minutes on the free tier.
- **Spend:** `calls.jsonl` logs $0.236 (of which $0.053 are free-tier calls from 2026-10-12, never billed), plus about $0.006 for the 21 test calls outside the log. So about $0.19 was billed, all estimated at $0.50/M in and $3/M out. The full run (about $15–20) needs more than the ₪15 (~$4) on the account.

## C. Stage 1 (2 videos: `Joe-Robinet_9-clip-2`, `4k-Relaxation_4-clip-4`)

Three runs, all kept (`piece_caption_check.py` reads every folder):
- `results/piece_captions/_with_audio/`: 2026-10-12, with audio, prompt v0 (10 outputs).
- `results/piece_captions/_prompt_v1/`: no audio, prompt v1 (26).
- `results/piece_captions/gemini-3.8-flash/`: no audio, **prompt v2, frozen** (26; all valid; median 11 s per isolated call; $0.087).

Check (24 captions per set; chance MRR 0.259, so differences of about ±0.1 are noise):

| Set | MRR | Words | Continuity cues | On-screen text | Repetition | Lead |
|---|---|---|---|---|---|---|
| isolated v2 (frozen) | 0.58 | 31 | 4% | 0% | 0.67 | 0.00 |
| isolated v1 | 0.44 | 36 | 8% | 50% (a watermark) | 0.69 | 0.00 |
| in-context v2 | 0.37 | 23 | 33% | 0% | 0.54 | +0.01 |
| in-context v1 | 0.44 | 21 | 33% | 0% | 0.51 | +0.02 |
| original, 5 per-second captions joined | 0.45 | 59 | 42% | 0% | 0.70 | +0.01 |

Read against frames (3 frames per piece; full resolution for disputed claims):
- **v1 → v2:** v1's isolated captions named the constant "ProArtInc" watermark in all 12 canyon pieces and listed the whole setting each time; v1's in-context captions invented a forward glide over static shots. v2 fixed all three. Isolated captions still restate the setting in Joe's pieces 6–11, where nothing in the scene changes.
- **The leak, in v2's in-context captions:**
  - Joe piece 1 "reveals a camper resting on a camouflage mat", who first appears in piece 2 (runs ahead).
  - Joe piece 8 "continues speaking".
  - Canyon pieces 8 and 10 have "tiny figures of hikers" that are not in the frames (checked at full resolution), carried over from pieces 4–7.

  So the leak moves content between pieces, not just cue words. These are the examples now in `new_plan.md`.
- **Errors that isolation does not fix:** the canyon's piece 4 calls a cut a pan; Joe's piece 9 is ambiguous (eating or wiping his face; his hand is at his mouth). The in-context caption of Joe's piece 0 calls the yellow tent green.
- **Orderability:** canyon pieces 0–3 and 8–11 are near-identical static shots, so their order can't be recovered from either modality. They are a ready example for the orderability measure.
- **Lead:** about 0 for every set, including the original captions (+0.01), whose 1–2 s lead is known. At 5 s pieces and 2 videos the metric can't detect it; re-check on the pilot before relying on it.

**Stage 1b: 5 more videos with v2, chosen for untested failure modes** (all from the 30-video pilot list, so they count toward it; 65 calls, all valid, $0.22): `Weathershot_1-clip-0` (storm compilation), `Military-Archive_6-clip-0` (archival, cuts), `Disaster-Compilations_6-clip-2` (phone and TV footage), `TreadmillTV_3-clip-12` (a near-uniform forest walk), `Olly's-Farm_11-clip-1` (cab-view farm work; the `'` file-name join works).
- **Check:** isolated MRR 0.57, in-context 0.54, original 0.48 (60 captions each).
- **No v3 needed.**
  - No clocks, subtitles or numbering were transcribed, and a TV channel logo ("PTQ") was correctly left out.
  - Aircraft numbers ("31086", "21066") were transcribed. They are object identifiers that link pieces of the same shot, not order cues, so they are allowed; watch their frequency in the pilot.
- **The leak again:** in-context captions narrate across pieces ("the overturned car skids", "the truck comes to a stop", "re-enters the shed", "another heavy load").
- **The isolation cost again:** one tilted frame from a camera jolt (Olly's piece 8) became "the vehicle suddenly tips over onto its right side" in the isolated caption, while the in-context caption reads it correctly ("the camera jolts"). One case in 60, so the prompt is not changed.
- **Orderability:** TreadmillTV's isolated captions are near-identical (repetition 0.84), as the footage is, apart from the pond in pieces 8–10.
- **Remaining pilot:** 23 videos (299 calls, about $1.00). The command in E.2 skips the 7 videos already done.

**Step 2, the 30-video pilot: done 2026-10-13** (30 videos, 30 channels; 360 isolated + 30 in-context captions, all valid; about $1.10; `calls.jsonl` logs $1.56 in total).
- **Runner fixes during the run** (committed):
  - A bare 429 "Resource has been exhausted" (no quota id; the next call succeeded) stopped the run once. Bare 429s now get the 500/503 backoff; a 429 that names a quota still stops the run.
  - `Skip-Talbot_0-clip-9` in-context had one timestamp typo (00:10:00.000 for 00:00:10.000). A segment now counts as aligned if its start or its end is on the grid, and the output was rebuilt from the cache at no cost.
- **Check, paired per-video differences with 95% bootstrap CIs (30 videos = 30 channels):**

  | Metric | in-context − isolated | original − isolated |
  |---|---|---|
  | MRR (isolated 0.534) | −0.041 [−0.087, +0.002] | −0.059 [−0.094, −0.024] |
  | lead | **+0.015 [+0.010, +0.020]** | +0.004 [+0.000, +0.007] |
  | continuity cues | **+0.139 [+0.075, +0.206]** | +0.386 [+0.281, +0.494] |
  | repetition (isolated 0.61) | −0.066 [−0.085, −0.048] | +0.047 [+0.017, +0.078] |

  In-context captions run ahead into the next piece and use more continuity words, so the leak is measurable at pilot scale. Isolated captions match their own piece best.
- **Spot read of 20 random isolated pieces against frames:**
  - 17 clearly correct, 0 clearly wrong, 2 uncertain at contact-sheet resolution ("hailstones" on gravel; a person in a bamboo shelter).
  - Watermarks and channel logos were left out in 5 cases.
  - `Dan-Robinson_6-clip-6` has a burned-in running timecode and "NOT FOR BROADCAST", and nothing of either was transcribed.
  - In-context lapses: invented vehicle motion over a static storm shot; a dropped boat shot.
- **Open: title cards and labels are transcribed** in 4 of 30 videos:
  - an intro sequence ("Into the Breach", "With Erling & Geir": `Norwegian-Explorer_7-clip-0`);
  - country labels in a compilation ("Colombia", "Israel": `Climate-Change_6-clip-7`);
  - a chapter card (`BC-Bushcraft_3-clip-0`);
  - an animated channel intro (`Millennial-Farmer_10-clip-0`).

  These carry editing structure (intro first, credits after), not event dynamics.
  - **Decided (user, 2026-10-13): tag, don't re-caption.** `piece_caption_check.py` writes `results/piece_captions/check/text_tags.csv`, a broad tag (`TEXT_TAG`): quoted text or words like text, title, logo, labeled, marked. It tags 44 of 360 isolated pieces in 11 videos.
  - The pre-registration (kickoff 3.5) reports every main result again without the tagged pieces. Prompt v2 stays frozen.

## D. Code (all committed on `main`, not pushed)

- `scripts/caption_pieces.py`: audio removed; `audio: false` in outputs; an `ERROR` caption or an in-context run whose segments don't start at 0 s is saved as `*.invalid.json`.
- `scripts/piece_caption_check.py` (new): per caption set, SigLIP 2 retrieval of the true piece among 12 (MRR; chance 0.259), length, cue words, on-screen text candidates, repetition (MPNet), and lead (caption p vs. pieces p±1). It reads set-aside folders `results/piece_captions/_<tag>/` as `<mode>_<tag>` sets. Outputs: `results/piece_captions/check/{per_video,flagged}.csv`.
- The cue-word lists of the 2026-10-12 count were never saved; the check uses a reconstruction (`CUES` in the script).

## E. Next

1. **Step 0.3 (user, from 2026-10-14):** Cloud Console → Billing → Reports for the project. Is there a Gemini cost line, and are the ₪30.71 of Cloud credits applied? Compare with `sum(usd)` in `calls.jsonl`; if they differ by more than 2×, correct `--usd-in`/`--usd-out`.
2. **Done: Step 2, the 30-video pilot** (results in §C). Title cards and labels are handled by tagging plus a robustness analysis, with no re-caption. For reference, the pilot commands were:
   ```bash
   source ~/.secrets && .venv/bin/python scripts/caption_pieces.py in_context --pilot 30 --dry-run
   source ~/.secrets && .venv/bin/python scripts/caption_pieces.py in_context --pilot 30 --max-usd 3 --sleep 5
   source ~/.secrets && .venv/bin/python scripts/caption_pieces.py isolated   --pilot 30 --max-usd 3 --sleep 5
   .venv/bin/python scripts/piece_caption_check.py
   ```
   Stop on any 429 or 402, on more than 5% invalid outputs, or at the cap (cumulative). Then a summary table and a hand read of about 20 random pieces against frames.
3. **Step 3.5 before Step 3:** write and commit `docs/new_direction/unshuffle_pilot_preregistration.md`. Still open: the primary metric (τ or adjacency), the primary contrast (matched-method text − frames, or Llama − edge chaining across contiguous-6 vs. gapped-6), Llama's aggregation (DP or Borda), and the arm that defines orderability.
4. **Step 3:** build `scripts/unshuffle/` (`build_pieces.py`, `make_puzzles.py`, arms, `score.py`) with tests in `tests/`. Llama pairwise scoring runs on Kaggle (reuse `scripts/kaggle_forced_choice.py`).
5. **Literature review:** hand `docs/new_direction/literature_review_plan.md` to a research agent (web only, no paid APIs). The deliverable is `docs/new_direction/related_work.md`.
6. **Partners:** the four decisions at the end of `new_plan.md` are still open.
7. **Step 4 budget:** about $15–20 of Gemini credit, after the pilot gate.

---

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

**Haiku and an Opus probe (2026-10-09):**
- **Shuffle bug, fixed.** `run --shuffle` used to shuffle only the items *not yet saved*, so the "seeded" sample depended on what a model already had on disk. Sonnet's main 160 were drawn after its 40 earlier no-hint items were saved, and Haiku's after a 10-item pilot, so the two `--pool all --shuffle --limit 160` runs overlapped on only 33 items.
  - Fix: the full list is now shuffled before saved items are dropped.
  - New `--match MODEL` restricts a run to the items MODEL has valid answers for.
  - Sonnet's main estimate = its `choice_nohint` rows after the first 40 distinct items. That split reproduces 76.9% exactly, and `scripts/blind_choice_compare.py` uses it.
- **Haiku, all 200 Sonnet no-hint items plus 137 others** (337 calls, ~1.6M input tokens). On Sonnet's main 160 (136 videos, 36 channels):

  | Method | Accuracy |
  |---|---|
  | **Haiku** | **51.2% [44.3, 58.2]** |
  | Sonnet | 76.9% [70.1, 82.9] |
  | Caption copy | 45.6% |
  | Frame copy | 87.5% |

  - Paired: Haiku − caption copy +5.6 [−4.4, +16.6]; Haiku − Sonnet −25.6 [−33.7, −17.3].
  - Haiku is level with copy, so recognition grows with scale.
- **Opus probe** (`--pool all --shuffle --match sonnet --limit 20`; 15 of the 20 items are in the main 160):
  - Opus 17/20, Sonnet 13/20, Haiku 8/20 on the same items.
  - All calls valid, 1 turn, 3–7 s each.
- **Cost, measured against the account's 5-hour figure** (account-wide, so other sessions add noise):
  - Haiku: about 1 point per 160 calls.
  - Opus: 4 points per 20 calls, so about 0.2 per call against Sonnet's 0.15.
  - Opus on the remaining 145 main items ≈ 29 points of the 5-hour window and ~2 points of the 7-day one.
- **Opus on the full main 160 (done, same day).**
  - Command: `run --task choice_nohint --model opus --pool all --shuffle --items results/blind_llm/sonnet_main160_items.json`. The new `--items FILE` option restricts a run to a JSON list of ids; the file was written from `blind_choice_compare.sonnet_main_items()`.
  - Cost: 145 calls took the 5-hour figure from 31% to 54%, about 0.16 points per call, close to Sonnet.

  | Method | Accuracy |
  |---|---|
  | Haiku | 51.2% [44.3, 58.2] |
  | Sonnet | 76.9% [70.1, 82.9] |
  | **Opus** | **85.0% [78.7, 90.7]** |
  | Caption copy | 45.6% |
  | Frame copy | 87.5% (assignment-aware 95.6%) |

  - Paired: Opus − frame copy = **−2.5 [−10.1, +5.6]**, i.e. level; Opus − Sonnet = +8.1 [+4.1, +12.4].
  - Flat across width (83–88%).
- **Thesis implication (for the user to decide):**
  - "Captions cap text-only recovery" no longer holds for *recognition*: a frontier reader matches the frame-copy baseline from captions alone. It still holds for second-level timing (§3), and so far for *generation* (Sonnet ≈ caption copy on 10 videos).
  - The draft's abstract, §1, §4.2 and §6 now say this, with an author note in §4.2.
- **Sonnet reconstruction on 50 videos** (done, same day). `run --task recon --model sonnet --limit 200` (video-major, so 50 videos × W ∈ {1, 4, 8, 16}, 28 channels), then `score --model sonnet`. 199 valid of 200; `reparse` couldn't recover the last one. Over the 191 gaps all three arms filled, with per-video means and channel-bootstrap CIs:

  | Comparison | MPNet text rank | SigLIP 2 frame rank |
  |---|---|---|
  | Sonnet − caption copy | −1.4 [−3.0, +0.5], level | **+3.3 [+1.5, +5.1], worse** |
  | Sonnet − Llama | −4.8 [−6.7, −2.9] | −3.1 [−5.4, −1.0] |

  - **"Recognizes but can't generate" holds** at 50 videos: 77% on forced choice, yet no better than the boundary caption at writing the gap.
  - Cost: 160 new calls took the 5-hour figure from 55% to 74%.
- **Opus reconstruction, partial.** `run --task recon --model opus --limit 200` hit the `--max-5h 0.8` stop after 26 items (7 videos; 75% → 80%). On those 26 gaps, mean rank:

  | Arm | Text rank | Frame rank |
  |---|---|---|
  | Opus | 20.5 | 23.9 |
  | Sonnet | 21.3 | 21.3 |
  | Caption copy | 22.9 | 18.4 |
  | Llama | 22.2 | 24.1 |

  - Same pattern as Sonnet: about level in text rank, worse than copy in frame rank. Too few items for CIs, but no sign yet that generation scales.
- **Opus reconstruction, 32 videos (same day, later).** `run --task recon --model opus --limit 100` added 100 items, all valid, for 126 in total. The 5-hour figure went from 15% to 41% (~0.26 points per call; recon prompts are longer) and the 7-day one from 36% to 37%. Then `score --model opus`, and `scripts/blind_recon_compare.py sonnet opus` (new; per-video means, channel bootstrap). It reproduces Sonnet's 50-video numbers when run as `blind_recon_compare.py sonnet`. On the 120 gaps all four arms filled (32 videos, 24 channels):

  | Comparison | MPNet text rank | SigLIP 2 frame rank |
  |---|---|---|
  | **Opus − Sonnet** | **−0.7 [−2.7, +1.0], level** | **−0.1 [−1.4, +1.2], level** |
  | Opus − caption copy | −2.3 [−4.6, +0.2], borderline | **+2.6 [+0.9, +4.4], worse** |
  | Sonnet − caption copy | −1.5 [−4.0, +1.2] | +2.7 [+0.7, +4.6] |
  | Opus − Llama | −4.9 [−7.6, −1.9] | −3.8 [−6.4, −1.7] |

  - **Generation doesn't scale from Sonnet to Opus**, while recognition went from 77% to 85%. "Recognizes but can't generate" now holds at the largest model. The draft's §1, §4.2 and §6 now say this.
  - Opus's text-rank edge over copy is almost all at W = 1 (−7.6). At W = 4–16 it's −1.4 to +1.1.
  - Caveat for the frame-rank deficit: caption copy is the captioner's own text, so it matches the true captions' style and Claude's text doesn't. Part of the deficit may be style. A style-matched copy baseline (copy rephrased by Claude) would test this.
  - The other 74 items (videos 33–50) are optional. They would only tighten Opus − caption copy in text rank: `run --task recon --model opus --limit 74`, about 20 points of the 5-hour window.

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
