# Experiment kickoff plan: from paid-API check to the unshuffle pilot

Written 2026-10-13, revised the same day after review. Companion to `docs/new_direction/new_plan.md` (the why and the design). This file is the how: concrete steps, commands, checks and stop rules, in order. Each step has a gate; do not start the next step until the gate passes and, where marked, the user has looked at the results.

## Where things stand

- **Prompts (v2, frozen):** `prompts/yt_video_processing/piece_isolated.txt` and `piece_in_context.txt`, with the same rule sentences in both files:
  - on-screen text is described, but subtitles, clocks, timestamps, dates and numbering are not transcribed, and watermarks and channel logos are left out;
  - the setting gets a few words at most;
  - camera movement is mentioned only when the camera clearly moves.

  v2 is the one revision round of the Step 1 gate. It fixes what the first prompt (v1) showed on Stage 1: isolated captions listed the whole setting in every piece and named a constant watermark in all 12, and in-context captions invented a forward glide over static shots.
- **Runner:** `scripts/caption_pieces.py`.
  - Defaults: `gemini-3.8-flash`, temperature 0.8, dynamic thinking, seed 0x5EED, 1 fps.
  - **No audio:** pieces are cut with the audio track removed (`-an`), so the captioner sees what the frame arm sees and cannot hear narration. Each output records `audio: false`.
  - Auth: reads only `GEMINI_PAID_API_KEY` (exported in `~/.secrets`, sourced by `~/.bashrc`).
  - Outputs: `results/piece_captions/<model>/<mode>/`; call log with token usage and estimated cost: `results/piece_captions/calls.jsonl`.
  - **Spend cap:** `--max-usd` is cumulative. It is compared with the sum over every call in `calls.jsonl` (all modes and runs; $0.053 logged so far, from free-tier calls), not with the current run's spend.
- **Stage 1 history** (outputs set aside are kept for comparison; `piece_caption_check.py` reads them as `<mode>_<tag>` sets):
  - 2026-10-12, with audio, prompt v0: 10 outputs (in-context for `Joe-Robinet_9-clip-2` and `4k-Relaxation_4-clip-4`, isolated Joe pieces 0–7), with the Stage 0 smoke test, in `results/piece_captions/_with_audio/`.
  - 2026-10-13, no audio, prompt v1: all 26 calls, in `results/piece_captions/_prompt_v1/`.
  - 2026-10-13, no audio, prompt v2: all 26 calls, in `results/piece_captions/gemini-3.8-flash/`. This is the frozen setup.
- **Billing (resolved 2026-10-13):**
  - On 2026-10-12 the key's project was on Google's **free tier**. The 429 named `generate_content_free_tier_requests`, limit 20, quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`.
  - On 2026-10-13 the first call of 0.2 failed with **402** "Your prepayment credits are depleted": a paid billing account in Prepay mode with a zero balance.
  - After ₪15 of prepay credit was added, all 21 calls of 0.2 passed (about 2 s each), which proves the paid tier.
  - Per [Google's billing docs](https://ai.google.dev/gemini-api/docs/billing), under Prepay the Google Cloud promotional credits apply only after funds are added, and not all Cloud credits cover the Gemini API. Whether the ₪30.71 of credits are used shows in the billing report (0.3).
- **Frame embeddings:** `local/wild_videos_embs_siglip/<video_id>.npy`, shape (T, 768), SigLIP 2, row t = frame at t s.
- **Compute:** no local GPU. Llama-3.1-8B runs on Kaggle (see `scripts/kaggle_forced_choice.py`, `scripts/prepare_kaggle_kernel.py`).

## Step 0: confirm paid API access (user + 21 tiny calls)

**0.1 User, in AI Studio / Cloud Console:**
- [ ] AI Studio → API keys: find the key used as `GEMINI_PAID_API_KEY`, note its project.
- [ ] Link that project to the billing account showing the credits ("Set up billing" / upgrade). Alternatively, create a new key in a project already on that billing account and update `~/.secrets`.
- [ ] Confirm the project now shows a paid tier (e.g. "Tier 1"), not "Free".
- [ ] Check whether the billing account has a card attached. If it does, usage beyond the credits would be charged to it. Optionally set a budget alert (e.g. $15) in Cloud Billing.

**0.2 Proof by volume: 21 tiny text-only calls (a fraction of a cent in total). Passed 2026-10-13.** The free tier allows 20 calls a day per model, so 21 successful calls in one day prove the paid tier. Use the same key and model, no video:

```bash
source ~/.secrets && .venv/bin/python -c "
import os
from google import genai
c = genai.Client(api_key=os.environ['GEMINI_PAID_API_KEY'])
for i in range(21):
    r = c.models.generate_content(model='gemini-3.8-flash', contents='Reply with the single word: ok')
    print(i, r.text, r.model_version, r.usage_metadata.total_token_count)
"
```

Gate: all 21 calls return `ok`.
- A 429 whose message contains `FreeTier` means the project is **still on the free tier**. Stop and go back to 0.1. The failed loop also uses up the day's free quota, which doesn't matter: nothing else should run until billing is fixed.
- A 402 "prepayment credits are depleted" means a paid tier in Prepay mode with a zero balance (the state on 2026-10-13). Add credits in AI Studio, or switch the billing account to Postpay, then re-run. Billing changes can take a while to reach the API.
- These calls bypass the runner, so they are not in `calls.jsonl`. Their cost is negligible.

**0.3 Proof by billing (next day).** Cloud Console → Billing → Reports, filtered to the project. Gate: a Generative Language API (Gemini) cost line appears, with an equal credit against it. Reports lag by several hours.
- If cost shows with **no** credit applied, the credits don't cover this SKU. Stop and tell the user before spending more.
- Compare the reported cost with the runner's estimate, `sum(usd)` in `calls.jsonl`, which assumes $0.50/M input and $3/M output. If they differ by more than 2×, correct `--usd-in`/`--usd-out` before Step 2.

**Also watch latency.** On the free tier, calls spent 1–7 minutes in 503 "high demand" retries. If the paid tier still shows median latency above about 60 s (field `secs` in `calls.jsonl`), raise it with the user. Options are a less loaded stable model (`gemini-3.7-flash`, `gemini-3.5-flash`) or the Batch API for Step 4.

## Step 1: redo Stage 1 (2 videos) without audio, then read and check

Commit the runner and prompt changes first. The outputs record the prompt hash, not the commit.

```bash
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py in_context \
    --videos Joe-Robinet_9-clip-2 4k-Relaxation_4-clip-4 --max-usd 0.5 --sleep 10
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py isolated \
    --videos Joe-Robinet_9-clip-2 4k-Relaxation_4-clip-4 --max-usd 0.5 --sleep 10
```

26 calls (2 in-context, 24 isolated), about $0.10. The runner stops on any 429; the last line of `calls.jsonl` has the quota id.

**1.1 Write `scripts/piece_caption_check.py`** (no API calls). For each video and caption set it reports:
- **SigLIP retrieval:**
  - Embed each piece's caption with `SiglipTextEmbedder("google/siglip2-base-patch16-224")` (`src/llm/local_embedder.py`), lowercased. Use the matching SigLIP 2 text tower only; the SigLIP 1 tower is not in the frames' space.
  - Embed each piece's frames as the mean of rows 5p…5p+4 of the frame embeddings.
  - Report the rank of the true piece among the 12 (1 = best) and the mean reciprocal rank (MRR).
- **Length:** words per caption, and the share of captions over 45 words or over 64 SigLIP tokens (truncated).
- **Order cues:** share of captions with the cue words, using the regex lists from the 2026-10-12 count (continuity / onset-progress / temporal adverbs).
- **On-screen text:** share of captions with digits, a clock or date pattern (e.g. `\d{1,2}:\d{2}`) or quoted text, each listed for reading. The prompt says not to transcribe subtitles, clocks, timestamps, dates or numbering; this checks that it worked.
- **Repetition:** mean pairwise MPNet cosine between a video's 12 captions. High means the captions repeat the setting.
- **Lead:** SigLIP similarity of caption p to piece p+1 minus to piece p−1. Above 0 means the captions run ahead into the next piece, as the original captions do by 1–2 s; for in-context captions that would be a timing leak.

Caption sets to compare:
- (a) **isolated** (3.8 Flash)
- (b) **in-context** (3.8 Flash)
- (c) **original**: the existing per-second captions from `datasets/wildQA/captions__wild{4,5}/`, joined per piece (seconds 5p…5p+4)
- (d) **with audio**, only where it exists: the old outputs in `results/piece_captions/_with_audio/` (both in-context captions, isolated Joe pieces 0–7). Set against (a) and (b), they show what the audio contributed. They also predate the on-screen text rule.

**1.2 Read all 48 piece captions by hand**, side by side, with frames for any disagreement. Extract frames with the static ffmpeg from `imageio_ffmpeg`, and use `-ss` per frame, not `select`+`fps`.

**Gate (user review):** show the user the check table and examples. Decide whether to tighten the prompt for "repeats the setting", hallucinated actions or leftover on-screen text.
- Any prompt change must be applied to **both** prompt files identically.
- Commit it, then re-run Stage 1 into a fresh output folder: rename the old folder, because the runner skips existing outputs.
- At most one revision round, then freeze. **Used on 2026-10-13: prompt v2** (see "Where things stand"), re-run into a fresh folder, then frozen.

## Step 2: 30-video pilot (390 calls, about $1.50)

```bash
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py in_context --pilot 30 --max-usd 3 --sleep 5
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py isolated   --pilot 30 --max-usd 3 --sleep 5
```

- `--pilot 30` picks 30 videos from 30 different channels, seed 0 (deterministic). Check the list first with `--dry-run`. Both Stage 1 videos are on it, so 364 of the 390 calls are new.
- Run in the background. The runner resumes if interrupted.
- **Time:** at Stage 1's free-tier latency (median about 100 s per call, mostly 503 retries), 364 calls in sequence take about 10 hours. Optional speed-up before running: add `--workers N` (a thread pool around the per-job body, with a lock on `calls.jsonl` and the spend check). Keep N ≤ 4 until the paid tier's rate limits are known.
- **Stop rules:**
  - any 429
  - more than 5% invalid outputs (`*.invalid.json`)
  - logged spend reaching the cap
- **Gate:** re-run `piece_caption_check.py` on all 30 videos, give the user a summary table, and spot-read about 20 random pieces.

## Step 3: unshuffle pilot (local / Kaggle, free)

Build under `scripts/unshuffle/`, with tests in `tests/`.

**3.1 Data assembly:** `build_pieces.py`. One JSON per video, holding for each of the 12 pieces:
- the caption under each caption set: isolated, in-context, in-context with cues removed, original, original with cues removed
- the frame-embedding rows

Cue removal is a fixed regex substitution (e.g. "continues to X" → "X"s, drop "again", "another" → "a"), applied the same way to in-context and original captions. Unit-test it on 20 hand-picked captions and commit the word list.

**3.2 Puzzles:** `make_puzzles.py`, seeded and saved to disk so every arm sees identical puzzles. The arms see shuffled piece labels only, never the true positions.
- **Contiguous-12:** all 12 pieces.
- **Contiguous-6:** pieces 0–5 and 6–11, two puzzles per video. The control for gapped-6: same number of pieces, no gaps.
- **Gapped-6:** pieces {0,2,4,6,8,10} and {1,3,5,7,9,11}, two puzzles per video. Edge frames are 6 s apart going forward and 14 s going backward, so edge matching is weakened, not removed; the SigLIP edge arm shows how much is left.
- 3 shuffles per puzzle. They matter only for arms that read the pieces in the order given (Claude). The chaining, pairwise and probe arms give the same answer for every shuffle, so run them once per puzzle.

**3.3 Arms.** Matched across modalities, so that a text-vs-frames difference cannot come from only one side having a reader:

| | No reader: similarity chaining | Learned order probe | Reader |
| --- | --- | --- | --- |
| Frames | SigLIP 2 edge chaining; mean-embedding chaining | probe on SigLIP 2 piece means | vision model on a subset (optional) |
| Text | MPNet caption chaining | probe on MPNet caption embeddings | Llama-3.1-8B pairwise; Claude later |

- **Random:** uniform permutations, for the chance level (τ = 0 in expectation).
- **Similarity chaining:**
  - Frames, edges: similarity of piece i's last frame to piece j's first frame. This is directed, so it carries an order direction.
  - Frames, means, and text: cosine between mean piece embeddings (SigLIP 2) or caption embeddings (MPNet, `all-mpnet-base-v2`). This is symmetric: a path and its reverse score the same.
  - Order by the best Hamiltonian path. With 12 pieces, use exact DP over subsets (2^12 × 12² is fine) rather than greedy.
  - **Direction for symmetric arms:** choose the path's direction by a seeded coin flip, not by the DP's tie-break, which can depend on piece labels. Such an arm knows nothing about direction, and its τ should show that; its local order is scored by undirected adjacency (3.4). Never report |τ|.
- **Learned order probe** (frames and text, same method): ridge regression from one piece's embedding to its position in the clip (0–11), trained leave-one-channel-out, with the regularization chosen inside the training folds. Pieces are ordered by predicted position, so one model serves every puzzle version. It sees one piece at a time and cannot use edge continuity: what it gets right is knowledge of what tends to come earlier or later. Train both modalities on the same videos.
- **Llama-3.1-8B pairwise scoring** (Kaggle):
  - For each ordered pair (i, j), compute log-probability of the captions in order i→j vs j→i, using the `prompts/dense_window/` style context framing.
  - Aggregate into a full order with the same subset DP, maximising the sum of pairwise log-odds along the path (adjacent pairs only, local), plus a Borda variant (all pairs, global). Name one as primary in 3.5.
  - About 132 pair scores per contiguous-12 puzzle.
  - Reuse the forced-choice Kaggle plumbing (`scripts/kaggle_forced_choice.py`).
- **Frames with a reader** (optional): a vision model reading a few frames per piece, on a subset. It costs money or quota; decide after the pilot.
- **Claude:** not in Step 3. Pilot it later with `scripts/blind_llm_runner.py`, `--limit` first, and ask the user before running.

**3.4 Metrics:** `score.py`.
- **Kendall τ** (global order), per puzzle, averaged per video.
- **Adjacent-pair accuracy** (local order): the share of true neighbour pairs (within the puzzle) that are neighbours in the predicted order, directed (right order) and undirected (either order). τ under-rewards local order: a prediction that gets two halves right but swaps them (pieces 6–11, then 0–5) has τ = −0.09 but directed adjacency 10/11.
- **Exact-position accuracy.**
- Pairwise order accuracy is not reported separately: for a full ordering it equals (τ + 1)/2.
- **Noise:** with no signal, per-puzzle τ has SD ≈ 0.22 for 12 pieces and ≈ 0.36 for 6, so per-video values are noisy. Compare arms on means over videos.
- **Reliability:** across videos, the correlation of τ between a video's two gapped-6 puzzles (and between its two contiguous-6 puzzles). They share no pieces, so this measures how stable a video's orderability is.
- Channel-cluster bootstrap CIs (40 channels; the `boot` function in `scripts/blind_recon_compare.py` does this: per-video mean, then resample channels).
- Paired differences between arms and between caption sets, on the same puzzles.

**3.5 Written up before looking at results:** commit the analysis plan below as `docs/new_direction/unshuffle_pilot_preregistration.md` before running the arms.
- **Primary metric and aggregation:** τ, with adjacency as secondary; for Llama, DP or Borda.
- **Context advantage:** τ(in-context) − τ(isolated) for Llama, paired by video. This is the leak plus whatever the captioner gets right from seeing the whole minute, so split it:
  - explicit cue words: τ(in-context) − τ(in-context, cues removed);
  - the rest, implicit leak plus accuracy: τ(in-context, cues removed) − τ(isolated), with each set's SigLIP retrieval MRR from `piece_caption_check.py` as the accuracy covariate.
- **Original captions, descriptive only:** τ(original) and τ(original, cues removed) next to τ(isolated). They differ from the new captions in model, in length (five per-second captions per piece) and in the 1–2 s lead, which puts the start of each next piece into the text. No decision rests on them.
- **Signal check:** each arm vs random on every puzzle version.
- **Main contrasts** (name one as primary):
  - text − frames within each matched method: similarity chaining (captions vs frame means, both symmetric; undirected adjacency) and the learned probe (τ);
  - Llama on isolated − SigLIP edge chaining, on contiguous-6 vs gapped-6 (same number of pieces). This interaction is expected from the design, because the edge arm relies only on continuity, so it cannot carry the claim alone;
  - contiguous-12 vs contiguous-6, which shows the effect of puzzle size alone.
- **Orderability:** τ of one pre-specified arm (or the mean of the reader arms), not the best τ over arms, which is biased upward. Report it per video only if the reliability above is adequate; otherwise per channel or event category.

**Gate (user review):** results table plus the decisions listed in `new_plan.md` ("Decisions made from the pilot"). Adjust piece length or gap here if the task has no signal or is saturated. A different piece length means re-captioning the pilot (about $1.50).

## Step 4: freeze and full run (after the user approves)

- Freeze the model, prompts, puzzle versions and arms: commit them, and record the commit hash in `docs/new_direction/`.
- Captions for all 335 videos: about 4,355 calls, about $15–20.
  - Check the remaining credits first; the user may need to redeem more from the $60.
  - `--max-usd` is cumulative, so set it to the total spend the log may reach (e.g. 25), not to this run's share.
  - At about 100 s per call, 4,000 calls in sequence take about 5 days. Use `--workers` or the Batch API (half price, but no per-call controls).
- Run the unshuffle arms on all 335 videos, with the probes trained leave-one-channel-out on all of them.
- The pilot videos helped tune the design, so report the main result on all 335 and, as a check, on the 305 outside the pilot.
- Then run Claude on a subset, which spends the subscription quota, so pilot with `--limit` and confirm with the user before the full subset.
- Update `docs/experiments/experiment_protocol_current.md` with the new standard.

## Rules that apply throughout

- **Cost and keys:**
  - Never use `GEMINI_API_KEY` (the free key) for this work.
  - Never print or log keys.
  - Keep `--max-usd` (a cumulative cap) just above what the next run needs, and raise it deliberately.
- **Visuals only:** captions are made from video with the audio removed (the runner does this). Don't caption with audio.
- **Commits:**
  - Commit prompts, scripts and pre-registrations **before** the runs that use them.
  - Results stay in gitignored `results/`.
  - A real `src/main.py` run fails on a dirty tree. The new scripts don't care, but commit anyway for reproducibility.
- **Statistics:** cluster all statistics by channel (`movie_id`, the video id prefix before `_<n>-clip`).
- **Name mismatch:** normalize `Olly's-Farm` (captions) vs `Olly_s-Farm` (files) when joining.
- **Staging:** don't scale up past a gate without the user. Every stage ends with a short report: what ran, cost, failures, and the numbers the gate asks for.
