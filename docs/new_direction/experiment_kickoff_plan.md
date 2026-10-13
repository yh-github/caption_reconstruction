# Experiment kickoff plan: from paid-API check to the unshuffle pilot

Written 2026-10-13. Companion to `docs/new_direction/new_plan.md` (the why and the design). This file is the how: concrete steps, commands, checks and stop rules, in order. Each step has a gate; do not start the next step until the gate passes and, where marked, the user has looked at the results.

## Where things stand

- **Prompts (committed):** `prompts/yt_video_processing/piece_isolated.txt` and `piece_in_context.txt`.
- **Runner (committed):** `scripts/caption_pieces.py`.
  - Defaults: `gemini-3.8-flash`, temperature 0.8, dynamic thinking, seed 0x5EED, 1 fps.
  - Auth: reads only `GEMINI_PAID_API_KEY` (exported in `~/.secrets`, sourced by `~/.bashrc`).
  - Outputs: `results/piece_captions/<model>/<mode>/`; call log with token usage and estimated cost: `results/piece_captions/calls.jsonl`.
- **Stage 1 so far:** in-context done for `Joe-Robinet_9-clip-2` and `4k-Relaxation_4-clip-4`; isolated done for Joe pieces 0–7. Missing: Joe pieces 8–11, all 12 pieces of 4k-Relaxation.
- **Blocker:** the key's project was on Google's **free tier**. The 429 on 2026-10-12 named `generate_content_free_tier_requests`, limit 20, quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. The ₪30.71 (~$10) of credits are on a billing account that the key's project is not linked to.
- **Frame embeddings:** `local/wild_videos_embs_siglip/<video_id>.npy`, shape (T, 768), SigLIP 2, row t = frame at t s.
- **Compute:** no local GPU. Llama-3.1-8B runs on Kaggle (see `scripts/kaggle_forced_choice.py`, `scripts/prepare_kaggle_kernel.py`).

## Step 0: confirm paid API access (user + 1 tiny call)

**0.1 User, in AI Studio / Cloud Console:**
- [ ] AI Studio → API keys: find the key used as `GEMINI_PAID_API_KEY`, note its project.
- [ ] Link that project to the billing account showing the credits ("Set up billing" / upgrade). Alternatively, create a new key in a project already on that billing account and update `~/.secrets`.
- [ ] Confirm the project now shows a paid tier (e.g. "Tier 1"), not "Free".
- [ ] Check whether the billing account has a card attached. If it does, usage beyond the credits would be charged to it. Optionally set a budget alert (e.g. $15) in Cloud Billing.

**0.2 One tiny text-only call (costs a fraction of a cent).** Use the same key and model, no video:

```bash
source ~/.secrets && .venv/bin/python -c "
import os
from google import genai
c = genai.Client(api_key=os.environ['GEMINI_PAID_API_KEY'])
r = c.models.generate_content(model='gemini-3.8-flash', contents='Reply with the single word: ok')
print(r.text, r.model_version, r.usage_metadata)
"
```

Gate: it returns `ok` without a 429.

On its own this does not prove the paid tier: the free tier also allows 20 calls a day. The proof comes from 0.3 and 0.4.

**0.3 Proof by volume, done as part of Step 1.** Step 1 makes 16 calls. Together with 0.2 and anything else made that day, the day's total for `gemini-3.8-flash` passes 20.
- If any call returns a 429 whose message contains `FreeTier`, the project is **still on the free tier**. Stop and go back to 0.1.
- The runner stops on any 429 automatically. Check the last line of `calls.jsonl` for the quota id.

**0.4 Proof by billing (next day).** Cloud Console → Billing → Reports, filtered to the project. Gate: a Generative Language API (Gemini) cost line appears, with an equal credit against it. Reports lag by several hours.
- If cost shows with **no** credit applied, the credits don't cover this SKU. Stop and tell the user before spending more.
- Compare the reported cost with the runner's estimate, `sum(usd)` in `calls.jsonl`, which assumes $0.50/M input and $3/M output. If they differ by more than 2×, correct `--usd-in`/`--usd-out` before Step 2.

**Also watch latency.** On the free tier, calls spent 1–7 minutes in 503 "high demand" retries. If the paid tier still shows median latency above about 60 s (field `secs` in `calls.jsonl`), raise it with the user. Options are a less loaded stable model (`gemini-3.7-flash`, `gemini-3.5-flash`) or the Batch API for Step 4.

## Step 1: finish Stage 1 (2 videos), then read and check

```bash
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py isolated \
    --videos Joe-Robinet_9-clip-2 4k-Relaxation_4-clip-4 --max-usd 0.5 --sleep 10
```

The runner skips the 8 pieces already done, so this makes 16 calls.

**1.1 Write `scripts/piece_caption_check.py`** (no API calls). For each video and caption set it reports:
- **SigLIP retrieval:**
  - Embed each piece's caption with `SiglipTextEmbedder("google/siglip2-base-patch16-224")` (`src/llm/local_embedder.py`), lowercased. Use the matching SigLIP 2 text tower only; the SigLIP 1 tower is not in the frames' space.
  - Embed each piece's frames as the mean of rows 5p…5p+4 of the frame embeddings.
  - Report the rank of the true piece among the 12 (1 = best) and the mean reciprocal rank (MRR).
- **Length:** words per caption, and the share of captions over 45 words or over 64 SigLIP tokens (truncated).
- **Order cues:** share of captions with the cue words, using the regex lists from the 2026-10-12 count (continuity / onset-progress / temporal adverbs).
- **Repetition:** mean pairwise MPNet cosine between a video's 12 captions. High means the captions repeat the setting.

Caption sets to compare:
- (a) **isolated** (3.8 Flash)
- (b) **in-context** (3.8 Flash)
- (c) **original**: the existing per-second captions from `datasets/wildQA/captions__wild{4,5}/`, joined per piece (seconds 5p…5p+4)

**1.2 Read all 48 piece captions by hand**, side by side, with frames for any disagreement. Extract frames with the static ffmpeg from `imageio_ffmpeg`, and use `-ss` per frame, not `select`+`fps`.

**Gate (user review):** show the user the check table and examples. Decide whether to tighten the prompt for "repeats the setting" or hallucinated actions.
- Any prompt change must be applied to **both** prompt files identically.
- Commit it, then re-run Stage 1 into a fresh output folder: rename the old folder, because the runner skips existing outputs.
- At most one revision round, then freeze.

## Step 2: 30-video pilot (390 calls, about $1.50)

```bash
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py in_context --pilot 30 --max-usd 3 --sleep 5
source ~/.secrets && .venv/bin/python scripts/caption_pieces.py isolated   --pilot 30 --max-usd 3 --sleep 5
```

- `--pilot 30` picks 30 videos from 30 different channels, seed 0 (deterministic). Check the list first with `--dry-run`.
- Run in the background. The runner resumes if interrupted.
- Optional speed-up before running: add `--workers N` (a thread pool around the per-job body, with a lock on `calls.jsonl` and the spend check). Keep N ≤ 4 until the paid tier's rate limits are known.
- **Stop rules:**
  - any 429
  - more than 5% invalid outputs (`*.invalid.json`)
  - logged spend reaching the cap
- **Gate:** re-run `piece_caption_check.py` on all 30 videos, give the user a summary table, and spot-read about 20 random pieces.

## Step 3: unshuffle pilot (local / Kaggle, free)

Build under `scripts/unshuffle/`, with tests in `tests/`.

**3.1 Data assembly:** `build_pieces.py`. One JSON per video, holding for each of the 12 pieces:
- the caption under each caption set: isolated, in-context, original, original with cues removed
- the frame-embedding rows

Cue removal is a fixed regex substitution (e.g. "continues to X" → "X"s, drop "again", "another" → "a"). Unit-test it on 20 hand-picked captions and commit the word list.

**3.2 Puzzles:** `make_puzzles.py`, seeded and saved to disk so every arm sees identical puzzles.
- Contiguous: all 12 pieces.
- Gapped: pieces {0,2,4,6,8,10} and {1,3,5,7,9,11} as two puzzles per video.
- 3 shuffles per puzzle.

**3.3 Arms:**
- **Random:** uniform permutations, for the chance level.
- **SigLIP edge chaining:**
  - Similarity of piece i's last frame to piece j's first frame.
  - Order by the best Hamiltonian path. With 12 pieces, use exact DP over subsets (2^12 × 12² is fine) rather than greedy.
  - Also run a variant using mean piece embeddings instead of edges.
- **Llama-3.1-8B pairwise scoring** (Kaggle):
  - For each ordered pair (i, j), compute log-probability of the captions in order i→j vs j→i, using the `prompts/dense_window/` style context framing.
  - Aggregate into a full order with the same subset DP, maximising the sum of pairwise log-odds along the path, plus a Borda variant.
  - About 132 pair scores per contiguous puzzle.
  - Reuse the forced-choice Kaggle plumbing (`scripts/kaggle_forced_choice.py`).
- **Claude:** not in Step 3. Pilot it later with `scripts/blind_llm_runner.py`, `--limit` first, and ask the user before running.

**3.4 Metrics:** `score.py`.
- Kendall τ, pairwise order accuracy and exact-position accuracy, per puzzle, averaged per video.
- Channel-cluster bootstrap CIs (40 channels; the `boot` function in `scripts/blind_recon_compare.py` does this: per-video mean, then resample channels).
- Paired differences between arms and between caption sets, on the same puzzles.

**3.5 Written up before looking at results:** commit the analysis plan below as `docs/new_direction/unshuffle_pilot_preregistration.md` before running the arms.
- **Leak:** τ(in-context) − τ(isolated) for Llama, paired by video.
- **Cleanup sufficiency:** τ(original with cues removed) vs τ(isolated).
- **Signal check:** each arm vs random on both puzzle versions.
- **Main contrast:** τ(SigLIP chaining) − τ(Llama on isolated), contiguous vs gapped.

**Gate (user review):** results table plus the decisions listed in `new_plan.md` ("Decisions made from the pilot"). Adjust piece length or gap here if the task has no signal or is saturated.

## Step 4: freeze and full run (after the user approves)

- Freeze the model, prompts, puzzle versions and arms: commit them, and record the commit hash in `docs/new_direction/`.
- Captions for all 335 videos: about 4,355 calls, about $15–20. Check the remaining credits first; the user may need to redeem more from the $60.
- Run the unshuffle arms on all 335 videos. Then run Claude on a subset, which spends the subscription quota, so pilot with `--limit` and confirm with the user before the full subset.
- Update `docs/experiments/experiment_protocol_current.md` with the new standard.

## Rules that apply throughout

- **Cost and keys:**
  - Never use `GEMINI_API_KEY` (the free key) for this work.
  - Never print or log keys.
  - Keep `--max-usd` low per run and raise it deliberately.
- **Commits:**
  - Commit prompts, scripts and pre-registrations **before** the runs that use them.
  - Results stay in gitignored `results/`.
  - A real `src/main.py` run fails on a dirty tree. The new scripts don't care, but commit anyway for reproducibility.
- **Statistics:** cluster all statistics by channel (`movie_id`, the video id prefix before `_<n>-clip`).
- **Name mismatch:** normalize `Olly's-Farm` (captions) vs `Olly_s-Farm` (files) when joining.
- **Staging:** don't scale up past a gate without the user. Every stage ends with a short report: what ran, cost, failures, and the numbers the gate asks for.
