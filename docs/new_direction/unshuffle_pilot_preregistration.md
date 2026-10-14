# Unshuffle pilot: analysis plan (written before any arm runs)

Written 2026-10-14, before any ordering arm has run on any video. It fixes what the pilot measures and how it is read, so that the pilot cannot be tuned to its own results. It covers kickoff plan §3.5 plus the pilot parts of E2–E5 and the Claude calibration in `paper_plan.md` §5. The pilot is **development data**: nothing here is a confirmatory claim. The confirmatory pre-registration for the 305 held-out videos (`unshuffle_preregistration.md`) is written after Gate 1 and committed before any arm runs on them.

## 1. Data

- **Videos:** the 30 pilot videos (`caption_pieces.select_pilot(30, seed 0)`), one per channel. Every statistic is clustered by channel; in the pilot that is one video per cluster.
- **Pieces:** 12 pieces of 5 s per video, piece p = seconds [5p, 5p + 5).
- **Caption sets** (Gemini 3.8 Flash, frozen prompts v2, no audio):
  - `isolated`: one call per piece, the piece only (main set);
  - `in_context`: one call per video, all 12 pieces described in one answer;
  - `in_context_nocue` and `original_nocue`: the same with order cue words removed by a fixed rule (`scripts/unshuffle/cues.py`, unit-tested, committed before use);
  - `original`: the benchmark's per-second captions, seconds 5p..5p+4 joined (descriptive only; it differs in model, length and the 1–2 s lead).
- **Frames:**
  - for the no-reader and learned arms, SigLIP 2 rows 5p..5p+4 of `local/wild_videos_embs_siglip/<video>.npy` (row t = the frame at t s, checked 2026-10-07);
  - for readers, the 5 frames the captioner saw: each piece cut exactly as `caption_pieces.cut` cuts it, sampled at 1 fps, saved as JPEG 448 px wide.

## 2. Puzzles

Built once by `scripts/unshuffle/puzzles.py`, saved to `results/unshuffle/puzzles.json`, and shared by every arm. Arms see neutral labels only.

| Set | Pieces | Puzzles per video | Gap between consecutive pieces |
|---|---|---|---|
| P12 | 0–11 | 1 | 0 s |
| C6 | 0–5; 6–11 | 2 | 0 s |
| G6 | 0,2,…,10; 1,3,…,11 | 2 | 5 s |
| K4, d = 1 | s, s+1, s+2, s+3 for s ∈ {0, 4, 8} | 3 | 0 s |
| K4, d = 2 | s, s+2, s+4, s+6 for s ∈ {0, 3, 5} | 3 | 5 s |
| K4, d = 3 | s, s+3, s+6, s+9 for s ∈ {0, 1, 2} | 3 | 10 s |

- **Shuffles (listwise readers only):** 2 per puzzle. Shuffle 1 is a seeded uniform permutation; shuffle 2 presents the pieces in the reverse of shuffle 1's order. Letters are given in presentation order (A is shown first). A reader that echoes the presentation order therefore scores τ = 0 exactly on the pair of shuffles, so position bias cannot raise the mean.
- **Robustness set:** each puzzle again without the pieces tagged `quotes_text` in `results/piece_captions/check/text_tags.csv` (isolated set: 44 of 360 pieces, 11 videos). No-reader and learned arms are re-run on the reduced puzzles. For readers in the pilot, the reader's order restricted to the untagged pieces is scored instead (exploratory: the reader still saw the tagged pieces).

## 3. Arms

Matched across modalities: every frame arm has a caption arm built the same way.

| Level | Frames | Captions |
|---|---|---|
| Chance | uniform random order: τ = 0, undirected adjacency 2/n, directed 1/n, exact position 1/n (computed, not sampled) | same |
| No reader, symmetric | SigLIP 2 piece-mean chaining | caption chaining: SigLIP 2 text tower; MPNet (`all-mpnet-base-v2`) |
| No reader, directed | SigLIP 2 edge chaining: last frame of piece i to first frame of piece j | none (captions have no edges) |
| Learned | position probe and pairwise order model on SigLIP 2 piece means | the same on SigLIP 2 text and on MPNet caption embeddings |
| Reader (listwise) | Claude Haiku, Sonnet, Opus on frames (calibration subset only, §5) | the same on isolated captions |

- **Chaining:** the order is the best Hamiltonian path through the pieces (exact DP over subsets): maximum total cosine similarity between consecutive pieces. Symmetric arms choose the path's direction by a seeded coin flip, never by the DP's tie-break. The edge arm's similarity is directed, so its path has a direction.
- **Position probe:** ridge regression from one piece's embedding to its relative position p / 11, trained leave-one-channel-out (in the pilot: leave-one-video-out on the other 29 videos), with α chosen by 5-fold group cross-validation inside the training videos from {0.1, 1, 10, 100, 1000}. Pieces are ordered by predicted position.
- **Pairwise order model:** logistic regression on the difference of two pieces' embeddings, trained on all ordered pairs within the training videos (same folds and α grid, C = 1/α). Its log-odds are aggregated by the exact linear-ordering optimum (subset DP maximising the summed log-odds of all pairs placed in order; Kemeny-style). On a linear model this equals sorting by the learned score, so it is the pairwise twin of the position probe.
- **Embeddings are unit-normalised** before similarities and probes. Piece means are re-normalised.
- **Kaggle pairwise readers** (Llama-3.1-8B on captions, Qwen2.5-VL-7B on frames and captions) are pre-registered in `paper_plan.md` and analysed the same way when they exist; they are not part of this run.

## 4. Metrics

Per puzzle, then averaged per video (and per puzzle set):
- **Kendall τ** between the predicted and true order (primary).
- **Structure:** undirected adjacency accuracy, the share of true neighbour pairs (neighbours within the puzzle) that are neighbours in the prediction.
- **Directed adjacency accuracy:** the same, in the right order.
- **Local direction:** directed / undirected adjacency, over puzzles with at least one undirected hit. Symmetric arms have expectation 0.5 exactly (Prop. 2): **gate**, |mean − 0.5| must not be significant for them.
- **Exact-position accuracy.**
- Pairwise accuracy is not reported (it equals (τ + 1)/2).
- **Listwise readers only:** invalid rate (after one retry), position bias (τ between the predicted order and the presentation order), input and output tokens, seconds, and quota used per call (5-hour and 7-day figures reported with each call).

Inference: per-video means; 95% CIs by channel-cluster bootstrap (10,000 resamples; the `boot` of `scripts/blind_recon_compare.py`); paired differences on the same puzzles. The pilot reports CIs as estimates of effect size and noise for planning, not as tests.

## 5. Claude calibration (choosing the main reader)

- **Smoke test first:** Haiku on 3 puzzles × frames and captions (6 calls). Stop and fix if any call is not single-turn, has tools, fails to parse, or reports a model other than the pinned one.
- **Calibration set:** the first 10 videos of the seeded pilot order × {P12, G6 even pieces} × 2 shuffles × {frames, isolated captions} = 80 calls per model, for `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`, effort medium, prompts `prompts/unshuffle/` (committed before the run). Stop at 80% of the 5-hour window.
- **Decision rule:** the main reader is the strongest model whose mean τ on frames has a bootstrap CI above 0 and whose full run (E1 + E2 on 335 videos, ≈ 10k calls, extrapolated from the measured 7-day use per call) fits the quota the user agrees to. If Sonnet's frame τ CI includes 0 while Opus's does not, Opus becomes the main reader on as many videos as the quota allows (at least 150). If a model's caption τ exceeds 0.9 on G6, harder puzzles (K4 d = 3) carry the comparison, not a weaker model. Opus runs on at least 100 videos regardless (E7).
- **Reported:** per model × modality × puzzle set, all §4 metrics; frames − captions paired by puzzle; quota per call.

## 6. Analyses (all descriptive in the pilot)

1. **Signal check:** every arm against chance, per puzzle set.
2. **Structure (H1 preview):** undirected adjacency, captions vs. frame means, for the symmetric chaining arms on P12, C6, G6 and K4; frames edge chaining alongside.
3. **Direction (H2 preview):** local direction and τ for the learned arms and the Claude readers, frames vs. captions. Symmetric arms must sit at 0.5 (gate, §4).
4. **Puzzle size:** C6 vs. P12 at the same pieces.
5. **Continuity profiles (Prop. 3):** per video and modality (SigLIP 2 frame means, SigLIP 2 text, MPNet), the mean similarity ρ(δ) of piece pairs δ = 5, 10, …, 55 s apart and the residual SD σ (pooled over δ); κ(Δ) = (ρ(Δ) − ρ(2Δ)) / σ. Monte Carlo under the model (similarities = ρ(δ) + i.i.d. normal noise of SD σ, 2,000 draws) gives each video's predicted undirected adjacency for each puzzle set under the chaining DP. **Calibration:** the correlation across videos between predicted and observed adjacency, and the mean predicted − observed, per modality and set. If the i.i.d. model is miscalibrated, a per-piece random effect is added and reported as such.
6. **Leak (E5 preview):** the no-reader and learned caption arms on `in_context`, `in_context_nocue`, `isolated`, `original`, `original_nocue`, paired by video: τ(in-context) − τ(isolated), split into cues (in-context − in-context without cues) and the rest; each set's SigLIP retrieval MRR from `piece_caption_check.py` alongside.
7. **Robustness:** analyses 1–3 without text-tagged pieces (§2).
8. **Power:** per-video SDs of the main paired contrasts → projected 95% CI half-width with 305 videos and 40 channels (target ≤ 0.04 τ), and the number of videos needed if the target is missed.

## 7. What the pilot decides (Gate 1)

- The main reader and the Opus subset (§5).
- Whether P12 is saturated or at floor for any arm; whether K4 has a usable gradient across d.
- Whether the continuity model is calibrated enough to carry Prop. 3's prediction.
- The primary contrasts, margins and sample sizes of the confirmatory pre-registration.

Changes to this plan after any arm has run are listed in a dated "Deviations" section at the end of this file, with the reason.
