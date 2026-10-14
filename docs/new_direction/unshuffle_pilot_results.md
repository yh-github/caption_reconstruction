# Unshuffle pilot results (Gate 1)

Written 2026-10-14. Development data: the free arms ran on the 30 pilot videos, and the Claude calibration on the first 10 of them. The analysis follows `unshuffle_pilot_preregistration.md`; its "Deviations" section lists the two implementation fixes made after the first run. Tables: `results/unshuffle/analysis/*.csv`, from `scripts/unshuffle/analyze.py`. Every CI is a 95% channel-cluster bootstrap; in the pilot each channel has one video.

**Headline: the pilot contradicts the plan's central bet.** For every Claude model, frames give more order than captions, in **direction** as well as in structure. H2 ("captions carry direction better") fails on contiguous puzzles and with 5 s gaps. The frames' advantage shrinks as gaps grow, which points to a different, testable thesis (§5).

## 1. What ran

| Run | Size | Validity | Cost |
|---|---|---|---|
| Free arms (no reader, learned): 34 arms × 420 puzzles × 2 variants | 30 videos | all | local, 4 min |
| Claude smoke test (Haiku) | 6 calls | 6/6 | ≈ 0 |
| Claude calibration: {P12, G6 even} × 2 shuffles × {frames, isolated captions} | 10 videos, 80 calls per model | **240/240 valid on the first try**; one turn, no tools, the pinned model reported on every call | quota (§4) |

## 2. Results

### 2.1 Structure from the free arms (30 videos)

Undirected adjacency: the share of true neighbours that the predicted order puts next to each other (chance: 0.17 on P12, 0.33 on 6 pieces, 0.50 on 4).

| Puzzle (gap) | Frames: chaining | Frames: edges | Captions: chaining (SigLIP 2 text) | Frames − captions, chaining |
|---|---|---|---|---|
| P12 (0 s) | 0.68 | 0.81 | 0.45 | **+0.23 [+0.16, +0.30]** |
| C6 (0 s) | 0.73 | 0.82 | 0.53 | +0.20 [+0.14, +0.26] |
| K4 d = 1 (0 s) | 0.79 | 0.85 | 0.63 | +0.16 [+0.09, +0.22] |
| G6 (5 s) | 0.54 | 0.56 | 0.49 | +0.05 [−0.02, +0.12] |
| K4 d = 2 (5 s) | 0.64 | 0.65 | 0.59 | +0.06 [−0.03, +0.14] |
| K4 d = 3 (10 s) | 0.64 | 0.60 | 0.62 | +0.02 [−0.06, +0.09] |

- **H1 holds** in the pilot: pixels carry more structure than captions when pieces touch, and gaps remove the advantage. MPNet captions give the same picture. The result holds without the text-tagged pieces (P12: +0.20 [+0.10, +0.29]).
- **Prop. 2 gate passes:** all 11 symmetric chaining arms have local direction 0.47–0.53, with every CI including 0.5.
- **The learned arms are at chance** (|τ| ≤ 0.13, CIs include 0). With 29 training videos per fold that was likely; they need the full set.

### 2.2 Claude reads order from frames far better than from captions (10 videos)

Same puzzles, same shuffles. τ is Kendall's τ (1 = perfect, 0 = chance). Local direction: of the true neighbour pairs placed next to each other, the share in the right order (0.5 = no sense of direction).

| P12 (12 pieces, no gaps) | τ frames | τ captions | adjacency frames | adjacency captions | local direction frames | local direction captions |
|---|---|---|---|---|---|---|
| Haiku | +0.37 | +0.16 | 0.65 | 0.41 | 0.76 | 0.66 |
| Sonnet | **+0.67** | +0.18 | 0.83 | 0.44 | 0.94 | 0.65 |
| Opus | **+0.76** | +0.26 | 0.90 | 0.53 | 0.96 | 0.77 |
| Edges (continuity only, no reader) | +0.60 | | 0.86 | | 0.93 | |

| G6 (6 pieces, 5 s gaps) | τ frames | τ captions | adjacency frames | adjacency captions | local direction frames | local direction captions |
|---|---|---|---|---|---|---|
| Haiku | +0.07 | −0.13 | 0.49 | 0.52 | 0.75 | 0.23 |
| Sonnet | +0.24 | +0.15 | 0.58 | 0.50 | 0.89 | 0.56 |
| Opus | **+0.53** | +0.17 | 0.62 | 0.52 | 0.79 | 0.47 |
| Edges (continuity only, no reader) | +0.23 | | 0.64 | | 0.78 | |

Paired frames − captions (same puzzle and shuffle):

| | τ, P12 | τ, G6 | adjacency, P12 | local direction, P12 | local direction, G6 |
|---|---|---|---|---|---|
| Haiku | +0.21 [+0.02, +0.40] | +0.20 [−0.10, +0.53] | +0.25 [+0.16, +0.33] | +0.10 [−0.00, +0.19] | +0.49 [+0.21, +0.71] |
| Sonnet | +0.50 [+0.20, +0.74] | +0.09 [−0.32, +0.43] | +0.40 [+0.29, +0.49] | +0.29 [+0.19, +0.40] | +0.33 [+0.08, +0.56] |
| Opus | +0.50 [+0.15, +0.80] | +0.35 [+0.11, +0.59] | +0.37 [+0.24, +0.49] | +0.19 [+0.10, +0.28] | +0.30 [−0.08, +0.63] |

What this shows:
- **Frames beat captions for every model,** in direction as well as in structure. Captions do carry some direction for readers (local direction 0.65–0.77 on P12, above the 0.5 of symmetric arms), but much less than frames.
- **A large part of the frames' direction needs no knowledge of events.** The edge arm, which only compares the last frame of one piece with the first frame of another, gets 0.93 local direction on P12 and 0.78 on G6. The frames inside each piece are in time order, so continuity across the boundary reveals which piece comes first. Sonnet and Opus match this on P12 and go beyond it on G6 (Opus τ +0.53 against the edge arm's +0.23).
- **Scale helps frames much more than captions,** the reverse of H6. Paired, frames: Sonnet − Haiku +0.31 τ on P12; Opus − Sonnet +0.29 τ on G6 [+0.07, +0.53]. Captions: Opus − Sonnet +0.08 [−0.12, +0.27] on P12, +0.02 on G6.
- **Position bias is negligible** (mean τ with the presentation order between −0.07 and +0.11). The two shuffles still give different orders for 70–90% of puzzles, so the second shuffle reduces noise.

### 2.3 Leak (E5 preview, free arms)

- A learned position probe finds position information in whole-video captions but none in isolated ones: τ +0.15 [+0.07, +0.24] (in-context, SigLIP 2 text) against −0.06 [−0.16, +0.05] (isolated); paired difference **+0.21 [+0.10, +0.32]**.
- **Removing cue words does not remove it** (in-context − in-context without cues: +0.01 [−0.01, +0.04]). The leak is implicit: what the captioner chose to say, not words like "continues". The cue rewrite works; in-context captions with a cue word drop from 107 to 6 of 360.
- In-context captions are *worse* for similarity chaining (adjacency 0.30 against 0.45 isolated, P12): describing the minute as a whole makes the pieces sound alike.

### 2.4 Prop. 3 (continuity profiles)

- **Captions:** the continuity model predicts chaining accuracy with the pre-registered per-piece effect. Predicted − observed CIs include 0 on every puzzle set (e.g. MPNet P12 −0.01 [−0.05, +0.03]), and the prediction tracks individual videos (r = 0.3–0.8).
- **Frames:** calibrated with gaps (G6 −0.03 [−0.08, +0.03], K4 d = 3 −0.02 [−0.06, +0.04]), but contiguous frames chain better than the model predicts (P12 −0.17 [−0.22, −0.12]). Likely cause: shot structure. Inside one shot, neighbouring pieces are far more alike than a single, stationary decay curve with independent noise allows. The theory section needs a shot-aware version for contiguous frames.
- **Mean profiles:** frames start far more continuous than captions (κ at 5 s: 0.92 against 0.59–0.66 with the piece effect) and lose more of it by 10 s (0.63 against 0.46–0.49). This is the mechanism behind §2.1.

## 3. The hypotheses after the pilot

| | Pilot verdict |
|---|---|
| H1 structure | **Supported**, including the gap interaction |
| H2 direction: captions > frames | **Contradicted** at 0 s and 5 s gaps, for Haiku, Sonnet and Opus. Untested at larger gaps. |
| H3 verbalization | Not run. After §2.2, "describe, then order" may well *lower* accuracy; that is now the more interesting outcome. |
| H4 where language wins | Not run (needs E6 and E8) |
| H5 leak | Supported for learned probes; the part that survives cue removal is all of it |
| H6 scale | Reversed: frames gain with scale, captions barely |

## 4. Quota (Claude subscription)

Measured per call. The 5-hour and 7-day figures are account-wide and reported as whole percents, so the 7-day numbers are rough.

| | Input tokens (P12 frames / captions) | 5-hour points per call | 7-day points per 80 calls |
|---|---|---|---|
| Haiku | 10,000 / 1,940 | not measurable (this session's own use dominated) | ≈ 1–2 |
| Sonnet | 10,000 / 1,940 | **0.175** (80 calls: 58% → 72%, little else running) | ≈ 1 |
| Opus | 10,000 / 1,940 | **0.18–0.19** (36 calls: +7; 44 calls: +8) | ≈ 2 |

- Opus costs about the same as Sonnet per call in the 5-hour window. It writes much less (median 45–450 output tokens against 215–740), which makes up for its higher price. In the 7-day figure it costs about twice as much.
- **E1 on 335 videos** (5 puzzles × 2 shuffles × 2 modalities = 6,700 calls): Sonnet ≈ 1,200 points of the 5-hour window (≈ 15 windows at 80%) and ≈ 85 points of the 7-day figure. Opus about the same in 5-hour windows and ≈ 170 points of the 7-day figure, i.e. close to two weeks of the full weekly quota.

## 5. Proposed reframing and next steps (decisions for Gate 1)

**Reframing.** A thesis consistent with the pilot, and with the literature where language wins (SPLICE, TempVS: separated steps, short human labels):
> *Pixels order events through continuity, and continuity has a range. Within it, frames beat words in structure and in direction, for every reader. Beyond it, only knowledge of how events unfold is left, and that is where language can compete.*

It keeps the information control (Prop. 1), the structure/direction split (Prop. 2) and the knob (Prop. 3, now measured and partly calibrated). It turns the knob from a robustness check into the main experiment: the frames − captions difference as a function of the gap, per reader. It also explains the apparent conflict with SPLICE: their clips are separated steps (large gaps); ours were contiguous.

**Decisions:**
1. **Main reader.** The pre-registered rule picks the strongest model whose frame τ is above 0 and whose full run fits the quota. Both qualify, and Opus is stronger on frames (G6 +0.29 τ over Sonnet). It costs the same per call in the 5-hour window but about twice as much in the 7-day figure. *Recommendation:* Opus as the main reader for the gap-knob experiment and the 7-day budget it needs; Sonnet and Haiku as the scale points.
2. **Next pilot: the gap knob with readers** (before any held-out run). Add a 3-piece set with 20 s gaps (K3 d = 5: pieces {0, 5, 10} and {1, 6, 11}) to reach beyond the edge arm's range. Then run frames vs. captions on K4 d = 1, 2, 3 and K3 d = 5 for the 30 pilot videos, with 1 shuffle: 30 × 11 puzzles × 2 modalities = 660 Opus calls, ≈ 120 points of the 5-hour window (two windows) and ≈ 15 of the 7-day figure. The question it answers: does the frames − captions difference in τ and in local direction shrink to 0, or reverse, once the edge arm is at chance?
3. **Redesign E2 (reversal test).** Reversing the order of pieces while the frames inside each piece still run forward leaves the boundaries discontinuous, so continuity alone solves it at small gaps. Run it at the knob's gap levels, with the edge arm as the continuity-only reference. Optionally add a full-reversal condition (frames reversed too), the classic arrow-of-time test (Pickup et al. 2014; Xue, Luo and Grauman 2025).
4. **Keep 2 shuffles in the confirmatory run** (answers change between shuffles), or accept a wider CI with 1.
5. **Literature, second pass, now targeted:** caption-then-reason video pipelines (LLoVi, VideoAgent, Vamos) and what they lose; arrow-of-time and motion-extrapolation work; why SPLICE's and TempVS's readers did poorly on images where Claude does well on frames (single images vs. 5-frame pieces).

The plan's thesis (`paper_plan.md` §1) is unchanged until the user and partners decide on the reframing.
