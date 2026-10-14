# Paper plan: where the order of events comes from, words or pixels

Written 2026-10-14, after the first-pass literature review (`related_work.md`) and the go for paid Gemini and Claude Code runs. This is the master plan for the paper: the claims, the theory, the experiments, the statistics, the budget and the order of work.

**Pilot results (2026-10-14, `unshuffle_pilot_results.md`):** H1 holds, but H2 fails at 0 s and 5 s gaps (frames beat captions in direction too, for every Claude model) and H6 is reversed. A reframing around the range of visual continuity is proposed there (§5); §1–§3 below are unchanged until the user and partners decide.

`new_plan.md` remains the why (generation → unshuffling) and the captioning design. `experiment_kickoff_plan.md` remains the commands and gates for captioning and the free arms. Where they conflict with this file, this file wins.

## 0. What changed and why

| Directive (2026-10-14) | What it changes |
|---|---|
| Literature review for novelty, pivot if needed | Done (first pass). There is no direct threat, but "language orders better than vision" is already known for models (TempVS, StripCipher, SPLICE, Wu et al. 2022). **Pivot:** from *how much* order each modality recovers to *why*. The paper rests on an information control that only our design has (Prop. 1), and on splitting order into continuity and direction (Props. 2–3). |
| Go for paid Gemini and Claude Code | Vision readers move from optional to central. The **same model reads frames or captions** (Claude Haiku, Sonnet and Opus through `claude -p`, which accepts images: probed 2026-10-14). That is the cleanest language-vs-perception test available. |
| Rigorous theory + top-tier experiments | Three propositions with predictions, and a pre-registered confirmatory test on a held-out set (the 305 videos outside the pilot). Matched arms at three levels (no reader, learned, reader), an open-weights replication, a second, script-rich dataset, a human baseline and a robustness battery. |
| Insight into language vs. visual perception | Three experiments aimed at it: verbalization inside one model (E4), structure vs. direction (E1–E3), and which videos language wins on (E6), plus humans (E9). |
| **Gemini only captions; Claude is the reader (decided 2026-10-14, later the same day)** | Gemini writes the isolated captions and nothing else. The frontier readers are Claude through `claude -p` (subscription quota): **Sonnet is the main reader**; Opus is the top of the scale curve on at least 100 videos (more, or the main reader, if the pilot shows Sonnet near chance on frames and the quota allows); Haiku runs on everything as the bottom of the curve. Consequences: the reader never reads its own captions; E4 needs Sonnet's own isolated captions to stay within one model; the open VLM on Kaggle becomes the required second family; Gemini spend falls to ≈ $25. |

## 1. Thesis and claims

**Working title:** *Seeing order, saying order: language makes the direction of events usable.*

**Thesis.**
- Each 5 s piece is captioned in isolation, from exactly the frames the vision arms see, so the captions contain no information about the order that the frames lack (Prop. 1).
- Any advantage readers get from captions is therefore *usable* information, created by computation, not new information.
- We predict, and test, that this advantage lies in the **direction** of events (what comes first), which needs knowledge of how events unfold.
- **Continuity** (which pieces belong next to each other) we expect to be carried at least as well by pixels.

**Claims, and where each is tested.**

| Claim | Tested by |
|---|---|
| C1. An information-controlled paradigm for comparing language and vision on event order, and evidence that whole-video captions leak order | E5, Props. 1 and 2 |
| C2. Order splits into structure (continuity) and direction. Pixels carry structure as well as or better than captions. Captions carry direction better, for readers with world knowledge. | E1, E2, E3 |
| C3. The language advantage is one of format as well as knowledge: the same model orders better after it puts what it sees into words | E4, E7 |
| C4. Where language wins (agents, goals, state changes) and where it does not (scenery, cyclic motion), across two datasets, and whether humans show the same split | E6, E8, E9 |

Generation (the old headline) becomes a short negative result in the appendix: scoring decides the answer, and generation stops improving with scale (`new_plan.md` §1–2).

## 2. Theory

**Setup.**
- A clip is cut into pieces X_1,…,X_n in true order (n = 12 pieces of 5 s).
- A puzzle shows the pieces under a uniformly random permutation π: the reader gets Y = (X_π(1),…,X_π(n)) with neutral labels and returns an order.
- Captions are C_i = g(X_i, U_i): g is the captioner and U_i is captioning noise, i.i.d. and independent of (X, π). This holds because each piece is captioned in its own call, from its frames only, without audio.
- The caption puzzle is Y^C = (C_π(1),…,C_π(n)).

**Proposition 1 (information control).**
- **Statement:** π → Y → Y^C is a Markov chain. Hence I(π; Y^C) ≤ I(π; Y), and for any loss the best achievable risk from captions is no better than from frames.
- **Proof:** Y^C = (g(Y_1, U′_1),…,g(Y_n, U′_n)), where U′ = U∘π is again i.i.d. and independent of π given Y. So Y^C is a randomized function of Y, and a reader of Y^C composed with g is a reader of Y.
- **Corollary:** if readers from captions beat readers from frames, the difference is a gap in *usable* information under computational constraints (Xu et al., ICLR 2020), not in Shannon information. In that framework, computation, here putting the pieces into words, can create usable information.
- **Remark (the leak, formally):** an in-context captioner writes C_i = g_ctx(X_1..X_n, i, U) while seeing the pieces in true order. Y^C then depends on π beyond Y and the chain breaks. That is why isolation is necessary, and why the in-context vs. isolated difference (E5) is a leak, not noise.
- **Conditions the design must keep:**
  - the vision readers get the **same frames** the captioner saw: 1 fps, 5 per piece, at no lower resolution (the captioner used about 66 tokens per frame, so low resolution is enough);
  - the audio stays removed;
  - the captioner never sees another piece.

  Arms that see less than the frames (SigLIP means) are lower bounds, which is allowed.

**Proposition 2 (symmetric scorers cannot know direction).**
- **Statement:** suppose a method scores orders with S(o) = S(reverse(o)) (for example, the shortest Hamiltonian path under a symmetric similarity), and breaks ties between an order and its reverse with a fair coin. Then its expected τ is 0, whatever the signal, because τ(reverse(o), π) = −τ(o, π).
- **Consequence:** all its order information is reversal-invariant, so we measure it with **structure** metrics: undirected adjacency accuracy, and |τ| for diagnostics only. **Direction** is measured separately by a reversal test (E2): the true order against its reverse, with the frames inside each piece still forward. Any E[τ] > 0 needs an asymmetric signal: physical arrows of time or knowledge of how events unfold.
- **Sanity gate:** symmetric arms must sit at direction accuracy 0.5.

**Proposition 3 (continuity under gaps; a model to test, not a theorem about video).**
- **Model:** for modality m, the similarity of pieces whose centres are δ seconds apart is s = ρ_m(δ) + ε, with ε i.i.d. N(0, σ_m²) and ρ_m decreasing.
- **One comparison:** with piece spacing Δ (5 s contiguous, 10 s gapped), the chance that a piece looks more similar to a piece 2Δ away than to its true neighbour Δ away is Φ(−κ_m(Δ)/√2), where κ_m(Δ) = (ρ_m(Δ) − ρ_m(2Δ))/σ_m.
- **Prediction:** adjacency accuracy from similarity chaining rises with κ_m, so gaps hurt each modality in proportion to how fast its similarity decays. For an exponential decay with time constant λ, κ peaks at Δ = λ ln 2 and vanishes for Δ ≫ λ.
- **Test:**
  1. Estimate each video's *continuity profile* ρ_m(δ), σ_m from all 66 piece pairs, for frames (SigLIP 2), for captions (SigLIP 2 text, the same embedding family) and for captions (MPNet).
  2. Map κ to expected accuracy by Monte Carlo under the model.
  3. Check calibration against the observed chaining accuracy per puzzle type (E3).

  The profiles are themselves a finding: how fast language and pixels forget across time.

**Pre-registered hypotheses** (direction fixed now; confirmatory tests fixed in the pre-registration after the pilot):

| | Hypothesis | Falsified if |
|---|---|---|
| H1 | Structure: on contiguous puzzles, frame similarity chaining recovers undirected adjacency at least as well as caption chaining. Gaps lower both, by the amounts Prop. 3 predicts from the measured profiles. | Captions beat frames on contiguous structure, or the predictions are uncalibrated |
| H2 | Direction: the same reader recovers direction (reversal test and τ) better from captions than from frames, for Claude Sonnet (primary), Haiku and Opus, and for the open VLM | No difference, or frames ≥ captions |
| H3 | Verbalization: for the same model, frames only < frames with "describe each piece, then order" ≤ isolated captions written by that model | Describing first does not help |
| H4 | Where language wins: the caption − frame direction advantage per video is larger for videos with agents, goal-directed actions and state changes, and larger on the script-rich dataset | No relation, or the opposite |
| H5 | Leak: readers order in-context captions better than isolated ones, and part of the advantage survives removing cue words | No difference |
| H6 (exploratory) | Scale: direction from captions rises with reader scale (Llama-8B < Haiku < Sonnet < Opus) more than direction from frames does | n/a (exploratory) |

## 3. Experiments

Puzzle sets, all cut from the same 12 pieces:
- **P12:** contiguous, 12 pieces.
- **C6:** pieces 0–5 and 6–11.
- **G6:** pieces {0,2,…,10} and {1,3,…,11}.
- **K4** (the knob, new): 4 pieces with fixed n and gaps of 0, 5 or 10 s (e.g. {0,1,2,3}, {0,2,4,6}, {0,3,6,9}); 3 puzzles per gap level per video.

Readers that read pieces in the order given see 2 shuffles per puzzle; position bias is known to be strong (Ianaro et al. 2026).

| | Question | Arms (frames ↔ captions, matched) | Data | Cost |
|---|---|---|---|---|
| **E1 Main comparison** | How much order, structure and direction does each modality give, at each level of reader? | No reader: SigLIP 2 mean chaining ↔ caption chaining (SigLIP 2 text and MPNet); SigLIP edge chaining (directed). Learned, leave-one-channel-out: position probe and pairwise order classifier on frame embeddings ↔ caption embeddings. Readers: Claude Sonnet (main) and Haiku on all videos, Opus on ≥ 100 (frames ↔ captions, listwise); Llama-3.1-8B (captions) and an open VLM such as Qwen2.5-VL-7B (frames ↔ captions), pairwise log-odds on Kaggle | 335 WildQA; confirmatory on the 305 outside the pilot; P12, C6, G6 | Claude quota; Kaggle free |
| **E2 Direction (reversal test)** | Which end is the start? | Every reader, both modalities: the pieces shown in true or in reversed order (frames inside each piece still forward), forward or backward answer. Symmetric arms: 0.5 by Prop. 2 (gate) | 335 | Claude quota |
| **E3 Gap knob and continuity profiles** | Does continuity decay as Prop. 3 predicts, per modality? | No-reader and learned arms on all videos; Claude Sonnet frames ↔ captions on 100 videos | K4, C6, G6 | Claude quota |
| **E4 Verbalization within one model** | Format or knowledge? | Claude Sonnet: (a) frames only; (b) frames, "describe each piece, then order", one call; (c) isolated captions written by Sonnet from the same frames with the frozen captioning prompt; (c′) Gemini's isolated captions; (d) frames + captions. Perception check: Sonnet matches each caption to its frames. (c) vs. (c′) also serves as the second-captioner check of E10 | 100 videos; P12, G6 | Claude quota (≈ 1,200 caption calls + readers) |
| **E5 Leak** | Do whole-video captions carry order? | Llama-8B, Claude Haiku and Sonnet (P12 only) on isolated / in-context / in-context without cue words / original / original without cue words | 335 (in-context captions for all 335: +$3.40) | Claude quota |
| **E6 Where language wins** | Which videos and events favour captions? | Per-video caption − frame advantage in structure and direction, regressed on: WildQA domain (5), visual change rate, a manual tag of 60 videos (agent, goal-directed, state change, scenic, cyclic), text tags | 335 | free (60 manual tags) |
| **E7 Scale** | Does knowledge grow the advantage? | On captions: Llama-8B, Haiku, Sonnet, Opus (≥ 100-video subset). On frames: open VLM, Haiku, Sonnet, Opus (subset) | 100–335 | Opus quota |
| **E8 Second dataset (script-rich)** | Does the language direction advantage grow with how scripted the video is? | The core of E1 and E2 | about 200 COIN clips of 60 s with ≥3 annotated steps; COIN is also SPLICE's source, so our results are directly comparable | ≈ $8 of captions + Claude quota |
| **E9 Human baseline** (decision needed) | Do people show the same split? | Captions only vs. frames (5 s clips, no audio), on C6 and G6 | 40 puzzles × 2 conditions × 15 people (Prolific, about 120 participants) | about £300; ethics approval |
| **E10 Robustness** | Is it the captioner, the encoder or the text tags? | Without tagged pieces; captions cut to 15 words; a second captioner (Sonnet's captions from E4, 100 videos); caption resampling at a new seed (60 videos, test–retest); a second vision encoder (DINOv2) for the no-reader arms | subsets | ≈ $2 |

**Reader protocol (all reader arms).**
- **Inputs:**
  - Pieces get neutral letter labels.
  - Frames go in as the 5 frames per piece the captioner saw (the piece cut exactly as for captioning, sampled at 1 fps), as JPEGs 448 px wide through stream-json: about 150 Claude tokens per frame, against about 66 Gemini tokens per frame for the captioner.
  - Captions go in as plain text.
  - The prompt states the puzzle's geometry (piece length, and the gap between consecutive pieces), worded identically for both modalities.
- **Output:** a permutation in a fixed JSON format, parsed strictly. One retry on a format failure, then the puzzle is scored as invalid, and invalid counts as random (its expected score: τ = 0, adjacency 2/n undirected, 1/n directed).
- **Pinned settings, recorded per call:** full model ID (`claude-sonnet-5-5`, not the `sonnet` alias, which moves when a new version ships), effort level (medium), and the model ID the CLI reports.
- **Pairwise local readers:** log P(A before B), averaged over both presentation orders, computed once per piece pair and video and reused across all puzzle sets. Aggregated by the exact linear-ordering optimum (Kemeny-style subset DP over all pairs, primary) and by the best Hamiltonian path over adjacent-pair scores (secondary).

## 4. Measurement and statistics

- **Metrics:**
  - Kendall τ (overall);
  - undirected and directed adjacency accuracy (structure);
  - reversal-test accuracy (direction);
  - exact position.

  Pairwise accuracy is not reported separately, since it equals (τ+1)/2.
- **Unit and clustering:** the video, clustered by channel (40 channels). CIs come from a channel-cluster bootstrap (per-video means, channels resampled; `boot` in `scripts/blind_recon_compare.py`). Effects are estimated with mixed models (metric ~ modality × puzzle × reader + (1 | channel/video)), and they must agree with the bootstrap.
- **Confirmatory vs. exploratory:**
  - The 30 pilot videos are for development.
  - The confirmatory hypotheses, metrics, arms and contrasts are fixed in `unshuffle_preregistration.md` and committed **before** any arm runs on the 305 held-out videos.
  - Up to 4 primary tests, Holm-corrected; everything else is exploratory and labelled so.
  - Null claims use equivalence tests (TOST) with a margin fixed in advance, e.g. ±0.05 τ.
- **Power:** from the pilot's per-video SDs. The target is a 95% CI half-width ≤ 0.04 τ on the primary contrasts with 305 videos; if the pilot says that is out of reach, report it, and add videos (WildQA has more clips per channel) rather than change the tests.
- **Orderability:** per video, the mean τ of a pre-specified reader set, separately for captions and frames. Their difference is the per-video language advantage used in E6. Reliability comes from the two C6 and the two G6 puzzles, which share no pieces; if it is low, report by channel or category.
- **Robustness reporting:** every primary result is repeated without text-tagged pieces, with 15-word captions, and with the second captioner (where it exists).

## 5. Phases, gates and budget

**Phase 1: build and pilot (now; mostly free).**
1. Write `docs/new_direction/unshuffle_pilot_preregistration.md`: the pilot's analysis plan, as in kickoff §3.5, plus E2–E4.
2. Build `scripts/unshuffle/` with tests: `build_pieces`, `make_puzzles` (P12, C6, G6, K4), the no-reader, learned and pairwise arms, `score`, and the continuity profiles. Run them on the 30 pilot videos.
3. Write the Claude listwise reader (`scripts/unshuffle/claude_reader.py`, the blind runner's isolation plus stream-json image input) and pilot it:
   - **Smoke test:** Haiku, 3 puzzles × frames and captions (6 calls): images arrive, labels and JSON parse.
   - **Calibration:** Haiku, Sonnet and Opus on identical puzzles: the first 10 pilot videos (seeded pilot order) × {P12, G6 even pieces} × 2 shuffles × {frames, isolated captions} = 80 calls per model. At the October rates, about 12 points of the 5-hour window for Sonnet, 13–21 for Opus, under 1 for Haiku. Measures accuracy per model and modality, quota per image call, format failures and position bias.
   - **Decision rule (fixed before the run):** the main reader is the strongest model whose frame τ is clearly above 0 and whose full run fits the quota the user is willing to spend; Opus covers at least 100 videos for E7. Captions at ceiling mean harder puzzles (G6, K4), not a weaker model.
4. Kaggle: Llama-8B and open-VLM pairwise scores on the pilot.
5. Write the theory section (Props. 1–3, proofs, Monte Carlo for Prop. 3).

**Gate 1 (user and partners):** pilot results table, measured costs, the confirmatory pre-registration draft, and the budget decision.

**Phase 2: full WildQA run (after Gate 1).**
1. Caption the remaining 305 videos: 3,660 isolated calls, ≈ $10, plus in-context captions for all of them, ≈ $3.40. The captioning prompt stays frozen.
2. Commit the pre-registration, then run all arms on the 335 videos. Confirmatory analysis on the 305.

**Gate 2:** main results, then a decision on E8 and E9.

**Phase 3:** second dataset (E8), human baseline (E9), robustness (E10), second-pass literature review (citation chase of SPLICE, TempVS and VIST; a monthly arXiv check for concurrent work).

**Phase 4:** writing. Partners choose the venue against current deadlines.

**Budget (Gemini at the measured median $0.0028 per isolated caption call; Gemini only captions, decided 2026-10-14):**

| Item | Estimate |
|---|---|
| Spent so far (captions, Stage 1 and the pilot) | $1.56 logged |
| Captions, remaining 305 videos (isolated + in-context) | ≈ $14 |
| E8 second dataset (captions) | ≈ $8 |
| E10 caption resampling | ≈ $2 |
| **Total Gemini** | **≈ $25**; the credits work (confirmed 2026-10-14) |
| Claude (subscription quota) | Sonnet (main): E1 and E2 on all 335 videos, roughly 10k calls, about 60 points of the 7-day figure at October's text-call rates (image calls measured in the calibration); E3 and E4 on 100. Haiku: everything (nearly free). Opus: at least 100 videos (about a full week of quota for all 335) |
| Kaggle (Llama, open VLM) | free |
| Human study (optional) | about £300 + ethics approval |

**Spending rules** (unchanged):
- `--max-usd` is cumulative and is raised only deliberately.
- Each phase starts with `--limit` and a cost readout.
- Report to the user at every gate before the larger run.

## 6. Risks

| Risk | Handling |
|---|---|
| A concurrent paper does this exact comparison | The information control, the knob and the theory are hard to scoop together. Second-pass review, and a monthly arXiv check (SPLICE and TempVS citations). |
| Readers are at ceiling on captions for P12 | G6 and K4 are harder; τ and adjacency are reported per puzzle type |
| Frame readers fail at *seeing*, not at ordering | Perception check in E4 (caption ↔ frames matching by the same model). Frames at the captioner's resolution. |
| Position bias in listwise readers | 2 shuffles per puzzle; bias is measured and reported |
| Findings are about Gemini's captioning style | Second captioner (Sonnet, E4) on 100 videos; the reader never reads its own captions |
| Findings are about Claude | The open VLM (Qwen2.5-VL-7B) and Llama-8B on Kaggle are a second family, so that arm is required |
| Contamination: models have seen these YouTube clips | Low for the order of random 60 s clips; the held-out design and the human baseline bound it; stated as a limitation |
| Quota | The calibration measures quota per image call before any large run; subsets for E3, E4 and Opus; `--max-5h` stop |
| Ethics delay for E9 | E9 is not on the critical path |

## 7. Decisions needed

- [ ] **Framing:** the paper's claim is C1–C4 (an information-controlled language-vs-pixels comparison, with structure vs. direction), not "language orders better".
- [x] **Gemini budget:** Gemini only captions (2026-10-14), ≈ $25 in total; the credits work.
- [x] **Readers:** Claude through `claude -p`; Sonnet main, Haiku everywhere, Opus on a subset; confirmed at Gate 1 by the calibration's decision rule (2026-10-14).
- [ ] **Second dataset (E8):** COIN (recommended: script-rich, and SPLICE's source) or none.
- [ ] **Human baseline (E9):** yes or no; who handles ethics approval.
- [ ] **Opus subset size** (quota; at least 100 videos), at Gate 1.
- [ ] **Target venue and deadline.**
