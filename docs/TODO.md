# TODO

Full reasoning, verified numbers and the evaluation design are in the paper-readiness assessment (2026-10-05):
https://claude.ai/code/artifact/26353452-f893-49df-85ac-d3a85c9becd2

**Current state of the thesis:** on direct (per-second similarity) metrics, LLM in-filling loses to copying the nearest caption. The downstream WildQA result looked like a win, but controls show it measures video topic, not gap content, and WildQA questions barely localize their evidence even for the oracle. The lead candidate is now the forced-choice gap test (section 4). The old "procedural vs. stochastic" thesis (H1/H2 in `docs/paper/draft.md`) was rejected by our own tests; `draft.md` needs a rewrite.

**Video-based arms:** must match the text arm's size (Llama-3.1-8B → a ~7B text+video model). On hold until the API budget decision.

**Goal for new evaluations:** show the LLM beats simple baselines, and compare it fairly with video baselines. Normalize each arm to its own modality: recovery = (arm − masked) / (oracle − masked), and content-specific gain = arm − shuffled control.

Legend: (Claude) = can be done locally; (GPU) = Claude prepares the config and hands over the exact command, the user runs it on the remote GPU; (API) = waiting on the API LLM budget.

## 1. Data cleanup

- [x] Wild4 W=3 Llama outputs re-scored with `pool_scope: "video"` (`scripts/rescore_window_pool_run.py` → `wild4_llama_w3_window_v3_videopool/`). `rebuild_unified_master.py` and `build_combined_benchmark_suite.py` now use it; master CSV, combined suite and rank-difference outputs regenerated (W=3 MRR 0.089, 335 videos). The explorer's `exclude_w3_anomaly` toggle is now unnecessary.
- [ ] (Claude) Reconcile counts across `docs/paper/draft.md`, `outline.md` and `experiments_plan.md`: 40 channels, 335 videos (wild5 = 235), MPNet as the text evaluation space, W ∈ {1,2,3,4,6,8,12,16}.
- [ ] (Claude) Remove the unsupported "Wild-Key-Events / LLM judge" claim in `draft.md` §5.1, and the 60-pool ranks of 258 and 450 in §4.5.

## 2. E1: question-conditioned evidence retrieval (demoted: cannot anchor the paper)

- [x] Shuffled-reconstruction control (`scripts/downstream_shuffle_control.py`). MPNet: recon 0.237 against shuffled 0.143 and repeat 0.137 (MRR, 176 questions), so the gain is content-specific. SigLIP: shuffled alone reaches 0.213, so "recon beats oracle" is a style artifact. Use MPNet as primary.
- [x] Full control set (`scripts/e1_evidence_retrieval.py`): Llama text from a *different gap of the same video* scores the same as the real reconstruction (MPNet 0.238 vs 0.237), so the gain is video topic, not gap content (that control leaks the evidence captions, but see the next item). Frames from another video beat the frame oracle (0.354 vs 0.246): per-video similarity offsets make within-video retrieval reward anything unusual.
- [x] Symmetric design (`scripts/e1_symmetric_feasibility.py`, evidence + 2 decoys, 297 questions): even the oracle is near chance (MPNet 0.671, frames 0.618, chance 0.611). WildQA questions are video-level, so E1 and E2 lack temporal headroom. Not worth a decoy GPU run.

## 3. Prompt ablation

- [ ] (Claude + GPU) A neutral prompt directory without "do NOT repeat / distinct, progressive actions" (`prompts/dense_window/default.txt`), on W ∈ {1,4,8,16} and on the evidence gaps.
- [ ] (Claude + GPU) A prompt variant that tells the model to use the context after the gap (failure case: `Army-military-2018_0-clip-8`, a day→night cut).

## 4. New evaluations

- [x] Feasibility checks (`scripts/feasibility_gap_level_checks.py`, `scripts/evidence_gap_intrinsic_ranks.py`):
  - Llama's within-gap ordering is at chance (calibrated c ≈ 0), so evaluate at the level of the whole gap.
  - On evidence gaps, per-second rank still favors copy (25.8 against 21.4).
  - Llama's win rate against visual copy doubles (8% → 17%) under high visual boundary disagreement (scene changes).
- [x] Forced-choice gap test (`scripts/forced_choice_gap.py`): gap + 3 same-length distractor spans masked jointly; every method picks the gap's content among 4 candidates (chance 25%). Baselines (1,325 items): caption copy 47%, assignment-aware 54%; frame copy 86%, assignment-aware 94%. Pre-registered LLM score: PMI (conditional minus no-context log-prob).
- [ ] (GPU) LLM scoring for the forced-choice test: `python scripts/forced_choice_gap.py llm --model-key llama-3.1-8b --upload` (pilot with `--limit 40` first). Then locally: `.venv/bin/python scripts/forced_choice_gap.py report --download`.
- [ ] (Claude) If the LLM clears the text baselines: harder distractors (spans near the gap) so frames no longer saturate, to look for regimes where text beats video.
- [ ] (Claude) E4: set-level (best-of-gap) matching across all widths, as a diagnostic.
- [ ] (API) E3: gap QA probes (now the main generation-based alternative, since WildQA questions are video-level). Generate 3–5 questions per gap from the true gap captions, answer them from each arm's transcript, and judge. Pilot on 50 gaps first.
- [ ] (API or GPU) E2: reader QA on WildQA questions, with a text reader for the text arms and a video LLM reader for the frame arms. Pre-registered strata: visual boundary disagreement and gap width. Open decision: API model, or a local Qwen2-VL-7B.
- [ ] (API) A stronger text model as a scaling point, on direct similarity, E1 and E3.

## 5. Paper

- [ ] Rewrite `docs/paper/draft.md` around the new thesis. Cut H2 and the Δ/N spectrum to one paragraph (scene continuity explains it).
- [ ] Figure 1: direct-similarity curves for the text arms. Figure 2: E1/E2 recovery and content-specific gain, text arms against video arms.
- [ ] Move to the venue's 4-page LaTeX template.

## Done (infrastructure)

- [x] Captions for wild4 (100) and wild5 (235); SigLIP 768-d embeddings for all 335; a-priori scores in `results/apriori_full_scores.csv`.
- [x] Llama-3.1-8B runs: wild4 and wild5 multi-width sweeps (W = 1–16, i=29); wild5 W=3/6 at i ∈ {0, 29, 59}; WildQA evidence gaps (dev and test).
- [x] Four baselines (caption/visual × repeat/mean-closest) across all widths. `results/unified_benchmark_master.csv` is assembled, but see the cleanup in section 1.
- [x] Streamlit explorer (`scripts/evaluation_explorer_app.py`); channel-clustered statistics (`scripts/compute_cluster_and_continuity_stats.py`, `scripts/run_hypothesis_regression_tests.py`).
