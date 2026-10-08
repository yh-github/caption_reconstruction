# TODO

Full reasoning, verified numbers and the evaluation design are in the paper-readiness assessment (2026-10-05):
https://claude.ai/code/artifact/26353452-f893-49df-85ac-d3a85c9becd2

Latest session summary and decisions: `docs/handover.md` (newest session at the top; 2026-10-08). Working paper draft: `docs/paper/captions_vs_frames.md`.

**Current state of the thesis (2026-10-08):**
- **Reconstruction as the paper's core: NO-GO at 8B.** Llama loses to copying on every metric, including forced choice (35% against 52% for caption copy).
- **New direction: GO.** The paper becomes "what dense LLM captions keep and lose relative to visual embeddings, and why that caps text-based reconstruction" (section 0). Captions share topic with the frames but barely track second-to-second change, and they run 1–2 s early in every channel (Gemini 3 Flash, one call per video). Videos where captions are better grounded show smaller caption-copy deficits.
- The old "procedural vs. stochastic" thesis (H1/H2 in `docs/paper/draft.md`) was rejected by our own tests; `draft.md` needs a rewrite.

**Correction (2026-10-07):** the frame embeddings are **SigLIP 2** (timm `v2_webli`), not SigLIP 1. Text compared against frames must use `SiglipTextEmbedder("google/siglip2-base-patch16-224")`. Earlier text-vs-frame results made with SigLIP 1 text are invalid (details in `docs/handover.md` §1).

**Video-based arms:** must match the text arm's size (Llama-3.1-8B → a ~7B text+video model). On hold until the API budget decision.

**Goal for new evaluations:** show the LLM beats simple baselines, and compare it fairly with video baselines. Normalize each arm to its own modality: recovery = (arm − masked) / (oracle − masked), and content-specific gain = arm − shuffled control.

Legend: (Claude) = can be done locally; (GPU) = Claude prepares the config and hands over the exact command, the user runs it on the remote GPU; (API) = waiting on the API LLM budget.

## 0. Captions vs. frames (new core direction)

- [x] Audit with no masking (`scripts/caption_vs_siglip_audit.py` → `results/caption_audit/`):
  - **Q1, grounding:** MRR 0.142 against 0.079 chance. Captions run **about 2 s early**: caption t best matches frame t+1 or t+2 in 222 of 335 videos. Frame timing was verified with `scripts/check_frame_timestamps.py`.
  - **Q2, shared variation:** within-video R² ≈ 0.02 in every direction.
  - **Q3, change timing:** caption change and frame change are uncorrelated.
  - **Link:** Q1 and Q2 predict where caption copy trails frame copy (partial ρ ≈ −0.36 to −0.39, controlling for change rates) and where Llama trails caption copy (partial ρ ≈ +0.25).
- [x] Caption-lag robustness (`scripts/caption_lag_robustness.py`; details in `docs/paper/captions_vs_frames.md` §3.2):
  - Peak at +1 or +2 s in 39 of 40 channels (+3 in one, none at 0 or earlier). The same holds in wild4 and wild5 and in every caption-length tertile.
  - Mostly a constant offset; the lead shrinks slightly late in the clip (slope −0.33 s per minute).
  - Re-aligning by +2 s improves the oracle caption's frame rank (22.9 → 20.2) but not caption copy (+0.4, CI touches 0) or Llama (28.5, about chance).
  - Q3 with the lag corrected: event coincidence is 33% against 27% chance; Spearman ≈ 0.
- [x] Shared-target evaluation re-run with SigLIP 2 text (`configs/eval_shared_target.yaml` fixed; old frame-home text results in `results/invalid_siglip1_text/`). All gates pass.
  - **Topic kept, timing lost.** In frame space the true caption scores c = 0.80 against other-channel frames, but only 0.18 against frames within ±10 s. There it equals caption copy (0.19), while frame copy reaches 0.52. Llama is 0.05 near and 0.70 against other channels.
  - C1–C4: Llama is INFERIOR to both caption baselines.
  - Details: `docs/paper/captions_vs_frames.md` §3.5.
- [x] Near-pool comparison with lag-corrected captions (`caption_lag_robustness.py` part 5):
  - The true caption goes from c = 0.17 to 0.28 at +2 s (+0.11 [0.07, 0.15]), so it now beats caption copy (0.20) but stays far below frame copy (0.52).
  - The lead explains about a third of the near-second gap.
- [ ] (Claude or API) Q4: caption correctness audit (hallucinations such as the "runner" camera-holder) on about 50 sampled seconds, manually or with a VLM judge.
- [ ] (Claude) Check whether the lag and grounding findings generalize to how the captions were produced (single Gemini call over the whole video), e.g. by re-captioning a few videos per second as a control (API).

## 1. Data cleanup

- [x] Wild4 W=3 Llama outputs re-scored with `pool_scope: "video"` (`scripts/rescore_window_pool_run.py` → `wild4_llama_w3_window_v3_videopool/`). `rebuild_unified_master.py` and `build_combined_benchmark_suite.py` now use it; master CSV, combined suite and rank-difference outputs regenerated (W=3 MRR 0.089, 335 videos). The explorer's `exclude_w3_anomaly` toggle is now unnecessary.
- [ ] (Claude) Reconcile counts across `docs/paper/draft.md`, `outline.md` and `experiments_plan.md`: 40 channels, 335 videos (wild5 = 235), MPNet as the text evaluation space, W ∈ {1,2,3,4,6,8,12,16}.
- [ ] (Claude) Remove the unsupported "Wild-Key-Events / LLM judge" claim in `draft.md` §5.1, and the 60-pool ranks of 258 and 450 in §4.5.

## 2. E1: question-conditioned evidence retrieval (demoted: cannot anchor the paper)

- [x] Shuffled-reconstruction control (`scripts/downstream_shuffle_control.py`). MPNet: recon 0.237 against shuffled 0.143 and repeat 0.137 (MRR, 176 questions), so the gain is content-specific. SigLIP: shuffled alone reaches 0.213, so "recon beats oracle" is a style artifact. Use MPNet as primary.
- [x] Full control set (`scripts/e1_evidence_retrieval.py`): Llama text from a *different gap of the same video* scores the same as the real reconstruction (MPNet 0.238 vs 0.237), so the gain is video topic, not gap content (that control leaks the evidence captions, but see the next item). Frame arms re-run with SigLIP 2 text (2026-10-07): oracle 0.333, repeat 0.217, other-video 0.219. The earlier "other-video frames beat the oracle" was an artifact of mixing SigLIP 1 text with SigLIP 2 frames. In SigLIP 2 text, recon (0.322) still beats the oracle (0.272), so this is a style artifact; MPNet stays primary.
- [x] Symmetric design (`scripts/e1_symmetric_feasibility.py`, evidence + 2 decoys, 297 questions), re-run with SigLIP 2 text: the oracle has little headroom (frames 0.707, MPNet 0.671, chance 0.611). WildQA questions are video-level, so E1 and E2 lack temporal headroom. Not worth a decoy GPU run.

## 3. Prompt ablation

- [ ] (Claude + GPU) A neutral prompt directory without "do NOT repeat / distinct, progressive actions" (`prompts/dense_window/default.txt`), on W ∈ {1,4,8,16} and on the evidence gaps.
- [ ] (Claude + GPU) A prompt variant that tells the model to use the context after the gap (failure case: `Army-military-2018_0-clip-8`, a day→night cut).

## 4. New evaluations

- [x] Feasibility checks (`scripts/feasibility_gap_level_checks.py`, `scripts/evidence_gap_intrinsic_ranks.py`):
  - Llama's within-gap ordering is at chance (calibrated c ≈ 0), so evaluate at the level of the whole gap.
  - On evidence gaps, per-second rank still favors copy (25.8 against 21.4).
  - Llama's win rate against visual copy doubles (8% → 17%) under high visual boundary disagreement (scene changes).
- [x] Forced-choice gap test (`scripts/forced_choice_gap.py`): gap + 3 same-length distractor spans masked jointly; every method picks the gap's content among 4 candidates (chance 25%). Baselines (1,325 items): caption copy 47%, assignment-aware 54%; frame copy 86%, assignment-aware 94%. Pre-registered LLM score: PMI (conditional minus no-context log-prob).
- [x] (GPU) LLM scoring for the forced-choice test, stopped at 180 of 1,325 items (`scripts/kaggle_forced_choice.py`). On the same items: Llama PMI 35% (CI 28–42), caption copy 52%, assignment-aware 59%, frame copy 85%. Llama's errors are independent of copy's, but fusion gives no gain. Report: `.venv/bin/python scripts/forced_choice_gap.py report --download`.
- [x] Continuity-matched distractors (`scripts/forced_choice_matched_feasibility.py`): **NO-GO**. Frame continuity is too strong for matched distractors to exist; with both modalities matched, frame copy is still 65% on 286 items. Caption-only matching (675 items, caption copy 16%, frames 87%) remains possible as a supporting experiment.
- [ ] (Claude) E4: set-level (best-of-gap) matching across all widths, as a diagnostic.
- [ ] (API) E3: gap QA probes (now the main generation-based alternative, since WildQA questions are video-level). Generate 3–5 questions per gap from the true gap captions, answer them from each arm's transcript, and judge. Pilot on 50 gaps first.
- [ ] (API or GPU) E2: reader QA on WildQA questions, with a text reader for the text arms and a video LLM reader for the frame arms. Pre-registered strata: visual boundary disagreement and gap width. Open decision: API model, or a local Qwen2-VL-7B.
- [ ] (API) A stronger text model as a scaling point, on direct similarity, E1 and E3.

## 5. Paper

- [ ] (in progress: `docs/paper/captions_vs_frames.md` replaces `draft.md`, which stays as the record of the old thesis) Write the paper around the captions-vs-frames thesis (section 0), with reconstruction as the consequence (LLM loses to copy everywhere). Cut H2 and the Δ/N spectrum to one paragraph (scene continuity explains it). Every mention of the visual encoder must say SigLIP 2.
- [x] Figures (`scripts/make_paper_figures.py` → `docs/paper/figures/fig{1..4}_*`): lag profile + near-pool shift, topic vs. timing by pool, forced choice, grounding vs. deficit. Choose which two fit in 4 pages.
- [x] Abstract and related work drafted in `captions_vs_frames.md`, with a reference list. Most entries were checked against venue pages; those marked † were cited from memory and need checking before submission.
- [ ] Polish: tighten to 4 pages and move to the venue's LaTeX template (below). Decide which two figures to keep.
- [ ] Move to the venue's 4-page LaTeX template.

## Done (infrastructure)

- [x] Captions for wild4 (100) and wild5 (235); SigLIP 768-d embeddings for all 335; a-priori scores in `results/apriori_full_scores.csv`.
- [x] Llama-3.1-8B runs: wild4 and wild5 multi-width sweeps (W = 1–16, i=29); wild5 W=3/6 at i ∈ {0, 29, 59}; WildQA evidence gaps (dev and test).
- [x] Four baselines (caption/visual × repeat/mean-closest) across all widths. `results/unified_benchmark_master.csv` is assembled, but see the cleanup in section 1.
- [x] Streamlit explorer (`scripts/evaluation_explorer_app.py`); channel-clustered statistics (`scripts/compute_cluster_and_continuity_stats.py`, `scripts/run_hypothesis_regression_tests.py`).
