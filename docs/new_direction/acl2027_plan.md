# ACL 2027 paper: plan and next steps

Written 2026-10-19. Revised 2026-10-20 after a review of the plan with the user: a Claude captioner arm, Gemini kept only where it adds something, new terms, cost stated per question in full, a long draft, and SMART goals (§0). The experiment record is in `research_log.md` (C1 *Result*, C2, D2). The session history is in `docs/handover.md`. Each item below is marked **decided** (agreed with the user) or **proposed** (my recommendation, still open).

## 0. Goals (decided 2026-10-20)

Each goal has a test for done, an owner and a date. "Claude" means a Claude Code session in this repo. Dates are in 2026 unless marked. The 7-day Claude quota resets on Tuesdays at 17:00 UTC (Oct 20, Oct 27, …; from the `resetsAt` field in the reader's event logs).

| # | Goal | Done when | Owner | By |
|---|---|---|---|---|
| G1 | C2 complete | ≥ 98% of the 761 windows captioned; 297 Haiku reads; `reader.py c2` output in the log as C2's *Result* | User raises the AI Studio cap; Claude runs | **Done Oct 20** |
| G2 | Open choices decided | Each marked decided in §6: the 448 px arm, Sonnet on its own captions, the open-weights reader, the thinking analysis, the effort sweep | User | **Done Oct 20** |
| G3 | C3 run | Code and tests; the log entry with predictions committed before any call; 761 Sonnet captions and 297 Haiku reads; the result in the log | Claude; user approves the run | Entry Thu Oct 22; result Sun Oct 25 |
| G4 | Cost account | Per arm: input, output and thinking tokens, list-price $, latency; index cost per video; Q\* with bootstrap CIs (§5, step 6) | Claude | Sun Oct 25 |
| G5 | Claim fixed (go/no-go) | The rule below applied, and the claim written in §3 | User and Claude | Sun Oct 25 |
| G6 | Literature pass | Log entry N2 and §10 of `related_work_caption_index.md`, with a verdict (scoop / close / related) per question Q1–Q7 | Claude | Sun Nov 1 |
| G7 | Chosen optional arms | Each with a log entry written before its run, and a result | Claude | Sun Nov 1 (open-weights reader: Sun Nov 8) |
| G8 | Full long draft | All sections; Tables 1–2 and Figure 1 from final numbers; limitations; appendix | Claude drafts, user edits | Sun Nov 15 |
| G9 | Long or short | Long if C3 replicated the captioner effect *and* at least one robustness arm (448 px or open-weights reader) is in; otherwise short | User | Mon Nov 30 |
| G10 | Team review | Reviewers named by Nov 15; every comment resolved or answered | User | Tue Dec 15 |
| G11 | Submission package | ARR checklist; anonymous release (code, captions, outputs); the pre-registration archived publicly (§5, step 9) | Claude and user | Thu Dec 31 |
| G12 | Submit | Submitted through ARR; the deadline checked against the ACL 2027 call on Nov 15 and Dec 1 | User | ARR deadline (Jan 4, 2027 assumed) |

**Go/no-go rule for G5** (C3 is the exact visual-token match, C2 the cross-family one; the same rule is in the log, C3):
1. **C3a's CI excludes 0, positive:** the headline is the captioner effect; target ACL main.
2. **C3a not positive, C2a positive:** the captioner effect is shown only across families. The claim names the captioner; length and venue are decided at G9.
3. **C3a and C2a not positive, C2b positive:** the claim becomes "spend the visual budget once, at captioning time".
4. **None positive:** the paper is the cost result plus the bound on the modality effect; target Findings.

**Not covered by the goals:** the user's available hours and the teammates' availability. G8 and G10 depend on both.

## 1. Target

- **Decided:** ACL 2027, through the **January 2027 ARR cycle**.
  - Checked 2026-10-20: ARR's dates page lists ACL 2027 as "January, 2027", with no day. The ACL 2027 site says TBA.
  - Secondary sources give **Jan 4, 2027**; one deadline tracker gives Feb 16. Plan to Jan 4 and confirm against the ACL 2027 call when it is published.
- **Decided (2026-10-20):** draft a **long paper** (8 pages), then decide long or short at the end of November, when C2, C3 and the optional arms are in. ARR fixes the choice only at submission. Cutting 8 pages to 4 is easier than the reverse.
- Fallbacks: Findings, or committing the ARR reviews to a later ACL-family venue. TMLR (rolling) for a longer version later.
- **Check:** ARR's rules for the submission (Responsible NLP checklist, reviewing duties for authors, anonymity, the limitations section).

## 2. Terms (decided 2026-10-20)

The definitions are in the *Terms* block of `research_plan_caption_index.md`; use only those words. Each effect is the difference between two arms that changes one thing, with the same reader in both:

| Effect | Changes | Held fixed |
|---|---|---|
| **Source** | what the captioner was given: the video (or more, larger frames) instead of the reader's frames | captioner, prompt, visual tokens |
| **Visual budget** | the visual tokens the captioner spends on the same pixels | captioner, pixels |
| **Captioner** | which model writes the caption | pixels, visual tokens, prompt |
| **Modality** | the reader reads its own caption (text) instead of the frames (images) | model, pixels |

With visual tokens held about fixed, the total caption effect = source + captioner + modality.
- **Retired:** "coverage" (now source) and "format" (now modality). C1 was pre-registered with the old names; its entry keeps them, and D2 in the log maps them.
- **Not a claim:** "knowledge". No arm separates what a captioner perceives, recognizes or writes, so the claims say "captioner effect".

## 3. What we have (C1: pre-registered, 297 held-out TempCompass questions)

| Reader, input | Accuracy | Video tokens | Input tokens | Cost per question |
|---|---|---|---|---|
| Haiku, question only | 0.41 | 0 | 1,025 | $0.00011 |
| Haiku, 1 frame per 5 s window | 0.58 | 164 | 1,189 | $0.00015 |
| Haiku, 5 frames per window (224 px) | 0.61 | 495 | 1,525 | $0.00020 |
| Haiku, its own captions of those frames | 0.64 | 258 | 1,283 | $0.00019 |
| Haiku, Gemini captions of those frames | 0.79 | 204 | 1,229 | $0.00017 |
| Haiku, Gemini captions of the video | 0.74 | 211 | 1,236 | $0.00017 |
| Sonnet, 5 frames per window | 0.80 | 495 | 1,527 | $0.00411 |
| Sonnet, Gemini captions of the video | 0.79 | 211 | 1,238 | $0.00287 |

Input tokens are medians per call. Cost is the mean of the CLI's `costUSD` per call (list basis, output and thinking tokens included), from `results/index_qa/claude/*.jsonl`. Step 6 cross-checks the list prices. The one-time captioning cost is not in the table: Gemini on the video costs ≈ $0.0028 per window, ≈ $0.0073 per video (2.57 windows on average).

**Findings, in the order the paper should give them:**
1. **Captioner effect (P3, pre-registered, passed):** Gemini's captions of the reader's own frames beat Haiku's own captions by +0.148 [98.33% CI +0.088, +0.209], positive in every aspect.
   - **C2 (exploratory, 2026-10-20): about half of it survives at matched visual tokens.** At ≈ 330 visual tokens per window instead of 5,500, the captioner effect is still +0.081 [+0.030, +0.131]. The other +0.067 [+0.030, +0.108] is the visual budget.
   - The source effect is ≈ 0: +0.013 [−0.030, +0.061].
   - C3 (Sonnet, exactly Haiku's 200 tokens) is still to come.
2. **Modality effect (P2, not passed):** +0.030 [−0.034, +0.094]. No evidence that text is easier for the reader than the pixels it came from: at most +0.09, against IsoBench's 15–29 points for exact text renderings.
3. **Secondary estimates** (95% CIs; their predictions were further off):
   - Haiku on Gemini video captions − Haiku on 5 frames: +0.125 [+0.071, +0.178] (S3, predicted 0.00). Part of it comes from Haiku's frame arm dropping from 0.70 (development) to 0.61 (confirmatory), so it is not the headline.
   - Sonnet on captions − Sonnet on 5 frames: −0.007 [−0.054, +0.040] (S1).
   - Haiku on captions − Sonnet on 5 frames: −0.061 [−0.111, −0.010] (S6).
4. **Which aspects captions lose: not confirmed** (P1: −0.10 [−0.23, +0.03]). Speed is the only consistent loss (Sonnet −0.10); exploratory.

**Cost** (log A2, `reader.py cost`; list prices, which match the CLI's `costUSD` exactly):
- **For one reader, the saving per question is small.** The fixed prompt (≈ 1,025 tokens) and the thinking tokens dominate. Captions cut the video tokens to 43%, but the input tokens only to 81%.
  - Haiku: captions instead of 5 frames save $0.000036 per question (18%).
  - Sonnet: they save $0.00124 per question (30%).
- **Break-even Q\*** (questions per video that repay the captioning; 95% CIs):
  - Haiku, same reader: 202 [156, 279].
  - Sonnet, same reader: 5.8 [5.2, 6.5].
  - The reader downgrade, Haiku on Gemini video captions instead of Sonnet on frames: 1.8 [1.7, 2.0]. That is 24× cheaper per question, but 6 points lower.
  - **Haiku on Gemini frame captions instead of Sonnet on frames:** 3.7 [3.5, 4.0], at equal accuracy (−0.007 [−0.057, +0.044]). Exploratory; its captioner spent 27× the visual tokens (C2), and that cost is in the index.
- **So the economic result is the reader downgrade, not fewer tokens for one reader.** Don't headline "43% of the tokens".

**Proposed claim:** a strong model's question-blind captions of the same frames make a cheap reader more accurate than its own view of those frames, at a fraction of the query cost. The gain comes from the captioner, not from the text modality.

## 4. Design: Claude for the main decomposition, Gemini as the second family (decided 2026-10-20)

**Main, within Claude (C3, new).** Sonnet captions exactly the 5 frames per window (224 px) that the reader sees, with the self-caption prompt and settings. Haiku reads the Sonnet captions.
- **Same pixels, same visual tokens, same prompt.** Every Claude model uses the same image-token rule, so Sonnet spends exactly the 200 tokens per window that Haiku spent on its self-captions. Haiku on Sonnet captions − Haiku on its own captions is the captioner effect, with no visual budget term.
- **Source effect = 0 by design.** The captioner sees only the reader's frames, so by Prop. 1 (garbling bound) the captions carry no information about the video that the frames lack. Any gain is usable information.
- **The strong model's direct answer is already measured:** Sonnet on 5 frames (0.798). The comparison for the index is: Sonnet captions once and Haiku answers each question, vs. Sonnet answering each question from the frames. No "ask the captioner directly" arm is needed.

**Gemini, only where it adds something Claude can't:**
1. **Video input.** The source effect (original video vs. the reader's frames) exists only with a captioner that takes video.
2. **A second model family.** A Claude-only study invites "Haiku just reads its sibling's prose well". The Gemini arms answer that.
3. **It costs dollars, not subscription quota.**

**No new Gemini arms.** C2 is finished, because it makes the Gemini captioner effect comparable at matched visual tokens.

**Pre-registered vs. exploratory.** P3 (Gemini) stays the pre-registered result. C3 is exploratory, designed after C1, on the same 297 questions. Its log entry, with predictions, is committed before any call. C3 replicates P3 within one family at identical input; it does not replace P3 as the headline.

| Effect (Haiku reads) | Claude chain | Gemini chain (C2 results, 2026-10-20) |
|---|---|---|
| Source | 0 by design | C2c: video captions − frame-clip captions = +0.013 [−0.030, +0.061] |
| Visual budget | 0 by design | C2b: frame captions (image path) − frame-clip captions = +0.067 [+0.030, +0.108] |
| Captioner | C3: Sonnet captions − Haiku's own captions (to run) | C2a: frame-clip captions − Haiku's own captions = +0.081 [+0.030, +0.131] |
| Modality | P2: Haiku's own captions − 5 frames = +0.030 [−0.034, +0.094] (98.33%) | (same) |

## 5. Next steps, in order

Steps 1–4 (C2) were **done on 2026-10-20**; the result is in the log, C2. They stay here as the record of how C2 was run.

1. **User: raise the AI Studio monthly spend cap** (https://ai.studio/spend). Done 2026-10-20; the logged Gemini total is $11.24.
2. **Resume the C2 captions.** Same command, and done windows are skipped:
   ```bash
   source ~/.secrets && for k in 0 1 2 3; do nohup .venv/bin/python scripts/index_qa/caption.py --set confirm \
       --input frameclip --max-usd 11.5 --shard $k/4 > logs/c2_frameclip_shard$k.log 2>&1 & done
   ```
   - Watch with `ls results/index_qa/captions/gemini-3.8-flash-frameclip/*/*.json | wc -l`.
   - Don't wait with `pgrep -f` on the command text: the waiting shell matches its own command line.
   - Stop rule: more than 2% of windows without a caption.
3. **Haiku reader on the frame-clip captions** (297 calls):
   `.venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms frameclipcaptions --skip-missing`
4. **Run `reader.py c2` once.** Record its full output as C2's *Result* in the log, each prediction next to its result. Read it by C2's mapped outcomes, in the new terms:
   - **C2a positive:** the captioner effect holds across families at about matched visual tokens. **This is the outcome (2026-10-20).**
   - **C2a null and C2b positive:** Gemini's P3 is mostly a visual budget effect. The captioner claim then rests on C3.
5. **C3: Claude captioner** (the Sonnet run after the quota reset of Oct 20, 17:00 UTC).
   - **Code (done 2026-10-20):** `self_caption.py --model sonnet` (folder `claude-sonnet-5-5-frames`, with a `--max-7d` stop), the `sonnetcaptions` reader arm, and `reader.py c3`; tests in `tests/test_index_qa.py`.
   - **Log entry C3** (written 2026-10-20): predictions, outcome mapping, stop rules and the commands. **The user approves it before the run.**
   - **Sonnet captions:** 761 windows. Caption calls take ≈ 1,300–1,500 input tokens plus thinking, so ≈ 10–15 points of the 7-day figure, and about two 5-hour windows at `--max-5h 0.7`.
   - **Haiku reads them:** 297 calls (little quota).
6. **Cost account and break-even** (no new runs).
   - **Inputs:**
     - Gemini caption cost per window, by mode, from `results/piece_captions/calls.jsonl` (video path ≈ $0.0028, mostly thinking).
     - Sonnet caption cost per window, from C3's calls.
     - Reader tokens (input, output, thinking), `costUSD` and latency per arm, from `results/index_qa/claude/*.jsonl`.
     - Claude list prices for Haiku and Sonnet 5.5, to check the CLI's `costUSD`. Look them up (claude-api skill); don't quote them from memory.
   - **Formula:** Prop. 2 in `research_plan_caption_index.md` (Q\*(α) = C(τ) / (c\*_f(α) − c\*_τ(α))). Compute it for α at the Haiku-frames, Haiku-captions and Sonnet accuracies, with bootstrap CIs.
   - **Output:** per question and arm, the input tokens, output and thinking tokens, list-price dollars and latency; the one-time index cost per video; Q\*.
7. **Optional arms, as decided** (§6; G2 done 2026-10-20): the 448 px Haiku arm; Sonnet on its own captions if quota allows after C3; the thinking analysis.
8. **Literature pass, after the quota reset of Oct 20, 17:00 UTC.** Run `final_search_brief.md`, reordered:
   - **Q5** (captioner effect from identical input) first, then Q1, Q2, Q4, Q6 and Q7. Q3 (aspects) last, since P1 failed.
   - **Deliverable:** §10 of `related_work_caption_index.md` and log entry N2. Do it before writing the introduction.
   - **Leads from the user's Gemini Deep Research report** (https://share.gemini.google/wZjHqQfnkeOL; render it with headless Chrome, `--dump-dom`). The report has no source list, so verify each lead in its full text:
     - **T3**, "Temporal Reasoning Transfer from Text to Video", arXiv 2410.06166. Checked 2026-10-19:
       - It replaces frames with GPT-4o captions only on its own synthetic probe videos.
       - Its only TempCompass row is LongVA *with* video.
       - The report's "LongVA without video" numbers are not in the paper.
       - Related (Q3) at most.
     - ShareGPT4Video (Q5, Related).
     - FlexBench and IsoVRBench (claimed motion-caption failures; unverified).
     - October 2026 preprints: arXiv 2610.11019 (mid-training, captions vs. visual tokens), 2610.16066 (PIGEONS), 2610.23098 (InstAgent).
     - T-MCoT and Zebra-CoT (verbalized reasoning; Q4).
   - **Another lead (from memory, not verified):** Vamos (ECCV 2024), free-text descriptions as a video representation for a language model (Q1, Q5).
9. **Make the pre-registration verifiable.** Reviewers can't check local git timestamps. Before submission, put the C1 commit (or an archive with its hash and date) somewhere public, timestamped and anonymizable: an anonymous repository for review, or OSF.
10. **Write** (§7), starting with the outline and the figures.

## 6. Optional arms (decided by the user 2026-10-20, G2: the recommendations below were accepted)

| Arm | Why | Cost | Decision |
|---|---|---|---|
| Haiku on 5 frames at 448 px (144 tokens each, ≈ 1,870 video tokens per question) | Answers "you starved the frames": 40-token frames, and Haiku barely gains from 1 to 5 of them (0.58 → 0.61). Shows where frames catch up on the accuracy–cost curve. | 297 Haiku calls (little quota). Frames re-extracted at 448 px (`tempcompass.py` has `LONG_EDGE = 224`). A log entry with predictions before the run. | **Do it** (G7) |
| Sonnet reads Sonnet's own captions | Sonnet's modality effect (P2 for a second reader), and the case where one model both indexes and answers | 297 Sonnet calls, ≈ 4 points of the 7-day quota | **Do it if quota allows after C3** (G7) |
| Open-weights reader on Kaggle (e.g. Qwen2.5-VL-7B) on the main arms | Answers "all readers are closed models"; reproducible | Free GPU; a few sessions of work | **Decide after C3** (G5, Oct 25) |
| Sonnet, question only and 1 frame (C1's skipped secondary arms) | Completes Sonnet's curve and gives its retention | 600 Sonnet calls, ≈ 7–8 points of the 7-day quota | Skip unless quota is spare |
| Opus as a third captioner | A captioner ladder (Haiku, Sonnet, Opus) | ≈ 2× C3's quota | Skip |
| Gemini answers the questions directly | Not needed: in the main design the captioner's direct answer is Sonnet on frames, already measured | ≈ $1–3 | Skip |
| Thinking tokens as an outcome (exploratory analysis of C1's calls) | See below. Thinking is part of the cost account either way (§5, step 6). | No calls | **Do it**: an exploratory log entry, and the analysis moved into `reader.py` |
| Effort sweep: Haiku at low and high effort, on captions and on 5 frames | The only way to learn whether thinking *helps* more on captions than on frames; the analysis below shows only that it marks hard questions | ≈ 1,200 Haiku calls (little quota) | **No** |

**Thinking tokens in C1** (computed 2026-10-20, after the results; exploratory; no new calls). At effort medium the model decides per call whether to think, so thinking is observed, not set. Thinking tokens are above 0 exactly when a call has a thinking block (all 3,042 reader calls).
- **How much:**
  - Haiku thinks on 25% of 5-frame calls and 45% of Gemini-caption calls, with about the same mean (79 and 74 tokens).
  - Sonnet thinks 3.3× less on captions (mean 24 vs. 80; 24% vs. 44% of calls).
  - Thinking is 2–5% of reader tokens.
- **Correlation across arms:** the same questions draw thinking from frames and from captions (Spearman +0.38 Haiku, +0.47 Sonnet). Within an aspect, it is +0.25 and +0.31. Speed and direction draw the most thinking in every arm, action almost none.
- **Thinking flags errors, except for Haiku on frames.** Accuracy without vs. with thinking:
  - Haiku, captions: 0.84 vs. 0.62.
  - Haiku, frames: 0.61 vs. 0.61.
  - Sonnet, captions: 0.84 vs. 0.64.
  - Sonnet, frames: 0.93 vs. 0.63.
  - The 100 development questions (other videos) show the same pattern: Haiku on captions 0.85 vs. 0.51, Haiku on frames 0.67 vs. 0.77, Sonnet on captions 0.86 vs. 0.50, Sonnet on frames 0.94 vs. 0.67.
- **Routing illustration (post hoc):** Haiku answers from captions and escalates to Sonnet on frames when it thought (45% of questions). That gives 0.774 at $0.0022 per question, against Sonnet on frames for every question at 0.798 and $0.0041.
- Adopted (G2, 2026-10-20): an exploratory log entry, and the analysis moved from the session scratchpad into `reader.py`.

## 7. Paper outline (long draft, 8 pages; cut to 4 if we go short)

- **Title candidates:** "Caption Once, Ask Cheaply: …"; "Who Writes the Caption Matters: …". Settle after C2 and C3.
- **1. Introduction:** the caption index as a one-time cost; the question (accuracy at query cost, and why). Findings in §3's order: the captioner effect (P3, C3), the bound on the modality effect (P2), then the cost result (the reader downgrade).
- **2. Setup:**
  - TempCompass multiple choice; 5 s windows; the arms. Readers Haiku and Sonnet; captioners Sonnet (main) and Gemini 3.8 Flash (second family).
  - The four effects and how the arms isolate them (§2, §4).
  - Pre-registration: 100 development questions, 297 confirmatory questions; statistics; which analyses are exploratory (C2, C3).
- **3. Theory, short:** Prop. 1 (garbling bound: a caption of the same frames can add usable information, never information about the video) and Prop. 2 (break-even).
- **4. Results:**
  - **Table 1:** accuracy and the full cost per question, both readers.
  - **Table 2:** the decomposition, Claude and Gemini chains side by side.
  - **Figure 1:** accuracy vs. list-price dollars per question (and total tokens), with the one-time captioning cost amortized at Q = 1, 10 and 100.
  - One paragraph on aspects; the per-aspect tables go to the appendix.
- **5. Cost and break-even:** Q\*, and why the reader downgrade, not token savings, is what pays.
- **6. Related work:** IsoBench, TokenSwap, Prism, LLoVi, CapMem, CFD, VidCtx, Sprout, LoHi, plus the lit pass.
- **Limitations:** see §8.
- **Appendix:** prompts; the development phase (K0–K2) and how it set the hypotheses; amendments, deviations and the term mapping (D2); per-aspect tables; release details.

## 8. Reviewer risks and our answers

| Risk | Answer |
|---|---|
| Closed readers, run through the Claude Code CLI (its system prompt is included; no temperature control) | Pinned model IDs; isolation checked on every call; release prompts, captions and raw outputs. Optional open-weights reader (§6). |
| Frames at 224 px are too small | Comparison at matched or lower query cost; optional 448 px arm (§6) |
| One benchmark | Scope; stated in the limitations |
| Captioner effect = larger visual budget | C3 matches visual tokens exactly by construction; C2 matches them about for Gemini |
| Captioner and reader from one family (Claude reading Claude) | The Gemini chain replicates the captioner effect across families |
| Why not ask the strong model directly? | It is in the design: Sonnet on frames, compared on accuracy and on cost per question |
| Haiku 5 frames: 0.70 on development, 0.61 on confirmatory | Report as is; the headline is P3, not S3; the development data only set the hypotheses |
| Multiple choice only | TempCompass's other formats are out of scope; stated |
| Post-hoc analyses | Every entry in the log is dated and marked pre-registered or exploratory; C2 and C3 are exploratory; the pre-registration is made publicly verifiable (§5, step 9) |

## 9. Timeline

The dated goals are in §0 (G1–G12). They replace the week-by-week timeline of 2026-10-19. TempCompass is CC BY-NC, which the release package (G11) must respect.

## 10. Budget

- **Gemini:** $11.24 logged against the approved $13, with C2 complete. No other Gemini spending is planned.
- **Claude:** the 7-day quota was 92% at the last read (2026-10-20, before the reset). It resets on Tuesdays at 17:00 UTC, next on Oct 20. In the week after that reset: C3 ≈ 10–15 points; Sonnet on its own captions ≈ 4; the 448 px Haiku arm and the effort sweep little; the literature pass uses this session's quota too. That is ≈ 20–30 points, which fits in one week.
