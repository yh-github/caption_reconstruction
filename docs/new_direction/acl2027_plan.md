# ACL 2027 short paper: plan and next steps

Written 2026-10-19, to pick up the next day. The experiment record is in `research_log.md` (C1 *Result*, C2). The session history is in `docs/handover.md`. Each item below is marked **decided** (agreed with the user) or **proposed** (my recommendation, still open).

## 1. Target

- **Decided:** ACL 2027, through the **January 2027 ARR cycle**. ARR's dates page lists ACL 2027 for January 2027. Secondary sources give **Jan 4, 2027**; confirm against the ACL 2027 call when it is published.
- **Proposed:** a short paper (4 pages, plus limitations, references and appendix). One pre-registered study with one clear result fits that format. No second benchmark.
- Fallbacks: Findings, or committing the ARR reviews to a later ACL-family venue. TMLR (rolling) would suit a longer version later.
- **Check:** ARR's rules for the submission (Responsible NLP checklist, reviewing duties for authors, anonymity).

## 2. What we have (C1: pre-registered, 297 held-out TempCompass questions)

| Reader, input | Accuracy | Video tokens per question |
|---|---|---|
| Haiku, question only | 0.41 | 0 |
| Haiku, 1 frame per 5 s window | 0.58 | 164 |
| Haiku, 5 frames per window (224 px) | 0.61 | 495 |
| Haiku, its own captions of those frames | 0.64 | 258 |
| Haiku, Gemini captions of those frames | 0.79 | 204 |
| **Haiku, Gemini captions of the video** | **0.74** | **211** |
| Sonnet, 5 frames per window | 0.80 | 495 |
| **Sonnet, Gemini captions of the video** | **0.79** | **211** |

Findings:
1. **A cheap reader does better on a strong model's captions than on the frames:** Haiku gains +12.5 points over 5 frames, with 43% of the tokens. At about the tokens of 1 frame, it gains +15.5.
2. **A strong reader loses nothing on captions:** Sonnet −0.007 [−0.05, +0.04], with 43% of the tokens.
3. **A cheap reader on captions comes within 6 points of a strong reader on frames:** −0.06 [−0.11, −0.01].
4. **The gain comes from the captioner, not the text format.**
   - Captioner effect, P3: +0.15 [+0.09, +0.21]. Passed.
   - Format effect, P2: +0.03, at most +0.09, against IsoBench's 15–29 points in favour of text.
5. **Which aspects captions lose: not confirmed** (P1: −0.10 [−0.23, +0.03]). Speed is the only consistent loss (Sonnet −0.10); exploratory.

**Proposed claim, pending C2:** a strong captioner's question-blind captions carry its knowledge to a cheap reader. C2 decides whether "knowledge" is the right word, or whether the effect comes from the 27× larger visual budget Gemini spent on the frames.

## 3. Next steps, in order

1. **User: raise the AI Studio monthly spend cap** (https://ai.studio/spend).
   - C2 captioning stopped at 450 of 761 windows on that cap (log, C2 *Run status*).
   - The rest costs ≈ $0.90. The logged total would then be ≈ $11.2, within the approved $13 and the run's `--max-usd 11.5`.
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
4. **Run `reader.py c2` once.** Record its full output as C2's *Result* in the log, each prediction next to its result. Then word the claim by C2's mapped outcomes:
   - C2a positive: "knowledge".
   - C2a null and C2b positive: "spend the visual budget once, at captioning time".
5. **Cost account and break-even** (no new runs).
   - **Inputs:**
     - Gemini caption cost per window, from `results/piece_captions/calls.jsonl` (≈ $0.0028 on the video path, mostly thinking).
     - Reader input and output tokens per arm, from `results/index_qa/claude/*.jsonl`.
     - Claude list prices for Haiku and Sonnet 5.5. Look them up (claude-api skill); don't quote them from memory.
   - **Formula:** Prop. 2 in `research_plan_caption_index.md` (Q\*(α) = C(τ) / (c\*_f(α) − c\*_τ(α))). Compute it for α at the Haiku-frames, Haiku-captions and Sonnet-captions accuracies, with bootstrap CIs.
   - **Output:** cost per question per arm, plus Q\*.
6. **Optional arms: the user decides** (see §5).
7. **Literature pass, after the weekly quota reset.** Run `final_search_brief.md`, reordered:
   - **Q5** (captioner knowledge from identical input) first, then Q1, Q2, Q4, Q6 and Q7. Q3 (aspects) last, since P1 failed.
   - **Deliverable:** §10 of `related_work_caption_index.md` and log entry N2.
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
8. **Write** (§4), starting with the outline and the figures.

## 4. Paper outline (4 pages)

- **Title candidates:** "Caption Once, Ask Cheaply: …"; "Who Writes the Caption Matters: …". Settle after C2.
- **1. Introduction:** the caption index as a one-time cost; the question (accuracy at query cost, and why); the three findings.
- **2. Setup:**
  - TempCompass multiple choice; 5 s windows; the arms; readers Haiku and Sonnet; captioner Gemini 3.8 Flash.
  - The decomposition into coverage (source), captioner, format (and visual budget, from C2).
  - Pre-registration: 100 development questions, 297 confirmatory questions; statistics.
- **3. Results:**
  - **Table 1:** accuracy and video tokens, both readers.
  - **Figure 1:** accuracy vs. video tokens per reader.
  - **Table 2:** decomposition (C1 terms plus C2 terms).
  - One paragraph on aspects.
- **4. Cost and break-even:** cost per question, Q\*, and the reader downgrade.
- **5. Related work:** IsoBench, TokenSwap, Prism, LLoVi, CapMem, CFD, VidCtx, Sprout, LoHi, plus the lit pass.
- **Limitations:** see §6.
- **Appendix:** prompts; the development phase (K0–K2) and how it set the hypotheses; amendments and deviations; per-aspect tables; release details.

## 5. Optional arms (proposed; the user decides)

| Arm | Why | Cost | Recommendation |
|---|---|---|---|
| Haiku on 5 frames at 448 px (144 tokens each, ≈ 1,870 video tokens per question) | Answers "you starved the frames". Shows where frames catch up on the accuracy–tokens curve. | 297 Haiku calls (little quota). Frames re-extracted at 448 px (`tempcompass.py` has `LONG_EDGE = 224`). A log entry (C3) with predictions before the run. | **Do it** |
| Sonnet, question only and 1 frame (C1's skipped secondary arms) | Completes Sonnet's curve and gives its retention | 600 Sonnet calls, ≈ 7–8 points of the 7-day quota; after the reset; C1 Amendment 3 | Nice to have |
| Open-weights reader on Kaggle (e.g. Qwen2.5-VL-7B) on the main arms | Answers "all readers are closed models"; reproducible | Free GPU; a few sessions of work | Decide after C2 |

## 6. Reviewer risks and our answers

| Risk | Answer |
|---|---|
| Closed readers, run through the Claude Code CLI (its system prompt is included; no temperature control) | Pinned model IDs; isolation checked on every call; release prompts, captions and raw outputs. Optional open-weights reader (§5). |
| Frames at 224 px are too small | Comparison at matched or lower query cost; optional 448 px arm (§5) |
| One benchmark, one captioner | Scope of a short paper; stated in the limitations |
| Captioner effect = larger visual budget | C2 |
| Haiku 5 frames: 0.70 on development, 0.61 on confirmatory | Report as is; the development data only set the hypotheses |
| Multiple choice only | TempCompass's other formats are out of scope; stated |
| Post-hoc analyses | Every entry in the log is dated and marked pre-registered or exploratory; C2 is exploratory |

## 7. Timeline to the deadline (Jan 4, 2027, to be confirmed)

| When | What |
|---|---|
| Week of Oct 19 | Finish C2; cost account; decide the optional arms |
| Week of Oct 26 | Optional arms; literature pass after the quota reset; outline and figures |
| November | Full draft by mid-November |
| December | Team review and revisions; ARR checklist; release package (code, captions, outputs; TempCompass is CC BY-NC) |
| By Jan 4 | Submit |

## 8. Budget

- **Gemini:** $10.29 logged against the approved $13. C2 needs ≈ $0.90 more. The AI Studio project cap is separate, and currently blocking.
- **Claude:** the 7-day quota was 86% at the last read. Haiku runs barely move it. Sonnet moves it about 1 point per 80 calls. Check the reset date before planning Sonnet runs or the literature pass.
