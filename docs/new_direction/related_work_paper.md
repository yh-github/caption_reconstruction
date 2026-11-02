# Caption index paper: literature review (fifth pass, 2026-11-02)

This review is organized by the paper's claims as they stand in `docs/paper/acl2027/caption_index.tex` (commit `047ea80`). It replaces `related_work_caption_index.md` as the review the paper is written from. That file stays as background: the third pass (state of the art, 2026-10-15) and the fourth pass (§10, 2026-10-21, questions Q1–Q7 of `final_search_brief.md`). The research log entry is N3.

**Why a new pass and new questions.** The brief of 2026-10-19 asked about the claims of draft v1. Since then the paper has gained the readers' tie, question-aware captions (C13), an audit of wrong answers, a decision rule (A6–A8), a cost account with thinking tokens and a long-video projection, and a contamination check (A4). P1 (aspects) failed and is no longer a claim. The questions below follow the paper's five contributions.

**Marks.** *(full)* full text read; *(skim)* abstract, setup, arms and main results read; *(searched)* full text fetched and searched for the controls that would raise its level (self-captions, identical frames, matched tokens, TempCompass), with no hit, so the level rests on the abstract; *(abs)* abstract only; *(earlier)* checked in an earlier pass, not re-read. "In bib" means the key is already in `caption_index.bib`.

**Full-text update (2026-11-02, later the same day).** OpenReview serves ICLR 2027 PDFs only behind a browser check, so scripts get a verification page. The author downloaded the two close submissions, *Revisiting Caption Gains* and *Read First, Look on Doubt*. Both are now *(full)*, and both read differently from their abstracts (§3, R3 and R7). The PDFs are in `local/papers/iclr2027/`, which is gitignored. The other ICLR 2027 submissions remain *(abs)*. The arXiv papers below that were *(abs)* now have full text (fetched with `scripts/lit_search/fetch_fulltext.py`). After that check, *Semantic Serialization* moves from close to related, and no level rises.

## 1. Bottom line

**No scoop on any claim.** Four new papers are close and should be cited. One of them (REEL) also uses our information control.

| | Claim (paper section) | Verdict | Closest work |
|---|---|---|---|
| R1 | Decomposition into source, visual budget, captioner and modality, with the captioner given exactly the reader's frames (contribution 1; §3) | **Open, narrower.** REEL (Oct 28) gives a text memory, a visual memory and a KV memory exactly the frames the frame path sees. It has no captioner swap, so no captioner term, and one model writes and reads. | REEL, CapMem, Caption This |
| R2 | Captioner effect at identical input, across model families (§4.1) | **Close, not scooped.** Captioner swaps in front of a fixed reader are common. None has a self-caption arm, the reader's frames and matched visual tokens across families. | CapMem, Prism, MetaCaptioner, CapRL, OACap, Many Captioners or One? |
| R3 | No reader gains measurably from its self-captions over its frames (§4.2) | **Close (images, long video); the sign varies with the reader.** REEL: self-text and frames have nearly the same mean accuracy, but 20–29% of answers flip. Revisiting Caption Gains (images, seven open VLMs): on Visual7W, removing the image from caption + image costs every model accuracy (11.3 points on average), so no model gains from its own captions alone there. | REEL, Revisiting Caption Gains, Caption This, GEASS |
| R4 | Readers' tie: Haiku and Sonnet within 0.02 on Sonnet's captions, 0.19 apart on frames (§4.1, §5) | **Close in text RAG, open for video.** A fixed compressed text shrinks reader upgrades from 31.8 to 7.8 points, and weaker readers gain more. | Compression Is Not Evaluation-Neutral; MemoryTax; What a Memory Can and Cannot Carry |
| R5 | Cost per question with thinking tokens, break-even Q_BE with a reader downgrade, projection to long videos (contribution 3; §5) | **Open.** REEL amortizes GPU time over questions for one model, with no cheaper reader. Nobody solves Q_BE with measured costs and a reader downgrade. | REEL, CapMem, CFD, Sprout, SAFE-Cascade |
| R6 | What question-blind captions lose (copies of a clip, reversals); question-aware captions recover the comparisons but not the reversals; most wrong answers come from captions that leave the question undecided (contribution 4; §4.3–4.5) | **Open on video.** Related work gives vocabulary (target inclusion vs. reader recovery; representation failure; task scope loss) but no question-aware vs. question-blind comparison on identical frames. | REEL, Beyond Visual Token Count, Keep the Question, VidCtx, MotionBlind |
| R7 | Decision rule: the question's aspect and the model pair predict whether index or frames is right; video and caption scores add nothing (contribution 5; §4.6) | **Close on long video; open in our form** (a frozen formula tested on new questions, with a null for video scores). | Read First, Look on Doubt; CFD; EARS; MMR-Bench; SwitchMem |
| R8 | Contamination check: questions asked of another video with another answer (A4); question-only arm (Limitations) | **Related only.** The A4 design was not found elsewhere. | Most of the Gain Is Not Video; CleanSlide; Benchmark Déjà Vu; TVBench |
| R9 | New work since 2026-10-18 | **4 close, about 20 related** (§4). | REEL; Revisiting Caption Gains; Read First, Look on Doubt; Compression Is Not Evaluation-Neutral (June, missed before) |

**What has to change in the paper** (details in §5):
1. Cite REEL next to CapMem as the closest work. The information control for self-captions is no longer ours alone. What stays ours is the captioner swap on identical input, across families, on short video, and the cheaper reader.
2. Frame the readers' tie with *Compression Is Not Evaluation-Neutral*: a fixed text artifact compresses reader differences. That paper predates ours (June 2026), so it must be cited.
3. Add *Revisiting Caption Gains* to "Text or pixels", and *Read First, Look on Doubt* to the decision rule (§5 items 3–4). *Semantic Serialization* and EARS are optional.
4. Everything else is optional or appendix material.

## 2. Contemporaneity (ARR)

ARR treats papers that appear less than three months before the submission deadline as contemporaneous; authors need not compare with them in detail (check the current wording in the ACL policy on citation and comparison). With a deadline of about January 4, 2027, the line is about October 4, 2026.
- **Contemporaneous:** REEL (Oct 28) and the other October arXiv papers. The ICLR 2027 submissions went public on OpenReview after the full-paper deadline of September 25, at the edge of the window. Cite them in a sentence, with no detailed comparison needed.
- **Not contemporaneous:** *Compression Is Not Evaluation-Neutral* (arXiv June 20, 2026; v2 October 2), which a reviewer can fairly expect, and *Semantic Serialization* (August 29, 2026), which is related only after the full-text check.

## 3. Per claim

### R1. The decomposition and the information control

**Verdict: open, narrower than before.** No paper splits a caption effect into source, visual budget, captioner and modality, on video or images. But giving every representation exactly the same frames is now published for self-written text (REEL).

- **[REEL](https://arxiv.org/abs/2610.38158)** (Liang & He, WACV 2027 Datasets and Evaluation track; arXiv 2026-10-28) *(full, §1–4; appendix headings only)*. **Close.**
  - Setting: long video (CG-Bench MQ-50: 50 videos, 6 questions each; HD-EPIC-51: 51 videos, 20–86 questions), six open 4–12B VLMs (Qwen3-VL-8B/4B, InternVL3/3.5-8B, Gemma-3-12B, MiniCPM-V-4.5), 64 uniform frames.
  - Arms: Frames (reprocess per question); Text (the same backbone writes a question-blind record, ≤ 24 words per frame, then answers from it); Visual (cached encoder outputs); Prefix State (cached KV). All see the same timestamps, "so that a change in answers can come only from what the system remembered".
  - Findings: Text and Frames differ in correctness on 19.7–28.7% of questions even when mean accuracy is nearly equal; Visual and Prefix State flip at most 2%. Changing only the writer's priorities flips 6.8–9.8%. On HD-EPIC, Text hurts object-content questions and helps movement questions in all six backbones. GPU-time break-even per question horizon; Prefix State is the cheapest memory that keeps almost all of Frames' correct answers.
  - Difference from us: one model writes and reads, so it has the modality term but no captioner, source or visual-budget term; no cross-family arm, no cheaper reader, no list-price or thinking-token cost; long video with open models only.
- **[CapMem](https://arxiv.org/abs/2609.17688)** (EMNLP 2026) *(earlier, full)*. In bib. Six Qwen captioners of the same frames and own captions; matched-frame CaptionQA–VideoQA control; long egocentric video. Does not separate source from visual budget, cross families or price a cheaper reader.
- **[Caption This, Reason That](https://arxiv.org/abs/2505.21538)** (NeurIPS 2025) *(earlier, full)*. In bib. Images, synthetic; image vs. self-captions vs. ground-truth captions, same model.
- **Pre-registration is no longer rare in this area:** ASP (2608.22975), *Semantic Serialization* (2609.29601, "prospectively frozen criterion"), *Compression Is Not Evaluation-Neutral* (a plan fixed in advance), and an ICLR 2027 submission titled *A Pre-Registered Audit of Hierarchical Retrieval* (OpenReview `dHrg5BhrsI`). The paper should not present pre-registration as unusual. It does not now (checked: no "first" or novelty wording around it in `caption_index.tex`).

### R2. Captioner effect at identical input

**Verdict: close, not scooped** (as in N2a, now with two more image or agent papers).

- In bib: [Prism](https://arxiv.org/abs/2406.14544), [MetaCaptioner](https://arxiv.org/abs/2510.12126), [CapRL](https://arxiv.org/abs/2509.22647), [LLoVi](https://arxiv.org/abs/2312.17235), CapMem. All vary the captioner in front of a fixed text reader. None has the reader's own frames and self-captions as further arms.
- **[OACap](https://arxiv.org/abs/2610.29327)** (2026-10-21) *(skim)*. Related. An agentic captioner that requests more observations. Its question-blind whole-video captions improve a fixed Qwen3.8-27B text-only reader on Video-MME (short, medium, long) and ActivityNet-QA. The single-pass baselines are Qwen3.5 captioners of different sizes. A captioner effect with a fixed reader. The only frame arm is the reader answering from the video, marked as an oracle with "different input" and left out of the comparisons. There is no self-caption arm, and the captioners do not get identical frames.
- **Many Captioners or One? The Price and Payout of Mixing Frozen VLMs** (ICLR 2027 submission, OpenReview `ZbVM5aOJRs`) *(abs)*. Related. 30 VLMs caption the same ImageNet images. The best captioner changes with the use and the prompt, and is hard to know in advance. Images, training uses; not QA from frames.
- **Our claim stays:** a captioner effect at identical input (C3: Sonnet captions exactly Haiku's frames) and at matched visual tokens (C2), with the reader's frames and self-captions as further arms, across Claude, Gemini and Qwen.

### R3. The modality effect

**Verdict: close on images and long video; the sign depends on the reader.** Our result (at most +0.09 for Haiku; Sonnet and Qwen lose 0.09–0.12) fits the pattern.

- **REEL** (R1): self-text ≈ frames in mean accuracy, with many flips both ways; type-dependent. The closest video result, on long video with open models.
- **Revisiting Caption Gains When the Image Remains Available** (ICLR 2027 submission, OpenReview `imjcY1Z6JP`) *(full, from the PDF)*. **Close (images), on one experiment.**
  - Setting: seven open VLMs (GLM-4.1V, InternVL3.5, Gemma-4-12B, Qwen3.8-27B, Qwen2.5-VL-3B, InternVL2-8B, Ministral-3-14B). Each writes one question-blind generic caption per image ("one short sentence"), which is then frozen. Datasets: Visual7W (2,900 images) and SLAKE (333 medical images), open-ended answers scored by a gpt-4o-mini judge. The paper reports no cost.
  - The main question is not ours. It asks what a caption adds *next to* the image, compared with length-matched neutral text in the same message slot. The answer is at most +0.35 points (Visual7W, caption in the history), and much of the gain from caption history comes from adding any text.
  - The close part is the image-removal replay. It reruns every caption and its neutral texts without the image. Removing the image costs the captions 11.3 points on Visual7W (history delivery, average over models) and costs neutral text 40.5. On Visual7W every model loses accuracy without the image, in both deliveries (Figure 4, about 9–17 points). On SLAKE one model (InternVL2-8B, inline delivery) gains slightly without it; the paper's text claims only that every model's caption loss is smaller than its neutral-text loss. So on Visual7W, no model does as well on its own question-blind captions as with the image. *(Corrected 2026-11-02 after re-reading Figure 4; the first full-text note said every model loses on both datasets.)* The comparison is caption alone vs. caption + image, but caption + image is within 1 point of the image alone (+0.77 on Visual7W, from their Q1 and caption-history rows). Our result is the video analogue, with a captioner swap added: no reader gains from its self-captions, and two of three lose.
  - Difference: images, open-ended QA, one-sentence captions, self-captions only (no captioner swap, no identical-input control across models), no cost.
- **[Free the Language Model From the Vision Encoder: Semantic Serialization](https://arxiv.org/abs/2609.29601)** (Xu & Sankar, arXiv 2026-08-29) *(skim)*. **Related (was close on the abstract).** The text is not a caption. Fine-tuned YOLO detectors and range estimation feed deterministic code that writes the scene state in the decision's vocabulary ("under 5 m", "left"). The interface is task-aware by design. A text-only Qwen2.5 reading it beats a same-size zero-shot Qwen2.5-VL on the image (0.789 vs. 0.746 at 7B; 0.767 vs. 0.691 at 3B). The gain sits in the two families that reduce to arithmetic over the state. Most of it (71–83%) is the pre-banded vocabulary, and a raw object list alone loses. The advantage grows as the reader shrinks to 1.5B and reverses at 0.5B. It is a reader-dependent sign like ours, but for engineered, task-aligned text rather than question-blind captions. One aside fits our captioner result: at fixed ground-truth content, its serializer beats a prose caption.
- **[Position: Reasoning After Perception Means Reasoning Without Vision](https://arxiv.org/abs/2507.16863)** (v2 2026-06) *(searched)*. Related. For tasks hard to verbalize, text-only reasoning cannot fix perceptual failures (Turing Eye Test). Background for the reversals.
- **Stop Describing, Start Answering** (ICLR 2027 submission, `UyUWSPyoTO`) *(abs)*. Related. VLMs answer better after describing the image ("description reliance"; a 16.6-point caption–direct gap on Qwen3-VL). Relevant to C5 (describe, then answer).
- In bib: [IsoBench](https://arxiv.org/abs/2404.01266), [TokenSwap](https://arxiv.org/abs/2607.28640), [GEASS](https://arxiv.org/abs/2605.01733), [Seeing and Solving](https://arxiv.org/abs/2609.33694). Not in bib, from earlier passes: the missing-modality gap (2605.12517).

### R4. The readers' tie

**Verdict: close in text RAG and agent memory; open for a video caption index.** The paper calls it a readers' tie; it is reported as new and cites only CapMem, where the stronger reader *does* gain on fixed captions.

- **[Compression Is Not Evaluation-Neutral: Fixed RAG Compression Can Distort Reader Comparisons](https://arxiv.org/abs/2606.21807)** (Panthi & Abdelfattah, arXiv 2026-06-20, v2 2026-10-02; also an ICLR 2027 submission, `MpDEPJO8Aa`) *(full, §1–10; appendix headings only)*. **Close.**
  - Text RAG. One stored compressed text per question goes to every reader, verified by content hash, with 8–20 readers from up to eleven families (7B open models to hosted frontier models). HotpotQA and MuSiQue are confirmatory under a plan fixed in advance. LongMemEval, NQ-Open and TriviaQA (added later) are replications.
  - On HotpotQA the lowest and highest readers are 31.8 points apart on raw evidence and 7.8 apart on one RECOMP output ("upgrade retention" 24.5%). On MuSiQue the gap nearly vanishes (15.4 → 0.6 EM). Across 13 panels, retention runs from 4.5% to 83.6%. Lower readers gain more, after correcting for shared sampling noise. Rescue and damage coexist: on HotpotQA 19.1% of row–reader pairs are rescued and 35.7% of correct ones damaged. Under compression, reader pairs also change order more often than in raw replays (+16.7 to +32.7 points), and on MuSiQue the F1 winner changes. The average accuracy and the retention diverge: a shared summary raises the mean by 13.6 points but keeps 18.1% of the upgrade. Their remedy is to report raw results and the surviving upgrade on the same compressed text ("Fixed-Artifact Reader Replay").
  - What it lacks for us: no images or video, and no frames-vs-text question. Its "raw" arm is the uncompressed text, not another modality. Compressor calls are left out of its cost estimates.
  - Ours is the video analogue: on Sonnet's captions, Sonnet's 0.19 lead over Haiku on frames shrinks to 0.02. In their terms, the upgrade retention is about 10%; compute it from the reports before quoting it.
- **[MemoryTax](https://arxiv.org/abs/2610.29823)** (2026-10-21) *(full)*. Related (R4, also R2 and R5). Agent memory on one LoCoMo conversation (152 questions, one run per cell), with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5. Retrieval calls no LLM, so the reader of each store can be swapped. On Zep's graph written by GPT-4o-mini the three readers tie (43%, 43%, 45%); on the graph Opus wrote they spread out (66%, 71%, 87%). Mem0 shows the same order with smaller gaps. So the writer sets the ceiling (the captioner effect, in text memory), and readers tie when the store leaves nothing for a stronger reader to use. Our tie is on Sonnet's captions, a strong writer's index, so it is not the same situation. MemoryTax also gives a break-even in questions with list prices: Mem0's build pays for itself after 372–596 questions on the history, against LoCoMo's 152. One model writes and reads there, so there is no reader downgrade.
- **What a Memory Can and Cannot Carry** (ICLR 2027 submission, `cOvTv63ZI7`) *(abs)*. Related. A writer × reader matrix over five models finds no author effect: the one model that prose memory did not harm was its writer, but the matrix does not support an own-words advantage. This agrees with C14 (own-words interaction −0.030 [−0.088, +0.027]).

### R5. Cost and break-even

**Verdict: open.** No paper solves for the number of questions per video at which captioning pays, with a cheaper reader, from measured costs.

- **REEL** amortizes GPU time over a question horizon and reports the first horizon at which a memory is cheaper and keeps answers. It also shows that token prices favour input shortening and ignore reuse of computation. One model; no reader downgrade; GPU-seconds, not list prices.
- **Read First, Look on Doubt** (ICLR 2027 submission, `6FPJ94irae`) *(full, from the PDF)*. 93.7% of long-video questions are answered without a frame (GPT-5). It prices queries in calls and tokens against WorldMM with the same reasoning model (Video-MME Long: 5.0 vs. 24.4 calls and 17.7k vs. 29.6k tokens per question), and memory construction in calls (226 vs. 1,209 per hour of footage). It solves no break-even, and the build cost is kept out of the query-time comparison by design.
- **MemoryTax** (R4): a break-even in questions for an agent memory build, with list prices, one model writing and reading.
- **IR³: Index Once, Retrieve, Restore, and Reduce** (`n3SeGUrAX6`), **Write Once, Query Many / Video-EMem** ([2610.24537](https://arxiv.org/abs/2610.24537); *searched*), **Multiple Questions, Shared Evidence / QShare** (`N6TRQNR42K`) *(abs)*: long-video reuse across questions, reported as latency, storage or visual-token budgets. Related.
- In bib: CFD, CapMem, Sprout, SAFE-Cascade, FrugalGPT, RouteLLM, AI Agents That Matter, MIT³, Industrial ReMEmbR.

### R6. What question-blind captions lose, and question-aware recovery

**Verdict: open on video.** Nobody compares question-blind and question-aware captions written from identical frames on minimal pairs (stacked copies, reversals). Related work offers terms the paper could use.

- **Beyond Visual Token Count: Target Inclusion and Reader Recovery** (ICLR 2027 submission, `s9lUh0Sl2N`) *(abs)*. Related. Splits failures into *target inclusion* (did the queried fact reach the reader?) and *conditional recovery* (can the reader recover a fact that is present?). Compact query-independent text recovers every target it retains but loses on inclusion. Our audit's main finding, that captions leave the question undecided, is an inclusion failure in these terms. Using the terms is a terminology decision for the author.
- **REEL**: "representation failure" (the frame was seen, the detail not kept), and writer priorities change which details survive.
- **Keep the Question: Task-Preserving Memory for Visual PDF Reading** (`vZfzyOh23P`) *(abs)*. Related. "Task scope loss": a local observation replaces a required comparison, count or sum. The stacked comparisons are a video case of this.
- **Reversal:** **Video Reversal Curse** (`uA4RIBZqBX`; uses TempCompass) and **Tracing Arrow-of-Time** (`UeqjOP0eqf`) *(abs)*: models ignore reversals, and temporal information is lost between encoder and LLM. Pixels only. In bib already: Seeing the Arrow of Time, the reversal-drop paper, Which Way Did It Move?, MotionBlind, TimeBlind.
- In bib: VidCtx (question-aware captions beat generic ones by 1–2 points on NExT-QA).

### R7. The decision rule

**Verdict: close on long video; open as a frozen predictive formula on short video.** Routing between a text index and frames per question is published (CFD; Read First). Neither tests whether scores of the video or the captions predict the choice, and neither freezes a rule and tests it on new questions.

- **Read First, Look on Doubt: A Text-First Framework for Long-Video QA** (ICLR 2027 submission, `6FPJ94irae`) *(full, from the PDF)*. **Close on the routing, not on what the text is.**
  - Setting: five long-video benchmarks (EgoLife, Ego-R1, HippoVlog, LVBench, Video-MME Long; 4,194 four-way questions), with GPT-5 or MiniMax-M3 as the reasoning model.
  - The text is mostly speech. The offline memory joins diarized ASR with 10 s captions written from 1 fps contact sheets (by MiniMax-M3, or EgoLife's released captions), then adds coarser summaries. Stage 0 answers from the transcript alone when it fits in 32k tokens (39% of questions), and the captions add +7.0 on EgoLife (Table 3).
  - **The headline "+14.6 → −13.7" is not about captions.** In that modality ablation (MiniMax-M3, Video-MME by duration, 16 uniform frames), "text" is the subtitle track. The value of adding frames to subtitles falls from +14.6 on 0–2 min clips to −13.7 at 45–60 min. Do not cite it as a caption result.
  - The routing: answer from the text memory, re-derive the answer from the same evidence, and look at frames only if the two derivations disagree. The gate fires on 10% of EgoLife questions (GPT-5) and beats a confidence threshold (67.4 vs. 64.6, firing less often). On the 237 questions it escalates across the five benchmarks (GPT-5), accuracy goes from 34.2 to 51.1. A third text draw with no new evidence reaches 43.0, so part of the lift is re-sampling. With the gate held open (EgoLife, MiniMax-M3), looking adds +8.1 on the questions the gate would escalate and +0.0 on those it passes, at 3.19× the vision calls.
  - Relation to our rule: both choose text or frames per question. Their signal is the reader's own instability on the text, which costs one more text call. Ours uses only what is known before reading (aspect and model pair) and finds no help from video or caption scores. Their positive result suggests the signal our rule lacks within a pair is answer-time instability; we did not test it. Long video, no frozen-rule test on new questions, no comparison at matched cost of text vs. frames.
- **[EARS](https://arxiv.org/abs/2610.35536)** (2026-10-26) *(skim)*. Related. A query-free frame selector from optical-flow motion and audio energy, tested on anomaly classification (XD-Violence, UCF-Crime). Its "routing" is per video, between its selector and uniform sampling, not between text and frames. A router on eleven modality-intrinsic statistics scores below never routing, and no statistic predicts the per-video gain. This is the same kind of null as ours for motion and cut scores (A7), in a different choice.
- **[MMR-Bench](https://arxiv.org/abs/2601.17814)** (2026-01) *(searched)*. Related. Per-query routing across MLLMs under cost models; a routed system beats the best single model at about 33% of its cost.
- **Disagreement-Guided Weak-to-Strong Routing** (`gvNBXwLEt1`) *(abs)*. Related. Routing to a stronger model by model disagreement, without correctness labels.
- In bib: CFD (visual-need router; routing accuracy 68.5% against an outcome oracle), SAFE-Cascade, RouteLLM. Not in bib: SwitchMem (2610.17928; memory modality chosen per step for GUI agents).

### R8. Contamination and question-only baselines

**Verdict: related only.** The A4 design (the same question and options asked of another TempCompass video with another answer) was not found elsewhere.

- **Most of the Gain Is Not Video** (ICLR 2027 submission, `QybPLnt1Ew`) *(abs)*. An adapter trained on the question text alone recovers 89–91% of a fine-tuned video-QA gain. Supports reporting a question-only arm.
- **CleanSlide / Contamination, Prior, or Evidence?** (`o3hO7x2lXq`) *(abs)*. Audits blind solvability and uses counterfactual pairs with opposite findings: the same logic as A4, in pathology.
- **Benchmark Déjà Vu** (`DEo5cLf1K3`) *(abs)*. Multiple-choice contamination detectors with explicit nulls, across 35 models.
- From earlier passes: TVBench (2410.07752), TempCore (2509.01167).

## 4. New papers in this pass

Levels per claim: **close** (same question in another domain or setting, cite and state the difference), **related** (shares a component), as in the brief. All dates are first posting.

| Paper | Date, venue | Level (claim) | Key point | Read |
|---|---|---|---|---|
| [REEL](https://arxiv.org/abs/2610.38158) | 2026-10-28, WACV 2027 | Close (R1, R3, R5) | Same frames for Text, Visual, Prefix State and Frames; Text flips 20–29% of answers at near-equal mean accuracy; GPU-time amortization | Full (§1–4) |
| [Compression Is Not Evaluation-Neutral](https://arxiv.org/abs/2606.21807) | 2026-06-20 (v2 10-02); ICLR 2027 sub. | Close (R4) | A fixed compressed text shrinks reader upgrades 31.8 → 7.8 points; weaker readers gain more | Full (§1–10) |
| Revisiting Caption Gains When the Image Remains Available (`imjcY1Z6JP`) | ICLR 2027 sub. | Close (R3) | Main question: caption next to the image (≤ +0.35 over neutral text). Image-removal replay: on Visual7W every model loses without the image (−11.3 on average); images | Full (PDF) |
| Read First, Look on Doubt (`6FPJ94irae`) | ICLR 2027 sub. | Close (R7); related (R5) | Speech + caption memory on long video; frames only when two text answers disagree. The +14.6 → −13.7 figure is subtitles vs. frames, not captions | Full (PDF) |
| [Semantic Serialization](https://arxiv.org/abs/2609.29601) | 2026-08-29 | Related (R3; was close) | Detector-to-text serializer in the task's vocabulary beats a same-size VLM; reverses at 0.5B | Skim |
| [MemoryTax](https://arxiv.org/abs/2610.29823) | 2026-10-21 | Related (R4, R2, R5) | Readers tie on a weak writer's store, spread on a strong one's; build break-even 372–596 questions | Full |
| What a Memory Can and Cannot Carry (`cOvTv63ZI7`) | ICLR 2027 sub. | Related (R4) | Writer × reader matrix; no author effect | Abs |
| Many Captioners or One? (`ZbVM5aOJRs`) | ICLR 2027 sub. | Related (R2) | 30 captioners of the same images; the best one depends on the use | Abs |
| [OACap](https://arxiv.org/abs/2610.29327) | 2026-10-21 | Related (R2) | Agentic captioner; better captions for a fixed text-only reader; no self-caption or same-frames arm | Skim |
| Beyond Visual Token Count (`s9lUh0Sl2N`) | ICLR 2027 sub. | Related (R6) | Target inclusion vs. reader recovery | Abs |
| Keep the Question (`vZfzyOh23P`) | ICLR 2027 sub. | Related (R6) | Task scope loss in decomposed document QA | Abs |
| Stop Describing, Start Answering (`UyUWSPyoTO`) | ICLR 2027 sub. | Related (R3, C5) | Description reliance in VLMs | Abs |
| [Position: Reasoning After Perception…](https://arxiv.org/abs/2507.16863) | 2025-07 (v2 2026-06) | Related (R3, R6) | Hard-to-verbalize tasks are not fixed by text reasoning | Searched |
| [EARS](https://arxiv.org/abs/2610.35536) | 2026-10-26 | Related (R7) | Motion/audio statistics cannot route per video between frame selectors (not text vs. frames) | Skim |
| [MMR-Bench](https://arxiv.org/abs/2601.17814) | 2026-01-25 | Related (R7) | Multimodal routing benchmark | Searched |
| Disagreement-Guided Weak-to-Strong Routing (`gvNBXwLEt1`) | ICLR 2027 sub. | Related (R7) | Routing without correctness labels | Abs |
| [Video-EMem / Write Once, Query Many](https://arxiv.org/abs/2610.24537) | 2026-10-17 | Related (R5) | Reusable text + sparse-image memory for long video; writer/reader split | Searched |
| IR³ (`n3SeGUrAX6`); QShare (`N6TRQNR42K`) | ICLR 2027 sub. | Related (R5) | Query-independent visual index; shared evidence across questions | Abs |
| VideoMem-Bench (`hnWMRIpNYZ`) | ICLR 2027 sub. | Related (R6) | Query-agnostic memory benchmark: compress first, then answer | Abs |
| [Retrieval-Augmented Online Textual Memory](https://arxiv.org/abs/2610.31284) | 2026-10-22 | Related (R5) | Egocentric caption memory; more retrieved captions can lower accuracy | Searched |
| [Similar Scores, Opposite Failures](https://arxiv.org/abs/2610.38511) | 2026-10-28 | Related (R6, audit) | Six frontier captioners, 196 short clips; windowed review finds 4.7× more camera and shot errors; systems as served, not identical inputs | Searched |
| [CodeCap-Bench](https://arxiv.org/abs/2610.37341) | 2026-10-27 | Related (R3) | Code as a video caption beats natural language on QA for structured 2D content | Searched |
| Video Reversal Curse (`uA4RIBZqBX`); Tracing Arrow-of-Time (`UeqjOP0eqf`) | ICLR 2027 sub. | Related (R6) | Reversal blindness from pixels | Abs |
| Most of the Gain Is Not Video (`QybPLnt1Ew`); CleanSlide (`o3hO7x2lXq`); Benchmark Déjà Vu (`DEo5cLf1K3`) | ICLR 2027 sub. | Related (R8) | Question-only baselines; blind-solvability audits; MC contamination detectors | Abs |

**Also seen, not above related:** Grounding on Request (2610.33841; modality control under image–caption conflict), InfiniteWorld (2610.33777; matched text-only and frame settings for physics), PACE-Mem (2610.24378), VideoMemBench (2610.40131; Claude Opus 5 best at 45.7%), Designing Textual Interfaces for AVQA (2610.20085; captioner-reasoner vs. omni model under prompt optimization), ZoomCap (2610.21320), Supervising the Visible State (2610.34926), Training-Free Inference Acceleration survey (2610.36327), V-Mem (2608.01543), Personal Visual Memory (2605.28806), Windsock (2510.22694), An Empirical Study of Video Representations in VideoQA (2510.12299; subtitles as a cheap alternative on long video), Frames Don't Add Up (`6FFJF8bQhV`), Learning When to Look / Route2K (`LoSVmTO1ZZ`; TempCompass), Video Understanding at a Glance (`GqKNRY7qN5`), Same Frames, Explicit Referents (`RBHEHW8fX0`), SMORE minimal-pair videos (`1XQKFnGKk4`).

## 5. Changes for the paper

**Applied 2026-11-02** (with the cut to 8 pages; log N3, *Applied*). Main text: items 1–2 (REEL in "Caption, then reason"; *Compression* in §4.2 and in a new "Readers on a fixed text" paragraph of §6), item 3 (*Revisiting*, Visual7W only), and item 4 (*Read First* in §4.6 and in future work, now in Limitations). Appendix "Further related work": MemoryTax, *Semantic Serialization*, *Beyond Visual Token Count* (item 5), the *Read First* gate, EARS, and the cost papers. Item 6 (*Most of the Gain Is Not Video*) was not added. Bib keys: `reel`, `compressneutral`, `revisitcaption`, `readfirst`, `semserial`, `memorytax`, `ears`, `inclusion`.

The main text was about 3¼ pages over the limit, so only items 1–2 were planned for the main text. The rest can go in an appendix paragraph, "Further related work", following the include-by-default rule.

1. **§6 "Caption, then reason":** "CapMem is the closest work" becomes "CapMem and REEL are the closest work". Add one sentence on REEL. It gives a self-written text record, cached visual tokens and a cached KV prefix exactly the frames the frame path sees, on long video with six open models. Text flips 20–29% of answers at nearly equal mean accuracy. It has no captioner swap and no cheaper reader. Say that our information control extends theirs to a captioner other than the reader.
2. **§4.1 / §6, the readers' tie:** cite *Compression Is Not Evaluation-Neutral*: a fixed compressed text shrinks reader differences in text RAG (31.8 → 7.8 points on HotpotQA; 4.5–84% retained across 13 panels), and weaker readers gain more. Our tie on Sonnet's captions is the same effect for a video caption index. Optionally add MemoryTax. Its readers tie on a weak writer's store and spread out on a strong writer's, so a tie is not automatic.
3. **§6 "Text or pixels":** add *Revisiting Caption Gains*, citing its image-removal replay and not its headline. On Visual7W, seven open VLMs, each on its own question-blind caption, all lose accuracy when the image is removed (11.3 points on average). Do not extend this to SLAKE, where one model gains slightly. *Semantic Serialization* is optional here, and only as engineered, task-aware text (the advantage grows as the reader shrinks and reverses at 0.5B). It is not caption evidence.
4. **§4.6 / §6, the decision rule:** cite *Read First, Look on Doubt* for routing text vs. frames per question on long video. Its gate looks at frames only when two text answers disagree, and the gated look is worth +8.1 on the questions it picks and 0 on the rest. One sentence can contrast it with our rule, which uses only what is known before reading. Answer-time instability is an untested signal for us, which fits Limitations or future work. Do not cite its "+14.6 → −13.7": that figure is subtitles vs. frames, not captions. EARS is optional, for the null of cheap video statistics in a different choice.
5. **§4.4, the audit (optional, author's call):** the "target inclusion vs. reader recovery" split names our finding that captions leave the question undecided.
6. **Limitations, contamination (optional):** *Most of the Gain Is Not Video* supports the question-only arm.

**Bib entries to add** (verify each against its page when adding):

| Key (suggested) | Source |
|---|---|
| `reel` | Liang & He, "REEL: Same Frames, Different Answers and Different Costs", WACV 2027; arXiv 2610.38158 |
| `compressneutral` | Panthi & Abdelfattah, "Compression Is Not Evaluation-Neutral: Fixed RAG Compression Can Distort Reader Comparisons", arXiv 2606.21807 |
| `revisitcaption` | Anonymous, "Revisiting Caption Gains When the Image Remains Available", ICLR 2027 submission, OpenReview `imjcY1Z6JP` |
| `readfirst` | Anonymous, "Read First, Look on Doubt: A Text-First Framework for Long-Video Question Answering", ICLR 2027 submission, OpenReview `6FPJ94irae` |
| `semserial` | Xu & Sankar, "Free the Language Model From the Vision Encoder: Semantic Serialization as a Perception Interface for Small Language Models", arXiv 2609.29601 |
| `memorytax` | Peng, "MemoryTax: The Bitter Lesson of Agent Memory", arXiv 2610.29823 |
| `ears` | Lin, "EARS: Evidence-Aware Retrieval via Sound and Motion for Budgeted Video Reasoning", arXiv 2610.35536 |
| `inclusion` | Anonymous, "Beyond Visual Token Count: Target Inclusion and Reader Recovery", ICLR 2027 submission, OpenReview `s9lUh0Sl2N` |

If an ICLR submission gets an arXiv version or is accepted by January, cite that version instead.

## 6. Search log

- **Tools.** Scripts are in `scripts/lit_search/` (`arxiv_query.py`, `arxiv_abstracts.py`, `s2_citations.py`, `openreview_search.py`, `fetch_fulltext.py`); `run_pass.sh` re-runs every query and chase below.
  - The arXiv API, newest first, at one request per 3 s.
  - Semantic Scholar citations, rate-limited without a key.
  - The OpenReview search API over `ICLR.cc/2027/Conference`, whose submissions are now public; the previous pass could not search them.
  - WebSearch: 2 extended searches and 2 standard ones (one of them the ARR deadline).
  - arXiv HTML full text for REEL and *Compression Is Not Evaluation-Neutral*.
  - Later the same day: full text of the two close ICLR 2027 submissions, from PDFs the author downloaded in a browser (OpenReview shows scripts a browser check), and arXiv full text of the 10 other arXiv papers in §4 that had been read as abstracts only. REEL was already read in full. The result is in the full-text update at the top: no level rises; *Semantic Serialization* moves to related; the descriptions of *Revisiting Caption Gains* and *Read First* are corrected.
- **arXiv queries, since 2026-10-18** (overlapping the fourth pass, which covered up to 2026-10-21):
  - caption + video + frames; caption + memory + video; captions + "visual tokens"; caption + cost + video; "text-only" + video + question;
  - "own captions" / "self-generated captions" / "self-captioning"; describe-then-answer / verbalize + video; captioner + reader; decoupled perception;
  - broad listings: video + caption(s); video QA + text; efficient video understanding; video + routing + modality.
- **arXiv queries, since 2025-01:**
  - question-aware / query-aware / question-agnostic / task-agnostic captions; generic captions + question; caption + reversed;
  - caption + speed + QA; when to look; text-or-image routing; route + text-only + VLM; visual need; modality selection;
  - TempCompass + caption / text-only; contamination + video benchmark; question-only + video; weak-to-strong + caption; "same frames" + caption;
  - exact-title lookups of the ICLR submissions.
  - About 200 unique arXiv entries were screened by title.
- **OpenReview (ICLR 2027), 38 queries**, about 670 title hits in all, screened by title:
  - captions vs. frames; self-captions; own captions; captioner–reader decomposition; describe then answer; text-only reader; caption utility;
  - question-blind captions; query-agnostic memory; question-aware captions; text vs. visual tokens; modality gap; verbalization loss;
  - amortized cost per query; break-even captioning; cheaper-model routing; per-query modality routing; TempCompass; arrow of time; contamination; blind baselines;
  - caption once; caption memory on short video; weak reader / strong writer; reader upgrade; information control; data processing inequality; pre-registration; question-only baseline;
  - minimal pairs; speed comparison; Claude Haiku / Sonnet; self-description; caption-then-answer cost; text index; when captions suffice; caption omission.
- **Citation chases (Semantic Scholar), 82 citing papers in all:**
  - since 2026-10-01: CapMem 1 (REEL), CFD 0, Caption This 0, REEL 0, Sprout 0, CapQuiz 2 (REEL, CodeCap-Bench);
  - since 2026-10-10: TempCompass 43, IsoBench 2, Prism 3, CaptionQA 3, LLoVi 25 (the first LLoVi chase), GEASS 0;
  - Caption This since 2025-06: 3, including the position paper above.
- **Stop rule.** The last 20 titles checked (the final OpenReview queries: minimal pairs, speed, Claude, self-description, caption-then-answer cost, text index, when captions suffice, omission) gave nothing above related.
- **Not done:**
  - full text of the ICLR 2027 submissions other than the two close ones (OpenReview serves PDFs only after a browser check; download them by hand if one becomes a citation in the main text);
  - Google Scholar; CVPR 2027 (not public);
  - arXiv after about 2026-10-30, since the API's newest entries were from Oct 30;
  - a citation chase of *Compression Is Not Evaluation-Neutral*;
  - a check for new post-cutoff short-video benchmarks (none was seen in passing).

## 7. Repeat pass before submission

In the week before the deadline (about 2026-12-28 to 2027-01-01), re-run §6 with `scripts/lit_search/run_pass.sh 2026-10-28 <outdir>` (no model calls; about 15 minutes at the APIs' rate limits), then screen the titles by hand:
- the arXiv queries;
- the OpenReview queries, since ICLR 2027 reviews will be public by then: check whether REEL, *Revisiting Caption Gains* or *Read First* changed, and whether any submission has an arXiv version;
- citation chases of REEL, CapMem, *Compression Is Not Evaluation-Neutral*, Caption This and TempCompass;
- post-cutoff short-video benchmarks.

Append the result as §8 here and as log entry N4, and update the bib.
