# Caption once, query cheaply: related work and feasibility (second pass)

Written 2026-10-14, after the Gate 1 pilot (`unshuffle_pilot_results.md`). The user's question: captions are a one-time index, and later video tasks should be solved from them instead of paying for video again. Has anyone compared captions with video at **equal query cost**, or scored with query cost in mind? What do the closest papers leave as future work? Can we do it? The first pass (`related_work.md`) covered ordering and modality; this pass covers captions as a reusable index and token-matched comparisons. Every paper below was checked at its arXiv abstract or full text, except where marked.

## 1. Short answer

- **Captioning once and reusing it across queries is published.** CFD and CapMem (September 2026, overlapping authors) both write the amortized cost formula, both on long video (tens of minutes to hours). SiLVR, LLoVi and Vamos had already shown that text-only readers over captions are competitive.
- **No one we found compares accuracy at matched query tokens, as a curve, per task type.**
  - ViSIL is the closest: 4 summary formats with their token counts, but 57 videos and aggregate accuracy only.
  - CapMem matches frames, not tokens, and tests a single frame cap.
  - CFD counts frames and seconds, not tokens.
- **No one separates coverage from language.** In CFD and CapMem, the captioner sees every frame at 1 fps, in short windows. The frame arm sees a subsample in one long context.
  - So "captions win on long video" mixes three effects: the captioner saw more, the captioner saw it in short windows, and the content was coded as words.
  - CFD's headline "visual–textual duality" (language carries long-range temporal structure better than frames) is measured this way. On InfiniBench it compares 1-fps captions with a frame arm capped at 768 frames, on videos of about 53 minutes (≈0.24 fps; our arithmetic).
  - Our pilot controls the information. There, frames ordered 1-minute clips far better than captions, though at 11× the query tokens.
- **Verdict: there is an open, testable contribution, but it is narrow, and the area is moving fast.** CFD, CapMem, CapQuiz and VEI all appeared in the last five weeks.
  - The parts that are hard to copy are the information control (coverage vs. language) and a task map with a theory behind it.
  - Stage 1 can run on what we have. A task map needs a second dataset.

## 2. Closest work

| Paper | Setting | Cost accounting | Matched query tokens? | Information control? | Main numbers | What it leaves open |
|---|---|---|---|---|---|---|
| [CFD: Caption-once, Frames-on-Demand](https://arxiv.org/abs/2609.11899) (Cai et al., 2026, arXiv) | Hour-long video (Video-MME, InfiniBench, LVBench). Qwen3-VL-8B captions once (events, plus 30 s clips at 1 fps); Qwen3-VL-32B reasons over the captions. A router fetches ≤ 8 frames only for "perceptual" questions. | Time: 222 s per video offline, 68 s per query online; T(Q) = T_offline/Q + T_online. Storage 0.035 MB per video. No token counts. | No. Frames per query (≈5.8) and seconds. | No. The captions come from all frames at 1 fps; the video arm gets ≤ 768 frames. | InfiniBench, captions vs. 768-frame video: chronological 56.5 vs 48.4; scene transitions 58.7 vs 54.0; global appearance 53.5 vs 70.5; character actions 53.1 vs 67.0. Video-MME short (< 2 min): no frames 68.8, router 72.2, always frames 72.3. | The router is binary; the authors want a calibrated score. Shot segmentation does not suit continuous footage. The captioning pass is "the bottleneck". No head-to-head with DVD or EGAgent. |
| [CapMem](https://arxiv.org/abs/2609.17688) (Liang et al., 2026, arXiv) | 75 egocentric videos (33.7 h; 56 from Ego4D), 1,000 MCQs, 12 models. Each model captions its own video in 30 s or 60 s windows at 1 fps, independent of the question (≤ 220 output tokens per window). | Caption tokens logged; video tokens by proxy (frames × 1,024). C(N) = (T_caption + Σ T_QA)/N. Gemini 3 Flash, 100 questions: 11.8 M input tokens with captions vs. 76.3 M direct. | No. Matched frames (the same ≤ 768 frames feed both arms), one frame cap. | Partly. The frames are matched, but windowed captioning vs. one long context still differs. | Captions win on videos over 20 min (frame-aligned +3.2 / +2.6 points for 30 s / 60 s windows). They lose at 20 min or less (−2.1 / −5.3; the 60 s interval excludes 0). Gemini 3 Flash does better on direct video. | Only one frame cap, so the 20-minute crossover is "an empirical regime, not a universal threshold". Controls run on Qwen only. Per-tag results are only a figure. The cost analysis leaves out retrieval. The authors ask for larger scale and more model families. |
| [ViSIL](https://arxiv.org/abs/2601.09851) (Li et al., 2026, arXiv) | A metric for how much video information a summary loses: log P(C\|V) / P(C\|summary), with a detailed caption C, scored by Gemini 2.0 Flash. Formats: text, 1 image + text, 3 images + text, full video. | Tokens: ≈78 / 337 / 873 / 54,588 on LongVideoBench; human response time. | **Yes, as a Pareto frontier:** at the cheapest end, its selected summaries get 63% QA accuracy vs. 56% for text at the same efficiency. | No. | 57 videos (20 from MVBench episodic reasoning, about 40 s; 37 from LongVideoBench sequence of scenes, about 208 s). Aggregate only. | No breakdown by task type. The authors want to use the score to generate summaries, and to add audio. **Closest to our core figure, but small and aggregate.** |
| [CaptionQA](https://arxiv.org/abs/2511.21025) (Yang et al., CVPR 2026) | Images: 4 domains, 33k MCQs that need the image; a text-only LLM answers from the caption. | None. | No. | n/a | Strong proprietary models lose 9–16% going from image to caption; open models lose 11–32%; Embodied AI loses over 40%. | Images only, no token axis. A per-category map of caption utility exists for images, not video. |
| [CapQuiz](https://arxiv.org/abs/2609.09973) (Wang et al., ACL 2026) | Scores video captions by how well they answer MCQs: 10 question types, 24 domains. | None. | No. | n/a | Correlates with human judgments better than reference metrics. | It evaluates captioners, not captions vs. video. Its question types could serve as our taxonomy. |
| [Aligned video captions for RAG](https://arxiv.org/abs/2405.17706) (Dela Rosa, 2024, arXiv) | Captions as the video stand-in for chat RAG. | An estimate: 1-fps frames at 256 tokens each would need ≈ 69× the tokens of the captions. | No accuracy at matched tokens. | No. | (estimate only) | The paper is cited for the token argument. |
| [SEG](https://arxiv.org/abs/2601.06097) (Dixit & Liang, 2026, arXiv) | A symbolic event graph (YOLO tracks to START/END events), pruned per query and read by Gemini 2.5 Flash. | Tokens per query: 3.47k vs. 40.39k for the full log. | Text vs. text only. | No. | 65.0% vs. 62.5% for the full log, on 5 videos and 120 generated questions. | Small; no frame arm. |
| [VEI](https://arxiv.org/abs/2610.00757) (Guan et al., 2026, arXiv) | A visual "index": a low-resolution preview at 1, 6, 12 or 24 tokens per frame, then a few high-resolution frames per question. | Visual token budgets. | Visual vs. visual. | No. | Better accuracy under limited visual budgets. | **A visual index can be very small per frame**, so a per-token comparison must include tiny frames, not just full ones. |
| [Survey of video LLM efficiency](https://arxiv.org/abs/2609.10355) (Steunou et al., 2026, arXiv) | Covers efficiency mechanisms for video LLMs. | n/a | n/a | n/a | n/a | It **leaves out** caption ("analyzer") pipelines because they "would require accounting for the upstream expert models". It also says budgets "are not standardized across papers". This states our accounting gap in print. |

**Older systems precedent (databases).** The split between a cheap ingest-time index and expensive query-time models is old:
- [Focus](https://arxiv.org/abs/1801.03493) (Hsieh et al., OSDI 2018): cheap CNNs index objects at ingest; expensive CNNs run at query time.
- [TASTI](https://arxiv.org/abs/2009.04540) (Kang et al., SIGMOD 2022): one *task-agnostic* index serves many queries without a proxy model per query. It is ≈ 10× cheaper to build, with queries up to 24× faster (numbers from the paper as summarized by search; not re-checked in the full text).

Their index is labels or embeddings, not language. Cite them for the cost model and for the idea of a task-agnostic index.

**Per-token modality, background:**
- [Text or Pixels? It Takes Half](https://arxiv.org/abs/2510.18279) (2025): text *rendered as an image* needs about half the decoder tokens at equal accuracy. For text content, pixels can be the cheaper code.
- [TokenSwap](https://arxiv.org/abs/2607.28640) (2026): the same content as interleaved images instead of text loses 4–47% accuracy across 42 MLLMs. Reasoning models lose less (10% vs. 26%). Readers process the two modalities unequally, so the reader is part of any per-token result.
- [SpatialEval](https://arxiv.org/abs/2406.14852) (NeurIPS 2024): text-only descriptions often beat images for spatial reasoning in VLMs.
- [Parametric multimodal user memory](https://arxiv.org/abs/2608.28609) (2026): caption-based memory recovers as little as 0.11 of a dedicated encoder's re-identification recall, "collapsing toward chance on non-nameable signals". **Nameability** predicts what a caption index loses.

## 3. Has anyone compared at equal query cost?

| | Amortizes captioning | Counts query tokens | Same reader on both sides | Matched budget | Budget sweep | Coverage vs. language separated | By task type | Short videos |
|---|---|---|---|---|---|---|---|---|
| CFD | yes | no (frames, seconds) | yes | no | frames only (1.8 / 5.8 / 8.9 per query, captions always present) | no | 4 InfiniBench skills | Video-MME short split |
| CapMem | yes | captions yes; video by proxy | yes (self-captioning) | frames | no (one cap) | partly | tags, in a figure | the ≤ 20 min split |
| ViSIL | no | yes | yes | Pareto frontier | 4 formats | no | no | 40 s and 208 s subsets |
| Aligned-caption RAG | no | estimate | n/a | no | no | no | no | n/a |
| **Proposed here** | yes | yes | yes | yes | yes | **yes** | **yes** | **yes (1 min)** |

So: query cost is now taken into account, both as amortized time (CFD, CapMem) and as a token frontier (ViSIL). No one has built the token-matched curve with an information control and a task breakdown.

## 4. Their open problems we can pick up

1. **CapMem: a single frame cap.** A budget sweep shows where captions stop paying, as a function of query tokens rather than of video length.
2. **CapMem: controls on one model family.** Our reader is Claude, a different family. Self-captioning is not our design: Gemini captions and Claude reads.
3. **CFD: a binary router; the authors want a calibrated score.** The calibrated quantity is caption retention, meaning caption accuracy over frame accuracy at matched tokens, measured per task (and per video). We can measure it; we would not build the router.
4. **CFD: the captioning pass is the bottleneck.** Caption length and granularity are a rate axis: terse vs. full per-piece captions vs. dense per-second captions.
5. **The survey: no standard accuracy–cost accounting for caption pipelines.** We would report query tokens and bytes, the one-time captioning cost, and the break-even number of queries Q* = C_caption / (c_frames − c_captions) at matched accuracy.
6. **ViSIL: small and aggregate.** Per-task curves on more videos.

## 5. What is open (our candidate contribution)

Working title: **"Caption once, query cheaply: what a task-agnostic text index of a video keeps and loses, per token."**

1. **Accuracy–query-token curves** for each representation and task, with the same reader.
2. **Coverage vs. language.**
   - Captions made from exactly the frames the frame arm sees isolate language. This is our isolated design.
   - Captions made from more frames add coverage: per-second captions from 60 frames, against 1 or 5 query frames per piece.
   - The difference between the two splits a caption advantage into its parts.
3. **A test of the published duality claim** (CFD) under the control and at matched tokens.
4. **A task map with a predictor: nameability.** The hypothesis is that per-task caption retention rises with how nameable the answer-relevant content is: actions and objects survive; motion direction, fine appearance and layout do not. This is the "language vs. visual perception" insight. Background from cognitive science, cited from memory and to be checked before citing:
   - codability (Brown & Lenneberg, 1954);
   - verbal overshadowing (Schooler & Engstler-Schooler, 1990);
   - dual coding (Paivio, 1971).
5. **Short videos.** CapMem finds captions lose at 20 minutes or less with full frames. Per token, no one knows.

**Theory sketch.**
- *Cost model and break-even:* as above.
- *Bound:* with equal coverage, captions cannot carry more Shannon information about the task than the frames they came from (data processing). Any caption win is therefore *usable* information ([Xu et al. 2020](https://arxiv.org/abs/2002.10689)): the language effect. Wins beyond that come from coverage.
- *Rate:* captions and frames are two lossy codes at a rate R (tokens or bytes), and each task has its own utility curve.

**On the user's concern that our original captions are not a realistic index:** task-agnostic, query-independent captions are exactly what CapMem (30 s and 60 s windows) and CFD (events plus 30 s clips) use. Our dense per-second originals are the same kind of index at a finer grain, so they are a fair arm. The open question is grain and length per token, which we can vary.

## 6. Can we do it?

**Already in hand:**
- 30 pilot videos, each with three caption sets. Median words per 5 s piece, measured today:
  - isolated: ≈ 32;
  - whole-video in-context: ≈ 24;
  - original per-second, task-agnostic: ≈ 52.
- 335 videos with the original captions.
- 12-piece puzzles with metrics and a cluster bootstrap.
- Frames at 448 px.
- The Claude listwise reader and the analysis code.

**Stage 1, ordering budget curve (no new tasks):**
- Text arms:
  - the existing sets: isolated, in-context, original;
  - a terse set, ≤ 10 words per piece;
  - captions from a single frame per piece.

  The last two are new Gemini calls, under $1 for 30 videos.
- Frame arms, at Claude's ≈ w×h/750 tokens per image:
  - 1 small frame per piece, ≈ 300×170 px, ≈ 70 tokens, matched to one caption;
  - 1 frame per piece at 448 px, ≈ 150 tokens;
  - 5 frames per piece, ≈ 766 tokens (already run).
- Reader: Sonnet, with Opus on a subset. Puzzles: P12 and G6, 2 shuffles each.
- Size: the 10 calibration videos need ≈ 200 new reader calls, most of them small. All 30 videos need ≈ 600.
- What it answers:
  - Does a 70-token frame beat a 70-token caption at ordering?
  - Does coverage (60 frames vs. 1) explain the gains of the dense captions?
  - Where does each arm sit on the accuracy–token plane?

**Stage 2, a task map (needs a decision after Stage 1):**
- Use a short-video QA benchmark with a question-type taxonomy, and caption it once, task-agnostically.
- Candidates:
  - [MVBench](https://arxiv.org/abs/2311.17005): 20 temporal tasks that a single frame cannot solve.
  - TVBench: text-only models are at chance; see `related_work.md`.
  - CapQuiz's 10 question types.
- WildQA is not suitable: its questions are video-level, so they have no temporal headroom (`docs/TODO.md`, E1).
- Several thousand reader calls, to be planned separately.

**Risks:**
- *Scooping.* Four close papers in five weeks; someone could publish the matched-token curve. The information control and the task map are the hard parts to copy.
- *Scope.* Ordering alone is too narrow for a top venue; reviewers will want QA. Stage 2 is not optional for a strong paper.
- *Token units.* Image and text tokens differ by model (Gemini ≈ 66–258 per frame, Claude by pixel area). Report tokens per reader and bytes, and show that the conclusions hold under both.
- *Reader asymmetry.* TokenSwap shows readers handle text and images unequally. That is part of the practical answer, but the effort setting must match across arms; at medium effort, Opus thought ≈ 400 tokens on frame puzzles and 0 on caption puzzles.

## 7. Search log (2026-10-14, second pass)

- **Tools:** WebSearch (standard and extended), the arXiv API through the scratchpad client, and the Semantic Scholar citations API.
- **Full text fetched:** CFD, CapMem (both halves), ViSIL, the aligned-caption RAG paper, and the survey (scope section).
- **Queries:**
  - caption vs. end-to-end at the same token budget;
  - caption once, query many, amortized;
  - text vs. visual tokens at equal budget;
  - task-agnostic index (TASTI, Focus, Boggart);
  - caption utility benchmarks (CaptionQA, CapQuiz, CapProbe);
  - SiLVR;
  - the token efficiency of text vs. pixels;
  - rate-accuracy and coding for machines (no caption-as-code work found);
  - caption length vs. QA;
  - question-aware vs. generic captions (Q-ViD and others; they help, but cannot be amortized);
  - LLM-era video analytics systems (LazyVLM);
  - Pareto frontiers of video QA cost;
  - visual necessity.
- **Citation chase:**
  - ViSIL: 2 citing papers (VEGAS, CodecCap), neither a threat.
  - CFD: 1 relevant citing paper (VEI).
  - CaptionQA: 15 citing papers, mostly captioning RL and evaluation; CapMem, CFD and CapQuiz are among them.
  - CapMem: no citations yet.
- **Not done:**
  - full text of CapQuiz, VEI and SiLVR;
  - Boggart, which search did not find;
  - the cognitive-science citations in §5;
  - the Gemini "agentic video" Pareto claims, which are a vendor blog.
