# Caption once, query cheaply: literature review (third pass)

Second pass written 2026-10-14, after the Gate 1 pilot (`unshuffle_pilot_results.md`). Expanded on 2026-10-15 into a full review: the state of the art in each thread the idea touches, the gaps, and the niche. The research plan built on it is `research_plan_caption_index.md`.

**The question.** A video is captioned once, independently of any question, and the captions are kept as an index. Later questions are answered from the index instead of from the frames. When does this pay, in accuracy per unit of query cost, and in total cost once the captioning is counted? What does the language channel keep and lose?

**How this was checked.** Every paper listed was opened at its arXiv abstract, its full text, or its venue page. Marks: *(full)* full text read; *(abs)* abstract only; *(search)* a claim taken from a search summary and not re-checked, used only where marked. The first pass (`related_work.md`) covers ordering and modality and is not repeated here.

## 1. Bottom line

**What is known (state of the art, October 2026):**
- **Language pipelines are competitive readers of video.** Caption-then-reason systems (LLoVi, VideoTree, SiLVR, Deep Video Discovery) hold or held the best reported results on long-video QA. They are mostly evaluated on long video, and almost never with a cost axis.
- **"Caption once, reuse" is published, for long video.** CFD and CapMem (September 2026) write the amortized cost formula; Sprout (late September 2026) builds the memory lazily because eager indexing costs more than answering a few questions. CapMem finds captions lose to direct video on videos of 20 minutes or less.
- **The visual side can be made very cheap.** 1–24 visual tokens per frame (VEI), and dense low-resolution frames beat sparse full-resolution frames at matched visual-token budgets (LoHi, NeurIPS 2026). Any fair "captions vs. frames per token" comparison has to give frames their best allocation, not one full-resolution frame.
- **Tokens are not the cost.** Multimodal inference costs 17–94% more energy than text-only at the *same* number of input tokens (Modality Inflation); image and video inputs cost 1.1–15× the energy per token of text (ML.ENERGY v3). Vision encoders and CPU-side decoding become the bottleneck once tokens are cut (LiteFrame, LoHi).
- **Caption quality is now judged by downstream QA** (CaptionQA, CapQuiz, CapProbe, CAPEval, CodecCap's VidCapQA). The only token-vs-accuracy frontier for video summaries is ViSIL: 57 videos, aggregate accuracy.

**What is open (as far as this search found):**
1. Accuracy against query cost as a **curve, per task type, with the same reader**, on short video, with both modalities at their best allocation per budget.
2. **Source vs. modality.** Published caption wins mix "the captioner saw more frames" with "words are easier to use than pixels". No one controls what the captioner saw.
3. **A cost account that goes beyond tokens:** bytes, latency, energy, the vision encoder, the reader's own thinking tokens, and the one-time captioning cost, with a break-even number of queries. The 2026 efficiency survey leaves caption pipelines out for exactly this reason.
4. **Why some content survives the language channel and some does not.** Cognitive science has a name and a theory for it (codability; efficient naming as an information bottleneck); no video paper tests it.

**The niche in one sentence.** A resource-complete, information-controlled rate–utility study of language as a reusable video index: per task type, how much accuracy a caption index buys per token, byte, joule and dollar at query time; how much of that is the source rather than language; what predicts it (codability); and when indexing pays for itself.

**Verdict.** Open, publishable, and narrow enough to finish on our budget. The area is moving weekly (CFD, CapMem, CapQuiz, VEI, Sprout, LoHi, OneStreamer and SwitchMem all appeared between 9 September and 13 October 2026), so the parts that are hard to copy (the information control, the theory, the full cost account) must lead.

## 2. Vocabulary used below

| Term | Meaning here |
|---|---|
| Index | A representation computed once per video, before any question is known (task-agnostic). Captions are one; cached frames or embeddings are others. |
| Query cost | What the reader consumes per question: input tokens (text and image), output and thinking tokens, payload bytes, latency, energy, list-price dollars. |
| Matched budget | Two representations given the same query cost in a stated unit. Matching in tokens does not match bytes or energy. |
| Source | What the representation was made from (the video, or which frames at which fps and resolution). Called "coverage" in this review's first version; renamed 2026-10-20. |
| Modality | The type of input at query time (words vs. pixels), from the same source. Called "code" in this review's first version; renamed 2026-10-20. The full term set is the *Terms* block of `research_plan_caption_index.md`. |
| Break-even Q\* | The number of questions per video after which the one-time captioning cost is repaid by cheaper queries at equal accuracy. |
| Routing headroom | How much accuracy a router that picks a representation per question type could gain at a fixed budget over the best single representation. |

## 3. State of the art, thread by thread

### 3.1 Language as the interface to video (caption-then-reason)

**What the field knows.** Turning video into text and letting a strong LLM reason over it works, especially for long video. Systems differ in how the text is built (per-clip captions, trees, memories, retrieval) and are compared on accuracy, rarely on cost.

| Paper | What it shows |
|---|---|
| [Socratic Models](https://arxiv.org/abs/2204.00598) (Zeng et al., 2022) *(abs)* | Composing models through language, zero-shot. The origin of the pattern. |
| [LLoVi](https://arxiv.org/abs/2312.17235) (Zhang et al., EMNLP 2024) *(abs)* | Short-clip captions (0.5–8 s) plus an LLM; 52–55% on EgoSchema *(search)*. The choice of captioner and LLM matters most. |
| [VideoAgent](https://arxiv.org/abs/2403.10517) (Wang et al., 2024) *(abs)* | An LLM agent that looks at few frames iteratively. |
| [VideoTree](https://arxiv.org/abs/2405.19209) (Wang et al., CVPR 2025) *(abs)* | Adaptive tree of captions; 61.1% EgoSchema, 75.6% NExT-QA *(search)*. |
| [LangRepo](https://arxiv.org/abs/2403.14622) (Kahatapitiya et al., 2024) *(abs)* | A pruned textual repository; longer caption inputs lowered QA accuracy even when they fit the context *(search)*. |
| [Vamos](https://arxiv.org/abs/2311.13627) (Wang et al., 2023) *(full, first pass)* | Shuffling 12 captions costs 2 points on EgoSchema. |
| [Video-RAG](https://arxiv.org/abs/2411.13093) (Luo et al., NeurIPS 2025) *(abs)* | Visually aligned auxiliary texts (OCR, ASR, detections) retrieved for long-video QA. |
| [VideoRAG](https://arxiv.org/abs/2502.01549) (Ren et al., 2025) *(abs)* | RAG over extremely long video. |
| [SiLVR](https://arxiv.org/abs/2505.24869) (Zhang et al., TMLR 2026) *(abs)* | Clip captions + speech into a reasoning LLM; best reported results on Video-MME (long) and others at publication. |
| [Deep Video Discovery](https://arxiv.org/abs/2505.18079) (Zhang et al., NeurIPS 2025) *(abs)* | Agentic search over a multi-granular video database; 74.2% on LVBench (76.0% with transcripts). |
| [VideoMultiAgents](https://arxiv.org/abs/2504.20091) (Kugo et al., 2025) *(abs)*, [Q-ViD](https://arxiv.org/abs/2402.10698) (Romero & Solorio, 2024) *(abs)* | Question-guided captions help; no matched-length comparison with generic captions was found. |

**Left open.** Cost is reported as wall time or frame counts at best. Short video is rarely studied. Question-aware captions help but cannot be amortized; their gain over task-agnostic captions at equal length is unmeasured (§3.10, "price of task-agnosticism").

### 3.2 Caption once, reuse: memories and indexes

**What the field knows.** Long-video systems now cache a text memory and reuse it across questions; the newest work asks *when* to build it.

| Paper | Setting | Cost accounting | Main numbers | What it leaves open |
|---|---|---|---|---|
| [CFD: Caption-once, Frames-on-Demand](https://arxiv.org/abs/2609.11899) (Cai et al., EMNLP 2026) *(full)* | Hour-long video (Video-MME, InfiniBench, LVBench). Qwen3-VL-8B captions once (events, plus 30 s clips at 1 fps); Qwen3-VL-32B reasons over the captions. A router fetches ≤ 8 frames only for "perceptual" questions. | Time: 222 s per video offline, 68 s per query online; T(Q) = T_offline/Q + T_online. Storage 0.035 MB per video. No token counts. | InfiniBench, captions vs. 768-frame video: chronological 56.5 vs 48.4; scene transitions 58.7 vs 54.0; global appearance 53.5 vs 70.5; character actions 53.1 vs 67.0. Video-MME short (< 2 min): no frames 68.8, router 72.2, always frames 72.3. | The router is binary; the authors want a calibrated score. The captioning pass is "the bottleneck". Its headline "visual–textual duality" compares 1-fps captions with ≤ 768 frames on ≈ 53-minute videos (≈ 0.24 fps; our arithmetic), so coverage and code are confounded. |
| [CapMem](https://arxiv.org/abs/2609.17688) (Liang et al., EMNLP 2026) *(full)* | 75 egocentric videos (33.7 h), 1,000 MCQs, 12 models; each model captions its own video in 30 s or 60 s windows at 1 fps. | Caption tokens logged; video tokens by proxy (frames × 1,024). C(N) = (T_caption + Σ T_QA)/N. Gemini 3 Flash, 100 questions: 11.8 M input tokens with captions vs. 76.3 M direct. | Captions win on videos over 20 min (+3.2 / +2.6 points, frame-matched, 30 s / 60 s windows) and lose at 20 min or less (−2.1 / −5.3). | One frame cap, so the 20-minute crossover is "an empirical regime, not a universal threshold". Controls only on Qwen. Per-tag results only in a figure. |
| [Sprout](https://arxiv.org/abs/2609.35497) (Chen et al., 2026) *(abs)* | Builds a temporal tree of text memory *while* answering; video inputs are replaced by text once recorded. | Context per question. | Competitive with offline memories, no upfront stage, lower context per question. | States the amortization problem in print: "with few questions, building memory for the whole video costs far more than answering them". No break-even analysis. |
| [iRAG](https://arxiv.org/abs/2404.12309) (Arefeen et al., CIKM 2024) *(abs)* | Incremental (lazy) video-to-text conversion for RAG. | Ingestion time. | 23–25× faster ingestion than converting everything upfront, comparable answer quality. | The eager vs. lazy trade-off, without accuracy-per-cost curves. |
| [OneStreamer](https://arxiv.org/abs/2610.01762) (Zeng et al., 2026) *(abs)* | Streaming video LLM that writes query-independent caption memory while watching. | n/a | Retaining generated captions improves historical QA without hurting real-time perception. | Trains the model; no cost comparison with visual memory. |
| [SwitchMem](https://arxiv.org/abs/2610.17928) (Roh et al., 2026) *(abs)* | GUI agents: route each step's memory to text or image. | n/a | The best memory modality is step-dependent; routing beats either. | Not video; supports "the modality choice is per task, not global". |
| [ASP / Budget-Constrained Embodied Perception](https://arxiv.org/abs/2608.22975) (Lin et al., 2026) *(abs)* | Pre-registered evaluation of 7 open models under a fixed 4,096-token decision budget, on a synthetic benchmark. | Tokens per decision. | Query-conditioned access beats equal-budget query-independent sampling (75–94% vs 3–19%); a prompted compressive text state *hurt* (removing it raised the mean from 35.4 to 58.0). | Synthetic scenes only (its natural-video benchmarks were not run). A rare pre-registered, budget-matched design: a model for ours. |

**Left open.** All on long or synthetic video. No information control. No per-task matched-budget curve. Break-even is written as a formula (CFD, CapMem) but never solved with measured numbers against an equal-accuracy frame arm.

### 3.3 How cheap can the visual side get?

**What the field knows.** Visual tokens per frame can be cut by one to two orders of magnitude, and how a fixed visual budget is split between frames and resolution matters as much as its size.

| Paper | Finding |
|---|---|
| [Survey of inference efficiency in video LLMs](https://arxiv.org/abs/2609.10355) (Steunou et al., 2026) *(full, scope section)* | Covers sampling, encoders, token reduction, prefill/decoding. **Leaves out** caption ("analyzer") pipelines because they "would require accounting for the upstream expert models"; budgets "are not standardized across papers". |
| [LLaMA-VID](https://arxiv.org/abs/2311.17043) (Li et al., 2023) *(abs)* | "An image is worth 2 tokens": two tokens per frame for long video. |
| [Du et al., design space of visual context](https://arxiv.org/abs/2410.13694) (2024) *(abs)* | Frames × tokens-per-frame as a constrained optimization; fits scaling curves and derives the optimal split. |
| [Apollo](https://arxiv.org/abs/2412.10360) (Zohar et al., 2024) *(abs)* | fps sampling beats uniform sampling; design decisions transfer across scale ("Scaling Consistency"). |
| [VEI](https://arxiv.org/abs/2610.00757) (Guan et al., 2026) *(abs)* | A low-resolution preview at 1, 6, 12 or 24 tokens per frame, then a few high-resolution frames per question; better accuracy under limited visual budgets. |
| [LoHi](https://arxiv.org/abs/2610.04318) (Dong et al., NeurIPS 2026) *(abs)* | **Dense low-resolution frames beat sparse native-resolution frames at matched token budgets**; front-end video decoding dominates wall time on hour-long video; +10.6 points over the native-resolution baseline at matched budget. |
| [LiteFrame](https://arxiv.org/abs/2605.17260) (Kim et al., 2026) *(abs)* | After post-hoc token reduction the vision encoder becomes the latency bottleneck; a distilled encoder moves the latency–accuracy frontier. |

**Consequence for us.** A frame arm at a given budget must be the best of several allocations (one small frame, several tiny frames, one large frame), or a reviewer will say the frames were strawmanned. Token budgets alone hide the encoder and decoding costs.

### 3.4 Text vs. pixels as codes for the same content

**What the field knows.** The question "which modality is the cheaper code" has been asked the other way round, for text: rendering text as an image can *save* decoder tokens. The same model also treats equivalent text and image inputs unequally.

| Paper | Finding |
|---|---|
| [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) (Wei et al., 2025) *(abs)* | "Optical compression": ~97% OCR precision when text tokens are under 10× the vision tokens, ~60% at 20×. |
| [Glyph](https://arxiv.org/abs/2510.17800) (Cheng et al., 2025) *(abs)* | Rendering long text as images gives 3–4× token compression at comparable accuracy, ≈ 4× faster prefill and decoding. |
| [Text or Pixels? It Takes Half](https://arxiv.org/abs/2510.18279) (Li et al., EMNLP 2025 Findings) *(abs)* | Text as one image needs about half the decoder tokens at equal accuracy. |
| [TokenSwap](https://arxiv.org/abs/2607.28640) (Hua et al., 2026) *(abs)* | The same content as interleaved images instead of text loses 4–47% across 42 MLLMs; reasoning models lose less. The reader is part of any per-token result. |
| [SpatialEval](https://arxiv.org/abs/2406.14852) (Wang et al., NeurIPS 2024) *(abs)* | Text-only descriptions often beat images for spatial reasoning in VLMs. |
| [Missing-modality gap](https://arxiv.org/abs/2605.12517) (Kim et al., ICLR 2026 workshop) *(abs)* | A VLM given a description instead of the image loses accuracy *and* calibration, beyond what missing information explains. |
| [Parametric multimodal user memory](https://arxiv.org/abs/2608.28609) (2026) *(abs, second pass)* | Caption memory recovers as little as 0.11 of a dedicated encoder's re-identification recall, "collapsing toward chance on non-nameable signals". |

**Our angle.** DeepSeek-OCR asks how many vision tokens a page of text needs; we ask how many text tokens a minute of video needs, per task. Per token, the image channel has far more capacity: our own numbers below put a 40-token frame at ≈ 7.7 KB and a 32-word caption at 184 bytes. A caption that wins per token therefore wins by *selection* and *usability*, not capacity.

### 3.5 Measuring what captions keep (caption utility)

**What the field knows.** Reference metrics (BLEU, CIDEr) are being replaced by QA-based utility: a text-only reader answers questions from the caption.

| Paper | What it measures | Token axis? |
|---|---|---|
| [CaptionQA](https://arxiv.org/abs/2511.21025) (Yang et al., CVPR 2026) *(abs)* | Images, 4 domains, 33k MCQs; strong models lose 9–16% from image to caption, open models 11–32%, embodied AI > 40% *(second pass)*. | No |
| [CapQuiz](https://arxiv.org/abs/2609.09973) (Wang et al., ACL 2026) *(abs)* | Video captions scored by MCQs over 10 question types, 24 domains. | No |
| [CapProbe](https://arxiv.org/abs/2608.11074) (Huang et al., 2026) *(abs)* | Dense region-aligned QA for image captions; density metrics penalize verbose captions; a "competency–efficiency trade-off". | Partly (density) |
| [CAPEval](https://arxiv.org/abs/2608.02589) (Liu et al., 2026) *(abs)* | Coverage vs. precision; coverage predicts understanding, precision predicts generation. | No |
| [CodecCap / VidCapQA](https://arxiv.org/abs/2605.26967) (Lin et al., 2026) *(abs)* | Codec-style keyframe + residual captions; a caption-then-QA benchmark (1,000 questions, 14 capability dimensions). Strong VLM captions "still miss many visual details". | No |
| [ViSIL](https://arxiv.org/abs/2601.09851) (Li et al., 2026) *(full)* | Information loss of a summary, log P(C\|V)/P(C\|summary). Formats: text ≈ 78 tokens, 1 image + text ≈ 337, 3 images + text ≈ 873, full video ≈ 54,588 (LongVideoBench). | **Yes, a Pareto frontier** on 57 videos, aggregate only |
| [CapMem](https://arxiv.org/abs/2609.17688) *(full)* | See §3.2. | Caption tokens only |

**Left open.** A per-task utility map for video with a cost axis. CodecCap's "keyframe + residual" captions and our grain arms (terse, per-window, whole clip) are rate points on the same curve.

### 3.6 Benchmarks, shortcuts, and what "needs the video"

**What the field knows.** Many video QA items are answerable from one frame, from the question text, or from world knowledge, so any caption-vs-frames comparison needs a blind (question-only) arm and a temporally sensitive subset.

| Benchmark or study | Relevance |
|---|---|
| [TempCompass](https://arxiv.org/abs/2403.00476) (Liu et al., 2024) *(abs; counts from full text)* | 410 short videos (mostly ≤ 30 s), 7,540 instructions; aspects action, speed, direction, order, attribute change. "Conflicting videos" share static content and differ in one temporal aspect, so single-frame and language priors fail. CC BY-NC 4.0. **Our best fit for the low-codability end.** |
| [Perception Test](https://arxiv.org/abs/2305.13786) (Pătrăucean et al., NeurIPS 2023 D&B) *(abs)* | 11.6k real videos, 23 s on average; skill areas (memory, abstraction, physics, semantics) and reasoning types (descriptive, explanatory, predictive, counterfactual). CC-BY. The validation videos are a 70.2 GB archive (GitHub page). |
| [NExT-QA](https://arxiv.org/abs/2105.08276) (Xiao et al., CVPR 2021) *(abs)* | Causal, temporal and descriptive questions; the standard benchmark of caption pipelines (LLoVi, VideoTree), so our index can be compared with published numbers. |
| [MVBench](https://arxiv.org/abs/2311.17005) (Li et al., CVPR 2024) *(abs)* | 20 temporal tasks; much of it is answerable without video (see TVBench). |
| [Video-MME](https://arxiv.org/abs/2405.21075) (Fu et al., 2024) *(abs)* | Short / medium / long splits; CFD reports the short split. |
| [TVBench / Lost in Time](https://arxiv.org/abs/2410.07752) (Cores et al., 2024) *(abs)* | Existing benchmarks are solvable from single frames, question text or world knowledge; text-only models are at chance on TVBench. |
| [TempCore](https://arxiv.org/abs/2509.01167) (Ok & Lee, EMNLP 2026) *(abs)* | Only 5.5–31% of samples across six benchmarks are temporally sensitive (frame-selection sensitivity × language independence). |
| [MotionBench](https://arxiv.org/abs/2501.02955) (Hong et al., 2025), [FAVOR-Bench](https://arxiv.org/abs/2503.14935) (Tu et al., NeurIPS 2025 D&B) *(abs)* | Fine-grained motion; current models are weak, and higher frame rates help. Candidates for the "hard to name" end. |

### 3.7 Accounting for cost: tokens are not the cost

**What the field knows.** Cost-aware evaluation is an established demand in ML; for multimodal inference, energy per token depends heavily on the modality.

| Paper | Finding |
|---|---|
| [Green AI](https://arxiv.org/abs/1907.10597) (Schwartz et al., 2019); [Energy and Policy Considerations](https://arxiv.org/abs/1906.02243) (Strubell et al., ACL 2019) *(abs)* | Report efficiency alongside accuracy. |
| [Show Your Work](https://arxiv.org/abs/1909.03004) (Dodge et al., 2019) *(abs)* | Report performance as a function of the computation budget. |
| [Efficiency Pentathlon](https://arxiv.org/abs/2307.09701) (Peng et al., 2023) *(abs)* | A standardized arena for efficiency evaluation. |
| [AI Agents That Matter](https://arxiv.org/abs/2407.01502) (Kapoor et al., TMLR) *(abs)* | Evaluate on an accuracy–cost Pareto frontier in dollars; simple baselines dominate costly agents. Our evaluation follows this standard. |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) (Chen et al., 2023) *(abs)* | Cascades and routing across LLMs to cut cost. An index that lets a cheaper text-only reader answer is a route of this kind. |
| [Power Hungry Processing](https://arxiv.org/abs/2311.16863) (Luccioni et al., FAccT 2024) *(abs)* | Energy per 1,000 inferences across tasks; general-purpose generative models are orders of magnitude costlier than task-specific ones. |
| [How Hungry is AI?](https://arxiv.org/abs/2505.09598) (Jegham et al., 2025) *(abs)* | Energy, water and carbon of LLM inference. |
| [Where Do the Joules Go? / ML.ENERGY v3](https://arxiv.org/abs/2601.22076) (Chung et al., 2026) *(abs)*; [blog](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/) *(full)* | 25× energy differences by task type. Qwen 3 VL 8B vs. Qwen 3 8B: text + image 1.1–5.2× and text + video 1.3–15.0× the energy per token of text; CPU-side vision preprocessing limits batch size. |
| [Modality Inflation](https://arxiv.org/abs/2512.22695) (Moghadampanah et al., 2025) *(abs)* | Stage-level energy (vision encoding, prefill, decode) on A100: multimodal inference uses 17–94% more energy than text-only **for identical inputs**; "token count alone does not determine energy" *(search)*. |

**Left open.** No video QA paper reports accuracy against measured energy per query for a text index vs. frames, with the index cost amortized.

### 3.8 Systems precedent: indexes, views, lazy vs. eager

**What the field knows.** Databases have long split cheap ingest-time indexes from expensive query-time models, and chosen what to precompute from the expected query workload.

- [Focus](https://arxiv.org/abs/1801.03493) (Hsieh et al., OSDI 2018) *(abs)*: cheap CNNs index objects at ingest; expensive CNNs run at query time.
- [TASTI](https://arxiv.org/abs/2009.04540) (Kang et al., SIGMOD 2022) *(abs)*: one task-agnostic semantic index serves many queries without a per-query proxy model, and is cheaper to build than prior indexes.
- [Boggart](https://arxiv.org/abs/2106.15315) (Agarwal & Netravali) *(abs)*: model-agnostic acceleration of retrospective video analytics.
- [Zelda](https://arxiv.org/abs/2305.03785) (Romero et al., 2023) *(abs)*: video analytics with VLMs and natural-language queries. [LOVO](https://arxiv.org/abs/2507.14301) (Liu et al., 2025) *(abs)*: a one-time, query-independent embedding index for complex object queries.
- [Atom](https://arxiv.org/abs/2512.17108) (Panchal et al., 2025) *(abs)*: on-device video-language pipelines that reuse modules across captioning, reasoning and indexing.
- **Materialized views:** [Harinarayan, Rajaraman & Ullman, "Implementing data cubes efficiently"](https://dsf.berkeley.edu/cs286/papers/datacube-sigmod1996.pdf) (SIGMOD 1996): which views to precompute, given their cost and the queries they speed up. A caption index is a materialized view of a video; iRAG and Sprout are lazy materialization.

**Left open.** These indexes hold labels or embeddings, not language. The view-selection framing (precompute given a workload) has not been applied to caption indexes.

### 3.9 Coding for machines and semantic compression

**What the field knows.** Text is used as an ultra-low-rate code for images and video, but it is scored by perceptual reconstruction, not by machine task accuracy. Machine-task coding standards score bitrate against task accuracy, but carry pixels or features, not text.

- [Text + Sketch](https://arxiv.org/abs/2307.01944) (Lei et al., ICML 2023 workshop) *(abs)* and [PerCo](https://arxiv.org/abs/2310.10325) (Careil et al., 2023) *(abs)*: captions plus side information as image codes at ultra-low bitrates.
- [CMVC](https://arxiv.org/abs/2408.08093) (2024), [M3-CVC](https://arxiv.org/abs/2411.15798) (ICASSP 2025), [DiSCo](https://arxiv.org/abs/2512.00408) (2025) *(abs, via search)*: MLLM descriptions + keyframes + generative decoders for video at ultra-low bitrates; scored perceptually.
- [Emerging standards for machine-to-machine video coding](https://arxiv.org/abs/2512.10230) (Eimon et al., 2025) *(abs)*: MPEG VCM (pixel domain) and FCM (feature domain); rate vs. detection/segmentation/tracking accuracy.

**Left open.** A rate–task curve with *text* as the code and *QA* as the task. Our bytes axis puts captions on the same chart as frame JPEGs.

### 3.10 Theory we can build on

- **Comparison of experiments.** Blackwell (1953), "Equivalent comparisons of experiments", *Ann. Math. Statist.* 24:265–272 (venue and pages via a citing [survey](https://arxiv.org/abs/2604.06621); not opened): if one signal is a garbling of another, it is worse for *every* decision problem. Captions made from the same frames are a garbling of those frames.
- **Usable information.** [Xu et al., ICLR 2020](https://arxiv.org/abs/2002.10689) *(abs; full in first pass)*: under computational constraints, information can be created by computation. This is the only way captions can beat the frames they came from.
- **Information bottleneck and indirect rate–distortion.** [Tishby, Pereira & Bialek (1999)](https://arxiv.org/abs/physics/0004057) *(abs)*. The remote (indirect) source-coding problem of Dobrushin & Tsybakov (1962) and Witsenhausen (1980), in which the encoder sees only a noisy observation of the source, reduces to ordinary rate–distortion with a modified distortion, and IB is its log-loss case (summarized in the tutorial by [Zaidi, Estella Aguerri & Shamai](https://arxiv.org/abs/2002.00008), Entropy 2020 *(abs)*; the 1962/1980 papers not opened). A captioner is such an encoder: it sees frames, not the task, and must serve many tasks.
- **Rational inattention.** Sims (2003), "Implications of rational inattention", *J. Monetary Economics* 50(3):665–690, doi:10.1016/S0304-3932(03)00029-1 (checked via a [listing](https://collaborate.princeton.edu/en/publications/implications-of-rational-inattention/)): an agent with a cost on information chooses what to attend to. Our reader with a token budget is such an agent.
- **Efficient naming.** [Zaslavsky, Kemp, Regier & Tishby, "Efficient compression in color naming and its evolution"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6077716) (PNAS 2018) *(abs)*: languages compress perceptual space into words near the IB optimum; [follow-up](https://arxiv.org/abs/1808.03353) generalizes the principle. It predicts that a language channel keeps the distinctions a lexicon names and drops the rest.

### 3.11 Cognitive science: what language keeps

- **Codability.** Brown & Lenneberg (1954), "A study in language and cognition", *J. Abnormal and Social Psychology* 49:454–462, doi:10.1037/h0057814 (checked via [reference lists](https://opentextbc.ca/psyclanguage/chapter/references-4/)): name agreement predicts recognition memory, more so on harder tasks *(secondary source)*.
- **Verbal overshadowing.** Schooler & Engstler-Schooler (1990), *Cognitive Psychology* 22(1):36–71 ([lab page](https://labs.psych.ucsb.edu/schooler/jonathan/node/406)): describing a face impairs later recognition. A multi-lab [Registered Replication Report](https://www.academia.edu/143180751/Registered_Replication_Report) (Perspectives on Psychological Science 2014, doi:10.1177/1745691614545653) found a smaller effect; cite both.
- **The model analogue.** [Mind Your Step (by Step)](https://arxiv.org/abs/2410.21333) (Liu et al., ICML 2025) *(abs)*: chain-of-thought lowers accuracy on tasks where verbal thinking hurts humans, including a verbal-overshadowing face task (up to 36.3 points for o1-preview vs. GPT-4o).
- **Dual coding** (Paivio, 1971, a book) and event segmentation (Zacks et al. 2007, in `related_work.md`): background only; Paivio not checked.

**What this gives the paper.** A prior, theory-backed prediction of *which* question types a caption index loses (low codability: motion manner and direction, speed, fine appearance, layout, timing) and a mechanism (the captioner names what has words and is salient).

## 4. Closest work, head to head

| | Amortizes captioning | Counts query tokens | Same reader both sides | Matched budget | Budget sweep | Best visual allocation per budget | Source vs. modality separated | By task type | Energy / bytes | Short videos |
|---|---|---|---|---|---|---|---|---|---|---|
| CFD | yes | no (frames, seconds) | yes | no | frames only | no | no | 4 InfiniBench skills | storage only | Video-MME short |
| CapMem | yes | captions; video by proxy | yes (self-captioning) | frames | no (one cap) | no | partly | tags, in a figure | no | ≤ 20 min split |
| ViSIL | no | yes | yes | Pareto frontier | 4 formats | no | no | no | no | 40 s and 208 s subsets |
| Sprout | lazy | context per question | yes | no | no | no | no | no | no | no |
| LoHi | no | yes (visual) | yes | yes (visual only) | yes | **yes** | n/a | resolution-sensitive tasks | decode latency | no |
| ASP | n/a | yes | yes | yes | yes | n/a | n/a | by "wall" | no | synthetic |
| Modality Inflation | n/a | yes | yes | iso-token | no | n/a | n/a | n/a | **energy** | n/a |
| **Proposed** | yes, with Q\* | yes | yes (3 Claude models + an open VLM) | yes | yes | **yes** | **yes** | **yes** | **yes** | **yes (≤ 1 min)** |

## 5. Their open problems we can pick up

1. **CapMem: one frame cap.** A budget sweep shows where captions stop paying, as a function of query cost rather than video length.
2. **CapMem: controls on one model family.** Our readers are Claude (three sizes) and an open VLM; Gemini captions. No self-captioning.
3. **CFD: a binary router; the authors want a calibrated score.** We measure per-task caption retention at matched cost and the *routing headroom* that bounds what any such router can gain.
4. **CFD and our own logs: captioning is the bottleneck.** In our Gemini log, a median isolated caption call has 811 thinking tokens out of 1,366, and thinking is ≈ 87% of its cost (median $0.0028); caption grain and the captioner's thinking level are rate knobs.
5. **The survey: no standard accounting for caption pipelines.** We report a per-query and per-index cost card (tokens by type, bytes, latency, list-price dollars, energy on the open model, model size and context).
6. **ViSIL: small and aggregate.** Per-task curves on two public benchmarks plus our ordering task.
7. **Sprout and iRAG: eager vs. lazy indexing.** The price of task-agnosticism (question-aware vs. generic captions at equal length) is what lazy indexing buys back, at the cost of a captioner call per question.

## 6. Candidate niche and theory sketch

Working title: **"Caption once, query cheaply? A resource-complete rate–utility study of language as a video index."** Full design, hypotheses and budget: `research_plan_caption_index.md`.

1. **Accuracy–cost frontiers** per representation (caption grains; frame counts × resolutions) and per task family, with the same readers, in several cost units.
2. **Source vs. modality.** Captions made from exactly the frames the frame arm sees isolate the modality; captions from more frames add a source effect. The difference splits any caption advantage into the two.
3. **A test of CFD's duality claim** (language carries temporal structure better) under the information control, at matched cost, on short video. Our ordering pilot already contradicts it there: frames ordered 1-minute clips far better than captions (at 11× the tokens).
4. **Codability.** A pre-registered prediction that caption retention rises with how codable the answer-relevant content is (§3.11).
5. **Break-even and routing headroom** from measured frontiers, with the captioning cost counted.

Theory:
- **Garbling bound.** Captions made from frame set S are a garbling of any frame arm that sees S or more, so no reader can do better from them in principle (Blackwell). Any observed win is usable information created by the captioner (Xu et al.), or a source effect.
- **Amortization.** Per-question cost C_index/Q + c_query; break-even Q\*(α) = C_index / (c_frames(α) − c_captions(α)) at accuracy α; undefined if captions never reach α.
- **Routing.** Mixed strategies make the routed frontier the per-task upper concave envelope; its gap to the best single representation bounds any visual-need router.
- **Rate.** Captions and frames are two lossy codes at rate R (tokens or bytes); each task has its own utility curve. A question-aware captioner is a code with encoder side information; its advantage at equal rate is the price of task-agnosticism.

**On the concern that our original captions are not a realistic index:** task-agnostic, question-independent captions are what CapMem (30 s and 60 s windows) and CFD (events plus 30 s clips) use. Our per-piece captions are the same kind of index at a finer grain.

## 7. Facts for cost accounting (checked 2026-10-15)

| Item | Value | Source |
|---|---|---|
| Claude image tokens | ⌈w/28⌉ × ⌈h/28⌉ per image; 448×252 → 144, 224×126 → 40, 112×63 → 12. Max 2576 px long edge (4,784 tokens) on Claude 4.7 and later. Images under 200 px may be misread. | [Vision docs](https://platform.claude.com/docs/en/build-with-claude/vision) |
| Claude prices (per M tokens, in / out) | Haiku 5.5 $0.10 / $0.50; Sonnet 5.5 $2 / $10; Opus 5.5 $4 / $20 (cached table, 2026-10-06) | claude-api skill |
| Gemini 3 video tokens | 70 per frame at low, medium and default media resolution; 280 at high | [Media resolution docs](https://ai.google.dev/gemini-api/docs/media-resolution) |
| Our captioner, measured | Gemini 3.8 Flash isolated piece: 330 video tokens for 5 frames (66 per frame) + 187 text; median 811 thinking and 37 caption tokens; thinking ≈ 87% of the cost; median $0.0028 and 11.8 s per call (393 calls) | `results/piece_captions/calls.jsonl` |
| Our caption size, measured | Isolated piece caption: median 32 words, 184 bytes (360 captions); ≈ 45–50 Claude tokens (from pilot call totals) | `results/piece_captions/gemini-3.8-flash/isolated/` |
| Our frame size, measured | JPEG 448×252: median 22.0 KB (1,800 frames); re-encoded at quality 85: 224 px wide 7.7 KB, 112 px wide 2.9 KB | `local/unshuffle_frames/` |
| Claude reader call, measured | P12 frames: 10,002 input tokens; P12 captions: 1,940; Sonnet 0.175 points of the 5-hour window per frame call | `unshuffle_pilot_results.md` §4 |

So at a matched ≈ 40 tokens per 5 s, a caption is ≈ 40× smaller in bytes than the frame. Token matching and byte matching give different comparisons, and the paper must show both.

## 8. Corrections to the second pass (2026-10-14)

- Claude image tokens: the second pass used w×h/750. The current rule is ⌈w/28⌉×⌈h/28⌉: 448 px frames cost 144 tokens, not ≈ 150; 5 frames per piece cost 720, not ≈ 766.
- Gemini tokens per frame: "≈ 66–258" becomes 70 per frame at default media resolution on Gemini 3 (280 at high); our captioner measured 66.
- The cognitive-science citations are now checked (§3.11), with the replication caveat for verbal overshadowing. Paivio (1971) is still unchecked.
- "No one compares at matched tokens": still true for captions vs. frames, but LoHi (visual vs. visual) and ASP (synthetic, text state vs. sampling) now match budgets. The gap is narrower than stated.

## 9. Search log

**Third pass (2026-10-15).**
- **Tools:** WebSearch (standard and extended), the arXiv API through a scratchpad client (1 request per 3 s), the Semantic Scholar citations API, full text of the ML.ENERGY blog, the TempCompass paper, the Perception Test and NExT-QA repositories, and the Claude vision docs.
- **Queries:** matched token budget captions vs. frames; text vs. visual tokens at equal budget; caption memory and cost (arXiv, newest first); energy of multimodal inference; semantic and machine video coding; cost-controlled evaluation; DeepSeek-OCR and Glyph; caption-pipeline SOTA (SiLVR, LLoVi, VideoTree, DVD); question-aware vs. generic captions; caption length vs. QA; frames vs. tokens per frame; benchmarks with skill taxonomies and text-only baselines; fine-grained motion benchmarks; nameability and efficient naming; verbal overshadowing and chain of thought; Blackwell, indirect rate–distortion, rational inattention, materialized views; video databases (Zelda, LOVO, Boggart); Gemini and Claude image token rules; Kaggle quotas; ARR deadlines.
- **Citation chase (Semantic Scholar):** ViSIL: VEGAS, CodecCap. CFD: VEI and a scoping review. CapMem, CapQuiz: none listed. CaptionQA: 16 citing papers, new relevant ones CapProbe, CAPEval and SwitchMem.
- **New close papers found:** Sprout, LoHi, ASP, OneStreamer, SwitchMem, CodecCap, TempCore, Modality Inflation, ML.ENERGY v3.
- **Not done:** full text of LoHi, Sprout, ASP, CodecCap and Modality Inflation (abstracts only); the Dobrushin–Tsybakov, Witsenhausen and Blackwell originals; Paivio; Perception Test per-file download availability; current official Kaggle quota.

**Second pass (2026-10-14).** Full text of CFD, CapMem, ViSIL, the aligned-caption RAG paper and the survey's scope section. Also cited from that pass: [Aligned video captions for RAG](https://arxiv.org/abs/2405.17706) (Dela Rosa, 2024; 1-fps frames would need ≈ 69× the tokens of the captions, an estimate) and [SEG](https://arxiv.org/abs/2601.06097) (Dixit & Liang, 2026; a pruned symbolic event graph at 3.47k vs. 40.39k tokens per query, 5 videos).
