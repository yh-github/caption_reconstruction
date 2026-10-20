# Research plan: caption once, query cheaply?

Written 2026-10-15. It builds on the literature review `related_work_caption_index.md` (third pass) and on what we already have (`unshuffle_pilot_results.md`, `docs/report/status_report_2026-10-14.md`). **It is a proposal until the user and partners decide (§12).** If adopted, it replaces `paper_plan.md` as the master plan, and the unshuffling study becomes one task family inside it (§5.3).

**Terms (fixed 2026-10-18, revised 2026-10-20; use only these).**
- **Codability:** how consistently and briefly people put the answer-relevant content into words, measured by agreement between independent descriptions (Brown & Lenneberg 1954). It replaces "nameability", which earlier drafts used for the same idea. A question type has high codability when different describers would state its answer the same way, in few words.
- **Visual tokens:** the tokens a model spends on visual input (frames or video) in one call.
- Four **effects**. Each is the accuracy difference between two arms that changes one thing, with the same reader in both:
  - **Source effect:** what the captioner was given (the video, or more or larger frames) instead of the frames the reader sees. Captioner, prompt and visual tokens are held fixed. It replaces "coverage" (2026-10-15 to 2026-10-19), which suggested only how much of the video was seen. In C1 and C2 both inputs are the same 1 fps instants; what differs is the input itself (the video file vs. the reader's 224 px JPEGs).
  - **Visual budget effect:** the visual tokens the captioner spends on the same pixels. Captioner and pixels are held fixed.
  - **Captioner effect:** a different captioning model instead of the reader model itself. Pixels and prompt are held fixed, and visual tokens about fixed. It includes everything that differs between the two models: what they perceive, what they recognize, how they write, their errors. No arm separates these, so claims say "captioner effect", not "knowledge".
  - **Modality effect:** the reader reads its own caption of the frames (text) instead of the frames themselves (images). The model and the pixels are held fixed. It replaces "format effect" (2026-10-18 to 2026-10-20), which suggested a change of encoding (JSON to YAML), not a different type of input. Before that, drafts called it the "code" term, which clashed with codability. "Format" keeps only its plain sense: an answer in the required output format.
  - With visual tokens held about fixed, the total caption effect = source + captioner + modality. Where two arms differ in visual tokens, that difference is reported as a visual budget effect.
- **Information about the video** (Shannon, Blackwell) vs. **usable information** (Xu et al. 2020: what a given bounded reader can extract). Captions never add the first. They can add or remove the second.

## 0. At a glance

| | |
|---|---|
| **Question** | For short videos (≤ 1 min), when does a one-time, question-independent caption index answer questions better per unit of query cost than frames, why, and when does the indexing pay for itself? |
| **Contribution** | Accuracy–cost frontiers per task family, with the same readers and both modalities at their best allocation per budget; an information control that splits any caption advantage into source, visual budget, captioner and modality effects; a full cost account (tokens by type, bytes, latency, list-price dollars, energy, model size); a tested explanation (codability); break-even and routing-headroom numbers from measurements. |
| **Theory** | Garbling bound (Blackwell), usable information, amortized break-even, routing headroom by LP duality, price of task-agnosticism (§3). |
| **Data** | 30 WildQA pilot videos (ordering, reused), TempCompass (fine temporal aspects), Perception Test or NExT-QA (breadth). |
| **Readers** | Claude Haiku / Sonnet / Opus 5.5 through `claude -p`; one open VLM on Kaggle, where energy and memory are measured. Gemini only writes captions. |
| **Cost** | Gemini ≈ $43 at the current thinking level (too tight against ≈ $50 of credits), ≈ $12 if lower thinking passes a quality check (E0). Claude ≈ 15k Sonnet, 15k Haiku and 3k Opus calls, mostly small: about 75 points of the 7-day figure over 5 weeks (≈ $135 at list prices, paid by the subscription). Kaggle ≈ 15 GPU-hours. |
| **Time** | About 6 weeks of experiments and 3 of writing. Target: ACL 2027 through ARR (cycle expected around January 2027; not yet confirmed) or TMLR (rolling). |

## 1. Research question, thesis and contributions

**Motivation.** Captioning a video is a one-time cost; afterwards, many questions could be answered from the captions instead of paying for the video again (the user's framing, 2026-10-14). The literature has the amortized formula (CFD, CapMem) but not the measurement that decides it: accuracy against query cost, per task, with what the captioner saw controlled. And the 2026 survey of video-LLM efficiency leaves caption pipelines out because their costs are hard to account for.

**Research questions.**
- **RQ1 (rate–utility).** At a fixed query budget, which representation gives a fixed reader the highest accuracy: captions at several grains, or frames at several counts and resolutions? How does the answer change by task family and by cost unit (tokens, bytes, energy, dollars)?
- **RQ2 (mechanism).** When captions win at a matched budget, is it because the captioner was given more of the video (source effect), spent more visual tokens on it (visual budget effect), is a model that perceives, recognizes or writes differently from the reader (captioner effect), or because text is easier for the reader to use than pixels (modality effect)?
- **RQ3 (explanation).** Does the codability of the answer-relevant content predict how much accuracy the caption index keeps?
- **RQ4 (economics).** Counting the captioning cost, after how many questions per video does the index pay off, and how much could a router that sends some question types to frames gain?

**Thesis (to be tested, not assumed).** Language is a selective, low-rate channel: it keeps what has names and drops what does not. Per token, captions should be competitive only where (a) the content has high codability and (b) the captioner saw more than the query budget lets the frames show. Where neither holds (motion, speed, fine appearance, timing), frames should win at every budget.

**Contributions.**
1. The first accuracy–cost frontiers of a caption index vs. frames on short video, per task family, with each modality at its best allocation per budget, the same readers, and a blind (question-only) arm.
2. An information control that splits a caption advantage into source, captioner and modality effects (with visual tokens held fixed), and a test, under that control, of the published claim that language carries temporal structure better (CFD).
3. A cost card for video QA that counts what token budgets hide: image vs. text tokens, thinking tokens, payload bytes, latency, energy and memory on an open model, and the one-time index cost.
4. A pre-registered test of codability as the predictor of what survives the language channel, linking the result to cognitive science and to efficient-naming theory.
5. Break-even query counts and routing headroom from measured frontiers, which say when caption indexing pays and how much any visual-need router (CFD-style) could gain.

## 2. Setting and definitions

- A video X is cut into windows of 5 s (the unit we already use). A question Q has a type t (task family) and a correct answer Y = y(X, Q).
- A **representation** φ maps X to an input Z for the reader, through a Markov kernel K_φ(z | x). It is **task-agnostic** if it does not depend on Q. Captions and cached frames are task-agnostic; question-aware captions are not.
- A **reader** R maps (Z, Q) to an answer, possibly randomly. We use fixed readers (Claude models at pinned IDs and effort; one open VLM), so the reader class is restricted.
- **Accuracy:** A_t(φ, R) = P(R(Z, Q) = Y | type t).
- **Query cost:** c(φ, R) ≥ 0 in an additive unit u (input tokens, total tokens, bytes, joules, list-price dollars). **Index cost:** C(φ) ≥ 0 per video, paid once (captioning). Frames have C ≈ 0 (decoding and storage are logged but small).
- **Retention:** ρ_t = (A_t(captions) − A_t(blind)) / (A_t(frames) − A_t(blind)), relative to the question-only arm, so language priors do not count as caption utility.

## 3. Theory

Each statement says what it lets us conclude from the experiments. Proofs are short because the value is in the measurements they license, not in the mathematics.

**Proposition 1 (garbling bound; information control).** Let ψ = G ∘ φ for a kernel G that does not depend on (X, Q): the captioner sees only φ's output and not the question. Then for every task type t,

sup_R A_t(ψ, R) ≤ sup_R A_t(φ, R),

where the supremum is over all (randomized) readers.

*Proof.* For any reader R of ψ, the reader R′(z, q) = R(z′, q) with z′ ~ G(· | z) reads φ and produces the same joint law of (answer, Y), hence the same accuracy. ∎ (This is the easy direction of Blackwell's 1953 theorem; it holds for every loss, not only accuracy.)

*What it licenses, and what it does not.* The bound is about information **about this video**, and it holds because the supremum includes readers that know everything the captioner knows. The captioner's world knowledge is part of the fixed kernel G, so it adds no information about X. Our readers are fixed and bounded, and the captioner is a different model from the reader, so for them captions **can** do better than frames. The captioner can name what the reader would not recognize (a species, a sport's move, a tool). That is information the reader lacked, though not information about the video that was missing from the frames. Captions can also do worse, through errors and hallucinations. So with the captioner given exactly the frame the frame arm sees (arm T3 vs. V1, §5.1), an observed caption win is a gain in **usable** information for this reader (Xu et al. 2020), coming from the captioner (what it perceives, recognizes or writes that the reader does not) or from the text modality. The unshuffling plan's Proposition 1 is the special case where the task is order.

**Corollary (source + captioner + modality decomposition).** Let T_S be captions made by the captioner from frame set S, S′ ⊆ S the frames the frame arm shows, and T^R_S′ captions of S′ written by the reader model itself (self-captioning). Then exactly

A(T_S) − A(V_S′) = [A(T_S) − A(T_S′)] + [A(T_S′) − A(T^R_S′)] + [A(T^R_S′) − A(V_S′)] = source + captioner + modality.

- **Source:** the captioner was given more than the reader's frames. Either sign.
- **Captioner:** what a different model's description adds over the reader's own description of the same frames. Either sign.
- **Modality:** the reader's own description, as text, instead of the frames it described. Proposition 1 bounds this term only in the limit of unbounded readers. For a real reader, it can have either sign.
- The identity holds for any arms. Each term isolates one change only if the two arms it compares spend about the same visual tokens; otherwise a visual budget effect is mixed in, and needs its own pair of arms (C2 in `research_log.md`).

**Proposition 2 (amortized cost and break-even).** For Q questions on one video, the total cost is TC_φ(Q) = C(φ) + Σ_q c_q(φ). For a target accuracy α, let c*_φ(α) = inf { c(φ_b, R) : A(φ_b, R) ≥ α } over the budgets b and readers R in a family (the inverse of its frontier). A caption index τ costs less in total than frames f for every Q ≥ Q*(α), where

Q*(α) = C(τ) / (c*_f(α) − c*_τ(α)) if c*_f(α) > c*_τ(α), and never otherwise (including when τ never reaches α).

*What it licenses.* Q* is the number to report, with a bootstrap CI over the measured frontiers, per task family and cost unit. If the index lets a cheaper reader reach α (Haiku on captions vs. Sonnet on frames), c*_τ uses that reader; the index can then pay off through the reader downgrade, not only through fewer tokens. Lazy indexing (iRAG, Sprout) is the same formula with a question-aware captioning call inside c_τ and C = 0.

**Example with our measured prices (illustration, not a result).** Our isolated caption costs $0.0028 per 5 s window (median). Replacing one 448 px frame (144 tokens) with a caption (≈ 47 tokens) saves ≈ 97 Sonnet input tokens, ≈ $0.0002 per window per question, so Q* ≈ 14 questions per video, *if* accuracy is equal. Replacing 5 such frames (720 tokens) gives Q* ≈ 2. At a lower captioner thinking level (§9), both fall several-fold. So Q* can fall on either side of a realistic workload, and only measurement decides it.

**Proposition 3 (routing headroom).** Let the workload have types t with weights w_t, and let Φ be a finite set of (representation, reader) options with accuracies a_t(φ) and costs c_t(φ). A router that knows the type picks a distribution π_t over Φ for each t; a single policy uses the same π for all t. Let R(B) and S(B) be the best average accuracy at average cost ≤ B for routed and single policies. Then:
1. R and S are concave, non-decreasing and piecewise linear in B, and R(B) ≥ S(B).
2. R(B) = min_{λ ≥ 0} [ λB + Σ_t w_t max_φ ( a_t(φ) − λ c_t(φ) ) ].

*Proof.* R(B) is a finite linear program in the π_t (linear objective, one budget constraint, simplex constraints). It is feasible for B ≥ min cost and bounded, so strong duality holds; dualizing only the budget constraint separates the simplices, and each inner maximum is attained at a vertex. Concavity and piecewise linearity follow from parametric LP. S(B) is the same LP with π_t tied, a restriction, so S ≤ R. ∎

*What it licenses.* The headroom H(B) = R(B) − S(B) is an upper bound on what any router that uses only the question type can gain at budget B (CFD's visual-need router is of this kind when its signal is the question type). An item-level oracle (best option per question) bounds any router at all. Both come from our per-type frontiers at no extra cost.

**Proposition 4 (price of task-agnosticism).** Among captioners limited to L words, one that also sees the question can do at least as well as one that does not, for an optimal reader, since it can ignore the question. We define the price of task-agnosticism Δ_TA(L) = A(question-aware captions, L) − A(generic captions, L) for our fixed captioner and reader, and measure it (E5). In coding terms, a question-aware caption is a code with side information at the encoder as well as the decoder; generic captions have it only at the decoder (the reader sees the question). Lossy coding with decoder-only side information pays a rate penalty in general (the Wyner–Ziv setting); we use this as an analogy, not a bound.

**Remark (capacity per token).** A text token from a vocabulary of about 2^17 words carries at most about 17 bits, so a 40-token caption carries at most ≈ 680 bits; its actual size is 184 bytes. The 40-token frame we would match it with is a 7.7 KB JPEG (≈ 62 kbit). At matched tokens the visual input is about two orders of magnitude larger. A caption that wins per token therefore wins by **selection** (the captioner chose what to keep, often from more frames) and **usability** (the captioner and the text modality), never by capacity. This is why the source/captioner/modality split is the right decomposition.

**What the theory does not give.** No proposition predicts which task families win. That is an empirical claim, made in advance through the codability hypothesis (H4) and tested.

## 4. Hypotheses

Fixed now in direction. The confirmatory versions (exact arms, budgets, margins, and the codability scores) are frozen in a pre-registration after the development split (§7).

| | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 Modality | At equal source (caption of exactly the frame the reader sees, T3, vs. that frame, V1), the captioner + modality effect is negative for low-codability families and not negative for high-codability families. | Arm × codability-group interaction, paired by question | No interaction, or the reverse sign |
| H2 Source | A caption made from all 5 frames of a window beats a caption made from 1 of them at the same length (T2 > T3), most for action, order and change questions. | Paired difference, pooled; by family as secondary | T2 ≤ T3 |
| H3 Frontier | For low-codability families (direction, speed, order of continuous motion, fine appearance), a frame arm at no more tokens than the dense captions is more accurate (V1 > T2). For high-codability families (objects, actions, scene, semantics), the dense captions are non-inferior to V1 within 3 points. | One-sided test; non-inferiority test (margin 3 points) | Frames do not win on low-codability; captions are inferior on high-codability |
| H4 Codability | Across question types (≥ 20 over both benchmarks), retention at the ≈ 40-token budget rises with a priori codability, and the caption's mention rate of the answer-relevant content mediates it. | Spearman ρ > 0 (one-sided); mediation as secondary | ρ ≤ 0 |
| H5 Break-even (estimation, no direction) | Q\*(α) per family and cost unit, with CIs; with and without a reader downgrade (Haiku on captions). | Pre-registered computation | n/a |
| H6 Routing (estimation) | Routing headroom H(B) at the dense-caption budget, on each benchmark's natural question mix. | LP on measured frontiers, bootstrap CI | n/a |
| H7 Duality (CFD) | Under the information control, captions do *not* carry the order of events better than frames on short video (the ordering pilot already points this way). | Ordering τ, T3 vs. V1 and T2 vs. V4 | Captions order better at equal source |

Primary confirmatory tests: H1, H2, H3 (its two parts) and H4: five tests, Holm-corrected. Everything else is labelled exploratory.

## 5. Design

### 5.1 Representations (per 5 s window; Claude tokens by the official ⌈w/28⌉×⌈h/28⌉ rule)

| Arm | What | Tokens | Bytes | Role |
|---|---|---|---|---|
| T0 Blind | Question and options only | 0 | 0 | Language-prior baseline (TVBench lesson) |
| T1 Terse | Caption ≤ 10 words, from the window at 1 fps | ≈ 15 | ≈ 60 | Low rate point |
| T2 Dense | Caption ≤ 45 words from the window's 5 frames (1 fps, the V4 JPEGs; frozen prompt v2) | ≈ 47 | 184 (measured) | The index; source arm |
| T3 Matched | Caption ≤ 45 words from **the single JPEG the V1 arm shows** | ≈ 47 | ≈ 184 | Information control (Prop. 1) |
| T4 Clip | One caption ≤ 80 words for the whole clip | ≈ 9 per window on 60 s | ≈ 40 per window | Coarse grain |
| T5 Aware | Like T2, but the captioner sees the question | ≈ 47 | ≈ 184 | Price of task-agnosticism (not amortizable) |
| V0 | 1 frame, 112×63 | 12 | 2.9 KB | Smallest frame; Claude warns that images under 200 px may be misread |
| V1 | 1 frame, 224×126 | 40 | 7.7 KB | Matched to T2 and T3 in tokens |
| V2 | 1 frame, 448×252 | 144 | 22 KB | Full-resolution single frame |
| V3 | 5 frames, 112×63 | 60 | ≈ 15 KB | Dense low-resolution allocation (LoHi) |
| V4 | 5 frames, 224×126 | 200 | ≈ 38 KB | Dense medium |
| V5 | 5 frames, 448×252 | 720 | ≈ 110 KB | The pilot's frame arm |
| H1 | T2 + V1 | ≈ 87 | ≈ 7.9 KB | Hybrid (CFD-like) |

Byte figures are measured on our frames (JPEG quality 85 for the re-encodes) and captions; tokens for captions are estimated from the pilot's call totals and will be logged exactly.

- Frames are always taken at the same instants the captioner saw (1 fps, t = 5p + k, as in `scripts/unshuffle/frames.py`).
- **T2 and T3 are made from the same JPEGs the frame arms show** (224×126; 5 per window for T2, the V1 frame for T3). Then T2 is a garbling of V4 and T3 of V1 (Prop. 1), and T2 − T3 isolates the number of frames the captioner saw, not their resolution. The existing WildQA isolated captions were made from the video file at Gemini's own resolution; they stay as a robustness variant, and a JPEG-based T2 is added (360 calls, ≈ $1).
- **Each modality gets its best allocation per budget.** The frame frontier is the upper envelope over V0–V5, chosen on the development split; the caption frontier the envelope over T1, T2, T4. Comparing one caption with one arbitrary frame would strawman the frames (LoHi, VEI).
- Audio stays removed in every arm.

### 5.2 Readers

- **Claude Sonnet 5.5** (main), **Haiku 5.5** (every arm; also the "cheap reader" of the index), **Opus 5.5** (core arms on subsets). Pinned IDs `claude-{haiku,sonnet,opus}-5-5`, effort medium, no tools, one isolated `claude -p` session per item, images through stream-json: the existing `scripts/unshuffle/claude_reader.py` machinery.
- **One open VLM on Kaggle** (candidates Qwen2.5-VL-7B, Qwen3-VL-8B or a 4B variant, whichever runs stably in fp16 on the available GPU; chosen in Phase 0). The same model reads text-only and image inputs. This is where energy, peak memory and stage timing are measured, and it is the second model family that reviewers will ask for.
- Thinking and output tokens count as query cost. In the pilot, Opus thought ≈ 400 tokens on frame puzzles and 0 on caption puzzles, so the representation also changes how much the reader writes.

### 5.3 Tasks and data

| Set | Use | Size | Why |
|---|---|---|---|
| **WildQA ordering** (30 pilot videos, P12 and G6, 2 shuffles) | E1: the "when" family; pipeline and quota calibration | 120 calls per arm per reader | Reuses captions, frames, puzzles, metrics and reader; the strongest information-control precedent; tests H7 |
| **TempCompass** (MCQ format) | E2: fine temporal aspects (action, speed, direction, order, attribute change) | 410 videos (mostly ≤ 30 s); 700 questions sampled, stratified by aspect (100 development, 600 confirmatory) | "Conflicting videos" defeat single-frame and language priors; spans high codability (action) to low (speed, direction). CC BY-NC 4.0 (research use). |
| **Perception Test** (MCQ, validation split) | E3: breadth (memory, abstraction, physics, semantics × descriptive, explanatory, predictive, counterfactual) | 700 questions on ≈ 350 videos of ≈ 23 s, stratified (100 development, 600 confirmatory) | Real-world, short, CC-BY, built to probe skills. **Risk:** the validation videos are a 70.2 GB archive; if single files cannot be fetched, fall back to NExT-QA. |
| Fallback: **NExT-QA** (val) | E3 alternative | 700 questions | Causal / temporal / descriptive; the benchmark of caption pipelines (LLoVi, VideoTree), so the index can be compared with published numbers |

**Why short video.** It is the hardest case for captions (frames are cheapest there), CapMem reports captions losing at ≤ 20 minutes with matched frames but never tested per token, and it fits the budget. Long video is stated as out of scope.

### 5.4 Codability (frozen before the confirmatory runs)

- **A priori score per question.** Raters see the question and options only, never a video or a result: "How likely is a short, general-purpose description of the video, written without knowing this question, to state what is needed to choose the right option?" and "How consistently would different people put the distinguishing information into words?" (1–5 each). Raters: Claude Haiku and Sonnet, one open LLM, and two authors on a 100-question sample. Report agreement (Krippendorff's α).
- **Type score:** the mean over a question type (TempCompass sub-aspect; Perception Test area × reasoning type; ordering as one type).
- **Mediator:** the mention rate, i.e. whether the T2 captions contain the answer-relevant information (Haiku judge given the question, the caption and the correct option; validated by hand on 50).
- **Background:** codability (Brown & Lenneberg 1954), efficient naming as an information bottleneck (Zaslavsky et al. 2018), verbal overshadowing and its replication (§3.11 of the review).

### 5.5 The cost card (logged for every call)

| Per query | Per index (per video) |
|---|---|
| input tokens by type (text, image), output tokens, thinking tokens | captioner input tokens by type, output and thinking tokens |
| payload bytes | dollars at list price, with the price date |
| wall-clock latency; CPU time for frame extraction and JPEG encoding | wall-clock time; CPU decode time |
| list-price dollars (Claude prices as of 2026-10-06: Haiku $0.10/$0.50, Sonnet $2/$10, Opus $4/$20 per M tokens) | storage bytes |
| open VLM only: energy (NVML counter, idle-subtracted), peak GPU memory, time split into vision encoding, prefill and decoding | open captioner only: energy and peak memory |
| model ID reported, effort, parameter count (open model), context used | captioner model version, prompt hash |

Every frontier is drawn in at least two units (tokens and dollars for Claude; tokens, joules and seconds for the open VLM; bytes for all), and every conclusion states the units in which it holds.

### 5.6 Metrics

- MCQ: accuracy, blind-adjusted retention ρ_t.
- Ordering: Kendall τ, undirected adjacency, local direction (as in the pilot).
- Frontier summaries: area under the accuracy–log-cost curve over a common range; dominance at matched cost; Q\*(α) at α set to the frame arm's accuracy at 40 tokens per window; H(B).

## 6. Experiments

| | Question | Arms × readers | Data | Calls |
|---|---|---|---|---|
| **E0 Setup** | Do the instruments work? | Gemini thinking level check (20 pieces, default vs. low, MRR + hand read); open-VLM smoke test with energy logging; token counts per arm within ±10% of §5.1 | 2 videos | ≈ 40 Gemini, ≈ 50 Haiku |
| **E1 Ordering budget curve** | Does any caption arm order better than frames at matched cost (H7)? Calibrates quota per small call. | T1, T2, T3, V0, V1, V3, V4, V5, H1 × Sonnet and Haiku; T2, T3, V1, V4 × Opus; all arms × open VLM | 30 WildQA videos, P12 + G6, 2 shuffles | Sonnet ≈ 1,100 new, Haiku ≈ 1,100, Opus ≈ 400 |
| **E2 TempCompass map** | RQ1–RQ3 on fine temporal aspects | Core arms T0, T1, T2, T3, V1, V3, V4, H1 × Sonnet and Haiku; T2, T3, V1, V4 × Opus (300 questions); all arms incl. V0, V2, V5, T4 × open VLM | 700 MCQs | Sonnet 5,600, Haiku 5,600, Opus 1,200 |
| **E3 Breadth map** | RQ1–RQ3 across skills | As E2 | 700 MCQs (Perception Test or NExT-QA) | Sonnet 5,600, Haiku 5,600, Opus 1,200 |
| **E4 Cost and energy** | Do the conclusions hold in joules, seconds and bytes? Index cost on free hardware. | Open VLM on every arm of E1–E3 with the full cost card; an open captioner (e.g. a small Qwen-VL) writes T2 for one dataset, so the index cost is also measured in joules | E1–E3 | ≈ 20k open-VLM calls (Kaggle) |
| **E5 Price of task-agnosticism** | How much do question-aware captions add at equal length (Prop. 4)? | T5 vs. T2 × Sonnet | 300 TempCompass + 300 E3 questions | Sonnet 600 |
| **E6 Codability** | H4 | LLM raters on all 1,400 questions; authors on 100; mention-rate judge | E2, E3 | Haiku ≈ 2,800, Sonnet ≈ 1,400 (tiny calls) |
| **E7 Robustness** | Is it Gemini, one seed, one reader? | Second captioner (Sonnet writes T2 for 100 videos); caption resampling at a new seed (10%); the open VLM as a second reader family throughout | subsets | Sonnet ≈ 1,000, Gemini ≈ $3 |

**Analysis outputs (all computed from these runs, no extra calls):** frontiers per family and unit; the source/captioner/modality decomposition (source T2 − T3; captioner T3 − self-captions of the same frame; modality self-captions − V1; and T2 − V4 at five frames) per family; Q\*(α) per family and unit, with and without the reader downgrade; routing headroom H(B) and the item-level oracle bound; codability regression and mediation.

## 7. Statistics and pre-registration

- **Unit and clustering:** the question, nested in video (and in channel for WildQA). CIs by cluster bootstrap over videos (channels for WildQA), using `boot` and the unshuffle bootstrap code. Effects by mixed logistic models (correct ~ arm × family + (1 | video / question)); the two must agree.
- **Development vs. confirmatory:** 100 development questions per benchmark choose prompts, the best allocation per budget, and the codability items. The pre-registration (arms, budgets, margins, tests, exclusion rules) is committed before any confirmatory run. E1 is development data for the ordering family (its 30 videos were the pilot).
- **Multiplicity:** five primary tests (H1, H2, H3a, H3b, H4), Holm-corrected; the rest exploratory and labelled. Null claims use equivalence tests with margins fixed in advance (3 points).
- **Power (rough, to be redone from development data):** with 600 confirmatory questions per benchmark and about 25% discordant pairs, a paired test detects about 4 points at 80% power on the pooled set and about 6 points within each codability half. H4 with ≈ 25 question types detects ρ ≈ 0.5.
- **Format failures** count as wrong for MCQ (as random for ordering, as in the pilot) and are reported per arm.
- **Position bias:** MCQ option order shuffled with a fixed seed per question, identical across arms.

## 8. What we reuse

| Asset | Where | Reused for |
|---|---|---|
| 30 pilot videos with isolated and in-context captions | `results/piece_captions/gemini-3.8-flash/` | E1: the isolated captions as a robustness variant of T2 (made from the video file); in-context captions for the leak result |
| Reader frames at 448 px, at the captioner's instants | `local/unshuffle_frames/` (`scripts/unshuffle/frames.py`) | V2, V5, and downscaled V0, V1, V3, V4 |
| Puzzles, metrics, bootstrap, analysis | `scripts/unshuffle/core.py`, `analyze.py` | E1 and the shared bootstrap |
| Claude reader with images, quota and usage logging | `scripts/unshuffle/claude_reader.py` | All Claude arms; add MCQ item sets, arm builders and the cost card |
| Pilot reader results (Haiku, Sonnet, Opus on T2 and V5, 10 videos) | `results/unshuffle/claude/` | Part of E1 (no re-run) |
| Gemini runner with cost cap, resume, no audio, prompt hash | `scripts/caption_pieces.py` | All captioning; add an image-input mode (T3), terse and clip prompts |
| Edge arm and SigLIP 2 embeddings | `scripts/unshuffle/arms.py`, `local/wild_videos_embs_siglip/` | Continuity-only reference for ordering |
| Timing findings (captions run 1–2 s early, keep topic and lose timing) | status report §5 | Background and motivation for the "when" family |
| Kaggle credentials and the earlier Llama setup | `local/kaggle.json` | Open VLM and energy runs |

Not reused: generation (already demoted), the original whole-minute captions (they leak), and the recognition items (built on those captions; appendix at most).

## 9. Budget

**Gemini (captions only).** Measured median $0.0028 per isolated window call at the current setting (`calls.jsonl`, 393 calls); the median call has 811 thinking tokens out of 1,366, and thinking is ≈ 87% of its cost.

| Item | Calls | At current thinking | If low thinking passes E0 |
|---|---|---|---|
| E0 thinking check | 40 | $0.10 | — |
| E1: T1, T2 from JPEGs and T3 for 30 videos (3 × 360 windows), T4 (30) | 1,110 | $3.2 | ≈ $0.9 |
| E2 TempCompass: T1, T2, T3 (3 × ≈ 1,230 windows), T4 (410 clips), T5 (300 questions × ≈ 3 windows) | ≈ 5,000 | $15 | ≈ $4 |
| E3: T1, T2, T3 (3 × ≈ 1,750 windows), T4 (350 clips), T5 (300 questions × ≈ 5 windows) | ≈ 7,100 | $21 | ≈ $6 |
| E7 resampling | ≈ 1,000 | $3 | ≈ $1 |
| **Total** | ≈ 14,300 | **≈ $43** | **≈ $12** |

Assumptions: $0.0028 per window call and ≈ $0.006 per whole-clip call at the current setting; ≈ $0.0008 per window call at low thinking (the same input, ≈ 150 output tokens), which E0 must confirm. Sending 5 JPEGs as images instead of a video file adds ≈ 1,000 input tokens (≈ $0.0005) per call at Gemini 3's image rate; packing them into a 1-fps clip avoids it. Window counts assume ≈ 3 windows per TempCompass video and ≈ 5 per Perception Test video; E0 replaces them with real counts. **At the current setting the plan uses ≈ $43 of the ≈ $50 of credits**, which is too tight: either lower thinking passes E0, or T5 drops to 150 questions per benchmark (−$3) and T4 is made from T2 instead of the video (−$4). A lower thinking level is itself a reportable index-cost result. Spending rules as before: `--max-usd` is cumulative, each phase starts with `--limit`, and the user confirms before each larger run.

**Claude (subscription quota).** Pilot measurements: a 10k-token Sonnet call costs 0.175 points of the 5-hour window, and about 80 such calls move the 7-day figure by 1 point; Opus costs about the same per call in the 5-hour window and twice as much in the 7-day figure. Most calls here are 1–3k tokens; E1 measures their real cost before anything larger runs.

| Reader | Calls | Estimated 7-day points | List-price equivalent |
|---|---|---|---|
| Sonnet | ≈ 15,200 | ≈ 50 | ≈ $100 |
| Opus | ≈ 2,800 | ≈ 20 | ≈ $30 |
| Haiku | ≈ 15,000 | ≈ 5 | ≈ $5 |
| **Total** | | **≈ 75 over ≈ 5 weeks** (≈ 15 a week) | **≈ $135** |

The 5-hour windows needed: roughly 8–10 for Sonnet and 3 for Opus, spread over the weeks.

**Kaggle.** About 20k open-VLM calls with short answers (≈ 1–2 s each) plus an open captioning pass: ≈ 15 GPU-hours, about half of one week's quota (commonly reported as ≈ 30 h; check the account).

**Storage and people.** TempCompass and a Perception Test subset: a few GB. Authors: about 6 hours (codability ratings, mention-rate check, caption spot reads).

## 10. Phases, gates and timeline

| Week | Phase | Work | Gate |
|---|---|---|---|
| 1 | 0. Setup (free) | Theory section (Props. 1–4, Monte Carlo of H(B) on synthetic frontiers); cost-card logger for Claude and Kaggle; E0; download and sample TempCompass and Perception Test (or NExT-QA); codability ratings; pre-registration draft | — |
| 2 | 1. Ordering | E1 | **Gate A:** token counts as designed; format failures < 2%; energy logging works; quota per small call measured; Opus subset sizes fixed |
| 3–4 | 2. TempCompass | Captioning; development split; freeze and commit the pre-registration; confirmatory run; open VLM | **Gate B:** pre-registration committed before the confirmatory runs |
| 4–5 | 3. Breadth | E3 | — |
| 6 | 4. The rest | E5, E6 mediation, E7; full analysis | **Gate C:** results to the user and partners |
| 7–9 | 5. Writing | Paper and artifacts (index, logs, cost card, code) | Venue decision |

**Venue.** ACL 2027 through ARR (the cycle is expected around January 2027; the date is not yet announced, so check it) fits the timeline, and EMNLP-style venues already published CFD and CapMem. TMLR (rolling, values rigor over novelty) fits a careful measurement-and-theory paper. A CVPR 2027 deadline would be too early.

## 11. Risks

| Risk | Handling |
|---|---|
| Scooping: CFD, CapMem, ViSIL or LoHi authors publish per-task matched-budget curves | Lead with what is hard to copy: the information control, the full cost card, the theory. Monthly arXiv and citation watch on CFD, CapMem, ViSIL, LoHi, Sprout. |
| "Obviously frames win on motion and captions on objects" | Directional predictions are pre-registered; the paper's value is in the magnitudes, the source/captioner/modality split, Q\*, the headroom, and the test of codability as the explanation. |
| Token units differ across readers | Report per reader, plus bytes, dollars and joules; claim only what holds in at least two units. |
| Readers treat text and images unequally (TokenSwap), and think more on images | Same effort for every arm; thinking tokens counted as cost; the open VLM as a second reader. |
| Claude misreads images under 200 px (V0, V3) | That is a real constraint of the reader and part of its frontier; the open VLM shows whether it is reader-specific. |
| Benchmarks may be in training data | Affects both modalities; the blind arm measures priors; retention is blind-adjusted. |
| Perception Test download (70.2 GB archive) | Check single-file access in Phase 0; otherwise NExT-QA. |
| Gemini credits (≈ $43 of ≈ $50 at the current thinking level) | E0 thinking check; `--max-usd`; halve T5 and derive T4 from T2 if needed (§9). |
| Claude quota | Calibrated at Gate A; Haiku everywhere, Opus on subsets; the run spreads over 5 weeks. |
| Kaggle hardware (fp16 on older GPUs; quota) | Model size chosen in Phase 0; 4B fallback; ≈ 15 of ≈ 30 weekly hours. |
| Model drift | Pinned IDs, reported model logged per call; each experiment runs within a short window. |

**What we will not claim:** anything about long video; energy for closed models (only tokens, bytes, latency and dollars); that our captioner is the best possible (we measure one strong captioner and one open one).

## 12. Decisions needed

- [ ] **Direction:** adopt this as the main paper, with unshuffling as the ordering family (E1) and generation as a short negative result? (Recommended.)
- [ ] **Breadth benchmark:** Perception Test (recommended: skill taxonomy, short, CC-BY) or NExT-QA (comparable with published caption pipelines).
- [ ] **Captioner thinking level:** decided by the E0 check.
- [ ] **Opus subset:** the subset sizes above, confirmed at Gate A.
- [ ] **Venue:** ACL 2027 via ARR, or TMLR.
- [ ] **Human codability raters** beyond the two authors (optional; a few colleagues, about 1 hour each).

**Alternatives considered.**
- *Continue unshuffling as the main paper.* Its central bet failed in the pilot (frames beat captions in direction too), and it does not answer the user's cost question.
- *A short paper on caption timing* ("frontier captions keep topic, lose timing, run 1–2 s early"). Robust and nearly written, but narrow; it becomes motivation here.
- *An energy-only systems study.* Measurable on Kaggle, but without the accuracy side it says nothing about whether the index is worth it; folded in as E4.
