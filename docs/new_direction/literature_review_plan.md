# Literature review plan: is unshuffling a meaningful contribution?

Hand-off brief for a research agent. Written 2026-10-13. Read `docs/new_direction/new_plan.md` first; it describes the study this review has to position.

## The question

Has anyone already compared **how much of a video's temporal order can be recovered from language versus from pixels, on the same clips, under a control that separates visual continuity from knowledge of how events unfold?**

Ordering as a task is old and well covered. Finding ordering papers is expected and is not a threat. The review must establish where our specific comparison sits, and either confirm the gap or find the paper that already fills it.

## Our study in one paragraph (what we must position)

Each 60 s clip (WildQA, 335 clips, 40 YouTube channels) is cut into 12 pieces of 5 s. Each piece is captioned **in isolation** by Gemini (no context from other pieces), from the pictures only (no audio). Readers get the pieces shuffled and restore the order, with matched arms in both modalities: similarity chaining with no reader (SigLIP 2 similarity of edge frames; MPNet similarity of captions), a learned order probe on single-piece embeddings, and readers (Llama-3.1-8B and Claude on captions; optionally a vision model on frames). Three puzzle versions: contiguous (all 12 pieces, edges can be matched), contiguous half (6 adjacent pieces) and gapped (every other piece, 5 s unseen between them, so edge matching is weakened). Metrics: Kendall τ and adjacent-pair accuracy, clustered by channel. Side results: captions written with whole-video context leak order (in-context vs isolated captions); per-video "orderability" as a measure of how predictable a video's events are.

## Kill criterion

A paper is a **direct threat** only if it has all three:

1. the same items (video clips, image sequences or procedure steps) presented both as text (captions or descriptions) and as pixels,
2. a control for visual continuity (gaps between pieces, matched distractors, or similar),
3. a comparison of how much order each modality recovers.

Two of three → **close prior work**: we cite it and state the difference. One or none → **background**.

Image and step sequences count for criterion 1: a reviewer will treat ordering a photo album or a set of instruction steps from text versus images as the same comparison as ours.

## Areas to search, in priority order

The paper names below come from memory as starting points. **Verify every one** (open its page) before listing it; drop any you cannot find.

### A. Closest threats (search these first and most thoroughly)

1. **Caption-only / language-only baselines for video understanding.** LLMs reading captions instead of frames on video QA. Starting points: "Revisiting the 'Video' in Video-Language Understanding" (atemporal probe), LLoVi, VideoAgent, Socratic Models. Question: has anyone run caption-vs-frame on a *temporal ordering* task, and controlled caption leakage?
2. **Step ordering in instructional and procedural video.** CrossTask, COIN, YouCook2, Ego4D keysteps, procedure planning, step-order verification. Question: do any compare ordering from step text vs from video on the same clips? This is the most likely overlap.
3. **Temporal benchmarks for video LLMs.** TempCompass, VITATECS, Vinoground, TemporalBench, ReXTime, MVBench ordering subtasks, and papers showing video LLMs are insensitive to frame shuffling. Question: does any include a text-only arm on captions, or vary the gap between pieces?
4. **Ordering image or step sequences from text versus images.** Starting points: Sort Story (Agrawal 2016; orders SIND image–caption sequences from text, images or both), sequencing multimodal instruction manuals (Wu et al. 2022, ACL; text vs images vs both), and RecipeQA's visual ordering task. Question: how much order did text and images each recover, and did anyone control for visual continuity? These are the most likely close prior work.

### B. Prior work for the task (cite; not threats)

5. **Self-supervised temporal order in video:** Shuffle & Learn (Misra 2016), Order Prediction Network (Lee 2017), clip order prediction / VCOP (Xu 2019), arrow of time (Pickup 2014; Wei 2018), temporal jigsaw tasks.
6. **Text ordering:** sentence ordering (entity grid, neural sentence orderers), ROCStories, script and narrative-chain knowledge (Chambers & Jurafsky; Regneri 2010), LLM procedural or script-ordering benchmarks.
7. **Spatial jigsaw self-supervision** (Noroozi 2016), for framing and naming.

### C. Our specific design choices (has anyone done these?)

8. **Gaps between pieces to defeat edge matching.** Any ordering work that removes frames between clips or controls continuity. Expect to find the device itself: jigsaw self-supervision leaves gaps between tiles, and clip order prediction samples clips with intervals, both to block low-level shortcuts. The question is whether anyone varies the gap as a graded knob, or uses it to compare modalities.
9. **Leakage in dense video captions:** captions written with whole-video context carrying information about other segments; evaluation contamination from context-aware captioners. Starting point: VIST (Huang 2016), which collected descriptions of images in isolation and in sequence (DII vs DIS), the closest precedent for our isolated vs in-context captions. Has anyone measured how much order the in-sequence descriptions carry?
10. **Predictability / orderability of a video as a per-video quantity:** event boundary detection (Kinetics-GEBD), action anticipation, video prediction in embedding space (V-JEPA), surprise or violation-of-expectation measures.

### D. Framing (cognitive science and LLM knowledge)

11. Event cognition and event segmentation (Zacks), and the "Mary's room" knowledge argument as applied to language models.
12. Temporal commonsense in language models: MC-TACO, TimeDial, script-knowledge probes.

## Method

- **Sources:** arXiv, Semantic Scholar, ACL Anthology, CVF Open Access (CVPR/ICCV/ECCV), OpenReview (NeurIPS/ICLR), Google Scholar.
- **Tools:** the repo's literature-search skills, `.agents/skills/literature_search_arxiv` and `.agents/skills/literature_search_openalex`. OpenAlex can also list the works citing a paper, for the citation chasing below.
- **Queries (starting set; extend as you learn the field's terms):**
  - "caption-only baseline video temporal reasoning"
  - "language-only baseline video ordering"
  - "temporal ordering video clips language model captions"
  - "shuffled frames video LLM sensitivity"
  - "step ordering instructional video text versus visual"
  - "clip order prediction gap between clips"
  - "dense video captioning temporal leakage context"
  - "video event predictability language prior"
  - "sorting jumbled images and captions into stories"
  - "visual storytelling descriptions in isolation versus in sequence"
  - "sequencing multimodal instructions text versus images"
  - "jigsaw puzzle gap between tiles shortcut"
- **Citation chasing:** for every paper in area A rated threat or close prior work, list who cites it (Semantic Scholar "cited by"), filtered to 2024 and later.
- **Recency:** the field moves fast. Search explicitly for 2025–2026 papers in areas A and C.
- **Timebox:** about 3–4 hours of searching. Stop early and report immediately if a direct threat (kill criterion 3/3) turns up.

## Rules

- Every listed paper must have a working link (arXiv, DOI, ACL Anthology or venue page) that you opened. A search snippet is not a source. No citations from memory.
- Record what the paper actually does, from its abstract or method section, not from its title.
- Do not spend money: no paid APIs, no LLM API calls. Web search and reading only.
- Do not download large files into the repo. Small PDFs of the closest papers may go in `docs/literature/` (follow its README table format).

## Deliverable

Write `docs/new_direction/related_work.md` with:

1. **Verdict (top, 3–5 sentences):** is the contribution intact? Name the closest paper and the one-line difference.
2. **Threat table**, one row per paper in areas A and C, sorted by closeness:

   | Paper (linked) | Year, venue | What it does | Criteria met (1/2/3) | Rating (threat / close / background) | How we differ |
   | --- | --- | --- | --- | --- | --- |

3. **Prior work list** for areas B and D: one line each, linked, with the sentence we would cite it for.
4. **Gaps confirmed:** which of our design choices (the gap as a graded continuity knob, caption leakage measurement, matched arms across modalities, orderability measure, language-vs-pixels on the same clips) have no prior work, as far as the search found.
5. **Suggested framing:** 3–5 bullets on how to position the paper given what was found, including any baselines or datasets reviewers will expect us to compare against.
6. **Search log:** queries run, databases, date, and any area searched only shallowly.

Also add the verified papers to the table in `docs/literature/README.md`.
