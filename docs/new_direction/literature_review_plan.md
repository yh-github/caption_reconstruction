# Literature review plan: is unshuffling a meaningful contribution?

Hand-off brief for a research agent. Written 2026-10-13. Read `docs/new_direction/new_plan.md` first; it describes the study this review has to position.

## The question

Has anyone already compared **how much of a video's temporal order can be recovered from language versus from pixels, on the same clips, under a control that separates visual continuity from knowledge of how events unfold?**

Ordering as a task is old and well covered. Finding ordering papers is expected and is not a threat. The review must establish where our specific comparison sits, and either confirm the gap or find the paper that already fills it.

## Our study in one paragraph (what we must position)

Each 60 s clip (WildQA, 335 clips, 40 YouTube channels) is cut into 12 pieces of 5 s. Each piece is captioned **in isolation** by Gemini (no context from other pieces). Readers get the pieces shuffled and restore the order: text readers (Llama-3.1-8B, Claude) read the captions; a frame baseline chains pieces by SigLIP 2 similarity of their edge frames (no understanding). Two puzzle versions: contiguous (all 12 pieces, edges can be matched) and gapped (every other piece, 5 s unseen between them, edges cannot be matched). Metric: Kendall τ, clustered by channel. Side results: captions written with whole-video context leak order (in-context vs isolated captions); per-video "orderability" as a measure of how predictable a video's events are.

## Kill criterion

A paper is a **direct threat** only if it has all three:

1. the same clips presented both as text (captions or descriptions) and as pixels,
2. a control for visual continuity (gaps between pieces, matched distractors, or similar),
3. a comparison of how much order each modality recovers.

Two of three → **close prior work**: we cite it and state the difference. One or none → **background**.

## Areas to search, in priority order

The paper names below come from memory as starting points. **Verify every one** (open its page) before listing it; drop any you cannot find.

### A. Closest threats (search these first and most thoroughly)

1. **Caption-only / language-only baselines for video understanding.** LLMs reading captions instead of frames on video QA. Starting points: "Revisiting the 'Video' in Video-Language Understanding" (atemporal probe), LLoVi, VideoAgent, Socratic Models. Question: has anyone run caption-vs-frame on a *temporal ordering* task, and controlled caption leakage?
2. **Step ordering in instructional and procedural video.** CrossTask, COIN, YouCook2, Ego4D keysteps, procedure planning, step-order verification. Question: do any compare ordering from step text vs from video on the same clips? This is the most likely overlap.
3. **Temporal benchmarks for video LLMs.** TempCompass, VITATECS, Vinoground, TemporalBench, ReXTime, MVBench ordering subtasks, and papers showing video LLMs are insensitive to frame shuffling. Question: does any include a text-only arm on captions, or vary the gap between pieces?

### B. Prior work for the task (cite; not threats)

4. **Self-supervised temporal order in video:** Shuffle & Learn (Misra 2016), Order Prediction Network (Lee 2017), clip order prediction / VCOP (Xu 2019), arrow of time (Pickup 2014; Wei 2018), temporal jigsaw tasks.
5. **Text ordering:** sentence ordering (entity grid, neural sentence orderers), ROCStories, SIND / "Sort Story" (Agrawal 2016), script and narrative-chain knowledge (Chambers & Jurafsky; Regneri 2010), LLM procedural or script-ordering benchmarks.
6. **Spatial jigsaw self-supervision** (Noroozi 2016), for framing and naming.

### C. Our specific design choices (has anyone done these?)

7. **Gaps between pieces to defeat edge matching.** Any ordering work that removes frames between clips or controls continuity.
8. **Leakage in dense video captions:** captions written with whole-video context carrying information about other segments; evaluation contamination from context-aware captioners.
9. **Predictability / orderability of a video as a per-video quantity:** event boundary detection (Kinetics-GEBD), action anticipation, video prediction in embedding space (V-JEPA), surprise or violation-of-expectation measures.

### D. Framing (cognitive science and LLM knowledge)

10. Event cognition and event segmentation (Zacks), and the "Mary's room" knowledge argument as applied to language models.
11. Temporal commonsense in language models: MC-TACO, TimeDial, script-knowledge probes.

## Method

- **Sources:** arXiv, Semantic Scholar, ACL Anthology, CVF Open Access (CVPR/ICCV/ECCV), OpenReview (NeurIPS/ICLR), Google Scholar.
- **Queries (starting set; extend as you learn the field's terms):**
  - "caption-only baseline video temporal reasoning"
  - "language-only baseline video ordering"
  - "temporal ordering video clips language model captions"
  - "shuffled frames video LLM sensitivity"
  - "step ordering instructional video text versus visual"
  - "clip order prediction gap between clips"
  - "dense video captioning temporal leakage context"
  - "video event predictability language prior"
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
4. **Gaps confirmed:** which of our design choices (continuity control by gaps, caption leakage measurement, orderability measure, language-vs-pixels on the same clips) have no prior work, as far as the search found.
5. **Suggested framing:** 3–5 bullets on how to position the paper given what was found, including any baselines or datasets reviewers will expect us to compare against.
6. **Search log:** queries run, databases, date, and any area searched only shallowly.

Also add the verified papers to the table in `docs/literature/README.md`.
