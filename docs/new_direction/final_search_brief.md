# Caption index: final literature search brief

2026-10-19

## Purpose and what the study claims

The search must establish that no paper published or posted up to submission already makes our three claims on video. If one does, it must tell us which claim is lost. It replaces the quick novelty pass of 2026-10-18 (N1 in `docs/new_direction/research_log.md`: 7 web searches, not a full review).

**The study in two sentences.** A strong captioner (Gemini 3.8 Flash) writes question-blind captions of each 5 s window of a short video, once. A reader (Claude Haiku or Sonnet) then answers multiple-choice TempCompass questions from those captions instead of the frames, and we compare accuracy at matched query cost (input tokens).

**How the effect is split.** With Haiku as the reader, the total caption effect (Gemini video captions − 5 frames) splits exactly into three terms:

| Term | Comparison | What differs |
| --- | --- | --- |
| Coverage | Gemini video captions − Gemini frame captions | what the captioner saw (the video vs. the reader's 5 small frames) |
| Captioner | Gemini frame captions − Haiku self-captions | who wrote the caption, from identical input |
| Format | Haiku self-captions − 5 frames | the same model's own words vs. the pixels it saw |

**Pre-registered hypotheses (C1, 297 questions, held-out videos).**

- **P1, aspect:** captions lose continuous quantities (direction, speed) more than discrete content (action, order, attribute change). Sonnet reader, predicted contrast −0.20.
- **P2, format:** no format advantage. Self-captions − 5 frames, predicted −0.08. This is the opposite of IsoBench's finding for exact text renderings.
- **P3, captioner:** a stronger captioner adds usable information. Gemini frame captions − self-captions, predicted +0.06.

Secondary claims the search should also cover: break-even counts (how many questions repay the captioning cost), and a cheap reader on captions vs. a strong reader on frames.

## Known closest work

None of these makes our claims, so the baseline is clear. The search looks for newer or missed papers that close the gaps in the last column. Full details are in `docs/new_direction/related_work_caption_index.md` (third pass, 2026-10-15) and N1 in the research log.

| Paper | What it does | What it leaves open for us |
| --- | --- | --- |
| [CapMem](https://arxiv.org/abs/2609.17688) (Liang et al., EMNLP 2026) | Each model captions its own video into a memory and answers from it. Token cost per question; 75 egocentric videos, 33.7 h. Captions win above 20 min and lose at 20 min or less. | Long video, one frame cap, self-captioning only (no separate captioner), no coverage/captioner/format split. |
| [CFD](https://arxiv.org/abs/2609.11899) (Cai et al., EMNLP 2026) | Caption once, then a router fetches frames for "perceptual" questions. Amortized time formula; hour-long video. | No token counts, no matched budget, coverage confounded with format. |
| [IsoBench](https://arxiv.org/abs/2404.01266) (COLM 2024) | The same problems as text and as images; models do 15–29 points better on text. | Images, and text that exactly renders the problem; not lossy, question-blind video captions. |
| [Prism](https://arxiv.org/abs/2406.14544) (NeurIPS 2024) | A VLM describes an image, an LLM reasons over the description; a 2B captioner + GPT-3.5 matches much larger VLMs. | Images; no self-caption control, so captioner and format are not separated. |
| [LLoVi](https://arxiv.org/abs/2312.17235) (2023) | Captions short clips, an LLM reasons over the text; captioner choice matters (oracle captions +10.8 on EgoSchema). | Long video; no matched cost, no decomposition. |
| [VidCtx](https://arxiv.org/abs/2412.17415) (2024) | Question-aware captions beat generic ones on NExT-QA by 1–2 points. | Prices task-agnosticism only; no frame arm at matched cost. |
| [Sprout](https://arxiv.org/abs/2609.35497) (2026) | Builds a text memory lazily, because eager indexing costs more than answering a few questions. | Eager vs. lazy; no information control. |
| [LoHi](https://arxiv.org/abs/2610.04318) (NeurIPS 2026) | Dense low-resolution frames beat sparse full-resolution frames at matched visual-token budget. | Visual side only; relevant to whether our frame arms got a fair allocation. |
| [TokenSwap](https://arxiv.org/abs/2607.28640) (2026) | The same content as images instead of text loses 4–47% across 42 MLLMs. | Format effect for exact content, not captions; closest risk to P2's framing. |
| [Missing-modality gap](https://arxiv.org/abs/2605.12517) (ICLR 2026 workshop) | A VLM given a description instead of the image loses accuracy and calibration beyond the missing information. | Images; close to P2 and P3 in spirit. |

## Search questions

Each question is a claim we want to make. For each one, find any paper that already makes it, and judge how close it comes (see the next sections). Q1 and Q2 matter most: they are the paper's contribution.

1. **Q1. The decomposition.** Has anyone split the benefit of captions over frames into what the captioner saw, who wrote the caption, and words vs. pixels, on video?
   - A scoop is any video QA paper with a self-caption control (the reader captions the same frames itself) or a captioner given exactly the reader's frames.
   - Also check images: a self-caption control on image QA would weaken our novelty claim but not remove it.
2. **Q2. Question-blind captions vs. frames at matched query cost, on short video.** Same reader on both sides, cost counted in input tokens, clips of about 1 minute or less.
   - A scoop is a TempCompass, MVBench, Perception Test or NExT-QA paper with a caption-only arm and a frame arm at a stated equal token budget.
   - Long-video results (CapMem, CFD) don't count as a scoop, but a short-video split inside such a paper does.
3. **Q3. Which aspects survive the caption (P1).** Has anyone shown that captions keep action, order and state change but lose speed and direction?
   - Look for per-aspect or per-skill caption-only results on temporal benchmarks: TempCompass, MVBench, VITATECS, TemporalBench, FAVOR-Bench, MotionBench.
   - Also the motion-captioning literature: papers reporting that captioners miss motion direction or speed.
   - The original TempCompass paper reports no caption-input baseline (checked).
4. **Q4. No format advantage for lossy text (P2).** Is the IsoBench or TokenSwap "text beats pixels" result known to reverse when the text is a lossy description written without the question?
   - Look for describe-then-answer or verbalization studies where the same model describes, then answers, and does worse than answering from the image.
5. **Q5. Captioner knowledge (P3).** Is "a stronger captioner adds usable information even from identical input" shown with a controlled design, on images or video?
   - Prism and LLoVi vary the captioner, but not with identical input to a self-caption arm.
6. **Q6. Break-even and reader downgrade.** Has anyone solved for the number of questions at which captioning pays for itself, counting a cheaper reader, with measured numbers?
7. **Q7. New closest work.** Anything after 2026-10-15 on caption memories, video indexes, or text vs. visual tokens. The area produced 8 close papers between 2026-09-09 and 2026-10-13.

## Queries and sources

Search from 2023-01 to the day of the search, newest first. The previous passes covered work up to 2026-10-15; anything after that is new by definition.

**Sources, in order:**

1. arXiv listing search (cs.CV, cs.CL, cs.AI, cs.MM), sorted by submission date.
2. Semantic Scholar: citing papers of the anchors below, plus keyword search.
3. Google Scholar, for venue versions and workshop papers that arXiv search misses.
4. Proceedings and OpenReview: ACL / EMNLP / NAACL 2025–2026, CVPR / ICCV / ECCV 2025–2026, NeurIPS 2025–2026, ICLR 2026–2027 submissions, COLM 2025–2026, TMLR.

**Citation chase.** List every paper citing these anchors, and check titles and abstracts:

- TempCompass (Liu et al. 2024): the most likely place for a caption-only arm on our benchmark.
- IsoBench, Prism, CapMem, CFD, LLoVi.
- CaptionQA and CapQuiz (caption utility measured by QA).

**Query terms** (combine one from each group where it helps):

| Group | Terms |
| --- | --- |
| Setting | caption memory, caption index, video index, caption once, amortized video understanding, text memory for video, describe then answer, verbalize then reason |
| Comparison | captions vs frames, text vs visual tokens, matched token budget, equal budget, cost-accuracy trade-off, Pareto frontier, iso-token |
| Decomposition | self-captioning, self-description, captioner effect, modality gap, format effect, perception vs reasoning, decoupled perception, information control |
| Content | motion direction, speed, temporal reasoning, fine-grained motion, what captions miss, caption hallucination of motion |
| Benchmarks | TempCompass, MVBench, Perception Test, NExT-QA, TemporalBench, VITATECS, MotionBench, FAVOR-Bench, Video-MME short |

**Example queries:** "TempCompass caption-only baseline"; "self-caption control video question answering"; "captions versus frames matched token budget short video"; "describe then answer worse than image"; "video captions lose motion direction speed"; "caption memory short video cost per query".

## How to judge a hit

Rate every relevant paper on one of four levels, per search question (Q1–Q7). One paper can score differently on different questions.

| Level | Meaning | Action |
| --- | --- | --- |
| Scoop | Makes the same claim on video with a comparable control (e.g. a self-caption arm, or matched-cost captions vs. frames on short clips). | Flag to the team the same day; we reframe that claim. |
| Close | Same question on another domain (images, long video) or without the control. | Cite and state the difference in one sentence. |
| Related | Shares a component (caption-then-reason, caption QA metric, token budgets). | Cite if space allows. |
| Not relevant | Matched a keyword only. | Log the title in the search log, nothing else. |

**Read the full text** of every Scoop or Close candidate: methods, arms and tables, not just the abstract. Abstracts in this area routinely overstate controls. Example: CFD's headline "visual–textual duality" compares 1-fps captions with up to 768 frames on 53-minute videos, so coverage and format are confounded.

**Record per paper:** title and link; venue and date; level per question; the setting (domain, video length, benchmark); the arms and whether the reader is the same on both sides; whether cost is matched and how; the key number; one sentence on how it differs from us; whether you read the full text or the abstract only.

## Deliverable and stop rule

The output is a verdict per search question (Q1–Q7: still open, weakened, or scooped) with the papers behind it. It goes into the repo, so the paper's related-work section can be written from it.

- **Where:** a new section "10. Fourth pass" at the end of `docs/new_direction/related_work_caption_index.md`, with the paper table (fields from the previous section) and a search log in the style of §9 there: tools, queries, citation chases, what was not done.
- **Summary:** one entry in `docs/new_direction/research_log.md` (N2), with the verdict per question and any change it forces on the paper's claims.
- **Stop rule:** stop when every question has had its queries and citation chases, and the last 20 checked titles turned up nothing above Related. A Scoop on Q1 or Q2 stops the search early: report it at once.
- **Timing:** run it once now, and repeat Q7 (new work) in the week before submission, because this area moves weekly.

Open question: the submission venue and deadline are not fixed yet. They set the date of the repeat pass.
