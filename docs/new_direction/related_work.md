# Related work for the unshuffling study (first pass)

Written 2026-10-14 from `docs/new_direction/literature_review_plan.md`. Every paper listed was checked at its arXiv abstract, its full text or its venue page; search snippets alone were not used. A first pass, about half the brief's timebox: area A and the 2025–2026 papers were searched in most depth, and the search log at the end lists what was searched only shallowly.

## 1. Verdict

**The contribution is intact. No direct threat was found** (no paper meets all three kill criteria).

- **Closest paper:** SPLICE (Ballout et al., 2025). It reorders 2–7 COIN clips from video, text, or both, with humans and VLMs. Its text is human-written step labels (4.84 words on average), so the text can carry information the pixels lack. It neither controls nor varies visual continuity, and it has no matched non-reader arms.
- **Our difference in one line:** our captions are written in isolation from exactly the frames the vision arms see. Language can therefore add no information about the order (Proposition 1 in `paper_plan.md`), and any language advantage measures how *usable* the order information is, not how much of it there is.

**One part of the comparison is already known and must not be our headline:** multimodal models order sentences much better than images.
- TempVS: best sentence ordering 86.3% against best image ordering 53.8%.
- StripCipher: GPT-4o reorders image sequences at 23.9%, against 80% for humans.
- Wu et al. 2022: text-only τ 0.82 against image-only 0.16 for models; humans also do better from text.
- SPLICE: text helps VLMs but not humans.

What is new is *why*, measured under an information control, separated into continuity and direction, and set against matched non-reader arms.

## 2. Threat table (areas A and C), closest first

Criteria:
1. the same items as text and as pixels;
2. a control for visual continuity;
3. a comparison of how much order each modality recovers.

| Paper | Year, venue | What it does | Criteria met | Rating | How we differ |
|---|---|---|---|---|---|
| [SPLICE](https://arxiv.org/abs/2509.24640) (Ballout et al.) | 2025, arXiv | Reorders 2–7 event clips from 3,381 COIN videos. Conditions: vision only, vision+text, text only; no audio. Whole-sequence accuracy: Gemini-2.0-Flash 0.51 (vision), 0.69 (vision+text), 0.53 (text); humans 0.85 / 0.83. VLMs put look-alike first and last clips next to each other (57% vs 27% at random, 2% for humans). | 1 (partly: the text is step labels, not descriptions of the clip), 2 (partly: non-step moments are left out, so there are gaps, but they are not controlled or varied), 3 | **Close** | Our captions are made from the same frames, in isolation, so the information is controlled. We vary the gap as a knob, use matched non-reader and learned arms, and separate continuity from direction. Our clips are in-the-wild, not instructional. |
| [Sequencing multimodal instructional manuals](https://aclanthology.org/2022.acl-long.310) (Wu et al.) | 2022, ACL | Orders WikiHow and RecipeQA steps from text, images, or both. Models: text-only τ 0.82 / 0.84, image-only 0.16 / 0.52, multimodal 0.85 / 0.88. Humans: image-only 0.72 / 0.86, text-only 0.89 / 0.93, both 0.94 / 0.95. | 1 (partly: the text is the instruction, not a description of the image), 3 | **Close** | Their text and images are complementary parts of a manual, not two views of the same content. They have no continuity control and no video. |
| [Sort Story](https://arxiv.org/abs/1606.07493) (Agrawal et al.) | 2016, EMNLP | Sorts aligned image–caption pairs from SIND stories with unary (position) and pairwise (order) predictors. Text and image features are complementary. | 1, 3 (partly) | **Close** | Its captions are story text written in sequence, which our leak result says carries order. It uses images, not video, and has no continuity control. Its unary + pairwise design is the ancestor of our probe + pairwise arms; cite it for that. |
| [TempVS / "Burn After Reading"](https://arxiv.org/abs/2506.10415) (Song, Du, Paperno, Gatt) | 2025, ACL Findings | Image sequences from FlintstonesSV, PororoSV, VIST and VWP. Three tests: event-relation inference, sentence ordering given images, image ordering given text. Best: sentence 86.3%, image 53.8%; chain of thought helps ordering. Image sequences that are too similar (CLIP) are filtered out. | 1 (partly), 2 (partly: similarity filtering), 3 (partly) | **Close / background** | It never swaps the modality of the same input. Each test mixes modalities, so it cannot say how much order each modality carries alone. |
| [VECTOR / MECOT](https://arxiv.org/abs/2512.08979) (Ahn et al.) | 2025, arXiv (cited as WACV 2026) | Video LMMs keep more than 82% of their accuracy on 9 benchmarks with shuffled frames, so they rely on scenario priors. Proposes an event-order benchmark. Fine-tuning on event-by-event descriptions plus chain of thought helps. | 3 (partly) | Background | It is QA, not ordering. That descriptions help is consistent with our "language makes order usable" hypothesis, and we test it under an information control. |
| [Temporal Reasoning Transfer from Text to Video (T3)](https://arxiv.org/abs/2410.06166) (Li et al.) | 2024, arXiv | Probes on video LLM representations decode temporal relations almost perfectly. The bottleneck is the LLM's temporal reasoning, and text-only temporal training transfers to video. | none directly | Background, **important for framing** | The same "the information is there, using it is the problem" claim, about training. We test it at inference, with captions as the representation and matched readers. |
| [Vamos](https://arxiv.org/abs/2311.13627) (Wang et al.) | 2023, arXiv | An LLM reasons over per-frame captions. Shuffling 12 captions costs only 2 points on EgoSchema (41.2% → 39.2%), which the authors read as "LLMs may auto-correct the order". | 1 (partly) | Background | An anecdote of LLMs restoring order from captions. We measure it directly. |
| [TVBench / "Lost in Time"](https://arxiv.org/abs/2410.07752) | 2024 arXiv, BMVC 2025 | Benchmarks solvable from a single frame, the question text or world knowledge. Text-only models are at chance on TVBench; shuffling barely hurts MVBench. | (QA) | Background | Shuffle and text-only *controls* in QA, not a modality comparison of ordering. |
| [VBenchComp](https://arxiv.org/abs/2505.14321) (Feng et al.) | 2025, arXiv | Splits benchmark questions into LLM-answerable, shuffle-invariant ("semantic") and temporal. | (QA) | Background | Same idea of separating priors from order, for QA. |
| [StripCipher](https://arxiv.org/abs/2502.13925) (Wang et al.) | 2025, arXiv | Image-sequence comprehension and reordering. GPT-4o reorders at 23.93%, 56 points below humans. | 3 (vision only) | Background | Vision only. |
| [TimeCatch](https://arxiv.org/abs/2608.23474) (Hradil, Sánchez Villegas) | 2026, arXiv | Swaps consecutive frames; VLMs are near chance at detecting it, humans near ceiling. | (vision only) | Background | Supports "VLMs see but do not order". |
| [Order Matters: LVLMs as judges](https://arxiv.org/abs/2608.10908) (Ianaro et al.) | 2026, arXiv | LVLMs fail at pairwise temporal-order discrimination of image sequences, with primacy and recency biases. | (vision only) | Background | Position bias: our readers get several shuffles per puzzle. |
| [Reversal-drop](https://arxiv.org/abs/2607.12304) (Rahman) | 2026, arXiv | Separates whether a video VLM reads order from the pixels or from positional encoding, by reversing the frames. | (vision only) | Background | Related to our direction probe (forward vs reversed). |
| [VCOP](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.html) (Xu et al.) | 2019, CVPR | Clip order prediction with a fixed 8-frame interval between clips, "required to avoid trivial solutions": with even 1 frame of overlap, "a simple comparison of frames pixels can solve the task". | 2 (fixed gap) | **Prior art for the gap device** | The gap is fixed, used for training and vision only. We vary it and compare modalities. |
| [OmniJigsaw](https://arxiv.org/abs/2604.08209) (Jia et al.) | 2026, arXiv | Reordering of shuffled audio-visual clips as an RL proxy task. Modality masking per clip; a "bi-modal shortcut"; filters out puzzles without clear temporal structure. | 2 (partly) | Background | Training, not measurement. Its filter for videos without temporal structure is the closest thing to our orderability, used there as a data filter. |
| [VideoSSR](https://arxiv.org/abs/2511.06281) (He et al.) | 2025, arXiv | "Temporal Jigsaw": a video is split into equal segments that MLLMs must reorder (VIUBench); used for RL. | (vision only) | Background | Same puzzle, vision only, as training. |
| [VIST](https://arxiv.org/abs/1604.03968) (Huang et al.) | 2016, NAACL | Three text tiers for the same photo sequences: descriptions in isolation (DII), descriptions in sequence (DIS), stories (SIS). | 1 (partly) | **Precedent for the leak design** | We found no measurement of how much order DIS carries beyond DII (searched shallowly). Our isolated vs. in-context captions measure exactly that, on video. |
| [MATEO](https://arxiv.org/abs/2602.14589) (Roccabruna et al.) | 2026, arXiv | Recipe steps with images and step-dependency graphs (DAGs) for LVLM temporal planning. | 1 (partly), 3 (partly) | Background | Procedural planning, not ordering of the same content. |

## 3. Prior work for areas B and D (cite; not threats)

**Temporal order in video (self-supervised and arrow of time)**
- [Shuffle and Learn](https://arxiv.org/abs/1603.08561) (Misra et al., ECCV 2016): verifies whether frames are in temporal order; the first order-as-supervision paper.
- [Odd-One-Out networks](https://arxiv.org/abs/1611.06646) (Fernando et al., CVPR 2017): finds the wrongly ordered subsequence.
- [Sorting sequences / OPN](https://arxiv.org/abs/1708.01246) (Lee et al., ICCV 2017): sorts shuffled frames with pairwise features; the pairwise-to-order design we reuse.
- [Seeing the Arrow of Time](https://openaccess.thecvf.com/content_cvpr_2014/html/Pickup_Seeing_the_Arrow_2014_CVPR_paper.html) (Pickup et al., CVPR 2014): forward vs backward from motion asymmetries.
- [Learning and Using the Arrow of Time](https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html) (Wei et al., CVPR 2018): cinematographic conventions are artificial direction cues, as our title cards are.
- [Seeing the Arrow of Time in LMMs](https://arxiv.org/abs/2506.03340) (Xue, Luo, Grauman, 2025): LMMs barely use temporal direction; ArrowRL and AoTBench.

**Video-language and temporal benchmarks**
- [ATP](https://arxiv.org/abs/2206.01720) (Buch et al., CVPR 2022) and [single-frame bias](https://arxiv.org/abs/2206.03428) (Lei et al., 2022): much of video-language is solvable from one frame.
- [Test of Time](https://arxiv.org/abs/2301.02074) (Bagad et al., CVPR 2023): video-language models fail before/after consistency.
- [Paxion / ActionBench](https://arxiv.org/abs/2305.10683) (Wang et al., NeurIPS 2023): video reversal probe.
- [TempCompass](https://arxiv.org/abs/2403.00476), [VITATECS](https://arxiv.org/abs/2311.17404), [Vinoground](https://arxiv.org/abs/2410.02763), [TemporalBench](https://arxiv.org/abs/2410.10818): temporal perception benchmarks; models are weak.
- [Mementos](https://arxiv.org/abs/2401.10529): reasoning over image sequences.

**Language as the interface to video**
- [Socratic Models](https://arxiv.org/abs/2204.00598) (Zeng et al., 2022): compose models through language.
- [LLoVi](https://arxiv.org/abs/2312.17235) (Zhang et al., 2023): clip captioner plus LLM for long video QA.
- [VLEP](https://arxiv.org/abs/2010.07999) (Lei et al., EMNLP 2020): future-event prediction from video and dialogue; most of the signal is in the text.

**Text ordering and scripts**
- [Narrative event chains](https://aclanthology.org/P08-1090/) (Chambers & Jurafsky, ACL 2008): script knowledge learned from text.
- [MC-TACO](https://arxiv.org/abs/1909.03065) (Zhou et al., EMNLP 2019): temporal commonsense, including order.
- [Arrows of Time for LLMs](https://arxiv.org/abs/2401.17505) (Papadopoulos et al., ICML 2024): forward vs backward perplexity asymmetry in language; a language-side arrow of time.
- [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207) (Gurnee & Tegmark, 2023): linear time representations (dates, not event order).

**Event cognition and predictability**
- [Event perception: a mind/brain perspective](https://pubmed.ncbi.nlm.nih.gov/17338600/) (Zacks et al., Psych. Bull. 2007): event boundaries from prediction error; perception uses sensory cues *and* script knowledge, the two sources our decomposition separates.
- [Kinetics-GEBD](https://arxiv.org/abs/2101.10511) (Shou et al., ICCV 2021): generic event boundaries.
- [Intuitive physics from self-supervised video](https://arxiv.org/abs/2502.11831) (Garrido et al., 2025): V-JEPA-style models show violation-of-expectation; a vision-side predictability measure.

**Theory**
- [Usable (V-)information](https://arxiv.org/abs/2002.10689) (Xu et al., ICLR 2020): information under computational constraints; unlike Shannon information it can be *created* by computation. This is the backbone of our framing (Proposition 1 and its corollary).
- [SerialRank / spectral seriation](https://papers.nips.cc/paper_files/paper/2014/file/f0e52b27a7a5d6a1a87373dffa53dbe5-Paper.pdf) (Fogel, d'Aspremont, Vojnovic, NeurIPS 2014; JMLR 2016) and Atkins et al. 1998 (cited there): ordering from a similarity matrix, recovered only up to reversal. This is the basis of our structure/direction split (Proposition 2).
- [Platonic representation hypothesis](https://proceedings.mlr.press/v235/huh24a.html) (Huh et al., ICML 2024): vision and language representations converge, which predicts small modality differences at scale. Our direction result is a test case.

## 4. Gaps confirmed (as far as this search found)

| Design choice | Prior work found | Status |
|---|---|---|
| Captions written in isolation from **exactly** the frames the vision arms see (an information control) | None. SPLICE, Wu, Sort Story and TempVS use human text written with the sequence or the task in view. | **Novel** |
| The gap between pieces as a **graded** continuity knob, compared across modalities | Fixed gaps in self-supervision (VCOP, jigsaw); trimmed clip edges (OmniJigsaw) | **Novel as a knob and as a cross-modal control** |
| Matched arms in both modalities (similarity chaining, learned probe, same reader with frames vs captions) | Only on the reader side (SPLICE, TempVS, Wu) | **Novel** |
| Splitting order recovery into structure (continuity) and direction | Arrow of time (vision only); seriation theory (no modality comparison) | **Novel in this setting** |
| Leak of whole-video captioning, measured by readers | VIST collected DII vs DIS but no one measured order in them (searched shallowly) | **Probably novel**; check in the second pass |
| Per-video orderability (language vs vision) | OmniJigsaw filters "no clear temporal structure" | **Novel as a measure** |
| Within-model verbalization effect on ordering (the same model, frames vs its own captions) | Chain of thought helps ordering (TempVS, MECOT); text helps (SPLICE) | **Novel under an information control** |

## 5. Suggested framing

- **Do not sell "language orders better than vision".** TempVS, StripCipher, SPLICE and Wu already show it for models. Sell the controlled explanation: with no extra information in the captions, the advantage is *usable* information (Xu et al. 2020), and it lives in **direction**, not in continuity.
- **Lead with the information control and the theory.** Isolated captioning makes caption order information a garbling of frame order information (Proposition 1). Symmetric similarity can never recover direction (Proposition 2). Continuity decays with the gap at a measurable rate (Proposition 3). Each gives a prediction we test.
- **Expected baselines and comparisons from reviewers:**
  - SPLICE-style direct VLM ordering of the clips (our Gemini and Claude frame readers);
  - a text-only LLM on captions (ours);
  - human performance on a subset (to add);
  - shuffle sensitivity and position bias (several shuffles per puzzle);
  - an open-weights model for reproducibility (Llama, and Qwen-VL for frames).
- **Generality:** reviewers will ask whether this holds beyond one in-the-wild dataset. A script-rich set (instructional, e.g. COIN as in SPLICE) is the natural second dataset and gives the dose-response test of "language helps where scripts exist".
- **The leak result is a standalone warning** for caption-based video benchmarks (LLoVi- and Socratic-style pipelines): captions written with whole-video context carry order.

## 6. Search log (2026-10-14)

- **Tools:** WebSearch (standard and extended), arXiv API (export.arxiv.org, 1 request per 3 s, through a small client in the session scratchpad), Semantic Scholar citations API, full-text fetches of SPLICE, TempVS, Wu et al., Vamos and VCOP (PDF).
- **Queries (main ones):**
  - caption vs frames temporal ordering
  - shuffled clips with captions vs frames
  - Sort Story
  - multimodal instruction sequencing results
  - TVBench / VBenchComp / text-only baselines
  - language vs vision ordering with the arrow of time
  - VIST DII vs DIS
  - verbalize or caption-then-order vs direct
  - temporal jigsaw with LLM captions
  - event predictability language vs vision
  - caption leakage in dense captioning
  - step ordering text vs video (COIN, CrossTask, YouCook2)
  - LLM sentence ordering (ROCStories)
  - seriation / SerialRank
  - V-information
  - Platonic representation
  - self-supervised order (Shuffle & Learn, O3N, OPN, VCOP)
  - arrow of time (Pickup, Wei)
  - Zacks; Chambers & Jurafsky
  - Socratic vs end-to-end on temporal order
  - 2026 caption-vs-frame controlled comparisons
- **Citation chase (Semantic Scholar, 2024 and later):**
  - Wu et al. 2022: 32 citing papers; checked MATEO, CaT-Bench and CompAct by title or abstract.
  - Sort Story: 62 citing papers; checked WeGO (2408.00305).
  - SPLICE: 2 citing papers; checked 2605.28779.
  - TempVS: 10 citing papers; checked TimeCatch and TPRU.

  None is a threat.
- **Searched only shallowly (for the second pass):**
  - VIST DIS order content;
  - procedural step-order verification in video (e.g. [Learning and Verification of Task Structure](https://arxiv.org/abs/2303.13519), seen only in a search listing);
  - sentence ordering with current LLMs;
  - the cognitive science of language vs. perception for event order (e.g. verbal labels and event memory);
  - human studies ordering from text vs. images beyond Wu et al.
- **Also not done:** Regneri et al. 2010 (script knowledge) could not be verified, so it is not listed.
