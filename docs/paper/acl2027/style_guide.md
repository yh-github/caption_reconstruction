# Style guide for the caption-index paper

This guide is for the editing pass that removes AI-writing patterns from `caption_index.tex`. The pass changes wording and structure only. Every number, citation, claim and hedge stays.

## Sources

- **Checklists.** Two skills, read in full (2026-10-31):
  - [AIScientists-Dev/academic-humanizer](https://github.com/AIScientists-Dev/academic-humanizer) v0.3.3 (3.1k stars). It covers academic tells and claim–evidence discipline, and it never changes numbers.
  - [blader/humanizer](https://github.com/blader/humanizer) v3.1.0 (62.9k stars). It is built on Wikipedia's "Signs of AI writing" and is strongest on structural tells.
  - The skill at claudskills.com is dongshuyan's `academic-humanizer`, which sits inside the 763-star `compass-skills` bundle. It has no stars or reviews of its own. Only its rule E13 (mechanical rhythm across a document) is used here.
- **Voice.** The abstracts and introductions of eight papers, measured with `style_stats.py` (in the session scratchpad):
  - Two ACL 2024 best papers: Kallini et al., *Mission: Impossible Language Models*; Lesci et al., *Causal Estimation of Memorisation Profiles*.
  - Six papers we cite: LLoVi, TempCompass, IsoBench, *AI Agents That Matter*, FrugalGPT, and Xu et al. 2020.

## What the measurements show

These figures are for the abstract and introduction, before the pass.

| | Exemplars (range) | Ours |
|---|---|---|
| Numbers per 100 words | 0.6–3 (FrugalGPT 8) | 18 |
| "we"/"our" per 1,000 words | 12–36 | 3 |
| Colons per 1,000 words | 2–10 | 19 |
| Semicolons per 1,000 words | 0–7 | 8 |
| Sentences under 12 words | 3–22% | 10% |

The exemplars follow the same order:
1. They state the setting and why it matters.
2. They name the gap, citing the closest work.
3. They say what they did ("Here, we…", "We present…").
4. They state the findings in words, with one to three headline numbers. "Our core finding is that…" and "(i) … (ii) … (iii)" are typical.

Results and CIs belong in the results sections.

## Rules

1. **Abstract.** Give at most four numbers. Name the main result in words, then give its size. Leave secondary effects to the body.
2. **Introduction.** Motivate the problem before reporting results. Then use one results paragraph with two or three headline numbers, and give no CIs. The contribution list names results in words.
3. **Use "we" as the subject** of what we did and found: "We find that…", "We froze the rule…". Do not personify arms ("The captioner matters").
4. **No staged contrasts** (blader §1). This covers "X, not Y", "not X but Y", "X rather than Y", and the split form "This does not mean X. It means Y." Keep a contrast only when both halves carry information, as when a definition rules out a reading a reader would hold. For example, "usable information, not new information" stays, because it is Proposition 1's point.
5. **No colon reveals.** Avoid patterns like "The saving comes from a cheaper reader:" or "The losses are concentrated:". State the claim as a sentence. Colons stay before lists and in definitions.
6. **No one-line closers or verdict sentences** that repeat the paragraph, such as "Much of the loss thus lies in…" placed after the evidence that already shows it.
7. **Use semicolons sparingly.** Two independent claims become two sentences.
8. **Keep internal labels out of the running text** (C11b, C12g, A6, A7, A8, P1–P3). Describe the test instead. Labels stay in the appendix's prediction table, which maps them. P2 and P3 may stay where the pre-registered test is named, defined once.
9. **Make sentences vary.** Paragraphs should not all follow claim → number → CI → next number. Open some with the reason or the example. A short sentence is fine when it carries a fact.
10. **Use an example where it helps.** The paper has a few, such as the jump-rope and salt clips. They belong where the reader needs a picture.
11. **Keep calibrated hedges.** "suggests", "fits", "we cannot tell" stay (academic-humanizer Layer 3). Do not add a hedge that the source does not support.
12. **Use one term per concept** (project rule). Keep the defined terms exactly: *index*, *self-captions*, *captioner effect*, *modality effect*, *visual budget*, *retention*, *omission*, *\qbe{}*, *reader downgrade*. Do not switch to synonyms.
13. **Bold** marks run-in paragraph heads only, plus the one headline number per subsection that is already bold. Do not use it for emphasis inside a sentence.
14. **Avoid the usual word-level tells.** The academic-humanizer Layer 1–2 and blader §12 word lists apply: no em-dashes, and no Notably, Moreover, delve, crucial, robust (outside the technical sense), or showcase.

## Process

1. Snapshot the file: `cp caption_index.tex <scratchpad>/pre_pass.tex`.
2. Edit one section at a time.
3. After each section, run `.venv/bin/python scripts/paper_numbers.py docs/paper/acl2027/caption_index.tex --old <scratchpad>/pre_pass.tex`. It must exit 0. Every number that left a section must still exist elsewhere, and no number may occur more often than before.
4. Rebuild with `latexmk -pdf caption_index.tex`. Check that the main text still ends on page 8.
5. Before the final check, search for the surviving tells: `rg ', not |: [a-z]|;' caption_index.tex`.
