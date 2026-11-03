# Ethical considerations: applied (2026-11-03)

The section below is in `caption_index.tex`. The author confirmed the facts: much of the code was written by hand, with Gemini and Antigravity helping to debug, refactor and write snippets; the base code is the author's, and Claude Code reworked much of it; the author designed the study and revised the text. The author will revise the wording again.

## Section text (review version)

> **Data.** We use TempCompass's videos and questions under its licence (CC BY-NC 4.0, for academic research). Some clips show people, whom we could not ask for consent. We therefore release only our captions, under the same licence, and not the videos or frames.
>
> **AI assistance.** We wrote much of the project's code by hand. Gemini and Google Antigravity helped to debug and refactor it and wrote sample snippets. For this paper we used Claude Code, which reworked much of that code, wrote most of the code the new experiments needed, ran the experiments we specified, searched the literature, and drafted text that we revised. We designed the study, and every reported number is computed by released scripts from the logged outputs. We are responsible for all content.

## Moved out of the section

- **Compute:** now a "Compute" paragraph in the Release appendix, with the same numbers.
- **Human subjects:** dropped. The 40 hand labels are described in §4.4.

## Checklist answers (for the OpenReview form)

- **A2 (risks):** Yes, in Ethical considerations (Data): the clips show people whom we could not ask for consent, so we release captions only.
- **C1 (compute):** Yes, Appendix "Release", paragraph "Compute". The closed models' parameter counts are not public. Qwen2.5-VL has 7B parameters.
- **D (human participants):** N/A. No participants. One author labelled 40 audit items (§4.4).
- **E1 (AI assistants):** Yes, in Ethical considerations (AI assistance). Coding: Gemini and Google Antigravity helped debug and refactor hand-written code and wrote sample snippets. For this paper, Claude Code reworked much of that code and wrote most of the new experiment and analysis code. Writing: Claude Code drafted text, which we revised. Literature search: Claude Code. Every reported number is computed by released scripts from the logged outputs.

## At camera-ready

Review versions must not include acknowledgements (`formatting.md`, line 47), but the ARR call wants the AI-assistance details in the Acknowledgements. So at camera-ready, move the *AI assistance* paragraph into an Acknowledgements section. Coding details may also go in the README.

## Sources

- [ARR call for papers](https://aclrollingreview.org/cfp)
- [ARR Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/)
- [TempCompass licence](https://github.com/llyx97/TempCompass)
