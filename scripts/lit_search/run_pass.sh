#!/usr/bin/env bash
# Re-run the literature pass of docs/new_direction/related_work_paper.md §6 (no model calls).
# Usage: scripts/lit_search/run_pass.sh SINCE OUTDIR
#   SINCE   YYYY-MM-DD; arXiv entries and citing papers before it are dropped.
#   OUTDIR  where the raw listings go (use the session scratchpad, not the repo).
# Then screen OUTDIR/*.txt by title, fetch abstracts (arxiv_abstracts.py, openreview_search.py abs)
# and full texts (fetch_fulltext.py) for candidates, and write the result into related_work_paper.md.
set -euo pipefail
SINCE=${1:?SINCE (YYYY-MM-DD)}
OUT=${2:?OUTDIR}
DIR=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
mkdir -p "$OUT"

echo "arXiv: recent work since $SINCE"
"$PY" -I "$DIR/arxiv_query.py" "$SINCE" 200 \
  'abs:caption AND abs:video AND abs:frames' \
  'abs:caption AND abs:memory AND abs:video' \
  'abs:captions AND abs:"visual tokens"' \
  'abs:caption AND abs:cost AND abs:video' \
  'abs:"text-only" AND abs:video AND abs:question' \
  'abs:"own captions" OR abs:"self-generated captions" OR abs:"self-captioning" OR abs:"self-caption"' \
  'abs:"describe then answer" OR abs:"describe-then-answer" OR abs:"verbalize" AND abs:video' \
  'abs:captioner AND abs:reader' \
  'abs:"decoupled perception" OR abs:"perception and reasoning" AND abs:video AND abs:caption' \
  'abs:video AND abs:caption' \
  'abs:video AND abs:"question answering" AND abs:text' \
  'abs:"video understanding" AND abs:efficient' \
  'abs:video AND abs:"captions"' \
  'abs:video AND abs:routing AND abs:modality' \
  > "$OUT/arxiv_recent.txt"

echo "arXiv: claim-specific queries since $SINCE"
"$PY" -I "$DIR/arxiv_query.py" "$SINCE" 60 \
  'abs:"question-aware" AND abs:caption AND abs:video' \
  'abs:"query-aware" AND abs:caption AND abs:video' \
  'abs:"question-agnostic" AND abs:caption' \
  'abs:"task-agnostic" AND abs:caption AND abs:video' \
  'abs:"generic captions" AND abs:question' \
  'abs:caption AND abs:video AND abs:reversed' \
  'abs:caption AND abs:"speed" AND abs:video AND abs:"question answering"' \
  'abs:"text or image" AND abs:routing' \
  'abs:route AND abs:"text-only" AND abs:"vision-language"' \
  'abs:"modality selection" AND abs:question' \
  'abs:TempCompass AND abs:caption' \
  'abs:contamination AND abs:video AND abs:benchmark' \
  'abs:"question-only" AND abs:video' \
  'abs:"weak-to-strong" AND abs:caption' \
  'abs:"same frames" AND abs:caption' \
  'abs:reader AND abs:compression AND abs:upgrade' \
  > "$OUT/arxiv_claims.txt"

echo "OpenReview (ICLR 2027 unless OR_GROUP is set)"
"$PY" -I "$DIR/openreview_search.py" search 25 \
  'captions frames same model video' 'self-generated captions video question answering' 'own captions' \
  'captioner reader decomposition' 'describe then answer video' 'text-only reader captions video' \
  'caption utility question answering video' 'question-blind captions' 'query-agnostic memory video' \
  'question-aware captions' 'text versus visual tokens video' 'modality gap text image same content' \
  'verbalization visual reasoning loss' 'cost per query video understanding amortized' 'break-even captioning' \
  'cheaper model routing multimodal' 'routing text image modality per query' 'TempCompass' \
  'reversed video arrow of time' 'contamination video benchmark question-only' 'blind baseline video question answering' \
  'caption once' 'caption memory short video' 'weak reader strong writer memory' 'reader upgrade captions' \
  'stronger captioner weaker reader' 'information control captions frames' 'data processing inequality captions' \
  'pre-registered video evaluation' 'question-only baseline multiple choice video' 'minimal pairs video representation' \
  'speed comparison video captions' 'Claude Haiku Sonnet video' 'self-description image question answering same model' \
  'caption then answer cost tokens' 'text index video retrieval cost' 'when do captions suffice' 'video caption omission question' \
  > "$OUT/openreview.txt"

echo "Semantic Scholar citation chases since $SINCE"
# CapMem, CFD, Caption This, REEL, Sprout, CapQuiz, TempCompass, IsoBench, Prism, CaptionQA, LLoVi, GEASS,
# Compression Is Not Evaluation-Neutral
"$PY" -I "$DIR/s2_citations.py" "$SINCE" \
  2609.17688 2609.11899 2505.21538 2610.38158 2609.35497 2609.09973 2403.00476 \
  2404.01266 2406.14544 2511.21025 2312.17235 2605.01733 2606.21807 \
  > "$OUT/citations.txt" 2> "$OUT/citations.err"

echo "done: $OUT"; wc -l "$OUT"/*.txt
