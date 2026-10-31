#!/bin/bash
# C13 and C14 (research log), resumable: stop at any time and run again; every step skips what is done (caption
# files are written atomically; a partial reader row is skipped and its item rerun).
#   scripts/index_qa/run_c13_c14.sh check   # 3 question-aware captions per captioner and 1 Sonnet read; then stop
#   scripts/index_qa/run_c13_c14.sh run     # captions (3 shards per captioner), with Haiku and Sonnet reading as
#                                           # captions appear; then the analyses
#   scripts/index_qa/run_c13_c14.sh stop
# Quota stop rules: captions --max-5h 0.7 --max-7d 0.5, reads --max-5h 0.8; after a stop, wait for the 5-hour reset.
# Logs: logs/orchestrator_c13.log, logs/c13_captions_{haiku,sonnet}_K.log, logs/c13_read_{haiku,sonnet}.log.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
QW=scripts/index_qa/quota_wait.py
CAP="scripts/index_qa/self_caption.py --set confirm --question"
READ="scripts/index_qa/reader.py run --set confirm --skip-missing"
ARMS_haiku="selfcaptions_rerun selfqcaptions sonnetqcaptions"
ARMS_sonnet="selfcaptions framecaptions frameclipcaptions sonnetqcaptions"
mkdir -p logs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a logs/orchestrator_c13.log; }

left_captions() { $PY $CAP --model "$1" --dry-run | awk 'NR==1{print $1}'; }
arms() { if [ "$1" = haiku ]; then echo "$ARMS_haiku"; else echo "$ARMS_sonnet"; fi; }
left_reads() { $PY $READ --model "$1" --arms $(arms "$1") --dry-run | grep -o 'running [0-9]*' | awk '{print $2}'; }

captions() {  # model
  for i in $(seq 1 20); do
    n=$(left_captions "$1"); log "C13 captions $1: $n windows left"
    [ "$n" = "0" ] && { touch "logs/c13_captions_$1.done"; log "C13 captions $1 done"; return 0; }
    for k in 0 1 2; do
      $PY $CAP --model "$1" --shard "$k/3" --max-5h 0.7 --max-7d 0.5 >> "logs/c13_captions_$1_$k.log" 2>&1 &
    done
    wait
    [ "$(left_captions "$1")" = "0" ] && { touch "logs/c13_captions_$1.done"; log "C13 captions $1 done"; return 0; }
    $PY $QW 0.65 >> logs/orchestrator_c13.log
  done
  log "C13 captions $1: gave up after 20 attempts"; touch "logs/c13_captions_$1.failed"; return 1
}

reads() {  # model: read every available item, wait for more captions, until all captions exist and all are read
  for i in $(seq 1 500); do
    stopped=$($PY $READ --model "$1" --arms $(arms "$1") --max-5h 0.8 2>&1 | grep -v '^skipped, ' \
              | tee -a "logs/c13_read_$1.log" | grep -c "quota use reached")
    n=$(left_reads "$1")
    if [ -e logs/c13_captions_haiku.done ] && [ -e logs/c13_captions_sonnet.done ] && [ "$n" = "0" ]; then
      log "reads $1: done"; return 0
    fi
    if [ -e logs/c13_captions_haiku.failed ] || [ -e logs/c13_captions_sonnet.failed ]; then
      log "reads $1: captions failed; stopping"; return 1
    fi
    if [ "$stopped" != "0" ]; then $PY $QW 0.75 >> logs/orchestrator_c13.log; else sleep 60; fi
  done
  log "reads $1: gave up after 500 rounds"; return 1
}

case "$1" in
  check)
    $PY $CAP --model haiku --limit 3 --max-5h 0.7 --max-7d 0.5 >> logs/c13_captions_haiku_check.log 2>&1
    $PY $CAP --model sonnet --limit 3 --max-5h 0.7 --max-7d 0.5 >> logs/c13_captions_sonnet_check.log 2>&1
    $PY $READ --model sonnet --arms selfcaptions --limit 1 --max-5h 0.8 >> logs/c13_read_sonnet.log 2>&1
    log "check done" ;;
  run)
    rm -f logs/c13_captions_{haiku,sonnet}.{done,failed}
    log "run: captions, with reads following"
    captions haiku &
    captions sonnet &
    reads haiku &
    reads sonnet &
    wait
    if [ -e logs/c13_captions_haiku.done ] && [ -e logs/c13_captions_sonnet.done ] \
       && [ "$(left_reads haiku)" = "0" ] && [ "$(left_reads sonnet)" = "0" ]; then
      log "analyses"
      for c in c13 c14; do
        $PY scripts/index_qa/followups.py "$c" > "results/index_qa/reports/$c.txt" 2>&1
      done
      log "done"
    else
      log "stopped before the end; run again to resume"
    fi ;;
  stop)
    pkill -f "run_c13_c14.sh run"
    pkill -f "self_caption.py --set confirm --question"
    pkill -f "reader.py run --set confirm --skip-missing"
    pkill -f "scripts/index_qa/quota_wait.py"
    log "stopped by hand" ;;
  *) echo "usage: $0 check|run|stop"; exit 2 ;;
esac
