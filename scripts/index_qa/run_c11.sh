#!/bin/bash
# C11 (research log), resumable: stop it at any time and run it again later; every step skips what is done (caption
# files are written atomically; a partial reader row is skipped and its item rerun).
#   scripts/index_qa/run_c11.sh check   # the 3-call check: 3 videos captioned, then stop for inspection
#   scripts/index_qa/run_c11.sh run     # captions, with Haiku and Sonnet reading each video as soon as its caption
#                                       # exists; then the analyses
#   scripts/index_qa/run_c11.sh stop    # stop everything (calls in flight are lost and redone on resume)
# Quota stop rules: captions --max-5h 0.7 --max-7d 0.5, reads --max-5h 0.8; after a stop, wait for the 5-hour reset.
# Logs: logs/orchestrator_c11.log, logs/c11_captions.log, logs/c11_read_{haiku,sonnet}.log.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
QW=scripts/index_qa/quota_wait.py
CAP="scripts/index_qa/self_caption.py --set confirm --model sonnet --whole --aspects"
READ="scripts/index_qa/reader.py run --set confirm --arms sonnetaspects --skip-missing"
DONE=logs/c11_captions.done
FAILED=logs/c11_captions.failed
mkdir -p logs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a logs/orchestrator_c11.log; }

left_captions() { $PY $CAP --dry-run | awk 'NR==1{print $1}'; }
left_reads() { $PY $READ --model "$1" --dry-run | grep -o 'running [0-9]*' | awk '{print $2}'; }

captions() {
  for i in $(seq 1 20); do
    n=$(left_captions); log "C11 captions: $n videos left"
    [ "$n" = "0" ] && { touch $DONE; log "C11 captions done"; return 0; }
    $PY $CAP --max-5h 0.7 --max-7d 0.5 >> logs/c11_captions.log 2>&1
    [ "$(left_captions)" = "0" ] && { touch $DONE; log "C11 captions done"; return 0; }
    $PY $QW 0.65 >> logs/orchestrator_c11.log
  done
  log "C11 captions: gave up after 20 attempts"; touch $FAILED; return 1
}

reads() {  # model: read every captioned video, wait for more captions, until all are captioned and read
  for i in $(seq 1 500); do
    stopped=$($PY $READ --model "$1" --max-5h 0.8 2>&1 | grep -v '^skipped, ' | tee -a "logs/c11_read_$1.log" \
              | grep -c "quota use reached")
    n=$(left_reads "$1")
    if [ -e $DONE ] && [ "$n" = "0" ]; then log "reads $1: done"; return 0; fi
    [ -e $FAILED ] && { log "reads $1: captions failed; stopping"; return 1; }
    if [ "$stopped" != "0" ]; then $PY $QW 0.75 >> logs/orchestrator_c11.log; else sleep 120; fi
  done
  log "reads $1: gave up after 500 rounds"; return 1
}

case "$1" in
  check)
    $PY $CAP --limit 3 --max-5h 0.7 --max-7d 0.5 >> logs/c11_captions.log 2>&1
    log "check done" ;;
  run)
    rm -f $DONE $FAILED
    log "run: captions, with reads following"
    captions &
    reads haiku &
    reads sonnet &
    wait
    if [ -e $DONE ] && [ "$(left_reads haiku)" = "0" ] && [ "$(left_reads sonnet)" = "0" ]; then
      log "analyses"
      for c in c11 cost; do
        $PY scripts/index_qa/reader.py "$c" > "results/index_qa/reports/$c.txt" 2>&1
      done
      log "done"
    else
      log "stopped before the end; run again to resume"
    fi ;;
  stop)
    pkill -f "run_c11.sh run"
    pkill -f "self_caption.py --set confirm --model sonnet --whole --aspects"
    pkill -f "reader.py run --set confirm --arms sonnetaspects"
    pkill -f "scripts/index_qa/quota_wait.py"
    log "stopped by hand" ;;
  *) echo "usage: $0 check|run|stop"; exit 2 ;;
esac
