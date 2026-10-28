#!/bin/bash
# E2 (research log): the EgoSchema confirmatory run on ego_confirm (160 questions), E1's arms and settings; run_e1.sh
# with ego_dev -> ego_confirm and e1 -> e2. Resumable: stop it at any time and run it again later; every step skips
# what is done (caption files are written atomically; a partial reader row is skipped and its item rerun).
#   scripts/index_qa/run_e2.sh check   # 3 windows per captioner and 1 five-frames read, then stop for inspection
#   scripts/index_qa/run_e2.sh run     # captions (Haiku and Sonnet, NSHARDS processes each; Sonnet's whole-video
#                                      # captions after its windows), with Haiku and Sonnet reading each video once
#                                      # its captions exist
#   scripts/index_qa/run_e2.sh status  # what is left
#   scripts/index_qa/run_e2.sh stop    # stop everything (calls in flight are lost and redone on resume)
# Quota stop rules: --max-5h 0.99, --max-7d 0.9 (E1's final settings); after a stop, wait for the 5-hour reset.
# NSHARDS=3 by default (environment variable). Logs: logs/orchestrator_e2.log, logs/e2_cap_<job>_<k>.log,
# logs/e2_read_{haiku,sonnet}.log. Edit this file only while it is stopped.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
QW=scripts/index_qa/quota_wait.py
NSHARDS=${NSHARDS:-3}
MAX5H=0.99
SC="scripts/index_qa/self_caption.py --set ego_confirm"
declare -A CAP=([haiku]="$SC --model haiku" [sonnet]="$SC --model sonnet" [whole]="$SC --model sonnet --whole")
declare -A ARMS=([haiku]="blind frame1 frames5 selfcaptions sonnetcaptions sonnetwhole"
                 [sonnet]="blind frames5 sonnetcaptions")
mkdir -p logs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a logs/orchestrator_e2.log; }

left_captions() { $PY ${CAP[$1]} ${2:+--shard $2} --dry-run | awk 'NR==1{print $1}'; }  # job [K/N]
read_cmd() { echo "scripts/index_qa/reader.py run --set ego_confirm --model $1 --arms ${ARMS[$1]} --skip-missing"; }
left_reads() { $PY $(read_cmd "$1") --dry-run | grep -o 'running [0-9]*' | awk '{print $2}'; }

shard() {  # job k: caption shard k of NSHARDS until none of its windows (or videos) is left
  local s="$2/$NSHARDS"
  for i in $(seq 1 30); do
    [ "$(left_captions "$1" "$s")" = "0" ] && return 0
    $PY ${CAP[$1]} --shard "$s" --max-5h $MAX5H --max-7d 0.9 >> "logs/e2_cap_$1_$2.log" 2>&1
    [ "$(left_captions "$1" "$s")" = "0" ] && return 0
    $PY $QW 0.9 >> logs/orchestrator_e2.log
  done
  log "captions $1 shard $s: gave up after 30 attempts"; touch logs/e2_cap.failed; return 1
}

captions() {  # job: all its shards in parallel, then the done flag
  log "captions $1: $(left_captions "$1") left, $NSHARDS shards"
  for k in $(seq 0 $((NSHARDS - 1))); do shard "$1" "$k" & done
  wait
  [ "$(left_captions "$1")" = "0" ] && { touch "logs/e2_cap_$1.done"; log "captions $1 done"; return 0; }
  return 1
}

all_captions_done() { [ -e logs/e2_cap_haiku.done ] && [ -e logs/e2_cap_sonnet.done ] && [ -e logs/e2_cap_whole.done ]; }

reads() {  # model: read every item whose captions exist, wait for more, until all captions exist and all is read
  for i in $(seq 1 1000); do
    stopped=$($PY $(read_cmd "$1") --max-5h $MAX5H 2>&1 | grep -v '^skipped, ' | tee -a "logs/e2_read_$1.log" \
              | grep -c "quota use reached")
    n=$(left_reads "$1")
    if all_captions_done && [ "$n" = "0" ]; then log "reads $1: done"; return 0; fi
    [ -e logs/e2_cap.failed ] && { log "reads $1: captions failed; stopping"; return 1; }
    if [ "$stopped" != "0" ]; then $PY $QW 0.9 >> logs/orchestrator_e2.log; else sleep 120; fi
  done
  log "reads $1: gave up after 1000 rounds"; return 1
}

case "$1" in
  check)
    $PY ${CAP[haiku]} --limit 3 --max-5h $MAX5H --max-7d 0.9 2>&1 | tee -a logs/e2_cap_haiku_check.log
    $PY ${CAP[sonnet]} --limit 3 --max-5h $MAX5H --max-7d 0.9 2>&1 | tee -a logs/e2_cap_sonnet_check.log
    $PY scripts/index_qa/reader.py run --set ego_confirm --model haiku --arms frames5 --limit 1 --max-5h $MAX5H 2>&1 \
      | tee -a logs/e2_read_haiku.log
    log "check done" ;;
  run)
    rm -f logs/e2_cap_*.done logs/e2_cap.failed
    log "run: captions (haiku, sonnet, then sonnet whole; $NSHARDS shards each), with reads following"
    captions haiku &
    (captions sonnet && captions whole) &
    reads haiku &
    reads sonnet &
    wait
    if all_captions_done && [ "$(left_reads haiku)" = "0" ] && [ "$(left_reads sonnet)" = "0" ]; then
      log "done: all captions written and all reads saved; next, the E2 analysis (reader.py e2, run once)"
    else
      log "stopped before the end; run again to resume"
    fi ;;
  status)
    for j in haiku sonnet whole; do echo "captions $j: $(left_captions $j) left"; done
    for m in haiku sonnet; do echo "reads $m: $(left_reads $m) runnable now (items without captions not counted)"; done ;;
  stop)
    pkill -f "run_e2.sh run"
    pkill -f "self_caption.py --set ego_confirm"
    pkill -f "reader.py run --set ego_confirm"
    pkill -f "scripts/index_qa/quota_wait.py"
    log "stopped by hand" ;;
  *) echo "usage: $0 check|run|status|stop"; exit 2 ;;
esac
