#!/bin/bash
# watch_ensemble.sh — watchdog for a chained FastEddy ensemble (submit_chain.sh).
# cgf / 2026-11-05
#
# Every POLL_S seconds:
#   - every FE_member_* / pod_all_bases job in the chain: log each state change
#     (PENDING -> RUNNING -> COMPLETED / FAILED / TIMEOUT / CANCELLED / OUT_OF_MEMORY)
#   - the running member: validate its new output frames (check_member_output.py:
#     finite, physical bounds, forcing, time stamps, sizes) and project its runtime
#   - stall: no new frame for STALL_S while the member is RUNNING
#   - a member that ends: final full check (all 417 frames present and clean)
# Lines starting with "ALERT" need attention; "STATE"/"DONE"/"OK" are progress.
# Exits when no chain job is left in the queue. Runs independently of any Claude session:
#   nohup setsid bash watch_ensemble.sh ensemble-lowvar > /dev/null 2>&1 &
# Log: <ensemble>/watchdog.log
set -u
ENS=${1:-ensemble-lowvar}
ROOT=/home/cego6160/workspace/ensemble_runs_paper
ENS_DIR=$ROOT/$ENS
CHECK=$ROOT/SESP-WFE/preprocessing/check_member_output.py
LOG=$ENS_DIR/watchdog.log
POLL_S=${POLL_S:-600}
STALL_S=${STALL_S:-900}
EXPECTED=417

PY=${PY:-$HOME/.conda/envs/fasteddy/bin/python}   # fasteddy env (netCDF4, numpy)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "START watchdog for $ENS (pid $$, poll ${POLL_S}s, stall ${STALL_S}s)"

declare -A last_state
chain_jobs() {   # "jobid name state" for this chain: queued jobs from squeue (sacct hides jobs held
                 # on dependencies), finished ones from sacct; queue wins for a job in both
    { squeue -u "$USER" -h -o "%i|%j|%T|%Z" 2>/dev/null | sed 's/^/Q|/'
      sacct -u "$USER" -S now-30days -X -n -P -o JobID,JobName,State,WorkDir 2>/dev/null | sed 's/^/A|/'
    } | awk -F'|' -v d="$ENS_DIR" '
        (($3 ~ /^FE_member_/ && index($5, d) == 1) || ($3 == "pod_all_bases" && $2 >= 22000)) {
            if (!($2 in st) || $1 == "Q") { st[$2] = $4; nm[$2] = $3 } }
        END { for (j in st) print j, nm[j], st[j] }' | sort -n
}

while true; do
    in_queue=0
    while read -r jid name state; do
        [ -z "$jid" ] && continue
        state=${state%% *}
        key="$jid"
        if [ "${last_state[$key]:-}" != "$state" ]; then
            last_state[$key]=$state
            case "$state" in
                PENDING|RUNNING|COMPLETED) log "STATE $name ($jid) -> $state" ;;
                *) log "ALERT $name ($jid) -> $state" ;;
            esac
            if [[ "$name" == FE_member_* && "$state" != PENDING && "$state" != RUNNING ]]; then
                id=${name#FE_member_}
                out=$("$PY" "$CHECK" "$id" --ensemble "$ENS" --brief 2>>"$LOG.stderr"); rc=$?; [ -z "$out" ] && out="(checker crashed; see $LOG.stderr)"
                n=${out%%/*}
                if [ $rc -eq 0 ] && [ "$n" = "$EXPECTED" ] && [ "$state" = COMPLETED ]; then
                    log "DONE $name: $n/$EXPECTED frames, all clean"
                else
                    log "ALERT $name ended $state with $n/$EXPECTED frames (check rc=$rc): $out"
                fi
            fi
        fi
        if [ "$state" = PENDING ] || [ "$state" = RUNNING ]; then in_queue=1; fi
        if [[ "$name" == FE_member_* && "$state" == RUNNING ]]; then
            id=${name#FE_member_}
            out=$("$PY" "$CHECK" "$id" --ensemble "$ENS" --brief 2>>"$LOG.stderr"); rc=$?; [ -z "$out" ] && out="(checker crashed; see $LOG.stderr)"
            if [ $rc -ne 0 ]; then
                log "ALERT $name data check failed: $out"
            else
                log "OK $name $out"
                [[ "$out" == *"fits=False"* ]] && log "ALERT $name projected runtime exceeds the 24 h limit"
            fi
            outdir=$ENS_DIR/member_$id/output
            newest=$(ls -t "$outdir" 2>/dev/null | head -1)
            if [ -n "$newest" ]; then
                age=$(( $(date +%s) - $(stat -c %Y "$outdir/$newest") ))
                [ $age -gt $STALL_S ] && log "ALERT $name stalled: newest frame $newest is ${age}s old"
            fi
        fi
    done < <(chain_jobs)
    if [ $in_queue -eq 0 ]; then
        log "END no chain jobs left in the queue; watchdog exiting"
        exit 0
    fi
    sleep "$POLL_S"
done
