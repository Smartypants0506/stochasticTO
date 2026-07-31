#!/usr/bin/env bash
# Status dashboard for the long-running study jobs.
#
#   ./scripts/watch_progress.sh            one snapshot
#   watch -n 60 ./scripts/watch_progress.sh   live, refreshing each minute
#
# Reads ONLY the log files, so it needs no sudo, no docker exec, and cannot
# disturb a running job. Liveness is inferred from log mtime: a log that has not
# been written to in 3 minutes is either finished or wedged, and the two are
# distinguished by whether its DONE marker is present.
set -u
cd "$(dirname "$0")/.." || exit 1

# A long silence is NORMAL here, so this threshold has to be generous or it
# cries wolf. Two reasons these jobs go quiet for many minutes at a time:
#   1. Only rank 0's sample-parallel GROUP logs progress. With 4 groups the
#      other three are invisible, so a straggler group looks like a freeze.
#   2. Each new correlation length is a KL cache MISS -> a dense O(N_nodes^2)
#      eigensolve on ~14k nodes, which is minutes of silent work.
# Measured: the l_c sweep routinely goes 6-10 min between log lines while
# perfectly healthy. 180 s produced constant false alarms.
STALL_SECONDS="${STALL_SECONDS:-1200}"

hr() { printf '%.0s-' {1..64}; echo; }

age_of() {  # seconds since last write, or "-" if missing
    [ -f "$1" ] || { echo "-"; return; }
    echo $(( $(date +%s) - $(stat -c %Y "$1") ))
}

liveness() {  # $1=log $2=done-marker-regex
    local age; age=$(age_of "$1")
    [ "$age" = "-" ] && { echo "MISSING"; return; }
    if grep -qE "$2" "$1" 2>/dev/null; then echo "DONE"
    elif [ "$age" -gt "$STALL_SECONDS" ]; then echo "STALLED (${age}s idle)"
    else echo "LIVE (${age}s ago)"; fi
}

fmt_eta() {  # $1=seconds remaining
    local s=$1
    [ "$s" -le 0 ] && { echo "now"; return; }
    printf '%dh%02dm  (ETA %s)' $((s/3600)) $(((s%3600)/60)) "$(date -d "+${s} seconds" '+%H:%M')"
}

echo "=== $(date '+%Y-%m-%d %H:%M:%S')  |  load:$(cut -d' ' -f1-3 /proc/loadavg)  |  cores:$(nproc) ==="

# Is the push notifier alive? If it died, jobs finish silently -- which is the
# one failure mode that is invisible precisely when you are relying on it.
if pgrep -f "job_watch.sh" > /dev/null 2>&1; then
    st=/tmp/stochasticTO_watch
    echo "notifier: RUNNING (ntfy)  notified: $(ls "$st"/*.notified 2>/dev/null | wc -l)  watching: $(ls "$st"/*.seen 2>/dev/null | wc -l)"
else
    echo "notifier: *** NOT RUNNING *** -> nohup scripts/job_watch.sh > watch_log.txt 2>&1 &"
fi

# ---------------------------------------------------------------- gap study
hr
LOG=gap_log.txt
if [ -f "$LOG" ]; then
    done_reps=$(grep -c "^INFO:__main__:replication [0-9]*: in-sample" "$LOG")
    cur=$(grep "=== replication" "$LOG" | tail -1 | grep -oE "replication [0-9]+/[0-9]+")
    total=10
    echo "SAA GAP STUDY (64 ranks)          $(liveness "$LOG" 'SAA gap at N=')"
    echo "  completed replications : ${done_reps}/${total}   (in flight: ${cur:-n/a})"
    # Rate from the log's own span, so it survives restarts of this script.
    start=$(stat -c %Y "$LOG"); first=$(head -1 "$LOG" >/dev/null 2>&1 && echo ok)
    if [ "$done_reps" -gt 0 ]; then
        elapsed=$(( $(date +%s) - $(stat -c %W "$LOG" 2>/dev/null || echo 0) ))
        [ "$elapsed" -le 0 ] && elapsed=$(( $(date +%s) - $(stat -c %Y "$LOG") + done_reps*10944 ))
        per=$(( elapsed / done_reps ))
        echo "  rate                   : $(( per/3600 ))h$(( (per%3600)/60 ))m per replication"
        echo "  remaining              : $(fmt_eta $(( (total-done_reps-1) * per )))  .. $(fmt_eta $(( (total-done_reps) * per )))"
    fi
    echo "  last sigma optimism    : $(grep 'optimism' "$LOG" | tail -1 | grep -oE 'optimism [+-][0-9.]+%' || echo n/a)"
    echo "  writes saa_gap.json only at the very end"
else
    echo "SAA GAP STUDY: no gap_log.txt"
fi

# ------------------------------------------------------------------ phase 0
hr
LOG=phase0_log.txt
if [ -f "$LOG" ]; then
    echo "PHASE 0 (32 ranks)                $(liveness "$LOG" 'PHASE0 DONE')"
    for m in "P0-A uniform_eta" "P0-B correlation_length"; do
        tag=${m%% *}
        if grep -q "$tag" "$LOG"; then
            state="started"
            grep -q "EXIT_${tag##*-}=0" "$LOG" && state="finished ok"
            grep -qE "EXIT_${tag##*-}=[1-9]" "$LOG" && state="FAILED"
            echo "  $m : $state"
        else
            echo "  $m : queued"
        fi
    done
    its=$(grep -c "\[SAA\] outer_iter=" "$LOG")
    stage=$(grep -oE "stage [0-9]/5" "$LOG" | tail -1)
    [ "$its" -gt 0 ] && echo "  SAA iterations         : ${its}/150   ${stage:-}"
    lc=$(grep -c "^INFO:__main__:l_c=" "$LOG")
    [ "$lc" -gt 0 ] && echo "  l_c levels evaluated   : ${lc}/7"
    echo "  last line: $(tail -1 "$LOG" | cut -c1-72)"
else
    echo "PHASE 0: no phase0_log.txt"
fi

# -------------------------------------------------------------- move probe
hr
LOG=probe_log.txt
if [ -f "$LOG" ]; then
    # A queued job polls silently, so its log is idle by design -- reporting
    # that as STALLED would raise a false alarm on every check until it starts.
    if grep -q "starting move-limit probe" "$LOG"; then
        state="$(liveness "$LOG" 'PROBE DONE')"
    else
        state="QUEUED (waiting for Phase 0)"
    fi
    echo "MOVE-LIMIT PROBE (32 ranks)       $state"
    if grep -q "starting move-limit probe" "$LOG"; then
        st=$(grep -oE "stage [0-9]/5" "$LOG" | tail -1)
        echo "  ${st:-starting}   baseline to beat: stat_rel 0.1244"
        grep -E "FINAL stat_rel|PASS:|FAIL:|MIXED:" "$LOG" | sed 's/^/  /'
    else
        echo "  queued behind Phase 0"
    fi
fi

# ----------------------------------------------------------------- results
hr
echo "RESULTS ON DISK"
for f in \
    output/studies/saa_gap/*/saa_gap.json \
    output/studies/uniform_eta/uniform_eta_comparison.json \
    output/studies/correlation_length/*/correlation_length_fixed_design.json \
    output/studies/baseline_comparison/*/baseline_comparison.json ; do
    [ -e "$f" ] && echo "  [x] $f"
done
compgen -G "output/studies/saa_gap/*/saa_gap.json" >/dev/null || echo "  [ ] saa_gap.json (gap study still running)"
hr
