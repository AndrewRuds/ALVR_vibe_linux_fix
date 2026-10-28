#!/usr/bin/env bash
# Logs NVIDIA GPU memory and load once a second while you play, so a crash can
# be matched against what the GPU was doing just before it.
#
# Usage: scripts/vram-log.sh [output file]   (default: ~/alvr_vram_log.txt)
# Stop with Ctrl+C; the peak memory use is printed at the end.
#
# Each line: time, VRAM used/total (MiB), GPU %, encoder %, temperature (C),
# core clock (MHz), power (W). Every 5 s the processes using the most VRAM are
# listed, and NVIDIA kernel errors ("Xid") are copied in as they happen.

set -u

out="${1:-$HOME/alvr_vram_log.txt}"
interval=1
proc_every=5

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found (it comes with nvidia-utils)" >&2
    exit 1
fi

peak=0
peak_time=""
xid_pid=""

finish() {
    [ -n "$xid_pid" ] && kill "$xid_pid" 2>/dev/null
    echo
    echo "Peak VRAM use: ${peak} MiB at ${peak_time:-n/a}" | tee -a "$out"
    echo "Log: $out"
    exit 0
}
trap finish INT TERM

echo "# VRAM log started $(date -Iseconds)" >>"$out"
echo "Logging to $out (Ctrl+C to stop)"

# NVIDIA reports GPU faults to the kernel log as "NVRM: Xid ...". Reading it may
# need the systemd-journal or wheel group; without it this part is skipped.
if command -v journalctl >/dev/null; then
    journalctl -k -f -n 0 -o short-iso 2>/dev/null \
        | grep --line-buffered -i "xid" \
        | sed -u 's/^/XID /' >>"$out" &
    xid_pid=$!
fi

tick=0
while true; do
    now="$(date +%H:%M:%S)"
    line="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.encoder,temperature.gpu,clocks.sm,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | head -n1)"

    if [ -z "$line" ]; then
        echo "$now nvidia-smi gave no data" | tee -a "$out"
    else
        IFS=', ' read -r used total gpu enc temp clock power <<<"$line"
        printf '%s vram %5s/%s MiB  gpu %3s%%  enc %3s%%  %3sC  %4s MHz  %s W\n' \
            "$now" "$used" "$total" "$gpu" "$enc" "$temp" "$clock" "$power" | tee -a "$out"

        if [ "${used%%.*}" -gt "$peak" ] 2>/dev/null; then
            peak="${used%%.*}"
            peak_time="$now"
        fi
    fi

    # Per-process use. pmon lists graphics processes too, unlike
    # --query-compute-apps. The fb column is VRAM in MiB.
    if [ $((tick % proc_every)) -eq 0 ]; then
        nvidia-smi pmon -c 1 -s m 2>/dev/null \
            | awk -v t="$now" '
                /^#/ { next }
                $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ {
                    cmd = "ps -o comm= -p " $2
                    name = ""
                    cmd | getline name
                    close(cmd)
                    if (name == "") name = "?"
                    print $4, $2, name
                }' \
            | sort -rn | head -n 6 \
            | while read -r fb pid name; do
                printf '%s   proc %5s MiB  %-7s %s\n' "$now" "$fb" "$pid" "$name"
            done | tee -a "$out"
    fi

    tick=$((tick + 1))
    sleep "$interval"
done
