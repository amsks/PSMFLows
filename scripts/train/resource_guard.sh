#!/usr/bin/env bash
# Passive resource watchdog for the DrQ pixel sweep. Every INTERVAL seconds it
# appends one status line (RAM/dev-shm/tmp/root + #runs) to logs/resource_guard.log
# and prefixes "WARNING" if any safety threshold is crossed. READ-ONLY: it never
# deletes or kills anything -- it only surfaces early warning. Detached via nohup.
#   INTERVAL=300  ROOT_MIN_GB=2  MEM_MIN_GB=30  SHM_MAX_PCT=90  TMP_MAX_PCT=90
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
INTERVAL="${INTERVAL:-300}"
ROOT_MIN_GB="${ROOT_MIN_GB:-2}"
MEM_MIN_GB="${MEM_MIN_GB:-30}"
SHM_MAX_PCT="${SHM_MAX_PCT:-90}"
TMP_MAX_PCT="${TMP_MAX_PCT:-90}"
LOG="logs/resource_guard.log"

echo "[$(date '+%F %T')] resource_guard start (interval=${INTERVAL}s thresholds: root<${ROOT_MIN_GB}G mem<${MEM_MIN_GB}G shm>${SHM_MAX_PCT}% tmp>${TMP_MAX_PCT}%)" >> "$LOG"
while true; do
    mem_avail=$(free -g | awk '/^Mem:/{print $7}')
    shm_pct=$(df --output=pcent /dev/shm | tail -1 | tr -dc '0-9')
    shm_used=$(df -h /dev/shm | awk 'NR==2{print $3}')
    tmp_pct=$(df --output=pcent /tmp | tail -1 | tr -dc '0-9')
    root_avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    nrun=$(pgrep -fc "main.py --env_name=visual-cube-single" 2>/dev/null || echo 0)
    warn=""
    (( root_avail < ROOT_MIN_GB )) && warn+=" ROOT_LOW"
    (( mem_avail  < MEM_MIN_GB  )) && warn+=" MEM_LOW"
    (( shm_pct   >= SHM_MAX_PCT )) && warn+=" SHM_HIGH"
    (( tmp_pct   >= TMP_MAX_PCT )) && warn+=" TMP_HIGH"
    prefix="[$(date '+%F %T')]"
    [ -n "$warn" ] && prefix="$prefix ⚠️ WARNING$warn"
    echo "$prefix mem_avail=${mem_avail}G shm=${shm_used}(${shm_pct}%) tmp=${tmp_pct}% root_avail~${root_avail}G runs=${nrun}" >> "$LOG"
    sleep "$INTERVAL"
done
