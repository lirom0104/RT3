#!/usr/bin/env bash
# Wait for the active shared run, then resume thermal-only LoRA on GPU 0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TZ=Asia/Shanghai

shared_launcher_pid="${1:?Usage: bash scripts/run_data_deal_gpu0_queue.sh SHARED_LAUNCHER_PID}"
[[ "$shared_launcher_pid" =~ ^[0-9]+$ ]]
exec 9>artifacts/data_deal_gpu0_queue.lock
flock -n 9 || { printf 'A DATA-DEAL GPU 0 queue is already running.\n' >&2; exit 1; }

queue_state=artifacts/data_deal_gpu0_queue_state.txt
printf 'waiting_for_shared\n' > "$queue_state"
printf '%s Waiting for shared launcher PID %s.\n' "$(date --iso-8601=seconds)" "$shared_launcher_pid"
if [[ -r "/proc/$shared_launcher_pid/cmdline" ]]; then
  tr '\0' ' ' < "/proc/$shared_launcher_pid/cmdline" | rg -q 'artifacts/thermal_giant_data_deal/launch.sh'
  shared_start="$(ps -p "$shared_launcher_pid" -o lstart=)"
  while [[ "$(ps -p "$shared_launcher_pid" -o lstart=)" == "$shared_start" ]]; do
    sleep 10
  done
fi

if [[ ! -f artifacts/thermal_giant_data_deal/exit_code.txt ]] ||
   [[ "$(cat artifacts/thermal_giant_data_deal/exit_code.txt)" != 0 ]]; then
  printf 'shared_failed\n' > "$queue_state"
  printf '%s Shared training did not finish successfully; inspect its train.log.\n' "$(date --iso-8601=seconds)" >&2
  exit 1
fi

printf 'running_thermal_only_on_gpu0\n' > "$queue_state"
printf '%s Shared finished; resuming thermal-only LoRA on GPU 0.\n' "$(date --iso-8601=seconds)"
if bash artifacts/thermal_giant_data_deal_ir_only/launch.sh; then
  printf 'completed\n' > "$queue_state"
else
  training_exit_code=$?
  printf 'thermal_only_failed\n' > "$queue_state"
  exit "$training_exit_code"
fi
