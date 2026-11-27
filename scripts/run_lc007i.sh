#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 AUTHORIZATION MANIFEST NEW_RUN_ROOT" >&2
  exit 2
fi
root=$3
mkdir "$root"
cp -- "$1" "$root/authorization.json"
date --iso-8601=seconds > "$root/launch_time.txt"
rc=0
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=5s 1795s \
  python -u -m scripts.run_lc007i --authorization "$root/authorization.json" \
  --manifest "$2" --output "$root/run" > "$root/runner.log" 2>&1 || rc=$?
date --iso-8601=seconds > "$root/exit_time.txt"
printf '%s\n' "$rc" > "$root/exit_status.txt"
exit "$rc"
