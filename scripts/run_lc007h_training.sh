#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 AUTHORIZATION MANIFEST NEW_RUN_ROOT" >&2
  exit 2
fi
authorization=$1
manifest=$2
root=$3
mkdir "$root"
cp -- "$authorization" "$root/authorization.json"
date --iso-8601=seconds > "$root/launch_time.txt"
rc=0
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=5s 7195s \
  python -u -m scripts.run_lc007h_training --authorization "$root/authorization.json" \
  --manifest "$manifest" --output "$root/run" > "$root/runner.log" 2>&1 || rc=$?
date --iso-8601=seconds > "$root/exit_time.txt"
printf '%s\n' "$rc" > "$root/exit_status.txt"
exit "$rc"
