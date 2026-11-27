#!/usr/bin/env bash
set -euo pipefail
# The Python guard reserves time between images. The wall timer also covers
# imports and prevents an individual stalled call from using an open-ended budget.
exec timeout --signal=TERM --kill-after=5s 295s python -u -m scripts.run_lc007h_e40 "$@"
