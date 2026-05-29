#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill http://10.56.160.38:30000 8998 \
  --decode http://10.56.160.36:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --worker-startup-timeout-secs 1800 \
  --worker-startup-check-interval 10 \
  --request-timeout-secs 7200 \
  --max-concurrent-requests 512 \
  --queue-size 2048
