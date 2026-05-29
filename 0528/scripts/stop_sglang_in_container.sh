#!/usr/bin/env bash
set -euo pipefail

ps -ef \
  | awk '/python3 -m sglang|sglang serve|bench_serving|sglang_router|launch_server/ && !/awk/ {print $2}' \
  | xargs -r kill -9

sleep 2
ps -ef | grep -E 'python3 -m sglang|sglang serve|bench_serving|sglang_router|launch_server' | grep -v grep || true
