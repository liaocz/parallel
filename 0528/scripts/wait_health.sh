#!/usr/bin/env bash
set -euo pipefail

host="${1:-127.0.0.1}"
port="${2:-30000}"
timeout_s="${3:-1800}"
log_path="${4:-/home/admin/0528/logs/wait_health_${host}_${port}.log}"
start_ts="$(date +%s)"
mkdir -p "$(dirname "${log_path}")"

while true; do
  code="$(curl -s -m 10 -o /tmp/sglang_wait_health.out -w "%{http_code}" "http://${host}:${port}/health" || true)"
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  echo "$(date '+%F %T') host=${host} port=${port} code=${code} elapsed=${elapsed}s" | tee -a "${log_path}"
  if [[ "${code}" == "200" ]]; then
    exit 0
  fi
  if (( elapsed >= timeout_s )); then
    echo "timeout waiting for http://${host}:${port}/health" | tee -a "${log_path}"
    exit 1
  fi
  sleep 10
done
