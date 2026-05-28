#!/usr/bin/env bash
set -u -o pipefail

mode="${1:?usage: run_bench_matrix.sh MODE}"
out_dir="${2:-/home/admin/dm_tp_8/raw}"
host="${BENCH_HOST:-127.0.0.1}"
port="${BENCH_PORT:-30000}"
model="${BENCH_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
num_prompts="${BENCH_NUM_PROMPTS:-100}"
timeout_s="${BENCH_TIMEOUT_S:-7200}"

mkdir -p "$out_dir"

cases=(
  "4k_500:4096:500"
  "32k_500:32768:500"
  "128k_500:131072:500"
)
concurrencies=(1 4 8 16 32 64 128)

health_code() {
  curl -s -m 10 -o /tmp/sglang_health.out -w "%{http_code}" "http://${host}:${port}/health" || true
}

models_code() {
  curl -s -m 10 -o /tmp/sglang_models.out -w "%{http_code}" "http://${host}:${port}/v1/models" || true
}

for case_spec in "${cases[@]}"; do
  IFS=: read -r case_name input_len output_len <<<"$case_spec"
  for concurrency in "${concurrencies[@]}"; do
    json_path="${out_dir}/${mode}_${case_name}_c${concurrency}.json"
    log_path="${out_dir}/${mode}_${case_name}_c${concurrency}.log"

    if [[ -s "$json_path" ]] && python3 - "$json_path" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
raise SystemExit(0 if data.get("completed", 0) > 0 and "output_throughput" in data else 1)
PY
    then
      echo "$(date '+%F %T') skip existing successful result: ${json_path}" | tee -a "${out_dir}/${mode}_matrix.log"
      continue
    fi

    {
      echo "BEGIN mode=${mode} case=${case_name} input=${input_len} output=${output_len} concurrency=${concurrency} at $(date '+%F %T')"
      echo "models_http=$(models_code)"
      cat /tmp/sglang_models.out || true
      echo
      echo "health_http=$(health_code)"
      df -h /dev/shm || true
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    } | tee "$log_path"

    export TOKENIZERS_PARALLELISM=false
    export RAYON_NUM_THREADS=1

    timeout "$timeout_s" python3 -m sglang.bench_serving \
      --backend sglang \
      --host "$host" \
      --port "$port" \
      --model "$model" \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len "$output_len" \
      --num-prompts "$num_prompts" \
      --max-concurrency "$concurrency" \
      --output-file "$json_path" \
      2>&1 | tee -a "$log_path"
    bench_rc=${PIPESTATUS[0]}

    {
      echo "bench_rc=${bench_rc}"
      echo "health_http_after=$(health_code)"
      df -h /dev/shm || true
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
      echo "END mode=${mode} case=${case_name} concurrency=${concurrency} at $(date '+%F %T')"
    } | tee -a "$log_path"

    if [[ "$bench_rc" -ne 0 ]]; then
      echo "$(date '+%F %T') failed mode=${mode} case=${case_name} concurrency=${concurrency} rc=${bench_rc}" | tee -a "${out_dir}/${mode}_matrix.log"
      if [[ "$(health_code)" != "200" ]]; then
        echo "$(date '+%F %T') service is unhealthy after failed benchmark; stopping matrix" | tee -a "${out_dir}/${mode}_matrix.log"
        exit "$bench_rc"
      fi
    else
      echo "$(date '+%F %T') completed mode=${mode} case=${case_name} concurrency=${concurrency}" | tee -a "${out_dir}/${mode}_matrix.log"
    fi
  done
done
