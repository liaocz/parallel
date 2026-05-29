#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: start_tp8_dp8_dpa_megamoe_profile.sh NODE_RANK [VARIANT] [CHUNKED_PREFILL] [DIST_PORT]}"
variant="${2:-tp8_dp8_dpa_megamoe_profile}"
chunked_prefill="${3:-2048}"
dist_port="${4:-29700}"

dist_init_addr="10.56.160.38:${dist_port}"
root_dir="/home/admin/parallel/dm_tp_dp_8"
run_dir="${root_dir}/profile_runs/${variant}"
metrics_dir="${root_dir}/metrics/${variant}"
profile_dir="${root_dir}/profile/${variant}"
log_path="${run_dir}/rank${rank}.server.log"

mkdir -p "$run_dir" "$metrics_dir" "$profile_dir" /home/admin/sglang_runs
rm -f "$log_path"

cd /sgl-workspace/sglang

export SGLANG_TORCH_PROFILER_DIR="$profile_dir"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3

{
  echo "=== start ${variant} rank${rank} ==="
  date --iso-8601=seconds
  echo "dist_init_addr=${dist_init_addr}"
  echo "chunked_prefill_size=${chunked_prefill}"
  echo "profile_dir=${profile_dir}"
} | tee "$log_path"

python3 -m sglang.launch_server \
  --model-path /home/admin/DeepSeek-V4-Flash/ \
  --tp-size 8 \
  --nnodes 2 \
  --dist-init-addr "$dist_init_addr" \
  --node-rank "$rank" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000 \
  --context-length 202752 \
  --max-total-tokens 202752 \
  --chunked-prefill-size "$chunked_prefill" \
  --max-prefill-tokens 16384 \
  --mem-fraction-static 0.80 \
  --attention-backend dsv4 \
  --kv-cache-dtype fp8_e4m3 \
  --pre-warm-nccl \
  --dp-size 8 \
  --enable-nccl-nvls \
  --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --enable-metrics \
  --enable-mfu-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir "$metrics_dir" \
  2>&1 | tee -a "$log_path"
