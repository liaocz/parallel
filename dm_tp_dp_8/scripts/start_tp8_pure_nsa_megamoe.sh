#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: start_tp8_pure_nsa_megamoe.sh NODE_RANK}"
variant="${SGLANG_VARIANT:-tp8_pure_nsa_megamoe}"
log_path="/home/admin/sglang_runs/${variant}_rank${rank}.log"
metrics_dir="/home/admin/sglang_metrics/${variant}"
dist_init_addr="${DIST_INIT_ADDR:-10.56.160.38:29600}"
extra_args=()

if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  extra_args+=(--disable-cuda-graph)
fi

mkdir -p /home/admin/sglang_runs "$metrics_dir" /home/admin/dm_tp_8/raw
rm -f "$log_path"

cd /sgl-workspace/sglang

export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3

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
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --mem-fraction-static 0.80 \
  --attention-backend nsa \
  --nsa-prefill-backend trtllm \
  --nsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --pre-warm-nccl \
  --moe-a2a-backend megamoe \
  --enable-nccl-nvls \
  --enable-flashinfer-allreduce-fusion \
  --enable-metrics \
  --enable-mfu-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir "$metrics_dir" \
  "${extra_args[@]}" \
  2>&1 | tee "$log_path"
