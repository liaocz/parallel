#!/usr/bin/env bash
set -euo pipefail

mode="${1:-tp4_dp4_dpa_nvl_cg16}"
root_dir="/home/admin/0528"
log_path="${root_dir}/server_logs/${mode}_server.log"
cmd_path="${root_dir}/commands/${mode}_server_command.sh"

mkdir -p "${root_dir}/server_logs" "${root_dir}/commands" "${root_dir}/metrics/${mode}"
rm -f "${log_path}" "${cmd_path}"

cd /sgl-workspace/sglang

export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_PROTO=LL128
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_DEBUG=INFO

cat > "${cmd_path}" <<'CMD'
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_PROTO=LL128
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_DEBUG=INFO

python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /home/admin/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --dp-size 4 \
  --enable-dp-attention \
  --quantization fp8 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --max-running-requests 256 \
  --schedule-conservativeness 0.3 \
  --moe-a2a-backend megamoe \
  --enable-cache-report \
  --enable-nccl-nvls \
  --cuda-graph-max-bs 16
CMD

{
  echo "=== ${mode} server start ==="
  date --iso-8601=seconds
  cat "${cmd_path}"
  echo
  df -h /dev/shm || true
  nvidia-smi topo -m || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "${log_path}"

bash "${cmd_path}" 2>&1 | tee -a "${log_path}"
