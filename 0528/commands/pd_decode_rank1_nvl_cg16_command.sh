#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

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
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
export MC_TCP_ENABLE_CONNECTION_POOL=true
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320

python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /home/admin/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30001 \
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.56.160.36:29521 \
  --dp-size 8 \
  --enable-dp-attention \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
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
  --pre-warm-nccl \
  --cuda-graph-max-bs 16 \
  --decode-log-interval 100
