#!/usr/bin/env bash
set -euo pipefail

name="${1:-sgl-0528}"
image="${2:-docker.io/lmsysorg/sglang:latest}"

podman rm -f "${name}" >/dev/null 2>&1 || true

podman run -d \
  --name "${name}" \
  --network host \
  --ipc host \
  --runtime /usr/bin/nvidia-container-runtime \
  --gpus all \
  --privileged \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_MNNVL_ENABLE=1 \
  -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_NVLS_ENABLE=1 \
  -e NCCL_P2P_LEVEL=NVL \
  -e NCCL_PROTO=LL128 \
  -e NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
  -e NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
  -e NCCL_DEBUG=INFO \
  -v /home/admin:/home/admin \
  -v /dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels:rw \
  -v /dev/infiniband:/dev/infiniband:rw \
  --entrypoint bash \
  "${image}" \
  -lc "tail -f /dev/null"
