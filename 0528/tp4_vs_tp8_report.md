# L20A SGLang TP4 vs TP8 NVLS/MNNVL Performance Report

- Generated at: 2026-05-29
- Model path: `/home/admin/DeepSeek-V4-Flash`
- Dataset: `random`; input/output cases: `4k_500, 32k_500, 128k_500`
- Concurrency: `1, 2, 4, 8, 16, 32, 64`; `num_prompts=64`; `warmup_requests=1`
- Both TP4 and TP8 deployments used NVLS/MNNVL environment and `--enable-nccl-nvls`; CUDA graph max batch size was set to 16.

## Artifacts

- Raw benchmark JSON/logs: `/home/admin/0528/raw`
- Summary CSV: `/home/admin/0528/summary_tp4_tp8.csv`
- Comparison CSV: `/home/admin/0528/tp4_vs_tp8_comparison.csv`
- Server logs: `/home/admin/0528/server_logs`
- Command files: `/home/admin/0528/commands`

## Test Status

| Mode | Nodes | GPUs | Completed cases | Result |
| --- | --- | --- | --- | --- |
| TP4 + DP4 + DPA + NVLS/MNNVL | 10.56.160.38 | 4 | 21/21 | PASS |
| TP8 + DP8 + DPA + NVLS/MNNVL | 10.56.160.38, 10.56.160.40 | 8 | 21/21 | PASS |

## Effective Server Args

The launch commands passed `--chunked-prefill-size 16384`, but SGLang reported a lower effective value through `/server_info` when DPA was enabled.

| Mode | Arg | Effective value |
| --- | --- | --- |
| TP4 + DP4 + DPA + NVLS/MNNVL | tp_size | 4 |
| TP4 + DP4 + DPA + NVLS/MNNVL | dp_size | 4 |
| TP4 + DP4 + DPA + NVLS/MNNVL | chunked_prefill_size | 4096 |
| TP4 + DP4 + DPA + NVLS/MNNVL | max_prefill_tokens | 16384 |
| TP4 + DP4 + DPA + NVLS/MNNVL | mem_fraction_static | 0.8 |
| TP4 + DP4 + DPA + NVLS/MNNVL | schedule_conservativeness | 0.09 |
| TP4 + DP4 + DPA + NVLS/MNNVL | moe_a2a_backend | megamoe |
| TP4 + DP4 + DPA + NVLS/MNNVL | enable_dp_attention | True |
| TP4 + DP4 + DPA + NVLS/MNNVL | enable_nccl_nvls | True |
| TP4 + DP4 + DPA + NVLS/MNNVL | cuda_graph_max_bs | 16 |
| TP4 + DP4 + DPA + NVLS/MNNVL | quantization | fp8 |
| TP4 + DP4 + DPA + NVLS/MNNVL | kv_cache_dtype | fp8_e4m3 |
| TP8 + DP8 + DPA + NVLS/MNNVL | tp_size | 8 |
| TP8 + DP8 + DPA + NVLS/MNNVL | dp_size | 8 |
| TP8 + DP8 + DPA + NVLS/MNNVL | chunked_prefill_size | 2048 |
| TP8 + DP8 + DPA + NVLS/MNNVL | max_prefill_tokens | 16384 |
| TP8 + DP8 + DPA + NVLS/MNNVL | mem_fraction_static | 0.8 |
| TP8 + DP8 + DPA + NVLS/MNNVL | schedule_conservativeness | 0.09 |
| TP8 + DP8 + DPA + NVLS/MNNVL | moe_a2a_backend | megamoe |
| TP8 + DP8 + DPA + NVLS/MNNVL | enable_dp_attention | True |
| TP8 + DP8 + DPA + NVLS/MNNVL | enable_nccl_nvls | True |
| TP8 + DP8 + DPA + NVLS/MNNVL | cuda_graph_max_bs | 16 |
| TP8 + DP8 + DPA + NVLS/MNNVL | quantization | fp8 |
| TP8 + DP8 + DPA + NVLS/MNNVL | kv_cache_dtype | fp8_e4m3 |

## NVLS/MNNVL Evidence

| Log | Path | MNNVL env lines | NVLS env lines | P2P/MNNVL channel lines |
| --- | --- | --- | --- | --- |
| TP4 | /home/admin/0528/server_logs/tp4_dp4_dpa_nvl_cg16_server.log | 4 | 4 | 399 |
| TP8 rank0 | /home/admin/0528/server_logs/tp8_dp8_dpa_nvl_cg16_rank0_server.log | 4 | 4 | 367 |
| TP8 rank1 | /home/admin/0528/server_logs/tp8_dp8_dpa_nvl_cg16_rank1_server.log | 4 | 4 | 445 |

## Peak Output Throughput

| Case | TP4 peak concurrency / tok/s | TP8 peak concurrency / tok/s | TP8/TP4 | TP8/TP4 per GPU |
| --- | --- | --- | --- | --- |
| 4k_500 | c64 / 2403.60 | c64 / 2149.17 | 0.89 | 0.45 |
| 32k_500 | c64 / 1205.63 | c64 / 1214.97 | 1.01 | 0.50 |
| 128k_500 | c64 / 307.70 | c64 / 291.07 | 0.95 | 0.47 |

## Scaling From c1 To c64

| Case | Mode | c1 output tok/s | c64 output tok/s | c64/c1 |
| --- | --- | --- | --- | --- |
| 4k_500 | TP4 + DP4 + DPA + NVLS/MNNVL | 90.26 | 2403.60 | 26.63 |
| 4k_500 | TP8 + DP8 + DPA + NVLS/MNNVL | 78.04 | 2149.17 | 27.54 |
| 32k_500 | TP4 + DP4 + DPA + NVLS/MNNVL | 84.68 | 1205.63 | 14.24 |
| 32k_500 | TP8 + DP8 + DPA + NVLS/MNNVL | 66.94 | 1214.97 | 18.15 |
| 128k_500 | TP4 + DP4 + DPA + NVLS/MNNVL | 48.68 | 307.70 | 6.32 |
| 128k_500 | TP8 + DP8 + DPA + NVLS/MNNVL | 32.73 | 291.07 | 8.89 |

## Detailed Comparison

### 4k_500

| Concurrency | TP4 out tok/s | TP8 out tok/s | TP8/TP4 | TP8/TP4 per GPU | TP4 mean TTFT ms | TP8 mean TTFT ms | TP4 p99 TPOT ms | TP8 p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 90.26 | 78.04 | 0.86 | 0.43 | 662 | 1080 | 8.5 | 8.6 |
| 2 | 180.72 | 170.34 | 0.94 | 0.47 | 306 | 399 | 14.4 | 16.3 |
| 4 | 333.41 | 305.24 | 0.92 | 0.46 | 190 | 294 | 14.9 | 18.9 |
| 8 | 545.00 | 474.82 | 0.87 | 0.44 | 232 | 309 | 16.9 | 23.3 |
| 16 | 813.48 | 725.79 | 0.89 | 0.45 | 414 | 418 | 25.5 | 67.0 |
| 32 | 1212.62 | 1066.80 | 0.88 | 0.44 | 608 | 881 | 31.5 | 173.9 |
| 64 | 2403.60 | 2149.17 | 0.89 | 0.45 | 914 | 1468 | 20.3 | 244.9 |

### 32k_500

| Concurrency | TP4 out tok/s | TP8 out tok/s | TP8/TP4 | TP8/TP4 per GPU | TP4 mean TTFT ms | TP8 mean TTFT ms | TP4 p99 TPOT ms | TP8 p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 84.68 | 66.94 | 0.79 | 0.40 | 833 | 1602 | 8.6 | 8.7 |
| 2 | 142.32 | 99.84 | 0.70 | 0.35 | 727 | 1549 | 29.4 | 41.9 |
| 4 | 203.84 | 142.92 | 0.70 | 0.35 | 777 | 1429 | 43.6 | 141.0 |
| 8 | 289.34 | 194.36 | 0.67 | 0.34 | 776 | 1434 | 153.5 | 97.7 |
| 16 | 414.72 | 309.76 | 0.75 | 0.37 | 1010 | 1257 | 349.6 | 117.5 |
| 32 | 645.64 | 552.20 | 0.86 | 0.43 | 1482 | 1792 | 333.3 | 479.3 |
| 64 | 1205.63 | 1214.97 | 1.01 | 0.50 | 3324 | 3742 | 889.5 | 941.5 |

### 128k_500

| Concurrency | TP4 out tok/s | TP8 out tok/s | TP8/TP4 | TP8/TP4 per GPU | TP4 mean TTFT ms | TP8 mean TTFT ms | TP4 p99 TPOT ms | TP8 p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 48.68 | 32.73 | 0.67 | 0.34 | 2986 | 5476 | 8.8 | 9.0 |
| 2 | 64.94 | 41.72 | 0.64 | 0.32 | 3170 | 5752 | 46.1 | 71.3 |
| 4 | 94.85 | 56.93 | 0.60 | 0.30 | 2948 | 6235 | 95.9 | 144.3 |
| 8 | 127.10 | 78.62 | 0.62 | 0.31 | 3113 | 5799 | 318.9 | 125.6 |
| 16 | 165.17 | 136.44 | 0.83 | 0.41 | 3528 | 5575 | 1236.4 | 207.7 |
| 32 | 194.06 | 165.34 | 0.85 | 0.43 | 5947 | 8317 | 2929.3 | 1984.4 |
| 64 | 307.70 | 291.07 | 0.95 | 0.47 | 17321 | 19781 | 6291.3 | 1361.0 |

## Summary

- TP8 + DP8 + DPA + NVLS/MNNVL did not deliver expected throughput scaling over single-node TP4. Peak absolute output throughput was lower for `4k_500` and `128k_500`, and only roughly equal for `32k_500`.
- Per-GPU efficiency regressed heavily in TP8. At peak, TP8 per-GPU output throughput was about 45-50% of TP4, even though NVLS/MNNVL was active in the logs.
- Low concurrency is especially weak for TP8: `128k_500_c1` was about 0.67x TP4 in absolute output throughput, while using twice the GPU count.
- Higher concurrency improves TP8 aggregate throughput, but does not remove the scaling gap. Tail latency grows sharply, especially on `128k_500`: TP8 c32 P99 E2E reached about 94s and c64 P99 TTFT reached about 45s.

## Initial Bottleneck Analysis

- NVLS/MNNVL was not simply disabled: rank logs contain `NCCL_MNNVL_ENABLE=1`, `NCCL_NVLS_ENABLE=1`, and many `via P2P/MNNVL` channel lines.
- The result is more consistent with runtime/scheduling and cross-rank efficiency loss than with missing link enablement. TP8 doubles TP ranks and introduces cross-node collectives, while this benchmark has only 64 prompts, so low and medium concurrency leave DP groups lightly loaded.
- Effective prefill chunk size reported by SGLang was lower than the launch value. TP8 reported `chunked_prefill_size=2048`; this increases scheduling turns for long prompts and can make the 128k cases more sensitive to per-rank stragglers.
- CUDA graph coverage is not uniform on long-context/high-concurrency cases. Server logs show long prefill steps with `cuda graph: False`, and some decode phases also fall back when batch/shape does not fit captured graphs. Since `--cuda-graph-max-bs=16`, c32/c64 can still suffer from uncaptured or fragmented shapes.
- `128k_500` tail behavior suggests straggler amplification across DP/TP ranks. Aggregate throughput improves with concurrency, but TTFT and TPOT tails grow, indicating that requests wait behind long prefill/decode phases rather than benefiting cleanly from the extra GPUs.

Profiling is intentionally deferred until the PD-separated report is completed, per task order. If TP8 remains worse after PD testing, the first profiling target should be `32k_500_c1` and `128k_500_c32/c64`, with focus on prefill kernels, NCCL/NVSHMEM collectives, CUDA graph hit rate, and DP scheduler imbalance.

## Container Creation Command

`/home/admin/0528/scripts/recreate_sgl_container.sh`

```bash
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
```

## Deployment Commands

### TP4 + DP4 + DPA + NVLS/MNNVL

`/home/admin/0528/commands/tp4_dp4_dpa_nvl_cg16_server_command.sh`

```bash
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
```

### TP8 + DP8 + DPA + NVLS/MNNVL

`/home/admin/0528/commands/tp8_dp8_dpa_nvl_cg16_rank0_server_command.sh`

```bash
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
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.56.160.38:29500 \
  --dp-size 8 \
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
  --pre-warm-nccl \
  --cuda-graph-max-bs 16
```

`/home/admin/0528/commands/tp8_dp8_dpa_nvl_cg16_rank1_server_command.sh`

```bash
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
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.56.160.38:29500 \
  --dp-size 8 \
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
  --pre-warm-nccl \
  --cuda-graph-max-bs 16
```
