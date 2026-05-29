# TP=8 Pure TP NVLS/MNNVL Rerun Report

- Generated at: 2026-05-29 15:00:21, timezone Asia/Shanghai
- Nodes: rank0 `10.56.160.38`, rank1 `10.56.160.40`
- Mode: `TP=8 pure TP`, `nnodes=2`, no `--dp-size`, no `--enable-dp-attention`
- Benchmark: `random`, cases `4k_500`, `32k_500`, `128k_500`, concurrency `1,2,4,8,16,32,64`, `num_prompts=64`, `warmup_requests=1`
- NVLS/MNNVL: enabled by env and `--enable-nccl-nvls`; server log contains `NVLS multicast support is available` and `via P2P/MNNVL`.

## Result Status

- Completed: `7` benchmark JSON files, all are `4k_500` c1-c64.
- Failed: `1` benchmark, first failure is `32k_500_c1` during warmup.
- Not run: `13` remaining cases because the server became unhealthy after the first 32k failure.
- Best `4k_500` pure TP output throughput: c16 = `654.91 tok/s`.

## 4k_500 Throughput

| conc | pureTP8 out tok/s | pureTP8 mean TTFT ms | pureTP8 p99 TPOT ms | TP4+DP4 out tok/s | pure/TP4 | TP8+DP8 out tok/s | pure/TP8DP8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.90 | 722 | 12.3 | 90.26 | 0.74 | 78.04 | 0.86 |
| 2 | 144.43 | 191 | 16.3 | 180.72 | 0.80 | 170.34 | 0.85 |
| 4 | 254.38 | 204 | 18.3 | 333.41 | 0.76 | 305.24 | 0.83 |
| 8 | 424.49 | 236 | 20.3 | 545.00 | 0.78 | 474.82 | 0.89 |
| 16 | 654.91 | 316 | 30.4 | 813.48 | 0.81 | 725.79 | 0.90 |
| 32 | 187.82 | 1703 | 177.2 | 1212.62 | 0.15 | 1066.80 | 0.18 |
| 64 | 275.87 | 669 | 200.8 | 2403.60 | 0.11 | 2149.17 | 0.13 |

## Failure Point

- The official `tp8_puretp_nvl_cg16` run completed `4k_500` c1, c2, c4, c8, c16, c32 and c64, then stopped at `32k_500_c1`.
- Client-side failure in `/home/admin/0528/raw/tp8_puretp_nvl_cg16_32k_500_c1.log`: `TransferEncodingError: Not enough data to satisfy transfer length header`, `bench_rc=1`, `health_http_after=000`.
- Server-side failure in `/home/admin/0528/server_logs/tp8_puretp_nvl_cg16_rank0_server.log`: FlashMLA decode scheduling error `get_decoding_sched_meta.cu:111 invalid argument`; scheduler crash detected: `True`.
- NVLS proof from the same run: `NVLS multicast support is available` = `True`, `via P2P/MNNVL` occurrences = `447`.

## Retry Experiments

- `--disable-cuda-graph` retry (`tp8_puretp_nvl_nocg`) still failed at `32k_500_c1` with the same FlashMLA `get_decoding_sched_meta.cu:111 invalid argument`; this rules out CUDA graph replay as the direct cause.
- `--decode-attention-backend triton` retry failed during startup with DeepSeek V4 KV pool incompatibility (`NotImplementedError` present: `True`).

## Analysis

- This rerun confirms the TP=8 pure TP path can serve short-context `4k_500`, but it is not stable for `32k_500` under the current FlashMLA/DeepSeek V4 pure TP configuration.
- The failure is unlikely to be caused by NVL/MNNVL not taking effect: NCCL initialized with NVLS support and P2P/MNNVL channels before the crash.
- The failure is also unlikely to be caused by CUDA graph max batch size, because the no-CUDA-graph retry reproduces the same FlashMLA decode scheduler invalid argument.
- The most likely root cause is a FlashMLA decode scheduling constraint or bug triggered by long prefill length in pure TP=8 for DeepSeek V4, rather than a generic cross-node bandwidth problem.
- A backend swap to generic Triton decode is not a viable workaround for this model in this container because DeepSeek V4 KV pool does not implement the buffer path expected by that backend.

## Artifacts

- Summary CSV: `/home/admin/0528/summary_tp8_puretp.csv`
- Raw results: `/home/admin/0528/raw/tp8_puretp_nvl_cg16_*`
- Official logs: `/home/admin/0528/server_logs/tp8_puretp_nvl_cg16_rank0_server.log`, `/home/admin/0528/server_logs/tp8_puretp_nvl_cg16_rank1_server.log`
- Retry logs: `/home/admin/0528/server_logs/tp8_puretp_nvl_nocg_rank0_server.log`, `/home/admin/0528/server_logs/tp8_puretp_nvl_cg16_decode_triton_rank0_server.log`

## Rank0 Command

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
  --dist-init-addr 10.56.160.38:29540 \
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

## Rank1 Command

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
  --dist-init-addr 10.56.160.38:29540 \
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
