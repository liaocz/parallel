# L20A SGLang TP4 / TP8 / PD 性能测试报告（含 Profiling 与算子延迟）

## 测试范围

- 测试日期：2026-05-29
- 模型路径：`/home/admin/DeepSeek-V4-Flash`
- benchmark：`python3 -m sglang.bench_serving --dataset-name random`
- 输入/输出：`4k_500`、`32k_500`、`128k_500`
- 并发：`1,2,4,8,16,32,64`
- `num_prompts=64`，`cuda_graph_max_bs=16`
- 所有模式均开启 NVLS/MNNVL/NVLink 相关环境变量和 `--enable-nccl-nvls`。

## 结果文件

- 全量指标 CSV：`/home/admin/0528/summary_all_modes.csv`
- 横向对比 CSV：`/home/admin/0528/comparison_all_modes.csv`
- 原始 JSON/日志：`/home/admin/0528/raw/`
- 服务日志：`/home/admin/0528/server_logs/`
- 部署命令：`/home/admin/0528/commands/`
- Profiling 瓶颈分析：`/home/admin/0528/profile_bottleneck_report.md`
- 主要算子延迟明细：`/home/admin/0528/profile_operator_latency.md`
- Decode batch 采样明细：`/home/admin/0528/decode_batch_by_case.csv`
- Decode batch 采样汇总：`/home/admin/0528/decode_batch_summary.md`
- 本最终版报告：`/home/admin/0528/tp4_tp8_pd_full_report_with_profile.md`

## 部署拓扑

| 模式 | 节点 | GPU 数 | 命令文件 |
| --- | --- | --- | --- |
| TP4+DP4+DPA+NVLS/MNNVL | 10.56.160.38 | 4 | commands/tp4_dp4_dpa_nvl_cg16_server_command.sh |
| TP8+DP8+DPA+NVLS/MNNVL | 10.56.160.38, 10.56.160.40 | 8 | commands/tp8_dp8_dpa_nvl_cg16_rank0_server_command.sh, commands/tp8_dp8_dpa_nvl_cg16_rank1_server_command.sh |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | prefill: 10.56.160.38,10.56.160.40; decode: 10.56.160.36,10.56.160.34 | 16 | commands/pd_prefill_rank0_nvl_cg16_command.sh, commands/pd_prefill_rank1_nvl_cg16_command.sh, commands/pd_decode_rank0_nvl_cg16_command.sh, commands/pd_decode_rank1_nvl_cg16_command.sh, commands/pd_router_nvl_cg16_command.sh |

## 完整性校验

| 模式 | 期望 JSON | 实际 JSON | 完成状态 |
| --- | --- | --- | --- |
| TP4+DP4+DPA+NVLS/MNNVL | 21 | 21/21 | completed=64 for all cases |
| TP8+DP8+DPA+NVLS/MNNVL | 21 | 21/21 | completed=64 for all cases |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | 21 | 21/21 | completed=64 for all cases |

## 有效运行配置

启动命令中 TP4/TP8/PD 的主要性能参数保持一致，但 SGLang 在启用 DP attention 后会按 TP/EP 自动调整部分运行时参数：TP4 实际 `chunked_prefill_size=4096`，TP8/PD 实际 `chunked_prefill_size=2048`，且 `schedule_conservativeness` 实际为 `0.09`。这些是 server_args/server_info 中的有效值。

| 模式 | 参数 | 有效值 |
| --- | --- | --- |
| TP4+DP4+DPA+NVLS/MNNVL | tp_size | 4 |
| TP4+DP4+DPA+NVLS/MNNVL | dp_size | 4 |
| TP4+DP4+DPA+NVLS/MNNVL | nnodes | 1 |
| TP4+DP4+DPA+NVLS/MNNVL | node_rank | 0 |
| TP4+DP4+DPA+NVLS/MNNVL | disaggregation_mode | null |
| TP4+DP4+DPA+NVLS/MNNVL | moe_a2a_backend | megamoe |
| TP4+DP4+DPA+NVLS/MNNVL | ep_size | 4 |
| TP4+DP4+DPA+NVLS/MNNVL | enable_dp_attention | True |
| TP4+DP4+DPA+NVLS/MNNVL | enable_nccl_nvls | True |
| TP4+DP4+DPA+NVLS/MNNVL | cuda_graph_max_bs | 16 |
| TP4+DP4+DPA+NVLS/MNNVL | chunked_prefill_size | 4096 |
| TP4+DP4+DPA+NVLS/MNNVL | max_prefill_tokens | 16384 |
| TP4+DP4+DPA+NVLS/MNNVL | max_running_requests | 256 |
| TP4+DP4+DPA+NVLS/MNNVL | effective_max_running_requests_per_dp | 64 |
| TP4+DP4+DPA+NVLS/MNNVL | schedule_conservativeness | 0.09 |
| TP4+DP4+DPA+NVLS/MNNVL | quantization | fp8 |
| TP4+DP4+DPA+NVLS/MNNVL | kv_cache_dtype | fp8_e4m3 |
| TP4+DP4+DPA+NVLS/MNNVL | pre_warm_nccl | False |
| TP8+DP8+DPA+NVLS/MNNVL | tp_size | 8 |
| TP8+DP8+DPA+NVLS/MNNVL | dp_size | 8 |
| TP8+DP8+DPA+NVLS/MNNVL | nnodes | 2 |
| TP8+DP8+DPA+NVLS/MNNVL | node_rank | 0 |
| TP8+DP8+DPA+NVLS/MNNVL | disaggregation_mode | null |
| TP8+DP8+DPA+NVLS/MNNVL | moe_a2a_backend | megamoe |
| TP8+DP8+DPA+NVLS/MNNVL | ep_size | 8 |
| TP8+DP8+DPA+NVLS/MNNVL | enable_dp_attention | True |
| TP8+DP8+DPA+NVLS/MNNVL | enable_nccl_nvls | True |
| TP8+DP8+DPA+NVLS/MNNVL | cuda_graph_max_bs | 16 |
| TP8+DP8+DPA+NVLS/MNNVL | chunked_prefill_size | 2048 |
| TP8+DP8+DPA+NVLS/MNNVL | max_prefill_tokens | 16384 |
| TP8+DP8+DPA+NVLS/MNNVL | max_running_requests | 256 |
| TP8+DP8+DPA+NVLS/MNNVL | effective_max_running_requests_per_dp | 32 |
| TP8+DP8+DPA+NVLS/MNNVL | schedule_conservativeness | 0.09 |
| TP8+DP8+DPA+NVLS/MNNVL | quantization | fp8 |
| TP8+DP8+DPA+NVLS/MNNVL | kv_cache_dtype | fp8_e4m3 |
| TP8+DP8+DPA+NVLS/MNNVL | pre_warm_nccl | True |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | tp_size | 8 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | dp_size | 8 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | nnodes | 2 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | node_rank | 0 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | disaggregation_mode | decode |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | moe_a2a_backend | megamoe |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | ep_size | 8 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | enable_dp_attention | True |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | enable_nccl_nvls | True |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | cuda_graph_max_bs | 16 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | chunked_prefill_size | 2048 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | max_prefill_tokens | 16384 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | max_running_requests | 256 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | effective_max_running_requests_per_dp | 32 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | schedule_conservativeness | 0.09 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | quantization | fp8 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | kv_cache_dtype | fp8_e4m3 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | pre_warm_nccl | True |

## 峰值与扩展性

| Case | 模式 | 峰值并发 | 峰值 output tok/s | 峰值每 GPU tok/s | c1 output | c64 output | c64/c1 | c64 TTFT(ms) | c64 P99 TPOT(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4k_500 | TP4+DP4+DPA+NVLS/MNNVL | c64 | 2403.60 | 600.90 | 90.26 | 2403.60 | 26.63 | 914 | 20.3 |
| 4k_500 | TP8+DP8+DPA+NVLS/MNNVL | c64 | 2149.17 | 268.65 | 78.04 | 2149.17 | 27.54 | 1468 | 244.9 |
| 4k_500 | PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | c64 | 2351.55 | 146.97 | 76.64 | 2351.55 | 30.68 | 1781 | 11.9 |
| 32k_500 | TP4+DP4+DPA+NVLS/MNNVL | c64 | 1205.63 | 301.41 | 84.68 | 1205.63 | 14.24 | 3324 | 889.5 |
| 32k_500 | TP8+DP8+DPA+NVLS/MNNVL | c64 | 1214.97 | 151.87 | 66.94 | 1214.97 | 18.15 | 3742 | 941.5 |
| 32k_500 | PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | c64 | 1105.94 | 69.12 | 69.45 | 1105.94 | 15.92 | 4163 | 10.9 |
| 128k_500 | TP4+DP4+DPA+NVLS/MNNVL | c64 | 307.70 | 76.92 | 48.68 | 307.70 | 6.32 | 17321 | 6291.3 |
| 128k_500 | TP8+DP8+DPA+NVLS/MNNVL | c64 | 291.07 | 36.38 | 32.73 | 291.07 | 8.89 | 19781 | 1361.0 |
| 128k_500 | PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | c64 | 247.50 | 15.47 | 32.46 | 247.50 | 7.62 | 20776 | 770.5 |

## 分 Case 明细

### 4k_500

| 并发 | TP4 out | TP8 out | TP8/TP4 | PD out | PD/TP4 | TP4 TTFT | TP8 TTFT | PD TTFT | TP4 P99 TPOT | TP8 P99 TPOT | PD P99 TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 90.26 | 78.04 | 0.86 | 76.64 | 0.85 | 662 | 1080 | 1169 | 8.5 | 8.6 | 8.9 |
| 2 | 180.72 | 170.34 | 0.94 | 175.83 | 0.97 | 306 | 399 | 690 | 14.4 | 16.3 | 9.0 |
| 4 | 333.41 | 305.24 | 0.92 | 335.70 | 1.01 | 190 | 294 | 699 | 14.9 | 18.9 | 10.0 |
| 8 | 545.00 | 474.82 | 0.87 | 680.27 | 1.25 | 232 | 309 | 373 | 16.9 | 23.3 | 12.2 |
| 16 | 813.48 | 725.79 | 0.89 | 1076.63 | 1.32 | 414 | 418 | 766 | 25.5 | 67.0 | 15.3 |
| 32 | 1212.62 | 1066.80 | 0.88 | 1771.03 | 1.46 | 608 | 881 | 748 | 31.5 | 173.9 | 14.0 |
| 64 | 2403.60 | 2149.17 | 0.89 | 2351.55 | 0.98 | 914 | 1468 | 1781 | 20.3 | 244.9 | 11.9 |

### 32k_500

| 并发 | TP4 out | TP8 out | TP8/TP4 | PD out | PD/TP4 | TP4 TTFT | TP8 TTFT | PD TTFT | TP4 P99 TPOT | TP8 P99 TPOT | PD P99 TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 84.68 | 66.94 | 0.79 | 69.45 | 0.82 | 833 | 1602 | 1499 | 8.6 | 8.7 | 8.8 |
| 2 | 142.32 | 99.84 | 0.70 | 137.03 | 0.96 | 727 | 1549 | 1516 | 29.4 | 41.9 | 9.6 |
| 4 | 203.84 | 142.92 | 0.70 | 266.10 | 1.31 | 777 | 1429 | 1543 | 43.6 | 141.0 | 9.2 |
| 8 | 289.34 | 194.36 | 0.67 | 497.81 | 1.72 | 776 | 1434 | 1520 | 153.5 | 97.7 | 10.3 |
| 16 | 414.72 | 309.76 | 0.75 | 711.28 | 1.72 | 1010 | 1257 | 2429 | 349.6 | 117.5 | 11.1 |
| 32 | 645.64 | 552.20 | 0.86 | 1040.09 | 1.61 | 1482 | 1792 | 3294 | 333.3 | 479.3 | 10.3 |
| 64 | 1205.63 | 1214.97 | 1.01 | 1105.94 | 0.92 | 3324 | 3742 | 4163 | 889.5 | 941.5 | 10.9 |

### 128k_500

| 并发 | TP4 out | TP8 out | TP8/TP4 | PD out | PD/TP4 | TP4 TTFT | TP8 TTFT | PD TTFT | TP4 P99 TPOT | TP8 P99 TPOT | PD P99 TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 48.68 | 32.73 | 0.67 | 32.46 | 0.67 | 2986 | 5476 | 5579 | 8.8 | 9.0 | 9.0 |
| 2 | 64.94 | 41.72 | 0.64 | 62.28 | 0.96 | 3170 | 5752 | 5774 | 46.1 | 71.3 | 9.9 |
| 4 | 94.85 | 56.93 | 0.60 | 107.60 | 1.13 | 2948 | 6235 | 6765 | 95.9 | 144.3 | 9.3 |
| 8 | 127.10 | 78.62 | 0.62 | 153.10 | 1.20 | 3113 | 5799 | 9436 | 318.9 | 125.6 | 9.7 |
| 16 | 165.17 | 136.44 | 0.83 | 235.37 | 1.42 | 3528 | 5575 | 10194 | 1236.4 | 207.7 | 46.5 |
| 32 | 194.06 | 165.34 | 0.85 | 239.80 | 1.24 | 5947 | 8317 | 18833 | 2929.3 | 1984.4 | 248.0 |
| 64 | 307.70 | 291.07 | 0.95 | 247.50 | 0.80 | 17321 | 19781 | 20776 | 6291.3 | 1361.0 | 770.5 |

## 结论

- TP8 相比 TP4 在低/中并发没有体现 2 倍 GPU 的吞吐扩展，`4k_500` 全并发基本为 TP4 的 0.86-0.94 倍，`32k_500` 在 c1-c32 为 0.67-0.86 倍，`128k_500` 在 c1-c32 为 0.60-0.85 倍；只有高并发 c64 在 32k/128k 接近 TP4。
- 按每 GPU 效率看，TP8 明显低于 TP4。即便总吞吐接近，TP8 使用 8 卡，TP4 使用 4 卡，因此 TP8 的单位 GPU output tok/s 大多只有 TP4 的 30%-50% 左右。
- PD 模式在中等并发表现更好：`4k_500` c8-c32 为 TP4 的 1.25-1.46 倍，`32k_500` c4-c32 为 1.31-1.72 倍，`128k_500` c4-c32 为 1.13-1.42 倍；但 c64 下 PD 在 32k/128k 回落，说明高并发长上下文下 PD 路由/KV 传输/排队开销开始显著。
- TP8/PD 长上下文低并发弱于 TP4 的主要现象是 TTFT 明显增加，而 TPOT 中位数多在 8-9ms 附近，说明瓶颈更偏 prefill、调度切块、跨节点/DP attention 协同，而不是纯 decode token 间隔。
- NVLS/MNNVL 已启用：服务日志出现 `NCCL_MNNVL_ENABLE=1`、`NVLS multicast support is available`、`via P2P/MNNVL`；PD 日志还包含 Mooncake NVLINK/MNNVL 相关环境配置。当前低扩展效率不能简单归因为 NVLink 未启用。

## 瓶颈判断

- 有效 `chunked_prefill_size` 不一致是最直接的可见因素：TP4 自动降为 4096，TP8/PD 自动降为 2048。长上下文 prefill 被切得更碎，TTFT 增加，调度和通信启动次数增加。
- TP8 的 EP/DP 规模更大，MegaMoE 下 `ep_size=8`，跨节点 TP/DP attention 协同会带来更多同步和路由开销。日志已确认 MNNVL/NVLS 路径可用，因此需要进一步用 profiling 区分 MoE dispatch/all-to-all、attention prefill、NCCL allreduce/allgather 的占比。
- PD 在 c4-c32 改善明显，说明把 prefill/decode 分离后能减少部分资源争用；c64 回落说明 router 排队、KV transfer 或 decode 侧批处理上限成为新的瓶颈。

## Decode Batch 采样

Decode batch 取自服务日志中的 `Decode batch, #running-req`，表示每个 scheduler/DP rank 采样到的 decode running requests，不是全局客户端并发。SGLang 按 `decode_log_interval` 周期采样打印，因此该值用于观察趋势，不保证覆盖每一个 decode step。

| 模式 | Case | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4+DP4+DPA | 4k_500 | 1 | 2 | 3 | 4 | 6 | 11 | 16 |
| TP4+DP4+DPA | 32k_500 | 1 | 2 | 2 | 3 | 6 | 10 | 16 |
| TP4+DP4+DPA | 128k_500 | 1 | 2 | 2 | 4 | 6 | 10 | 16 |
| TP8+DP8+DPA | 4k_500 | 1 | 1 | 2 | 3 | 4 | 6 | 8 |
| TP8+DP8+DPA | 32k_500 | 1 | 1 | 2 | 2 | 3 | 7 | 8 |
| TP8+DP8+DPA | 128k_500 | 1 | 1 | 2 | 2 | 4 | 6 | 8 |
| PD decode TP8+DP8 | 4k_500 | 1 | 1 | 1 | 2 | 3 | 5 | 7 |
| PD decode TP8+DP8 | 32k_500 | 1 | 1 | 1 | 2 | 3 | 4 | 6 |
| PD decode TP8+DP8 | 128k_500 | 1 | 1 | 1 | 2 | 2 | 3 | 3 |

观察：
- TP4 单 scheduler 采样到的最大 decode batch 可到 16；TP8 单 scheduler 最大到 8，符合 DP rank 更多后每个 rank 分到的 decode 请求更少的现象。
- PD decode 在长上下文下 batch 更小，`128k_500_c64` 最大采样只有 3。结合 PD c64 长上下文吞吐回落，瓶颈更可能出现在 router/KV transfer/排队和 decode 分发节奏，而不是 decode kernel 单步 TPOT。
- 该采样也解释了为什么 TP8/PD 增卡后单位 GPU 效率低：全局并发固定时，DP rank 越多，单 rank decode batch 越小，kernel 利用率和 cuda graph batch 利用率更难打满。

## Profiling 结果与算子延迟

Profiling 针对 `32k_500_c1` 进行，用于解释 TP8 相比 TP4 扩展效率偏低的问题。正式性能结论仍以完整矩阵 `num_prompts=64` 的结果为准；profiling 为降低 trace 体积，使用 `num_prompts=8`、`max_concurrency=1`、`profile_start_step=1`、`profile_steps=5`。

采集目录：
- TP4：`/home/admin/0528/profiles/tp4_32k_c1/`
- TP8：`/home/admin/0528/profiles/tp8_32k_c1/`
- PD prefill 辅助 profile：`/home/admin/0528/profiles/pd_prefill_32k_c1/`
- 解析汇总：`/home/admin/0528/profiles/profile_parse_summary.json`
- 主要算子延迟明细：`/home/admin/0528/profile_operator_latency.md`

### 正式矩阵中的 32k_500_c1 现象

| 模式 | 32k_500_c1 output tok/s | mean TTFT(ms) | P99 TPOT(ms) |
| --- | ---: | ---: | ---: |
| TP4+DP4+DPA+NVLS/MNNVL | 84.68 | 833 | 8.6 |
| TP8+DP8+DPA+NVLS/MNNVL | 66.94 | 1602 | 8.7 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | 69.45 | 1499 | 8.8 |

TP8/PD 的 TPOT 与 TP4 基本接近，但 TTFT 明显更高。因此低并发长上下文的差距主要来自 prefill、切块调度、DP/EP 同步和控制面等待，而不是 decode 阶段单 token 生成间隔。

### Profile 场景

| 场景 | 已解析 trace | chunked_prefill_size | profile output tok/s | mean TTFT(ms) | mean TPOT(ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | 4 | 4096 | 58.05 | 2423.61 | 8.53 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | 4 | 2048 | 57.21 | 2484.44 | 8.56 |
| PD prefill TP8+DP8, 32k_500_c1 | 1 | 2048 | 9.89 | 25741.53 | 8.43 |

说明：
- 表中的算子延迟是 profiler 窗口内累计时长，单位 ms，不是单次 kernel 调用耗时。
- `max/rank` 更接近单个 rank 的关键路径压力；`sum/ranks` 是已解析 trace 的累计和，会把并行 rank 相加，不能直接等同 wall time。
- PD 当前只解析到 prefill rank0 trace，其他 PD trace 文件很大，因此 PD profile 只作为辅助证据。

### Category Totals

| 场景 | 类别 | traces | min(ms) | avg(ms) | max(ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | GPU kernel | 4 | 573.2 | 1645.1 | 2004.4 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | GPU annotation | 4 | 2003.5 | 2006.8 | 2010.3 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | CPU/user annotation | 4 | 2015.2 | 2037.9 | 2046.8 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | CPU op | 4 | 42.8 | 305.8 | 1073.5 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | CUDA runtime API | 4 | 6.6 | 65.7 | 241.9 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | GPU kernel | 4 | 277.5 | 1660.4 | 2122.2 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | GPU annotation | 4 | 2131.3 | 2135.0 | 2137.1 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | CPU/user annotation | 4 | 2138.5 | 2161.5 | 2169.8 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | CPU op | 4 | 47.8 | 323.8 | 1145.3 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | CUDA runtime API | 4 | 6.9 | 66.3 | 244.1 |
| PD prefill TP8+DP8, 32k_500_c1 | GPU kernel | 1 | 1551.3 | 1551.3 | 1551.3 |
| PD prefill TP8+DP8, 32k_500_c1 | GPU annotation | 1 | 1564.5 | 1564.5 | 1564.5 |
| PD prefill TP8+DP8, 32k_500_c1 | CPU/user annotation | 1 | 2796.6 | 2796.6 | 2796.6 |
| PD prefill TP8+DP8, 32k_500_c1 | CPU op | 1 | 462.5 | 462.5 | 462.5 |
| PD prefill TP8+DP8, 32k_500_c1 | CUDA runtime API | 1 | 7.2 | 7.2 | 7.2 |

### TP4 32k_500_c1 主要算子

| 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | ---: | ---: | ---: |
| DeepGEMM MegaMoE fp8/fp4 | 1993.4 | 1546.3 | 6185.1 |
| Sparse MLA decode attention fp8 | 88.7 | 22.2 | 88.7 |
| DeepGEMM dense fp8/fp4 GEMM | 81.4 | 20.3 | 81.4 |
| MHC post tilelang | 18.4 | 4.6 | 18.4 |
| Fused Q norm + RoPE | 17.2 | 4.3 | 17.2 |
| DeepSeek RoPE | 15.3 | 3.8 | 15.3 |
| Per-token group quant 8bit | 15.3 | 3.8 | 15.3 |
| NCCL AllGather kernel | 4.0 | 3.0 | 12.0 |
| nvJet / TRT-LLM fused kernel | 3.4 | 2.5 | 10.2 |
| FlashInfer one-shot all-reduce push | 3.2 | 2.4 | 9.6 |

TP4 的 GPU 主耗时高度集中在 `DeepGEMM MegaMoE fp8/fp4`，NCCL AllGather 与 all-reduce push 均为毫秒级。控制面上，`gloo:all_gather` max/rank 约 1882 ms，`gloo:broadcast` max/rank 约 1854 ms，说明 scheduler/DP/EP 控制同步在 profile 窗口内已经是秒级。

### TP8 32k_500_c1 主要算子

| 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | ---: | ---: | ---: |
| DeepGEMM MegaMoE fp8/fp4 | 2109.9 | 1602.3 | 6409.2 |
| Sparse MLA decode attention fp8 | 45.2 | 11.3 | 45.2 |
| DeepGEMM dense fp8/fp4 GEMM | 44.9 | 11.2 | 44.9 |
| MHC post tilelang | 9.9 | 2.5 | 9.9 |
| NCCL AllReduce kernel | 8.9 | 6.7 | 26.7 |
| Fused Q norm + RoPE | 8.8 | 2.2 | 8.8 |
| DeepSeek RoPE | 7.9 | 2.0 | 7.9 |
| MHC pre big-fuse norm tilelang | 7.9 | 2.0 | 7.9 |
| NCCL AllGather kernel | 3.0 | 2.2 | 8.9 |
| MegaMoE pre-dispatch | 0.4 | 0.3 | 1.1 |

TP8 的 `DeepGEMM MegaMoE fp8/fp4` max/rank 约 2110 ms，比 TP4 的约 1993 ms 更高。TP8 的 NCCL AllReduce max/rank 约 8.9 ms、AllGather max/rank 约 3.0 ms，仍不是 GPU 侧主耗时。控制面上，`gloo:all_gather` max/rank 约 2030 ms，`gloo:broadcast` max/rank 约 2008 ms，均高于 TP4 对应值。

### PD Prefill 32k_500_c1 主要算子

| 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | ---: | ---: | ---: |
| DeepGEMM MegaMoE fp8/fp4 | 1541.3 | 1541.3 | 1541.3 |
| NCCL AllReduce kernel | 7.1 | 7.1 | 7.1 |
| NCCL AllGather kernel | 2.3 | 2.3 | 2.3 |
| MegaMoE pre-dispatch | 0.4 | 0.4 | 0.4 |
| nvJet / TRT-LLM fused kernel | 0.1 | 0.1 | 0.1 |
| ATen vectorized elementwise | 0.0 | 0.0 | 0.0 |
| CUB DeviceScan | 0.0 | 0.0 | 0.0 |

PD prefill 的 GPU 主耗时同样集中在 `DeepGEMM MegaMoE fp8/fp4`。NCCL AllReduce/AllGather 仍为毫秒级；但控制面 `gloo:broadcast` 约 2094 ms，`gloo:all_gather` 约 579 ms，同时 bench 中 mean TTFT 达 25741 ms，说明 PD profile 这次采样受到 prefill worker、router、KV transfer 或 profile 启停路径的额外影响，只能作为定位方向，不能直接代表完整矩阵性能。

### Profiling 结论

- TP8 弱于 TP4 不能简单归因于 NVLink/NVLS 没启用。服务日志已确认 `NCCL_MNNVL_ENABLE=1`、`NVLS multicast support is available`、`via P2P/MNNVL`，profiling 中 NCCL GPU kernel 也只有毫秒到十几毫秒级。
- GPU 侧主瓶颈是 `DeepGEMM MegaMoE fp8/fp4`，TP8 的 MegaMoE max/rank 累计耗时不低于 TP4，说明扩到 TP8/EP8 后并没有把单 rank MoE 路径摊薄，反而可能受 expert routing、token/rank 分布和跨节点协同影响。
- 控制面和调度同步开销明显。Gloo `broadcast/all_gather` 在 TP4/TP8 都达到秒级，TP8 更高；这类开销走的是控制面同步，不等同于 NVLink/NVLS 数据面带宽。
- 有效 `chunked_prefill_size` 是直接差异：TP4 为 4096，TP8/PD 为 2048。`32k` 输入下 TP8 需要更多 prefill chunk，放大了 scheduler 调度、Gloo 同步、DP attention 协同和 MoE dispatch 次数，因此 TTFT 更高。
- Decode 不是当前低并发长上下文差距的主因。正式矩阵里 TP4/TP8/PD 的 c1 P99 TPOT 都在 8.6-8.8 ms 附近，profiling 中 mean TPOT 也在 8.4-8.6 ms 附近；差距主要体现在 TTFT。
- PD 中等并发提升明显，说明 P/D 分离能减少 prefill/decode 资源争用；但 c64 长上下文回落，并且 decode batch 采样偏小，说明高并发长上下文下 router 排队、KV transfer 和 decode worker 批处理上限会成为新瓶颈。

### 后续调优建议

1. 做 chunk size 公平性实验：把 TP4 有效 chunk 固定到 2048，或确认安全后尝试让 TP8 使用更大的有效 chunk，单独验证 chunking 对 TTFT 的影响。
2. 针对 MegaMoE 做小矩阵对比：比较 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`、EP size、MoE backend 和 token/rank 分布，确认 TP8 下 MegaMoE kernel 不扩展的原因。
3. 针对控制面同步继续 profile：优先看 Gloo `broadcast/all_gather` 的触发频率、scheduler 同步点和 DP controller 路径，而不是只盯 NCCL bandwidth。
4. PD 需要拆分 profile prefill、decode、router 和 KV transfer。当前 prefill rank0 能说明方向，但不足以把 PD 高并发回落归因到单一算子。

## 容器启动命令

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

## 服务部署命令

### TP4+DP4+DPA+NVLS/MNNVL

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

### TP8+DP8+DPA+NVLS/MNNVL

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

### PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL

`/home/admin/0528/commands/pd_prefill_rank0_nvl_cg16_command.sh`

```bash
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
  --port 30000 \
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.56.160.38:29520 \
  --dp-size 8 \
  --enable-dp-attention \
  --disaggregation-mode prefill \
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
  --cuda-graph-max-bs 16
```

`/home/admin/0528/commands/pd_prefill_rank1_nvl_cg16_command.sh`

```bash
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
  --port 30000 \
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.56.160.38:29520 \
  --dp-size 8 \
  --enable-dp-attention \
  --disaggregation-mode prefill \
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
  --cuda-graph-max-bs 16
```

`/home/admin/0528/commands/pd_decode_rank0_nvl_cg16_command.sh`

```bash
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
  --node-rank 0 \
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
```

`/home/admin/0528/commands/pd_decode_rank1_nvl_cg16_command.sh`

```bash
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
```

`/home/admin/0528/commands/pd_router_nvl_cg16_command.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill http://10.56.160.38:30000 8998 \
  --decode http://10.56.160.36:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --worker-startup-timeout-secs 1800 \
  --worker-startup-check-interval 10 \
  --request-timeout-secs 7200 \
  --max-concurrent-requests 512 \
  --queue-size 2048
```

## 失败尝试归档

- `failed_attempts/pd_deepep_65536_assert`：DeepEP low-latency dispatcher token 限制断言。
- `failed_attempts/pd_deepep_ep8_weight_mismatch`：DeepEP EP8 与当前 FP8 权重 shape 不匹配。
- `failed_attempts/pd_bench_host_python_missing`：宿主机 Python 缺少 sglang 模块，benchmark 后续改在容器内执行。

## TP=8 Pure TP 重跑补充

- 补充报告：`/home/admin/0528/tp8_puretp_rerun_report.md`
- 汇总 CSV：`/home/admin/0528/summary_tp8_puretp.csv`
- 部署：2 机 `TP=8 pure TP`，rank0 `10.56.160.38`，rank1 `10.56.160.40`，未启用 DP/DPA；NVLS/MNNVL 环境变量和 `--enable-nccl-nvls` 均开启，`--cuda-graph-max-bs 16` 保持一致。
- 结果：`4k_500` 的 c1、c2、c4、c8、c16、c32、c64 全部完成；最佳为 c16，输出吞吐 `654.91 tok/s`。c32/c64 出现明显退化，分别为 `187.82 tok/s` 和 `275.87 tok/s`。
- 失败点：进入 `32k_500_c1` warmup 后服务崩溃，client 侧为 `TransferEncodingError` 且 `health_http_after=000`；server 侧为 FlashMLA decode scheduling 报错 `get_decoding_sched_meta.cu:111 invalid argument`，scheduler exit code 1。
- NVL 结论：日志中存在 `NVLS multicast support is available` 和大量 `via P2P/MNNVL`，因此本次失败不是 NVLS/MNNVL 未生效导致。
- 排障结论：`--disable-cuda-graph` 重试仍在 `32k_500_c1` 复现同一 FlashMLA 报错，说明不是 CUDA graph replay 直接导致；`--decode-attention-backend triton` 启动阶段因 DeepSeek V4 KV pool 不兼容报 `NotImplementedError`，不能作为当前容器内的直接替代方案。
