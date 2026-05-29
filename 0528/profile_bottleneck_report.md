# L20A SGLang 32k_500_c1 Profiling 与瓶颈分析

## Profiling 范围

- 目标 case：`32k_500_c1`
- 正式性能对比仍以 64 prompts 的矩阵结果为准；profiling 为降低 trace 体积，使用 `num_prompts=8`、`max_concurrency=1`、`profile_start_step=1`、`profile_steps=5`。
- 采集目录：
  - TP4：`/home/admin/0528/profiles/tp4_32k_c1/`
  - TP8：`/home/admin/0528/profiles/tp8_32k_c1/`
  - PD prefill 辅助 profile：`/home/admin/0528/profiles/pd_prefill_32k_c1/`
- 解析汇总：`/home/admin/0528/profiles/profile_parse_summary.json`

## 正式矩阵中的现象

| 模式 | 32k_500_c1 output tok/s | mean TTFT(ms) | P99 TPOT(ms) |
| --- | --- | --- | --- |
| TP4+DP4+DPA+NVLS/MNNVL | 84.68 | 833 | 8.6 |
| TP8+DP8+DPA+NVLS/MNNVL | 66.94 | 1602 | 8.7 |
| PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL | 69.45 | 1499 | 8.8 |

TP8/PD 的 TPOT 与 TP4 接近，但 TTFT 明显更高，说明问题主要在 prefill、切块调度、DP/EP 同步，而不是 decode 阶段单 token 生成速度。

## Profiling 观察

| Profile | bench output tok/s | mean TTFT(ms) | 说明 |
| --- | --- | --- | --- |
| TP4 32k_c1 profile | 58.05 | 2424 | profiler 开销下的参考值，不用于正式性能比较 |
| TP8 32k_c1 profile | 57.21 | 2484 | profiler 开销下 TP4/TP8 接近，主要用于 trace 对比 |
| PD prefill 32k_c1 profile | 9.89 | 25742 | prefill rank0 trace 极大，rank1 `/start_profile` 返回 404，作为辅助证据 |

代表性 trace 解析结果：

| 模式 | 代表 trace | GPU kernel 总时长 | Top GPU kernel | NCCL kernel | Gloo/user annotation |
| --- | --- | --- | --- | --- | --- |
| TP4 | local TP0 | 1999.9 ms | DeepGEMM MegaMoE 1988.9 ms | allgather/allreduce 约 7.2 ms | gloo broadcast/all_gather 约 1882 ms |
| TP8 | local TP0 | 2121.3 ms | DeepGEMM MegaMoE 2108.9 ms | allreduce/allgather 约 11.9 ms | gloo broadcast/all_gather 约 2038 ms |
| PD prefill | local TP0 | 1551.3 ms | DeepGEMM MegaMoE 1541.3 ms | allreduce/allgather 约 9.5 ms | gloo broadcast/all_gather 约 2673 ms |

## 判断

- 不是 NVLink/NCCL 原始带宽没有打通：服务日志已看到 `NCCL_MNNVL_ENABLE=1`、`NVLS multicast support is available`、`via P2P/MNNVL`，而 profiling 中 NCCL GPU kernel 时长为毫秒级，不是主耗时。
- 主 GPU 算子是 `deep_gemm::sm100_fp8_fp4_mega_moe_impl`。TP8 的 `ep_size=8`，MegaMoE 路径的 per-step kernel 时间不比 TP4 低，2 倍 GPU 没有换来对应吞吐扩展。
- Gloo broadcast/all_gather 的 user annotation 可到秒级，说明调度/控制同步开销明显。它走的是控制面同步，不等同于 NVLink 数据面带宽。
- SGLang 启用 DP attention 后自动改写有效配置：TP4 的 `chunked_prefill_size=4096`，TP8/PD 的 `chunked_prefill_size=2048`。TP8 长上下文 prefill 切块更多，调度和同步次数更多，这是 TTFT 偏高的直接原因之一。
- PD 在中等并发提升明显，说明 P/D 分离可以缓解资源争用；但 c64 长上下文回落，说明 router 排队、KV transfer、decode 侧批处理上限会成为新瓶颈。

## 调优方向

1. 先做 chunk size 公平性实验：把 TP4 也固定到有效 `chunked_prefill_size=2048` 后重跑小矩阵，或在确认 MoE kernel 风险可控后尝试让 TP8 使用更大的有效 chunk。当前 CLI 传 `16384` 会被 SGLang 自动改写，不能只看启动命令。
2. 针对 Gloo/control-plane：检查是否有可减少 DP controller 同步频率的参数，或改用更合适的控制面网络路径；当前 traces 显示 Gloo 同步比 NCCL kernel 更值得优先看。
3. 针对 MegaMoE：优先比较 MegaMoE 的 token/rank 参数、EP size、以及可用 MoE backend 的小矩阵表现。profile 指向 MoE kernel/dispatch 路径，而不是 decode attention。
4. 针对 PD：分别 profile prefill 与 decode worker。当前 prefill rank0 已能 profile；rank1 endpoint 404，需要用 rank0 汇总或改启动方式暴露 profile endpoint。高并发下应重点看 router queue、KV transfer 和 decode worker 批处理。

