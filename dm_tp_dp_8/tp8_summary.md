# TP=8 SGLang 测试结果总结

## 测试口径

- 目标目录：`/home/admin/parallel/dm_tp_dp_8/`。
- 主结果口径：`tp8_dp8_dpa_megamoe`，两机 `10.56.160.38` + `10.56.160.40`，`tp_size=8`，`dp_size=8`，`ep_size=8`，启用 `enable_dp_attention` 与 `enable_nccl_nvls`。
- 模型路径：`/home/admin/DeepSeek-V4-Flash/`。
- benchmark：`python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name random --num-prompts 100`，输入/输出长度按 case 替换，并发为 `1,4,8,16,32,64,128`。
- 主服务从 `server_info` 看使用 `attention_backend=dsv4`，`moe_a2a_backend=megamoe`，`kv_cache_dtype=fp8_e4m3`，`max_total_tokens=202752`。

## 结论

- TP=8 + DP=8 + DPA + NVLS/MNNVL 的主配置可完成 `4k_500` 与 `32k_500` 全并发矩阵。
- `128k_500` 只完成到并发 32；并发 64 在 warmup 阶段连接服务失败，说明服务已退出或不可用；并发 128 没有成功结果。
- 输出吞吐随并发上升明显，但长上下文下 decode 吞吐下降明显，`128k_500` 在并发 32 达到本轮最高 `103.57 tok/s`，继续提高并发不稳定。
- 纯 TP=8 的尝试只留下少量 `4k_500` 小并发结果，随后在 flash-mla decode schedule meta 触发 CUDA invalid argument，不能作为完整可用矩阵。
- 与 TP4 `tp4_nvl_off` 基线相比，本轮 TP=8 DP 配置没有表现出吞吐优势；短上下文差距最大，长上下文也未超过 TP4。对比数据已单独保存到 `tp8_vs_tp4_nvl_off_comparison.csv`。

## 主矩阵峰值

| case | 完成并发 | 最佳并发 | output tok/s | total tok/s | mean TTFT ms | mean TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 4k_500 | 1, 4, 8, 16, 32, 64, 128 | 128 | 2089.86 | 19622.43 | 2767.04 | 56.37 |
| 32k_500 | 1, 4, 8, 16, 32, 64, 128 | 128 | 810.74 | 53138.93 | 9152.75 | 135.88 |
| 128k_500 | 1, 4, 8, 16, 32 | 32 | 103.57 | 28283.48 | 38029.03 | 121.80 |

## Output Throughput 矩阵

| concurrency | 4k_500 | 32k_500 | 128k_500 |
| --- | --- | --- | --- |
| 1 | 85.79 | 70.37 | 32.19 |
| 4 | 304.45 | 147.19 | 52.85 |
| 8 | 485.73 | 182.83 | 72.09 |
| 16 | 744.29 | 271.17 | 97.49 |
| 32 | 1065.37 | 394.47 | 103.57 |
| 64 | 1593.77 | 625.80 | FAIL/未完成 |
| 128 | 2089.86 | 810.74 | FAIL/未完成 |

## 主矩阵明细

| case | conc | req/s | output tok/s | total tok/s | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 E2E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4k_500 | 1 | 0.34 | 85.79 | 805.51 | 815.65 | 6242.41 | 8.42 | 8.53 | 6975.58 |
| 4k_500 | 4 | 1.22 | 304.45 | 2858.58 | 316.45 | 1995.10 | 11.85 | 25.14 | 8208.78 |
| 4k_500 | 8 | 1.95 | 485.73 | 4560.68 | 285.30 | 618.57 | 14.91 | 23.89 | 8792.01 |
| 4k_500 | 16 | 2.98 | 744.29 | 6988.44 | 369.74 | 1254.77 | 19.86 | 35.11 | 10306.72 |
| 4k_500 | 32 | 4.27 | 1065.37 | 10003.18 | 527.71 | 1490.35 | 26.10 | 48.62 | 14268.37 |
| 4k_500 | 64 | 6.39 | 1593.77 | 14964.51 | 1080.98 | 2291.40 | 35.17 | 57.67 | 13541.67 |
| 4k_500 | 128 | 8.38 | 2089.86 | 19622.43 | 2767.04 | 4724.92 | 56.37 | 325.84 | 11821.89 |
| 32k_500 | 1 | 0.28 | 70.37 | 4612.10 | 1442.44 | 3029.93 | 8.45 | 8.57 | 6786.73 |
| 32k_500 | 4 | 0.59 | 147.19 | 9647.48 | 1425.20 | 3085.41 | 21.21 | 52.24 | 15489.65 |
| 32k_500 | 8 | 0.73 | 182.83 | 11983.69 | 1456.55 | 3059.57 | 36.69 | 81.81 | 28176.49 |
| 32k_500 | 16 | 1.09 | 271.17 | 17773.23 | 1713.36 | 7912.53 | 52.49 | 98.57 | 34141.64 |
| 32k_500 | 32 | 1.58 | 394.47 | 25854.86 | 1995.27 | 6258.31 | 74.81 | 136.08 | 43283.98 |
| 32k_500 | 64 | 2.51 | 625.80 | 41017.37 | 4353.73 | 11799.25 | 111.31 | 408.48 | 37963.51 |
| 32k_500 | 128 | 3.25 | 810.74 | 53138.93 | 9152.75 | 26073.91 | 135.88 | 752.77 | 29363.17 |
| 128k_500 | 1 | 0.13 | 32.19 | 8789.82 | 5613.15 | 11986.50 | 8.60 | 8.86 | 14526.84 |
| 128k_500 | 4 | 0.21 | 52.85 | 14432.93 | 5967.63 | 12562.71 | 50.90 | 170.32 | 40156.11 |
| 128k_500 | 8 | 0.29 | 72.09 | 19687.16 | 6306.53 | 30563.44 | 86.46 | 173.00 | 60370.19 |
| 128k_500 | 16 | 0.39 | 97.49 | 26621.83 | 12745.32 | 61069.99 | 107.55 | 164.13 | 106173.87 |
| 128k_500 | 32 | 0.42 | 103.57 | 28283.48 | 38029.03 | 107619.87 | 121.80 | 213.11 | 148684.68 |

## TP4 对比抽样

完整对比见 `tp8_vs_tp4_nvl_off_comparison.csv`。ratio 为 TP8/TP4，低于 1 表示 TP8 本轮低于 TP4 基线。

| case | conc | TP8 output | TP4 output | output ratio | TP8 total | TP4 total | total ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4k_500 | 1 | 85.79 | 109.76 | 0.78 | 805.51 | 1030.62 | 0.78 |
| 4k_500 | 32 | 1065.37 | 1288.81 | 0.83 | 10003.18 | 12101.08 | 0.83 |
| 4k_500 | 64 | 1593.77 | 1905.07 | 0.84 | 14964.51 | 17887.42 | 0.84 |
| 4k_500 | 128 | 2089.86 | 3670.97 | 0.57 | 19622.43 | 34468.08 | 0.57 |
| 32k_500 | 1 | 70.37 | 89.04 | 0.79 | 4612.10 | 5836.22 | 0.79 |
| 32k_500 | 32 | 394.47 | 579.88 | 0.68 | 25854.86 | 38007.39 | 0.68 |
| 32k_500 | 64 | 625.80 | 876.50 | 0.71 | 41017.37 | 57449.22 | 0.71 |
| 32k_500 | 128 | 810.74 | 1212.08 | 0.67 | 53138.93 | 79444.64 | 0.67 |
| 128k_500 | 1 | 32.19 | 49.27 | 0.65 | 8789.82 | 13454.71 | 0.65 |
| 128k_500 | 32 | 103.57 | 210.92 | 0.49 | 28283.48 | 57599.32 | 0.49 |

## 纯 TP=8 部分结果

纯 TP=8 没有完成完整矩阵，以下仅保留原始观测值，不作为主结论。

| run_set | case | conc | output tok/s | total tok/s | source |
| --- | --- | --- | --- | --- | --- |
| tp8_pure_megamoe | 4k_500 | 1 | 63.52 | 515.78 | tp8_pure_megamoe_4k_500_c1.json |
| tp8_pure_megamoe | 4k_500 | 4 | 228.00 | 1851.26 | tp8_pure_megamoe_4k_500_c4.json |
| tp8_pure_nsa_megamoe | 4k_500 | 1 | 60.55 | 491.64 | tp8_pure_nsa_megamoe_4k_500_c1.json |
| tp8_pure_nsa_megamoe | 4k_500 | 4 | 222.04 | 1802.87 | tp8_pure_nsa_megamoe_4k_500_c4.json |

## 失败与未完成项

| category | source | case | conc | reason |
| --- | --- | --- | --- | --- |
| main_matrix_missing | tp8_dp8_dpa_megamoe_128k_500_c64.log | 128k_500 | 64 | bench warmup 阶段服务端口拒绝连接，服务已退出或不可用 |
| main_matrix_missing | - | 128k_500 | 128 | not_run_or_no_log |
| failed_attempt_log | tp8_dp8_dpa_rank0.failed_flashinfer_trtllm.log | - | - | flashinfer_trtllm 初始化阶段 AssertionError |
| failed_attempt_log | tp8_dp8_dpa_rank0.failed_flashinfer_trtllm_routed.log | - | - | flashinfer_trtllm routed 路径不支持 apply_routed_scaling_factor_on_output |
| failed_attempt_log | tp8_dp8_dpa_rank1.failed_flashinfer_trtllm_routed.log | - | - | flashinfer_trtllm routed 路径不支持 apply_routed_scaling_factor_on_output |
| failed_attempt_log | tp8_dpa_rank0.failed_modelopt_fp4.log | - | - | model config 为 fp8，但启动参数指定 modelopt_fp4，量化配置不匹配 |
| failed_attempt_log | tp8_pure_megamoe_4k_500_c1.failed_flashmla.log | - | - | bench warmup 阶段服务端口拒绝连接，服务已退出或不可用 |
| failed_attempt_log | tp8_pure_megamoe_rank0.failed_flashmla.log | - | - | 纯 TP 路径在 flash-mla decode schedule meta 触发 CUDA invalid argument |
| failed_attempt_log | tp8_pure_megamoe_rank0.failed_flashmla_c8.server.log | - | - | 纯 TP 路径在 flash-mla decode schedule meta 触发 CUDA invalid argument |
| failed_attempt_log | tp8_pure_nsa_megamoe_rank0.failed_flashmla.log | - | - | 纯 TP 路径在 flash-mla decode schedule meta 触发 CUDA invalid argument |
| failed_attempt_log | tp8_pure_nsa_megamoe_rank0.failed_flashmla_c8.server.log | - | - | 纯 TP 路径在 flash-mla decode schedule meta 触发 CUDA invalid argument |

## 文件说明

- `tp8_results_all.csv`：所有 TP=8 JSON 结果，包括主矩阵与纯 TP 部分结果。
- `tp8_dp8_dpa_megamoe_success.csv`：主矩阵成功结果。
- `tp8_incomplete_or_failed_logs.csv`：缺失、失败、未完成项及原因摘要。
- `tp8_vs_tp4_nvl_off_comparison.csv`：与 TP4 `tp4_nvl_off` 的同 case 同并发对比。
- `raw/`：原始 bench JSON、bench log、服务日志和失败日志。
- `logs/`：容器启动、服务启动、环境检查过程日志。
- `scripts/`：压测脚本和本汇总脚本。
