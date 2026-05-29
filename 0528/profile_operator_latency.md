# Profile 主要算子延迟汇总

来源：`/home/admin/0528/profiles/profile_parse_summary.json` 以及对应 `bench.log`。

说明：
- 当前 profiler 只采了 `32k_500_c1`，用于定位 TP8 相比 TP4 的瓶颈；不是完整 4k/32k/128k 全矩阵的算子 profile。
- 表中的延迟是 profiler 窗口内累计时长，单位 ms，不是单次 kernel 调用的平均耗时。
- `max/rank` 更接近单个 rank 的关键路径压力；`sum/ranks` 是已解析 trace 的累计和，会把并行 rank 相加，不能直接等同 wall time。
- PD profile 当前解析到的是 prefill rank0 trace；其他 PD trace 文件很大，主报告中也只把它作为辅助证据。

## Profile 场景

| 场景 | 已解析 trace | chunked_prefill_size | profile output tok/s | mean TTFT(ms) | mean TPOT(ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | 4 | 4096 | 58.05 | 2423.61 | 8.53 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | 4 | 2048 | 57.21 | 2484.44 | 8.56 |
| PD prefill TP8+DP8, 32k_500_c1 | 1 | 2048 | 9.89 | 25741.53 | 8.43 |

## Category Totals

| 场景 | 类别 | traces | min(ms) | avg(ms) | max(ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | GPU kernel | 4 | 573.2 | 1645.1 | 2004.4 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | GPU annotation | 4 | 2003.5 | 2006.8 | 2010.3 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | CPU/user annotation | 4 | 2015.2 | 2037.9 | 2046.8 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | CPU op | 4 | 42.8 | 305.8 | 1073.5 |
| TP4+DP4+DPA+NVLS, 32k_500_c1 | CUDA runtime API | 4 | 6.642 | 65.7 | 241.9 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | GPU kernel | 4 | 277.5 | 1660.4 | 2122.2 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | GPU annotation | 4 | 2131.3 | 2135.0 | 2137.1 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | CPU/user annotation | 4 | 2138.5 | 2161.5 | 2169.8 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | CPU op | 4 | 47.8 | 323.8 | 1145.3 |
| TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1 | CUDA runtime API | 4 | 6.933 | 66.3 | 244.1 |
| PD prefill TP8+DP8, 32k_500_c1 | GPU kernel | 1 | 1551.3 | 1551.3 | 1551.3 |
| PD prefill TP8+DP8, 32k_500_c1 | GPU annotation | 1 | 1564.5 | 1564.5 | 1564.5 |
| PD prefill TP8+DP8, 32k_500_c1 | CPU/user annotation | 1 | 2796.6 | 2796.6 | 2796.6 |
| PD prefill TP8+DP8, 32k_500_c1 | CPU op | 1 | 462.5 | 462.5 | 462.5 |
| PD prefill TP8+DP8, 32k_500_c1 | CUDA runtime API | 1 | 7.193 | 7.193 | 7.193 |

## TP4+DP4+DPA+NVLS, 32k_500_c1

### GPU Kernel Top

| 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | ---: | ---: | ---: |
| DeepGEMM MegaMoE fp8/fp4 | 1993.4 | 1546.3 | 6185.1 |
| Sparse MLA decode attention fp8 | 88.7 | 22.2 | 88.7 |
| DeepGEMM dense fp8/fp4 GEMM | 81.4 | 20.3 | 81.4 |
| MHC post tilelang | 18.4 | 4.605 | 18.4 |
| Fused Q norm + RoPE | 17.2 | 4.292 | 17.2 |
| DeepSeek RoPE | 15.3 | 3.835 | 15.3 |
| Per-token group quant 8bit | 15.3 | 3.815 | 15.3 |
| NCCL AllGather kernel | 3.999 | 2.994 | 12.0 |
| nvJet / TRT-LLM fused kernel | 3.399 | 2.545 | 10.2 |
| FlashInfer one-shot all-reduce push | 3.215 | 2.404 | 9.616 |
| MegaMoE pre-dispatch | 0.379 | 0.280 | 1.119 |
| cuBLASLt splitK reduce | 0.022 | 0.016 | 0.065 |

### 通信/控制面/CPU Top

| 类别 | 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | --- | ---: | ---: | ---: |
| CPU/user annotation | step[EXTEND bs=1 toks=4096] | 2012.1 | 503.0 | 2012.1 |
| GPU annotation | step[EXTEND bs=1 toks=4096] | 2010.2 | 502.5 | 2010.2 |
| GPU annotation | step[IDLE bs=0] | 2003.3 | 1501.2 | 6004.8 |
| CPU/user annotation | gloo:all_gather | 1882.2 | 948.5 | 3794.0 |
| CPU/user annotation | gloo:broadcast | 1853.9 | 483.6 | 1934.3 |
| CPU op | sglang::deep_gemm_fp8_fp8_bf16_nt | 170.9 | 42.7 | 170.9 |
| CPU/user annotation | step[IDLE bs=0] | 160.8 | 102.2 | 408.8 |
| CPU op | aten::empty | 133.3 | 47.4 | 189.5 |
| CUDA runtime API | cudaLaunchKernelExC | 128.7 | 34.8 | 139.0 |
| CUDA runtime API | cudaLaunchKernel | 87.1 | 22.1 | 88.3 |
| CPU op | sgl_kernel::sgl_per_token_group_quant_8bit_v2 | 58.3 | 14.6 | 58.3 |
| CPU op | aten::view | 55.3 | 13.8 | 55.3 |
| CPU op | aten::to | 46.5 | 12.9 | 51.5 |
| CPU op | aten::_to_copy | 40.7 | 10.9 | 43.7 |
| CPU op | aten::new_empty | 40.7 | 14.3 | 57.4 |
| CPU op | aten::copy_ | 40.6 | 10.1 | 40.6 |
| CUDA runtime API | cudaPointerGetAttributes | 16.5 | 5.845 | 23.4 |
| CUDA runtime API | cudaMalloc | 4.061 | 1.015 | 4.061 |

## TP8+DP8+DPA+NVLS/MNNVL, 32k_500_c1

### GPU Kernel Top

| 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | ---: | ---: | ---: |
| DeepGEMM MegaMoE fp8/fp4 | 2109.9 | 1602.3 | 6409.2 |
| Sparse MLA decode attention fp8 | 45.2 | 11.3 | 45.2 |
| DeepGEMM dense fp8/fp4 GEMM | 44.9 | 11.2 | 44.9 |
| MHC post tilelang | 9.941 | 2.485 | 9.941 |
| NCCL AllReduce kernel | 8.914 | 6.667 | 26.7 |
| Fused Q norm + RoPE | 8.832 | 2.208 | 8.832 |
| DeepSeek RoPE | 7.888 | 1.972 | 7.888 |
| MHC pre big-fuse norm tilelang | 7.870 | 1.967 | 7.870 |
| NCCL AllGather kernel | 2.994 | 2.226 | 8.905 |
| MegaMoE pre-dispatch | 0.379 | 0.280 | 1.122 |
| nvJet / TRT-LLM fused kernel | 0.121 | 0.086 | 0.345 |
| ATen vectorized elementwise | 0.022 | 0.015 | 0.061 |

### 通信/控制面/CPU Top

| 类别 | 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | --- | ---: | ---: | ---: |
| CPU/user annotation | step[EXTEND bs=1 toks=2048] | 2133.0 | 533.2 | 2133.0 |
| GPU annotation | step[EXTEND bs=1 toks=2048] | 2131.1 | 532.8 | 2131.1 |
| GPU annotation | step[IDLE bs=0] | 2125.3 | 1593.3 | 6373.2 |
| CPU/user annotation | gloo:all_gather | 2030.2 | 1023.4 | 4093.5 |
| CPU/user annotation | gloo:broadcast | 2008.4 | 505.7 | 2022.8 |
| CPU op | sglang::deep_gemm_fp8_fp8_bf16_nt | 177.1 | 44.3 | 177.1 |
| CPU op | aten::empty | 144.2 | 49.6 | 198.5 |
| CPU/user annotation | step[IDLE bs=0] | 136.1 | 98.1 | 392.4 |
| CUDA runtime API | cudaLaunchKernelExC | 128.7 | 34.8 | 139.1 |
| CUDA runtime API | cudaLaunchKernel | 89.6 | 22.7 | 90.8 |
| CPU op | aten::view | 61.9 | 15.5 | 61.9 |
| CPU op | sgl_kernel::sgl_per_token_group_quant_8bit_v2 | 59.1 | 14.8 | 59.1 |
| CPU op | aten::to | 50.9 | 14.0 | 55.8 |
| CPU op | aten::_to_copy | 44.7 | 12.2 | 48.9 |
| CPU op | aten::copy_ | 43.8 | 11.0 | 43.8 |
| CPU op | aten::new_empty | 43.4 | 14.8 | 59.4 |
| CUDA runtime API | cudaPointerGetAttributes | 17.6 | 6.096 | 24.4 |
| GPU annotation | nccl:all_reduce | 8.914 | 6.703 | 26.8 |

## PD prefill TP8+DP8, 32k_500_c1

### GPU Kernel Top

| 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | ---: | ---: | ---: |
| DeepGEMM MegaMoE fp8/fp4 | 1541.3 | 1541.3 | 1541.3 |
| NCCL AllReduce kernel | 7.143 | 7.143 | 7.143 |
| NCCL AllGather kernel | 2.338 | 2.338 | 2.338 |
| MegaMoE pre-dispatch | 0.405 | 0.405 | 0.405 |
| nvJet / TRT-LLM fused kernel | 0.116 | 0.116 | 0.116 |
| ATen vectorized elementwise | 0.021 | 0.021 | 0.021 |
| CUB DeviceScan | 0.012 | 0.012 | 0.012 |

### 通信/控制面/CPU Top

| 类别 | 算子/事件 | max/rank累计(ms) | avg/rank累计(ms) | sum/ranks累计(ms) |
| --- | --- | ---: | ---: | ---: |
| CPU/user annotation | gloo:broadcast | 2093.6 | 2093.6 | 2093.6 |
| GPU annotation | step[IDLE bs=0] | 1555.0 | 1555.0 | 1555.0 |
| CPU/user annotation | gloo:all_gather | 579.1 | 579.1 | 579.1 |
| CPU/user annotation | step[IDLE bs=0] | 122.9 | 122.9 | 122.9 |
| CPU op | c10d::_allgather_base_ | 56.0 | 56.0 | 56.0 |
| CPU op | aten::select | 41.0 | 41.0 | 41.0 |
| CPU op | aten::empty | 32.1 | 32.1 | 32.1 |
| CPU op | aten::chunk | 29.9 | 29.9 | 29.9 |
| CPU op | aten::split | 28.2 | 28.2 | 28.2 |
| CPU op | aten::slice | 22.0 | 22.0 | 22.0 |
| CPU op | c10d::broadcast_ | 20.8 | 20.8 | 20.8 |
| CPU op | aten::index_put_ | 20.5 | 20.5 | 20.5 |
| GPU annotation | nccl:all_reduce | 7.143 | 7.143 | 7.143 |
| CUDA runtime API | cudaLaunchKernelExC | 3.497 | 3.497 | 3.497 |
| GPU annotation | nccl:_all_gather_base | 2.338 | 2.338 | 2.338 |
| CUDA runtime API | cudaPointerGetAttributes | 2.289 | 2.289 | 2.289 |
| CPU/user annotation | nccl:all_reduce | 0.626 | 0.626 | 0.626 |
| CPU/user annotation | nccl:_all_gather_base | 0.471 | 0.471 | 0.471 |

## 结论

- 三个已采 profile 场景里，GPU 侧主耗时都是 `DeepGEMM MegaMoE fp8/fp4`，TP4 max/rank 约 1993 ms，TP8 max/rank 约 2110 ms，PD prefill rank0 约 1541 ms。
- NCCL GPU kernel 累计耗时是毫秒到十几毫秒级：TP8 的 `NCCL AllReduce kernel` max/rank 约 8.9 ms，`NCCL AllGather kernel` max/rank 约 3.0 ms，不是 profile 窗口内的主要 GPU 耗时。
- Gloo `broadcast/all_gather` 的 user annotation 达到秒级，说明控制面同步/调度等待值得继续看；这和 NVLink/NVLS 数据面带宽不是同一个瓶颈。
- TP8 profile 中 `chunked_prefill_size=2048`，TP4 为 `4096`，TP8 在 32k prefill 下切块更多，会放大调度和同步次数，是 TTFT 偏高的重要原因之一。
