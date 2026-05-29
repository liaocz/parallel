# Decode Batch 统计

来源：`/home/admin/0528/server_logs/*` 中的 `Decode batch, #running-req: ...`。

说明：
- 这里的 decode batch 指每个 scheduler/DP rank 日志中的 `#running-req`。
- SGLang 按 `decode_log_interval` 周期采样打印，TP4/TP8 为 40，PD decode 为 100；因此这是日志采样到的值，不保证覆盖每一个 decode step。
- 明细 CSV：`/home/admin/0528/decode_batch_by_case.csv`。

## Max Decode Batch By Case

| 模式 | Case | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TP4+DP4+DPA | 4k_500 | 1 | 2 | 3 | 4 | 6 | 11 | 16 |
| TP4+DP4+DPA | 32k_500 | 1 | 2 | 2 | 3 | 6 | 10 | 16 |
| TP4+DP4+DPA | 128k_500 | 1 | 2 | 2 | 4 | 6 | 10 | 16 |
| TP8+DP8+DPA | 4k_500 | 1 | 1 | 2 | 3 | 4 | 6 | 8 |
| TP8+DP8+DPA | 32k_500 | 1 | 1 | 2 | 2 | 3 | 7 | 8 |
| TP8+DP8+DPA | 128k_500 | 1 | 1 | 2 | 2 | 4 | 6 | 8 |
| PD decode TP8+DP8 | 4k_500 | 1 | 1 | 1 | 2 | 3 | 5 | 7 |
| PD decode TP8+DP8 | 32k_500 | 1 | 1 | 1 | 2 | 3 | 4 | 6 |
| PD decode TP8+DP8 | 128k_500 | 1 | 1 | 1 | 2 | 2 | 3 | 3 |

## 读数解释

- TP4 是 DP4，单 scheduler 采样到的最大 decode batch 到 16。
- TP8 是 DP8，单 scheduler 采样到的最大 decode batch 到 8。
- PD decode 在长上下文下 batch 明显更小，`128k_500_c64` 采样最大只有 3，说明高并发长上下文下 decode 侧并没有形成很大的单 scheduler decode batch，瓶颈更可能在 prefill/KV transfer/router 排队或 decode 分发节奏。

