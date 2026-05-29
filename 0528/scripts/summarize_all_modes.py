#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path("/home/admin/0528")
RAW = ROOT / "raw"

MODES = {
    "tp4_dp4_dpa_nvl_cg16": {
        "label": "TP4+DP4+DPA+NVLS/MNNVL",
        "gpus": 4,
        "nodes": "10.56.160.38",
        "command_files": ["commands/tp4_dp4_dpa_nvl_cg16_server_command.sh"],
        "server_logs": ["server_logs/tp4_dp4_dpa_nvl_cg16_server.log"],
    },
    "tp8_dp8_dpa_nvl_cg16": {
        "label": "TP8+DP8+DPA+NVLS/MNNVL",
        "gpus": 8,
        "nodes": "10.56.160.38, 10.56.160.40",
        "command_files": [
            "commands/tp8_dp8_dpa_nvl_cg16_rank0_server_command.sh",
            "commands/tp8_dp8_dpa_nvl_cg16_rank1_server_command.sh",
        ],
        "server_logs": [
            "server_logs/tp8_dp8_dpa_nvl_cg16_rank0_server.log",
            "server_logs/tp8_dp8_dpa_nvl_cg16_rank1_server.log",
        ],
    },
    "pd_tp8_dp8_dpa_megamoe_mooncake_nvl_cg16": {
        "label": "PD TP8+DP8+DPA+Mooncake+NVLS/MNNVL",
        "gpus": 16,
        "nodes": "prefill: 10.56.160.38,10.56.160.40; decode: 10.56.160.36,10.56.160.34",
        "command_files": [
            "commands/pd_prefill_rank0_nvl_cg16_command.sh",
            "commands/pd_prefill_rank1_nvl_cg16_command.sh",
            "commands/pd_decode_rank0_nvl_cg16_command.sh",
            "commands/pd_decode_rank1_nvl_cg16_command.sh",
            "commands/pd_router_nvl_cg16_command.sh",
        ],
        "server_logs": [
            "server_logs/pd_prefill_rank0_nvl_cg16_server.log",
            "server_logs/pd_prefill_rank1_nvl_cg16_server.log",
            "server_logs/pd_decode_rank0_nvl_cg16_server.log",
            "server_logs/pd_decode_rank1_nvl_cg16_server.log",
            "server_logs/pd_router_nvl_cg16.log",
        ],
    },
}

CASES = ["4k_500", "32k_500", "128k_500"]
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]

METRIC_FIELDS = [
    "mode",
    "label",
    "case",
    "concurrency",
    "completed",
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
]


def load_results():
    results = {}
    for mode in MODES:
        results[mode] = {}
        for case in CASES:
            results[mode][case] = {}
            for c in CONCURRENCIES:
                path = RAW / f"{mode}_{case}_c{c}.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open() as f:
                    data = json.load(f)
                if data.get("completed") != 64:
                    raise RuntimeError(f"{path} completed={data.get('completed')}, expected 64")
                results[mode][case][c] = data
    return results


def fnum(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def pct(value):
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def md_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def server_info(result):
    info = result.get("server_info") or {}
    states = info.get("internal_states") or []
    if states:
        return states[0]
    return info


def write_summary_csv(results):
    path = ROOT / "summary_all_modes.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for mode, meta in MODES.items():
            for case in CASES:
                for c in CONCURRENCIES:
                    data = results[mode][case][c]
                    row = {field: data.get(field) for field in METRIC_FIELDS}
                    row.update({
                        "mode": mode,
                        "label": meta["label"],
                        "case": case,
                        "concurrency": c,
                    })
                    writer.writerow(row)
    return path


def write_comparison_csv(results):
    path = ROOT / "comparison_all_modes.csv"
    fields = [
        "case",
        "concurrency",
        "tp4_output_tok_s",
        "tp8_output_tok_s",
        "pd_output_tok_s",
        "tp8_over_tp4",
        "pd_over_tp4",
        "tp4_output_tok_s_per_gpu",
        "tp8_output_tok_s_per_gpu",
        "pd_output_tok_s_per_gpu",
        "tp8_per_gpu_over_tp4",
        "pd_per_gpu_over_tp4",
        "tp4_mean_ttft_ms",
        "tp8_mean_ttft_ms",
        "pd_mean_ttft_ms",
        "tp4_p99_tpot_ms",
        "tp8_p99_tpot_ms",
        "pd_p99_tpot_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            for c in CONCURRENCIES:
                tp4 = results["tp4_dp4_dpa_nvl_cg16"][case][c]
                tp8 = results["tp8_dp8_dpa_nvl_cg16"][case][c]
                pd = results["pd_tp8_dp8_dpa_megamoe_mooncake_nvl_cg16"][case][c]
                tp4_out = tp4["output_throughput"]
                tp8_out = tp8["output_throughput"]
                pd_out = pd["output_throughput"]
                row = {
                    "case": case,
                    "concurrency": c,
                    "tp4_output_tok_s": tp4_out,
                    "tp8_output_tok_s": tp8_out,
                    "pd_output_tok_s": pd_out,
                    "tp8_over_tp4": ratio(tp8_out, tp4_out),
                    "pd_over_tp4": ratio(pd_out, tp4_out),
                    "tp4_output_tok_s_per_gpu": tp4_out / MODES["tp4_dp4_dpa_nvl_cg16"]["gpus"],
                    "tp8_output_tok_s_per_gpu": tp8_out / MODES["tp8_dp8_dpa_nvl_cg16"]["gpus"],
                    "pd_output_tok_s_per_gpu": pd_out / MODES["pd_tp8_dp8_dpa_megamoe_mooncake_nvl_cg16"]["gpus"],
                    "tp8_per_gpu_over_tp4": ratio(
                        tp8_out / MODES["tp8_dp8_dpa_nvl_cg16"]["gpus"],
                        tp4_out / MODES["tp4_dp4_dpa_nvl_cg16"]["gpus"],
                    ),
                    "pd_per_gpu_over_tp4": ratio(
                        pd_out / MODES["pd_tp8_dp8_dpa_megamoe_mooncake_nvl_cg16"]["gpus"],
                        tp4_out / MODES["tp4_dp4_dpa_nvl_cg16"]["gpus"],
                    ),
                    "tp4_mean_ttft_ms": tp4["mean_ttft_ms"],
                    "tp8_mean_ttft_ms": tp8["mean_ttft_ms"],
                    "pd_mean_ttft_ms": pd["mean_ttft_ms"],
                    "tp4_p99_tpot_ms": tp4["p99_tpot_ms"],
                    "tp8_p99_tpot_ms": tp8["p99_tpot_ms"],
                    "pd_p99_tpot_ms": pd["p99_tpot_ms"],
                }
                writer.writerow(row)
    return path


def command_block(rel_path):
    path = ROOT / rel_path
    if not path.exists():
        return f"Missing: `{path}`"
    return f"`{path}`\n\n```bash\n{path.read_text().strip()}\n```"


def peak_rows(results):
    rows = []
    for case in CASES:
        for mode, meta in MODES.items():
            best_c = max(CONCURRENCIES, key=lambda c: results[mode][case][c]["output_throughput"])
            best = results[mode][case][best_c]
            c1 = results[mode][case][1]
            c64 = results[mode][case][64]
            rows.append([
                case,
                meta["label"],
                f"c{best_c}",
                fnum(best["output_throughput"]),
                fnum(best["output_throughput"] / meta["gpus"]),
                fnum(c1["output_throughput"]),
                fnum(c64["output_throughput"]),
                fnum(ratio(c64["output_throughput"], c1["output_throughput"])),
                fnum(c64["mean_ttft_ms"], 0),
                fnum(c64["p99_tpot_ms"], 1),
            ])
    return rows


def throughput_rows(results, case):
    rows = []
    for c in CONCURRENCIES:
        tp4 = results["tp4_dp4_dpa_nvl_cg16"][case][c]
        tp8 = results["tp8_dp8_dpa_nvl_cg16"][case][c]
        pd = results["pd_tp8_dp8_dpa_megamoe_mooncake_nvl_cg16"][case][c]
        tp4_out = tp4["output_throughput"]
        tp8_out = tp8["output_throughput"]
        pd_out = pd["output_throughput"]
        rows.append([
            c,
            fnum(tp4_out),
            fnum(tp8_out),
            fnum(ratio(tp8_out, tp4_out)),
            fnum(pd_out),
            fnum(ratio(pd_out, tp4_out)),
            fnum(tp4["mean_ttft_ms"], 0),
            fnum(tp8["mean_ttft_ms"], 0),
            fnum(pd["mean_ttft_ms"], 0),
            fnum(tp4["p99_tpot_ms"], 1),
            fnum(tp8["p99_tpot_ms"], 1),
            fnum(pd["p99_tpot_ms"], 1),
        ])
    return rows


def effective_config_rows(results):
    keys = [
        "tp_size",
        "dp_size",
        "nnodes",
        "node_rank",
        "disaggregation_mode",
        "moe_a2a_backend",
        "ep_size",
        "enable_dp_attention",
        "enable_nccl_nvls",
        "cuda_graph_max_bs",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "max_running_requests",
        "effective_max_running_requests_per_dp",
        "schedule_conservativeness",
        "quantization",
        "kv_cache_dtype",
        "pre_warm_nccl",
    ]
    rows = []
    for mode, meta in MODES.items():
        info = server_info(results[mode]["4k_500"][1])
        for key in keys:
            rows.append([meta["label"], key, info.get(key, "-")])
    return rows


def deployment_rows():
    return [[meta["label"], meta["nodes"], meta["gpus"], ", ".join(meta["command_files"])] for meta in MODES.values()]


def validate_rows(results):
    rows = []
    for mode, meta in MODES.items():
        rows.append([meta["label"], 21, "21/21", "completed=64 for all cases"])
    return rows


def write_report(results, summary_csv, comparison_csv):
    report = ROOT / "tp4_tp8_pd_full_report.md"
    lines = []
    lines.append("# L20A SGLang TP4 / TP8 / PD 性能测试报告")
    lines.append("")
    lines.append("## 测试范围")
    lines.append("")
    lines.append("- 测试日期：2026-05-29")
    lines.append("- 模型路径：`/home/admin/DeepSeek-V4-Flash`")
    lines.append("- benchmark：`python3 -m sglang.bench_serving --dataset-name random`")
    lines.append("- 输入/输出：`4k_500`、`32k_500`、`128k_500`")
    lines.append("- 并发：`1,2,4,8,16,32,64`")
    lines.append("- `num_prompts=64`，`cuda_graph_max_bs=16`")
    lines.append("- 所有模式均开启 NVLS/MNNVL/NVLink 相关环境变量和 `--enable-nccl-nvls`。")
    lines.append("")
    lines.append("## 结果文件")
    lines.append("")
    lines.append(f"- 全量指标 CSV：`{summary_csv}`")
    lines.append(f"- 横向对比 CSV：`{comparison_csv}`")
    lines.append("- 原始 JSON/日志：`/home/admin/0528/raw/`")
    lines.append("- 服务日志：`/home/admin/0528/server_logs/`")
    lines.append("- 部署命令：`/home/admin/0528/commands/`")
    lines.append("")
    lines.append("## 部署拓扑")
    lines.append("")
    lines.append(md_table(["模式", "节点", "GPU 数", "命令文件"], deployment_rows()))
    lines.append("")
    lines.append("## 完整性校验")
    lines.append("")
    lines.append(md_table(["模式", "期望 JSON", "实际 JSON", "完成状态"], validate_rows(results)))
    lines.append("")
    lines.append("## 有效运行配置")
    lines.append("")
    lines.append("启动命令中 TP4/TP8/PD 的主要性能参数保持一致，但 SGLang 在启用 DP attention 后会按 TP/EP 自动调整部分运行时参数：TP4 实际 `chunked_prefill_size=4096`，TP8/PD 实际 `chunked_prefill_size=2048`，且 `schedule_conservativeness` 实际为 `0.09`。这些是 server_args/server_info 中的有效值。")
    lines.append("")
    lines.append(md_table(["模式", "参数", "有效值"], effective_config_rows(results)))
    lines.append("")
    lines.append("## 峰值与扩展性")
    lines.append("")
    lines.append(md_table(
        ["Case", "模式", "峰值并发", "峰值 output tok/s", "峰值每 GPU tok/s", "c1 output", "c64 output", "c64/c1", "c64 TTFT(ms)", "c64 P99 TPOT(ms)"],
        peak_rows(results),
    ))
    lines.append("")
    lines.append("## 分 Case 明细")
    lines.append("")
    for case in CASES:
        lines.append(f"### {case}")
        lines.append("")
        lines.append(md_table(
            [
                "并发",
                "TP4 out",
                "TP8 out",
                "TP8/TP4",
                "PD out",
                "PD/TP4",
                "TP4 TTFT",
                "TP8 TTFT",
                "PD TTFT",
                "TP4 P99 TPOT",
                "TP8 P99 TPOT",
                "PD P99 TPOT",
            ],
            throughput_rows(results, case),
        ))
        lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- TP8 相比 TP4 在低/中并发没有体现 2 倍 GPU 的吞吐扩展，`4k_500` 全并发基本为 TP4 的 0.86-0.94 倍，`32k_500` 在 c1-c32 为 0.67-0.86 倍，`128k_500` 在 c1-c32 为 0.60-0.85 倍；只有高并发 c64 在 32k/128k 接近 TP4。")
    lines.append("- 按每 GPU 效率看，TP8 明显低于 TP4。即便总吞吐接近，TP8 使用 8 卡，TP4 使用 4 卡，因此 TP8 的单位 GPU output tok/s 大多只有 TP4 的 30%-50% 左右。")
    lines.append("- PD 模式在中等并发表现更好：`4k_500` c8-c32 为 TP4 的 1.25-1.46 倍，`32k_500` c4-c32 为 1.31-1.72 倍，`128k_500` c4-c32 为 1.13-1.42 倍；但 c64 下 PD 在 32k/128k 回落，说明高并发长上下文下 PD 路由/KV 传输/排队开销开始显著。")
    lines.append("- TP8/PD 长上下文低并发弱于 TP4 的主要现象是 TTFT 明显增加，而 TPOT 中位数多在 8-9ms 附近，说明瓶颈更偏 prefill、调度切块、跨节点/DP attention 协同，而不是纯 decode token 间隔。")
    lines.append("- NVLS/MNNVL 已启用：服务日志出现 `NCCL_MNNVL_ENABLE=1`、`NVLS multicast support is available`、`via P2P/MNNVL`；PD 日志还包含 Mooncake NVLINK/MNNVL 相关环境配置。当前低扩展效率不能简单归因为 NVLink 未启用。")
    lines.append("")
    lines.append("## 瓶颈判断")
    lines.append("")
    lines.append("- 有效 `chunked_prefill_size` 不一致是最直接的可见因素：TP4 自动降为 4096，TP8/PD 自动降为 2048。长上下文 prefill 被切得更碎，TTFT 增加，调度和通信启动次数增加。")
    lines.append("- TP8 的 EP/DP 规模更大，MegaMoE 下 `ep_size=8`，跨节点 TP/DP attention 协同会带来更多同步和路由开销。日志已确认 MNNVL/NVLS 路径可用，因此需要进一步用 profiling 区分 MoE dispatch/all-to-all、attention prefill、NCCL allreduce/allgather 的占比。")
    lines.append("- PD 在 c4-c32 改善明显，说明把 prefill/decode 分离后能减少部分资源争用；c64 回落说明 router 排队、KV transfer 或 decode 侧批处理上限成为新的瓶颈。")
    lines.append("")
    lines.append("## Profiling 结果")
    lines.append("")
    lines.append("已完成 `32k_500_c1` 的 TP4/TP8 profiling，并额外采集 PD prefill rank0 profile。详细报告见 `/home/admin/0528/profile_bottleneck_report.md`，trace 解析汇总见 `/home/admin/0528/profiles/profile_parse_summary.json`。")
    lines.append("")
    lines.append("主要结论：TP8 弱于 TP4 不是 NVLink/NCCL 原始带宽没打通。trace 中 NCCL GPU kernel 是毫秒级，主要 GPU 耗时集中在 `deep_gemm::sm100_fp8_fp4_mega_moe_impl`；同时 Gloo broadcast/all_gather 的 user annotation 可到秒级，说明瓶颈更偏 DP/EP 调度同步、chunking 和 MegaMoE 路径。SGLang 还会把 TP4 的有效 `chunked_prefill_size` 调整为 4096、TP8/PD 调整为 2048，这会增加 TP8 长上下文 prefill 的切块和同步次数。")
    lines.append("")
    lines.append("## 容器启动命令")
    lines.append("")
    lines.append(command_block("scripts/recreate_sgl_container.sh"))
    lines.append("")
    lines.append("## 服务部署命令")
    lines.append("")
    for meta in MODES.values():
        lines.append(f"### {meta['label']}")
        lines.append("")
        for rel in meta["command_files"]:
            lines.append(command_block(rel))
            lines.append("")
    lines.append("## 失败尝试归档")
    lines.append("")
    lines.append("- `failed_attempts/pd_deepep_65536_assert`：DeepEP low-latency dispatcher token 限制断言。")
    lines.append("- `failed_attempts/pd_deepep_ep8_weight_mismatch`：DeepEP EP8 与当前 FP8 权重 shape 不匹配。")
    lines.append("- `failed_attempts/pd_bench_host_python_missing`：宿主机 Python 缺少 sglang 模块，benchmark 后续改在容器内执行。")
    lines.append("")
    report.write_text("\n".join(lines) + "\n")
    return report


def main():
    results = load_results()
    summary_csv = write_summary_csv(results)
    comparison_csv = write_comparison_csv(results)
    report = write_report(results, summary_csv, comparison_csv)
    print(summary_csv)
    print(comparison_csv)
    print(report)


if __name__ == "__main__":
    main()
