#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


ROOT = Path("/home/admin/parallel/dm_tp_dp_8")
RAW = ROOT / "raw"
TP4_ALL = Path("/home/admin/dm_tp_4/tp4_results_all.csv")

CONCURRENCY_ORDER = [1, 4, 8, 16, 32, 64, 128]
WORKLOAD_ORDER = ["4k_500", "32k_500", "128k_500"]
MAIN_RUN_SET = "tp8_dp8_dpa_megamoe"


RESULT_FIELDS = [
    "source_file",
    "run_set",
    "workload",
    "input_len",
    "output_len",
    "concurrency",
    "row_type",
    "completed",
    "duration_s",
    "request_throughput_req_s",
    "input_throughput_tok_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "tp_size",
    "dp_size",
    "ep_size",
    "enable_nccl_nvls",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "attention_backend",
    "moe_a2a_backend",
    "moe_runner_backend",
    "kv_cache_dtype",
    "quantization",
    "status",
]


def fmt(value, digits=2):
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def sort_key(row):
    return (
        row.get("run_set", ""),
        WORKLOAD_ORDER.index(row["workload"]) if row.get("workload") in WORKLOAD_ORDER else 99,
        int(row.get("concurrency") or 0),
        row.get("source_file", ""),
    )


def read_json_rows():
    rows = []
    pattern = re.compile(r"(?P<run_set>.+)_(?P<workload>\d+k_500)_c(?P<concurrency>\d+)\.json$")
    for path in sorted(RAW.glob("tp8*_c*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        with path.open() as f:
            data = json.load(f)
        server_info = data.get("server_info") or {}
        run_set = match.group("run_set")
        row_type = "matrix" if run_set == MAIN_RUN_SET else "partial"
        row = {
            "source_file": path.name,
            "run_set": run_set,
            "workload": match.group("workload"),
            "input_len": data.get("random_input_len"),
            "output_len": data.get("random_output_len"),
            "concurrency": int(match.group("concurrency")),
            "row_type": row_type,
            "completed": data.get("completed"),
            "duration_s": data.get("duration"),
            "request_throughput_req_s": data.get("request_throughput"),
            "input_throughput_tok_s": data.get("input_throughput"),
            "output_throughput_tok_s": data.get("output_throughput"),
            "total_throughput_tok_s": data.get("total_throughput"),
            "mean_e2e_latency_ms": data.get("mean_e2e_latency_ms"),
            "median_e2e_latency_ms": data.get("median_e2e_latency_ms"),
            "p90_e2e_latency_ms": data.get("p90_e2e_latency_ms"),
            "p99_e2e_latency_ms": data.get("p99_e2e_latency_ms"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "median_ttft_ms": data.get("median_ttft_ms"),
            "p99_ttft_ms": data.get("p99_ttft_ms"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
            "median_tpot_ms": data.get("median_tpot_ms"),
            "p99_tpot_ms": data.get("p99_tpot_ms"),
            "mean_itl_ms": data.get("mean_itl_ms"),
            "p99_itl_ms": data.get("p99_itl_ms"),
            "max_output_tokens_per_s": data.get("max_output_tokens_per_s"),
            "max_concurrent_requests": data.get("max_concurrent_requests"),
            "tp_size": server_info.get("tp_size"),
            "dp_size": server_info.get("dp_size"),
            "ep_size": server_info.get("ep_size"),
            "enable_nccl_nvls": server_info.get("enable_nccl_nvls"),
            "enable_dp_attention": server_info.get("enable_dp_attention"),
            "enable_dp_lm_head": server_info.get("enable_dp_lm_head"),
            "attention_backend": server_info.get("attention_backend"),
            "moe_a2a_backend": server_info.get("moe_a2a_backend"),
            "moe_runner_backend": server_info.get("moe_runner_backend"),
            "kv_cache_dtype": server_info.get("kv_cache_dtype"),
            "quantization": server_info.get("quantization"),
            "status": "success",
        }
        rows.append(row)
    rows.sort(key=sort_key)
    return rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_reason(path):
    if not path.exists():
        return "not_run_or_no_log"
    text = path.read_text(errors="ignore")
    checks = [
        ("Quantization method specified in the model config", "model config 为 fp8，但启动参数指定 modelopt_fp4，量化配置不匹配"),
        ("apply_routed_scaling_factor_on_output", "flashinfer_trtllm routed 路径不支持 apply_routed_scaling_factor_on_output"),
        ("AssertionError", "flashinfer_trtllm 初始化阶段 AssertionError"),
        ("get_decoding_sched_meta.cu:111", "纯 TP 路径在 flash-mla decode schedule meta 触发 CUDA invalid argument"),
        ("ConnectionRefusedError", "bench warmup 阶段服务端口拒绝连接，服务已退出或不可用"),
        ("Warmup failed", "bench warmup 失败"),
        ("ClientConnectorError", "bench 请求无法连接服务"),
    ]
    for marker, reason in checks:
        if marker in text:
            return reason
    for line in text.splitlines():
        if any(token in line for token in ("Error", "error", "Traceback", "RuntimeError", "ValueError")):
            return line.strip()[:240]
    return "see_log"


def build_failure_rows(rows):
    existing = {(r["run_set"], r["workload"], int(r["concurrency"])) for r in rows}
    failure_rows = []

    for workload in WORKLOAD_ORDER:
        for concurrency in CONCURRENCY_ORDER:
            key = (MAIN_RUN_SET, workload, concurrency)
            if key in existing:
                continue
            log_name = f"{MAIN_RUN_SET}_{workload}_c{concurrency}.log"
            log_path = RAW / log_name
            failure_rows.append(
                {
                    "source_file": log_name if log_path.exists() else "",
                    "category": "main_matrix_missing",
                    "run_set": MAIN_RUN_SET,
                    "workload": workload,
                    "concurrency": concurrency,
                    "status": "failed_or_missing",
                    "reason": extract_reason(log_path),
                }
            )

    for path in sorted(RAW.glob("*.failed*.log")):
        failure_rows.append(
            {
                "source_file": path.name,
                "category": "failed_attempt_log",
                "run_set": path.name.split(".failed", 1)[0],
                "workload": "",
                "concurrency": "",
                "status": "failed",
                "reason": extract_reason(path),
            }
        )

    return failure_rows


def read_tp4_baseline():
    if not TP4_ALL.exists():
        return {}
    rows = {}
    with TP4_ALL.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_set") != "tp4_nvl_off":
                continue
            if row.get("status") != "success":
                continue
            key = (row.get("workload"), int(float(row.get("concurrency") or 0)))
            rows[key] = row
    return rows


def build_comparison(rows):
    tp4 = read_tp4_baseline()
    comparison = []
    for row in rows:
        if row["run_set"] != MAIN_RUN_SET:
            continue
        key = (row["workload"], int(row["concurrency"]))
        base = tp4.get(key)
        if not base:
            continue
        tp8_out = float(row["output_throughput_tok_s"])
        tp4_out = float(base["output_throughput_tok_s"])
        tp8_total = float(row["total_throughput_tok_s"])
        tp4_total = float(base["total_throughput_tok_s"])
        comparison.append(
            {
                "workload": row["workload"],
                "concurrency": row["concurrency"],
                "tp8_output_throughput_tok_s": tp8_out,
                "tp4_output_throughput_tok_s": tp4_out,
                "tp8_vs_tp4_output_ratio": tp8_out / tp4_out if tp4_out else "",
                "tp8_total_throughput_tok_s": tp8_total,
                "tp4_total_throughput_tok_s": tp4_total,
                "tp8_vs_tp4_total_ratio": tp8_total / tp4_total if tp4_total else "",
                "tp8_mean_ttft_ms": row["mean_ttft_ms"],
                "tp4_mean_ttft_ms": base["mean_ttft_ms"],
                "tp8_mean_tpot_ms": row["mean_tpot_ms"],
                "tp4_mean_tpot_ms": base["mean_tpot_ms"],
            }
        )
    comparison.sort(key=lambda r: (WORKLOAD_ORDER.index(r["workload"]), int(r["concurrency"])))
    return comparison


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def get_main(rows):
    return [r for r in rows if r["run_set"] == MAIN_RUN_SET]


def build_markdown(rows, failures, comparison):
    main = get_main(rows)
    by_key = {(r["workload"], int(r["concurrency"])): r for r in main}
    partial = [r for r in rows if r["run_set"] != MAIN_RUN_SET]

    summary_rows = []
    for workload in WORKLOAD_ORDER:
        workload_rows = [r for r in main if r["workload"] == workload]
        if not workload_rows:
            continue
        best = max(workload_rows, key=lambda r: float(r["output_throughput_tok_s"]))
        finished = ", ".join(str(r["concurrency"]) for r in sorted(workload_rows, key=lambda r: int(r["concurrency"])))
        summary_rows.append(
            [
                workload,
                finished,
                str(best["concurrency"]),
                fmt(best["output_throughput_tok_s"]),
                fmt(best["total_throughput_tok_s"]),
                fmt(best["mean_ttft_ms"]),
                fmt(best["mean_tpot_ms"]),
            ]
        )

    matrix_rows = []
    for concurrency in CONCURRENCY_ORDER:
        row = [str(concurrency)]
        for workload in WORKLOAD_ORDER:
            result = by_key.get((workload, concurrency))
            row.append(fmt(result["output_throughput_tok_s"]) if result else "FAIL/未完成")
        matrix_rows.append(row)

    latency_rows = []
    for workload in WORKLOAD_ORDER:
        for concurrency in CONCURRENCY_ORDER:
            result = by_key.get((workload, concurrency))
            if not result:
                continue
            latency_rows.append(
                [
                    workload,
                    str(concurrency),
                    fmt(result["request_throughput_req_s"]),
                    fmt(result["output_throughput_tok_s"]),
                    fmt(result["total_throughput_tok_s"]),
                    fmt(result["mean_ttft_ms"]),
                    fmt(result["p99_ttft_ms"]),
                    fmt(result["mean_tpot_ms"]),
                    fmt(result["p99_tpot_ms"]),
                    fmt(result["p99_e2e_latency_ms"]),
                ]
            )

    compare_rows = []
    for row in comparison:
        if row["concurrency"] not in (1, 32, 64, 128):
            continue
        compare_rows.append(
            [
                row["workload"],
                str(row["concurrency"]),
                fmt(row["tp8_output_throughput_tok_s"]),
                fmt(row["tp4_output_throughput_tok_s"]),
                fmt(row["tp8_vs_tp4_output_ratio"]),
                fmt(row["tp8_total_throughput_tok_s"]),
                fmt(row["tp4_total_throughput_tok_s"]),
                fmt(row["tp8_vs_tp4_total_ratio"]),
            ]
        )

    partial_rows = []
    for row in partial:
        partial_rows.append(
            [
                row["run_set"],
                row["workload"],
                str(row["concurrency"]),
                fmt(row["output_throughput_tok_s"]),
                fmt(row["total_throughput_tok_s"]),
                row["source_file"],
            ]
        )

    failure_md_rows = []
    for row in failures:
        if row["category"] == "main_matrix_missing" or row["status"] == "failed":
            failure_md_rows.append(
                [
                    row["category"],
                    row["source_file"] or "-",
                    row["workload"] or "-",
                    str(row["concurrency"] or "-"),
                    row["reason"],
                ]
            )

    lines = [
        "# TP=8 SGLang 测试结果总结",
        "",
        "## 测试口径",
        "",
        "- 目标目录：`/home/admin/parallel/dm_tp_dp_8/`。",
        "- 主结果口径：`tp8_dp8_dpa_megamoe`，两机 `10.56.160.38` + `10.56.160.40`，`tp_size=8`，`dp_size=8`，`ep_size=8`，启用 `enable_dp_attention` 与 `enable_nccl_nvls`。",
        "- 模型路径：`/home/admin/DeepSeek-V4-Flash/`。",
        "- benchmark：`python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name random --num-prompts 100`，输入/输出长度按 case 替换，并发为 `1,4,8,16,32,64,128`。",
        "- 主服务从 `server_info` 看使用 `attention_backend=dsv4`，`moe_a2a_backend=megamoe`，`kv_cache_dtype=fp8_e4m3`，`max_total_tokens=202752`。",
        "",
        "## 结论",
        "",
        "- TP=8 + DP=8 + DPA + NVLS/MNNVL 的主配置可完成 `4k_500` 与 `32k_500` 全并发矩阵。",
        "- `128k_500` 只完成到并发 32；并发 64 在 warmup 阶段连接服务失败，说明服务已退出或不可用；并发 128 没有成功结果。",
        "- 输出吞吐随并发上升明显，但长上下文下 decode 吞吐下降明显，`128k_500` 在并发 32 达到本轮最高 `103.57 tok/s`，继续提高并发不稳定。",
        "- 纯 TP=8 的尝试只留下少量 `4k_500` 小并发结果，随后在 flash-mla decode schedule meta 触发 CUDA invalid argument，不能作为完整可用矩阵。",
        "- 与 TP4 `tp4_nvl_off` 基线相比，本轮 TP=8 DP 配置没有表现出吞吐优势；短上下文差距最大，长上下文也未超过 TP4。对比数据已单独保存到 `tp8_vs_tp4_nvl_off_comparison.csv`。",
        "",
        "## 主矩阵峰值",
        "",
        md_table(
            ["case", "完成并发", "最佳并发", "output tok/s", "total tok/s", "mean TTFT ms", "mean TPOT ms"],
            summary_rows,
        ),
        "",
        "## Output Throughput 矩阵",
        "",
        md_table(["concurrency", "4k_500", "32k_500", "128k_500"], matrix_rows),
        "",
        "## 主矩阵明细",
        "",
        md_table(
            [
                "case",
                "conc",
                "req/s",
                "output tok/s",
                "total tok/s",
                "mean TTFT",
                "p99 TTFT",
                "mean TPOT",
                "p99 TPOT",
                "p99 E2E",
            ],
            latency_rows,
        ),
        "",
        "## TP4 对比抽样",
        "",
        "完整对比见 `tp8_vs_tp4_nvl_off_comparison.csv`。ratio 为 TP8/TP4，低于 1 表示 TP8 本轮低于 TP4 基线。",
        "",
        md_table(
            ["case", "conc", "TP8 output", "TP4 output", "output ratio", "TP8 total", "TP4 total", "total ratio"],
            compare_rows,
        ),
        "",
        "## 纯 TP=8 部分结果",
        "",
        "纯 TP=8 没有完成完整矩阵，以下仅保留原始观测值，不作为主结论。",
        "",
        md_table(["run_set", "case", "conc", "output tok/s", "total tok/s", "source"], partial_rows) if partial_rows else "无。",
        "",
        "## 失败与未完成项",
        "",
        md_table(["category", "source", "case", "conc", "reason"], failure_md_rows),
        "",
        "## 文件说明",
        "",
        "- `tp8_results_all.csv`：所有 TP=8 JSON 结果，包括主矩阵与纯 TP 部分结果。",
        "- `tp8_dp8_dpa_megamoe_success.csv`：主矩阵成功结果。",
        "- `tp8_incomplete_or_failed_logs.csv`：缺失、失败、未完成项及原因摘要。",
        "- `tp8_vs_tp4_nvl_off_comparison.csv`：与 TP4 `tp4_nvl_off` 的同 case 同并发对比。",
        "- `raw/`：原始 bench JSON、bench log、服务日志和失败日志。",
        "- `logs/`：容器启动、服务启动、环境检查过程日志。",
        "- `scripts/`：压测脚本和本汇总脚本。",
    ]
    return "\n".join(lines) + "\n"


def build_readme():
    return """# dm_tp_dp_8

该目录保存 TP=8 相关测试数据和总结。

- `tp8_summary.md`：中文总结和结论。
- `tp8_results_all.csv`：所有解析出的 TP=8 bench JSON 结果。
- `tp8_dp8_dpa_megamoe_success.csv`：主测试矩阵成功结果。
- `tp8_incomplete_or_failed_logs.csv`：失败和未完成项摘要。
- `tp8_vs_tp4_nvl_off_comparison.csv`：与 TP4 `tp4_nvl_off` 基线的对比。
- `raw/`：原始 bench 结果和服务日志。
- `logs/`：部署和启动过程日志。
- `scripts/`：运行和汇总脚本。
"""


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_json_rows()
    main_rows = [r for r in rows if r["run_set"] == MAIN_RUN_SET]
    failures = build_failure_rows(rows)
    comparison = build_comparison(rows)

    write_csv(ROOT / "tp8_results_all.csv", RESULT_FIELDS, rows)
    write_csv(ROOT / "tp8_dp8_dpa_megamoe_success.csv", RESULT_FIELDS, main_rows)
    write_csv(
        ROOT / "tp8_incomplete_or_failed_logs.csv",
        ["source_file", "category", "run_set", "workload", "concurrency", "status", "reason"],
        failures,
    )
    write_csv(
        ROOT / "tp8_vs_tp4_nvl_off_comparison.csv",
        [
            "workload",
            "concurrency",
            "tp8_output_throughput_tok_s",
            "tp4_output_throughput_tok_s",
            "tp8_vs_tp4_output_ratio",
            "tp8_total_throughput_tok_s",
            "tp4_total_throughput_tok_s",
            "tp8_vs_tp4_total_ratio",
            "tp8_mean_ttft_ms",
            "tp4_mean_ttft_ms",
            "tp8_mean_tpot_ms",
            "tp4_mean_tpot_ms",
        ],
        comparison,
    )
    (ROOT / "tp8_summary.md").write_text(build_markdown(rows, failures, comparison))
    (ROOT / "README.md").write_text(build_readme())

    print(f"wrote {len(rows)} result rows")
    print(f"wrote {len(main_rows)} main matrix rows")
    print(f"wrote {len(failures)} failure/missing rows")
    print(f"wrote {len(comparison)} TP4 comparison rows")


if __name__ == "__main__":
    main()
