#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


ROOT = Path("/home/admin/0528")
RAW = ROOT / "raw"

MODES = {
    "tp4_dp4_dpa_nvl_cg16": {
        "label": "TP4 + DP4 + DPA + NVLS/MNNVL",
        "gpus": 4,
        "nodes": "10.56.160.38",
        "commands": [ROOT / "commands/tp4_dp4_dpa_nvl_cg16_server_command.sh"],
    },
    "tp8_dp8_dpa_nvl_cg16": {
        "label": "TP8 + DP8 + DPA + NVLS/MNNVL",
        "gpus": 8,
        "nodes": "10.56.160.38, 10.56.160.40",
        "commands": [
            ROOT / "commands/tp8_dp8_dpa_nvl_cg16_rank0_server_command.sh",
            ROOT / "commands/tp8_dp8_dpa_nvl_cg16_rank1_server_command.sh",
        ],
    },
}

CASES = ["4k_500", "32k_500", "128k_500"]
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]

METRIC_FIELDS = [
    "mode",
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
    "median_tpot_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "max_concurrent_requests",
]


def load_results():
    rows = {}
    for mode in MODES:
        rows[mode] = {}
        for case in CASES:
            rows[mode][case] = {}
            for concurrency in CONCURRENCIES:
                path = RAW / f"{mode}_{case}_c{concurrency}.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open() as f:
                    data = json.load(f)
                rows[mode][case][concurrency] = data
    return rows


def fnum(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def code_block(path):
    if not path.exists():
        return f"`{path}` missing\n"
    return f"`{path}`\n\n```bash\n{path.read_text().strip()}\n```"


def write_csv(results):
    summary_path = ROOT / "summary_tp4_tp8.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for mode in MODES:
            for case in CASES:
                for c in CONCURRENCIES:
                    data = results[mode][case][c]
                    writer.writerow({field: data.get(field) for field in METRIC_FIELDS} | {
                        "mode": mode,
                        "case": case,
                        "concurrency": c,
                    })

    compare_path = ROOT / "tp4_vs_tp8_comparison.csv"
    fields = [
        "case",
        "concurrency",
        "tp4_output_tok_s",
        "tp8_output_tok_s",
        "tp8_over_tp4",
        "tp4_output_tok_s_per_gpu",
        "tp8_output_tok_s_per_gpu",
        "tp8_over_tp4_per_gpu",
        "tp4_mean_ttft_ms",
        "tp8_mean_ttft_ms",
        "tp4_p99_tpot_ms",
        "tp8_p99_tpot_ms",
    ]
    with compare_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            for c in CONCURRENCIES:
                tp4 = results["tp4_dp4_dpa_nvl_cg16"][case][c]
                tp8 = results["tp8_dp8_dpa_nvl_cg16"][case][c]
                tp4_out = tp4["output_throughput"]
                tp8_out = tp8["output_throughput"]
                writer.writerow({
                    "case": case,
                    "concurrency": c,
                    "tp4_output_tok_s": tp4_out,
                    "tp8_output_tok_s": tp8_out,
                    "tp8_over_tp4": ratio(tp8_out, tp4_out),
                    "tp4_output_tok_s_per_gpu": tp4_out / 4,
                    "tp8_output_tok_s_per_gpu": tp8_out / 8,
                    "tp8_over_tp4_per_gpu": ratio(tp8_out / 8, tp4_out / 4),
                    "tp4_mean_ttft_ms": tp4["mean_ttft_ms"],
                    "tp8_mean_ttft_ms": tp8["mean_ttft_ms"],
                    "tp4_p99_tpot_ms": tp4["p99_tpot_ms"],
                    "tp8_p99_tpot_ms": tp8["p99_tpot_ms"],
                })
    return summary_path, compare_path


def first_result(results, mode):
    return results[mode]["4k_500"][1]


def effective_args_table(results):
    keys = [
        "tp_size",
        "dp_size",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "mem_fraction_static",
        "schedule_conservativeness",
        "moe_a2a_backend",
        "enable_dp_attention",
        "enable_nccl_nvls",
        "cuda_graph_max_bs",
        "quantization",
        "kv_cache_dtype",
    ]
    rows = []
    for mode in MODES:
        info = first_result(results, mode).get("server_info", {})
        for key in keys:
            if key in info:
                rows.append([MODES[mode]["label"], key, info.get(key)])
    return rows


def peak_rows(results):
    rows = []
    for case in CASES:
        peaks = {}
        for mode in MODES:
            best_c = max(
                CONCURRENCIES,
                key=lambda c: results[mode][case][c]["output_throughput"],
            )
            best = results[mode][case][best_c]
            peaks[mode] = (best_c, best["output_throughput"])
        tp4_c, tp4_out = peaks["tp4_dp4_dpa_nvl_cg16"]
        tp8_c, tp8_out = peaks["tp8_dp8_dpa_nvl_cg16"]
        rows.append([
            case,
            f"c{tp4_c} / {fnum(tp4_out)}",
            f"c{tp8_c} / {fnum(tp8_out)}",
            fnum(ratio(tp8_out, tp4_out)),
            fnum(ratio(tp8_out / 8, tp4_out / 4)),
        ])
    return rows


def throughput_table(results, case):
    rows = []
    for c in CONCURRENCIES:
        tp4 = results["tp4_dp4_dpa_nvl_cg16"][case][c]
        tp8 = results["tp8_dp8_dpa_nvl_cg16"][case][c]
        tp4_out = tp4["output_throughput"]
        tp8_out = tp8["output_throughput"]
        rows.append([
            c,
            fnum(tp4_out),
            fnum(tp8_out),
            fnum(ratio(tp8_out, tp4_out)),
            fnum(ratio(tp8_out / 8, tp4_out / 4)),
            fnum(tp4["mean_ttft_ms"], 0),
            fnum(tp8["mean_ttft_ms"], 0),
            fnum(tp4["p99_tpot_ms"], 1),
            fnum(tp8["p99_tpot_ms"], 1),
        ])
    return rows


def scaling_rows(results):
    rows = []
    for case in CASES:
        for mode in MODES:
            c1 = results[mode][case][1]["output_throughput"]
            c64 = results[mode][case][64]["output_throughput"]
            rows.append([
                case,
                MODES[mode]["label"],
                fnum(c1),
                fnum(c64),
                fnum(ratio(c64, c1)),
            ])
    return rows


def check_log(path, pattern):
    if not path.exists():
        return 0
    text = path.read_text(errors="ignore")
    return len(re.findall(pattern, text))


def write_report(results, summary_csv, compare_csv):
    report = ROOT / "tp4_vs_tp8_report.md"
    lines = []
    lines.append("# L20A SGLang TP4 vs TP8 NVLS/MNNVL Performance Report")
    lines.append("")
    lines.append(f"- Generated at: 2026-05-29")
    lines.append(f"- Model path: `/home/admin/DeepSeek-V4-Flash`")
    lines.append(f"- Dataset: `random`; input/output cases: `{', '.join(CASES)}`")
    lines.append(f"- Concurrency: `{', '.join(map(str, CONCURRENCIES))}`; `num_prompts=64`; `warmup_requests=1`")
    lines.append("- Both TP4 and TP8 deployments used NVLS/MNNVL environment and `--enable-nccl-nvls`; CUDA graph max batch size was set to 16.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Raw benchmark JSON/logs: `{RAW}`")
    lines.append(f"- Summary CSV: `{summary_csv}`")
    lines.append(f"- Comparison CSV: `{compare_csv}`")
    lines.append(f"- Server logs: `{ROOT / 'server_logs'}`")
    lines.append(f"- Command files: `{ROOT / 'commands'}`")
    lines.append("")
    lines.append("## Test Status")
    lines.append("")
    lines.append(md_table(
        ["Mode", "Nodes", "GPUs", "Completed cases", "Result"],
        [
            [MODES[m]["label"], MODES[m]["nodes"], MODES[m]["gpus"], f"{len(CASES) * len(CONCURRENCIES)}/21", "PASS"]
            for m in MODES
        ],
    ))
    lines.append("")
    lines.append("## Effective Server Args")
    lines.append("")
    lines.append("The launch commands passed `--chunked-prefill-size 16384`, but SGLang reported a lower effective value through `/server_info` when DPA was enabled.")
    lines.append("")
    lines.append(md_table(["Mode", "Arg", "Effective value"], effective_args_table(results)))
    lines.append("")
    lines.append("## NVLS/MNNVL Evidence")
    lines.append("")
    log_rows = []
    logs = [
        ("TP4", ROOT / "server_logs/tp4_dp4_dpa_nvl_cg16_server.log"),
        ("TP8 rank0", ROOT / "server_logs/tp8_dp8_dpa_nvl_cg16_rank0_server.log"),
        ("TP8 rank1", ROOT / "server_logs/tp8_dp8_dpa_nvl_cg16_rank1_server.log"),
    ]
    for name, path in logs:
        log_rows.append([
            name,
            str(path),
            check_log(path, r"NCCL_MNNVL_ENABLE set by environment to 1"),
            check_log(path, r"NCCL_NVLS_ENABLE set by environment to 1"),
            check_log(path, r"via P2P/MNNVL"),
        ])
    lines.append(md_table(["Log", "Path", "MNNVL env lines", "NVLS env lines", "P2P/MNNVL channel lines"], log_rows))
    lines.append("")
    lines.append("## Peak Output Throughput")
    lines.append("")
    lines.append(md_table(
        ["Case", "TP4 peak concurrency / tok/s", "TP8 peak concurrency / tok/s", "TP8/TP4", "TP8/TP4 per GPU"],
        peak_rows(results),
    ))
    lines.append("")
    lines.append("## Scaling From c1 To c64")
    lines.append("")
    lines.append(md_table(["Case", "Mode", "c1 output tok/s", "c64 output tok/s", "c64/c1"], scaling_rows(results)))
    lines.append("")
    lines.append("## Detailed Comparison")
    lines.append("")
    for case in CASES:
        lines.append(f"### {case}")
        lines.append("")
        lines.append(md_table(
            [
                "Concurrency",
                "TP4 out tok/s",
                "TP8 out tok/s",
                "TP8/TP4",
                "TP8/TP4 per GPU",
                "TP4 mean TTFT ms",
                "TP8 mean TTFT ms",
                "TP4 p99 TPOT ms",
                "TP8 p99 TPOT ms",
            ],
            throughput_table(results, case),
        ))
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- TP8 + DP8 + DPA + NVLS/MNNVL did not deliver expected throughput scaling over single-node TP4. Peak absolute output throughput was lower for `4k_500` and `128k_500`, and only roughly equal for `32k_500`.")
    lines.append("- Per-GPU efficiency regressed heavily in TP8. At peak, TP8 per-GPU output throughput was about 45-50% of TP4, even though NVLS/MNNVL was active in the logs.")
    lines.append("- Low concurrency is especially weak for TP8: `128k_500_c1` was about 0.67x TP4 in absolute output throughput, while using twice the GPU count.")
    lines.append("- Higher concurrency improves TP8 aggregate throughput, but does not remove the scaling gap. Tail latency grows sharply, especially on `128k_500`: TP8 c32 P99 E2E reached about 94s and c64 P99 TTFT reached about 45s.")
    lines.append("")
    lines.append("## Initial Bottleneck Analysis")
    lines.append("")
    lines.append("- NVLS/MNNVL was not simply disabled: rank logs contain `NCCL_MNNVL_ENABLE=1`, `NCCL_NVLS_ENABLE=1`, and many `via P2P/MNNVL` channel lines.")
    lines.append("- The result is more consistent with runtime/scheduling and cross-rank efficiency loss than with missing link enablement. TP8 doubles TP ranks and introduces cross-node collectives, while this benchmark has only 64 prompts, so low and medium concurrency leave DP groups lightly loaded.")
    lines.append("- Effective prefill chunk size reported by SGLang was lower than the launch value. TP8 reported `chunked_prefill_size=2048`; this increases scheduling turns for long prompts and can make the 128k cases more sensitive to per-rank stragglers.")
    lines.append("- CUDA graph coverage is not uniform on long-context/high-concurrency cases. Server logs show long prefill steps with `cuda graph: False`, and some decode phases also fall back when batch/shape does not fit captured graphs. Since `--cuda-graph-max-bs=16`, c32/c64 can still suffer from uncaptured or fragmented shapes.")
    lines.append("- `128k_500` tail behavior suggests straggler amplification across DP/TP ranks. Aggregate throughput improves with concurrency, but TTFT and TPOT tails grow, indicating that requests wait behind long prefill/decode phases rather than benefiting cleanly from the extra GPUs.")
    lines.append("")
    lines.append("Profiling is intentionally deferred until the PD-separated report is completed, per task order. If TP8 remains worse after PD testing, the first profiling target should be `32k_500_c1` and `128k_500_c32/c64`, with focus on prefill kernels, NCCL/NVSHMEM collectives, CUDA graph hit rate, and DP scheduler imbalance.")
    lines.append("")
    lines.append("## Container Creation Command")
    lines.append("")
    lines.append(code_block(ROOT / "scripts/recreate_sgl_container.sh"))
    lines.append("")
    lines.append("## Deployment Commands")
    lines.append("")
    for mode in MODES:
        lines.append(f"### {MODES[mode]['label']}")
        lines.append("")
        for command in MODES[mode]["commands"]:
            lines.append(code_block(command))
            lines.append("")
    report.write_text("\n".join(lines))
    return report


def main():
    results = load_results()
    summary_csv, compare_csv = write_csv(results)
    report = write_report(results, summary_csv, compare_csv)
    print(report)
    print(summary_csv)
    print(compare_csv)


if __name__ == "__main__":
    main()
