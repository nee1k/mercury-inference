"""Summarize validated evidence; provisional capability proxy is not quality."""
import argparse
import collections
import statistics
from pathlib import Path

from common import ROOT, load, save


def collect(root=ROOT):
    groups = collections.defaultdict(list)
    rejected = []
    for path in (Path(root) / "results").glob("**/cell.json"):
        record = load(path)
        if record.get("kind") not in ("sweep", "frontier-repeat", "pilot"):
            continue
        if record.get("status") != "VALID":
            rejected.append(str(path))
            continue
        raw_path = Path(record["raw_file"])
        if not raw_path.is_file():
            rejected.append(str(path))
            continue
        from benchmark import validate_result
        raw = load(raw_path)
        if validate_result(raw, record["prompts"], record["output_tokens"], record["input_tokens"]):
            rejected.append(str(path))
            continue
        key = (record["config"], record["shape"], record["concurrency"], "pilot" if record["kind"] == "pilot" else "study")
        groups[key].append((record, raw, str(path)))
    rows = []
    for key, items in sorted(groups.items()):
        config, shape, concurrency, scope = key
        first = items[0][0]
        row = {"config": config, "shape": shape, "concurrency": concurrency, "scope": scope,
               "gpus": first["gpus"], "precision": first["precision"], "reasoning": "off",
               "total_parameters_b": first["total_parameters_b"], "runs": len(items),
               "sources": [path for _, _, path in items], "vllm_versions": sorted({record["vllm_version"] for record, _, _ in items})}
        for metric in ("ttft", "tpot", "itl", "e2el"):
            for percentile in (50, 95, 99):
                name = "p%d_%s_ms" % (percentile, metric)
                row[name] = statistics.median(raw[name] for _, raw, _ in items)
        row["output_tokens_per_second"] = statistics.median(raw["output_throughput"] for _, raw, _ in items)
        row["output_tokens_per_second_per_gpu"] = row["output_tokens_per_second"] / row["gpus"]
        row["failed_requests"] = sum(raw["failed"] for _, raw, _ in items)
        originals = [(record, raw) for record, raw, _ in items if record["kind"] == "sweep"]
        row["median_movement_percent"] = None
        if len(items) >= 3 and len(originals) == 1 and originals[0][1]["p95_e2el_ms"] > 0:
            row["median_movement_percent"] = 100 * (row["p95_e2el_ms"] / originals[0][1]["p95_e2el_ms"] - 1)
        rows.append(row)
    return rows, rejected


def dominates(a, b):
    left = (a["gpus"] <= b["gpus"], a["p95_e2el_ms"] <= b["p95_e2el_ms"], a["total_parameters_b"] >= b["total_parameters_b"])
    strict = (a["gpus"] < b["gpus"] or a["p95_e2el_ms"] < b["p95_e2el_ms"] or a["total_parameters_b"] > b["total_parameters_b"])
    return all(left) and strict


def frontiers(rows):
    result = {}
    for shape in ("chat", "long-doc", "long-gen"):
        candidates = [row for row in rows if row["shape"] == shape and row["concurrency"] == 16 and row["scope"] == "study"]
        result[shape] = [row["config"] for row in candidates if not any(dominates(other, row) for other in candidates)]
    return result


def run(root=ROOT):
    root = Path(root)
    rows, rejected = collect(root)
    frontier = frontiers(rows)
    result = {"rows": rows, "rejected_cells": rejected, "frontiers": frontier,
              "repeat_union": sorted(set(config for configs in frontier.values() for config in configs)),
              "capability_proxy_warning": "Total parameters are a rough stand-in for capability, not a quality measure"}
    save(root / "results" / "summary.json", result)
    lines = ["# Serving study summary", "", "Provisional workload; serving cost only. Total parameters are a rough capability proxy, not measured quality.", "",
             "Pilot and gate runs do not count as the size sweep or frontier repeats. All required percentiles are retained in summary.json and raw result files.", "",
             "| Configuration | Scope | Shape | C | Runs | p95 TTFT ms | p95 TPOT ms | p95 ITL ms | p95 E2E ms | Output tok/s/GPU | Failed | Evidence |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        sources = ", ".join("[cell](" + str(Path(path).relative_to(root / "results")) + ")" for path in row["sources"])
        lines.append("| %s | %s | %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %.2f | %d | %s |" % (
            row["config"], row["scope"], row["shape"], row["concurrency"], row["runs"], row["p95_ttft_ms"], row["p95_tpot_ms"],
            row["p95_itl_ms"], row["p95_e2el_ms"], row["output_tokens_per_second_per_gpu"], row["failed_requests"], sources))
    lines.extend(["", "Frontiers use concurrency 16 only, one per shape. Axes: fewer GPUs, lower p95 E2E, more total parameters.", ""])
    for shape, configs in frontier.items():
        lines.append("- " + shape + ": " + (", ".join(configs) or "No eligible evidence yet"))
    lines.extend(["", "Rejected cells: " + str(len(rejected)) + ". See [summary.json](summary.json) for sources and repeat movement.", ""])
    (root / "results" / "summary.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print("Summary written; repeat union: " + ", ".join(result["repeat_union"]))
