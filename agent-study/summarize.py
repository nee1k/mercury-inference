"""Summarize agent replay results and link to each run's raw evidence."""

import argparse
from collections import Counter
import json
import os

from replay import COUNTERS, FULL_SESSIONS, KV_USAGE, METRICS, RUNNING, WAITING
from study import BF16_CHECKS, CONFIGS, RESULTS, ROOT, save


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = fraction * (len(values) - 1)
    index = int(position)
    return values[index] + (position - index) * (values[min(index + 1, len(values) - 1)] - values[index])


def summarize_run(path):
    run = json.loads(path.read_text())
    turns = read_jsonl(path.with_name("turns.jsonl"))
    batches = read_jsonl(path.with_name("metrics.jsonl"))
    first = [row["ttft_seconds"] for row in turns if row["turn"] == 1 and not row["error"]]
    later = [row["ttft_seconds"] for row in turns if row["turn"] > 1 and not row["error"]]
    latency = [row["end_to_end_seconds"] for row in turns if not row["error"]]
    series = {replica: [] for replica in run["replicas"]}
    for batch in batches:
        for row in batch:
            if row["replica"] in series and set(METRICS) <= row.get("metrics", {}).keys():
                series[row["replica"]].append(row["metrics"])
    preempt, hits, queries = COUNTERS
    deltas = {metric: None for metric in (preempt, hits, queries)}
    metrics_complete = len(batches) >= 2 and all(
        len(batch) == len(series) and {row["replica"] for row in batch} == set(series)
        and all(set(METRICS) <= row.get("metrics", {}).keys() for row in batch)
        for batch in batches
    )
    if metrics_complete:
        for metric in deltas:
            if all(a[metric] <= b[metric] for values in series.values()
                   for a, b in zip(values, values[1:])):
                deltas[metric] = sum(
                    int(values[-1][metric] - values[0][metric]) for values in series.values()
                )
    successful = [row for row in turns if row["error"] is None]
    prompt_tokens = sum(row["prompt_tokens"] for row in successful)
    # Historical traces predate cached-token capture.
    request_hit_pct = None
    if prompt_tokens and all("cached_tokens" in row for row in successful):
        request_hit_pct = 100 * sum(row["cached_tokens"] for row in successful) / prompt_tokens
    contexts = [row["prompt_tokens"] + row["completion_tokens"] for row in turns
                if row["turn"] == run["turns"] and not row["error"]]
    replica_values = [row for rows in series.values() for row in rows]
    by_turn = {}
    for row in turns:
        if not row["error"]:
            by_turn.setdefault(row["turn"], Counter())[row["replica"]] += 1
    turn_counts = Counter((row["session"], row["turn"]) for row in turns
                          if not row["error"] and row.get("stream_done")
                          and row["finish_reason"] and row["ttft_seconds"] is not None
                          and type(row["prompt_tokens"]) is int
                          and type(row["completion_tokens"]) is int)
    valid = (run["status"] == "complete" and len(turns) == len(turn_counts)
             == run["sessions"] * run["turns"]
             and all(count == 1 for count in turn_counts.values())
             and len({session for session, _ in turn_counts}) == run["sessions"]
             and {turn for _, turn in turn_counts} == set(range(1, run["turns"] + 1))
             and all(row["replica"] in series for row in turns)
             and all(value is not None for value in deltas.values())
             and 0 <= deltas[hits] <= deltas[queries])
    return {
        "run": run, "path": path, "first_p50": percentile(first, 0.50),
        "first_p95": percentile(first, 0.95), "later_p50": percentile(later, 0.50),
        "later_p95": percentile(later, 0.95), "e2e_p95": percentile(latency, 0.95),
        "valid": valid, "counter_increases": deltas, "preemptions": deltas[preempt],
        "request_hit_pct": request_hit_pct,
        "prefix_hit_pct": 100 * deltas[hits] / deltas[queries]
        if deltas[hits] is not None and deltas[queries] else None,
        "prefix_hit_rate": deltas[hits] / deltas[queries]
        if deltas[hits] is not None and deltas[queries] else None,
        "peak_kv": max((row[KV_USAGE] for row in replica_values), default=None),
        "peak_waiting": max((row[WAITING] for row in replica_values), default=None),
        "peak_running": {replica: max((row[RUNNING] for row in values if RUNNING in row),
                                      default=None) for replica, values in series.items()},
        "context_max": max(contexts, default=None), "context_min": min(contexts, default=None),
        "by_turn": by_turn,
    }


def relative(path):
    return str(path.relative_to(ROOT))


def write_run_result(path):
    row = summarize_run(path)
    result = path.with_name("result.json")
    save(result, {**row, "path": relative(path),
                  "turns_path": relative(path.with_name("turns.jsonl")),
                  "metrics_path": relative(path.with_name("metrics.jsonl"))})
    return result, row


def sweep_onset(rows):
    if set(rows) != set(FULL_SESSIONS) or not all(row["valid"] for row in rows.values()):
        return None
    return min((size for size, row in rows.items() if row["preemptions"]), default="none")


def paper_entry(path, field, what, format_spec=""):
    data = json.loads(path.read_text())
    number = data[field]
    value = str(number)
    if number is None or value not in path.read_text():
        raise ValueError(f"No source value for {field} in {path}")
    return {"value": value, "display": format(number, format_spec),
            "source": relative(path), "what": what}


def write_paper_values(sweeps, output=RESULTS / "paper_values.json", *, include_request_hits=False):
    paper = {}

    def add(key, path, field, what, spec=""):
        paper[key] = paper_entry(path, field, what, spec)

    for config, (path, sweep) in sweeps.items():
        key = config.lower()
        if config in ("C1", "C3") and sweep["ready"]:
            ready = ROOT / sweep["ready"][0]
            add(f"{key}_weights_GiB", ready, "model_loading_gib", "Model memory at load in GiB")
            add(f"{key}_kv_tokens", ready, "kv_cache_tokens", "GPU KV cache tokens per replica", ",d")
            add(f"{key}_conc32k", ready, "max_concurrency", "Estimated concurrency per replica at 32K")
        if config == "C3" and sweep.get("queueing_onset_s") is not None:
            add("c3_queueing_onset_s", path, "queueing_onset_s", "Smallest tested S with sampled waiting requests")
        if sweep["onset_s"] is not None:
            add(f"{key}_onset_s", path, "onset_s", "Smallest tested S with preemption")
        for size, source in sweep["results"].items():
            result = ROOT / source
            row = json.loads(result.read_text())
            if not row["valid"]:
                continue
            add(f"{key}_s{size}_preemptions", result, "preemptions", "Preemption increase across both GPUs")
            if include_request_hits:
                add(f"{key}_s{size}_request_hit_pct", result, "request_hit_pct",
                    "Percentage of successful prompt tokens served from cache", ".1f")
            if int(size) == 24 and row["prefix_hit_pct"] is not None:
                add(f"{key}_s24_hit_pct", result, "prefix_hit_pct", "Prefix hit rate as a percentage across both GPUs", ".1f")
            if int(size) in (24, 32):
                add(f"{key}_s{size}_ttft_later_s", result, "later_p95", "Turns 2–8 p95 TTFT in seconds", ".3f")
        if config in ("C3", "C4") and sweep["function_checks"]:
            add(f"{key}_function_checks", ROOT / sweep["function_checks"], "status", "Protocol check outcome")
    save(output, paper)


def current_sweeps():
    latest = {}
    for path in sorted(RESULTS.glob("C*/sweep-*.json")):
        sweep = json.loads(path.read_text())
        latest[sweep["config"]] = (path, sweep)
    return latest


def format_number(number, pattern):
    return format(number, pattern) if number is not None else "—"


def make_summary(output=RESULTS / "summary.md"):
    latest = {}
    sweeps = current_sweeps()
    for path in sorted(RESULTS.glob("C*/*/run.json")):
        row = summarize_run(path)
        run = row["run"]
        key = (run["phase"], run["config"], run["sessions"])
        if run["phase"] == "full":
            current = sweeps.get(run["config"])
            if not current or run.get("run_id") != current[1]["run_id"]:
                key = ("historical-full", *key[1:])
        latest[key] = row
    rows = [
        "# Agent serving results", "",
        "Latency uses successful turns and is in seconds. Context is the final-turn range "
        "of prompt plus completion tokens. Counter changes are summed across replicas. "
        "Peak KV is the highest per-replica sample, with 100% meaning full. "
        "Peak running is the highest sampled running-request count on each replica.", "",
        "| Config | Run | Sessions | Turn 1 TTFT p50 / p95 | Later TTFT p50 / p95 | "
        "End to end p95 | Preemptions | Prefix hit rate | Peak KV | Peak waiting | Peak running per replica | "
        "Errors / attempted | Final context | Result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, row in sorted(latest.items()):
        run = row["run"]
        phase, name, sessions = key
        first = " / ".join(format_number(row[f"first_{p}"], ".3f") for p in ("p50", "p95"))
        later = " / ".join(format_number(row[f"later_{p}"], ".3f") for p in ("p50", "p95"))
        e2e = format_number(row["e2e_p95"], ".3f")
        preemptions = format_number(row["preemptions"], "d")
        hits = format_number(row["prefix_hit_rate"], ".1%")
        peak = format_number(row["peak_kv"], ".1%")
        waiting = format_number(row["peak_waiting"], ".0f")
        running = " / ".join(
            f"GPU {run['replica_gpus'][replica]} ({peak:.0f})"
            if replica in run.get("replica_gpus", {}) else f"{replica.rsplit(':', 1)[-1]} ({peak:.0f})"
            for replica, peak in row["peak_running"].items() if peak is not None
        ) or "—"
        context = "–".join(format_number(row[f"context_{kind}"], ",d")
                           for kind in ("min", "max")) if row["context_min"] is not None else "—"
        link = os.path.relpath(row["path"], output.parent)
        result = row["path"].with_name("result.json")
        result_link = (f"[result]({os.path.relpath(result, output.parent)}) · "
                       if result.is_file() else "")
        turns_link = os.path.relpath(row["path"].with_name("turns.jsonl"), output.parent)
        metrics_link = os.path.relpath(row["path"].with_name("metrics.jsonl"), output.parent)
        rows.append(f"| {name} | {phase} | {sessions} | {first} | {later} | {e2e} | "
                    f"{preemptions} | {hits} | {peak} | {waiting} | {running} | "
                    f"{run['errors']} / {run['attempted']} | {context} | [{run['status']}]({link}) · "
                    f"{result_link}[turns]({turns_link}) · [metrics]({metrics_link}) |")
    rows.extend(["", "## First observed preemption", ""])
    for config in CONFIGS:
        if config == BF16_CHECKS:
            continue
        text = "full sweep pending."
        if config in sweeps:
            path, sweep = sweeps[config]
            link = os.path.relpath(path, output.parent)
            onset = sweep["onset_s"]
            if onset == "none":
                text = f"none observed through [S={FULL_SESSIONS[-1]}]({link})."
            elif onset is not None:
                text = f"first observed at [S={onset}]({link})."
            else:
                text = f"[{sweep['status']}]({link})."
        rows.append(f"- {config} — {text}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("\n".join(rows) + "\n")
    temporary.replace(output)
    return latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = make_summary()
    print(f"Wrote {len(rows)} rows to {RESULTS / 'summary.md'}")
