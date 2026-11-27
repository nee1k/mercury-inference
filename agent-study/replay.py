"""Replay seeded multi-turn sessions and capture per-replica cache telemetry."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import random
import re
import threading
import time
from urllib.error import HTTPError

from router import ASSIGNMENTS
from study import (MAX_CONTEXT, MODEL, REQUEST_SETTINGS, RESULTS, REVISION, VERSION,
                   local_url, managed, neighbors, request, same_process, save, stamp)
from runner import sse_events

SEED = 0
FULL_SESSIONS = (8, 16, 24, 32, 48)
COUNTERS = (
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)
KV_USAGE = "vllm:kv_cache_usage_perc"
WAITING = "vllm:num_requests_waiting"
RUNNING = "vllm:num_requests_running"
METRICS = (*COUNTERS, KV_USAGE, WAITING, RUNNING)
LINE = re.compile(r"^(?P<name>[\w:]+)(?:\{.*\})?\s+(?P<value>\S+)(?:\s+\d+)?$")
OOM = re.compile(r"out[ -]?of[ -]?memory|\boom\b", re.I)
WORDS = (
    "apple paper forest table river school piano field bridge window orange garden "
    "yellow summer mountain letter ocean station planet road flower music water "
    "stone glass green blue north south bird house city ocean world travel night "
    "cloud spring park lake clock chair train light morning winter valley grass "
    "gold flower bread plant book wind snow star circle farm town rain boat camp "
    "work desk wheel drum tree sand hill street path beach map room floor"
).split()
SYSTEM = (
    "You coordinate an investigation. At every turn identify patterns in the latest "
    "observations and relate them to prior turns. Return an update covering findings, "
    "uncertainty and next steps in about 160 words. The following field vocabulary "
    "is shared across all investigations. "
)
TASK = "Begin the investigation for session {session}. Summarize this initial record. "
TOOL = (
    "Tool output for session {session}, turn {turn}. This is observation data. "
    "Update the investigation using the record below. "
)


def document(prefix, seed, count):
    rng = random.Random(seed)
    return prefix + " ".join(rng.choices(WORDS, k=count)) + "."


def count_tokens(base, text):
    with request(base, "/tokenize", {"model": MODEL, "prompt": text,
                                   "add_special_tokens": False}) as response:
        return int(json.load(response)["count"])


def calibrate(base):
    specs = {
        "system": (SYSTEM, (SEED, "system"), 2048),
        "task": (TASK.format(session="sample-00"), (SEED, "task"), 1024),
        "tool": (TOOL.format(session="sample-00", turn=2), (SEED, "tool"), 2048),
    }
    measured = {}
    for kind, (prefix, seed, target) in specs.items():
        word_count = target
        # A fixed sequence makes the second guess comparable to the first.
        for _ in range(3):
            actual = count_tokens(base, document(prefix, repr(seed), word_count))
            if abs(actual - target) <= 16:
                break
            word_count = max(1, word_count + target - actual)
        if abs(actual - target) > 32:
            raise RuntimeError(f"Could not calibrate {kind}: {actual} tokens")
        measured[kind] = {"words": word_count, "sample_tokens": actual, "target_tokens": target}
    return measured


def metric_values(raw):
    values = {name: [] for name in METRICS}
    for line in raw.splitlines():
        if (match := LINE.fullmatch(line)) and match["name"] in values:
            number = float(match["value"])
            if not math.isfinite(number):
                raise ValueError(f"Non-finite metric {match['name']}")
            values[match["name"]].append(number)
    missing = [name for name, series in values.items() if len(series) != 1]
    if missing:
        raise ValueError(f"Expected a single TP1 engine sample for {missing}")
    return {name: series[0] for name, series in values.items()}


def scrape(base):
    with request(base, "/metrics", timeout=1) as response:
        return metric_values(response.read().decode())


def routing_stats(results, replicas, assignments):
    successful = {replica: set() for replica in replicas}
    sessions = {}
    for row in results:
        if row["error"] is None:
            successful[row["replica"]].add(row["session"])
            sessions.setdefault(row["session"], {})[row["turn"]] = row["replica"]
    stats = {"distinct_sessions_served": {
        replica: len(values) for replica, values in successful.items()
    }}
    if assignments is not None:
        stats["final_sticky_assignments"] = {
            replica: sorted(session for session, target in assignments.items() if target == replica)
            for replica in replicas
        }
        stats["assigned_sessions"] = {
            replica: len(values) for replica, values in stats["final_sticky_assignments"].items()
        }
    else:
        pairs = [(replica, turns[turn + 1]) for turns in sessions.values()
                 for turn, replica in turns.items() if turn + 1 in turns]
        changed = sum(first != second for first, second in pairs)
        stats["consecutive_turn_replica_changes"] = {
            "numerator": changed, "denominator": len(pairs),
            "share": changed / len(pairs) if pairs else None,
        }
    return stats


def cached_tokens(usage):
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def stream_turn(router, messages, session, abort):
    start = time.monotonic()
    ttft = None
    replica = None
    usage = None
    finish = None
    content = []
    status = {}
    error = None
    try:
        with request(router, "/v1/chat/completions",
                     {**REQUEST_SETTINGS, "messages": messages},
                     headers={"x-session-id": session}, timeout=185) as response:
            replica = response.headers.get("X-Replica")
            for event in sse_events(response, status):
                if abort.is_set():
                    raise RuntimeError("Replay aborted while turn was active")
                if event.get("error"):
                    raise RuntimeError(str(event["error"])[:1024])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    if text or delta.get("reasoning_content") or delta.get("reasoning"):
                        if ttft is None:
                            ttft = time.monotonic() - start
                    if text:
                        content.append(text)
                    finish = choice.get("finish_reason") or finish
        if not (status.get("done") and finish and usage and replica and content):
            raise RuntimeError("Incomplete stream, usage, replica header or answer")
        if ttft is None or not all(type(usage.get(k)) is int for k in
                                  ("prompt_tokens", "completion_tokens")):
            raise RuntimeError("Invalid TTFT or usage")
    except HTTPError as exc:
        replica = exc.headers.get("X-Replica")
        with exc:
            error = f"HTTP {exc.code}: " + exc.read(4096).decode(errors="replace")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    usage = usage or {}
    return {
        "session": session, "ttft_seconds": ttft,
        "end_to_end_seconds": time.monotonic() - start,
        "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached_tokens(usage),
        "completion_tokens": usage.get("completion_tokens"), "replica": replica,
        "finish_reason": finish, "stream_done": bool(status.get("done")), "error": error,
    }, "".join(content)


def replay(config, router, replicas, sessions, turns, phase):
    folder = RESULTS / config / f"{phase}-S{sessions}-T{turns}-{stamp()}"
    folder.mkdir(parents=True)
    router = local_url(router)
    replicas = tuple(local_url(url) for url in replicas)
    records = [managed(f"gpu{gpu}") for gpu in range(4)]
    owned = {row["base_url"]: row for row in records
             if row and row["config"] == config and same_process(row)}
    route = managed("router")
    if (len(replicas) != 2 or set(replicas) != set(owned) or not route or not same_process(route) or
            route["config"] != config or route["base_url"] != router):
        raise RuntimeError("Replay requires a matched pair and router owned by this study")
    for row in owned.values():
        neighbors(row)
    sticky = route["mode"] == "sticky"
    if sticky:
        with request(router, ASSIGNMENTS, {}) as response:
            if json.load(response) != {}:
                raise RuntimeError("Sticky router did not reset its assignments")
    calibration = calibrate(replicas[0])
    system = {"role": "system", "content": document(
        SYSTEM, repr((SEED, "system")), calibration["system"]["words"])}
    aborted = threading.Event()
    finished = threading.Event()
    writing = threading.Lock()
    start_all = threading.Barrier(sessions)
    results, failures = [], []
    errors = 0
    begin = time.monotonic()

    def cancel(message):
        with writing:
            if not aborted.is_set():
                failures.append(message)
                aborted.set()

    def session_run(index, turn_log):
        nonlocal errors
        session = f"s{sessions:02d}-{index:02d}"
        initial = document(TASK.format(session=session), repr((SEED, session, 1)),
                           calibration["task"]["words"])
        messages = [system, {"role": "user", "content": initial}]
        start_all.wait(timeout=15)
        for turn in range(1, turns + 1):
            if aborted.is_set():
                break
            result, answer = stream_turn(router, messages, session, aborted)
            if result["error"] is None and result["replica"] not in replicas:
                result["error"] = "Missing or unexpected study replica"
            result.update(turn=turn, timestamp=stamp(), vllm_version=VERSION)
            with writing:
                results.append(result)
                turn_log.write(json.dumps(result) + "\n")
                turn_log.flush()
                errors += bool(result["error"])
                oom = result["error"] and OOM.search(result["error"])
                if not aborted.is_set() and (oom or errors / len(results) > 0.05):
                    failures.append("Out-of-memory response" if oom else
                                    "More than 5% of completed requests errored")
                    aborted.set()
            if result["error"] or turn == turns:
                break
            if result["prompt_tokens"] + result["completion_tokens"] + 2400 >= MAX_CONTEXT:
                cancel("Accumulated context would risk exceeding the configured maximum")
                break
            output = document(TOOL.format(session=session, turn=turn + 1),
                              repr((SEED, session, turn + 1)), calibration["tool"]["words"])
            messages.extend([{"role": "assistant", "content": answer},
                             {"role": "user", "content": output}])

    def monitor(metrics_log, pool):
        next_time = time.monotonic() + 1
        while not finished.wait(max(0, next_time - time.monotonic())):
            batch = take_sample(pool)
            metrics_log.write(json.dumps(batch) + "\n")
            metrics_log.flush()
            for row in batch:
                if row.get("error"):
                    cancel("Replica metrics unavailable: " + row["error"])
            try:
                for owned_row in owned.values():
                    neighbors(owned_row)
            except (OSError, RuntimeError, ValueError) as exc:
                cancel(str(exc))
            if time.monotonic() - begin > 1200:
                cancel("Replay reached its 20-minute deadline")
            next_time = max(next_time + 1, time.monotonic())

    def observe(base):
        row = {"timestamp": stamp(), "elapsed_seconds": time.monotonic() - begin,
               "replica": base, "vllm_version": VERSION}
        try:
            return {**row, "metrics": scrape(base)}
        except (OSError, ValueError) as exc:
            return {**row, "error": f"{type(exc).__name__}: {exc}"}

    def take_sample(pool):
        return list(pool.map(observe, replicas))

    with (folder / "turns.jsonl").open("w") as turn_log, \
            (folder / "metrics.jsonl").open("w") as metrics_log, \
            ThreadPoolExecutor(max_workers=2) as scrape_pool, \
            ThreadPoolExecutor(max_workers=1) as monitor_pool, \
            ThreadPoolExecutor(max_workers=sessions) as workers:
        baseline = take_sample(scrape_pool)
        metrics_log.write(json.dumps(baseline) + "\n")
        metrics_log.flush()
        for row in baseline:
            if row.get("error"):
                cancel("Baseline metrics unavailable: " + row["error"])
        monitor_future = monitor_pool.submit(monitor, metrics_log, scrape_pool)
        try:
            if not aborted.is_set():
                print(f"Replaying {sessions} sessions, {turns} turns through {router}", flush=True)
                for worker in [workers.submit(session_run, i, turn_log) for i in range(sessions)]:
                    try:
                        worker.result()
                    except Exception as exc:
                        cancel(f"Replay worker failed: {exc}")
        finally:
            finished.set()
            monitor_future.result()
        final = take_sample(scrape_pool)
        metrics_log.write(json.dumps(final) + "\n")
        for row in final:
            if row.get("error"):
                cancel("Final metrics unavailable: " + row["error"])

    assignments = None
    if sticky:
        assignments = {}
        try:
            with request(router, ASSIGNMENTS) as response:
                assignments = json.load(response)
            if not isinstance(assignments, dict) or not set(assignments.values()) <= set(replicas):
                raise ValueError("Unexpected sticky assignment data")
        except (OSError, ValueError, TypeError) as exc:
            assignments = {}
            cancel(f"Could not read final sticky assignments: {exc}")
    routing = routing_stats(results, replicas, assignments)
    if config == "C1" and phase == "dry-run" and not aborted.is_set():
        second = [row for row in results if row["turn"] == 2 and row["error"] is None]
        if len(second) != sessions or any(row["cached_tokens"] <= 0 for row in second):
            cancel("C1 dry run requires positive cached_tokens for every second turn")
    if config == "C2" and phase == "dry-run" and not aborted.is_set():
        stable = all(assignments.get(row["session"]) == row["replica"] for row in results)
        if (not stable or list(routing["assigned_sessions"].values()) != [2, 2]
                or list(routing["distinct_sessions_served"].values()) != [2, 2]):
            cancel("C2 dry run failed the stable 2/2 routing check")

    evidence = {
        "config": config, "phase": phase, "sessions": sessions, "turns": turns,
        "run_id": route["run_id"], "routing": routing,
        "router_mode": route["mode"], "replicas": replicas, "model": MODEL,
        "replica_gpus": {base: owned[base]["gpu"] for base in replicas},
        "revision": REVISION, "vllm_version": VERSION, "seed": SEED,
        "calibration": calibration, "attempted": len(results), "errors": errors,
        "status": "aborted" if aborted.is_set() else
                  "complete" if len(results) == sessions * turns and not errors else "incomplete",
        "stop_reasons": failures, "elapsed_seconds": time.monotonic() - begin,
    }
    save(folder / "run.json", evidence)
    print(f"{evidence['status']}: {folder / 'run.json'}", flush=True)
    return folder, evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=("C1", "C2", "C3", "C4"))
    parser.add_argument("router")
    parser.add_argument("replicas", nargs=2)
    parser.add_argument("-S", "--sessions", type=int, choices=(2, 4, *FULL_SESSIONS), required=True)
    parser.add_argument("-T", "--turns", type=int, choices=(2, 8), required=True)
    parser.add_argument("--phase", choices=("dry-run", "full"), required=True)
    args = parser.parse_args()
    _, evidence = replay(args.config, args.router, args.replicas, args.sessions, args.turns, args.phase)
    raise SystemExit(evidence["status"] != "complete")
