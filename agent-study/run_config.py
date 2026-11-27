"""Run one approved configuration and clean up all verified study processes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time
from unittest.mock import patch
from urllib.error import URLError

import c1_probe
from replay import FULL_SESSIONS, replay
from serve import FLAGS, verify_flags
from shutdown import release
from study import (
    CONFIGS, MODEL, PORTS, RESULTS, ROOT, VERSION, available_port, base_url, gpu_snapshot,
    launch, managed, neighbors, request, same_process, sample_idle, save, stamp,
)
from summarize import (current_sweeps, make_summary, relative, sweep_onset, write_paper_values,
                       write_run_result)


def run_function_checks(config, record):
    # The existing checks are scoped to their 8B deployment. Supply the
    # verified 32B study record and GPU telemetry without changing that study.
    import function_checks

    def owned_server(base):
        if base != record["base_url"]:
            raise ValueError("Function checks must target the selected study replica")
        neighbors(record)
        return record

    def check_neighbors(state, sample):
        try:
            neighbors(state)
        except (OSError, RuntimeError, ValueError) as error:
            return [str(error)]
        return []

    folder = RESULTS / config / f"function-checks-{stamp()}"
    with patch.multiple(function_checks, MODEL=MODEL, owned_server=owned_server,
                        gpu_snapshot=lambda: neighbors(record), check_neighbors=check_neighbors):
        result_path = function_checks.run(record["base_url"], folder)
    evidence = json.loads(result_path.read_text())
    print(f"Function checks {evidence['status']}: {result_path}", flush=True)
    return result_path


def start_router(config, run_id, router_port, replicas):
    mode = CONFIGS[config][0]
    endpoint = base_url(router_port)
    folder = RESULTS / config
    record = launch(
        "router", [sys.executable, "-B", str(ROOT / "router.py"), mode,
                   str(router_port), *replicas],
        folder / f"router-{stamp()}.log",
        config=config, run_id=run_id, port=router_port, base_url=endpoint, mode=mode,
        vllm_version=VERSION,
    )
    expires = time.monotonic() + 20
    while same_process(record) and time.monotonic() < expires:
        try:
            with request(endpoint, "/v1/models", timeout=2) as response:
                payload = json.load(response)
            if MODEL in [row["id"] for row in payload["data"]]:
                return endpoint
        except (URLError, TimeoutError):
            pass
        time.sleep(0.4)
    raise RuntimeError("Study router did not become ready")


def controllers(config, run_id, gpus, ports):
    launchers = []
    for gpu, port in zip(gpus, ports):
        command = [str(ROOT / "serve.sh"), config, str(gpu), str(port), *CONFIGS[config][1]]
        env = {**os.environ, "MERCURY_AGENT_STUDY_RUN_ID": run_id}
        launchers.append(subprocess.Popen(command, env=env))
    expires = time.monotonic() + 950
    while any(process.poll() is None for process in launchers):
        if time.monotonic() >= expires:
            raise TimeoutError("Replica startup exceeded the controller deadline")
        time.sleep(2)
    if any(process.returncode != 0 for process in launchers):
        raise RuntimeError("Replica startup failed; see launcher logs")


def select_gpus(preferred, folder, run_id):
    checks = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks.update(zip(preferred, pool.map(sample_idle, preferred)))
        if not all(check["idle"] for check in checks.values()):
            remaining = [gpu for gpu in range(4) if gpu not in preferred]
            checks.update(zip(remaining, pool.map(sample_idle, remaining)))
    idle = [gpu for gpu, check in checks.items() if check["idle"]]
    path = folder / f"prelaunch-{run_id}.json"
    save(path, {"checks": checks, "idle": idle, "vllm_version": VERSION})
    if len(idle) < 2:
        raise RuntimeError(f"Fewer than two GPUs passed the idle check; see {path.name}")
    return idle[:2]


def select_ports(ports):
    if ports:
        if len(set(ports)) != 3:
            raise ValueError("Choose three distinct study ports")
        for port in ports:
            available_port(port)
        return ports
    free = []
    for port in PORTS:
        try:
            available_port(port)
        except OSError:
            continue
        free.append(port)
        if len(free) == 3:
            return free
    raise RuntimeError("Fewer than three study ports are free")


def run_config(config, phase, gpus, ports):
    dry_shapes = {"C1": (2, 2), "C2": (4, 2)}
    if phase == "dry-run" and config not in dry_shapes:
        raise ValueError("No dry-run shape for this configuration")
    if len(set(gpus)) != 2:
        raise ValueError("Choose two distinct preferred GPU indices")
    folder = RESULTS / config
    folder.mkdir(parents=True, exist_ok=True)
    run_id = stamp()
    sweep = {"config": config, "run_id": run_id, "results": {}, "ready": [],
             "function_checks": None, "onset_s": None, "cleanup": None,
             "status": "incomplete", "vllm_version": VERSION}
    rows = {}
    attempted_launch = False
    try:
        try:
            verify_flags((*FLAGS, "--port", *CONFIGS[config][1]))
        except ValueError as exc:
            sweep.update(status="skipped", reason=str(exc))
            print(str(exc), flush=True)
            return False
        gpus = select_gpus(gpus, folder, run_id)
        ports = select_ports(ports)
        replicas = tuple(base_url(port) for port in ports[:2])
        print(f"Starting {config} {phase} on GPUs {gpus}, replicas {replicas}", flush=True)
        if config == "C1" and phase == "full":
            c1_probe.record(["date", "-u", "+%FT%TZ"])
            c1_probe.gpu_facts(gpus)
        attempted_launch = True
        controllers(config, run_id, gpus, ports[:2])
        sweep["ready"] = [relative(folder / f"gpu{gpu}-{run_id}-ready.json") for gpu in gpus]
        router = start_router(config, run_id, ports[2], replicas)
        if config in ("C3", "C4"):
            record = managed(f"gpu{gpus[0]}")
            sweep["function_checks"] = relative(run_function_checks(config, record))
        shapes = [dry_shapes[config]] if phase == "dry-run" else [(size, 8) for size in FULL_SESSIONS]
        for sessions, turns in shapes:
            run_folder, result = replay(config, router, replicas, sessions, turns, phase)
            path, rows[sessions] = write_run_result(run_folder / "run.json")
            sweep["results"][sessions] = relative(path)
            if result["status"] == "aborted" or (phase == "dry-run" and not rows[sessions]["valid"]):
                raise RuntimeError("Replay did not complete; remaining shapes are paused")
        if phase == "full":
            sweep["onset_s"] = sweep_onset(rows)
            if config == "C1" and all(row["valid"] for row in rows.values()):
                c1_probe.record(["c1_probe.py", "live", str(gpus[0])], study_python=True)
        if all(row["valid"] for row in rows.values()):
            sweep["status"] = "complete"
    except Exception as exc:
        sweep["reason"] = type(exc).__name__
        raise
    finally:
        try:
            if attempted_launch:
                cleanup = release(config, run_id)
                sweep["cleanup"] = cleanup["source"]
                if not cleanup["memory_released"]:
                    sweep["status"] = "cleanup incomplete"
        finally:
            prefix = "sweep" if phase == "full" else "dry-checkpoint"
            path = folder / f"{prefix}-{run_id}.json"
            save(path, sweep)
            print(f"Checkpoint evidence: {path}", flush=True)
            if config == "C1":
                if phase == "full" and sweep["status"] == "complete":
                    write_paper_values({"C1": (path, sweep)},
                                       RESULTS / "paper_values_c1_rerun.json",
                                       include_request_hits=True)
            else:
                write_paper_values(current_sweeps())
            if config == "C1" and phase == "full" and sweep["status"] == "complete":
                for gpu in gpus:
                    c1_probe.record(["c1_probe.py", "copies", str(gpu)], study_python=True)
            summary = make_summary()
            print(f"Summary contains {len(summary)} runs", flush=True)
    return sweep["status"] == "complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("phase", choices=("dry-run", "full"))
    parser.add_argument("--gpus", nargs=2, type=int, choices=range(4), default=(2, 3))
    parser.add_argument("--ports", nargs=3, type=int)
    args = parser.parse_args()
    try:
        ok = run_config(args.config, args.phase, args.gpus, args.ports)
    except Exception as exc:
        save(RESULTS / args.config / f"failure-{stamp()}.json",
             {"error": type(exc).__name__, "vllm_version": VERSION})
        raise
    raise SystemExit(not ok)
