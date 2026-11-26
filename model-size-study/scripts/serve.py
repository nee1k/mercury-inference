"""Offline, GPU-0-only server lifecycle with process-identity checks."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from common import (CONFIG, MODEL, ROOT, VLLM, check_neighbors, gpu_snapshot, http,
                    identity, idle_sample, load, owned_server, runtime_env,
                    same_process, save, stamp, state_path, verify_cache, version)


def free_port():
    for port in range(8100, 8200):
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            pass
    raise RuntimeError("No free study port")


def server_metrics(state):
    with open(state["server_log"], "rb") as handle:
        handle.seek(state["log_start_byte"])
        text = handle.read().decode(errors="replace")
    cache = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    concurrency = re.findall(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", text)
    return {"kv_cache_tokens": int(cache[-1].replace(",", "")) if cache else None,
            "maximum_concurrency": float(concurrency[-1][1]) if concurrency else None,
            "concurrency_context_tokens": int(concurrency[-1][0].replace(",", "")) if concurrency else None,
            "source": state["server_log"], "start_byte": state["log_start_byte"]}


def start(config, gpu_ids, port):
    print("Preparing selected offline configuration; checking cache and existing study state", flush=True)
    if config != CONFIG or gpu_ids != "0" or not 8100 <= port <= 8199:
        raise ValueError("Only selected configuration on GPU 0, port 8100–8199 is allowed")
    selected = load(ROOT / "state" / "selection.json")
    if selected["selected"] != [MODEL]:
        raise ValueError("Selection does not match this launcher")
    path = state_path(config)
    if path.exists() and same_process(load(path)):
        raise ValueError("Study server is already alive; refusing a second launch")
    cache = verify_cache()
    (ROOT / "results" / config).mkdir(exist_ok=True)
    launch_dir = ROOT / "results" / config / ("launch-" + stamp())
    launch_dir.mkdir()
    print("Reading installed vLLM version (imports may take a while)", flush=True)
    installed_version = version()
    with (ROOT / "state" / "gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Checking GPU 0 idleness throughout the required sample", flush=True)
        observed = idle_sample()
        save(launch_dir / "idle-preflight.json", observed)
        if not observed["idle"]:
            raise ValueError("GPU 0 is not idle; see " + str(launch_dir / "idle-preflight.json"))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        args = ["setsid", "nohup", VLLM, "serve", "--config",
                str(ROOT / "configs" / (config + ".yaml")), "--host", "127.0.0.1", "--port", str(port)]
        log_path = ROOT / "results" / config / "server.log"
        with log_path.open("ab") as log, open(os.devnull, "rb") as null:
            offset = log.tell()
            begin = time.monotonic()
            process = subprocess.Popen(args, stdin=null, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=ROOT, env=runtime_env([0]))
        with (ROOT / "state" / "pids.txt").open("a") as pids:
            pids.write(str(process.pid) + "\n")
            pids.flush()
            os.fsync(pids.fileno())
        state = identity(process.pid)
        state.update(config=config, model=MODEL, revision=cache["revision"], gpu_ids=[0],
                     gpu_uuid=observed["samples"][-1]["uuid"], base_url="http://127.0.0.1:" + str(port),
                     command=args, vllm_version=installed_version, cache_verification=cache,
                     launched_at=stamp(), status="STARTING", server_log=str(log_path),
                     log_start_byte=offset, launch_directory=str(launch_dir),
                     fixed_settings={"max_model_len": 32768, "gpu_memory_utilization": 0.90,
                                     "seed": 0, "prefix_caching": "installed default", "dtype": "bfloat16", "tp": 1})
        save(path, state)
        save(launch_dir / "launch.json", state)
        print("Server launched; initialization log: " + str(log_path), flush=True)
        time.sleep(2)
        if not same_process(state):
            state["status"] = "STARTUP_EXITED"
            save(path, state)
            raise RuntimeError("Server did not survive launcher return; inspect server.log")
        state["survived_detachment_check"] = True
        save(path, state)
        print("Detached server survived. Waiting for health and first completion; do not send test requests until READY", flush=True)
        observations = []
        try:
            while time.monotonic() - begin < 1800:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup; inspect server.log")
                sample = gpu_snapshot()
                observations.append(sample)
                neighbors = check_neighbors(state, sample)
                if neighbors:
                    raise RuntimeError("Startup invalidated: " + "; ".join(neighbors))
                try:
                    # /health returns an empty HTTP body, so use the model list as
                    # the JSON readiness probe and independently require /health.
                    import urllib.request
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(state["base_url"] + "/health", timeout=5) as response:
                        if response.status != 200:
                            raise ValueError("Health endpoint not ready")
                    models = http(state["base_url"], "/v1/models", timeout=5)
                    if MODEL not in [item["id"] for item in models.get("data", [])]:
                        raise ValueError("Model list does not match selected model")
                    completion = http(state["base_url"], "/v1/chat/completions", {
                        "model": MODEL, "messages": [{"role": "user", "content": "Reply with OK."}],
                        "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
                    if not completion.get("choices", [{}])[0].get("message", {}).get("content"):
                        raise ValueError("No first good completion yet")
                    break
                except (OSError, ValueError):
                    time.sleep(5)
            else:
                raise TimeoutError("Readiness timed out; server left running for inspection")
            state.update(status="READY", ready_at=stamp(), load_seconds=time.monotonic() - begin,
                         server_metrics=server_metrics(state), gpu_after_ready=gpu_snapshot())
            if check_neighbors(state, state["gpu_after_ready"]):
                raise RuntimeError("Unrelated process appeared during first completion")
            save(launch_dir / "first-completion.json", completion)
            save(launch_dir / "models.json", models)
            save(launch_dir / "startup-gpu.json", observations)
            save(launch_dir / "ready.json", state)
            save(path, state)
            print("READY " + state["base_url"] + "; evidence " + str(launch_dir), flush=True)
            return state
        except Exception as error:
            state.update(status="STARTUP_FAILED_REVIEW", error=str(error))
            save(launch_dir / "startup-gpu.json", observations)
            save(path, state)
            raise


def stop(config):
    state = load(state_path(config))
    if not same_process(state):
        raise ValueError("Recorded process is not alive with the same identity; no signal sent")
    owned_server(state["base_url"])
    descriptor = os.pidfd_open(state["pid"])
    try:
        owned_server(state["base_url"])
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)
    observations = []
    for _ in range(120):
        sample = gpu_snapshot()
        observations.append(sample)
        if not same_process(state) and not sample["pids"] and sample["used_mib"] < 1024:
            state.update(status="STOPPED", stopped_at=stamp())
            save(state_path(config), state)
            save(Path(state["launch_directory"]) / "stop-gpu.json", observations)
            print("Study server stopped; GPU memory released", flush=True)
            return
        time.sleep(1)
    save(Path(state["launch_directory"]) / "stop-gpu.json", observations)
    raise RuntimeError("SIGTERM sent; clean exit/GPU release unconfirmed. No stronger signal sent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("config", choices=(CONFIG,))
    parser.add_argument("gpu_ids", nargs="?", default="0")
    parser.add_argument("port", nargs="?", type=int, default=8100)
    args = parser.parse_args()
    if args.action == "start":
        start(args.config, args.gpu_ids, args.port)
    elif args.action == "stop":
        stop(args.config)
    else:
        print(state_path(args.config).read_text())
