"""Launch and control Mercury's private single-node Ray Serve cluster."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import socketserver
import stat
import threading
import time


STATE = Path(f"/tmp/mercury-qwen3-{os.getuid()}")
CONTROL = STATE / "control.sock"
API_PORT = 8180
INDICES = range(4)


class Manager:
    def __init__(self, serve, config, inventory, snapshot):
        self.serve = serve
        self.config = config
        self.inventory = inventory
        self.snapshot = snapshot
        self.ports = {index: API_PORT + 1 + index for index in INDICES}
        self.apps = {}
        self.lifecycle = dict.fromkeys(INDICES, "starting")
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()

    @staticmethod
    def name(index):
        return f"qwen3-gpu-{index}"

    def target(self, index):
        gpu = self.inventory[index]
        app = self.config.ModelReplica.options(
            ray_actor_options={"num_cpus": 2, "resources": {f"mercury_gpu_{index}": 1}}
        ).bind(
            index,
            gpu["uuid"],
            gpu["memory_mib"],
            self.ports[index],
            str(self.snapshot),
            str(STATE),
            str(STATE / "vllm-cache"),
        )
        return self.serve.RunTarget(app, name=self.name(index), route_prefix=None)

    def start(self):
        self.serve.start(
            proxy_location="HeadOnly",
            http_options={"host": self.config.HOST, "port": API_PORT},
        )
        self.router = self.serve.run(
            self.config.Router.bind(self.ports), name="qwen3-router"
        )
        handles = self.serve.run_many([self.target(index) for index in INDICES])
        for index, handle in zip(INDICES, handles, strict=True):
            self.apps[index] = handle
            self.router.include.remote(index).result(timeout_s=10)
            self.lifecycle[index] = "serving"

    def require_idle(self, index):
        gpu = self.inventory[index]
        self.config.require_idle(index, gpu["uuid"], gpu["memory_mib"])

    def status(self):
        live = self.config.gpu_inventory()
        routing = self.router.status.remote().result(timeout_s=10)
        return {
            "endpoint": f"http://{self.config.HOST}:{API_PORT}/v1",
            "gpus": {
                index: {
                    "state": self.lifecycle[index],
                    **routing[index],
                    **live[index],
                    "idle_mib": self.inventory[index]["memory_mib"],
                    "expected_uuid": self.inventory[index]["uuid"],
                }
                for index in INDICES
            },
        }

    def verify_free(self, index, timeout=25):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.require_idle(index)
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

    def release(self, index, timeout):
        if self.lifecycle[index] == "released":
            return f"GPU {index} is already released"
        if self.lifecycle[index] == "serving":
            self.lifecycle[index] = "draining"
            try:
                drained = self.router.drain.remote(index, timeout).result(
                    timeout_s=timeout + 10
                )
            except Exception:
                self.router.include.remote(index).result(timeout_s=10)
                self.lifecycle[index] = "serving"
                raise
            if not drained:
                self.lifecycle[index] = "serving"
                raise RuntimeError(
                    f"GPU {index} drain timed out after {timeout:g}s. GPU is still held."
                )
        self.lifecycle[index] = "stopping"
        try:
            if index in self.apps:
                self.apps[index].stop.remote().result(timeout_s=30)
        finally:
            try:
                self.serve.delete(self.name(index))
                self.apps.pop(index, None)
                self.verify_free(index)
            except Exception:
                self.lifecycle[index] = "cleanup_pending"
                raise
            else:
                self.lifecycle[index] = "released"
        return f"GPU {index} released to its idle VRAM baseline"

    def restore(self, index):
        if self.lifecycle[index] == "serving":
            return f"GPU {index} is already serving"
        self.require_idle(index)
        self.lifecycle[index] = "starting"
        try:
            (self.apps[index],) = self.serve.run_many([self.target(index)])
            self.router.include.remote(index).result(timeout_s=10)
        except Exception:
            self.lifecycle[index] = "cleanup_pending"
            # Keep the GPU out of rotation and verify cleanup before accepting a retry.
            self.router.drain.remote(index, 1).result(timeout_s=10)
            self.serve.delete(self.name(index))
            self.apps.pop(index, None)
            self.verify_free(index)
            self.lifecycle[index] = "released"
            raise
        self.lifecycle[index] = "serving"
        return f"GPU {index} restored and serving"

    def dispatch(self, request):
        verb = request["command"]
        if verb == "status":
            return self.status()
        with self.lock:
            if self.stop_requested.is_set():
                raise RuntimeError("Service is stopping")
            if verb == "stop":
                for index in INDICES:
                    self.release(index, 120)
                self.stop_requested.set()
                return "All model instances stopped"
            index = request["gpu"]
            if index not in INDICES:
                raise ValueError("GPU index must be between 0 and 3")
            if verb == "restore":
                return self.restore(index)
            if verb == "release":
                timeout = request["timeout"]
                if not math.isfinite(timeout) or not 0 < timeout <= 120:
                    raise ValueError(
                        "Drain timeout must be greater than zero and at most 120 seconds"
                    )
                return self.release(index, timeout)
            raise ValueError(f"Unknown command {verb}")


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        try:
            result = {
                "result": self.server.manager.dispatch(
                    json.loads(self.rfile.readline(4096))
                )
            }
        except Exception as exc:
            result = {"error": str(exc)}
        try:
            self.wfile.write(json.dumps(result).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def model_snapshot():
    hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (
            Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
            / "hub"
        )
    )
    repo = hub / "models--Qwen--Qwen3-8B"
    revision = (repo / "refs/main").read_text().strip()
    snapshot = repo / "snapshots" / revision
    if not (snapshot / "config.json").is_file() or not list(
        snapshot.glob("*.safetensors")
    ):
        raise RuntimeError(f"Cached Qwen3-8B weights not found at {snapshot}")
    return snapshot.resolve()


def launch():
    STATE.mkdir(mode=0o700, exist_ok=True)
    info = STATE.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise RuntimeError(
            f"{STATE} must be a private directory owned by the current user"
        )
    with (STATE / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "A Qwen3 service is already starting or running"
            ) from exc
        run_cluster()


def run_cluster():
    import ray
    from ray import serve
    import qwen3_serve as config

    snapshot = model_snapshot()
    for port in range(API_PORT, API_PORT + 5):
        config.require_port(port)
    inventory = config.gpu_inventory()
    if (
        set(inventory) != set(INDICES)
        or len({gpu["uuid"] for gpu in inventory.values()}) != 4
    ):
        raise RuntimeError("Expected four distinct physical GPUs numbered 0 through 3")
    if sorted(inventory, key=lambda index: inventory[index]["pci_bus_id"]) != list(
        INDICES
    ):
        raise RuntimeError("The physical GPU indices do not match CUDA's PCI bus order")
    for index, gpu in inventory.items():
        if gpu["compute_pids"] or gpu["memory_mib"] > 64:
            raise RuntimeError(f"GPU {index} is already occupied: {gpu}")
    print(f"Launching from {snapshot}. GPU inventory: {inventory}", flush=True)

    def stop_startup(*_):
        raise SystemExit("Qwen3 startup interrupted")

    signal.signal(signal.SIGTERM, stop_startup)
    ray.init(
        address="local",
        _node_ip_address=config.HOST,
        _temp_dir=str(STATE / "ray"),
        include_dashboard=False,
        num_cpus=12,
        num_gpus=0,
        object_store_memory=256 * 1024 * 1024,
        resources={f"mercury_gpu_{index}": 1 for index in INDICES},
    )
    try:
        manager = Manager(serve, config, inventory, snapshot)
        manager.start()
        signal.signal(signal.SIGTERM, lambda *_: manager.stop_requested.set())
        CONTROL.unlink(missing_ok=True)
        with socketserver.ThreadingUnixStreamServer(
            str(CONTROL), ControlHandler
        ) as server:
            server.manager = manager
            server.timeout = 0.25
            print(
                f"Ready at http://{config.HOST}:{API_PORT}/v1. Logs: {STATE}",
                flush=True,
            )
            while not manager.stop_requested.is_set():
                server.handle_request()
    finally:
        CONTROL.unlink(missing_ok=True)
        try:
            serve.shutdown()
        finally:
            ray.shutdown()


def control(args):
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(CONTROL))
        client.sendall(json.dumps(vars(args)).encode() + b"\n")
        with client.makefile("rb") as stream:
            response = json.loads(stream.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    result = response["result"]
    print(result if isinstance(result, str) else json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for verb in ("start", "status", "release", "restore", "stop"):
        command = commands.add_parser(verb)
        if verb in ("release", "restore"):
            command.add_argument("gpu", type=int, choices=INDICES)
        if verb == "release":
            command.add_argument(
                "--timeout", type=float, default=120, help="Drain seconds, up to 120"
            )
    args = parser.parse_args()
    try:
        if args.command == "start":
            launch()
        else:
            control(args)
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
