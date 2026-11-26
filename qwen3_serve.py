"""Ray Serve deployments for the Mercury Qwen3-8B endpoint."""

import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
from ray import serve
from starlette.responses import JSONResponse, Response, StreamingResponse


HOST = "127.0.0.1"
MODEL = "Qwen/Qwen3-8B"
START_TIMEOUT = 600
IDLE_VRAM_SLACK_MIB = 8


def require_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, port))


def gpu_inventory():
    def query(fields, kind):
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return [line.split(", ") for line in output.splitlines() if line.strip()]

    inventory = {
        int(index): {
            "uuid": uuid,
            "pci_bus_id": pci_bus_id,
            "memory_mib": int(memory),
            "compute_pids": [],
        }
        for index, uuid, pci_bus_id, memory in query(
            "index,uuid,pci.bus_id,memory.used", "gpu"
        )
    }
    by_uuid = {gpu["uuid"]: gpu for gpu in inventory.values()}
    for uuid, pid in query("gpu_uuid,pid", "compute-apps"):
        by_uuid[uuid]["compute_pids"].append(int(pid))
    return inventory


def require_idle(index, uuid, baseline):
    live = gpu_inventory()[index]
    if live["uuid"] != uuid:
        raise RuntimeError(f"GPU {index} no longer maps to {uuid}")
    # NVIDIA's driver can allocate a few MiB without adding a compute process.
    if live["compute_pids"] or live["memory_mib"] > baseline + IDLE_VRAM_SLACK_MIB:
        raise RuntimeError(
            f"GPU {index} is occupied ({live['memory_mib']} MiB, "
            f"idle baseline {baseline} MiB, compute PIDs {live['compute_pids']})"
        )


@serve.deployment(graceful_shutdown_timeout_s=25)
class ModelReplica:
    def __init__(self, index, uuid, baseline, port, snapshot, log_dir, cache_dir):
        require_idle(index, uuid, baseline)
        require_port(port)
        self.port = port
        read_fd, self.lifeline = os.pipe()
        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # vLLM 0.21 interprets this value as an integer NVML index.
            "CUDA_VISIBLE_DEVICES": str(index),
            "HF_HUB_OFFLINE": "1",
            "VLLM_CACHE_ROOT": cache_dir,
        }
        command = [
            sys.executable,
            str(Path(__file__).with_name("qwen3_group.py")),
            str(read_fd),
            str(Path(sys.executable).with_name("vllm")),
            "serve",
            snapshot,
            "--served-model-name",
            MODEL,
            "--host",
            HOST,
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            "0.30",
            "--max-model-len",
            "4096",
        ]
        try:
            with open(Path(log_dir) / f"gpu-{index}.log", "ab") as log:
                self.guard = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(read_fd,),
                )
        except BaseException:
            os.close(self.lifeline)
            raise
        finally:
            os.close(read_fd)
        try:
            deadline = time.monotonic() + START_TIMEOUT
            with httpx.Client(timeout=2, trust_env=False) as client:
                while time.monotonic() < deadline:
                    self.check_health()
                    try:
                        models = client.get(f"http://{HOST}:{port}/v1/models")
                        if models.is_success and any(
                            model["id"] == MODEL for model in models.json()["data"]
                        ):
                            return
                    except httpx.RequestError:
                        pass
                    time.sleep(1)
            raise RuntimeError(f"GPU {index} vLLM did not become ready")
        except BaseException:
            self.stop()
            raise

    def check_health(self):
        if self.lifeline is not None and self.guard.poll() is not None:
            raise RuntimeError("vLLM process group exited")

    def stop(self):
        if self.lifeline is not None:
            os.close(self.lifeline)
            self.lifeline = None
        try:
            self.guard.wait(timeout=16)
        except subprocess.TimeoutExpired:
            os.killpg(self.guard.pid, signal.SIGKILL)
            self.guard.wait(timeout=5)
        try:
            os.killpg(self.guard.pid, 0)
        except ProcessLookupError:
            return
        raise RuntimeError(f"vLLM process group {self.guard.pid} still exists")

    def __del__(self):
        if hasattr(self, "guard"):
            self.stop()


class ProxiedResponse(Response):
    """Hold a routing slot until the HTTP response finishes or is cancelled."""

    def __init__(self, router, request, body):
        self.router = router
        self.request = request
        self.body = body

    async def __call__(self, scope, receive, send):
        router = self.router
        choices = sorted(router.ready)
        if not choices:
            await unavailable("No Qwen3-8B workers are ready")(scope, receive, send)
            return
        # Start after the last GPU selected, including after membership changes.
        index = next((gpu for gpu in choices if gpu > router.last), choices[0])
        router.last = index
        router.active[index] += 1
        router.drained[index].clear()
        upstream = None
        try:
            headers = {
                name: self.request.headers[name]
                for name in ("content-type", "accept", "authorization", "x-request-id")
                if name in self.request.headers
            }
            url = httpx.URL(
                f"http://{HOST}:{router.ports[index]}{self.request.url.path}"
            ).copy_with(query=self.request.scope["query_string"])
            try:
                upstream = await router.client.send(
                    router.client.build_request(
                        self.request.method,
                        url,
                        headers=headers,
                        content=self.body,
                    ),
                    stream=True,
                )
            except httpx.RequestError:
                await unavailable(f"Qwen3-8B worker on GPU {index} is unavailable")(
                    scope, receive, send
                )
                return
            headers = {
                name: upstream.headers[name]
                for name in (
                    "content-type",
                    "content-encoding",
                    "cache-control",
                    "x-request-id",
                )
                if name in upstream.headers
            }
            headers["x-mercury-gpu"] = str(index)
            response = StreamingResponse(
                upstream.aiter_raw(), status_code=upstream.status_code, headers=headers
            )
            await response(scope, receive, send)
        finally:
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                router.active[index] -= 1
                if router.active[index] == 0:
                    router.drained[index].set()


def unavailable(message):
    return JSONResponse({"error": {"message": message, "type": "server_error"}}, 503)


@serve.deployment(ray_actor_options={"num_cpus": 1}, max_ongoing_requests=256)
class Router:
    def __init__(self, ports):
        self.ports = ports
        self.ready = set()
        self.last = -1
        self.active = dict.fromkeys(ports, 0)
        self.drained = {index: asyncio.Event() for index in ports}
        for event in self.drained.values():
            event.set()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300, connect=3), trust_env=False
        )

    async def __call__(self, request):
        if (request.method, request.url.path) not in {
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        }:
            return JSONResponse({"error": {"message": "Not found"}}, 404)
        return ProxiedResponse(self, request, await request.body())

    async def include(self, index):
        self.ready.add(index)

    async def drain(self, index, timeout):
        self.ready.discard(index)
        try:
            await asyncio.wait_for(self.drained[index].wait(), timeout=timeout)
        except TimeoutError:
            self.ready.add(index)
            return False
        return True

    async def status(self):
        return {
            index: {
                "routing": index in self.ready,
                "active_requests": self.active[index],
            }
            for index in self.ports
        }
