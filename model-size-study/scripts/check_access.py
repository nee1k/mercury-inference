"""Phase-0 access checks. Run using the installed vLLM Python. Standard library only."""
import datetime
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
report = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "python": sys.executable, "checks": [], "status": "RUNNING"}
target = ROOT / "results" / "access-check.json"


def check(name, operation, permission):
    try:
        evidence = operation()
        report["checks"].append({"name": name, "status": "PASS", "evidence": evidence})
    except Exception as error:
        report["checks"].append({"name": name, "status": "FAIL",
                                 "error": str(error), "required_access": permission})
        report["status"] = "BLOCKED"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


check("nvidia-smi", lambda: command(["nvidia-smi"]), "NVIDIA device/driver access outside the sandbox")
check("torch CUDA", lambda: command([sys.executable, "-c",
      'import torch; print(torch.__version__); print(torch.cuda.is_available()); '
      'assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.device_count())']),
      "NVIDIA device/driver access outside the sandbox")


def localhost():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"phase0-ok")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:%s/" % server.server_port, timeout=10) as response:
            assert response.read() == b"phase0-ok"
        return {"host": "127.0.0.1", "port": server.server_port, "status": 200}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


check("localhost HTTP", localhost, "Local socket binding and loopback HTTP access")


def cache_write():
    hf_home = Path(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
    cache = Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))))
    # Do not create a missing cache directory outside the study folder.
    with tempfile.NamedTemporaryFile(prefix="mercury-phase0-", dir=cache, mode="w+") as handle:
        handle.write("phase0-ok")
        handle.flush()
        handle.seek(0)
        assert handle.read() == "phase0-ok"
        test_path = Path(handle.name)
    assert not test_path.exists()
    return {"cache": str(cache), "test_removed": True}


check("HF cache write/remove", cache_write, "Write access to the existing Hugging Face cache")


def huggingface():
    with urllib.request.urlopen("https://huggingface.co", timeout=20) as response:
        return {"url": "https://huggingface.co", "status": response.status}


check("Hugging Face HTTPS", huggingface, "Outbound HTTPS/DNS access to huggingface.co")
report["status"] = "PASS"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
