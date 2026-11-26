"""Loopback OpenAI router with round-robin and session-affine routing."""

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from study import base_url, local_url

PATHS = {("GET", "/v1/models"), ("POST", "/v1/chat/completions")}
UNAVAILABLE = {502, 503, 504}
ASSIGNMENTS = "/study/assignments"


class Router(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, mode, port, replicas):
        self.mode = mode
        self.replicas = tuple(local_url(url) for url in replicas)
        if len(self.replicas) != 2 or len(set(self.replicas)) != 2:
            raise ValueError("Router requires two distinct study replicas")
        if base_url(port) in self.replicas:
            raise ValueError("Router cannot be its own upstream")
        self.next_index = 0
        self.lock = threading.Lock()
        self.assignments = {}
        self.counts = Counter()
        self.replay = 0
        super().__init__(("127.0.0.1", port), Proxy)

    def order(self, session):
        with self.lock:
            if self.mode == "sticky":
                if not session:
                    return self.replicas, self.replay
                if session not in self.assignments:
                    replica = min(self.replicas, key=self.counts.__getitem__)
                    self.assignments[session] = replica
                    self.counts[replica] += 1
                replica = self.assignments[session]
                return (replica, self.replicas[1 - self.replicas.index(replica)]), self.replay
            index = self.next_index
            self.next_index = 1 - index
            return (self.replicas[index],), self.replay

    def move(self, session, original, target, replay):
        with self.lock:
            if replay == self.replay and self.assignments.get(session) == original:
                self.assignments[session] = target
                self.counts[original] -= 1
                self.counts[target] += 1
                print(f"Moved session {session!r} from {original} to {target}", flush=True)

    def assignment_snapshot(self, reset=False):
        with self.lock:
            if reset:
                self.assignments.clear()
                self.counts.clear()
                self.replay += 1
            return dict(self.assignments)


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        if self.server.mode == "sticky" and self.path == ASSIGNMENTS:
            assignments = self.server.assignment_snapshot(reset=self.command == "POST")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(assignments).encode())
            return
        if (self.command, self.path) not in PATHS:
            self.send_error(404)
            return
        data = None
        if self.command == "POST":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= 16 * 1024 * 1024:
                self.send_error(400, "Expected a bounded JSON request body")
                return
            data = self.rfile.read(length)
        session = self.headers.get("x-session-id") if self.command == "POST" else None
        candidates, replay = self.server.order(session)
        for position, replica in enumerate(candidates):
            request = Request(replica + self.path, data=data, method=self.command,
                              headers={"Content-Type": "application/json"})
            try:
                response = build_opener(ProxyHandler({})).open(request, timeout=180)
            except HTTPError as exc:
                if exc.code in UNAVAILABLE and position + 1 < len(candidates):
                    error_body = exc.read().lower()
                    exc.close()
                    if b"out of memory" not in error_body and b"outofmemory" not in error_body:
                        continue
                    self.send_bytes(exc.code, replica, error_body)
                    return
                response = exc
            except (URLError, OSError):
                continue
            with response:
                if position and session and 200 <= response.code < 300:
                    self.server.move(session, candidates[0], replica, replay)
                self.copy_response(response, replica)
            return
        self.send_error(502, "Replica unavailable")

    def send_headers(self, code, replica, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Replica", replica)
        self.end_headers()

    def send_bytes(self, code, replica, data):
        self.send_headers(code, replica, "application/json")
        self.wfile.write(data)

    def copy_response(self, response, replica):
        content_type = response.headers.get("Content-Type", "application/json")
        self.send_headers(response.code, replica, content_type)
        try:
            if content_type.startswith("text/event-stream"):
                for line in response:
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                self.wfile.write(response.read())
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("rr", "sticky"))
    parser.add_argument("port", type=int)
    parser.add_argument("replicas", nargs=2)
    args = parser.parse_args()
    with Router(args.mode, args.port, args.replicas) as server:
        print(f"Listening on {base_url(args.port)} using {args.mode}", flush=True)
        server.serve_forever()
