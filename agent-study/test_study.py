"""Offline checks for the replay harness and paper evidence."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from replay import EXTERNAL_COUNTERS, FULL_SESSIONS, METRICS, cached_tokens, metric_values, routing_stats
from router import Proxy, Router
from serve import FLAGS, startup_evidence, verify_flags
from study import CONFIGS
from summarize import paper_entry, sweep_onset, write_run_result

REPLICAS = ("http://127.0.0.1:8120", "http://127.0.0.1:8121")


class StudyTests(unittest.TestCase):
    def router(self, mode):
        with patch.object(ThreadingHTTPServer, "__init__", return_value=None):
            return Router(mode, 8122, REPLICAS)

    def test_sticky_split_and_reset(self):
        router = self.router("sticky")
        sessions = [f"session-{i}" for i in range(24)]
        with ThreadPoolExecutor(max_workers=8) as workers:
            initial = dict(zip(sessions, workers.map(router.order, sessions)))
        self.assertEqual(Counter(order[0][0] for order in initial.values()),
                         dict.fromkeys(REPLICAS, 12))
        self.assertEqual(initial, {session: router.order(session) for session in sessions})
        self.assertEqual(router.assignment_snapshot(reset=True), {})
        self.assertEqual(router.order("next")[0][0], REPLICAS[0])

    def test_down_replica_moves_once(self):
        router = self.router("sticky")
        proxy = object.__new__(Proxy)
        proxy.server = router
        proxy.command = "POST"
        proxy.path = "/v1/chat/completions"
        proxy.headers = {"Content-Length": "2", "x-session-id": "session"}
        with patch.object(proxy, "copy_response") as copy, patch("router.print") as log:
            with patch("router.build_opener") as build:
                opener = build.return_value
                response = opener.open.return_value
                response.code = 200
                opener.open.side_effect = [URLError("down"), response, response]
                for _ in range(2):
                    proxy.rfile = io.BytesIO(b"{}")
                    proxy.forward()
                urls = [call.args[0].full_url for call in opener.open.call_args_list]
                self.assertEqual(urls, [replica + proxy.path for replica in
                                        (REPLICAS[0], REPLICAS[1], REPLICAS[1])])
            self.assertEqual(copy.call_count, 2)
            log.assert_called_once()
        self.assertEqual(router.assignment_snapshot(), {"session": REPLICAS[1]})

    def test_round_robin_only_counts_adjacent_successes(self):
        router = self.router("rr")
        self.assertEqual([router.order("s")[0][0] for _ in range(4)], list(REPLICAS) * 2)
        turns = [(4, 0, None), (2, 1, None), (1, 0, None), (3, 1, "error")]
        stats = routing_stats([{"session": "s", "turn": turn, "replica": REPLICAS[i], "error": error}
                               for turn, i, error in turns], REPLICAS, None)
        self.assertEqual(stats["consecutive_turn_replica_changes"],
                         {"numerator": 1, "denominator": 1, "share": 1.0})
        self.assertEqual(stats["distinct_sessions_served"], dict.fromkeys(REPLICAS, 1))

    def test_cached_tokens(self):
        for usage, expected in [
            ({"prompt_tokens_details": {"cached_tokens": 32}}, 32),
            ({"prompt_tokens_details": {"cached_tokens": None}}, 0),
            ({"prompt_tokens_details": {}}, 0),
            ({"prompt_tokens_details": None}, 0),
            ({}, 0),
        ]:
            with self.subTest(usage=usage):
                self.assertEqual(cached_tokens(usage), expected)

    def test_saved_flags_and_metric_presence(self):
        for _, extra in CONFIGS.values():
            verify_flags((*FLAGS, "--port", *extra))
        metrics = [f'{metric}{{engine="0"}} {i}.0' for i, metric in enumerate(METRICS)]
        self.assertEqual(metric_values("\n".join(metrics)), dict(zip(METRICS, range(len(METRICS)))))
        external = [f'{metric}{{engine="0"}} 0' for metric in EXTERNAL_COUNTERS]
        parsed = metric_values("\n".join(metrics + external))
        self.assertEqual([parsed[name] for name in EXTERNAL_COUNTERS], [0, 0])
        with self.assertRaises(ValueError):
            metric_values("\n".join(metrics[:-1]))

    def test_startup_evidence_and_paper_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "server.log"
            line = "INFO [gpu_model_runner.py:4959] Model loading took 61.03 GiB memory and 6.9 seconds"
            log.write_text(line + "\nGPU KV cache size: 251,104 tokens\n"
                           "Maximum concurrency for 32,768 tokens per request: 7.66x\n")
            evidence = startup_evidence(log)
            self.assertEqual(evidence["model_loading_line"], line)
            self.assertEqual(evidence["model_loading_gib"], 61.03)
            self.assertEqual(evidence["max_concurrency"], 7.66)
            ready = root / "ready.json"
            ready.write_text(json.dumps(evidence))
            with patch("summarize.ROOT", root):
                self.assertEqual(paper_entry(ready, "kv_cache_tokens", "tokens", ",d"),
                                 {"value": "251104", "display": "251,104",
                                  "source": "ready.json", "what": "tokens"})

    def test_raw_metrics_and_onset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run.json"
            run.write_text(json.dumps({"status": "complete", "sessions": 1, "turns": 2,
                                       "replicas": REPLICAS}))
            turns = [dict(session="s", turn=turn, replica=REPLICAS[0], error=None,
                          stream_done=True, finish_reason="stop", ttft_seconds=0.5 * turn,
                          end_to_end_seconds=2, prompt_tokens=turn, cached_tokens=turn - 1, completion_tokens=1)
                     for turn in (1, 2)]
            (root / "turns.jsonl").write_text("\n".join(map(json.dumps, turns)))
            samples = [[{"replica": replica, "metrics": dict.fromkeys(METRICS, level)}
                        for replica in REPLICAS] for level in (10, 12)]
            (root / "metrics.jsonl").write_text("\n".join(map(json.dumps, samples)))
            with patch("summarize.ROOT", root):
                _, row = write_run_result(run)
            self.assertTrue(row["valid"])
            self.assertAlmostEqual(row["request_hit_pct"], 100 / 3)
            failed = {**turns[0], "error": "stream interrupted", "prompt_tokens": 1000}
            with (root / "turns.jsonl").open("a") as log:
                log.write("\n" + json.dumps(failed))
            with patch("summarize.ROOT", root):
                _, partial = write_run_result(run)
            self.assertAlmostEqual(partial["request_hit_pct"], 100 / 3)
            self.assertEqual(row["preemptions"], 4)
            self.assertEqual(row["prefix_hit_pct"], 100)
            self.assertEqual(row["later_p95"], 1)
            rows = {size: {**row, "preemptions": 0} for size in FULL_SESSIONS}
            self.assertEqual(sweep_onset(rows), "none")
            rows[32]["preemptions"] = 1
            self.assertEqual(sweep_onset(rows), 32)
            rows[48]["valid"] = False
            self.assertIsNone(sweep_onset(rows))


if __name__ == "__main__":
    unittest.main()
