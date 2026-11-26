"""Hardware-free tests for evidence validation, stream timing and gating."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark import validate_result
from common import ROOT, check_neighbors, gpu_snapshot, idle_sample, same_process, save, terminate_child
from gate import compare, read_chat
from runner import analyze_chunks, sse_events
from serve import server_metrics
from summarize import collect, frontiers, run as summarize


def fixture_raw(count=10, input_len=1024, output_len=300):
    result = {"completed": count, "failed": 0, "output_lens": [output_len] * count,
              "input_lens": [input_len] * count, "errors": [""] * count, "output_throughput": 100}
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for percentile in (50, 95, 99):
            result["p%d_%s_ms" % (percentile, metric)] = 100.0
    return result


class HarnessTests(unittest.TestCase):
    def test_idle_rule_rejects_memory_utilization_or_process_at_any_sample(self):
        good = {"uuid": "test-gpu", "used_mib": 0, "gpu_util_percent": 0, "pids": []}
        cases = [None, {"used_mib": 1024}, {"gpu_util_percent": 1}, {"pids": [123]}]
        for change in cases:
            samples = [dict(good) for _ in range(11)]
            if change:
                samples[5].update(change)
            with patch("common.gpu_snapshot", side_effect=samples), patch("common.time.sleep"), patch("common.time.monotonic", side_effect=range(30)):
                self.assertEqual(idle_sample()["idle"], change is None)

    def test_gpu_xml_includes_graphics_process_and_unknown_neighbors_fail(self):
        xml = '<nvidia_smi_log><gpu><uuid>test-gpu</uuid><fb_memory_usage><used>5 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes><process_info><pid>123</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>'
        import subprocess
        with patch("common.subprocess.run", return_value=subprocess.CompletedProcess([], 0, xml, "")):
            sample = gpu_snapshot()
        self.assertEqual(sample["pids"], [123])
        with patch("common.same_process", return_value=True), patch("common.descendant", return_value=False):
            self.assertTrue(check_neighbors({"pid": 100, "gpu_uuid": "test-gpu"}, sample))

    def test_pid_reuse_and_foreign_owner_never_receive_signal(self):
        record = {"pid": 123, "start_ticks": 100, "uid": os.getuid(), "boot_id": "test", "state": "S"}
        with patch("common.identity", return_value={**record, "start_ticks": 101}):
            self.assertFalse(same_process(record))
        with tempfile.TemporaryDirectory(dir=ROOT / "state", prefix="test-owner-") as temporary:
            root = Path(temporary)
            (root / "state").mkdir()
            (root / "state" / "pids.txt").write_text("123\n")
            with patch("common.ROOT", root), patch("common.os.pidfd_open") as pidfd:
                with self.assertRaises(ValueError):
                    terminate_child({**record, "uid": os.getuid() + 1})
                pidfd.assert_not_called()

    def test_exact_length_and_missing_evidence(self):
        good = fixture_raw()
        self.assertEqual(validate_result(good, 10, 300, 1024), [])
        wrong_length = copy.deepcopy(good)
        wrong_length["output_lens"][2] = 299
        self.assertTrue(validate_result(wrong_length, 10, 300, 1024))
        missing = copy.deepcopy(good)
        del missing["p95_e2el_ms"]
        self.assertTrue(validate_result(missing, 10, 300, 1024))
        failure = copy.deepcopy(good)
        failure["errors"][0] = "HTTP 500"
        self.assertTrue(validate_result(failure, 10, 300, 1024))

    def test_gate_boundary_failures_and_missing_cells(self):
        reference = {c: {"p95_ttft_ms": 100.0, "p95_tpot_ms": 100.0} for c in (1, 4)}
        boundary = {c: {"p95_ttft_ms": 120.0, "p95_tpot_ms": 120.0} for c in (1, 4)}
        self.assertEqual(compare(reference, boundary, True)["status"], "PASS")
        boundary[4]["p95_tpot_ms"] = 120.001
        self.assertEqual(compare(reference, boundary, True)["status"], "FAIL")
        self.assertEqual(compare(reference, {1: reference[1]}, True)["status"], "FAIL")
        self.assertEqual(compare(reference, reference, False)["status"], "FAIL")
        invalid = copy.deepcopy(reference)
        invalid[1]["p95_ttft_ms"] = float("nan")
        self.assertEqual(compare(reference, invalid, True)["status"], "FAIL")

    def test_split_think_tag_and_separate_field_names(self):
        chunks = [(0.1, {"choices": [{"delta": {"role": "assistant"}}]}),
                  (0.2, {"choices": [{"delta": {"content": "<think>working</thi"}}]}),
                  (0.4, {"choices": [{"delta": {"content": "nk>\n"}}]}),
                  (0.6, {"choices": [{"delta": {"content": "Answer"}}]})]
        result = analyze_chunks(chunks, "default")
        self.assertEqual(result["ttft_any_seconds"], 0.2)
        self.assertEqual(result["ttft_answer_seconds"], 0.6)
        for field in ("reasoning", "reasoning_content"):
            chunks = [(0.2, {"choices": [{"delta": {field: "work"}}]}),
                      (0.7, {"choices": [{"delta": {"content": "Answer"}}]})]
            self.assertEqual(analyze_chunks(chunks, "default")["ttft_answer_seconds"], 0.7)
        self.assertIsNone(analyze_chunks([(0.2, {"choices": [{"delta": {"content": "Unknown"}}]})], "default")["ttft_answer_seconds"])

    def test_sse_empty_choices_usage_and_done(self):
        response = io.BytesIO(b': ping\n\ndata: {"choices": []}\n\ndata: {"usage": {"completion_tokens": 3}}\n\ndata: [DONE]\n\n')
        result = list(sse_events(response))
        self.assertEqual(result[-1]["usage"]["completion_tokens"], 3)
        with self.assertRaises(json.JSONDecodeError):
            list(sse_events(io.BytesIO(b'data: broken\n\n')))

    def test_saved_summary_and_gate_baseline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "state", prefix="test-harness-") as temporary:
            root = Path(temporary)
            sources = []
            for c in (1, 4):
                directory = root / "results" / "fixture" / ("c" + str(c))
                raw = fixture_raw(count=10 if c == 1 else 40)
                save(directory / "raw.json", raw)
                record = {"config": "fixture", "shape": "chat", "concurrency": c, "kind": "pilot",
                    "status": "VALID", "raw_file": str(directory / "raw.json"), "gpus": 1,
                    "precision": "BF16", "total_parameters_b": 8.2, "vllm_version": "fixture",
                    "prompts": 10 if c == 1 else 40, "output_tokens": 300, "input_tokens": 1024}
                save(directory / "cell.json", record)
                sources.append(str(directory / "cell.json"))
            save(root / "baseline.json", {"cells": sources})
            baseline = read_chat(root / "baseline.json")
            self.assertEqual(compare(baseline, baseline, True)["status"], "PASS")
            result = summarize(root)
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["repeat_union"], [])
            self.assertTrue((root / "results" / "summary.md").is_file())
            cell = json.loads(Path(sources[0]).read_text())
            cell["status"] = "INVALID"
            save(sources[0], cell)
            with self.assertRaises(ValueError):
                read_chat(root / "baseline.json")
            self.assertEqual(len(collect(root)[1]), 1)

    def test_frontier_uses_only_c16_and_all_three_axes(self):
        def row(name, gpu, latency, params, concurrency=16):
            return {"config": name, "shape": "chat", "concurrency": concurrency,
                    "scope": "study", "gpus": gpu, "p95_e2el_ms": latency, "total_parameters_b": params}
        result = frontiers([row("small", 1, 10, 8), row("large", 1, 20, 32),
                            row("dominated", 2, 30, 16), row("wrong-cell", 1, 1, 235, 64)])
        self.assertEqual(set(result["chat"]), {"small", "large"})
        self.assertEqual(result["long-doc"], [])

    def test_startup_metrics_ignore_previous_launch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "state", prefix="test-startup-") as temporary:
            path = Path(temporary) / "server.log"
            old = "GPU KV cache size: 1 tokens\n"
            path.write_text(old + "GPU KV cache size: 123,456 tokens\nMaximum concurrency for 32,768 tokens per request: 3.77x\n")
            result = server_metrics({"server_log": str(path), "log_start_byte": len(old.encode())})
            self.assertEqual(result["kv_cache_tokens"], 123456)
            self.assertEqual(result["concurrency_context_tokens"], 32768)
            self.assertEqual(result["maximum_concurrency"], 3.77)


if __name__ == "__main__":
    unittest.main()
