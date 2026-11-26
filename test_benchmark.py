import io
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import urllib.error
import jsonschema
import benchmark as b
import report

SCHEMA = {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value'], 'additionalProperties': False}
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)

class FakeResponse(io.BytesIO):
    status = 200
    def __init__(self, data):
        super().__init__(data)
        self.fp = types.SimpleNamespace(raw=types.SimpleNamespace(_sock=types.SimpleNamespace(settimeout=lambda n: None)))

def stream(content='{"value":4}', reason='stop', usage=True, done=True):
    chunks = [{'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]},
              {'choices': [{'delta': {'content': content[:4]}, 'finish_reason': None}]},
              {'choices': [{'delta': {'content': content[4:]}, 'finish_reason': None}]},
              {'choices': [{'delta': {}, 'finish_reason': reason}]}]
    if usage: chunks.append({'choices': [], 'usage': {'completion_tokens': 5, 'prompt_tokens': 10}})
    return (''.join('data: ' + json.dumps(c) + '\r\n\r\n' for c in chunks) + ('data: [DONE]\n\n' if done else '')).encode()

class BenchmarkTests(unittest.TestCase):
    def test_sse_comments_and_multiline(self):
        self.assertEqual(list(b.sse_events(io.BytesIO(b': ping\r\ndata: first\r\ndata: second\r\n\r\ndata: tail'))), ['first\nsecond', 'tail'])

    def test_stream_usage_trailer_and_empty_chunk(self):
        with patch.object(b.urllib.request, 'urlopen', return_value=FakeResponse(stream())):
            row = b.stream_request('http://fixture', {})
        self.assertIsNone(row['error'])
        self.assertEqual(row['content'], '{"value":4}')
        self.assertIsNotNone(row['ttft_seconds'])
        self.assertAlmostEqual(row['tpot_seconds'], row['decode_seconds']/4)
        self.assertGreaterEqual(row['latency_seconds'], row['ttft_seconds'] + row['decode_seconds'])

    def test_incomplete_stream_and_missing_usage(self):
        with patch.object(b.urllib.request, 'urlopen', return_value=FakeResponse(stream(done=False))):
            self.assertIn('Incomplete stream', b.stream_request('http://fixture', {})['error'])
        with patch.object(b.urllib.request, 'urlopen', return_value=FakeResponse(stream(usage=False))):
            row = b.stream_request('http://fixture', {})
        self.assertIsNone(row['error'])
        self.assertIsNone(row['tpot_seconds'])

    def test_http_and_timeout(self):
        error = urllib.error.HTTPError('http://fixture', 503, 'Unavailable', {}, io.BytesIO(b'engine failed'))
        with patch.object(b.urllib.request, 'urlopen', side_effect=error):
            row = b.stream_request('http://fixture', {})
        self.assertEqual(row['http_status'], 503)
        self.assertEqual(row['error_body'], 'engine failed')
        with patch.object(b.urllib.request, 'urlopen', side_effect=TimeoutError('timed out')):
            self.assertIn('TimeoutError', b.stream_request('http://fixture', {})['error'])

    def test_total_deadline(self):
        with patch.object(b.urllib.request, 'urlopen', return_value=FakeResponse(stream())), patch.object(b.time, 'perf_counter', side_effect=[0, 181, 182]):
            self.assertIn('total timeout', b.stream_request('http://fixture', {})['error'])

    def test_quality(self):
        self.assertTrue(b.score('{"value":4}', {'value': 4}, VALIDATOR, 'stop')['exact_match'])
        self.assertFalse(b.score('{"value":4}', {'value': 4}, VALIDATOR, 'length')['exact_match'])
        self.assertFalse(b.score('```json\n{}\n```', {}, VALIDATOR, 'stop')['json_valid'])
        self.assertFalse(b.score('{"value":"4"}', {'value': 4}, VALIDATOR, 'stop')['schema_valid'])
        self.assertFalse(b.strict_equal({'value': True}, {'value': 1}))
        self.assertTrue(b.strict_equal({'a': 1, 'b': 2}, {'b': 2, 'a': 1}))
        self.assertFalse(b.strict_equal([1, 2], [2, 1]))

    def test_safety_thresholds(self):
        safety = b.Safety()
        for _ in range(2): safety.observe({'error': 'HTTPError', 'http_status': 400})
        self.assertFalse(safety.stop.is_set())
        safety.observe({'error': 'HTTPError', 'http_status': 400})
        self.assertTrue(safety.stop.is_set())
        safety = b.Safety()
        for i in range(20): safety.observe({'error': 'bad' if i in (1, 3) else None, 'http_status': 400 if i in (1, 3) else 200})
        self.assertIn('5%', safety.reason)
        safety = b.Safety()
        safety.observe({'error': 'CUDA out of memory', 'http_status': 500})
        self.assertEqual(safety.reason, 'OOM reported')

    def test_percentile(self):
        self.assertAlmostEqual(report.percentile([1, 2, 3, 4], .95), 3.85)
        self.assertEqual(report.percentile([2], .95), 2)
        self.assertIsNone(report.percentile([], .95))

    def test_workload_contract_and_boundaries(self):
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda msg, **kw: {'input_ids': [[0] * int(msg[0]['content'].split('Text:\n')[1])], 'attention_mask': []})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            schema, data = tmp/'schema.json', tmp/'data.jsonl'
            schema.write_text(json.dumps(SCHEMA))
            rows = [{'id': str(n), 'input_text': str(n), 'reference_output': {'value': 4}} for n in (512, 513, 1536, 1537, 4097)]
            data.write_text('\n'.join(json.dumps(r) for r in rows))
            records, _, _ = b.load_workload(data, schema, tokenizer)
            self.assertEqual([r['length_group'] for r in records], ['short', 'medium', 'medium', 'long', 'long'])
            data.write_text('\n'.join(json.dumps(r) for r in rows + rows[:1]))
            with self.assertRaisesRegex(ValueError, 'unique'): b.load_workload(data, schema, tokenizer)
            rows[0]['reference_output'] = {'value': 'bad'}
            data.write_text(json.dumps(rows[0]))
            with self.assertRaises(jsonschema.ValidationError): b.load_workload(data, schema, tokenizer)

    def test_closed_loop_and_reporting(self):
        active = peak = 0
        lock = threading.Lock()
        def request(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(.005)
            with lock: active -= 1
            return {'error': None, 'content': '{"value":4}', 'usage': {'completion_tokens': 5}, 'finish_reason': 'stop',
                    'http_status': 200, 'latency_seconds': .02, 'ttft_seconds': .01, 'decode_seconds': .01, 'tpot_seconds': .0025}
        records = [{'id': str(i), 'messages': [], 'reference_output': {'value': 4}, 'length_group': 'short', 'prompt_tokens_expected': 20} for i in range(20)]
        args = types.SimpleNamespace(endpoint='http://fixture', timeout=180)
        with tempfile.TemporaryDirectory() as tmp, patch.object(b, 'stream_request', side_effect=request):
            tmp = Path(tmp)
            recorder = b.Recorder(tmp/'requests.jsonl')
            stage = b.run_stage(records, 4, 128, 1, False, args, recorder, VALIDATOR, b.Safety())
            recorder.close()
            self.assertLessEqual(peak, 4)
            self.assertGreater(peak, 1)
            b.write_json(tmp/'stages.json', [stage])
            summary = report.generate(tmp)
            self.assertEqual(summary['conditions'][0]['requests'], 20)
            self.assertEqual(summary['conditions'][0]['exact_match_rate'], 1)
            self.assertEqual(summary['conditions'][0]['completion_tokens_reported'], 100)
            self.assertAlmostEqual(summary['conditions'][0]['requests_per_second'], 20/stage['duration_seconds'])
            with self.assertRaises(FileExistsError): b.Recorder(tmp/'requests.jsonl')

    def test_stop_does_not_schedule_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = b.Recorder(Path(tmp)/'requests.jsonl')
            safety = b.Safety()
            safety.stop.set()
            with patch.object(b, 'stream_request') as request:
                b.run_stage([{}] * 20, 4, 128, 1, False, types.SimpleNamespace(), recorder, VALIDATOR, safety)
                request.assert_not_called()
            recorder.close()

    def test_telemetry(self):
        observed = report.telemetry([{'gpu': {'returncode': 0, 'stdout': 'index,uuid,memory.used,utilization.gpu\n0, GPU-a, 43673 MiB, 17 %\n'},
             'server_metrics': '# HELP comment\nvllm:request_queue_time_seconds_sum{model="Qwen"} 1.5\n'}])
        self.assertEqual(observed['gpus']['0']['peak_memory_mib'], 43673)
        self.assertEqual(len(observed['server_series']), 1)


    def test_full_runner_offline(self):
        fake_tokenizer = types.SimpleNamespace(chat_template='fixture template', apply_chat_template=lambda *a, **kw: [0] * 40)
        fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_tokenizer))
        inv = {'timestamp': 'fixture', 'gpus': {'returncode': 0, 'stdout': ''}, 'processes': {'returncode': 0, 'stdout': ''}}
        def sample(base, path, done): path.write_text('')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset, schema = tmp/'dataset.jsonl', tmp/'schema.json'
            dataset.write_text(json.dumps({'id':'case-1', 'input_text':'value 4', 'reference_output':{'value':4}}) + '\n')
            schema.write_text(json.dumps(SCHEMA))
            argv = ['benchmark.py', '--dataset', str(dataset), '--schema', str(schema), '--concurrency', '1', '4', '--output-dir', str(tmp/'runs')]
            with patch.dict('sys.modules', {'transformers': fake_transformers}), patch('sys.argv', argv), patch.object(b, 'inventory', return_value=inv), patch.object(b, 'get_json', return_value={'data':[{'id':b.MODEL, 'max_model_len':4096}]}), patch.object(b, 'sampler', side_effect=sample), patch.object(b.urllib.request, 'urlopen', side_effect=lambda *a, **kw: FakeResponse(stream())):
                b.main()
            run = next((tmp/'runs').iterdir())
            summary = json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['warmup_requests'], 10)
            self.assertEqual(len(summary['conditions']), 4)
            self.assertEqual(len(summary['repetitions']), 12)
            self.assertEqual(summary['status']['status'], 'complete')
            self.assertEqual((run/'dataset.jsonl').read_bytes(), dataset.read_bytes())
            self.assertEqual(len((run/'requests.jsonl').read_text().splitlines()), 22)

    def test_high_load_gate(self):
        with patch('sys.argv', ['benchmark.py', '--dataset', 'missing', '--schema', 'missing', '--concurrency', '8']):
            with self.assertRaises(SystemExit) as error:
                b.main()
            self.assertEqual(error.exception.code, 2)


    @unittest.skipUnless(os.environ.get('MERCURY_SOCKET_TESTS') == '1', 'requires localhost socket permission')
    def test_real_http_stream(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                data = stream()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                for offset in range(0, len(data), 7):
                    self.wfile.write(data[offset:offset+7])
                    self.wfile.flush()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            row = b.stream_request('http://127.0.0.1:' + str(server.server_port), {})
            self.assertIsNone(row['error'])
            self.assertEqual(row['content'], '{"value":4}')
            self.assertEqual(row['usage']['completion_tokens'], 5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

if __name__ == '__main__':
    unittest.main()
