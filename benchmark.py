#!/usr/bin/env python3
"""Closed-loop extraction benchmark. Uses existing E4S Python dependencies."""
import argparse
import concurrent.futures
from collections.abc import Mapping
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MODEL = 'Qwen/Qwen3-8B'
PROMPT = ('Extract the information requested by this JSON Schema from the supplied text. '
          'Return only one JSON object matching the schema, without markdown or commentary.\n'
          'JSON Schema:\n{schema}\n\nText:\n{text}')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'command': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command': args, 'error': str(e)}


def inventory():
    result = {'timestamp': utc(), 'host': socket.gethostname(),
              'gpus': command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu', '--format=csv']),
              'processes': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv'])}
    pids = []
    for line in result['processes'].get('stdout', '').splitlines()[1:]:
        fields = line.split(',')
        if len(fields) >= 2 and fields[1].strip().isdigit():
            pids.append(fields[1].strip())
    if pids:
        result['owners'] = command(['ps', '-p', ','.join(pids), '-o', 'pid,user,lstart,args'])
    return result


def get_json(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def score(text, reference, validator, finish_reason):
    result = {'json_valid': False, 'schema_valid': False, 'exact_match': False,
              'truncated': finish_reason == 'length'}
    try:
        parsed = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        result['json_valid'] = True
    except (ValueError, TypeError) as e:
        result['quality_error'] = str(e)
        return result
    errors = list(validator.iter_errors(parsed))
    result['schema_valid'] = not errors
    result['quality_error'] = '; '.join(e.message for e in errors)
    result['exact_match'] = (not errors and not result['truncated'] and strict_equal(parsed, reference))
    return result


def sse_events(response, deadline=None):
    """Parse SSE event boundaries, including multiple data lines and comments."""
    data = []
    iterator = iter(response)
    while True:
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError('Request exceeded total timeout')
            response.fp.raw._sock.settimeout(remaining)
        try:
            raw = next(iterator)
        except StopIteration:
            break
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
    if data:
        yield '\n'.join(data)


def stream_request(base, body, timeout=180):
    started = time.perf_counter()
    result = {'started_at': utc(), 'http_status': None, 'content': '', 'usage': None,
              'finish_reason': None, 'ttft_seconds': None, 'decode_seconds': None,
              'tpot_seconds': None, 'error': None}
    last_content = None
    try:
        request = urllib.request.Request(base + '/v1/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['http_status'] = response.status
            done = False
            for event in sse_events(response, started + timeout):
                if event == '[DONE]':
                    done = True
                    break
                chunk = json.loads(event)
                if chunk.get('error'):
                    raise ValueError('Server stream error: ' + json.dumps(chunk['error']))
                if chunk.get('usage') is not None:
                    result['usage'] = chunk['usage']
                for choice in chunk.get('choices', []):
                    content = choice.get('delta', {}).get('content') or ''
                    if content:
                        elapsed = time.perf_counter() - started
                        if result['ttft_seconds'] is None:
                            result['ttft_seconds'] = elapsed
                        last_content = elapsed
                        result['content'] += content
                    if choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
            if not done or result['finish_reason'] is None:
                raise ValueError('Incomplete stream: missing DONE or finish reason')
            if not result['content']:
                raise ValueError('Empty content response')
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        if isinstance(e, urllib.error.HTTPError):
            result['http_status'] = e.code
            result['error_body'] = e.read(8192).decode(errors='replace')
    result['latency_seconds'] = time.perf_counter() - started
    if last_content is not None:
        result['decode_seconds'] = last_content - result['ttft_seconds']
    tokens = (result['usage'] or {}).get('completion_tokens')
    if tokens is not None and tokens > 1 and result['decode_seconds'] is not None:
        result['tpot_seconds'] = result['decode_seconds'] / (tokens - 1)
    return result


class Safety:
    def __init__(self):
        self.stop = threading.Event()
        self.reason = None
        self.lock = threading.Lock()
        self.total = self.failures = self.consecutive = 0

    def reset_counts(self):
        with self.lock:
            self.total = self.failures = self.consecutive = 0

    def observe(self, row):
        with self.lock:
            failed = bool(row['error'])
            self.total += 1
            self.failures += int(failed)
            self.consecutive = self.consecutive + 1 if failed else 0
            error = (str(row.get('error', '')) + str(row.get('error_body', ''))).lower()
            reason = None
            if 'out of memory' in error or 'outofmemory' in error or 'oom' in error:
                reason = 'OOM reported'
            elif row.get('http_status') in (502, 503, 504) or (failed and row.get('http_status') is None):
                reason = 'Endpoint unavailable or request timed out'
            elif self.consecutive >= 3:
                reason = 'Three consecutive request failures'
            elif self.total >= 20 and self.failures / self.total > .05:
                reason = 'Error rate exceeded 5% after 20 requests'
            if reason and not self.stop.is_set():
                self.reason = reason
                self.stop.set()


class Recorder:
    def __init__(self, path):
        self.file = path.open('x')
        self.lock = threading.Lock()

    def append(self, row):
        with self.lock:
            self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.file.flush()

    def close(self):
        self.file.close()


def sampler(base, path, done):
    with path.open('x') as file:
        while not done.is_set():
            started = time.monotonic()
            row = {'timestamp': utc(), 'gpu': command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'])}
            try:
                with urllib.request.urlopen(base + '/metrics', timeout=2) as response:
                    row['server_metrics'] = response.read().decode()
            except Exception as e:
                row['metrics_error'] = str(e)
            file.write(json.dumps(row) + '\n')
            file.flush()
            done.wait(max(0, 1 - (time.monotonic() - started)))


def messages(schema, text):
    return [{'role': 'user', 'content': PROMPT.format(schema=schema, text=text)}]


def load_workload(dataset, schema_path, tokenizer):
    import jsonschema
    schema_bytes = schema_path.read_bytes()
    reject_constant = lambda value: (_ for _ in ()).throw(ValueError('Nonstandard JSON constant: ' + value))
    schema = json.loads(schema_bytes, parse_constant=reject_constant)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    # External references would make scoring dependent on network/files outside the contract.
    def reject_external(value):
        if isinstance(value, dict):
            for key in ('$ref', '$dynamicRef', '$recursiveRef'):
                if key in value and not value[key].startswith('#'):
                    raise ValueError('Schema must be self-contained; external references are unsupported')
            for child in value.values(): reject_external(child)
        elif isinstance(value, list):
            for child in value: reject_external(child)
    reject_external(schema)
    validator = cls(schema)
    schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    records, ids = [], set()
    raw = dataset.read_bytes()
    for line_number, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip(): continue
        row = json.loads(line, parse_constant=reject_constant)
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError(f'Line {line_number}: id must be a unique nonempty string')
        if not isinstance(row.get('input_text'), str) or not isinstance(row.get('reference_output'), dict):
            raise ValueError(f'Line {line_number}: input_text string and reference_output object required')
        validator.validate(row['reference_output'])
        ids.add(row['id'])
        msg = messages(schema_text, row['input_text'])
        encoded = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        token_ids = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError('Unexpected tokenizer batch size')
            token_ids = token_ids[0]
        count = len(token_ids)
        records.append({**row, 'messages': msg, 'prompt_tokens_expected': count,
                        'length_group': 'short' if count <= 512 else 'medium' if count <= 1536 else 'long'})
    if not records:
        raise ValueError('Dataset has no examples')
    return records, validator, {'dataset_sha256': digest(raw), 'schema_sha256': digest(schema_bytes),
                              'prompt_sha256': digest(PROMPT.encode()), 'schema': schema}


def run_stage(records, concurrency, cap, repetition, warmup, args, recorder, validator, safety):
    started = time.perf_counter()
    start_wall = utc()
    safety.reset_counts()
    source = iter(records)
    source_lock = threading.Lock()

    def worker():
        while not safety.stop.is_set():
            with source_lock:
                if safety.stop.is_set(): return
                record = next(source, None)
            if record is None: return
            body = {'model': MODEL, 'messages': record['messages'], 'temperature': 0,
                    'max_tokens': cap, 'stream': True, 'stream_options': {'include_usage': True},
                    'chat_template_kwargs': {'enable_thinking': False}}
            row = stream_request(args.endpoint, body, args.timeout)
            row.update({'id': record['id'], 'concurrency': concurrency, 'output_cap': cap,
                        'repetition': repetition, 'warmup': warmup, 'length_group': record['length_group'],
                        'prompt_tokens_expected': record['prompt_tokens_expected']})
            row['quality'] = score(row['content'], record['reference_output'], validator, row['finish_reason'])
            safety.observe(row)
            recorder.append(row)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        try:
            for future in futures: future.result()
        except BaseException:
            safety.stop.set()
            raise
    return {'concurrency': concurrency, 'output_cap': cap, 'repetition': repetition,
            'warmup': warmup, 'started_at': start_wall, 'duration_seconds': time.perf_counter() - started,
            'planned_requests': len(records), 'stop_reason': safety.reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8001')
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1])
    parser.add_argument('--output-caps', nargs='+', type=int, default=[128, 512])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--revision', default='b968826d9c46dd6066d109eabc6255188de91218')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results' / 'pilot')
    parser.add_argument('--coordinated-window', action='store_true', help='Assert a coordinated high-load window is in effect')
    parser.add_argument('--prepare-only', action='store_true', help='Validate local inputs without contacting or loading the endpoint')
    args = parser.parse_args()
    if any(c not in (1, 4, 8, 16, 32) for c in args.concurrency): parser.error('Concurrency must be 1, 4, 8, 16, or 32')
    if max(args.concurrency) > 4 and not args.coordinated_window: parser.error('Concurrency >4 requires --coordinated-window')
    if args.repetitions < 1 or args.timeout <= 0 or any(c <= 0 or c >= 4096 for c in args.output_caps): parser.error('Invalid repetitions, timeout, or output cap')
    if len(set(args.concurrency)) != len(args.concurrency) or len(set(args.output_caps)) != len(args.output_caps): parser.error('Duplicate conditions are not allowed')
    args.endpoint = args.endpoint.rstrip('/')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.revision, local_files_only=True)
    records, validator, data_info = load_workload(args.dataset, args.schema, tokenizer)
    run_dir = args.output_dir / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + str(os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(data_info)
    config.update({'created_at': utc(), 'python_version': sys.version, 'model': MODEL, 'tokenizer_revision': args.revision,
                   'checkpoint_revision_status': 'Expected local checkpoint; endpoint revision must be corroborated with launch provenance',
                   'tokenizer_chat_template_sha256': digest(str(tokenizer.chat_template).encode()),
                   'generation': {'temperature': 0, 'enable_thinking': False}, 'context_limit': 4096,
                   'versions': {p: importlib.metadata.version(p) for p in ('vllm', 'torch', 'transformers', 'triton', 'jsonschema')},
                   'interpretation': 'Closed-loop fixed concurrency. Client TTFT includes queueing; streaming chunks are not tokens.'})
    config['tool_sha256'] = {name: digest((Path(__file__).resolve().parent / name).read_bytes()) for name in ('benchmark.py', 'report.py', 'run-benchmark.sh')}
    config['environment'] = {key: os.environ.get(key) for key in ('LOADEDMODULES', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')}
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'schema.json', data_info['schema'])
    for name in ('benchmark.py', 'report.py', 'run-benchmark.sh', 'launch-qwen3-8b.sh'):
        (run_dir / name).write_bytes((Path(__file__).resolve().parent / name).read_bytes())
    (run_dir / 'dataset.jsonl').write_bytes(args.dataset.read_bytes())
    write_json(run_dir / 'workload.json', [{'id': r['id'], 'prompt_tokens': r['prompt_tokens_expected'], 'length_group': r['length_group']} for r in records])
    exclusions = [{'id': r['id'], 'output_cap': cap, 'reason': 'input_plus_output_exceeds_4096', 'prompt_tokens': r['prompt_tokens_expected']}
                  for cap in args.output_caps for r in records if r['prompt_tokens_expected'] + cap > 4096]
    write_json(run_dir / 'exclusions.json', exclusions)
    print('Run directory:', run_dir, flush=True)
    if digest((run_dir / 'dataset.jsonl').read_bytes()) != data_info['dataset_sha256']:
        raise RuntimeError('Dataset changed during preparation; retry with stable input')
    if args.prepare_only:
        write_json(run_dir / 'status.json', {'status': 'prepared', 'examples': len(records), 'exclusions': len(exclusions)})
        return
    try:
        before = inventory()
        write_json(run_dir / 'inventory-before.json', before)
        if before['gpus'].get('returncode') != 0 or before['processes'].get('returncode') != 0:
            raise RuntimeError('Cannot inspect GPU allocations; rerun with appropriate GPU access')
        models = get_json(args.endpoint + '/v1/models')
        write_json(run_dir / 'models.json', models)
        model = next((m for m in models['data'] if m['id'] == MODEL), None)
        if model is None: raise RuntimeError('Endpoint does not serve expected model')
        if model.get('max_model_len', 4096) != 4096: raise RuntimeError('Endpoint context differs from baseline')
    except Exception as e:
        write_json(run_dir / 'status.json', {'status': 'preflight_failed', 'error': str(e), 'ended_at': utc()})
        from report import generate
        generate(run_dir)
        raise
    recorder = Recorder(run_dir / 'requests.jsonl')
    done = threading.Event()
    thread = threading.Thread(target=sampler, args=(args.endpoint, run_dir / 'samples.jsonl', done), daemon=True)
    thread.start()
    safety, stages = Safety(), []
    status = 'complete'
    try:
        for cap in args.output_caps:
            eligible = [r for r in records if r['prompt_tokens_expected'] + cap <= 4096]
            if not eligible: continue
            for group in ('short', 'medium', 'long'):
                selected = [r for r in eligible if r['length_group'] == group]
                if selected and not safety.stop.is_set():
                    warmup = [selected[i % len(selected)] for i in range(5)]
                    stages.append(run_stage(warmup, 1, cap, 0, True, args, recorder, validator, safety))
            for concurrency in sorted(args.concurrency):
                if safety.stop.is_set(): break
                stage_inventory = inventory()
                write_json(run_dir / f'inventory-cap{cap}-c{concurrency}.json', stage_inventory)
                if stage_inventory['gpus'].get('returncode') != 0 or stage_inventory['processes'].get('returncode') != 0:
                    raise RuntimeError('GPU inventory failed before stage')
                for repetition in range(1, args.repetitions + 1):
                    if safety.stop.is_set(): break
                    shuffled = eligible.copy()
                    random.Random(args.seed + repetition).shuffle(shuffled)
                    stage = run_stage(shuffled, concurrency, cap, repetition, False, args, recorder, validator, safety)
                    stages.append(stage)
                    write_json(run_dir / 'stages.json', stages)
        if safety.stop.is_set(): status = 'stopped'
        if not any(not s['warmup'] for s in stages): status = 'no_measured_requests'
    except BaseException as e:
        status = 'interrupted' if isinstance(e, KeyboardInterrupt) else 'failed'
        write_json(run_dir / 'exception.json', {'type': type(e).__name__, 'message': str(e)})
        raise
    finally:
        done.set()
        thread.join(timeout=15)
        recorder.close()
        write_json(run_dir / 'stages.json', stages)
        write_json(run_dir / 'inventory-after.json', inventory())
        write_json(run_dir / 'status.json', {'status': status, 'stop_reason': safety.reason, 'ended_at': utc()})
        from report import generate
        generate(run_dir)
    print('Status:', status, flush=True)
    if status != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
