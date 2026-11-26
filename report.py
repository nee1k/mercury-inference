#!/usr/bin/env python3
"""Regenerate JSON and Markdown reports from one immutable pilot run."""
import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values: return None
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows, duration=None):
    ok = [r for r in rows if not r['error']]
    result = {'requests': len(rows), 'successes': len(ok), 'errors': len(rows) - len(ok),
              'timeouts': sum('timeout' in (r['error'] or '').lower() or 'timed out' in (r['error'] or '').lower() for r in rows),
              'error_rate': (len(rows) - len(ok)) / len(rows) if rows else None,
              'truncations': sum(r['quality']['truncated'] for r in rows),
              'small_sample_warning': len(ok) < 100,
              'completion_tokens_reported': sum((r['usage'] or {}).get('completion_tokens', 0) for r in ok),
              'prompt_token_mismatches': sum((r['usage'] or {}).get('prompt_tokens') is not None and r['usage']['prompt_tokens'] != r['prompt_tokens_expected'] for r in ok),
              'missing_usage_successes': sum(not r['usage'] or r['usage'].get('completion_tokens') is None for r in ok)}
    for name in ('latency_seconds', 'ttft_seconds', 'tpot_seconds', 'decode_seconds'):
        values = [r[name] for r in ok if r[name] is not None]
        result[name] = {'n': len(values), 'p50': percentile(values, .5), 'p95': percentile(values, .95)}
    for name in ('json_valid', 'schema_valid', 'exact_match'):
        passed = sum(not r['error'] and not r['quality']['truncated'] and r['quality'][name] for r in rows)
        result[name + '_rate'] = passed / len(rows) if rows else None
    if duration is not None:
        result['duration_seconds'] = duration
        result['requests_per_second'] = len(ok) / duration if duration > 0 else None
        result['output_tokens_per_second'] = (result['completion_tokens_reported'] / duration
            if duration > 0 and not result['missing_usage_successes'] else None)
    return result


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def telemetry(samples):
    gpus = defaultdict(lambda: {'peak_memory_mib': 0, 'utilization_percent': []})
    metrics = defaultdict(list)
    for sample in samples:
        stdout = sample['gpu'].get('stdout', '')
        for row in list(csv.reader(io.StringIO(stdout)))[1:]:
            try:
                index, uuid, memory, utilization = [x.strip() for x in row]
                gpu = gpus[index]
                gpu['uuid'] = uuid
                gpu['peak_memory_mib'] = max(gpu['peak_memory_mib'], float(memory.split()[0]))
                gpu['utilization_percent'].append(float(utilization.split()[0]))
            except (ValueError, IndexError): continue
        for line in sample.get('server_metrics', '').splitlines():
            if line.startswith('#') or not any(s in line.split(' ', 1)[0] for s in ('queue', 'prefill', 'decode', 'cache_usage', 'num_requests_running', 'num_requests_waiting')): continue
            try:
                name, value = line.rsplit(' ', 1)
                number = float(value)
                if math.isfinite(number): metrics[name].append(number)
            except ValueError: continue
    return {'samples': len(samples), 'metrics_errors': sum('metrics_error' in s for s in samples),
            'gpu_errors': sum(s['gpu'].get('returncode') != 0 for s in samples),
            'gpus': {key: {**value, 'utilization_percent': {'p50': percentile(value['utilization_percent'], .5),
                          'p95': percentile(value['utilization_percent'], .95)}} for key, value in gpus.items()},
            'server_series': {name: {'first': values[0], 'last': values[-1], 'min': min(values), 'max': max(values)} for name, values in metrics.items()}}


def generate(run_dir):
    rows = read_jsonl(run_dir / 'requests.jsonl')
    measured = [r for r in rows if not r['warmup']]
    stages = [s for s in read_json(run_dir / 'stages.json', []) if not s['warmup']]
    groups = defaultdict(list)
    for row in measured:
        groups[(row['output_cap'], row['concurrency'], row['repetition'])].append(row)
    repetitions = []
    for stage in stages:
        key = (stage['output_cap'], stage['concurrency'], stage['repetition'])
        repetitions.append({**stage, **summarize(groups[key], stage['duration_seconds'])})
    conditions = []
    for cap, concurrency in sorted({(r['output_cap'], r['concurrency']) for r in measured}):
        subset = [r for r in measured if r['output_cap'] == cap and r['concurrency'] == concurrency]
        duration = sum(s['duration_seconds'] for s in stages if s['output_cap'] == cap and s['concurrency'] == concurrency)
        conditions.append({'output_cap': cap, 'concurrency': concurrency, **summarize(subset, duration),
                           'by_length': {group: summarize([r for r in subset if r['length_group'] == group])
                                         for group in ('short', 'medium', 'long')}})
    workload = read_json(run_dir / 'workload.json', [])
    report = {'status': read_json(run_dir / 'status.json', {}), 'conditions': conditions,
              'repetitions': repetitions, 'warmup_requests': len(rows) - len(measured),
              'exclusions': read_json(run_dir / 'exclusions.json', []),
              'missing_length_groups': [g for g in ('short', 'medium', 'long') if not any(r['length_group'] == g for r in workload)],
              'telemetry': telemetry(read_jsonl(run_dir / 'samples.jsonl')),
              'notes': ['P95 uses linear interpolation; fewer than 100 successful requests is flagged as a small sample.',
                        'Quality rates count transport failures and truncated responses as failures.',
                        'Token throughput is unavailable if any successful response lacks usage.',
                        'TTFT includes transport, queueing, and prefill. TPOT uses completion token counts, not SSE chunk counts.',
                        'Server metrics are endpoint-wide observations, not per-request attribution; unrelated traffic can contribute.',
                        'Cache state is not reset. Repeated examples may benefit from prefix caching.',
                        'Nonzero prompt-token mismatches require investigating tokenizer/endpoint revision before interpreting length groups.',
                        'No task quality or latency acceptance threshold has been selected.']}
    (run_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def fmt(value): return 'N/A' if value is None else f'{value:.3f}'
    lines = ['# Qwen3-8B pilot report', '', f"Status: {report['status'].get('status', 'unknown')}", '',
             '| Output cap | Concurrency | Requests | Errors | Truncated | P95 TTFT (s) | P95 latency (s) | Req/s | Output tok/s | Exact match |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for c in conditions:
        lines.append('| ' + ' | '.join(str(x) for x in (c['output_cap'], c['concurrency'], c['requests'], c['errors'], c['truncations'],
            fmt(c['ttft_seconds']['p95']), fmt(c['latency_seconds']['p95']), fmt(c['requests_per_second']), fmt(c['output_tokens_per_second']), fmt(c['exact_match_rate']))) + ' |')
    lines += ['', f"Warmup requests excluded: {report['warmup_requests']}",
              f"Excluded example/output-cap pairs: {len(report['exclusions'])}",
              'Missing length groups: ' + (', '.join(report['missing_length_groups']) or 'none'), '',
              'Full P50/P95, TPOT, per-repetition and per-length quality/latency results are in summary.json.', '']
    for index, gpu in report['telemetry']['gpus'].items():
        lines.append(f"GPU {index}: peak sampled memory {gpu['peak_memory_mib']:.0f} MiB; utilization P50/P95 {fmt(gpu['utilization_percent']['p50'])}/{fmt(gpu['utilization_percent']['p95'])}%.")
    lines += ['', *('- ' + note for note in report['notes']), '']
    (run_dir / 'report.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    generate(parser.parse_args().run_directory)
