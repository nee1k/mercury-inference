#!/usr/bin/env python3
"""Validate the local endpoint and preserve actual responses and GPU usage."""
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

out = Path(__file__).resolve().parent / 'results'
out.mkdir(exist_ok=True)
base = 'http://127.0.0.1:8001'

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
        assert response.status == 200
    return result, time.perf_counter() - start

models, model_seconds = request('/v1/models')
assert any(m['id'] == 'Qwen/Qwen3-8B' for m in models['data'])
completion, chat_seconds = request('/v1/chat/completions', {
    'model': 'Qwen/Qwen3-8B',
    'messages': [{'role': 'user', 'content': 'What is 2 + 2? Answer briefly.'}],
    'temperature': 0, 'max_tokens': 128,
    'chat_template_kwargs': {'enable_thinking': False},
})
assert completion['choices'][0]['message']['content'].strip()
(out / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
(out / 'chat-completion.json').write_text(json.dumps(completion, indent=2) + '\n')
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv'], text=True)
report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'models_http_status': 200, 'chat_http_status': 200,
          'models_seconds': model_seconds, 'chat_seconds': chat_seconds,
          'answer': completion['choices'][0]['message']['content'],
          'usage': completion.get('usage'), 'gpu': gpu, 'processes': processes}
(out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
