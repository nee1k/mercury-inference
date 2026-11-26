#!/usr/bin/env python3
"""Read-only remote checkpoint inspection; writes only user deployment records."""
import concurrent.futures
import datetime
import importlib.metadata
import json
from pathlib import Path
import requests
from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url

ROOT = Path(__file__).resolve().parent

def inspect(model):
    result = dict(model)
    try:
        token = get_token()
        info = HfApi(token=token).model_info(model['repo'], files_metadata=True)
        result.update(revision=info.sha, gated=info.gated, license=(info.card_data or {}).get('license'), config=info.config)
        files = [{'name': f.rfilename, 'size': f.size} for f in info.siblings]
        result['files'] = files
        weight_files = [f for f in files if f['name'].endswith('.safetensors') and '/' not in f['name']]
        # Mistral format uses consolidated weights; other models use HF safetensors.
        if model.get('mistral'):
            weight_files = [f for f in weight_files if f['name'].startswith('consolidated')]
        else:
            weight_files = [f for f in weight_files if not f['name'].startswith('consolidated')]
        result['weight_bytes'] = sum(f['size'] or 0 for f in weight_files)
        result['weight_files'] = [f['name'] for f in weight_files]
        if not weight_files: raise ValueError('No expected safetensors files found')
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        config_url = hf_hub_url(model['repo'], 'config.json', revision=info.sha)
        response = requests.get(config_url, headers=headers, timeout=30)
        response.raise_for_status()
        result['checkpoint_config'] = response.json()
        get_hf_file_metadata(hf_hub_url(model['repo'], weight_files[0]['name'], revision=info.sha), token=token, timeout=30)
        result['access'] = 'available'
    except Exception as e:
        # Do not persist exception URLs, headers, or credentials.
        result.update(access='blocked', error_type=type(e).__name__, http_status=getattr(getattr(e, 'response', None), 'status_code', None))
    return result

if __name__ == '__main__':
    models = json.loads((ROOT/'models.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, models))
    report = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'token_available': bool(get_token()), 'versions': {}, 'models': results}
    for name in ('vllm','torch','triton','transformers','mistral-common','huggingface-hub'):
        try: report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['versions'][name] = None
    (ROOT/'state'/'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in results:
        print(row['key'], row['access'], row.get('http_status'), row.get('revision'), round(row.get('weight_bytes',0)/2**30,2), 'GiB weights', flush=True)
