"""Read public HF metadata only: no weight requests, credentials or model imports."""
import ast
import datetime
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "candidates"
OUT.mkdir(exist_ok=True)
proposals = json.loads((ROOT / "configs" / "candidate-proposals.json").read_text())
inventory = json.loads((ROOT / "results" / "inventory" / "inventory.json").read_text())
cache = Path(inventory["hf_cache"])
site = Path(sys.executable).parent.parent / "lib" / "python3.12" / "site-packages" / "vllm"
registry = (site / "model_executor" / "models" / "registry.py").read_text()
tool_help = (ROOT / "results" / "inventory" / "vllm-serve-help-all.txt").read_text()
reasoning_source = (site / "reasoning" / "__init__.py").read_text()
report = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "status": "METADATA_COLLECTION", "vllm_version": inventory["package_versions"]["vllm"],
          "models": [], "weight_downloads": False}


def fetch(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "mercury-study-metadata/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        # Never permit a weight request, and bound metadata response size.
        raw = response.read(12 * 1024 * 1024 + 1)
    if len(raw) > 12 * 1024 * 1024:
        raise ValueError("Metadata response exceeds limit")
    data = json.loads(raw)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


selection_path = ROOT / "state" / "selection.json"
selected = set(json.loads(selection_path.read_text())["selected"]) if selection_path.is_file() else None
for proposal in proposals["models"]:
    if selected is not None and proposal["id"] not in selected:
        continue
    model = proposal["id"]
    print("Reading public metadata: " + model, flush=True)
    entry = dict(proposal)
    directory = OUT / model.replace("/", "--")
    directory.mkdir(exist_ok=True)
    try:
        cached = next((item for item in inventory["cached_models"] if item["model"] == model), None)
        pinned = cached["snapshots"][0]["revision"] if cached and cached["snapshots"] else "main"
        api = "https://huggingface.co/api/models/" + model + "/revision/" + pinned + "?blobs=true"
        metadata = fetch(api, directory / "hf-api.json")
        entry["api_url"] = api
        if metadata.get("gated") is not False or metadata.get("private", False):
            entry["status"] = "DROP_GATED_PRIVATE_OR_UNVERIFIED_ACCESS"
            report["models"].append(entry)
            continue
        revision = metadata["sha"]
        entry["revision"] = revision
        base = "https://huggingface.co/" + model + "/resolve/" + revision + "/"
        config = fetch(base + "config.json", directory / "config.json")
        architectures = config.get("architectures", [])
        entry["installed_architecture_registered"] = bool(architectures) and all('"' + name + '"' in registry for name in architectures)
        entry["architectures"] = architectures
        entry["remote_code_policy"] = "Never enable; require installed architecture implementation and validate at dry run"
        entry["tool_parser_listed_in_help"] = entry["tool_parser"] in tool_help
        entry["reasoning_parser_registered_in_installed_source"] = ('"' + entry["reasoning_parser"] + '"') in reasoning_source
        siblings = {item["rfilename"]: item for item in metadata.get("siblings", [])}
        index = fetch(base + "model.safetensors.index.json", directory / "model.safetensors.index.json")
        shard_names = set(index["weight_map"].values())
        shards = [siblings[name] for name in sorted(shard_names)]
        if any(not isinstance(item.get("size"), int) for item in shards):
            raise ValueError("HF API did not provide exact shard sizes")
        weight_bytes = sum(item["size"] for item in shards)
        selected = shard_names | {name for name in siblings if "/" not in name and
                                 name.endswith((".json", ".jinja", ".txt", ".model"))}
        entry["weight_bytes_hf_api"] = weight_bytes
        entry["selected_download_bytes_hf_api"] = sum(siblings[name]["size"] for name in selected)
        entry["selected_files"] = sorted(selected)
        local = cache / ("models--" + model.replace("/", "--"))
        missing_bytes = 0
        for name in selected:
            item = siblings[name]
            snapshot_file = local / "snapshots" / revision / name
            blob = item.get("lfs", {}).get("sha256", item.get("blobId", ""))
            present = snapshot_file.is_file() and snapshot_file.stat().st_size == item["size"]
            if not present and blob:
                blob_file = local / "blobs" / blob
                present = blob_file.is_file() and blob_file.stat().st_size == item["size"]
            if not present:
                missing_bytes += item["size"]
        entry["additional_download_bytes_estimate"] = missing_bytes
        entry["cache_note"] = "Size/presence check, not a checksum verification"
        text = config.get("text_config", config)
        layers = text["num_hidden_layers"]
        heads = text["num_key_value_heads"]
        head_dim = text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])
        layer_types = text.get("layer_types", [])
        full_layers = layer_types.count("full_attention") if "linear_attention" in layer_types else layers
        kv_bytes = 2 * full_layers * heads * head_dim * 2
        entry["kv_bytes_per_token_bf16"] = kv_bytes
        entry["kv_formula"] = "2 * attention_layers * KV_heads * head_dim * 2 bytes"
        recurrent_bytes = 0
        if "linear_attention" in layer_types:
            linear_layers = layer_types.count("linear_attention")
            recurrent_bytes = linear_layers * text["linear_num_value_heads"] * text["linear_key_head_dim"] * text["linear_value_head_dim"] * 4
            conv_channels = 2 * text["linear_num_key_heads"] * text["linear_key_head_dim"] + text["linear_num_value_heads"] * text["linear_value_head_dim"]
            recurrent_bytes += linear_layers * conv_channels * text["linear_conv_kernel_dim"] * 2
            entry["recurrent_state_note"] = "One-sequence estimate: FP32 recurrent state plus BF16 convolution; prefix-state caching and allocator overhead excluded"
        if model == "openai/gpt-oss-120b":
            entry["kv_note"] = "Conservative full-context estimate across all layers; actual alternating sliding-window storage can be smaller"
        entry["additional_recurrent_state_bytes_per_sequence_estimate"] = recurrent_bytes
        entry["fit_estimates"] = []
        for gpus in entry["gpu_tiers"]:
            # Each estimate describes one full-context sequence, not c64 capacity.
            budget = 143771 * 1024 * 1024 * 0.90
            lower = (weight_bytes + kv_bytes * 32768 + recurrent_bytes) / gpus
            entry["fit_estimates"].append({"gpus": gpus,
                "weights_and_one_32k_sequence_bytes_per_gpu_estimate": lower,
                "budget_bytes_per_gpu": budget,
                "headroom_before_runtime_overhead_bytes_per_gpu": budget - lower,
                "verdict": "PLAUSIBLE_NOT_LAUNCH_VALIDATED" if lower < budget else "DOES_NOT_FIT_ESTIMATE",
                "note": "TP division is approximate; replicated tensors, workspaces, CUDA graphs and concurrent KV/state allocations excluded"})
        entry["status"] = "METADATA_VERIFIED_LAUNCH_UNTESTED" if entry["installed_architecture_registered"] else "DROP_ARCHITECTURE_NOT_REGISTERED"
    except urllib.error.HTTPError as error:
        entry["status"] = "ACCESS_BLOCKED" if error.code in (401, 403) else "METADATA_FAILED"
        entry["error"] = "HTTP " + str(error.code)
    except Exception as error:
        entry["status"] = "METADATA_FAILED"
        entry["error"] = type(error).__name__ + ": " + str(error)
    report["models"].append(entry)
    (OUT / "candidates.json").write_text(json.dumps(report, indent=2) + "\n")
report["status"] = "COLLECTED_REVIEW_REQUIRED"
(OUT / "candidates.json").write_text(json.dumps(report, indent=2) + "\n")
print("Metadata collection finished; no weights requested", flush=True)
if not report["models"] or any(item["status"] != "METADATA_VERIFIED_LAUNCH_UNTESTED" for item in report["models"]):
    raise SystemExit(1)
