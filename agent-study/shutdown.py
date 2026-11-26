"""Stop recorded study processes and verify the selected GPUs released memory."""

import argparse
import json
import time

from study import CONFIGS, RESULTS, ROOT, VERSION, gpu_snapshot, managed, same_process, save, stamp, stop


def release(config, run_id=None):
    records = [managed(name) for name in ("router", "gpu0", "gpu1", "gpu2", "gpu3")]
    records = [row for row in records if row and row["config"] == config and
               (run_id is None or row.get("run_id") == run_id)]
    replicas = [row for row in records if row.get("gpu") is not None]
    errors = []
    for row in records:
        try:
            stop(row)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{row['name']}: {exc}")
    end = time.monotonic() + 90
    while True:
        samples = [gpu_snapshot(row["gpu"]) for row in replicas]
        live = [row["pid"] for row in records if same_process(row)]
        cleared = all(
            sample["uuid"] == row["gpu_uuid"] and not sample["pids"] and
            sample["used_mib"] <= row["idle_used_mib"] + 8 and sample["gpu_util_percent"] == 0
            for row, sample in zip(replicas, samples)
        )
        if (cleared and not live) or time.monotonic() >= end:
            break
        time.sleep(2)
    evidence = {"config": config, "run_id": run_id, "vllm_version": VERSION, "timestamp": stamp(),
                "gpu_samples": samples, "live_study_pids": live, "errors": errors,
                "memory_released": bool(replicas) and cleared and not live and not errors}
    path = RESULTS / config / f"released-{stamp()}.json"
    evidence["source"] = str(path.relative_to(ROOT))
    save(path, evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    print(f"Cleanup evidence: {path}", flush=True)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=CONFIGS)
    args = parser.parse_args()
    raise SystemExit(0 if release(args.config)["memory_released"] else 1)
