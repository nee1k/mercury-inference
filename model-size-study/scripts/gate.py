"""Function and provisional latency gate; missing evidence fails closed."""
import argparse
import math
from pathlib import Path

import benchmark
import function_checks
from common import CONFIG, MODEL, ROOT, load, owned_server, save, stamp


def compare(baseline, candidate, functions_pass):
    problems = []
    if not functions_pass:
        problems.append("Function checks failed or missing")
    for concurrency in (1, 4):
        for metric in ("p95_ttft_ms", "p95_tpot_ms"):
            reference = baseline.get(concurrency, {}).get(metric)
            observed = candidate.get(concurrency, {}).get(metric)
            if not all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in (reference, observed)):
                problems.append("Missing/invalid c%d %s evidence" % (concurrency, metric))
            elif observed > 1.20 * reference:
                problems.append("c%d %s exceeds baseline by more than 20%%" % (concurrency, metric))
    return {"status": "FAIL" if problems else "PASS", "problems": problems,
            "baseline": baseline, "candidate": candidate, "threshold_increase_percent": 20}


def read_chat(manifest):
    values = {}
    for filename in load(manifest).get("cells", []):
        cell = load(filename)
        if cell.get("shape") != "chat" or cell.get("concurrency") not in (1, 4):
            continue
        if cell.get("status") != "VALID":
            raise ValueError("Invalid cell cannot serve as gate evidence")
        raw = load(cell["raw_file"])
        if benchmark.validate_result(raw, cell["prompts"], cell["output_tokens"], cell["input_tokens"]):
            raise ValueError("Gate cell evidence is incomplete")
        concurrency = cell["concurrency"]
        if concurrency in values:
            raise ValueError("Ambiguous duplicate gate baseline cell")
        values[concurrency] = {key: raw[key] for key in ("p95_ttft_ms", "p95_tpot_ms")}
    return values


def run(base, model, baseline):
    owned_server(base, model)
    directory = ROOT / "results" / CONFIG / ("gate-" + stamp())
    directory.mkdir()
    try:
        reference = read_chat(baseline)
        functions = function_checks.run(base, directory / "functions")
        passed = load(functions)["status"] == "PASS"
        if passed:
            manifest = benchmark.run(base, dry=True, kind="gate")
            measured = read_chat(manifest)
        else:
            manifest, measured = None, {}
        result = compare(reference, measured, passed)
        result.update(function_checks=str(functions), candidate_manifest=str(manifest), baseline_manifest=str(baseline))
    except Exception as error:
        result = {"status": "FAIL", "problems": [type(error).__name__ + ": " + str(error)], "baseline_manifest": str(baseline)}
    save(directory / "gate.json", result)
    print("Gate " + result["status"] + ": " + str(directory / "gate.json"), flush=True)
    return directory / "gate.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("baseline_result")
    args = parser.parse_args()
    path = run(args.base_url, args.model, args.baseline_result)
    raise SystemExit(0 if load(path)["status"] == "PASS" else 1)
