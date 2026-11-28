"""Protocol checks, not answer-quality scoring; never execute model tool output."""
import json
from pathlib import Path
from urllib.error import HTTPError

from common import MODEL, OFF, check_neighbors, gpu_snapshot, http, owned_server, save, stamp
from runner import stream_request

ADD_TOOL = {"type": "function", "function": {"name": "add_numbers",
    "description": "Add two numbers.", "parameters": {"type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"], "additionalProperties": False}}}


def run(base, directory):
    state = owned_server(base)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checks = []
    evidence = {}
    def request(prompt, **extra):
        return http(base, "/v1/chat/completions", {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}], "max_tokens": 128,
            "seed": 0, "temperature": 0.6, **OFF, **extra}, evidence=evidence)
    def check(name, operation):
        evidence.update(http_status=None, response_body=None)
        try:
            response = operation()
            checks.append({"name": name, "status": "PASS", "response": response})
        except Exception as error:
            if isinstance(error, HTTPError):
                with error:
                    evidence.update(http_status=error.code, response_body=error.read().decode("utf-8", errors="replace"))
            checks.append({"name": name, "status": "FAIL", **evidence, "error": type(error).__name__ + ": " + str(error)})
        save(directory / "function-checks.json", {"config": state["config"],
            "vllm_version": state["vllm_version"], "checks": checks})
    def models():
        result = http(base, "/v1/models", evidence=evidence)
        assert MODEL in [item["id"] for item in result.get("data", [])], "Model missing"
        return result
    def chat():
        result = request("Reply briefly with hello.")
        assert result["choices"][0]["message"].get("content"), "Empty chat content"
        return result
    def streaming():
        result = stream_request(base, {"model": MODEL,
            "messages": [{"role": "user", "content": "Reply with one short greeting."}],
            "max_tokens": 32, "seed": 0, **OFF, "stream": True,
            "stream_options": {"include_usage": True}}, "off", directory / "stream-chunks.jsonl", evidence=evidence)
        assert result["content"] and result["ttft_any_seconds"] is not None, "Missing streaming content"
        assert result["usage"] and result["usage"].get("completion_tokens"), "Missing usage"
        assert result["stream_done"] and result["finish_reason"], "Incomplete SSE response"
        return result
    def tool():
        result = request("Use add_numbers to add 3 and 5. Return only the tool call.",
                         tools=[ADD_TOOL], tool_choice="auto")
        calls = result["choices"][0]["message"].get("tool_calls", [])
        assert calls, "No structured tool call"
        function = calls[0]["function"]
        assert function["name"] == "add_numbers", "Wrong tool name"
        arguments = json.loads(function["arguments"])
        assert set(arguments) == {"a", "b"} and all(type(arguments[key]) in (int, float) for key in arguments), "Invalid tool arguments"
        return result
    def schema():
        specification = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
                         "required": ["status"], "additionalProperties": False}
        result = request("Return a JSON object whose status is ok.", response_format={"type": "json_schema",
            "json_schema": {"name": "status", "strict": True, "schema": specification}})
        content = json.loads(result["choices"][0]["message"]["content"])
        assert content == {"status": "ok"}, "Response violates the requested schema"
        return result
    before = gpu_snapshot()
    if check_neighbors(state, before):
        raise RuntimeError("Unrelated GPU process before function checks")
    for name, operation in (("model_list", models), ("chat", chat), ("streaming", streaming),
                            ("tool_call", tool), ("json_schema", schema)):
        print("Function check: " + name, flush=True)
        check(name, operation)
    after = gpu_snapshot()
    problems = check_neighbors(state, after)
    result = {"timestamp": stamp(), "config": state["config"], "vllm_version": state["vllm_version"],
              "checks": checks, "gpu_before": before, "gpu_after": after, "problems": problems,
              "status": "PASS" if not problems and all(item["status"] == "PASS" for item in checks) else "FAIL"}
    save(directory / "function-checks.json", result)
    return directory / "function-checks.json"
