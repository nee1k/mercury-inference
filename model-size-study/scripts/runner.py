"""Sequential streaming natural completions; raw chunks remain untrusted data."""
import argparse
import json
import time
import urllib.request

from common import MODEL, ROOT, local_url, save, stamp, version


def sse_events(response, status=None):
    if status is not None:
        status["done"] = False
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                text = "\n".join(data)
                data = []
                if text == "[DONE]":
                    if status is not None:
                        status["done"] = True
                    return
                yield json.loads(text)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def analyze_chunks(chunks, mode):
    content, spans, fields = "", [], set()
    first_any = first_content = first_tool = None
    usage, finish, tools = None, None, []
    for elapsed, event in chunks:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            fields.update(key for key in ("reasoning", "reasoning_content") if key in delta)
            text = delta.get("content") or ""
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            calls = delta.get("tool_calls") or []
            if text or reasoning or calls:
                if first_any is None:
                    first_any = elapsed
            if text:
                if first_content is None and text.strip():
                    first_content = elapsed
                content += text
                spans.append((len(content), elapsed))
            if calls:
                tools.extend(calls)
                if first_tool is None:
                    first_tool = elapsed
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    end = content.find("</think>")
    method, answer = None, None
    if end >= 0:
        offset = end + len("</think>")
        while offset < len(content) and content[offset].isspace():
            offset += 1
        answer = next((elapsed for length, elapsed in spans if length > offset), None)
        method = "content </think> delimiter"
    elif fields:
        method = "separate reasoning field: " + ",".join(sorted(fields))
        answer = first_content if first_content is not None else first_tool
    elif mode == "off" and not content.lstrip().startswith("<think>"):
        answer = first_content if first_content is not None else first_tool
        method = "reasoning explicitly disabled"
    else:
        method = "UNAVAILABLE: no separate reasoning field or closing </think> delimiter"
    return {"ttft_any_seconds": first_any, "ttft_answer_seconds": answer,
            "first_answer_method": method, "reasoning_fields_seen": sorted(fields),
            "usage": usage, "finish_reason": finish, "content": content, "tool_call_deltas": tools}


def stream_request(base, body, mode, output_path):
    base = local_url(base)
    request = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    chunks, begin, status = [], time.perf_counter(), {}
    with open(output_path, "w") as log:
        with opener.open(request, timeout=900) as response:
            for event in sse_events(response, status):
                elapsed = time.perf_counter() - begin
                chunks.append((elapsed, event))
                log.write(json.dumps({"elapsed_seconds": elapsed, "chunk": event}) + "\n")
                log.flush()
    result = analyze_chunks(chunks, mode)
    result["end_to_end_seconds"] = time.perf_counter() - begin
    result["stream_done"] = status.get("done", False)
    result["raw_chunks"] = str(output_path)
    result["first_raw_choice_chunk"] = next((event for _, event in chunks if event.get("choices")), None)
    return result


def run(base, model):
    base = local_url(base)
    directory = ROOT / "results" / "natural-runner" / stamp()
    directory.mkdir(parents=True)
    installed_version = version()
    prompts = [json.loads(line) for line in (ROOT / "prompts" / "runner.jsonl").read_text().splitlines() if line.strip()]
    results = []
    for prompt in prompts:
        for mode in ("off", "default"):
            # Byte count is a conservative allowance for the short provisional
            # Qwen byte-BPE inputs. Live APIs are never sent /tokenize requests.
            input_allowance = len(json.dumps(prompt["messages"], ensure_ascii=False).encode()) + 512
            if prompt.get("tools"):
                input_allowance += len(json.dumps(prompt["tools"]).encode())
            cap = min(16384, 32768 - input_allowance)
            if cap < 1:
                raise ValueError("Prompt exceeds conservative context budget")
            body = {"model": model, "messages": prompt["messages"], "max_tokens": cap,
                    "stream": True, "stream_options": {"include_usage": True}, "seed": 0,
                    "temperature": 0.6, "top_p": 0.95, "top_k": 20}
            if model.startswith("Qwen/"):
                if mode == "off":
                    body["chat_template_kwargs"] = {"enable_thinking": False}
            elif model.startswith("openai/gpt-oss"):
                if mode == "off":
                    body["reasoning_effort"] = "low"
            else:
                raise ValueError("Reasoning controls not implemented for this model")
            if prompt.get("tools"):
                body.update(tools=prompt["tools"], tool_choice="auto")
            record = {"prompt_id": prompt["id"], "type": prompt["type"], "provisional": True,
                      "base_url": base, "model": model, "mode": mode, "max_tokens": cap,
                      "input_allowance_policy": "UTF-8 bytes plus conservative template allowance; short provisional prompts only",
                      "vllm_version": installed_version, "version_command": "vllm --version (local installed environment; live server version unverified)",
                      "request": body}
            print("Natural runner " + prompt["id"] + " " + mode, flush=True)
            try:
                record.update(stream_request(base, body, mode, directory / (prompt["id"] + "-" + mode + "-chunks.jsonl")))
                usage = record.get("usage") or {}
                count = usage.get("completion_tokens")
                record["completion_tokens"] = count
                record["censored"] = record.get("finish_reason") == "length" or isinstance(count, int) and count >= cap
                record["status"] = "VALID" if isinstance(count, int) and record["ttft_any_seconds"] is not None and record["stream_done"] and record["finish_reason"] else "MISSING_EVIDENCE"
                if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] + cap > 32768:
                    record.update(status="INVALID", error="Actual prompt plus requested cap exceeds fixed context")
            except Exception as error:
                record.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
            results.append(record)
            save(directory / "results.json", {"provisional": True, "concurrency": 1, "results": results})
            if record["status"] not in ("VALID",):
                raise RuntimeError("Natural runner stopped on failed/missing evidence; see saved results")
    return directory / "results.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("model")
    args = parser.parse_args()
    print(run(args.base_url, args.model))
