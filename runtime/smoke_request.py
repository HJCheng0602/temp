"""Send one real, short text request to the local mounted-model server."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--long", action="store_true", help="Exercise indexer and multiple KV pages")
args = parser.parse_args()
prompt = "Reply with exactly the word READY."
if args.long:
    prompt = (
        "The following is padding text.\n" + "sample " * 1300
        + "\nEnd of padding. What is 1 + 1? Reply with only the number."
    )
payload = {
    "model": "deepseek-v41-flash",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": 32,
    "chat_template_kwargs": {"thinking": False},
    "stream": True,
    "stream_options": {"include_usage": True},
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
started = time.perf_counter()
first_content = None
chunks = []
content = []
with urllib.request.urlopen(request, timeout=300) as response:
    for line in response:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        chunks.append(chunk)
        for choice in chunk.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if text:
                if first_content is None:
                    first_content = time.perf_counter() - started
                content.append(text)
record = {
    "request": payload,
    "content": "".join(content),
    "first_content_seconds": first_content,
    "elapsed_seconds": time.perf_counter() - started,
    "chunks": chunks,
    "scope": "Functional smoke request; not a 32K performance benchmark.",
}
filename = "smoke_long_response.json" if args.long else "smoke_response.json"
path = Path("/home/admin/dsv41-vllm-20260917") / filename
path.write_text(json.dumps(record, indent=2))
print(json.dumps({k: v for k, v in record.items() if k not in ("request", "chunks")}, indent=2))
print("USAGE", [c["usage"] for c in chunks if c.get("usage")])
print("SAVED", path)
if not record["content"].strip():
    raise RuntimeError("No generated text returned")
