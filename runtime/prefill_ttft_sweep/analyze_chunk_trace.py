#!/usr/bin/env python3
"""Summarize chunked-prefill Kineto traces without loading all ranks.

The analyzer deliberately maps a kernel to a chunk only when the trace provides
an explicit chunk-labelled CPU/user annotation or an exact External id or
runtime-correlation link to such an annotation. Generic ``execute_*`` scopes
are reported as unmapped unless ``--single-request-chunks`` validates the
one-request ordinal contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
from typing import Any, Iterable


GPU_CATEGORIES = {"kernel", "cuda_kernel", "gpu_kernel", "cuda kernels", "gpu kernels"}
CHUNK_RE = re.compile(
    r"(?:chunk|microbatch|prefill)[^0-9]{0,16}(?:idx|index|id)?[^0-9]{0,4}(\d+)",
    re.IGNORECASE,
)
PREFILL_SCOPE_RE = re.compile(
    r"^execute_(\d+)_context_1\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)",
    re.IGNORECASE,
)
RANK_RE = re.compile(r"(?:^|[_-])rank[_-]?(\d+)(?:\.|[_-]|$)", re.IGNORECASE)
PP_RE = re.compile(r"(?:^|[_-])pp[_-]?(\d+)(?:\.|[_-]|$)", re.IGNORECASE)


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    """Yield traceEvents.item through ijson when installed, one rank otherwise."""
    try:
        import ijson  # type: ignore
    except ImportError:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
        yield from value.get("traceEvents", value if isinstance(value, list) else [])
        return
    with gzip.open(path, "rb") as stream:
        yield from ijson.items(stream, "traceEvents.item", use_float=True)


def arg(event: dict[str, Any], *keys: str) -> Any:
    args = event.get("args")
    if not isinstance(args, dict):
        args = {}
    for key in keys:
        if key in args:
            return args[key]
        if key in event:
            return event[key]
    return None


def link_ids(event: dict[str, Any]) -> set[str]:
    result = set()
    for key in ("External id", "external id", "external_id"):
        value = arg(event, key)
        if value is not None:
            result.add(f"external:{value}")
    for key in ("correlation", "Correlation"):
        value = arg(event, key)
        if value is not None:
            result.add(f"correlation:{value}")
    return result


def is_execute_model_scope(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and str(event.get("cat", "")).lower()
        in {"user_annotation", "cpu_user_annotation", "cpu_op"}
        and "execute_model" in str(event.get("name", "")).lower()
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    )


def prefill_scope_fields(event: dict[str, Any]) -> dict[str, int] | None:
    """Parse the detailed annotation used by the single-request profiler."""
    if event.get("ph") != "X" or event.get("cat") not in {
        "user_annotation", "cpu_user_annotation", "cpu_op"
    }:
        return None
    name = str(event.get("name", ""))
    match = PREFILL_SCOPE_RE.match(name)
    if not match:
        return None
    tokens, sq, sk, sqsq, sqsk = (int(value) for value in match.groups())
    return {"tokens": tokens, "sq": sq, "sk": sk, "sqsq": sqsq, "sqsk": sqsk}


def token_count(event: dict[str, Any]) -> int | None:
    parsed = prefill_scope_fields(event)
    if parsed is not None:
        return parsed["tokens"]
    name_match = re.search(r"num_tokens\s*[=:]\s*(\d+)", str(event.get("name", "")), re.I)
    if name_match:
        return int(name_match.group(1))
    args = event.get("args")
    if isinstance(args, dict):
        for key in ("num_tokens", "Num tokens", "tokens", "Tokens"):
            value = args.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def chunk_number(name: str) -> int | None:
    match = CHUNK_RE.search(name)
    if not match:
        return None
    # Trace producers differ on zero- versus one-based labels. Preserve the
    # observed number and expose the label convention in the manifest.
    return int(match.group(1))


def is_gpu_event(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and str(event.get("cat", "")).lower() in GPU_CATEGORIES
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
        and float(event["dur"]) > 0
    )


def category(name: str) -> str:
    low = name.lower()
    if "nccl" in low or any(token in low for token in ("allreduce", "allgather", "reducescatter", "sendrecv")):
        return "nccl"
    if "memcpy" in low or "memset" in low:
        return "memory"
    return "compute"


def union_length(intervals: list[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return sum(stop - start for start, stop in merged)


def rank_info(path: Path) -> tuple[str, str | None]:
    rank_match = RANK_RE.search(path.name)
    pp_match = PP_RE.search(path.name)
    return (rank_match.group(1) if rank_match else path.stem,
            pp_match.group(1) if pp_match else None)


def trace_paths(root: Path, ranks: set[str]) -> list[Path]:
    paths = sorted(root.rglob("*.trace.json.gz"))
    selected = []
    for path in paths:
        rank, _ = rank_info(path)
        if rank in ranks:
            selected.append(path)
    return selected


def analyze_rank(
    path: Path,
    requested_chunks: int,
    top_n: int,
    single_request_chunks: int | None = None,
    chunk_tokens: int = 8192,
) -> dict[str, Any]:
    rank, pp = rank_info(path)
    scopes: list[dict[str, Any]] = []
    execute_scopes: list[dict[str, Any]] = []
    cpu_records: list[dict[str, Any]] = []
    runtime_events: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    event_count = 0
    for event in iter_events(path):
        if not isinstance(event, dict):
            continue
        event_count += 1
        name = str(event.get("name", ""))
        event_category = str(event.get("cat", "")).lower()
        if (
            event.get("ph") == "X"
            and event_category in {"cpu_op", "user_annotation", "cpu_user_annotation"}
            and isinstance(event.get("ts"), (int, float))
            and isinstance(event.get("dur"), (int, float))
        ):
            cpu_records.append({
                "ts": float(event["ts"]),
                "end": float(event["ts"]) + float(event["dur"]),
                "links": link_ids(event),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
            })
        detailed_prefill = prefill_scope_fields(event)
        if is_execute_model_scope(event) or detailed_prefill is not None:
            execute_scopes.append({
                "name": name,
                "ts": float(event["ts"]),
                "end": float(event["ts"]) + float(event["dur"]),
                "links": link_ids(event),
                "tokens": token_count(event),
                "shape": detailed_prefill,
                "pid": event.get("pid"),
                "tid": event.get("tid"),
            })
        if event.get("ph") == "X" and str(event.get("cat", "")).lower() in {
            "user_annotation", "cpu_user_annotation", "cpu_op"
        }:
            number = chunk_number(name)
            if number is not None:
                scopes.append({
                    "name": name,
                    "chunk_label": number,
                    "ts": float(event.get("ts", 0)),
                    "end": float(event.get("ts", 0)) + float(event.get("dur", 0)),
                    "links": link_ids(event),
                    "pid": event.get("pid"),
                    "tid": event.get("tid"),
                })
        if is_gpu_event(event):
            events.append({
                "name": name,
                "ts": float(event["ts"]),
                "end": float(event["ts"]) + float(event["dur"]),
                "dur": float(event["dur"]),
                "category": category(name),
                "links": link_ids(event),
                "stream": arg(event, "stream", "Stream", "stream id"),
            })
        if event.get("ph") == "X" and (
            "runtime" in event_category or "driver" in event_category
        ):
            runtime_events.append({
                "ts": float(event.get("ts", 0)),
                "end": float(event.get("ts", 0)) + float(event.get("dur", 0)),
                "links": link_ids(event),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
            })

    validation: dict[str, Any] | None = None
    if single_request_chunks is not None:
        ordered = sorted(execute_scopes, key=lambda scope: scope["ts"])
        counts = [scope["tokens"] for scope in ordered]
        sequence = [
            scope.get("shape", {}).get("sk")
            if isinstance(scope.get("shape"), dict)
            else None
            for scope in ordered
        ]
        expected_sequence = [chunk_tokens * (index + 1) for index in range(single_request_chunks)]
        detailed_sequence_valid = (
            all(value is None for value in sequence) or sequence == expected_sequence
        )
        valid = (
            len(ordered) == single_request_chunks
            and all(count == chunk_tokens for count in counts)
            and detailed_sequence_valid
        )
        validation = {
            "valid": valid,
            "expected_scopes": single_request_chunks,
            "observed_scopes": len(ordered),
            "expected_tokens_per_scope": chunk_tokens,
            "observed_tokens": counts,
            "expected_sk_sequence": expected_sequence,
            "observed_sk_sequence": sequence,
            "scope_names": [scope["name"] for scope in ordered],
        }
        if not valid:
            raise ValueError(
                f"{path.name}: single-request validation failed: "
                f"expected {single_request_chunks} execute_model scopes x "
                f"{chunk_tokens} tokens, observed {len(ordered)} x {counts}"
            )
        # The ordinal is the only chunk identity in this mode. The independent
        # one-request/count validation makes this exact rather than a timestamp
        # guess. Preserve zero-based labels for downstream comparisons.
        for ordinal, scope in enumerate(ordered):
            scope["chunk_label"] = ordinal
            scopes.append(scope)

    explicit_by_link: dict[str, set[int]] = defaultdict(set)
    for scope in scopes:
        for identifier in scope["links"]:
            explicit_by_link[identifier].add(scope["chunk_label"])

    # CUDA kernels are asynchronous and often outlive their CPU annotation.
    # Link them through runtime External id -> runtime correlation -> kernel
    # correlation. Runtime records may be nested in the CPU scope, so this is
    # still based on CPU-side profiler evidence rather than GPU containment.
    for scope in scopes:
        scope_links = set(scope["links"])
        # The execute_model annotation is a parent scope. Leaf CPU ops inside
        # it carry the External ids used by CUDA runtime records, so collect
        # those ids before resolving runtime correlation to kernel events.
        for cpu in cpu_records:
            same_thread = (
                scope.get("pid") is None
                or cpu.get("pid") is None
                or (scope.get("pid") == cpu.get("pid") and scope.get("tid") == cpu.get("tid"))
            )
            if same_thread and scope["ts"] <= cpu["ts"] and cpu["end"] <= scope["end"]:
                scope_links.update(cpu["links"])
        for runtime in runtime_events:
            same_thread = (
                scope.get("pid") is None
                or runtime.get("pid") is None
                or (scope.get("pid") == runtime.get("pid") and scope.get("tid") == runtime.get("tid"))
            )
            if not same_thread or not (scope["ts"] <= runtime["ts"] and runtime["end"] <= scope["end"]):
                continue
            scope_links.update(runtime["links"])
        for identifier in scope_links:
            explicit_by_link[identifier].add(scope["chunk_label"])

    buckets: dict[str, dict[str, Any]] = {}
    unmapped = Counter()
    for event in events:
        labels = set()
        for identifier in event["links"]:
            labels.update(explicit_by_link.get(identifier, ()))
        if len(labels) != 1:
            unmapped["no_explicit_chunk_link" if not labels else "ambiguous_chunk_link"] += 1
            continue
        label = str(next(iter(labels)))
        bucket = buckets.setdefault(label, {
            "chunk_label": int(label), "events": 0, "sum_gpu_us": 0.0,
            "intervals": [], "by_category": Counter(), "by_kernel": Counter(),
            "kernel_sum_us": Counter(),
        })
        bucket["events"] += 1
        bucket["sum_gpu_us"] += event["dur"]
        bucket["intervals"].append((event["ts"], event["end"]))
        bucket["by_category"][event["category"]] += event["dur"]
        bucket["by_kernel"][event["name"]] += 1
        bucket["kernel_sum_us"][event["name"]] += event["dur"]

    rows = []
    for label, bucket in sorted(buckets.items(), key=lambda item: int(item[0])):
        row = {
            "chunk_label": bucket["chunk_label"],
            "events": bucket["events"],
            "sum_gpu_us": bucket["sum_gpu_us"],
            "union_gpu_us": union_length(bucket["intervals"]),
            "by_category_sum_gpu_us": dict(bucket["by_category"]),
            "top_kernels_by_sum_gpu_us": [
                {"name": name, "calls": bucket["by_kernel"][name], "sum_gpu_us": duration}
                for name, duration in bucket["kernel_sum_us"].most_common(top_n)
            ],
        }
        rows.append(row)

    labels = [row["chunk_label"] for row in rows]
    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    return {
        "rank": rank,
        "pp_rank": pp,
        "source": str(path),
        "event_count": event_count,
        "gpu_event_count": len(events),
        "explicit_chunk_scope_count": len(scopes),
        "execute_model_scope_count": len(execute_scopes),
        "single_request_validation": validation,
        "requested_chunks": requested_chunks,
        "mapped_chunk_labels": labels,
        "mapping_status": "mapped" if rows else "unmapped",
        "unmapped_gpu_events": dict(unmapped),
        "chunks": rows,
        "first_vs_last": {"first": first, "last": last} if first and last else None,
        "trace_semantics": {
            "sum_gpu_us": "sum of individual GPU event durations; overlaps and waits are not removed",
            "union_gpu_us": "union of mapped GPU event intervals on this rank; still not a critical path",
            "wait_caveat": "CPU scope duration and PP/NCCL synchronization may include waiting; CUDA kernels alone do not prove pure model compute",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("chunk_trace_analysis.json"))
    parser.add_argument("--ranks", default="0,4", help="comma-separated representative rank ids")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument(
        "--single-request-chunks",
        type=int,
        default=None,
        help="validate this many ordered execute_model scopes as one request",
    )
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    ranks = {value.strip() for value in args.ranks.split(",") if value.strip()}
    paths = trace_paths(args.trace_dir, ranks)
    if not paths:
        raise SystemExit(f"no trace files for ranks {sorted(ranks)} below {args.trace_dir}")
    result = {
        "schema": "prefill_chunk_trace_analysis/v1",
        "trace_dir": str(args.trace_dir),
        "ranks_requested": sorted(ranks),
        "ranks_found": [rank_info(path)[0] for path in paths],
        "requested_chunks": args.chunks,
        "mapping_policy": "explicit chunk-labelled CPU/user annotation or exact External id/runtime-correlation link; in single-request mode, validated ordered execute_model or detailed execute_<tokens>_context_1 scopes provide ordinal chunk labels; GPU timestamp containment is never used",
        "single_request_chunks": args.single_request_chunks,
        "chunk_tokens": args.chunk_tokens,
        "ranks": [
            analyze_rank(
                path,
                args.chunks,
                args.top,
                args.single_request_chunks,
                args.chunk_tokens,
            )
            for path in paths
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "ranks": result["ranks_found"]}, indent=2))


if __name__ == "__main__":
    main()
