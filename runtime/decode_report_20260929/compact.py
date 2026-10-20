#!/usr/bin/env python3
"""Pack/unpack a small transferable decode report display archive.

The archive contains no raw profiler trace. Timeline events are quantized to
one-microsecond bins and RLE encoded for the two representative ranks.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import math


SCHEMA = "decode_report_compact_v1"
RESOLUTION_US = 1


def _read_report(path: Path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def _file_meta(path: Path):
    raw = path.read_bytes()
    return {"name": path.name, "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _small_extra(path: Path):
    """Keep useful metadata without copying large request-id summaries."""
    if path.suffix != ".json":
        return None
    if path.name in {"configuration.json", "COMPLETE.json"}:
        return {"kind": "raw_json", "text": path.read_text()}
    value = json.loads(path.read_text())
    if "unprofiled" in path.name:
        rows = value if isinstance(value, list) else value.get("rows", [])
        return {"kind": "timing_summary", "rows": [
            {key: row.get(key) for key in ("batch", "label", "repeat", "whole_batch_step")
             if key in row} for row in rows if isinstance(row, dict)]}
    return {"kind": "source_json", "value": value}


def _mask(event):
    name = str(event.get("name", ""))
    lower = name.lower()
    if "broadcast" in lower:
        return 0
    if "sendrecv" in lower:
        return 4
    if "allreduce" in lower or "allgather" in lower or "nccl" in lower:
        return 2
    return 1


def _rle(timeline):
    span = max(0, int(math.ceil(float(timeline.get("span_us", 0)))))
    bins = {0: [0] * span, 4: [0] * span}
    for event in timeline.get("events", []):
        rank = int(event.get("rank", -1))
        if rank not in bins:
            continue
        mask = _mask(event)
        if not mask:
            continue
        start = max(0, int(math.floor(float(event.get("start_us", 0)))))
        end = min(span, int(math.ceil(float(event.get("start_us", 0)) +
                                     max(0, float(event.get("duration_us", 0))))))
        if end <= start and start < span:
            end = start + 1
        for index in range(start, min(end, span)):
            bins[rank][index] |= mask

    encoded = {}
    for rank, values in bins.items():
        runs = []
        for value in values:
            if runs and runs[-1][1] == value:
                runs[-1][0] += 1
            else:
                runs.append([1, value])
        encoded[str(rank)] = {"resolution_us": RESOLUTION_US,
                              "bin_count": span, "runs": runs}
    return encoded


def pack(report_path: Path, output: Path, extras: list[Path]):
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = _read_report(report_path)
    if report.get("schema") != "decode_report_observed_dependency_v1":
        raise RuntimeError(f"unexpected report schema: {report.get('schema')}")
    batches = []
    for batch in report.get("batches", []):
        timeline = batch.get("timeline", {})
        batches.append({"batch": batch.get("batch"), "label": batch.get("label"),
                        "steps": batch.get("steps"), "mean_span_us": batch.get("mean_span_us"),
                        "rows": batch.get("rows", []), "accounting": batch.get("accounting", {}),
                        "limitations": batch.get("limitations", []),
                        "timeline": {"origin_us": timeline.get("origin_us"),
                                     "span_us": timeline.get("span_us", 0),
                                     "phases": timeline.get("phases", []),
                                     "display_steps": timeline.get("display_steps", 5),
                                     "quantization": "1us floor(start)/ceil(end), Broadcast excluded",
                                     "lanes": _rle(timeline)}})
    files = [_file_meta(report_path)]
    extras_data = {}
    for path in extras:
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(_file_meta(path))
        small = _small_extra(path)
        if small is not None:
            extras_data[path.name] = small
    archive = {"schema": SCHEMA, "source_schema": report["schema"],
               "resolution_us": RESOLUTION_US,
               "quantization": "display-only; raw trace remains at the remote source",
               "files": files, "extras": extras_data, "mode": report.get("mode"),
               "display_quantization": "timeline display uses 1us floor(start)/ceil(end); Broadcast excluded",
               "batches": batches,
               "limitations": report.get("limitations", [])}
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(archive, stream, separators=(",", ":"))


def _unrle(lane, origin):
    events = []
    cursor = 0
    for count, mask in lane.get("runs", []):
        count, mask = int(count), int(mask)
        if count < 0:
            raise RuntimeError("negative RLE count")
        for bit, label in ((1, "compute"), (2, "nccl collective"), (4, "SendRecv")):
            if mask & bit:
                events.append({"start_us": cursor, "duration_us": count,
                               "name": label, "kind": label,
                               "rank": None, "stream": None})
        cursor += count
    if cursor != int(lane.get("bin_count", cursor)):
        raise RuntimeError("RLE bin count mismatch")
    return events


def unpack(path: Path, output: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    if archive.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected compact schema: {archive.get('schema')}")
    batches = []
    for batch in archive.get("batches", []):
        timeline = batch["timeline"]
        events = []
        for rank, lane in timeline.get("lanes", {}).items():
            for event in _unrle(lane, timeline.get("origin_us", 0)):
                event["rank"] = int(rank)
                events.append(event)
        batches.append({"batch": batch.get("batch"), "label": batch.get("label"),
                        "steps": batch.get("steps"), "mean_span_us": batch.get("mean_span_us"),
                        "rows": batch.get("rows", []), "accounting": batch.get("accounting", {}),
                        "limitations": batch.get("limitations", []),
                        "timeline": {"origin_us": timeline.get("origin_us"),
                                     "span_us": timeline.get("span_us"),
                                     "events": events, "phases": timeline.get("phases", []),
                                     "display_steps": timeline.get("display_steps", 5),
                                     "quantization": timeline.get("quantization")}})
    output.parent.mkdir(parents=True, exist_ok=True)
    extracted = output.parent / "compact_extras"
    extracted.mkdir(exist_ok=True)
    validations = []
    metadata = {item["name"]: item for item in archive.get("files", [])}
    for name, value in archive.get("extras", {}).items():
        target = extracted / name
        if value.get("kind") == "raw_json":
            target.write_text(value["text"])
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            validations.append({"name": name, "sha256": actual,
                                "matches_source": actual == metadata.get(name, {}).get("sha256")})
        else:
            target.write_text(json.dumps(value, indent=2) + "\n")
    result = {"schema": "decode_report_compact_unpacked_v1", "mode": archive.get("mode"),
              "quantization": archive.get("quantization"),
              "display_quantization": archive.get("display_quantization"),
              "source_files": archive.get("files", []), "extracted_validations": validations,
              "batches": batches,
              "limitations": archive.get("limitations", [])}
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack_parser = sub.add_parser("pack")
    pack_parser.add_argument("--report", type=Path, required=True)
    pack_parser.add_argument("--output", type=Path, required=True)
    pack_parser.add_argument("--timing", type=Path)
    pack_parser.add_argument("--config", type=Path)
    pack_parser.add_argument("--complete", type=Path)
    unpack_parser = sub.add_parser("unpack")
    unpack_parser.add_argument("--archive", type=Path, required=True)
    unpack_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        extras = [p for p in (args.timing, args.config, args.complete) if p is not None]
        pack(args.report, args.output, extras)
    else:
        unpack(args.archive, args.output)


if __name__ == "__main__":
    main()
