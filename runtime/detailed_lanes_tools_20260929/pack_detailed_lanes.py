#!/usr/bin/env python3
"""Losslessly transport rounded 10-ns event ticks with delta varints."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

RESOLUTION_US = .01


def encode(value, output):
    assert value >= 0
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)


def decode(data):
    value = shift = 0
    for byte in data:
        value |= (byte & 127) << shift
        if byte < 128:
            yield value
            value = shift = 0
        else:
            shift += 7
            assert shift <= 63
    assert shift == 0


def transform(source, output, unpack=False):
    with gzip.open(source, "rt") as stream:
        data = json.load(stream)
    if unpack:
        assert data["schema"] == "decode_detailed_packed_v1"
        for batch in data["batches"]:
            events = []
            for lane in batch.pop("packed_lanes"):
                values = list(decode(base64.b64decode(lane["data"])))
                assert len(values) == lane["count"] * 4
                start = 0
                for index in range(0, len(values), 4):
                    delta, duration, symbol, meta = values[index:index+4]
                    start += delta
                    events.append([lane["rank"], round(start * RESOLUTION_US, 2),
                                   round(duration * RESOLUTION_US, 2), symbol,
                                   lane["streams"][meta >> 3], (meta & 7) - 1])
            batch["events"] = events
        data["schema"] = "decode_detailed_lanes_v1"
    else:
        assert data["schema"] == "decode_detailed_lanes_v1"
        data["source_export_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        for batch in data["batches"]:
            lanes = []
            events = batch.pop("events")
            for rank in (0, 4):
                selected = [e for e in events if e[0] == rank]
                streams = sorted({e[4] for e in selected})
                encoded = bytearray()
                previous = 0
                for event in selected:
                    _, start_us, duration_us, symbol, stream, step = event
                    start = round(start_us / RESOLUTION_US)
                    duration = max(1, round(duration_us / RESOLUTION_US))
                    assert -1 <= step <= 4
                    for value in (start-previous, duration, symbol, (streams.index(stream) << 3) | (step+1)):
                        encode(value, encoded)
                    previous = start
                lanes.append(dict(rank=rank, streams=streams, count=len(selected),
                                  data=base64.b64encode(encoded).decode()))
            batch["packed_lanes"] = lanes
        data["schema"] = "decode_detailed_packed_v1"
        data["resolution_us"] = RESOLUTION_US
        data["notes"].append("Transport uses 0.01-us rounded event boundaries/durations; this is not hardware timing accuracy")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("pack", "unpack"))
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    transform(args.source, args.output, args.command == "unpack")
