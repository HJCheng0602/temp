#!/usr/bin/env python3
"""Analyze exported Nsight Systems SQLite traces conservatively.

The useful dependency evidence in an eager CUDA capture is in
``CUPTI_ACTIVITY_KIND_KERNEL`` plus CUDA runtime API rows.  This parser adds
same-device/context/stream order and explicit event/synchronize barriers when
the SQLite export contains their identifiers.  It never joins NCCL rows by
name or timestamp; communicator/sequence metadata is required for that.

Nsight ``.nsys-rep`` files must be exported to SQLite first, for example with
``nsys export --type sqlite --output trace trace.nsys-rep``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

LENGTHS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
LEN_RE = re.compile(r"(?<!\d)(2|4|8|16|32|64|128)\s*[kK](?!\w)")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(db: sqlite3.Connection) -> dict[str, str]:
    rows = db.execute("select name from sqlite_master where type='table'").fetchall()
    return {str(row[0]).lower(): str(row[0]) for row in rows}


def _columns(db: sqlite3.Connection, table: str) -> dict[str, str]:
    return {str(row[1]).lower(): str(row[1])
            for row in db.execute(f"pragma table_info({_q(table)})")}


def _pick(columns: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


def _table(tables: dict[str, str], *needles: str) -> str | None:
    for key, value in tables.items():
        if all(needle.lower() in key for needle in needles):
            return value
    return None


def _strings(db: sqlite3.Connection, tables: dict[str, str]) -> dict[int, str]:
    table = tables.get("stringids") or _table(tables, "string")
    if not table:
        return {}
    cols = _columns(db, table)
    ident = _pick(cols, "id", "stringId")
    value = _pick(cols, "value", "string", "name")
    if not ident or not value:
        return {}
    return {int(row[0]): str(row[1]) for row in
            db.execute(f"select {_q(ident)}, {_q(value)} from {_q(table)}")}


def _length(path: Path) -> int | None:
    match = LEN_RE.search(str(path))
    return int(match.group(1)) * 1024 if match else None


def _value(row: sqlite3.Row, columns: dict[str, str], *names: str) -> Any:
    column = _pick(columns, *names)
    return row[column] if column else None


def _kernel_rows(db: sqlite3.Connection, tables: dict[str, str], strings: dict[int, str]):
    table = tables.get("cupti_activity_kind_kernel") or _table(tables, "kernel")
    if not table:
        return [], None
    cols = _columns(db, table)
    start = _pick(cols, "start", "start_ns")
    end = _pick(cols, "end", "end_ns")
    if not start or not end:
        return [], table
    query = f"select * from {_q(table)} where {_q(start)} is not null and {_q(end)} is not null"
    nodes = []
    for index, row in enumerate(db.execute(query)):
        name_id = _value(row, cols, "nameId", "name_id")
        name = strings.get(int(name_id), str(name_id)) if name_id is not None else table
        nodes.append({
            "id": index, "name": name,
            "start": float(_value(row, cols, "start", "start_ns")),
            "end": float(_value(row, cols, "end", "end_ns")),
            "device": _value(row, cols, "deviceId", "device_id", "device"),
            "context": _value(row, cols, "contextId", "context_id", "context"),
            "stream": _value(row, cols, "streamId", "stream_id", "stream"),
            "correlation": _value(row, cols, "correlationId", "correlation_id"),
            "deps": [],
        })
    return nodes, table


def _runtime_rows(db: sqlite3.Connection, tables: dict[str, str], strings: dict[int, str]):
    table = (tables.get("cupti_activity_kind_runtime") or
             tables.get("cupti_activity_kind_api") or _table(tables, "runtime"))
    if not table:
        return [], None
    cols = _columns(db, table)
    start = _pick(cols, "start", "start_ns")
    end = _pick(cols, "end", "end_ns")
    if not start or not end:
        return [], table
    rows = []
    for row in db.execute(f"select * from {_q(table)} where {_q(start)} is not null and {_q(end)} is not null"):
        name_id = _value(row, cols, "nameId", "name_id")
        name = strings.get(int(name_id), str(name_id)) if name_id is not None else ""
        rows.append({
            "name": name, "start": float(_value(row, cols, "start", "start_ns")),
            "end": float(_value(row, cols, "end", "end_ns")),
            "stream": _value(row, cols, "streamId", "stream_id", "stream"),
            "context": _value(row, cols, "contextId", "context_id", "context"),
            "event": _value(row, cols, "eventId", "event_id", "event"),
            "correlation": _value(row, cols, "correlationId", "correlation_id"),
        })
    return rows, table


def _add_edge(nodes: list[dict[str, Any]], child: int, parent: int) -> bool:
    if child != parent and parent not in nodes[child]["deps"]:
        nodes[child]["deps"].append(parent)
        return True
    return False


def _graph(nodes: list[dict[str, Any]], runtime: list[dict[str, Any]]) -> dict[str, int]:
    edges = defaultdict(int)
    streams: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
    for node in nodes:
        streams[(node["device"], node["context"], node["stream"])].append(node["id"])
    for values in streams.values():
        values.sort(key=lambda i: (nodes[i]["start"], nodes[i]["end"], i))
        for left, right in zip(values, values[1:]):
            if _add_edge(nodes, right, left):
                edges["stream_order"] += 1

    # CUDA event records/waits are only connected when both rows expose an
    # event identifier.  Names alone are deliberately insufficient.
    records: dict[Any, int] = {}
    for row in sorted(runtime, key=lambda item: (item["start"], item["end"])):
        name = row["name"].lower()
        event_id = row["event"]
        if event_id is not None and ("eventrecord" in name or "event_record" in name):
            prior = [n["id"] for n in nodes if n["end"] <= row["start"] and
                     n["stream"] == row["stream"] and n["context"] == row["context"]]
            if prior:
                records[event_id] = max(prior, key=lambda i: nodes[i]["end"])
        if event_id is not None and ("streamwaitevent" in name or "stream_wait_event" in name):
            parent = records.get(event_id)
            if parent is not None:
                following = [n["id"] for n in nodes if n["start"] >= row["end"] and
                             n["stream"] == row["stream"] and n["context"] == row["context"]]
                if following and _add_edge(nodes, min(following, key=lambda i: nodes[i]["start"]), parent):
                    edges["event_wait"] += 1

    # A device/context synchronize is an explicit barrier.  It creates edges
    # only from kernels ending before the API to the first later kernel on each
    # stream; this remains conservative because it does not infer NCCL peers.
    for row in runtime:
        name = row["name"].lower().replace(" ", "")
        if "synchronize" not in name:
            continue
        before = [n for n in nodes if n["end"] <= row["start"]]
        after = [n for n in nodes if n["start"] >= row["end"]]
        if "devicesynchronize" in name or "contextsynchronize" in name:
            groups = {(n["device"], n["context"], n["stream"]) for n in after}
            for key in groups:
                later = min((n for n in after if (n["device"], n["context"], n["stream"]) == key),
                            key=lambda n: n["start"], default=None)
                if later:
                    for parent in before:
                        if _add_edge(nodes, later["id"], parent["id"]):
                            edges["explicit_sync"] += 1
    return dict(edges)


def _path(nodes: list[dict[str, Any]]) -> list[int]:
    indegree = [len(n["deps"]) for n in nodes]
    children: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        for parent in node["deps"]:
            children[parent].append(node["id"])
    ready = [n["id"] for n in nodes if indegree[n["id"]] == 0]
    order = []
    while ready:
        current = ready.pop()
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        return []
    for i in order:
        node = nodes[i]
        node["score"] = node["end"] - node["start"]
        node["prev"] = None
        for parent in node["deps"]:
            gap = max(0.0, node["start"] - nodes[parent]["end"])
            score = nodes[parent]["score"] + gap + node["end"] - node["start"]
            if score > node["score"]:
                node["score"], node["prev"] = score, parent
    last = max(nodes, key=lambda n: n["score"])["id"] if nodes else None
    result = []
    while last is not None:
        result.append(last)
        last = nodes[last]["prev"]
    return list(reversed(result))


def analyze(path: Path) -> dict[str, Any]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = _tables(db)
        strings = _strings(db, tables)
        nodes, kernel_table = _kernel_rows(db, tables, strings)
        runtime, runtime_table = _runtime_rows(db, tables, strings)
        edges = _graph(nodes, runtime)
    finally:
        db.close()
    chain = _path(nodes)
    if nodes and not chain:
        status = "dependency_cycle"
    elif nodes:
        status = "partial_explicit_cuda_dependencies"
    else:
        status = "no_kernel_rows"
    top = []
    for i in sorted(chain, key=lambda x: nodes[x]["end"] - nodes[x]["start"], reverse=True)[:10]:
        n = nodes[i]
        top.append({"name": n["name"], "duration_us": (n["end"] - n["start"]) / 1000,
                    "device": n["device"], "context": n["context"], "stream": n["stream"]})
    return {"source": str(path), "critical_path_status": status,
            "critical_path_is_end_to_end": False, "kernel_count": len(nodes),
            "runtime_count": len(runtime), "edge_counts": edges,
            "partial_dependency_path_us": (nodes[chain[-1]]["end"] - nodes[chain[0]]["start"]) / 1000 if chain else 0,
            "partial_dependency_path_top10": top,
            "tables": {"kernel": kernel_table, "runtime": runtime_table},
            "unknowns": ["NCCL communicator/sequence dependencies are not inferred",
                         "CPU launch-to-kernel correlation is not a GPU dependency edge",
                         "missing event IDs prevent event-wait edges"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = ([args.input] if args.input.is_file() else
             sorted(set(args.input.rglob("*.sqlite")) | set(args.input.rglob("*.sqlite3"))))
    rows = [analyze(path) for path in paths]
    report = {"schema": "prefill_nsys_observed_dependency_v1", "traces": rows,
              "collection_requirements": ["nsys export --type sqlite", "cuda,nvtx tracing",
                                          "runtime API records with event IDs", "rank/role and prompt-length NVTX ranges",
                                          "explicit NCCL communicator and sequence metadata for collective edges"],
              "limitations": ["No end-to-end critical path is claimed without cross-stream, launch, event, and collective evidence"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "traces": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
