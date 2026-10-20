#!/usr/bin/env python3
"""Export dependency-relevant rows from an Nsight Systems SQLite database.

This is intended to run beside the captured database on the profiling host.
It emits gzip-compressed JSONL and never loads the kernel/runtime tables into
memory.  NCCL peer/sequence edges are not invented: the export retains NVTX
payloads and rows that can be decoded later, but marks them as opaque.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sqlite3


def quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def tables(db: sqlite3.Connection) -> dict[str, str]:
    return {str(row[0]).lower(): str(row[0]) for row in
            db.execute("select name from sqlite_master where type='table'")}


def columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in db.execute(f"pragma table_info({quote(table)})")]


def strings(db: sqlite3.Connection, table_map: dict[str, str]) -> dict[int, str]:
    table = table_map.get("stringids")
    if not table:
        return {}
    return {int(row[0]): str(row[1]) for row in
            db.execute(f"select id,value from {quote(table)}")}


def write_rows(db: sqlite3.Connection, out, table_map: dict[str, str], key: str,
               kind: str, keep=None, names=None) -> int:
    table = table_map.get(key)
    if table is None:
        # Nsight versions split memcpy/memset into numbered activity tables.
        table = next((value for name, value in table_map.items()
                      if name.startswith(key) and name[len(key):].isdigit()), None)
    if not table:
        return 0
    fields = columns(db, table)
    emitted = 0
    for row in db.execute(f"select * from {quote(table)}"):
        value = dict(zip(fields, row))
        if keep is not None and not keep(value):
            continue
        if names is not None:
            value["resolved_name"] = (names.get(value.get("nameId")) or
                                       names.get(value.get("demangledName")) or
                                       names.get(value.get("shortName")) or "")
            if value.get("text") is None and value.get("textId") is not None:
                value["text"] = names.get(value["textId"])
        for field, item in list(value.items()):
            if isinstance(item, bytes):
                value[field] = item.hex()
                value[field + "Encoding"] = "hex"
        out.write(json.dumps({"kind": kind, **value}, separators=(",", ":")) + "\n")
        emitted += 1
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path, help=".jsonl.gz output")
    parser.add_argument("--all-runtime", action="store_true",
                        help="retain every runtime API row instead of dependency APIs")
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    try:
        table_map = tables(db)
        name_map = strings(db, table_map)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        counts = {}
        with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as out:
            out.write(json.dumps({"kind": "header", "schema": "nsys_dependency_export_v1",
                                  "source": str(args.database),
                                  "tables": sorted(table_map.values()),
                                  "runtime_filter": "all" if args.all_runtime else "dependency_api_names"},
                                 separators=(",", ":")) + "\n")
            counts["kernel"] = write_rows(db, out, table_map, "cupti_activity_kind_kernel",
                                           "kernel", names=name_map)
            counts["runtime"] = write_rows(
                db, out, table_map, "cupti_activity_kind_runtime", "runtime",
                names=name_map,
                keep=(None if args.all_runtime else lambda row:
                      any(token in name_map.get(row.get("nameId"), "").lower()
                          for token in ("launch", "event", "stream", "synchron", "wait", "memcpy", "memset", "nccl"))))
            counts["synchronization"] = write_rows(db, out, table_map,
                                                    "cupti_activity_kind_synchronization",
                                                    "synchronization")
            counts["cuda_event"] = write_rows(db, out, table_map,
                                               "cupti_activity_kind_cuda_event", "cuda_event")
            counts["memcpy"] = write_rows(db, out, table_map,
                                           "cupti_activity_kind_memcpy", "memcpy")
            counts["memset"] = write_rows(db, out, table_map,
                                           "cupti_activity_kind_memset", "memset")
            counts["sync_enum"] = write_rows(db, out, table_map,
                                              "enum_cupti_sync_type", "sync_enum")
            counts["session_start"] = write_rows(db, out, table_map,
                                                  "target_info_session_start_time", "session_start")
            counts["strings"] = write_rows(db, out, table_map, "stringids", "string")
            counts["processes"] = write_rows(db, out, table_map, "processes", "process")
            counts["process_streams"] = write_rows(db, out, table_map,
                                                    "processstreams", "process_stream")
            counts["cuda_contexts"] = write_rows(db, out, table_map,
                                                  "target_info_cuda_context_info", "cuda_context")
            counts["cuda_streams"] = write_rows(db, out, table_map,
                                                 "target_info_cuda_stream", "cuda_stream")
            counts["nvtx_payload_schemas"] = write_rows(db, out, table_map,
                                                         "nvtx_payload_schemas", "nvtx_schema")
            counts["nvtx_payload_entries"] = write_rows(db, out, table_map,
                                                         "nvtx_payload_schema_entries", "nvtx_schema_entry")
            counts["nvtx"] = write_rows(
                db, out, table_map, "nvtx_events", "nvtx", names=name_map,
                keep=lambda row: (str(row.get("text") or name_map.get(row.get("textId"), "")).startswith("report|") or
                                  row.get("binaryData") is not None))
            out.write(json.dumps({"kind": "footer", "counts": counts},
                                 separators=(",", ":")) + "\n")
    finally:
        db.close()
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
