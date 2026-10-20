"""Emit bounded schema evidence before interpreting a new Nsight version."""
import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    result = {}
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for entry in tables:
        name = entry[0]
        quoted = '"' + name.replace('"', '""') + '"'
        columns = [dict(row) for row in db.execute(f"PRAGMA table_info({quoted})")]
        relevant = any(part in name.upper() for part in (
            "KERNEL", "RUNTIME", "DRIVER", "CUDA", "NCCL", "NVTX", "SESSION", "PROCESS", "STRING"))
        result[name] = {"columns": columns}
        if relevant:
            result[name]["count"] = db.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            result[name]["sample"] = [dict(row) for row in db.execute(f"SELECT * FROM {quoted} LIMIT 3")]
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"{len(tables)} tables; schema saved to {args.output}")


if __name__ == "__main__":
    main()
