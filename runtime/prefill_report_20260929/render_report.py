# -*- coding: utf-8 -*-
"""Render a prefill TTFT report from a verified report_data.json.

The renderer is deliberately data-only: it does not infer timings, kernel
names, or accuracy from unrelated artifacts.  See ``SCHEMA`` below for the
small input contract.  All generated artifacts are written below --output.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

PAGE = landscape(A4)
W, H = PAGE
FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

SCHEMA = {
    "metadata": {"ttft_definition": "token-ready to P+CPU handoff+D", "batch": "B1",
                 "chunk_tokens": 8192, "parallelism": "P4D4"},
    "lengths": [{
        "input_length": 2048,
        "ttft_ms": 123.4,                 # or ttft_samples_ms: [..]
        "lanes": [{"name": "PP1 TP4", "events": [
            {"name": "prefill_attention", "start_s": 0.0, "end_s": 0.42,
             "stage": "PP1", "tp": "TP4"}
        ]}],
        "top10_kernels": [{"kernel": "...", "time_ms": 12.3,
                           "percent": 23.4, "path": True}],
    }],
    "accuracy": {"status": "pass|fail|insufficient", "checks": [],
                  "evidence": "..."},
}


def configure_plot_fonts() -> str:
    """Select an installed CJK font; never silently render Chinese as tofu."""
    preferred = ("PingFang SC", "PingFang HK", "Noto Sans CJK SC", "Noto Sans CJK JP",
                 "Source Han Sans CN", "Heiti SC", "Songti SC", "Arial Unicode MS")
    installed = {font_manager.FontProperties(fname=p).get_name(): p
                 for p in font_manager.findSystemFonts()}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans (CJK unavailable)"


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def items(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("lengths", data.get("cases", data.get("rows", [])))
    if isinstance(raw, dict):
        raw = [dict(v, input_length=k) if isinstance(v, dict) else
               {"input_length": k, "ttft_ms": v} for k, v in raw.items()]
    out = []
    for row in raw or []:
        if not isinstance(row, dict) or row.get("input_length", row.get("prompt_tokens")) is None:
            continue
        item = dict(row)
        item["input_length"] = int(item.get("input_length", item.get("prompt_tokens")))
        samples = item.get("ttft_samples_ms", item.get("ttft_ms_samples"))
        if samples is not None:
            vals = [float(x) for x in samples if x is not None]
            item["ttft_ms"] = statistics.median(vals) if vals else None
            item["ttft_min_ms"] = min(vals) if vals else None
            item["ttft_max_ms"] = max(vals) if vals else None
        elif item.get("ttft_ms", item.get("median_ttft_ms")) is not None:
            item["ttft_ms"] = float(item.get("ttft_ms", item.get("median_ttft_ms")))
            item["ttft_min_ms"] = float(item.get("ttft_min_ms", item.get("min_ttft_ms", item["ttft_ms"])))
            item["ttft_max_ms"] = float(item.get("ttft_max_ms", item.get("max_ttft_ms", item["ttft_ms"])))
        else:
            item["ttft_ms"] = item["ttft_min_ms"] = item["ttft_max_ms"] = None
        out.append(item)
    return sorted(out, key=lambda x: x["input_length"])


def event_span(event: dict[str, Any]) -> tuple[float, float] | None:
    """Return seconds; accepts start/end_s, start/duration_s, or *_us."""
    start = event.get("start_s", event.get("start_sec"))
    end = event.get("end_s", event.get("end_sec"))
    if start is None and event.get("start_us") is not None:
        start = float(event["start_us"]) / 1e6
    if end is None and event.get("end_us") is not None:
        end = float(event["end_us"]) / 1e6
    if end is None:
        dur = event.get("duration_s", event.get("duration_sec"))
        if dur is None and event.get("duration_us") is not None:
            dur = float(event["duration_us"]) / 1e6
        if start is not None and dur is not None:
            end = float(start) + float(dur)
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        return None
    return start, end


def lane_events(item: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Flatten lane forms while retaining lane labels and real timestamps."""
    result = []
    for lane in item.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lname = str(lane.get("name", lane.get("label", lane.get("stage", "lane"))))
        for event in lane.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            span = event_span(event)
            if span is None:
                continue
            value = dict(event)
            value["_start"], value["_end"] = span
            value["_lane"] = lname
            result.append(value)
    # Also accept one event list directly on the input-length item.
    for event in item.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        span = event_span(event)
        if span is not None:
            value = dict(event)
            value["_start"], value["_end"] = span
            value["_lane"] = str(event.get("lane", event.get("stage", f"rank{event['rank']}" if event.get("rank") is not None else "lane")))
            result.append(value)
    if not result:
        return []
    metadata = metadata or {}
    # Prefer the request-level origin supplied by the collector.  Falling back
    # to the earliest observed event preserves real gaps when no origin exists.
    origin_value = item.get("request_origin_s", item.get("timeline_origin_s", item.get("origin_s", metadata.get("request_origin_s"))))
    origin = float(origin_value) if origin_value is not None else min(x["_start"] for x in result)
    for event in result:
        event["_start"] -= origin
        event["_end"] -= origin
    return [x for x in result if x["_end"] >= 0]


def top10(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only explicitly attributed critical-path rows.

    A regular time-sorted kernel list is intentionally not treated as a
    critical path: serializing it would make the report claim evidence that
    the profiler did not provide.
    """
    raw = item.get("critical_path_top10", item.get("critical_path"))
    if raw is None:
        candidates = item.get("top10_kernels", item.get("top10", item.get("kernels", []))) or []
        raw = [x for x in candidates if isinstance(x, dict) and
               (x.get("path") is True or x.get("critical_path") is True or x.get("path_role") == "critical")]
    raw = raw or []
    result = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        value = dict(row)
        if value.get("time_ms") is None:
            if value.get("duration_us") is not None:
                value["time_ms"] = float(value["duration_us"]) / 1000
            elif value.get("time_us") is not None:
                value["time_ms"] = float(value["time_us"]) / 1000
        if value.get("time_ms") is not None:
            value["time_ms"] = float(value["time_ms"])
        result.append(value)
    return sorted(result, key=lambda x: x.get("time_ms") or 0, reverse=True)[:10]


def kernel_label(row: dict[str, Any]) -> str:
    """Use a collector-provided concise alias while retaining full symbol in CSV."""
    return str(row.get("display_name", row.get("short_name", row.get("kernel", row.get("name", "unknown kernel")))))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "缺失"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def quality_rows(acc: Any) -> list[dict[str, Any]]:
    if not isinstance(acc, dict):
        return []
    rows = acc.get("rows", acc.get("checks", [])) or []
    return [x if isinstance(x, dict) else {"check": x} for x in rows]


def write_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=170)
    valid = [r for r in rows if r.get("ttft_ms") is not None]
    if valid:
        x = [r["input_length"] for r in valid]
        y = [r["ttft_ms"] for r in valid]
        lo = [r.get("ttft_min_ms", v) for r, v in zip(valid, y)]
        hi = [r.get("ttft_max_ms", v) for r, v in zip(valid, y)]
        ax.plot(x, y, marker="o", linewidth=2, color="#1769aa", label="TTFT")
        if any(a != b for a, b in zip(lo, hi)):
            ax.fill_between(x, lo, hi, color="#1769aa", alpha=.15, label="重复测量范围")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
    else:
        ax.text(.5, .5, "没有可用 TTFT 数据", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefill input length - TTFT")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best") if valid else None
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def event_color(event: dict[str, Any]) -> str:
    value = " ".join(str(event.get(k, "")) for k in ("event_type", "kind", "category", "name")).lower()
    if any(x in value for x in ("host", "cpu", "python", "scheduler")):
        return "#8b6bb1"
    if any(x in value for x in ("handoff", "send", "recv", "p2p", "pp")):
        return "#bd4b3e"
    if any(x in value for x in ("memcpy", "memset", "copy", "offload")):
        return "#d18b3c"
    if any(x in value for x in ("collective", "allreduce", "allgather", "reduce", "nccl")):
        return "#62a87c"
    return "#1769aa"


def write_lane(path: Path, item: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
    events = lane_events(item, metadata)
    names = []
    for event in events:
        if event["_lane"] not in names:
            names.append(event["_lane"])
    fig, ax = plt.subplots(figsize=(11, max(2.8, .48 * max(1, len(names)) + 1.3)), dpi=160)
    phases = item.get("phases", []) or []
    if events:
        for index, name in enumerate(names):
            for event in (e for e in events if e["_lane"] == name):
                ax.broken_barh([(event["_start"], max(0.0, event["_end"] - event["_start"]))],
                               (index - .32, .64), facecolors=event_color(event), alpha=.85)
        for phase in phases:
            span = event_span(phase)
            if span is not None:
                start, end = span
                origin_value = (item.get("request_origin_s", item.get("timeline_origin_s", metadata.get("request_origin_s") if metadata else None)))
                if origin_value is not None:
                    start -= float(origin_value); end -= float(origin_value)
                ax.axvspan(start, end, color="#9aa4ad", alpha=.10, zorder=0)
                if phase.get("name"):
                    ax.text((start + end) / 2, len(names) - .45, str(phase["name"]), ha="center", va="top", fontsize=7, color="#555")
        xmax = max(e["_end"] for e in events)
        ax.set_xlim(left=0, right=max(xmax * 1.04, .001))
        ax.set_yticks(range(len(names)), names)
    else:
        ax.text(.5, .5, "没有可用泳道事件", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
    ax.set_xlabel("Time from prefill start (s)")
    ax.set_title(f"Input length {item['input_length']} tokens | request origin = 0 s")
    legend = [Patch(facecolor="#1769aa", label="compute"), Patch(facecolor="#62a87c", label="collective"),
              Patch(facecolor="#d18b3c", label="memcpy/offload"), Patch(facecolor="#8b6bb1", label="host/scheduler"),
              Patch(facecolor="#bd4b3e", label="P+handoff/PP")]
    ax.legend(handles=legend, loc="upper right", fontsize=7, ncol=3, framealpha=.9)
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def para(c: canvas.Canvas, text: str, x: float, y: float, width: float, size=10, leading=15) -> float:
    c.setFont(FONT, size)
    line = ""
    lines = []
    for ch in str(text):
        if c.stringWidth(line + ch, FONT, size) > width and line:
            lines.append(line)
            line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def _wrap_cell(value: Any, width: float, size: float) -> list[str]:
    text = str(value)
    limit = max(6, int(width / max(size * .55, 1)))
    return [text[i:i + limit] for i in range(0, len(text), limit)] or [""]


def table(c: canvas.Canvas, headers: list[str], rows: list[list[str]], x: float, y: float, widths: list[float], rh=17, size=7.4) -> float:
    """Draw a wrapping table and return its bottom y coordinate."""
    total = sum(widths)
    table_top = y
    c.setFillColor(colors.HexColor("#e9eef3")); c.rect(x, y-rh, total, rh, fill=1, stroke=0)
    c.setFillColor(colors.black); c.setFont(FONT, size)
    for i, val in enumerate(headers):
        c.drawString(x + sum(widths[:i]) + 3, y - 12, str(val))
    y -= rh
    for row in rows:
        lines = [_wrap_cell(value, widths[i], size) for i, value in enumerate(row)]
        height = max(rh, max(len(x) for x in lines) * (size + 2) + 5)
        c.setStrokeColor(colors.HexColor("#d4d8dc")); c.line(x, y, x + total, y)
        for i, cell_lines in enumerate(lines):
            for line_index, txt in enumerate(cell_lines):
                c.setFillColor(colors.black)
                c.drawString(x + sum(widths[:i]) + 3, y - 11 - line_index * (size + 2), txt)
        y -= height
    c.setStrokeColor(colors.HexColor("#9aa4ad")); c.rect(x, y, total, table_top - y, fill=0, stroke=1)
    return y


def footer(c: canvas.Canvas, page: int, metadata: dict[str, Any]) -> None:
    c.setStrokeColor(colors.HexColor("#c4cbd1")); c.line(34, 24, W - 34, 24)
    c.setFillColor(colors.HexColor("#666666")); c.setFont(FONT, 7)
    config = " | ".join(str(metadata.get(k)) for k in ("batch", "parallelism", "chunk_tokens") if metadata.get(k) is not None)
    c.drawString(34, 13, f"Prefill performance | {config}"[:150])
    c.drawRightString(W - 34, 13, f"2026-09-29 | {page}")
    c.setFillColor(colors.black)


def accuracy_text(acc: Any) -> str:
    if not isinstance(acc, dict):
        return "精度结论：无法确认（未提供 accuracy 证据）。"
    status = str(acc.get("status", "insufficient")).lower()
    mapping = {"pass": "通过", "passed": "通过", "fail": "失败", "failed": "失败"}
    label = mapping.get(status, "无法确认")
    checks = acc.get("checks") or []
    detail = "；".join(str(x.get("name", x.get("check", x))) + (f": {x.get('status')}" if isinstance(x, dict) and x.get("status") else "") for x in checks)
    evidence = str(acc.get("evidence", "")).strip()
    return "精度结论：" + label + "。" + ((" 检查：" + detail + "。") if detail else "") + ((" 证据：" + evidence) if evidence else "")


def metadata_text(metadata: dict[str, Any]) -> str:
    """Render collector-provided workload/TTFT definitions without hardcoding."""
    if not metadata:
        return "测量口径与配置：未提供 metadata。"
    labels = {"ttft_definition": "TTFT 定义", "batch": "Batch", "chunk_tokens": "Chunk tokens",
              "parallelism": "并行度", "model": "模型", "backend": "后端", "device": "设备"}
    parts = [f"{labels.get(k, k)}={v}" for k, v in metadata.items() if v is not None and k != "request_origin_s"]
    return "测量口径与配置：" + ("；".join(parts) if parts else "未提供") + "。"


def critical_text(item: dict[str, Any]) -> str:
    if item.get("critical_path_note"):
        return "关键路径状态：" + str(item["critical_path_note"])
    rows = top10(item)
    if not rows:
        return "关键路径状态：无法确认（没有 critical path 归因证据；未将普通 kernel 排序当作关键路径）。"
    return f"关键路径状态：已提供 {len(rows)} 条归因记录，以下为按 observed time 排序的 Top10。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render prefill TTFT report")
    parser.add_argument("--data", type=Path, required=True, help="report_data.json")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    data = load(args.data)
    rows = items(data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    configure_plot_fonts()
    if not rows:
        raise ValueError("report_data.json contains no input lengths")
    root = args.output / "prefill_report_20260929"
    root.mkdir(parents=True, exist_ok=True)
    pdfdir = args.output / "pdf"; pdfdir.mkdir(parents=True, exist_ok=True)
    curve = root / "input_length_ttft_curve.png"; write_curve(curve, rows)
    lane_paths = {}
    for item in rows:
        path = root / f"input_{item['input_length']}_lane.png"
        write_lane(path, item, metadata); lane_paths[item["input_length"]] = path

    with (root / "input_length_ttft.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["input_length", "ttft_ms", "ttft_min_ms", "ttft_max_ms"])
        for r in rows: w.writerow([r["input_length"], r.get("ttft_ms", ""), r.get("ttft_min_ms", ""), r.get("ttft_max_ms", "")])
    with (root / "kernel_top10.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["input_length", "rank", "display_name", "full_kernel_symbol", "time_ms", "percent", "path"])
        for r in rows:
            for rank, k in enumerate(top10(r), 1):
                w.writerow([r["input_length"], rank, kernel_label(k), k.get("kernel", k.get("name", "")), k.get("time_ms", ""), k.get("percent", ""), k.get("path", "")])

    acc = accuracy_text(data.get("accuracy"))
    meta = metadata_text(metadata)
    md = ["# Prefill 性能报告", "", "本报告回答三个问题：不同输入长度的 TTFT 如何变化、每个长度的请求从 0 秒起经过哪些执行轨道、以及关键路径上哪些 kernel 占据最多时间。TTFT 的起止点以测量口径中的定义为准；泳道只表达观测到的事件顺序，不把重叠事件串行相加。", "", "## 结论", "", acc, "", meta, "", "## Input length - TTFT", "", "| Input length | TTFT p50/median (ms) | Min (ms) | Max (ms) |", "|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['input_length']} | {fmt(r.get('ttft_ms'))} | {fmt(r.get('ttft_min_ms'))} | {fmt(r.get('ttft_max_ms'))} |")
    md += ["", "![TTFT 曲线](input_length_ttft_curve.png)", "", "## 泳道与关键路径 Top10", "", "泳道横轴从该 input length 的观测起点归零到 0 秒；事件保留输入数据的相对顺序。Top10 按输入数据中的 `time_ms` 从大到小展示，只有标记 `path=true` 的记录才可称为关键路径 kernel。", ""]
    for r in rows:
        length = r["input_length"]
        md += [f"### Input length {length}", "", critical_text(r), "", f"![Input {length} 泳道](input_{length}_lane.png)", "", "| # | Kernel alias | Full kernel symbol | Time (ms) | % | Critical path |", "|---:|---|---|---:|---:|:---:|"]
        for rank, k in enumerate(top10(r), 1):
            md.append(f"| {rank} | {kernel_label(k)} | {k.get('full_kernel_symbol', k.get('kernel', k.get('name', '缺失')))} | {fmt(k.get('time_ms'))} | {fmt(k.get('percent'))} | {'是' if k.get('path') is True else '否'} |")
    md += ["", "## 数据与限制", "", f"{acc}", meta, "泳道横轴从 request origin 归一化为 0 秒；若未提供 request origin，才退回该长度的最早观测事件。缺少 critical path 归因时不会把普通 kernel 排序冒充关键路径 Top10。"]
    md_path = root / "prefill_report_20260929.md"; md_path.write_text("\n".join(md), encoding="utf-8")
    html_path = root / "prefill_report_20260929.html"
    quality_html = "<h2>精度质量证据</h2><table><tr><th>Input length</th><th>检查</th><th>状态</th><th>证据</th></tr>" + "".join("<tr><td>" + html.escape(str(q.get("input_length", q.get("prompt_tokens", "全部")))) + "</td><td>" + html.escape(str(q.get("name", q.get("check", "")))) + "</td><td>" + html.escape(str(q.get("status", "缺失"))) + "</td><td>" + html.escape(str(q.get("evidence", q.get("detail", "")))) + "</td></tr>" for q in quality_rows(data.get("accuracy"))) + "</table>"
    per_length_html = "".join("<h2>Input length " + str(r["input_length"]) + "</h2><p>" + html.escape(critical_text(r)) + "</p><img src='input_" + str(r["input_length"]) + "_lane.png' style='max-width:100%'><h3>Critical path Top10</h3><table><tr><th>#</th><th>Kernel</th><th>Time ms</th><th>%</th></tr>" + "".join(f"<tr><td>{i}</td><td>{html.escape(kernel_label(k))}</td><td>{fmt(k.get('time_ms'))}</td><td>{fmt(k.get('percent'))}</td></tr>" for i, k in enumerate(top10(r), 1)) + "</table>" for r in rows)
    html_body = "<h1>Prefill 性能报告</h1><p>" + html.escape(acc) + "</p><p>" + html.escape(meta) + "</p>" + quality_html + "<h2>Input length - TTFT</h2><img src='input_length_ttft_curve.png' style='max-width:100%'><table><tr><th>Input length</th><th>TTFT ms</th></tr>" + "".join(f"<tr><td>{r['input_length']}</td><td>{fmt(r.get('ttft_ms'))}</td></tr>" for r in rows) + "</table>" + per_length_html + "</body></html>"
    html_path.write_text("<html><meta charset='utf-8'><style>body{font-family:Arial,'PingFang SC',sans-serif;max-width:1200px;margin:24px auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:5px 10px}</style><body>" + html_body, encoding="utf-8")

    pdf_path = pdfdir / "prefill_performance_20260929.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=PAGE); c.setTitle("Prefill Performance 2026-09-29")
    c.setFont(FONT, 18); c.drawString(34, H-38, "Prefill 性能报告")
    c.setFont(FONT, 10); c.drawString(42, H-75, "Input length - TTFT、观测泳道、关键路径 Top10")
    cover_y = para(c, acc, 42, H-102, W-84, size=10)
    cover_y = para(c, meta, 42, cover_y - 5, W-84, size=9)
    chart_y = max(135, min(165, cover_y - 15 - 245))
    c.drawImage(str(curve), 42, chart_y, width=500, height=245, preserveAspectRatio=True, anchor="sw")
    table(c, ["Length", "TTFT ms", "Min", "Max"], [[str(r["input_length"]), fmt(r.get("ttft_ms")), fmt(r.get("ttft_min_ms")), fmt(r.get("ttft_max_ms"))] for r in rows], 540, chart_y + 245, [55, 78, 68, 68], rh=19)
    footer(c, 1, metadata)
    c.showPage()
    for page, r in enumerate(rows, 2):
        c.setFont(FONT, 16); c.drawString(34, H-38, f"Input length {r['input_length']}：泳道与关键路径 Top10")
        c.drawImage(str(lane_paths[r["input_length"]]), 34, H/2-5, width=W-68, height=H/2-60, preserveAspectRatio=True, anchor="sw")
        top = top10(r)
        para(c, critical_text(r), 42, H/2-28, W-84, size=8.5, leading=12)
        table(c, ["#", "Kernel", "ms", "%", "Path"], [[str(i), kernel_label(k), fmt(k.get("time_ms")), fmt(k.get("percent")), "是" if k.get("path") is True else "否"] for i,k in enumerate(top,1)], 42, H/2-48, [28, 350, 80, 70, 55], rh=16, size=7)
        footer(c, page, metadata)
        c.showPage()
    final_page = len(rows) + 2
    c.setFont(FONT, 16); c.drawString(34, H-38, "精度结论与限制")
    final_y = para(c, acc, 42, H-78, W-84, size=11, leading=18)
    final_y = para(c, meta, 42, final_y - 5, W-84, size=10, leading=16)
    final_y = para(c, "精度结论仅依据已提供的逐长度对照、参考输出、容差和质量指标。缺少这些证据时，本报告不会宣称 prefill 精度已确认。", 42, final_y - 8, W-84, size=10, leading=16)
    qrows = quality_rows(data.get("accuracy"))
    if qrows:
        for offset in range(0, len(qrows), 7):
            chunk = qrows[offset:offset + 7]
            if offset:
                final_page += 1
                c.setFont(FONT, 15); c.drawString(34, H-38, "精度质量证据（续）")
                quality_y = H - 70
            else:
                quality_y = final_y - 10
            table(c, ["Input length", "Metric", "Max abs", "RMSE", "KL", "Token match", "Evidence"], [[str(q.get("input_length", q.get("prompt_tokens", "全部"))), str(q.get("metric", q.get("name", q.get("check", "")))), fmt(q.get("max_abs", q.get("max_abs_error"))), fmt(q.get("rmse")), fmt(q.get("kl")), fmt(q.get("token_match")), str(q.get("evidence", q.get("detail", q.get("status", "缺失"))))] for q in chunk], 34, quality_y, [70, 100, 65, 55, 55, 75, 360], rh=17, size=6.8)
            footer(c, final_page, metadata)
            if offset + 7 < len(qrows):
                c.showPage()
    else:
        para(c, "未提供按长度的质量指标表。", 42, final_y - 10, W-84, size=10, leading=16)
        footer(c, final_page, metadata)
    c.save()
    print(pdf_path)


if __name__ == "__main__":
    main()
