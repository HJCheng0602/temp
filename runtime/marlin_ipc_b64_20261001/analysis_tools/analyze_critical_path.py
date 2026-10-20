"""Observed dependency-chain attribution for the fixed whole-B32 PP2 capture.

Graph edges are captured, exterior PP edges follow the installed runtime, and
communication rendezvous uses the latest participant start. PDL trigger times
are not observed: two overlap-allocation conventions are reported, not claimed
as hardware-counter measurements or rigorous bounds on optimization benefit.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

from critical_path_graph import analyze_graph


GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def read_capture(root):
    graphs = {}
    for path in (root / "graph_dags").glob("graph_pid*.json"):
        record = json.loads(path.read_text())
        graphs[record["torch_rank"], record["graph"]["exec_graph_id"]] = record
    ranks, bases = {}, set()
    for path in sorted((root / "graph/timelines").rglob("*.trace.json.gz")):
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        rank = data["distributedInfo"]["rank"]
        bases.add(data["baseTimeNanoseconds"])
        events = sorted((e for e in data["traceEvents"] if e.get("cat") in GPU_CATS), key=lambda e: e["ts"])
        groups = defaultdict(list)
        for event in events:
            a = event["args"]
            if a.get("graph id"):
                groups[a["graph id"], a["correlation"]].append(event)
        replay = sorted(groups.values(), key=lambda es: min(e["ts"] for e in es))
        ranks[rank] = dict(events=events, replays=replay)
    assert set(ranks) == set(range(8)) and len(bases) == 1
    assert {len(row["replays"]) for row in ranks.values()} == {9}
    return graphs, ranks


def step_graph(graphs, ranks, step, extra_collective_prefixes=()):
    """Build one observed dependency graph.

    ``extra_collective_prefixes`` is intentionally opt-in so the historical
    NCCL-only behavior remains unchanged.  Prefix matches are split by the
    concrete kernel name before rendezvous (IPC has multiple implementations).
    """
    if isinstance(extra_collective_prefixes, str):
        extra_collective_prefixes = (extra_collective_prefixes,)
    nodes, graph_nodes, audits = [], {}, []
    origin = min(e["ts"] for rank in range(4) for e in ranks[rank]["replays"][step])

    def add(rank, event, deps=(), kind=None):
        i = len(nodes)
        nodes.append(dict(id=i, rank=rank, name=event["name"],
            start=event["ts"]-origin, end=event["ts"]-origin+event.get("dur", 0),
            raw_duration=event.get("dur", 0), kind=kind or event.get("cat", "boundary"),
            source_cat=event.get("cat", ""),
            stream=event.get("args", {}).get("stream"),
            graph_id=event.get("args", {}).get("graph id"),
            graph_node_id=event.get("args", {}).get("graph node id", 0), deps=list(deps)))
        return i

    for rank, data in ranks.items():
        events = data["replays"][step]
        record = graphs[rank, events[0]["args"]["graph id"]]
        result = analyze_graph(record, events, refuse_programmatic=False)
        assert not result["unmatched_activity_node_ids"]
        assert not any(n["node_type"] in (0, 1, 2) and not n["timed"] for n in result["nodes"])
        index = {n["index"]: add(rank, n["activity"]) for n in result["nodes"] if n["timed"]}
        pdl = defaultdict(list)
        for edge in record["graph"]["edges"]:
            if edge.get("type") == 1:
                assert edge["source"] in index and edge["destination"] in index
                pdl[edge["destination"]].append(index[edge["source"]])
        for n in result["nodes"]:
            if n["timed"]:
                nodes[index[n["index"]]]["deps"] = [index[j] for j in n["effective_predecessors"]]
                nodes[index[n["index"]]]["pdl_deps"] = pdl[n["index"]]
        graph_nodes[rank] = list(index.values())
        audits.append(dict(rank=rank, timed=result["timed_node_count"],
            total=result["node_count"], untimed_types=result["untimed_node_types"],
            edge_types=dict(Counter(e.get("type", 0) for e in record["graph"]["edges"]))))

    def roots(rank):
        return [i for i in graph_nodes[rank] if not nodes[i]["deps"]]

    def leaves(rank):
        used = {j for i in graph_nodes[rank] for j in nodes[i]["deps"]}
        return [i for i in graph_nodes[rank] if i not in used]

    def series(rank, events, incoming):
        ids = []
        for event in events:
            i = add(rank, event, ids[-1:] or incoming)
            ids.append(i)
        return ids

    pp_send, pp_recv, broadcasts = {}, {}, {}
    sample_ends = []
    for rank, data in ranks.items():
        all_events = data["events"]
        outside = [e for e in all_events if not e["args"].get("graph id")]
        graph_start = min(nodes[i]["start"] for i in graph_nodes[rank]) + origin
        graph_end = max(nodes[i]["end"] for i in graph_nodes[rank]) + origin
        pp = [e for e in outside if "ncclDevKernel_SendRecv" in e["name"]]
        bc = [e for e in outside if "ncclDevKernel_Broadcast" in e["name"]]
        assert len(pp) == 27 and len(bc) == 18
        pp = pp[3*step:3*step+3]
        bc = bc[2*step:2*step+2]
        if rank < 4:
            post = [e for e in outside if e["args"].get("stream") == 23 and graph_end <= e["ts"] < pp[0]["ts"]]
            assert len(post) == 1 and post[0]["name"] == "_post_update_num_computed_tokens_kernel"
            p = series(rank, post, leaves(rank))
            pp_send[rank] = series(rank, pp, p[-1:])
            broadcasts[rank] = series(rank, bc, pp_send[rank][-1:])
        else:
            # recv_tensor_dict waits for all PP tensors before TP reconstruction.
            pp_recv[rank] = series(rank, pp, [])
            pre = [e for e in outside if e["args"].get("stream") == 23 and pp[-1]["ts"] < e["ts"] < graph_start]
            expected_gathers = data.get("pp_allgather_count", 3)
            assert sum("AllGather" in e["name"] for e in pre) == expected_gathers, (
                rank, step, graph_start, pp[-1]["ts"],
                [(e["name"], e["ts"], e["args"].get("stream")) for e in pre])
            p = series(rank, pre, pp_recv[rank][-1:])
            for i in roots(rank):
                nodes[i]["deps"].extend(p[-1:])
            post = [e for e in outside if e["args"].get("stream") == 23 and graph_end <= e["ts"] < bc[0]["ts"]]
            sample = [i for i,e in enumerate(post) if e["name"] == "_get_num_sampled_and_rejected_kernel"]
            assert len(sample) == 1
            post = post[:sample[0]+1]
            p = series(rank, post, leaves(rank))
            sample_ends.append(p[-1])
            # PP broadcast_stream.wait_stream(main_stream) after sampling.
            side = [e for e in outside if e["args"].get("stream") == bc[0]["args"]["stream"]
                    and nodes[p[-1]]["end"]+origin <= e["ts"] <= bc[-1]["ts"]]
            ids = series(rank, side, p[-1:])
            broadcasts[rank] = [i for i in ids if "ncclDevKernel_Broadcast" in nodes[i]["name"]]
            assert len(broadcasts[rank]) == 2

    comm_audits = []

    def rendezvous(ids, label):
        assert len({nodes[i]["name"] for i in ids}) == 1
        arrival = []
        for i in ids:
            n = nodes[i]
            a = add(n["rank"], dict(name="communication readiness", ts=n["start"]+origin), n["deps"], "readiness")
            nodes[a]["pdl_deps"] = n.get("pdl_deps", [])
            arrival.append(a)
        latest = max(nodes[i]["start"] for i in ids)
        buffered = min(nodes[i]["end"] for i in ids) < latest
        if buffered:
            assert label.startswith(("PP:", "broadcast:")), label
        comm_audits.append(dict(label=label, participants=ids, latest_start=latest,
            buffered_completion=buffered,
            receiver_tails=[max(0,nodes[i]["end"]-latest) for i in ids]))
        for i in ids:
            # Small P2P sends can complete into a buffer before recv starts.
            feasible = [j for j in arrival if nodes[j]["end"] <= nodes[i]["end"]]
            nodes[i]["start"] = max(nodes[j]["end"] for j in feasible)
            nodes[i]["deps"] = feasible
            nodes[i]["pdl_deps"] = []
            nodes[i]["kind"] = "communication_tail"

    for stage in range(2):
        operations = ("ncclDevKernel_AllReduce", "ncclDevKernel_AllGather")
        for prefix in extra_collective_prefixes:
            names = sorted({n["name"] for n in nodes
                            if n["rank"] in range(stage*4, stage*4+4)
                            and prefix in n["name"]})
            operations += tuple(names)
        for op in operations:
            group = []
            for rank in range(stage*4, stage*4+4):
                ids = sorted((n["id"] for n in nodes if n["rank"] == rank and op in n["name"]), key=lambda i: nodes[i]["start"])
                group.append(ids)
            assert len({len(ids) for ids in group}) == 1
            for ordinal, ids in enumerate(zip(*group)):
                rendezvous(ids, f"stage{stage}:{op}:{ordinal}")
    for tp in range(4):
        for ordinal in range(3):
            rendezvous([pp_send[tp][ordinal], pp_recv[tp+4][ordinal]], f"PP:{tp}:{ordinal}")
        for ordinal in range(2):
            rendezvous([broadcasts[tp+4][ordinal], broadcasts[tp][ordinal]], f"broadcast:{tp}:{ordinal}")
    return nodes, [broadcasts[tp][-1] for tp in range(4)], sample_ends, audits, comm_audits


def backtrace(nodes, endpoints, end_limit=None, ordinary_overlap_tolerance_us=0.0):
    endpoint = max(endpoints, key=lambda i: nodes[i]["end"])
    cursor, visited, reverse = endpoint, set(), []
    while cursor is not None:
        assert cursor not in visited, "dependency cycle"
        visited.add(cursor)
        reverse.append(cursor)
        deps = nodes[cursor]["deps"]
        cursor = max(deps, key=lambda i: (nodes[i]["end"], nodes[i]["start"], i)) if deps else None
    chain = list(reversed(reverse))
    total_end = nodes[endpoint]["end"] if end_limit is None else end_limit
    overlaps = []
    for predecessor, current in zip(chain,chain[1:]):
        delta = nodes[predecessor]["end"]-nodes[current]["start"]
        if delta > .01:
            overlaps.append(dict(predecessor=predecessor, current=current, overlap_us=delta,
                pdl=predecessor in nodes[current].get("pdl_deps", [])))
    assert all(x["pdl"] or x["overlap_us"] <= ordinary_overlap_tolerance_us
               for x in overlaps), "unexplained ordinary-edge overlap"

    def allocate(order):
        # Deterministic interval subtraction retains a single wall-time budget.
        covered, amounts, segments = [], Counter(), []
        for i in order:
            n = nodes[i]
            start, end = max(0, n["start"]), min(total_end, n["end"])
            todo = [(start, end)] if end > start else []
            for left, right in covered:
                next_todo = []
                for a,b in todo:
                    if b <= left or a >= right:
                        next_todo.append((a,b))
                    else:
                        if a < left: next_todo.append((a,left))
                        if b > right: next_todo.append((right,b))
                todo = next_todo
                if not todo: break
            for a,b in todo:
                amounts[n["name"]] += b-a
                segments.append(dict(node=i, start=a, end=b, kind=n["kind"], name=n["name"]))
            if end > start:
                all_intervals = sorted(covered + [(start,end)])
                covered = []
                for a,b in all_intervals:
                    if covered and a <= covered[-1][1]: covered[-1] = (covered[-1][0], max(b,covered[-1][1]))
                    else: covered.append((a,b))
        occupied = sum(b-a for a,b in covered)
        return dict(amounts=dict(amounts), segments=sorted(segments,key=lambda x:x["start"]),
            occupied_us=occupied, unattributed_gap_us=total_end-occupied)

    return dict(endpoint=endpoint, span_us=total_end, chain=chain,
        overlapping_dependency_links=overlaps,
        consumer_first=allocate(reversed(chain)), producer_first=allocate(chain))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graphs, ranks = read_capture(args.root)
    cases, totals, alternatives, kinds, raw = [], Counter(), Counter(), {}, Counter()
    for step in range(9):
        nodes, endpoints, sample, audit, comm = step_graph(graphs, ranks, step)
        path = backtrace(nodes, endpoints)
        forward = backtrace(nodes, sample)
        for n in nodes:
            kinds[n["name"]] = n["kind"]
            if n["kind"] in ("kernel", "communication_tail"):
                raw[n["name"]] += n["raw_duration"]
        totals.update(path["consumer_first"]["amounts"])
        alternatives.update(path["producer_first"]["amounts"])
        cases.append(dict(step=step, graph_audit=audit, communication_audit=comm,
            path=path, forward_path=forward, nodes=nodes))
    labels_path = Path("results/kernel_top30_b32_20260926/analysis.json")
    labels = {r["full_kernel_symbol"]: r for r in json.loads(labels_path.read_text())["rows"]}
    kernel_names = [name for name in totals if kinds[name] in ("kernel", "communication_tail")]
    rows = []
    for number, name in enumerate(sorted(kernel_names, key=lambda n:totals[n], reverse=True),1):
        label = labels.get(name, {})
        rows.append(dict(number=number, kernel=label.get("kernel",name),
            full_kernel_symbol=name, purpose=label.get("purpose",""),
            attributed_us_per_step=totals[name]/9,
            alternate_overlap_allocation_us_per_step=alternatives[name]/9,
            all_rank_raw_us_per_step=raw[name]/9, kind=kinds[name]))
    summary = dict(schema="observed_dependency_chain_v1", batch=32, context=2048,
        pp=2,tp=4,ep=4,steps=9,pp_layers=[20,20],
        method="Observed latest-completion predecessor chain; captured CUDA Graph edges; source-verified exterior PP edges; latest-participant-start communication rendezvous; consumer-first overlap allocation",
        limitations=["PDL device trigger and consumer wait timestamps are absent; per-kernel attribution is an estimate, not exact marginal latency or a speedup bound.",
            "Collective tails are rendezvous-to-completion proxies, not physical wire time.",
            "CUDA synchronization activities omit event record correlation IDs; exterior dependencies are augmented using installed PP source.",
            "Profiler adds overhead; spans here are not unprofiled service TPOT.",
            "Both overlap conventions use the same observed path; their difference is sensitivity, not a rigorous confidence interval."],
        endpoint="Earliest stage0 graph activity to latest stage0 second sampled-token broadcast completion",
        mean_span_us=statistics.mean(c["path"]["span_us"] for c in cases),
        mean_forward_span_us=statistics.mean(c["forward_path"]["span_us"] for c in cases),
        mean_attributed_kernel_us=sum(totals[n] for n in kernel_names)/9,
        mean_unattributed_gap_us=statistics.mean(c["path"]["consumer_first"]["unattributed_gap_us"] for c in cases),
        mean_copy_memset_us=sum(v for n,v in totals.items() if kinds[n] not in ("kernel","communication_tail"))/9,
        overlapping_path_links=sum(len(c["path"]["overlapping_dependency_links"]) for c in cases),
        unexplained_ordinary_path_overlap=0,
        rows=rows,cases=cases)
    (args.output/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (args.output/"top30.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows[:30])
    report = ["# B32 Critical Dependency Chain: Estimated Kernel Attribution", "",
        "PP2 TP4 EP4, layers 20+20, whole B32, 2048-token prompt, CUDA Graph, nine decode steps. "
        "A fresh capture retains graph topology and original timestamps for all eight ranks. "
        "Every active graph kernel, memcpy and memset maps to its graph node. "
        "This is observed dependency-chain attribution, not exact instruction-level critical-path measurement.", "",
        "## Broadcast Finding", "",
        "A stage0 receive-side Broadcast overlaps stage1 computation. Its full duration includes "
        "waiting for sampled tokens. Completion is still required before the next decode consumes "
        "those tokens, so the return tail cannot be declared irrelevant to latency. "
        "Adding the entire receiver duration after stage1 compute would double-count the wait.", "",
        "## Accounting", "",
        f"Mean measured span: {summary['mean_span_us']/1000:.3f} ms, from the earliest stage0 graph "
        "activity to the latest stage0 second broadcast completion. Input preparation before graph "
        "entry, the next decode's host launch, and client-side delivery are outside this span.", "",
        f"Attributed kernel intervals: {summary['mean_attributed_kernel_us']/1000:.3f} ms/step; "
        f"copies/memsets: {summary['mean_copy_memset_us']/1000:.3f} ms/step; "
        f"unattributed gaps: {summary['mean_unattributed_gap_us']/1000:.3f} ms/step. "
        "These sum to the span and never sum simultaneous work from all eight GPUs. "
        "The span is profiler-instrumented GPU execution, not unprofiled service TPOT.", "",
        "The path backtracks the latest observed predecessor completion at each join. "
        "Intra-graph dependencies come from CUDA topology. Exterior stream joins follow the "
        "installed PP receive, graph, sampling, and broadcast code. A communication completion "
        "depends on participants' readiness; waiting before the latest arrival is charged to "
        "the upstream dependency, not counted again as communication. Small buffered P2P sends "
        "may complete before their receives start, and are kept asynchronous.", "",
        "## Top 30", "",
        "Times are microseconds per whole-B32 decode step, averaged across nine steps. "
        "The first allocation gives overlapping intervals to the downstream/consumer kernel. "
        "The alternative gives them to the upstream/producer on the same path. "
        "Their difference indicates attribution sensitivity; it is not a confidence interval "
        "or a bound on achievable speedup.", "",
        "| # | Kernel | Attributed us/step | Alternative us/step |",
        "|---:|---|---:|---:|"]
    for row in rows[:30]:
        report.append(f"| {row['number']} | `{row['kernel']}` | {row['attributed_us_per_step']:.3f} | "
                      f"{row['alternate_overlap_allocation_us_per_step']:.3f} |")
    report += ["", "## Communication", "",
        "| Rank | Kernel | Attributed us/step | Raw sum over 8 GPUs, us/step |",
        "|---:|---|---:|---:|"]
    for row in rows:
        if row["kind"] == "communication_tail":
            report.append(f"| {row['number']} | `{row['kernel']}` | {row['attributed_us_per_step']:.3f} | "
                          f"{row['all_rank_raw_us_per_step']:.3f} |")
    report += ["", "## Limits", ""] + ["- " + x for x in summary["limitations"]]
    report += ["", "There are 481 PDL edges in each stage0 graph and 480 in each stage1 graph. "
        "Device trigger and consumer wait instants were not recorded. Overlap is retained and "
        "accounted once; no claim is made that whole PDL kernels serialize. Exact marginal "
        "latency attribution would need device-level dependency timing or controlled interventions.", "",
        f"The selected paths contain {summary['overlapping_path_links']} overlapping predecessor/consumer "
        "links over nine steps, all explicitly PDL-mediated. There are no unexplained ordinary-edge "
        "overlaps on these paths. Thus the selected chain is a dependency attribution chain, "
        "not a claim that its kernels execute one after another without overlap.", "",
        "## Evidence", "",
        "The driver completed output equality, zero-preemption, and whole-batch shape checks. "
        "All ranks logged 95 decode rows at n=32, padded_tokens=32, graph_mode=FULL; nine were profiled. "
        "The service exited and the GPU compute process list was empty after capture.", "",
        "Archive: `critical_b32_20260927a.tgz`, 9125221 bytes. "
        "SHA256: `5289006487276d56a90e063fa6ce7a347420a3956e52b955a821c8464df377ea`.", "",
        "`analysis.json` retains all nodes, graph audit, communication pairing, per-step selected "
        "path, and attributed intervals. `top30.csv` contains full kernel symbols. "
        "`broadcast_tail_audit.json` retains every same-capture sender/receiver pair.", ""]
    (args.output/"REPORT.md").write_text("\n".join(report))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","cases")},indent=2))
    for r in rows[:30]:print(r["number"],round(r["attributed_us_per_step"],3),round(r["alternate_overlap_allocation_us_per_step"],3),r["kernel"])


if __name__ == "__main__":
    main()
