"""Reconstruct a per-rank CUDA-Graph critical path from explicit topology.

This module intentionally does not infer cross-rank communication.  Callers
provide one replay's profiler activities; graph records come from the
critical-path sitecustomize hook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _activity_field(activity: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    value = _first(activity, *names, default=None)
    if value is not None:
        return value
    args = activity.get("args")
    return _first(args, *names, default=default) if isinstance(args, Mapping) else default


def _node_id(value: Any) -> int:
    """Normalize CUPTI IDs; graph records and traces may use different widths."""
    return int(value) & 0xFFFFFFFF


def _graph_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    graph = record.get("graph", record)
    if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), list):
        raise ValueError("graph record must contain a nodes list")
    return graph


def _activity_index(activities: Iterable[Mapping[str, Any]], exec_graph_id: int) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for activity in activities:
        graph_id = _activity_field(activity, "graph id", "graph_id", "graphId", "exec_graph_id", "execGraphId")
        node_id = _activity_field(activity, "graph node id", "graph_node_id", "graphNodeId", "node_id", "nodeId")
        if graph_id is None or node_id is None or int(graph_id) != exec_graph_id:
            continue
        key = _node_id(node_id)
        if key in result:
            raise ValueError(f"duplicate activity for graph {exec_graph_id}, node {key}")
        result[key] = activity
    return result


def analyze_graph(
    record: Mapping[str, Any],
    activities: Iterable[Mapping[str, Any]],
    *,
    refuse_programmatic: bool = True,
) -> dict[str, Any]:
    """Return an auditable longest path for one captured executable graph.

    Activity timestamps and durations are retained in their input units.  A
    graph node with no matching activity is structural/untimed; its completion
    is inherited from its explicit predecessors.  ``programmatic`` or other
    nonzero edge annotations raise by default because ordinary reachability
    does not preserve their CUDA semantics.
    """
    graph = _graph_payload(record)
    exec_graph_id = int(_first(graph, "exec_graph_id", "execGraphId", default=-1))
    if exec_graph_id < 0:
        raise ValueError("graph record lacks exec_graph_id")
    raw_nodes = graph["nodes"]
    count = len(raw_nodes)
    nodes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"node {index} is not an object")
        deps = [int(dep) for dep in _first(raw, "dependencies", default=[])]
        if any(dep < 0 or dep >= count for dep in deps):
            raise ValueError(f"node {index} contains an invalid dependency")
        nodes.append({
            "index": index,
            "node_id": int(_first(raw, "node_id", "nodeId", default=-1)),
            "node_type": _first(raw, "node_type", "nodeType"),
            "dependencies": deps,
            "dependents": [int(dep) for dep in _first(raw, "dependents", default=[])],
        })
    normalized_ids = [_node_id(node["node_id"]) for node in nodes]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("graph contains duplicate normalized node IDs")

    edges = graph.get("edges")
    if edges is not None:
        for edge in edges:
            edge_type = int(_first(edge, "type", "edge_type", default=0))
            if refuse_programmatic and edge_type != 0:
                raise ValueError(f"graph contains non-ordinary edge type {edge_type}")
            source = int(_first(edge, "from", "source"))
            target = int(_first(edge, "to", "target", "destination"))
            if source < 0 or target < 0 or source >= count or target >= count:
                raise ValueError("graph edge references an invalid node")

    activities = list(activities)
    by_node_id = _activity_index(activities, exec_graph_id)
    graph_node_ids = {_node_id(node["node_id"]) for node in nodes}
    unmatched_activity_ids = sorted(
        {_node_id(_activity_field(activity, "graph_node_id", "graphNodeId", "node_id", "nodeId"))
         for activity in activities
         if _activity_field(activity, "graph_id", "graphId", "exec_graph_id", "execGraphId") is not None
         and int(_activity_field(activity, "graph_id", "graphId", "exec_graph_id", "execGraphId")) == exec_graph_id
         and _node_id(_activity_field(activity, "graph_node_id", "graphNodeId", "node_id", "nodeId")) not in graph_node_ids}
    )
    for node in nodes:
        activity = by_node_id.get(_node_id(node["node_id"]))
        node["activity"] = dict(activity) if activity is not None else None
        node["timed"] = activity is not None

    indegree = [len(node["dependencies"]) for node in nodes]
    dependents = [[] for _ in nodes]
    for node in nodes:
        for dep in node["dependencies"]:
            dependents[dep].append(node["index"])
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    order = []
    while ready:
        index = ready.pop()
        order.append(index)
        for dependent in dependents[index]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if len(order) != count:
        raise ValueError("graph dependencies contain a cycle")

    solved: list[dict[str, Any] | None] = [None] * count
    for index in order:
        node = nodes[index]
        predecessors = [solved[dep] for dep in node["dependencies"]]
        predecessors = [item for item in predecessors if item is not None]
        activity = node["activity"]
        if activity is None:
            completion = max((item["completion"] for item in predecessors), default=0)
            duration = 0
            start = completion
        else:
            start = _activity_field(activity, "start", "ts", "timestamp", "start_ns")
            duration = _activity_field(activity, "duration", "dur", "duration_ns", default=0)
            if start is None:
                raise ValueError(f"timed activity for node {index} lacks a start timestamp")
            start = float(start)
            duration = float(duration)
            completion = start + duration
        effective: set[int] = set()
        for item in predecessors:
            if item["timed"]:
                effective.add(item["index"])
            else:
                effective.update(item["effective_predecessors"])
        result = {"index": index, "timed": activity is not None, "start": start, "duration": duration, "completion": completion, "effective_predecessors": effective}
        solved[index] = result
    solved = [item for item in solved if item is not None]
    terminal = max(solved, key=lambda item: item["completion"], default={"index": None, "completion": 0})
    path: list[int] = []
    cursor = terminal.get("index")
    while cursor is not None:
        path.append(cursor)
        deps = nodes[cursor]["dependencies"]
        latest = max((solved[dep]["completion"] for dep in deps), default=None)
        predecessor = [dep for dep in deps if solved[dep]["completion"] == latest]
        cursor = predecessor[-1] if predecessor else None
    path.reverse()
    timed = sum(node["timed"] for node in nodes)
    return {
        "exec_graph_id": exec_graph_id,
        "node_count": count,
        "activity_count": len(by_node_id),
        "timed_node_count": timed,
        "untimed_node_indexes": [node["index"] for node in nodes if not node["timed"]],
        "untimed_node_types": sorted({str(node["node_type"]) for node in nodes if not node["timed"]}),
        "unmatched_activity_node_ids": unmatched_activity_ids,
        "coverage": timed / count if count else 1.0,
        "nodes": [{**node, "critical": item["index"] in path, "effective_predecessors": solved[item["index"]]["effective_predecessors"]} for node, item in zip(nodes, solved)],
        "critical_path": path,
        "critical_completion": terminal["completion"],
    }
