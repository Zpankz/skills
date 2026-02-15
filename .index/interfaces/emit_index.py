#!/usr/bin/env python3
"""Emit deterministic skill index artifacts from the structured skill repository."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Sequence

BASE = Path(__file__).resolve().parents[2]


KNOWN_CLUSTER_PREFIXES = ("00_", "10_", "20_", "30_")
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "will",
    "when",
    "where",
    "into",
    "from",
    "about",
    "andor",
    "use",
    "uses",
    "used",
    "you",
    "your",
    "using",
    "then",
    "have",
    "has",
    "have",
    "had",
    "their",
    "they",
    "them",
    "that",
    "also",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_yaml_str(value: str) -> str:
    if value == "":
        return "\"\""
    return json.dumps(value)


def _yaml_dump(data: dict) -> str:
    lines: list[str] = []

    def _emit_scalar(key: str, value: object, indent: int = 0) -> None:
        if isinstance(value, list):
            lines.append(f"{'  ' * indent}{key}:")
            for item in value:
                if isinstance(item, dict):
                    _emit_dict(item, indent + 1)
                else:
                    lines.append(f"{'  ' * (indent + 1)}- {_safe_yaml_str(str(item))}")
            return

        if isinstance(value, dict):
            lines.append(f"{'  ' * indent}{key}:")
            _emit_mapping(value, indent + 1)
            return

        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif value is None:
            value_str = "null"
        elif isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = _safe_yaml_str(str(value))

        lines.append(f"{'  ' * indent}{key}: {value_str}")

    def _emit_mapping(mapping: dict, indent: int = 0) -> None:
        for key in sorted(mapping.keys(), key=lambda x: str(x)):
            _emit_scalar(key, mapping[key], indent=indent)

    def _emit_dict(value: dict, indent: int) -> None:
        _emit_mapping(value, indent)

    _emit_mapping(data, 0)
    return "\n".join(lines) + "\n"


def _tokenize(text: str) -> list[str]:
    parts: list[str] = []
    token: list[str] = []
    for char in text.lower():
        if char.isalnum():
            token.append(char)
        else:
            if token:
                parts.append("".join(token))
                token = []
    if token:
        parts.append("".join(token))
    return [p for p in parts if p and p not in STOP_WORDS and len(p) > 2]


def _extract_script_signatures(skill_root: Path) -> list[str]:
    signatures: set[str] = set()
    for file in sorted(skill_root.glob("**/*.py")):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                signatures.add(node.name)
    return sorted(signatures)


def _load_skill_records(base: Path) -> tuple[dict, dict, dict, list[str]]:
    repo_dir = base / "repo"
    records: dict[str, dict] = {}
    clusters: dict[str, list[str]] = {}
    cluster_skill_paths: dict[str, Path] = {}
    unresolved: list[str] = []

    for cluster_dir in sorted(repo_dir.iterdir(), key=lambda p: p.name):
        if not cluster_dir.is_dir():
            continue
        if not cluster_dir.name.startswith(KNOWN_CLUSTER_PREFIXES):
            continue
        clusters[cluster_dir.name] = []
        for skill_archive in sorted(cluster_dir.glob("*.skill")):
            skill_id = skill_archive.stem
            manifest_path = cluster_dir / skill_id / "manifest.json"
            bridge_path = cluster_dir / skill_id / "bridge.json"
            if not manifest_path.is_file():
                unresolved.append(f"{skill_id}: missing manifest")
                continue
            if not bridge_path.is_file():
                unresolved.append(f"{skill_id}: missing bridge")
            manifest = _read_json(manifest_path)

            cluster_skill_paths[skill_id] = skill_archive
            clusters[cluster_dir.name].append(skill_id)
            records[skill_id] = {
                "id": manifest.get("id", skill_id),
                "name": manifest.get("name", skill_id),
                "cluster": cluster_dir.name,
                "archive": f"{skill_archive.relative_to(base)}",
                "manifest": f"{manifest_path.relative_to(base)}",
                "bridge": f"{bridge_path.relative_to(base)}" if bridge_path.is_file() else None,
                "triggers": list(manifest.get("triggers", [])),
                "capabilities": list(manifest.get("capabilities", [])),
                "hard_deps": list(manifest.get("hard_deps", [])),
                "soft_refs": list(manifest.get("soft_refs", [])),
                "outputs": list(manifest.get("outputs", [])),
                "inputs": list(manifest.get("inputs", [])),
                "subgraph_type": manifest.get("subgraph_type", "modular_block"),
                "quality": manifest.get("quality", {}),
                "skill_root": f"{(cluster_dir / skill_id)}",
            }

    root = base
    aliases: dict[str, list[str]] = {skill_id: [] for skill_id in records}

    for symlink in sorted(root.glob("*.skill")):
        if not symlink.is_symlink():
            continue
        try:
            target = symlink.resolve()
        except FileNotFoundError:
            unresolved.append(f"{symlink.name}: broken symlink")
            continue
        if not str(target).startswith(str((base / "repo").resolve())):
            continue
        canonical = target.stem
        alias = symlink.stem
        if canonical not in aliases:
            continue
        if alias != canonical:
            aliases[canonical].append(alias)

    for skill_id, alias_list in aliases.items():
        alias_list.sort()

    return records, clusters, aliases, unresolved


def _normalise_dep(dep: str, alias_to_canonical: dict[str, str], canonical_nodes: set[str]) -> str | None:
    dep = dep.strip()
    if not dep:
        return None
    if dep.endswith(".skill"):
        dep = dep[:-6]
    dep = dep.lower().replace("_", "-")
    if dep in canonical_nodes:
        return dep
    return alias_to_canonical.get(dep)


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        v = value.strip()
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _build_raw_edges(
    records: dict[str, dict],
    alias_to_canonical: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    if alias_to_canonical is None:
        alias_to_canonical = {}
    canonical_nodes = set(records.keys())
    raw_hard: list[dict] = []
    raw_soft: list[dict] = []
    unresolved_refs: list[str] = []

    for src, entry in records.items():
        for target in _dedupe(entry.get("hard_deps", [])):
            resolved = _normalise_dep(target, alias_to_canonical, canonical_nodes)
            if resolved is None:
                unresolved_refs.append(f"{src} -> {target} (hard)")
                continue
            if resolved != src:
                raw_hard.append({"from": src, "to": resolved, "type": "hard_dep"})

        for target in _dedupe(entry.get("soft_refs", [])):
            resolved = _normalise_dep(target, alias_to_canonical, canonical_nodes)
            if resolved is None:
                unresolved_refs.append(f"{src} -> {target} (soft)")
                continue
            if resolved != src:
                raw_soft.append({"from": src, "to": resolved, "type": "soft_ref"})

    return raw_hard, raw_soft, unresolved_refs


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for edge in edges:
        key = (edge["from"], edge["to"], edge["type"])
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _tarjan_scc(nodes: list[str], hard_adj: dict[str, list[str]]) -> tuple[list[list[str]], dict[str, int]]:
    index = 0
    stack: list[str] = []
    onstack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    comp_id: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in hard_adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component: list[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp_id[w] = len(components)
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for v in nodes:
        if v not in indices:
            strongconnect(v)

    return components, comp_id


def _scc_metadata(components: list[list[str]], total_nodes: int) -> dict:
    nontrivial = [component for component in components if len(component) > 1]
    nontrivial_node_count = sum(len(c) for c in nontrivial)
    ratio = 1.0
    if total_nodes:
        ratio = 1.0 - (nontrivial_node_count / total_nodes)
    return {
        "nontrivial_components": len(nontrivial),
        "nontrivial_node_count": nontrivial_node_count,
        "nontrivial_members": sorted([node for component in nontrivial for node in component]),
        "acyclic_ratio": round(ratio, 4),
        "nodes": total_nodes,
    }


def _build_condensed_graph(nodes: list[str], hard_adj: dict[str, list[str]], comp_id: dict[str, int]) -> dict:
    comp_count = max(comp_id.values()) + 1 if comp_id else 0
    comp_members: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        comp_members[comp_id[node]].append(node)
    for comp_nodes in comp_members.values():
        comp_nodes.sort()

    comp_label_map = {
        comp: f"C{comp}" for comp in sorted(comp_members.keys())
    }
    edges: set[tuple[str, str]] = set()
    for node in nodes:
        source = comp_label_map[comp_id[node]]
        for target_node in hard_adj.get(node, []):
            target = comp_label_map[comp_id[target_node]]
            if source != target:
                edges.add((source, target))

    condensed_nodes = {
        comp_label_map[c]: {
            "size": len(comp_members[c]),
            "members": comp_members[c],
        }
        for c in sorted(comp_label_map.keys())
    }

    condensed_edges = [
        {"from": src, "to": dst}
        for src, dst in sorted(edges, key=lambda edge: (edge[0], edge[1]))
    ]
    return {
        "condensed_node_count": comp_count,
        "condensed_edge_count": len(condensed_edges),
        "node_map": condensed_nodes,
        "edges": condensed_edges,
    }


def _is_dag(node_count: int, edges: Sequence[dict]) -> bool:
    if node_count == 0:
        return True
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for edge in edges:
        u = edge["from"]
        v = edge["to"]
        nodes.add(u)
        nodes.add(v)
        if v not in adjacency[u]:
            adjacency[u].add(v)
            indegree[v] += 1
    for node in nodes:
        indegree.setdefault(node, 0)
        adjacency.setdefault(node, set())

    queue: deque[str] = deque([node for node in nodes if indegree[node] == 0])
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for neigh in sorted(adjacency[node]):
            indegree[neigh] -= 1
            if indegree[neigh] == 0:
                queue.append(neigh)
    return seen == len(nodes)


def _topological_order(nodes: list[str], edges: Sequence[dict]) -> list[str]:
    nodes_set = set(nodes)
    adjacency: dict[str, set[str]] = {node: set() for node in nodes_set}
    indegree: dict[str, int] = {node: 0 for node in nodes_set}
    for edge in sorted(edges, key=lambda item: (item["from"], item["to"])):
        src = edge["from"]
        dst = edge["to"]
        if src not in nodes_set or dst not in nodes_set:
            continue
        if dst not in adjacency[src]:
            adjacency[src].add(dst)
            indegree[dst] += 1

    queue: deque[str] = deque(sorted([node for node in nodes_set if indegree[node] == 0]))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in sorted(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    for node in nodes_set:
        if node not in order:
            order.append(node)
    return order


def _longest_path_in_dag(nodes: list[str], edges: Sequence[dict]) -> dict:
    topo = _topological_order(nodes, edges)
    index = {node: idx for idx, node in enumerate(topo)}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["from"]].append(edge["to"])

    dp: dict[str, int] = {node: 0 for node in nodes}
    for node in topo:
        node_depth = dp[node]
        for target in sorted(set(adjacency.get(node, []))):
            if dp[target] < node_depth + 1:
                dp[target] = node_depth + 1
    return {
        "topological_order": topo,
        "longest_path": dp,
        "max_depth": max(dp.values()) if dp else 0,
    }


def _control_scores(
    nodes: list[str],
    edges: Sequence[dict],
    damping: float = 0.85,
    iterations: int = 18,
) -> dict[str, float]:
    if not nodes:
        return {}
    score: dict[str, float] = {node: 1.0 for node in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["from"]].append(edge["to"])

    for _ in range(max(1, iterations)):
        next_score = {node: 1.0 - damping for node in nodes}
        for source in sorted(nodes):
            targets = sorted(set(outgoing.get(source, [])))
            if not targets:
                continue
            share = score[source] * damping / len(targets)
            for target in targets:
                next_score[target] += share
        score = next_score
    return {node: round(value, 6) for node, value in sorted(score.items())}


def _demote_cycle_edges(
    nodes: list[str],
    raw_hard: list[dict],
    comp_id: dict[str, int],
) -> tuple[list[dict], list[dict], list[dict]]:
    by_comp: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        by_comp[comp_id[node]].append(node)
    for _, members in by_comp.items():
        members.sort()
    positions: dict[str, int] = {}
    for members in by_comp.values():
        for idx, member in enumerate(members):
            positions[member] = idx

    kept: list[dict] = []
    demoted: list[dict] = []
    for edge in raw_hard:
        src = edge["from"]
        dst = edge["to"]
        if comp_id[src] != comp_id[dst]:
            kept.append(edge)
            continue
        if src == dst:
            continue
        if positions.get(src, 0) < positions.get(dst, 0):
            kept.append({"from": src, "to": dst, "type": "hard_dep"})
        else:
            demoted.append({"from": src, "to": dst, "type": "soft_ref"})

    return (
        _dedupe_edges(kept),
        _dedupe_edges(demoted),
        [{"from": edge["from"], "to": edge["to"], "reason": "scc_dag_order"} for edge in demoted],
    )


def _transitive_reduce_hard_edges(
    nodes: list[str],
    raw_hard_dag: list[dict],
) -> tuple[list[dict], list[dict]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in raw_hard_dag:
        if edge["from"] == edge["to"]:
            continue
        adjacency[edge["from"]].add(edge["to"])

    def _has_path_excluding_direct(u: str, v: str, skip_from: str, skip_to: str) -> bool:
        if u == v:
            return True
        seen: set[str] = {u}
        stack = [u]
        while stack:
            current = stack.pop()
            for nxt in adjacency[current]:
                if current == skip_from and nxt == skip_to:
                    continue
                if nxt == v:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    kept: list[dict] = []
    transitive_redundant: list[dict] = []
    for edge in sorted(raw_hard_dag, key=lambda item: (item["from"], item["to"])):
        src = edge["from"]
        dst = edge["to"]
        if _has_path_excluding_direct(src, dst, src, dst):
            transitive_redundant.append(
                {
                    "from": src,
                    "to": dst,
                    "type": "soft_ref",
                    "reason": "transitive_reduction",
                }
            )
        else:
            kept.append({"from": src, "to": dst, "type": "hard_dep"})

    return _dedupe_edges(kept), _dedupe_edges(transitive_redundant)


def _build_reverse_calls(edges: list[dict], nodes: list[str]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        reverse[edge["to"]].append(edge["from"])
    for node in reverse:
        reverse[node] = sorted(set(reverse[node]))
    return reverse


def _build_concepts(records: dict[str, dict], repo_root: Path) -> dict[str, set[str]]:
    concepts: dict[str, set[str]] = {}
    for skill_id, entry in records.items():
        tokens: set[str] = set()
        tokens.update(_tokenize(" ".join(entry.get("triggers", []))))
        tokens.update(_tokenize(" ".join(entry.get("capabilities", []))))
        tokens.update(_tokenize(" ".join(entry.get("outputs", []))))
        tokens.update(_tokenize(" ".join(entry.get("inputs", []))))
        root = repo_root / entry["skill_root"]
        tokens.update(_tokenize(" ".join(entry.get("skill_root", ""))))
        tokens.update(_tokenize(" ".join(_extract_script_signatures(root))))
        concepts[skill_id] = tokens
    return concepts


def _bridge_candidates(records: dict[str, dict], concepts: dict[str, set[str]], threshold: float = 0.14) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    ids = sorted(records.keys())
    for src, dst in combinations(ids, 2):
        set_a = concepts.get(src, set())
        set_b = concepts.get(dst, set())
        if not set_a or not set_b:
            continue
        union = len(set_a | set_b)
        if not union:
            continue
        score = len(set_a & set_b) / union
        if score >= threshold:
            rows.append((src, dst, round(score, 4)))
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    return rows


def _format_quality_markdown(
    timestamp: str,
    node_count: int,
    hard_edge_count: int,
    soft_edge_count: int,
    raw_scc: dict,
    scc_dag_edges: dict,
    demotion_count: int,
    transitive_reduction_count: int,
    hard_dag_ratio: float,
    atomic_ratio: float,
    mctsr_score: float | None = None,
) -> str:
    acyclic_raw = raw_scc["acyclic_ratio"]
    lines = [
        "# Quality Report",
        "",
        "## Graph health",
        f"- Generated: {timestamp}",
        f"- Node count: {node_count}",
        f"- Hard edges (raw): {hard_edge_count}",
        f"- Soft edges (raw): {soft_edge_count}",
        f"- Demotion edges (cycle + transitive): {demotion_count}",
        f"- Transitive-reduced hard edges: {transitive_reduction_count}",
        f"- Raw acyclic ratio (hard edges): {acyclic_raw:.4f}",
        f"- DAGified hard-edge acyclic ratio: {hard_dag_ratio:.4f}",
        "",
        "## Cycle decomposition",
        f"- Nontrivial SCC count: {raw_scc['nontrivial_components']}",
        f"- Nodes in nontrivial SCCs: {raw_scc['nontrivial_node_count']}",
        f"- Condensed node count: {scc_dag_edges['condensed_node_count']}",
        f"- Condensed edge count: {scc_dag_edges['condensed_edge_count']}",
        f"- Condensation DAG (hard deps): {'yes' if _is_dag(scc_dag_edges['condensed_node_count'], scc_dag_edges['edges']) else 'no'}",
        "",
    ]
    lines.extend([
        f"- Atomicity ratio: {atomic_ratio:.4f}",
        "",
    ])
    if mctsr_score is not None:
        lines.extend([
            "## Control-Traceability score",
            f"- MCTSR score: {mctsr_score:.4f}",
            "",
        ])
    return "\n".join(lines) + "\n"


def _format_processing_report(
    timestamp: str,
    node_count: int,
    canonical_count: int,
    unresolved: list[str],
    demotion_count: int,
    transitive_reduction_count: int,
    candidates: int,
    atomic_ratio: float,
    max_depth: int,
    mctsr_score: float | None = None,
) -> str:
    status = "resolved"
    if unresolved:
        status = "unresolved_refs"
    lines = [
        "# Skill Refactor Index",
        "",
        f"Generated: {timestamp}",
        f"- Canonical skills: {canonical_count}",
        f"- Total nodes: {node_count}",
        f"- Demotion edges for DAG/trace compatibility: {demotion_count}",
        f"- Transitive reduction edges reclassified as soft refs: {transitive_reduction_count}",
        f"- Bridge candidates: {candidates}",
        f"- Atomic node ratio: {atomic_ratio:.4f}",
        f"- Max DAG depth: {max_depth}",
        f"- MCTSR score: {mctsr_score:.4f}" if mctsr_score is not None else "- MCTSR score: N/A",
        f"- Validation status: {status}",
        "",
    ]
    if unresolved:
        lines.append("## Unresolved references")
        for entry in sorted(unresolved):
            lines.append(f"- {entry}")
    return "\n".join(lines) + "\n"


def _build_cluster_profile(clusters: dict[str, list[str]], records: dict[str, dict]) -> dict:
    profile: dict[str, dict] = {}
    for cluster_name in sorted(clusters.keys()):
        members = sorted(clusters.get(cluster_name, []))
        profile[cluster_name] = {
            "size": len(members),
            "members": members,
            "canonical_paths": [records[member]["archive"] for member in members if member in records],
        }
    return profile


def _build_control_ontology(
    records: dict[str, dict],
    clusters: dict[str, list[str]],
    hard_edges_dag: list[dict],
    soft_edges: list[dict],
    alias_to_canonical: dict[str, str],
    topology: list[str],
    comp_id: dict[str, int],
    comp_sizes: dict[int, int],
    control_scores: dict[str, float],
    max_depth: int,
    demotion_plan: list[dict],
) -> dict:
    nodes = sorted(records.keys())
    clusters_profile = _build_cluster_profile(clusters, records)
    hard_out: dict[str, int] = {node: 0 for node in nodes}
    hard_in: dict[str, int] = {node: 0 for node in nodes}
    soft_in: dict[str, int] = {node: 0 for node in nodes}
    soft_out: dict[str, int] = {node: 0 for node in nodes}

    for edge in hard_edges_dag:
        hard_out[edge["from"]] += 1
        hard_in[edge["to"]] += 1
    for edge in soft_edges:
        soft_out[edge["from"]] += 1
        soft_in[edge["to"]] += 1

    node_map: list[dict] = []
    for node in nodes:
        entry = records[node]
        if hard_in[node] == 0 and hard_out[node] == 0:
            control_state = "atomic_leaf"
        elif hard_in[node] == 0:
            control_state = "source"
        elif hard_out[node] == 0:
            control_state = "sink"
        else:
            control_state = "relay"
        node_map.append(
            {
                "id": node,
                "cluster": entry["cluster"],
                "cluster_ordinal": int(entry["cluster"].split("_")[0]),
                "is_atomic": comp_sizes.get(comp_id[node], 0) == 1,
                "scc_id": comp_id[node],
                "control_score": control_scores.get(node, 0.0),
                "control_state": control_state,
                "archive": entry["archive"],
                "quality": {
                    "word_count": entry["quality"].get("word_count", 0),
                    "hard_in_degree": hard_in[node],
                    "hard_out_degree": hard_out[node],
                    "soft_in_degree": soft_in[node],
                    "soft_out_degree": soft_out[node],
                },
            }
        )

    relations: list[dict] = []
    relation_id = 1
    for idx, edge in enumerate(sorted(hard_edges_dag, key=lambda item: (item["from"], item["to"]))):
        relations.append(
            {
                "id": f"R{idx+1:03d}",
                "kind": "runtime_hard",
                "source": edge["from"],
                "target": edge["to"],
                "policy": "required",
                "weight": 1.0,
                "control_plane": True,
                "meta": {
                    "source_control_score": control_scores.get(edge["from"], 0.0),
                    "target_control_score": control_scores.get(edge["to"], 0.0),
                },
            }
        )
        relation_id += 1
    for edge in sorted(soft_edges, key=lambda item: (item["from"], item["to"])):
        relations.append(
            {
                "id": f"R{relation_id:03d}",
                "kind": "runtime_soft",
                "source": edge["from"],
                "target": edge["to"],
                "policy": "advisory",
                "control_plane": False,
                "weight": 0.65,
            }
        )
        relation_id += 1
    for edge in demotion_plan:
        relations.append(
            {
                "id": f"R{relation_id:03d}",
                "kind": "demotion",
                "source": edge["from"],
                "target": edge["to"],
                "reason": edge.get("reason", "scc_order"),
                "policy": "compatibility",
                "control_plane": True,
                "weight": 0.45,
            }
        )
        relation_id += 1
    for alias, canonical in sorted(alias_to_canonical.items()):
        relations.append(
            {
                "id": f"R{relation_id:03d}",
                "kind": "compat_alias",
                "source": canonical,
                "target": canonical,
                "alias": alias,
                "policy": "resolution",
                "control_plane": False,
                "weight": 0.2,
            }
        )
        relation_id += 1

    atomic_nodes = [item["id"] for item in node_map if item["is_atomic"]]
    atomic_count = len(atomic_nodes)
    relation_kind_counts: dict[str, int] = defaultdict(int)
    for relation in relations:
        relation_kind_counts[relation["kind"]] += 1
    return {
        "generated_at": _now_iso(),
        "topology": topology,
        "control_graph": {
            "max_depth": max_depth,
            "node_count": len(nodes),
            "relation_count": len(relations),
            "atomic_count": atomic_count,
            "atomic_nodes": atomic_nodes,
            "relation_kind_counts": dict(sorted(relation_kind_counts.items())),
        },
        "clusters": clusters_profile,
        "nodes": node_map,
        "relations": relations,
    }


def _build_control_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Skill Control Ontology",
        "type": "object",
        "required": ["generated_at", "topology", "control_graph", "clusters", "nodes", "relations"],
        "properties": {
            "generated_at": {
                "type": "string",
                "format": "date-time",
            },
            "topology": {
                "type": "array",
                "items": {"type": "string"},
            },
            "control_graph": {
                "type": "object",
                "required": ["max_depth", "node_count", "relation_count", "atomic_count", "atomic_nodes", "relation_kind_counts"],
                "properties": {
                    "max_depth": {"type": "integer", "minimum": 0},
                    "node_count": {"type": "integer", "minimum": 0},
                    "relation_count": {"type": "integer", "minimum": 0},
                    "atomic_count": {"type": "integer", "minimum": 0},
                    "atomic_nodes": {"type": "array", "items": {"type": "string"}},
                    "relation_kind_counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                },
            },
            "clusters": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["size", "members", "canonical_paths"],
                    "properties": {
                        "size": {"type": "integer", "minimum": 0},
                        "members": {"type": "array", "items": {"type": "string"}},
                        "canonical_paths": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "id",
                        "cluster",
                        "cluster_ordinal",
                        "is_atomic",
                        "scc_id",
                        "control_score",
                        "control_state",
                        "archive",
                        "quality",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "cluster": {"type": "string"},
                        "cluster_ordinal": {"type": "integer"},
                        "is_atomic": {"type": "boolean"},
                        "scc_id": {"type": "integer", "minimum": 0},
                        "control_score": {"type": "number"},
                        "control_state": {"type": "string"},
                        "archive": {"type": "string"},
                        "quality": {
                            "type": "object",
                            "properties": {
                                "word_count": {"type": "integer", "minimum": 0},
                                "hard_in_degree": {"type": "integer", "minimum": 0},
                                "hard_out_degree": {"type": "integer", "minimum": 0},
                                "soft_in_degree": {"type": "integer", "minimum": 0},
                                "soft_out_degree": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "kind", "source", "target", "policy", "control_plane", "weight"],
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["runtime_hard", "runtime_soft", "demotion", "compat_alias"],
                        },
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "policy": {"type": "string"},
                        "control_plane": {"type": "boolean"},
                        "weight": {"type": "number"},
                        "reason": {"type": "string"},
                        "meta": {"type": "object"},
                    },
                },
            },
        },
    }


def _compute_mctsr(
    canon_nodes: list[str],
    hard_edges: list[dict],
    soft_edges: list[dict],
    hard_edges_dag: list[dict],
    soft_edges_aug: list[dict],
    unresolved_count: int,
    hard_dag_ratio: float,
    alias_to_canonical: dict[str, str],
    control_graph_node_count: int,
    control_graph_relation_count: int,
    schema_present: bool,
) -> dict[str, float | int | str | dict]:
    total_declared = len(hard_edges) + len(soft_edges)
    covered = len(hard_edges_dag) + len(soft_edges_aug)
    edge_coverage = 0.0 if total_declared == 0 else min(1.0, covered / total_declared)

    alias_health = 1.0
    for alias, canonical in alias_to_canonical.items():
        if canonical not in canon_nodes or alias in canon_nodes or not alias:
            alias_health = 0.0
            break

    declared_alias_count = len(alias_to_canonical)
    schema_health = 1.0 if schema_present else 0.0
    unresolved_health = 1.0 if unresolved_count == 0 else max(0.0, 1.0 - (unresolved_count / max(1, total_declared)))
    dag_health = 1.0 if hard_dag_ratio == 1.0 else hard_dag_ratio
    control_node_health = 1.0 if control_graph_node_count == len(canon_nodes) else min(1.0, control_graph_node_count / max(1, len(canon_nodes)))
    control_relation_health = 1.0 if control_graph_relation_count >= 1 else 0.0
    if denom := len(canon_nodes):
        alias_density = declared_alias_count / denom
    else:
        alias_density = 0.0

    raw_components = {
        "edge_coverage": round(edge_coverage, 6),
        "unresolved_health": round(unresolved_health, 6),
        "dag_health": round(dag_health, 6),
        "alias_health": round(alias_health, 6),
        "alias_density": round(alias_density, 6),
        "schema_health": round(schema_health, 6),
        "control_node_health": round(control_node_health, 6),
        "control_relation_health": round(control_relation_health, 6),
    }

    score = 0.30 * raw_components["edge_coverage"]
    score += 0.25 * raw_components["dag_health"]
    score += 0.15 * raw_components["unresolved_health"]
    score += 0.10 * raw_components["alias_health"]
    score += 0.10 * raw_components["schema_health"]
    score += 0.10 * raw_components["control_node_health"]
    score += 0.10 * raw_components["control_relation_health"]
    return {
        "score": round(min(100.0, score * 100.0), 4),
        "components": raw_components,
        "target": 95.0,
        "passed": score * 100.0 >= 95.0,
    }


def _emit_main_index_yaml(
    records: dict[str, dict],
    clusters: dict[str, list[str]],
    aliases: dict[str, list[str]],
    edge_payload: dict,
) -> str:
    nodes = sorted(records.keys())
    canonical_section = {}
    for node in nodes:
        entry = records[node]
        canonical_section[node] = {
            "cluster": entry["cluster"],
            "path": entry["archive"],
            "aliases": aliases.get(node, []),
            "hard_deps": [edge["to"] for edge in edge_payload["hard_edges_dag"] if edge["from"] == node],
            "soft_refs": [
                edge["to"]
                for edge in edge_payload["soft_edges_augmented"]
                if edge["from"] == node
                and edge["to"] not in {edge["to"] for edge in edge_payload["hard_edges_dag"] if edge["from"] == node}
            ],
            "outputs": entry.get("outputs", []),
            "quality": {
                "word_count": entry["quality"].get("word_count", 0),
                "hard_degree": len([edge for edge in edge_payload["hard_edges_dag"] if edge["from"] == node]),
                "soft_degree": len([edge for edge in edge_payload["soft_edges_augmented"] if edge["from"] == node]),
            },
        }
    payload = {
        "version": 1,
        "generated_at": edge_payload["generated_at"],
        "canonical": canonical_section,
        "aliases": {
            "canonical_to_aliases": aliases,
            "alias_to_canonical": {
                alias: canonical
                for canonical, names in aliases.items()
                for alias in names
            },
        },
    }
    return _yaml_dump(payload)


def dump_all(base: Path) -> dict:
    records, clusters, aliases, unresolved = _load_skill_records(base)
    canonical_nodes = sorted(records.keys())
    unresolved_refs: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    for canonical, names in aliases.items():
        for alias in names:
            alias_to_canonical[alias] = canonical
    hard_edges, soft_edges, unresolved_refs = _build_raw_edges(records, alias_to_canonical)
    unresolved.extend(unresolved_refs)

    hard_adj: dict[str, list[str]] = {node: [] for node in canonical_nodes}
    for edge in hard_edges:
        hard_adj[edge["from"]].append(edge["to"])

    components, comp_id = _tarjan_scc(canonical_nodes, hard_adj)
    raw_scc = _scc_metadata(components, len(canonical_nodes))
    condensed = _build_condensed_graph(canonical_nodes, hard_adj, comp_id)
    comp_sizes: dict[int, int] = {comp_id[node]: 0 for node in canonical_nodes}
    for node, cid in comp_id.items():
        comp_sizes[cid] += 1

    hard_edges_dag, demoted_soft, demotion_plan = _demote_cycle_edges(canonical_nodes, hard_edges, comp_id)
    hard_edges_dag, transitive_demotion = _transitive_reduce_hard_edges(canonical_nodes, hard_edges_dag)
    transitive_demotion_plan = [
        {"from": edge["from"], "to": edge["to"], "reason": edge.get("reason", "transitive_reduction")}
        for edge in transitive_demotion
    ]
    demotion_plan = demotion_plan + transitive_demotion_plan

    hard_adj_dag: dict[str, list[str]] = {node: [] for node in canonical_nodes}
    for edge in hard_edges_dag:
        hard_adj_dag[edge["from"]].append(edge["to"])
    components_dag, comp_id_dag = _tarjan_scc(canonical_nodes, hard_adj_dag)
    scc_dag = _scc_metadata(components_dag, len(canonical_nodes))
    condensed_dag = _build_condensed_graph(canonical_nodes, hard_adj_dag, comp_id_dag)

    soft_edge_set = {(edge["from"], edge["to"]) for edge in soft_edges}
    for edge in demoted_soft:
        soft_edge_set.add((edge["from"], edge["to"]))
    for edge in transitive_demotion:
        soft_edge_set.add((edge["from"], edge["to"]))
    soft_edges_aug = [{"from": src, "to": dst, "type": "soft_ref"} for src, dst in sorted(soft_edge_set)]

    depth_payload = _longest_path_in_dag(canonical_nodes, hard_edges_dag)
    topology = depth_payload["topological_order"]
    max_depth = depth_payload["max_depth"]
    control_scores = _control_scores(canonical_nodes, hard_edges_dag)
    atomic_count = sum(1 for node in canonical_nodes if comp_sizes.get(comp_id[node], 0) == 1)
    atomic_ratio = 0.0 if not canonical_nodes else atomic_count / len(canonical_nodes)
    hard_dag_ratio = 1.0 if _is_dag(len(canonical_nodes), hard_edges_dag) else 0.0

    call_graph = _build_reverse_calls(hard_edges_dag + soft_edges_aug, canonical_nodes)
    called_by = call_graph

    # Persist bridge metadata updates without altering the core skill semantics.
    for node, entry in records.items():
        bridge_file = base / entry["bridge"]
        if not bridge_file.is_file():
            continue
        try:
            bridge = _read_json(bridge_file)
        except (OSError, json.JSONDecodeError):
            continue
        bridge["depends_on"] = sorted(set(edge["to"] for edge in hard_edges_dag + soft_edges_aug if edge["from"] == node))
        bridge["called_by"] = called_by.get(node, [])
        bridge["hard_deps"] = sorted(set(edge["to"] for edge in hard_edges_dag if edge["from"] == node))
        bridge["soft_refs"] = sorted(set(edge["to"] for edge in soft_edges_aug if edge["from"] == node))
        _dump_json(bridge_file, bridge)

    # Update bridge index
    bridge_index: dict = {
        "generated_at": _now_iso(),
        "canonical_to_aliases": aliases,
        "alias_to_canonical": alias_to_canonical,
        "skills": {},
    }
    for node in canonical_nodes:
        entry = records[node]
        bridge_index["skills"][node] = {
            "id": node,
            "cluster": entry["cluster"],
            "archive": entry["archive"],
            "hard_deps": sorted(set(edge["to"] for edge in hard_edges_dag if edge["from"] == node)),
            "soft_deps": sorted(set(edge["to"] for edge in soft_edges_aug if edge["from"] == node)),
            "bridge_path": entry["bridge"],
            "manifest_path": entry["manifest"],
        }

    # Concept extraction for bridge candidate discovery
    concepts = _build_concepts(records, base)
    candidates = _bridge_candidates(records, concepts)
    candidate_path = base / ".index" / "bridge_candidates.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["source", "target", "score"])
        for source, target, score in candidates:
            writer.writerow([source, target, score])

    # Base graph payload
    skill_graph = {
        "generated_at": _now_iso(),
        "canonical_nodes": canonical_nodes,
        "node_count": len(canonical_nodes),
        "edge_count": len(hard_edges),
        "clusters": clusters,
        "hard_edges": _dedupe_edges(hard_edges),
        "soft_edges": _dedupe_edges(soft_edges),
        "hard_edges_dag": hard_edges_dag,
        "soft_edges_augmented": soft_edges_aug,
        "demotion_plan": demotion_plan,
        "scc": {"components": components, "memberships": comp_id},
        "scc_dag": {"components": components_dag, "memberships": comp_id_dag},
        "scc_summary": {
            **raw_scc,
            "hard_dag_acyclic_ratio": 1.0 if _is_dag(len(canonical_nodes), hard_edges_dag) else 0.0,
        },
        "condensed": condensed,
        "condensed_dag": condensed_dag,
    }
    _dump_json(base / ".index" / "skill_graph.json", skill_graph)

    # Condensed graph for external consumers
    clustered_graph = {
        "generated_at": _now_iso(),
        **condensed_dag,
    }
    _dump_json(base / ".index" / "clustered_graph.json", clustered_graph)

    control_ontology = _build_control_ontology(
        records,
        clusters,
        hard_edges_dag,
        soft_edges_aug,
        alias_to_canonical,
        topology,
        comp_id,
        comp_sizes,
        control_scores,
        max_depth,
        demotion_plan,
    )
    _dump_json(base / ".index" / "control_ontology.json", control_ontology)
    control_scores_payload = _compute_mctsr(
        canonical_nodes,
        hard_edges,
        soft_edges,
        hard_edges_dag,
        soft_edges_aug,
        len(unresolved),
        hard_dag_ratio,
        alias_to_canonical,
        control_ontology["control_graph"]["node_count"],
        control_ontology["control_graph"]["relation_count"],
        True,
    )
    _dump_json(base / ".index" / "control_ontology.schema.json", _build_control_schema())

    # Keep hyperedges stable where present; ensure valid JSON.
    hyperedges_path = base / ".index" / "hyperedges.json"
    if not hyperedges_path.exists():
        _dump_json(
            hyperedges_path,
            [
                {
                    "id": "H1",
                    "name": "SAQ_PIPELINE_V2",
                    "type": "workflow",
                    "kind": "hyperedge",
                    "input_nodes": [
                        "cicm-saq-rubric",
                        "saq-schema",
                        "saq-rubric-metaschema",
                        "pex",
                        "constraints",
                    ],
                    "output_nodes": ["saq-rubric", "saq", "saq-template-2"],
                    "confidence": 0.96,
                    "notes": "Fallback generated hyperedge skeleton.",
                }
            ],
        )
    else:
        try:
            hyperedges = _read_json(hyperedges_path)
            if not isinstance(hyperedges, list):
                hyperedges = []
            _dump_json(hyperedges_path, hyperedges)
        except (OSError, json.JSONDecodeError):
            _dump_json(hyperedges_path, [])

    _dump_json(
        base / ".index" / "bridge_index.json",
        bridge_index,
    )

    main_index_yaml = base / ".index" / "main_index.yaml"
    main_index_yaml.write_text(
        _emit_main_index_yaml(
            records,
            clusters,
            aliases,
            skill_graph,
        ),
        encoding="utf-8",
    )

    main_index_payload = {
        "generated_at": _now_iso(),
        "root": str(base),
        "repo_root": "repo",
        "cluster_order": sorted(clusters.keys()),
        "cluster_membership": clusters,
        "node_count": len(canonical_nodes),
        "metrics": {
            "node_count": len(canonical_nodes),
            "hard_edges": len(hard_edges),
            "soft_edges": len(soft_edges),
            "hard_edges_dag": len(hard_edges_dag),
            "demoted_edges": len(demotion_plan),
            "cycle_demotion_edges": len(demoted_soft),
            "transitive_reduction_edges": len(transitive_demotion),
            "acyclic_ratio_raw": raw_scc["acyclic_ratio"],
            "acyclic_ratio_hard_dag": hard_dag_ratio,
            "atomic_ratio": atomic_ratio,
            "max_depth": max_depth,
            "mctsr": control_scores_payload["score"],
            "mctsr_components": control_scores_payload["components"],
            "mctsr_passed": control_scores_payload["passed"],
        },
        "aliases": {
            "canonical_to_aliases": aliases,
            "alias_to_canonical": {
                alias: canonical
                for canonical, names in aliases.items()
                for alias in names
            },
        },
    }
    _dump_json(base / ".index" / "main_index_payload.json", main_index_payload)

    quality_path = base / ".index" / "quality_report.md"
    quality_path.write_text(
            _format_quality_markdown(
            _now_iso(),
            len(canonical_nodes),
            len(hard_edges),
            len(soft_edges),
            raw_scc,
            condensed_dag,
            len(demotion_plan),
            len(transitive_demotion),
            hard_dag_ratio,
            atomic_ratio,
            control_scores_payload["score"],
        ),
        encoding="utf-8",
    )

    processing_path = base / ".index" / "processing_report.md"
    processing_path.write_text(
        _format_processing_report(
            _now_iso(),
            len(canonical_nodes),
            len(canonical_nodes),
            unresolved,
            len(demotion_plan),
            len(transitive_demotion),
            len(candidates),
            atomic_ratio,
            max_depth,
            control_scores_payload["score"],
        ),
        encoding="utf-8",
    )

    return {
        "generated": _now_iso(),
        "node_count": len(canonical_nodes),
        "hard_edges": len(hard_edges),
        "soft_edges": len(soft_edges),
        "hard_edges_dag": len(hard_edges_dag),
        "demoted_edges": len(demotion_plan),
        "cycle_demotion_edges": len(demoted_soft),
        "transitive_reduction_edges": len(transitive_demotion),
        "mctsr": control_scores_payload["score"],
        "mctsr_passed": control_scores_payload["passed"],
        "acyclic_ratio_raw": raw_scc["acyclic_ratio"],
        "acyclic_ratio_hard_dag": 1.0 if _is_dag(len(canonical_nodes), hard_edges_dag) else 0.0,
        "unresolved_references": len(unresolved),
        "bridge_candidates": len(candidates),
        "hard_dag_acyclic_ratio": hard_dag_ratio,
        "atomic_ratio": atomic_ratio,
        "max_depth": max_depth,
    }


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(BASE),
        help="Root path containing the skill repository.",
    )
    args = parser.parse_args()
    report = dump_all(Path(args.base))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main_cli()
