#!/usr/bin/env python3
"""Compute higher-order hypergraph metrics for all 45 cities.

Metrics implemented (with literature provenance):
  1. (k,m)-hyper-core decomposition → hypercoreness [Mancastroppa et al. 2023, NatComm]
  2. Cross-order degree correlation ρ_{1,2}  [Zhang et al. 2023, NatComm]
  3. Hyperedge overlap O(v)               [defined in this study]

Output: results_higher_order_metrics/city_higher_order_metrics.csv
        (one row per city, columns = city + structural features + higher-order features)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_ROOT = ROOT / "results_build_transit_hypergraphs" / "walk_200m"
DEFAULT_CITY_CSV = ROOT / "metadata/cities_with_bus_and_metro.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results_higher_order_metrics"


# ====================================================================
# 1. (k,m)-HYPER-CORE DECOMPOSITION  [Mancastroppa et al. 2023]
# ====================================================================

def decompose_hyper_cores(
    node_to_edges: Dict[str, List[str]],
    edge_to_nodes: Dict[str, List[str]],
    m_min: int = 2,
    m_max: int | None = None,
) -> Dict[str, Dict[int, int]]:
    """Compute (k,m)-hyper-core decomposition.

    A node belongs to the (k,m)-hyper-core if it is incident to >= k hyperedges
    of size >= m, within the sub-hypergraph induced by iterative pruning.

    Algorithm (from Mancastroppa et al. 2023, Supplementary Note 1):
      For each m:
        1. Filter to hyperedges of size >= m
        2. Iteratively remove nodes with D_m(v) < k, incrementing k when stable
        3. Record shell index C_m(v) = max k for which v survives

    Returns: dict[node_id -> {m: C_m(node)}]
    """
    if m_max is None:
        m_max = max(len(nodes) for nodes in edge_to_nodes.values())

    # Precompute hyperedge sizes
    edge_sizes = {eid: len(nodes) for eid, nodes in edge_to_nodes.items()}

    # Initialize shell indices: C_m(v) = 0 for all m
    all_nodes = set(node_to_edges.keys())
    shells: Dict[str, Dict[int, int]] = {v: {} for v in all_nodes}

    for m in range(m_min, m_max + 1):
        # Filter to hyperedges with size >= m
        active_edges = {eid for eid, sz in edge_sizes.items() if sz >= m}
        if not active_edges:
            for v in all_nodes:
                shells[v][m] = 0
            continue

        # Working copies
        node_deg = {v: sum(1 for e in node_to_edges[v] if e in active_edges)
                     for v in all_nodes}
        edge_nodes = {eid: set(nodes) & all_nodes
                      for eid, nodes in edge_to_nodes.items()
                      if eid in active_edges}

        k = 1
        survivors = set(all_nodes)

        while survivors:
            # Find nodes with D_m < k
            to_remove = {v for v in survivors if node_deg.get(v, 0) < k}

            if not to_remove:
                # All survivors have D_m >= k → record shell and increase k
                for v in survivors:
                    shells[v][m] = k
                k += 1
                if k > max(node_deg.values(), default=0) + 1:
                    break
            else:
                # Remove these nodes and update degrees
                for v in to_remove:
                    survivors.discard(v)
                    # Decrement D_m for other nodes in the same hyperedges
                    affected_edges = [e for e in node_to_edges[v] if e in active_edges]
                    for e in affected_edges:
                        edge_nodes[e].discard(v)
                        # If hyperedge shrinks below m, it's no longer active
                        if len(edge_nodes[e]) < m:
                            active_edges.discard(e)
                            # Decrement D_m for all remaining nodes in this edge
                            for u in edge_nodes[e]:
                                if u in survivors:
                                    node_deg[u] = max(0, node_deg.get(u, 0) - 1)
                        else:
                            # Decrement D_m for other nodes still in this edge
                            for u in edge_nodes[e]:
                                if u in survivors and u != v:
                                    node_deg[u] = max(0, node_deg.get(u, 0) - 1)
                    node_deg[v] = 0

    return shells


def compute_hypercoreness(
    shells: Dict[str, Dict[int, int]],
    m_max: int,
    hyperedge_sizes: Dict[str, int],
) -> Dict[str, float]:
    """Compute size-independent hypercoreness R(v) from shell indices.

    R(v) = Σ_{m=2}^{M} C_m(v) / k_max^m

    where k_max^m = max_v C_m(v) for normalization.

    Also compute frequency-weighted hypercoreness R_w(v):
    R_w(v) = Σ_{m=2}^{M} Ψ(m) · C_m(v) / k_max^m

    where Ψ(m) = fraction of hyperedges of size m.
    """
    all_nodes = set(shells.keys())
    m_values = sorted(set().union(*(s.keys() for s in shells.values())))

    # Compute k_max^m for each m
    k_max = {}
    for m in m_values:
        vals = [shells[v].get(m, 0) for v in all_nodes]
        k_max[m] = max(vals) if vals and max(vals) > 0 else 1

    # Compute Ψ(m)
    total_edges = len(hyperedge_sizes)
    psi = {}
    for m in m_values:
        count = sum(1 for sz in hyperedge_sizes.values() if sz == m)
        psi[m] = count / total_edges if total_edges > 0 else 0

    R = {}
    R_w = {}
    for v in all_nodes:
        R[v] = sum(shells[v].get(m, 0) / k_max[m] for m in m_values)
        R_w[v] = sum(psi[m] * shells[v].get(m, 0) / k_max[m] for m in m_values)

    return R, R_w


# ====================================================================
# 2. CROSS-ORDER DEGREE CORRELATION  [Zhang et al. 2023, NatComm]
# ====================================================================

def compute_cross_order_correlation(
    node_to_edges: Dict[str, List[str]],
    edge_to_nodes: Dict[str, List[str]],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Pearson correlation between first-order and second-order hyperdegrees.

    First-order degree:  d_H^{(1)}(v) = number of hyperedges containing v
    Second-order degree: d_H^{(2)}(v) = Σ_{e∋v} (|e| - 1)
                                      = total number of co-members across all of v's hyperedges

    Zhang et al. (2023) showed that the sign and magnitude of ρ_{1,2}
    determines whether higher-order interactions enhance or impede synchronization.
    """
    nodes = list(node_to_edges.keys())
    d1 = np.array([len(node_to_edges[v]) for v in nodes], dtype=float)
    d2 = np.array([
        sum(len(edge_to_nodes[e]) - 1 for e in node_to_edges[v] if e in edge_to_nodes)
        for v in nodes
    ], dtype=float)

    # Filter zero-variance cases
    mask = (d1 > 0) & (d2 > 0)
    if mask.sum() < 3:
        return 0.0, d1, d2

    rho, _ = pearsonr(d1[mask], d2[mask])
    return rho, d1, d2


# ====================================================================
# 3. HYPEREDGE OVERLAP  [defined in this study]
# ====================================================================

def compute_hyperedge_overlap(
    node_to_edges: Dict[str, List[str]],
    edge_to_nodes: Dict[str, List[str]],
) -> Dict[str, float]:
    """Compute hyperedge overlap O(v) for each node.

    O(v) = fraction of other nodes that share >= 2 hyperedges with v.
    High O(v): v's routes are heavily overlapping → tension between
    alternative paths and shared propagation channels.
    """
    nodes = list(node_to_edges.keys())
    n = len(nodes)

    # Build reverse index: pair of edges → set of shared nodes
    # (expensive for large networks; use sampling if needed)
    overlap = {}
    for v in nodes:
        edges_v = set(node_to_edges[v])
        # Count co-members across v's edges
        co_members = defaultdict(int)
        for e in edges_v:
            if e in edge_to_nodes:
                for u in edge_to_nodes[e]:
                    if u != v:
                        co_members[u] += 1
        # Count nodes with >= 2 shared edges
        shared_2plus = sum(1 for cnt in co_members.values() if cnt >= 2)
        overlap[v] = shared_2plus / (n - 1) if n > 1 else 0.0

    return overlap


# ====================================================================
# 4. CITY-LEVEL AGGREGATION
# ====================================================================

def compute_city_higher_order_metrics(
    hyperedge_nodes_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> dict:
    """Compute all higher-order metrics for one city.

    Args:
        hyperedge_nodes_df: columns [edge_id, node_id, sequence, ...]
        nodes_df: columns [node_id, mode, ...]

    Returns:
        dict of city-level higher-order metrics
    """
    # Build node↔edge indices
    node_ids = nodes_df["node_id"].astype(str).tolist()
    edge_ids = hyperedge_nodes_df["edge_id"].astype(str).unique().tolist()

    node_to_edges: Dict[str, List[str]] = defaultdict(list)
    edge_to_nodes: Dict[str, List[str]] = defaultdict(list)

    for _, row in hyperedge_nodes_df.iterrows():
        nid = str(row["node_id"])
        eid = str(row["edge_id"])
        if nid in node_ids:
            node_to_edges[nid].append(eid)
            edge_to_nodes[eid].append(nid)

    # Filter to nodes with >= 1 edge
    node_to_edges = {k: v for k, v in node_to_edges.items() if v}
    edge_to_nodes = {k: v for k, v in edge_to_nodes.items() if v}

    n_nodes = len(node_to_edges)
    n_edges = len(edge_to_nodes)

    if n_nodes == 0 or n_edges == 0:
        return {"n_nodes": 0, "n_edges": 0, "error": "empty_hypergraph"}

    # Hyperedge size distribution
    edge_sizes = {eid: len(nodes) for eid, nodes in edge_to_nodes.items()}
    m_max = max(edge_sizes.values())

    # 1. Hypercoreness
    shells = decompose_hyper_cores(node_to_edges, edge_to_nodes, m_min=2, m_max=m_max)
    R, R_w = compute_hypercoreness(shells, m_max, edge_sizes)

    # Hypercoreness distribution statistics
    R_vals = np.array(list(R.values()))
    R_w_vals = np.array(list(R_w.values()))

    # Hyper-core profile: fraction of nodes in (k,m)-hyper-core
    core_profile = {}
    for m in range(2, min(m_max + 1, 51)):  # cap at m=50 for practicality
        for k in range(1, 21):  # cap at k=20
            n_in_core = sum(1 for v in node_to_edges if shells.get(v, {}).get(m, 0) >= k)
            if n_in_core > 0:
                core_profile[(k, m)] = n_in_core / n_nodes

    # Hyper-core area (normalized)
    A_core = sum(core_profile.values()) / (len(core_profile) or 1)

    # 2. Cross-order degree correlation
    rho_12, d1, d2 = compute_cross_order_correlation(node_to_edges, edge_to_nodes)

    # 3. Hyperedge overlap
    overlap = compute_hyperedge_overlap(node_to_edges, edge_to_nodes)
    O_vals = np.array(list(overlap.values()))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        # Hypercoreness
        "mean_hypercoreness": float(np.mean(R_vals)),
        "std_hypercoreness": float(np.std(R_vals)),
        "max_hypercoreness": float(np.max(R_vals)),
        "mean_hypercoreness_weighted": float(np.mean(R_w_vals)),
        "hypercore_area": float(A_core),
        "hypercore_entropy": float(-np.sum(R_vals / R_vals.sum() * np.log(R_vals / R_vals.sum() + 1e-12))
                                    if R_vals.sum() > 0 else 0),
        # Cross-order correlation
        "cross_order_corr": float(rho_12) if not np.isnan(rho_12) else 0.0,
        "mean_d1": float(np.mean(d1)),
        "mean_d2": float(np.mean(d2)),
        "d1_std": float(np.std(d1)),
        "d2_std": float(np.std(d2)),
        "d1_d2_ratio": float(np.mean(d2) / np.mean(d1)) if np.mean(d1) > 0 else 0.0,
        # Hyperedge overlap
        "mean_hyperedge_overlap": float(np.mean(O_vals)),
        "std_hyperedge_overlap": float(np.std(O_vals)),
        "max_hyperedge_overlap": float(np.max(O_vals)),
        "pct_high_overlap": float(np.mean(O_vals > np.percentile(O_vals, 90))),
    }


# ====================================================================
# 5. MAIN PIPELINE
# ====================================================================

def load_city_inventory(city_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(city_csv_path, encoding="utf-8-sig")


def process_all_cities(
    build_root: Path,
    city_csv_path: Path,
    output_dir: Path,
):
    """Compute higher-order metrics for all cities with built hypergraphs."""
    city_df = load_city_inventory(city_csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for _, row in city_df.iterrows():
        city_cn = str(row["城市中文"])
        city_en = str(row["城市英文"])
        output_folder = str(row["公交文件夹"])
        city_dir = build_root / output_folder

        # Check required files
        hn_file = city_dir / "hyperedge_nodes.csv"
        nodes_file = city_dir / "nodes.csv"
        if not hn_file.exists() or not nodes_file.exists():
            print(f"[SKIP] {city_cn}: missing hyperedge_nodes.csv or nodes.csv")
            continue

        print(f"[{city_cn}] Computing higher-order metrics...")
        try:
            hyperedge_nodes = pd.read_csv(hn_file)
            nodes = pd.read_csv(nodes_file)

            metrics = compute_city_higher_order_metrics(hyperedge_nodes, nodes)
            metrics["city"] = city_cn
            metrics["city_en"] = city_en
            all_metrics.append(metrics)

            print(f"  n={metrics['n_nodes']}, m={metrics['n_edges']}, "
                  f"ρ₁₂={metrics['cross_order_corr']:.3f}, "
                  f"R̄={metrics['mean_hypercoreness']:.3f}, "
                  f"Ō={metrics['mean_hyperedge_overlap']:.4f}")

        except Exception as e:
            print(f"[ERROR] {city_cn}: {e}")
            continue

    # Save results
    df = pd.DataFrame(all_metrics)
    output_file = output_dir / "city_higher_order_metrics.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} cities to {output_file}")

    # Quick summary
    print("\n=== Higher-order metrics summary ===")
    for col in ["cross_order_corr", "mean_hypercoreness", "mean_hyperedge_overlap",
                 "hypercore_area", "d1_d2_ratio"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}")

    return df


# ====================================================================
# 6. CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute higher-order hypergraph metrics for all cities")
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city", type=str, default=None,
                        help="Process single city (Chinese name)")
    args = parser.parse_args()

    if args.city:
        city_df = load_city_inventory(args.city_csv)
        mask = city_df["城市中文"].astype(str).eq(args.city)
        if not mask.any():
            raise ValueError(f"City not found: {args.city}")
        row = city_df[mask].iloc[0]
        city_dir = args.build_root / str(row["公交文件夹"])
        hyperedge_nodes = pd.read_csv(city_dir / "hyperedge_nodes.csv")
        nodes = pd.read_csv(city_dir / "nodes.csv")
        metrics = compute_city_higher_order_metrics(hyperedge_nodes, nodes)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        process_all_cities(args.build_root, args.city_csv, args.output_dir)


if __name__ == "__main__":
    main()
