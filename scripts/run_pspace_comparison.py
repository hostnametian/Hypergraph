#!/usr/bin/env python3
"""P-space baseline comparison for the 6 representative cities.

This script constructs a P-space (route-as-clique) pairwise graph for each
representative city and runs the same targeted attacks (T1-T3) and cascade
experiments (C1, C4) as the hypergraph pipeline. The purpose is to demonstrate
that the hypergraph representation produces substantively different cascade
dynamics compared to a standard pairwise projection.

Key difference:
- Hypergraph cascade: node fails when route-support fraction < tau
  (requires tracking per-node hyperedge membership)
- P-space cascade: node fails when alive-neighbor fraction < tau
  (only uses pairwise adjacency — cannot distinguish same-route vs cross-route)

Output: results_pspace_comparison/ with per-city CSVs and a combined summary.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from run_resilience_experiments import (
    DEFAULT_CITY_CSV,
    DEFAULT_BUILD_ROOT,
    DEFAULT_ANALYSIS_ROOT,
    GRAPH_METRIC_COLUMNS,
    load_city_inventory,
    select_cities,
    load_city_artifacts,
    build_csr_index,
    build_retention_index,
    attack_fraction_grid,
    sample_removed,
    compute_auc,
)
from run_resilience_targeted import compute_importance_orders


# ---------- configuration ----------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results_pspace_comparison"

REPRESENTATIVE_CITIES = ["贵阳", "成都", "滁州", "青岛", "石家庄", "福州"]

COMPARISON_METRICS = GRAPH_METRIC_COLUMNS + ["cascade_depth", "collapse_ratio"]

SUMMARY_COLUMNS = [
    "city", "representation", "experiment", "metric", "value",
]


# ---------- P-space graph construction ----------

def build_pspace_adjacency(hyperedge_nodes_df: pd.DataFrame, nodes_df: pd.DataFrame,
                           transfers_df: pd.DataFrame) -> csr_matrix:
    """Build a P-space directed adjacency matrix.

    P-space: within each route, consecutive stops are connected (same as the
    hypergraph projection's intra_route edges). Additionally, for each route,
    all member nodes form a clique (undirected). This is the standard P-space
    representation where co-route membership implies pairwise connectivity.

    Transfer edges are added identically to the hypergraph projection.
    """
    node_ids = nodes_df["node_id"].astype(str).to_numpy()
    n = len(node_ids)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build adjacency via lil_matrix (efficient for incremental construction)
    A = lil_matrix((n, n), dtype=np.int8)

    # P-space clique: for each route, connect all pairs of member nodes
    grouped = hyperedge_nodes_df.groupby("edge_id")["node_id"].apply(list)
    for edge_id, members in grouped.items():
        member_idx = [id2idx[str(m)] for m in members if str(m) in id2idx]
        # Directed: connect consecutive stops (preserves route direction)
        for k in range(len(member_idx) - 1):
            A[member_idx[k], member_idx[k + 1]] = 1
        # P-space addition: connect ALL pairs (bidirectional) to form clique
        for i_m in range(len(member_idx)):
            for j_m in range(i_m + 1, len(member_idx)):
                A[member_idx[i_m], member_idx[j_m]] = 1
                A[member_idx[j_m], member_idx[i_m]] = 1

    # Add transfer edges (bidirectional, same as hypergraph projection)
    for _, row in transfers_df.iterrows():
        bus_id = str(row["bus_node_id"])
        metro_id = str(row["metro_node_id"])
        if bus_id in id2idx and metro_id in id2idx:
            bi = id2idx[bus_id]
            mi = id2idx[metro_id]
            A[bi, mi] = 1
            A[mi, bi] = 1

    return A.tocsr()


def build_pspace_sequential_adjacency(hyperedge_nodes_df: pd.DataFrame,
                                       nodes_df: pd.DataFrame,
                                       transfers_df: pd.DataFrame) -> csr_matrix:
    """Build a sequential P-space adjacency (L-space + transfers).

    This is equivalent to the hypergraph projection: only consecutive stops
    within a route are connected. Used as a control to isolate the effect of
    the cascade rule (hyperedge-support vs neighbor-fraction) from the effect
    of graph density.
    """
    node_ids = nodes_df["node_id"].astype(str).to_numpy()
    n = len(node_ids)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    A = lil_matrix((n, n), dtype=np.int8)

    # Sequential (L-space): only consecutive stops
    grouped = hyperedge_nodes_df.sort_values(["edge_id", "sequence"]).groupby("edge_id")["node_id"].apply(list)
    for edge_id, members in grouped.items():
        member_idx = [id2idx[str(m)] for m in members if str(m) in id2idx]
        for k in range(len(member_idx) - 1):
            A[member_idx[k], member_idx[k + 1]] = 1

    # Transfer edges
    for _, row in transfers_df.iterrows():
        bus_id = str(row["bus_node_id"])
        metro_id = str(row["metro_node_id"])
        if bus_id in id2idx and metro_id in id2idx:
            bi = id2idx[bus_id]
            mi = id2idx[metro_id]
            A[bi, mi] = 1
            A[mi, bi] = 1

    return A.tocsr()


# ---------- P-space cascade (neighbor-fraction rule) ----------

def simulate_pspace_cascade(A: csr_matrix, initial_node_dead: np.ndarray,
                            tau: float, max_rounds: int = 50) -> tuple[np.ndarray, int]:
    """P-space cascade: node fails when alive-neighbor fraction drops below (1-tau).

    Equivalent formulation: node fails when lost-neighbor ratio >= tau.
    This uses only pairwise adjacency — no route membership information.
    """
    n = A.shape[0]
    # Initial degree (number of neighbors in the full graph)
    initial_degree = np.array(A.getnnz(axis=1)).flatten().astype(np.float64)
    initial_degree_safe = np.maximum(initial_degree, 1)

    node_dead = initial_node_dead.copy()
    rounds = 0
    while True:
        rounds += 1
        alive_vec = (~node_dead).astype(np.float64)
        alive_neighbors = np.array(A.dot(alive_vec)).flatten()
        lost_ratio = 1.0 - alive_neighbors / initial_degree_safe
        new_dead = (lost_ratio >= tau) & (initial_degree > 0) & (~node_dead)
        if not new_dead.any() or rounds >= max_rounds:
            break
        node_dead |= new_dead
    return node_dead, rounds


# ---------- graph metrics on arbitrary CSR ----------

def compute_graph_metrics_from_csr(A: csr_matrix, node_dead: np.ndarray,
                                    base_n: int,
                                    source_samples: int = 500,
                                    seed: int = 42) -> dict[str, float]:
    """Compute LWCC/LSCC/reachability/efficiency on the subgraph induced by alive nodes."""
    n = A.shape[0]
    alive_idx = np.where(~node_dead)[0]
    n_alive = len(alive_idx)

    if n_alive <= 1:
        return {col: 0.0 for col in GRAPH_METRIC_COLUMNS}

    # Build subgraph adjacency
    alive_mask = ~node_dead
    # Zero out rows/cols of dead nodes
    row_mask = alive_mask[A.tocoo().row]
    col_mask = alive_mask[A.tocoo().col]
    keep = row_mask & col_mask

    coo = A.tocoo()
    src = coo.row[keep]
    tgt = coo.col[keep]
    data = np.ones(len(src), dtype=np.int8)

    if len(src) == 0:
        return {col: 0.0 for col in GRAPH_METRIC_COLUMNS}

    A_sub = csr_matrix((data, (src, tgt)), shape=(n, n))

    # WCC / SCC on active nodes
    active = np.zeros(n, dtype=bool)
    active[src] = True
    active[tgt] = True
    n_active = int(active.sum())
    if n_active == 0:
        return {col: 0.0 for col in GRAPH_METRIC_COLUMNS}

    _, scc_lab = connected_components(A_sub, directed=True, connection="strong")
    _, wcc_lab = connected_components(A_sub, directed=True, connection="weak")
    lwcc = int(np.bincount(wcc_lab[active]).max())
    lscc = int(np.bincount(scc_lab[active]).max())

    # Sampled reachability + efficiency
    rng = np.random.default_rng(seed)
    K = min(source_samples, n_alive)
    source_indices = np.sort(rng.choice(alive_idx, size=K, replace=False)).astype(np.int32)

    D = dijkstra(A_sub, directed=True, unweighted=True, indices=source_indices)
    D[np.arange(K), source_indices] = np.inf
    denom = K * (base_n - 1)

    finite = np.isfinite(D)
    reachable = int(finite.sum())
    inv = np.zeros_like(D)
    np.reciprocal(D, out=inv, where=finite)
    eff_sum = float(inv.sum())

    return {
        "largest_weakly_connected_ratio": lwcc / base_n,
        "largest_strongly_connected_ratio": lscc / base_n,
        "reachable_ordered_pair_ratio": reachable / denom,
        "avg_directed_efficiency": eff_sum / denom,
    }


# ---------- targeted attack on P-space ----------

def run_pspace_targeted_attack(A: csr_matrix, node_ids: np.ndarray,
                                ordered_items: list[str], id2idx: dict,
                                base_n: int, step: float,
                                source_samples: int = 500) -> pd.DataFrame:
    """Deterministic targeted attack on P-space graph (no cascade, static only)."""
    fractions = attack_fraction_grid(step)
    total = len(ordered_items)
    rows = []
    for fraction in fractions:
        n_remove = min(total, int(round(total * fraction)))
        removed = set(ordered_items[:n_remove])
        node_dead = np.zeros(base_n, dtype=bool)
        for nid in removed:
            if nid in id2idx:
                node_dead[id2idx[nid]] = True
        gm = compute_graph_metrics_from_csr(A, node_dead, base_n, source_samples)
        row = {"attack_fraction": fraction}
        row.update(gm)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- cascade comparison runner ----------

def run_cascade_comparison(
    A_pspace: csr_matrix,
    A_sequential: csr_matrix,
    rctx: dict,
    idx: dict,
    node_ids: np.ndarray,
    id2idx: dict,
    base_n: int,
    attack_pool: list[str],
    tau: float,
    step: float,
    repetitions: int,
    seed_base: int,
    source_samples: int,
    deterministic_order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run cascade on both P-space (neighbor rule) and hypergraph (route-support rule).

    Returns dict with keys:
      'pspace_clique': P-space clique graph + neighbor-fraction cascade
      'pspace_sequential': Sequential graph + neighbor-fraction cascade
      'hypergraph': Sequential graph + route-support cascade (the paper's method)
    """
    from run_resilience_cascade import (
        simulate_hyperedge_cascade,
        build_node_initial_hyperdegree,
    )

    fractions = attack_fraction_grid(step)
    is_deterministic = deterministic_order is not None
    pool = deterministic_order if is_deterministic else attack_pool
    local_reps = 1 if is_deterministic else repetitions

    initial_hd = build_node_initial_hyperdegree(rctx)

    results = {
        "pspace_clique": [],
        "pspace_sequential": [],
        "hypergraph": [],
    }

    for fraction in fractions:
        bufs = {k: {"lwcc": [], "depth": [], "collapse": []} for k in results}

        for rep in range(local_reps):
            # Initial removal
            if is_deterministic:
                n_rm = min(len(pool), int(round(len(pool) * fraction)))
                removed = set(pool[:n_rm])
            else:
                rng = random.Random(seed_base + rep * 100003 + int(fraction * 10000))
                removed = sample_removed(pool, fraction, rng)

            # Build initial dead mask
            initial_dead = np.zeros(base_n, dtype=bool)
            for nid in removed:
                if nid in id2idx:
                    initial_dead[id2idx[nid]] = True

            # --- P-space clique + neighbor cascade ---
            dead_pc, depth_pc = simulate_pspace_cascade(A_pspace, initial_dead.copy(), tau)
            gm_pc = compute_graph_metrics_from_csr(A_pspace, dead_pc, base_n, source_samples)
            bufs["pspace_clique"]["lwcc"].append(gm_pc["largest_weakly_connected_ratio"])
            bufs["pspace_clique"]["depth"].append(depth_pc)
            bufs["pspace_clique"]["collapse"].append(float(dead_pc.mean()))

            # --- P-space sequential + neighbor cascade ---
            dead_ps, depth_ps = simulate_pspace_cascade(A_sequential, initial_dead.copy(), tau)
            gm_ps = compute_graph_metrics_from_csr(A_sequential, dead_ps, base_n, source_samples)
            bufs["pspace_sequential"]["lwcc"].append(gm_ps["largest_weakly_connected_ratio"])
            bufs["pspace_sequential"]["depth"].append(depth_ps)
            bufs["pspace_sequential"]["collapse"].append(float(dead_ps.mean()))

            # --- Hypergraph route-support cascade ---
            dead_hg, depth_hg = simulate_hyperedge_cascade(
                rctx, initial_dead.copy(), initial_hd, tau
            )
            gm_hg = compute_graph_metrics_from_csr(A_sequential, dead_hg, base_n, source_samples)
            bufs["hypergraph"]["lwcc"].append(gm_hg["largest_weakly_connected_ratio"])
            bufs["hypergraph"]["depth"].append(depth_hg)
            bufs["hypergraph"]["collapse"].append(float(dead_hg.mean()))

        for k in results:
            results[k].append({
                "attack_fraction": fraction,
                "largest_weakly_connected_ratio": float(np.mean(bufs[k]["lwcc"])),
                "cascade_depth": float(np.mean(bufs[k]["depth"])),
                "collapse_ratio": float(np.mean(bufs[k]["collapse"])),
            })

    return {k: pd.DataFrame(v) for k, v in results.items()}


# ---------- main city runner ----------

def run_city_comparison(
    city_row: pd.Series,
    build_root: Path,
    analysis_root: Path,
    network_version: str,
    step: float,
    repetitions: int,
    seed_base: int,
    source_samples: int,
    tau: float,
) -> dict:
    """Run full P-space vs hypergraph comparison for one city."""
    loaded = load_city_artifacts(city_row, build_root, analysis_root, network_version)
    nodes_df = loaded["nodes"]
    hyperedges_df = loaded["hyperedges"]
    transfers_df = loaded["transfers"]
    hyperedge_nodes_df = loaded["hyperedge_nodes"]
    edge_df = loaded["projection"]

    base_n = len(nodes_df)
    node_ids = nodes_df["node_id"].astype(str).to_numpy()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build indices for hypergraph cascade
    idx = build_csr_index(edge_df, nodes_df["node_id"])
    rctx = build_retention_index(nodes_df, hyperedges_df, transfers_df, hyperedge_nodes_df)

    # Build P-space graphs
    print(f"  Building P-space clique graph for {loaded['city']}...")
    A_pspace = build_pspace_adjacency(hyperedge_nodes_df, nodes_df, transfers_df)
    print(f"  Building P-space sequential graph for {loaded['city']}...")
    A_sequential = build_pspace_sequential_adjacency(hyperedge_nodes_df, nodes_df, transfers_df)

    print(f"  P-space clique: {A_pspace.nnz} edges | Sequential: {A_sequential.nnz} edges")

    # Get importance orders (same as hypergraph pipeline)
    orders = compute_importance_orders(idx, rctx)
    node_pool = nodes_df["node_id"].astype(str).tolist()

    city_results = {
        "city": loaded["city"],
        "base_n": base_n,
        "pspace_clique_edges": A_pspace.nnz,
        "sequential_edges": A_sequential.nnz,
    }

    # --- Experiment 1: Static targeted attacks (T1, T2, T3) on P-space vs sequential ---
    print(f"  Running static targeted attacks...")
    for attack_name in ["T1_node_hyperdegree", "T2_node_betweenness", "T3_node_transfer_first"]:
        order = orders[attack_name]
        curve_pspace = run_pspace_targeted_attack(
            A_pspace, node_ids, order, id2idx, base_n, step, source_samples
        )
        curve_seq = run_pspace_targeted_attack(
            A_sequential, node_ids, order, id2idx, base_n, step, source_samples
        )
        city_results[f"static_{attack_name}_pspace_clique_auc_lwcc"] = compute_auc(curve_pspace, "largest_weakly_connected_ratio")
        city_results[f"static_{attack_name}_sequential_auc_lwcc"] = compute_auc(curve_seq, "largest_weakly_connected_ratio")

    # --- Experiment 2: Cascade comparison (C1 random, C4 targeted) ---
    print(f"  Running cascade comparison (C1 random trigger, tau={tau})...")
    cascade_c1 = run_cascade_comparison(
        A_pspace, A_sequential, rctx, idx, node_ids, id2idx, base_n,
        node_pool, tau, step, repetitions, seed_base + 11, source_samples,
        deterministic_order=None,
    )
    for k, df in cascade_c1.items():
        city_results[f"C1_{k}_auc_lwcc"] = compute_auc(df, "largest_weakly_connected_ratio")
        city_results[f"C1_{k}_mean_depth"] = float(df["cascade_depth"].mean())
        city_results[f"C1_{k}_mean_collapse"] = float(df["collapse_ratio"].mean())

    print(f"  Running cascade comparison (C4 targeted trigger, tau={tau})...")
    t2_order = orders["T2_node_betweenness"]
    cascade_c4 = run_cascade_comparison(
        A_pspace, A_sequential, rctx, idx, node_ids, id2idx, base_n,
        node_pool, tau, step, 1, seed_base + 29, source_samples,
        deterministic_order=t2_order,
    )
    for k, df in cascade_c4.items():
        city_results[f"C4_{k}_auc_lwcc"] = compute_auc(df, "largest_weakly_connected_ratio")
        city_results[f"C4_{k}_mean_depth"] = float(df["cascade_depth"].mean())
        city_results[f"C4_{k}_mean_collapse"] = float(df["collapse_ratio"].mean())

    # Store curves for detailed output
    city_results["_curves_c1"] = cascade_c1
    city_results["_curves_c4"] = cascade_c4

    return city_results


# ---------- output ----------

def write_outputs(results_root: Path, all_results: list[dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)

    # Write per-city cascade curves
    for res in all_results:
        city = res["city"]
        city_dir = results_root / city
        city_dir.mkdir(parents=True, exist_ok=True)
        for exp_key in ["_curves_c1", "_curves_c4"]:
            if exp_key in res:
                prefix = exp_key.replace("_curves_", "")
                for repr_name, df in res[exp_key].items():
                    df.to_csv(city_dir / f"{prefix}_{repr_name}_curve.csv",
                              index=False, encoding="utf-8-sig")

    # Write combined summary
    summary_rows = []
    for res in all_results:
        city = res["city"]
        for key, val in res.items():
            if key.startswith("_") or key in ("city", "base_n"):
                continue
            summary_rows.append({
                "city": city,
                "metric": key,
                "value": val,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_root / "pspace_comparison_summary.csv",
                      index=False, encoding="utf-8-sig")

    # Write formatted comparison table
    comparison_rows = []
    for res in all_results:
        city = res["city"]
        row = {"city": city, "n_nodes": res["base_n"],
               "pspace_clique_edges": res.get("pspace_clique_edges", 0),
               "sequential_edges": res.get("sequential_edges", 0)}

        # Static attack AUCs
        for attack in ["T1_node_hyperdegree", "T2_node_betweenness", "T3_node_transfer_first"]:
            row[f"{attack}_pspace"] = res.get(f"static_{attack}_pspace_clique_auc_lwcc", None)
            row[f"{attack}_sequential"] = res.get(f"static_{attack}_sequential_auc_lwcc", None)

        # Cascade comparisons
        for exp in ["C1", "C4"]:
            for metric in ["auc_lwcc", "mean_depth", "mean_collapse"]:
                for repr_name in ["pspace_clique", "pspace_sequential", "hypergraph"]:
                    key = f"{exp}_{repr_name}_{metric}"
                    row[key] = res.get(key, None)
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(results_root / "pspace_vs_hypergraph_table.csv",
                   index=False, encoding="utf-8-sig")

    # Print summary
    print("\n" + "=" * 80)
    print("P-SPACE vs HYPERGRAPH COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'City':<12} {'Edges(clique)':<14} {'Edges(seq)':<12} "
          f"{'C1 depth(P)':<13} {'C1 depth(H)':<13} {'C4 depth(P)':<13} {'C4 depth(H)':<13}")
    print("-" * 90)
    for res in all_results:
        print(f"{res['city']:<12} "
              f"{res.get('pspace_clique_edges', 0):<14} "
              f"{res.get('sequential_edges', 0):<12} "
              f"{res.get('C1_pspace_clique_mean_depth', 0):<13.3f} "
              f"{res.get('C1_hypergraph_mean_depth', 0):<13.3f} "
              f"{res.get('C4_pspace_clique_mean_depth', 0):<13.3f} "
              f"{res.get('C4_hypergraph_mean_depth', 0):<13.3f}")

    print(f"\n{'City':<12} {'C1 LWCC(P-clq)':<16} {'C1 LWCC(P-seq)':<16} {'C1 LWCC(HG)':<14} "
          f"{'C4 LWCC(P-clq)':<16} {'C4 LWCC(P-seq)':<16} {'C4 LWCC(HG)':<14}")
    print("-" * 110)
    for res in all_results:
        print(f"{res['city']:<12} "
              f"{res.get('C1_pspace_clique_auc_lwcc', 0):<16.4f} "
              f"{res.get('C1_pspace_sequential_auc_lwcc', 0):<16.4f} "
              f"{res.get('C1_hypergraph_auc_lwcc', 0):<14.4f} "
              f"{res.get('C4_pspace_clique_auc_lwcc', 0):<16.4f} "
              f"{res.get('C4_pspace_sequential_auc_lwcc', 0):<16.4f} "
              f"{res.get('C4_hypergraph_auc_lwcc', 0):<14.4f}")

    print(f"\nResults written to: {results_root}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P-space vs Hypergraph comparison on 6 representative cities."
    )
    parser.add_argument("--network-version", default="walk_200m",
                        choices=["exact_name", "walk_100m", "walk_200m", "walk_300m"])
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--fraction-step", type=float, default=0.02)
    parser.add_argument("--repetitions", type=int, default=50,
                        help="MC reps for random-trigger cascades (C1). Default 50 for speed.")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--source-samples", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Cascade threshold. Default 0.2 (same as C1 in the paper).")
    parser.add_argument("--cities", nargs="*", default=None,
                        help="Override representative cities (Chinese names). "
                             "Default: the 6 cluster representatives.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_df = load_city_inventory(args.city_csv)

    target_cities = args.cities if args.cities else REPRESENTATIVE_CITIES
    print(f"P-space comparison: {len(target_cities)} cities | "
          f"version={args.network_version} tau={args.tau} reps={args.repetitions}")

    all_results = []
    for city_cn in target_cities:
        mask = city_df["城市中文"].astype(str).eq(city_cn)
        if not mask.any():
            print(f"[SKIP] City not found: {city_cn}")
            continue
        city_row = city_df[mask].iloc[0]
        print(f"\n[START] {city_cn} ({city_row['公交文件夹']})")

        try:
            res = run_city_comparison(
                city_row,
                build_root=args.build_root,
                analysis_root=args.analysis_root,
                network_version=args.network_version,
                step=args.fraction_step,
                repetitions=args.repetitions,
                seed_base=args.seed_base,
                source_samples=args.source_samples,
                tau=args.tau,
            )
            all_results.append(res)
            print(f"[OK] {city_cn}")
        except Exception as exc:
            print(f"[FAIL] {city_cn}: {exc!r}")
            import traceback
            traceback.print_exc()

    if all_results:
        write_outputs(args.results_root, all_results)
    else:
        print("No results to write.")


if __name__ == "__main__":
    main()
