#!/usr/bin/env python3
"""Phase 2 — Targeted attacks T1–T6 on transit hypergraph projection graphs.

T1: node hyperdegree desc            (kind=node)
T2: projection-graph node betweenness (kind=node)
T3: transfer-involved nodes first    (kind=node)  — secondary key: hyperdegree desc
T4: hyperedge size desc              (kind=hyperedge)
T5: hyperedge betweenness proxy = sum of intra_route edge betweenness  (kind=hyperedge)
T6: transfer importance = sum of endpoint hyperdegrees (kind=transfer)

All rankings are deterministic with lexicographic tie-break on ID. Each attack
fraction therefore needs only one metric call (no repetitions), so total work
is roughly Phase-1 / 100. Engine — CSR adjacency, retention indices, BFS-based
metrics — is reused from run_resilience_experiments.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

from run_resilience_experiments import (
    DEFAULT_CITY_CSV,
    DEFAULT_BUILD_ROOT,
    DEFAULT_ANALYSIS_ROOT,
    GRAPH_METRIC_COLUMNS,
    RETENTION_METRIC_COLUMNS,
    ALL_METRIC_COLUMNS,
    load_city_inventory,
    select_cities,
    load_city_artifacts,
    build_csr_index,
    build_edge_mask,
    compute_graph_metrics_csr,
    build_retention_index,
    compute_retention_metrics,
    attack_fraction_grid,
    compute_auc,
)


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results_run_resilience_targeted"

TARGETED_ATTACKS = [
    "T1_node_hyperdegree",
    "T2_node_betweenness",
    "T3_node_transfer_first",
    "T4_edge_size",
    "T5_edge_betweenness",
    "T6_transfer_endpoint_hd",
]

TARGETED_TO_KIND = {
    "T1_node_hyperdegree":      "node",
    "T2_node_betweenness":      "node",
    "T3_node_transfer_first":   "node",
    "T4_edge_size":             "hyperedge",
    "T5_edge_betweenness":      "hyperedge",
    "T6_transfer_endpoint_hd":  "transfer",
}

SUMMARY_COLUMNS_TARGETED = [
    "city",
    "network_version",
    "attack_type",
    "fraction_step",
    "source_samples",
    "auc_lwcc",
    "auc_lscc",
    "auc_reachable_pair_ratio",
    "auc_avg_directed_efficiency",
    "auc_nrr",
    "auc_her",
    "auc_clr",
]


def _sort_ids_by_score(ids: np.ndarray, scores: np.ndarray) -> list[str]:
    """Return ids ordered by score DESC, ties broken by id ASC (lexicographic).

    Implementation note: np.lexsort sorts by the LAST key as primary, ascending.
    We want primary = -score ascending (=score descending), secondary = id ascending.
    So keys = (id_asc, neg_score_asc).
    """
    order = np.lexsort((ids, -scores))
    return ids[order].tolist()


def _build_igraph_projection(idx: dict) -> ig.Graph:
    """Build an igraph DiGraph from the precomputed CSR endpoint arrays.

    Edge order in g.es matches `idx['src_idx']` order, so we can index back into
    the edge_kind / edge_id arrays directly.
    """
    n = idx["n"]
    edges = list(zip(idx["src_idx"].tolist(), idx["tgt_idx"].tolist()))
    return ig.Graph(n=n, edges=edges, directed=True)


def compute_importance_orders(idx: dict, rctx: dict) -> dict[str, list[str]]:
    """Return for each Tk the list of items to delete, most important first.

    Uses only the precomputed CSR + retention indices — no DataFrame ops in the hot path.
    """
    n_nodes = rctx["base_node_count"]
    n_hyperedges = rctx["base_hyperedge_count"]
    n_transfers = rctx["base_transfer_count"]

    # Node id <-> idx
    id2idx_nodes = rctx["node_id2idx"]
    node_ids = np.empty(n_nodes, dtype=object)
    for nid, i in id2idx_nodes.items():
        node_ids[i] = nid
    node_ids = node_ids.astype(str)

    # T1 node hyperdegree
    hyperdegree = np.bincount(rctx["hen_nidx"], minlength=n_nodes).astype(np.float64)

    # T4 hyperedge size
    edge_size = np.bincount(rctx["hen_eidx"], minlength=n_hyperedges).astype(np.float64)
    edge_ids = np.empty(n_hyperedges, dtype=object)
    for eid, i in rctx["edge_id2idx"].items():
        edge_ids[i] = eid
    edge_ids = edge_ids.astype(str)

    # T3 transfer-involved priority: transfer endpoints (high), then by hyperdegree desc
    transfer_involved = np.zeros(n_nodes, dtype=bool)
    transfer_involved[rctx["transfer_bus_idx"]] = True
    transfer_involved[rctx["transfer_metro_idx"]] = True
    # Compose a single score: 1e12 boost for transfer-involved nodes, then their hyperdegree.
    # 1e12 >> any plausible hyperdegree, so transfer nodes strictly come first.
    t3_score = transfer_involved.astype(np.float64) * 1e12 + hyperdegree

    # T2 / T5: igraph betweenness on projection (directed, exact)
    g = _build_igraph_projection(idx)
    node_bw = np.asarray(g.betweenness(directed=True), dtype=np.float64)
    edge_bw = np.asarray(g.edge_betweenness(directed=True), dtype=np.float64)

    # T5: aggregate edge betweenness to hyperedge level (sum over intra_route edges).
    # Edge order in g.es matches idx['src_idx']/edge_kind/edge_id order.
    intra_mask = idx["is_intra"]
    eid_intra = idx["edge_id"][intra_mask]                         # str array
    bw_intra = edge_bw[intra_mask]
    # group sum by hyperedge id; pandas groupby is fine here (one-time per city)
    grp = pd.Series(bw_intra).groupby(eid_intra).sum()
    edge_bw_proxy = np.zeros(n_hyperedges, dtype=np.float64)
    for eid, val in grp.items():
        edge_bw_proxy[rctx["edge_id2idx"][eid]] = float(val)

    # T6 transfer importance = hyperdegree[bus_endpoint] + hyperdegree[metro_endpoint]
    transfer_score = hyperdegree[rctx["transfer_bus_idx"]] + hyperdegree[rctx["transfer_metro_idx"]]
    transfer_ids = np.empty(n_transfers, dtype=object)
    for tid, i in rctx["transfer_id2idx"].items():
        transfer_ids[i] = tid
    transfer_ids = transfer_ids.astype(str)

    return {
        "T1_node_hyperdegree":     _sort_ids_by_score(node_ids, hyperdegree),
        "T2_node_betweenness":     _sort_ids_by_score(node_ids, node_bw),
        "T3_node_transfer_first":  _sort_ids_by_score(node_ids, t3_score),
        "T4_edge_size":            _sort_ids_by_score(edge_ids, edge_size),
        "T5_edge_betweenness":     _sort_ids_by_score(edge_ids, edge_bw_proxy),
        "T6_transfer_endpoint_hd": _sort_ids_by_score(transfer_ids, transfer_score),
    }


def run_targeted_attack_curve(
    idx: dict,
    rctx: dict,
    base_node_count: int,
    ordered_items: list[str],
    attack_type: str,
    step: float,
    source_indices: np.ndarray | None,
) -> pd.DataFrame:
    """Deterministic attack: at each fraction, delete top-floor(N*frac) items in order."""
    fractions = attack_fraction_grid(step)
    kind = TARGETED_TO_KIND[attack_type]
    total = len(ordered_items)
    rows = []
    for fraction in fractions:
        n_remove = min(total, int(round(total * fraction)))
        removed = set(ordered_items[:n_remove])
        edge_keep, _ = build_edge_mask(idx, kind, removed)
        gm = compute_graph_metrics_csr(idx, edge_keep, base_node_count, source_indices)
        rm = compute_retention_metrics(kind, removed, rctx)
        row = {"attack_fraction": fraction}
        for col in GRAPH_METRIC_COLUMNS:
            row[col] = gm[col]
        for col in RETENTION_METRIC_COLUMNS:
            row[col] = rm[col]
        rows.append(row)
    return pd.DataFrame(rows)


def run_city_targeted(
    city_row: pd.Series,
    build_root: Path,
    analysis_root: Path,
    network_version: str,
    step: float,
    seed_base: int,
    source_samples: int,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    loaded = load_city_artifacts(city_row, build_root, analysis_root, network_version)
    edge_df = loaded["projection"]
    nodes_df = loaded["nodes"]
    hyperedges_df = loaded["hyperedges"]
    transfers_df = loaded["transfers"]
    hyperedge_nodes_df = loaded["hyperedge_nodes"]

    base_node_count = len(nodes_df)
    idx = build_csr_index(edge_df, nodes_df["node_id"])
    rctx = build_retention_index(nodes_df, hyperedges_df, transfers_df, hyperedge_nodes_df)

    # Same source-sample policy as Phase 1: city-level fixed source set, K=500 default
    if source_samples and source_samples < base_node_count:
        sample_rng = np.random.default_rng(seed_base + 977)
        source_indices = np.sort(
            sample_rng.choice(base_node_count, size=source_samples, replace=False)
        ).astype(np.int32)
    else:
        source_indices = None
    effective_K = int(len(source_indices)) if source_indices is not None else base_node_count

    orders = compute_importance_orders(idx, rctx)
    curves: dict[str, pd.DataFrame] = {}
    for attack_type in TARGETED_ATTACKS:
        curves[attack_type] = run_targeted_attack_curve(
            idx, rctx, base_node_count, orders[attack_type], attack_type, step, source_indices,
        )

    summaries = []
    for attack_type, curve_df in curves.items():
        summaries.append({
            "city": loaded["city"],
            "network_version": network_version,
            "attack_type": attack_type,
            "fraction_step": step,
            "source_samples": effective_K,
            "auc_lwcc": compute_auc(curve_df, "largest_weakly_connected_ratio"),
            "auc_lscc": compute_auc(curve_df, "largest_strongly_connected_ratio"),
            "auc_reachable_pair_ratio": compute_auc(curve_df, "reachable_ordered_pair_ratio"),
            "auc_avg_directed_efficiency": compute_auc(curve_df, "avg_directed_efficiency"),
            "auc_nrr": compute_auc(curve_df, "node_retention_rate"),
            "auc_her": compute_auc(curve_df, "hyperedge_retention_rate"),
            "auc_clr": compute_auc(curve_df, "cross_layer_retention_rate"),
        })
    return summaries, curves


def write_city_outputs_targeted(
    results_root: Path,
    network_version: str,
    output_folder: str,
    curves: dict[str, pd.DataFrame],
    summaries: list[dict],
) -> None:
    city_dir = results_root / network_version / output_folder
    city_dir.mkdir(parents=True, exist_ok=True)
    for attack_type, curve_df in curves.items():
        curve_df.to_csv(city_dir / f"{attack_type}_attack_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS_TARGETED).to_csv(
        city_dir / "resilience_summary_targeted.csv", index=False, encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 directed attack resilience (T1–T6).")
    parser.add_argument("--city", help="Chinese city name or folder name from the inventory")
    parser.add_argument("--all", action="store_true", help="Run all cities")
    parser.add_argument(
        "--network-version", default="walk_200m",
        choices=["exact_name", "walk_100m", "walk_200m", "walk_300m"],
    )
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--fraction-step", type=float, default=0.02)
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Only affects the K source-sample draw; attack order itself is deterministic.")
    parser.add_argument("--source-samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args()


def _process_city(args_tuple):
    (city_row_dict, build_root, analysis_root, network_version, step, seed_base, source_samples) = args_tuple
    city_row = pd.Series(city_row_dict)
    summaries, curves = run_city_targeted(
        city_row,
        build_root=build_root,
        analysis_root=analysis_root,
        network_version=network_version,
        step=step,
        seed_base=seed_base,
        source_samples=source_samples,
    )
    return str(city_row["城市中文"]), str(city_row["公交文件夹"]), summaries, curves


def main() -> None:
    args = parse_args()
    city_df = load_city_inventory(args.city_csv)
    selected = select_cities(city_df, args.city, args.all)
    all_summaries: list[dict] = []

    job_args = []
    for idx, (_, city_row) in enumerate(selected.iterrows()):
        job_args.append((
            city_row.to_dict(),
            args.build_root,
            args.analysis_root,
            args.network_version,
            args.fraction_step,
            args.seed_base + idx * 1000,
            args.source_samples,
        ))

    n_workers = max(1, min(args.workers, len(job_args)))
    print(f"Phase 2 (targeted): {len(job_args)} cities on {n_workers} worker(s) "
          f"| version={args.network_version} step={args.fraction_step} K={args.source_samples}")

    if n_workers == 1:
        for task in job_args:
            city, output_folder, summaries, curves = _process_city(task)
            write_city_outputs_targeted(args.results_root, args.network_version, output_folder, curves, summaries)
            all_summaries.extend(summaries)
            print(f"[OK] {city} -> {output_folder} summaries={len(summaries)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_city, task): task for task in job_args}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    city, output_folder, summaries, curves = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed_city = futures[fut][0].get("城市中文", "?")
                    print(f"[FAIL {done}/{len(job_args)}] {failed_city}: {exc!r}")
                    continue
                write_city_outputs_targeted(args.results_root, args.network_version, output_folder, curves, summaries)
                all_summaries.extend(summaries)
                print(f"[OK {done}/{len(job_args)}] {city} -> {output_folder} summaries={len(summaries)}")

    out_dir = args.results_root / args.network_version
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries, columns=SUMMARY_COLUMNS_TARGETED).sort_values(
        ["city", "attack_type"]
    ).to_csv(out_dir / "all_cities_resilience_summary_targeted.csv", index=False, encoding="utf-8-sig")
    print(f"Finished Phase 2: cities={len(selected)} version={args.network_version} "
          f"results_root={args.results_root}")


if __name__ == "__main__":
    main()
