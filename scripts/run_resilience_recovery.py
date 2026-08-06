#!/usr/bin/env python3
"""Phase 6 — Recovery experiments REC1–REC6 on transit hypergraph projection graphs.

From a damaged state at fixed f_dmg=0.5, sweep the recovery fraction β ∈ [0, 1) and
compare six recovery strategies:

  REC1: random_node damage + random recovery               (baseline)
  REC2: random_node damage + hyperdegree-priority recovery (heuristic)
  REC3: T2_betweenness damage + betweenness-priority       (deterministic)
  REC4: T4_edge_size damage + edge-size-priority           (deterministic)
  REC5: random_transfer damage + endpoint-hd-priority      (heuristic)
  REC6: random_node damage + one-shot marginal-LWCC ranking (oracle proxy)

REC1 / REC2 / REC6 share the same 100 damage seeds — three strategies compete on the
same damaged graph each rep, enabling paired statistics. REC3 / REC4 are fully
deterministic (1 rep). REC5 has random damage + deterministic recovery (100 reps).

Engine — CSR adjacency, retention indices, BFS-based metrics — is reused from
run_resilience_experiments.
"""
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from run_resilience_experiments import (
    DEFAULT_CITY_CSV,
    DEFAULT_BUILD_ROOT,
    DEFAULT_ANALYSIS_ROOT,
    GRAPH_METRIC_COLUMNS,
    RETENTION_METRIC_COLUMNS,
    ALL_METRIC_COLUMNS,
    CI_METRIC_COLUMNS,
    load_city_inventory,
    select_cities,
    load_city_artifacts,
    build_csr_index,
    build_edge_mask,
    compute_graph_metrics_csr,
    build_retention_index,
    compute_retention_metrics,
    attack_fraction_grid,
    sample_removed,
    compute_auc,
)
from run_resilience_targeted import compute_importance_orders


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results_run_resilience_recovery"

# 6 recovery strategies, listed in display order
RECOVERY_LABELS = [
    "REC1_rand_dmg_rand_rec",
    "REC2_rand_dmg_hyperdegree_rec",
    "REC3_T2_dmg_betweenness_rec",
    "REC4_T4_dmg_edgesize_rec",
    "REC5_R7_dmg_transferendpoint_rec",
    "REC6_rand_dmg_marginal_rec",
]

SUMMARY_COLUMNS_RECOVERY = [
    "city", "network_version", "attack_type",
    "damage_kind", "recovery_strategy",
    "repetitions", "fraction_step", "damage_fraction", "source_samples",
    "baseline_lwcc", "baseline_reachable",
    "recovery_auc_lwcc", "recovery_auc_lscc",
    "recovery_auc_reachable_pair_ratio", "recovery_auc_avg_directed_efficiency",
    "recovery_auc_nrr", "recovery_auc_her", "recovery_auc_clr",
    "t90_lwcc", "t90_reachable_ordered_pair_ratio",
]


# ---------- one-time per-city precomputation ----------

def build_full_graph_csr_pair(idx: dict) -> tuple[csr_matrix, csr_matrix]:
    """Out-adjacency and in-adjacency CSR of the full (un-damaged) projection graph."""
    n = idx["n"]
    data = np.ones(len(idx["src_idx"]), dtype=np.int8)
    A_out = csr_matrix((data, (idx["src_idx"], idx["tgt_idx"])), shape=(n, n))
    A_in = A_out.T.tocsr()
    return A_out, A_in


def build_recovery_scores(idx: dict, rctx: dict, orders: dict) -> dict:
    """Per-item recovery-priority scores (higher = restore earlier). Reuses the same
    formulas as Phase 2 targeted attacks so 'priority recovery' = 'reverse of priority
    attack' on the same scoring axis.
    """
    n_nodes = rctx["base_node_count"]
    n_h = rctx["base_hyperedge_count"]

    # Node hyperdegree (T1 score)
    hyperdegree = np.bincount(rctx["hen_nidx"], minlength=n_nodes).astype(np.float64)

    # Recover the T2 order → re-compute node_betweenness scores from idx via igraph.
    # We don't need the absolute values — only the order — so reuse compute_importance_orders.
    # Convert orders to a rank-based score: top-of-order gets the highest score.
    def order_to_score(order_list: list[str], id2idx: dict, total: int) -> np.ndarray:
        score = np.zeros(total, dtype=np.float64)
        # Rank n-i so that index-0 in the order gets the highest score.
        for rank, item_id in enumerate(order_list):
            i = id2idx.get(item_id)
            if i is not None:
                score[i] = total - rank
        return score

    bw_score = order_to_score(orders["T2_node_betweenness"], rctx["node_id2idx"], n_nodes)
    edge_size = np.bincount(rctx["hen_eidx"], minlength=n_h).astype(np.float64)

    # Transfer endpoint-hd score = hd[bus] + hd[metro]
    transfer_score = (
        hyperdegree[rctx["transfer_bus_idx"]]
        + hyperdegree[rctx["transfer_metro_idx"]]
    )

    return {
        "node_hyperdegree": hyperdegree,
        "node_betweenness_rank": bw_score,
        "hyperedge_size": edge_size,
        "transfer_endpoint_hd": transfer_score,
    }


# ---------- recovery orderings ----------

def _sort_damaged_by_score(damaged_ids: list[str], score_lookup: dict[str, float]) -> list[str]:
    """Sort by score desc, tie-break by id asc (lexicographic)."""
    return sorted(damaged_ids, key=lambda x: (-score_lookup.get(x, 0.0), x))


def order_random(damaged_ids: list[str], seed: int) -> list[str]:
    rng = random.Random(seed + 5333)
    out = list(damaged_ids)
    rng.shuffle(out)
    return out


def order_by_node_score(damaged_node_ids: list[str], score_arr: np.ndarray, node_id2idx: dict) -> list[str]:
    lookup = {nid: float(score_arr[node_id2idx[nid]]) for nid in damaged_node_ids
              if nid in node_id2idx}
    return _sort_damaged_by_score(damaged_node_ids, lookup)


def order_by_edge_score(damaged_edge_ids: list[str], score_arr: np.ndarray, edge_id2idx: dict) -> list[str]:
    lookup = {eid: float(score_arr[edge_id2idx[eid]]) for eid in damaged_edge_ids
              if eid in edge_id2idx}
    return _sort_damaged_by_score(damaged_edge_ids, lookup)


def order_by_transfer_score(damaged_transfer_ids: list[str], score_arr: np.ndarray,
                            transfer_id2idx: dict) -> list[str]:
    lookup = {tid: float(score_arr[transfer_id2idx[tid]]) for tid in damaged_transfer_ids
              if tid in transfer_id2idx}
    return _sort_damaged_by_score(damaged_transfer_ids, lookup)


def order_by_marginal_lwcc(
    damaged_node_ids: list[str],
    idx: dict,
    A_out: csr_matrix,
    A_in: csr_matrix,
) -> list[str]:
    """One-shot marginal-LWCC oracle.

    For each damaged node u, score = (LWCC if u alone were restored) - (current LWCC).
    Implemented in O(N + E) per call:
      1) Build post-damage CSR; run scipy connected_components for WCC labels & sizes.
      2) For each damaged node u, gather its FULL-graph neighbors (via A_out, A_in);
         filter to alive ones; the candidate restored-component size = 1 + Σ sizes
         of distinct WCC labels touched.
      3) Marginal Δ = max(candidate_size, current_lwcc) - current_lwcc.
    """
    n = idx["n"]
    damaged_idx = np.fromiter(
        (idx["id2idx"][nid] for nid in damaged_node_ids if nid in idx["id2idx"]),
        dtype=np.int32,
    )
    damaged_mask = np.zeros(n, dtype=bool)
    damaged_mask[damaged_idx] = True

    edge_keep = ~(damaged_mask[idx["src_idx"]] | damaged_mask[idx["tgt_idx"]])
    src = idx["src_idx"][edge_keep]
    tgt = idx["tgt_idx"][edge_keep]
    if len(src) == 0:
        # No edges left — recovery order is by alphabetical id (deterministic).
        return sorted(damaged_node_ids)

    A_dmg = csr_matrix(
        (np.ones(len(src), dtype=np.int8), (src, tgt)),
        shape=(n, n),
    )
    _, wcc_lab = connected_components(A_dmg, directed=True, connection="weak")
    wcc_sizes = np.bincount(wcc_lab)
    current_lwcc = int(wcc_sizes.max()) if wcc_sizes.size else 0

    deltas = np.zeros(len(damaged_node_ids), dtype=np.int64)
    for i, nid in enumerate(damaged_node_ids):
        u = idx["id2idx"][nid]
        out_n = A_out[u].indices
        in_n = A_in[u].indices
        if len(out_n) == 0 and len(in_n) == 0:
            deltas[i] = 0  # isolate restoration adds at most 1 to a singleton — Δ=0 vs current LWCC
            continue
        all_neigh = np.unique(np.concatenate([out_n, in_n]))
        alive_neigh = all_neigh[~damaged_mask[all_neigh]]
        if len(alive_neigh) == 0:
            new_size = 1
        else:
            unique_labels = np.unique(wcc_lab[alive_neigh])
            new_size = int(wcc_sizes[unique_labels].sum()) + 1
        deltas[i] = max(new_size, current_lwcc) - current_lwcc

    # Sort by -delta then by id asc (lex tie-break)
    pairs = sorted(zip(damaged_node_ids, deltas), key=lambda p: (-int(p[1]), p[0]))
    return [p[0] for p in pairs]


# ---------- one recovery curve ----------

def _empirical_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def run_recovery_curve(
    idx: dict,
    rctx: dict,
    base_n: int,
    damage_kind: str,            # "node" | "hyperedge" | "transfer"
    damaged_states: list[list[str]],   # length R, each is the damaged-id list for rep r
    recovery_orders: list[list[str]],  # length R, each is the recovery order for rep r
    step: float,
    source_indices: np.ndarray | None,
) -> pd.DataFrame:
    """β-sweep over the recovery fraction. damaged_states & recovery_orders are paired
    per rep — for REC1/REC2/REC6 the damaged_states list is the SAME across strategies,
    while recovery_orders differ.
    """
    betas = attack_fraction_grid(step)
    R = len(damaged_states)
    rows = []
    for beta in betas:
        buf = {col: np.empty(R, dtype=np.float64) for col in ALL_METRIC_COLUMNS}
        for rep in range(R):
            damaged = damaged_states[rep]
            order = recovery_orders[rep]
            n_rec = min(len(order), int(round(len(order) * beta)))
            recovered_set = set(order[:n_rec])
            still_removed = set(damaged) - recovered_set

            edge_keep, _ = build_edge_mask(idx, damage_kind, still_removed)
            gm = compute_graph_metrics_csr(idx, edge_keep, base_n, source_indices)
            rm = compute_retention_metrics(damage_kind, still_removed, rctx)
            for col in GRAPH_METRIC_COLUMNS: buf[col][rep] = gm[col]
            for col in RETENTION_METRIC_COLUMNS: buf[col][rep] = rm[col]

        row = {"recovery_fraction": beta, "repetitions": R}
        for col in ALL_METRIC_COLUMNS:
            vals = buf[col]
            mean = float(vals.mean())
            row[col] = mean
            if R >= 2:
                row[f"std_{col}"] = float(vals.std(ddof=0))
                if col in CI_METRIC_COLUMNS:
                    lo, hi = _empirical_ci(vals)
                    row[f"ci_lo_{col}"] = lo
                    row[f"ci_hi_{col}"] = hi
            else:
                row[f"std_{col}"] = 0.0
                if col in CI_METRIC_COLUMNS:
                    row[f"ci_lo_{col}"] = mean
                    row[f"ci_hi_{col}"] = mean
        rows.append(row)
    return pd.DataFrame(rows)


def _t90(curve_df: pd.DataFrame, metric: str, baseline: float) -> float:
    """Smallest β with metric(β) >= 0.9 × baseline. Returns NaN if never reached.

    Uses linear interpolation between adjacent rows for sub-grid precision.
    """
    if baseline <= 0:
        return float("nan")
    target = 0.9 * baseline
    xs = curve_df["recovery_fraction"].to_numpy(dtype=float)
    ys = curve_df[metric].to_numpy(dtype=float)
    above = ys >= target
    if not above.any():
        return float("nan")
    first = int(np.argmax(above))
    if first == 0:
        return float(xs[0])
    # Linear interp between (xs[first-1], ys[first-1]) and (xs[first], ys[first])
    y0, y1 = ys[first - 1], ys[first]
    if y1 == y0:
        return float(xs[first])
    t = (target - y0) / (y1 - y0)
    return float(xs[first - 1] + t * (xs[first] - xs[first - 1]))


# ---------- city runner ----------

def run_city_recovery(
    city_row: pd.Series,
    build_root: Path,
    analysis_root: Path,
    network_version: str,
    step: float,
    repetitions: int,
    damage_fraction: float,
    seed_base: int,
    source_samples: int,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    loaded = load_city_artifacts(city_row, build_root, analysis_root, network_version)
    edge_df = loaded["projection"]
    nodes_df = loaded["nodes"]
    hyperedges_df = loaded["hyperedges"]
    transfers_df = loaded["transfers"]
    hyperedge_nodes_df = loaded["hyperedge_nodes"]

    base_n = len(nodes_df)
    idx = build_csr_index(edge_df, nodes_df["node_id"])
    rctx = build_retention_index(nodes_df, hyperedges_df, transfers_df, hyperedge_nodes_df)

    if source_samples and source_samples < base_n:
        sample_rng = np.random.default_rng(seed_base + 977)
        source_indices = np.sort(
            sample_rng.choice(base_n, size=source_samples, replace=False)
        ).astype(np.int32)
    else:
        source_indices = None
    effective_K = int(len(source_indices)) if source_indices is not None else base_n

    # Baseline metrics on the un-damaged graph (for T90 normalization)
    all_edges_keep = np.ones(len(idx["src_idx"]), dtype=bool)
    baseline_metrics = compute_graph_metrics_csr(idx, all_edges_keep, base_n, source_indices)
    baseline_lwcc = baseline_metrics["largest_weakly_connected_ratio"]
    baseline_reach = baseline_metrics["reachable_ordered_pair_ratio"]

    # Phase 2 importance orderings (for damage source in REC3/4)
    orders = compute_importance_orders(idx, rctx)
    scores = build_recovery_scores(idx, rctx, orders)
    A_out, A_in = build_full_graph_csr_pair(idx)

    # ----- generate damaged states -----
    node_pool = nodes_df["node_id"].astype(str).tolist()
    transfer_pool = transfers_df["transfer_id"].astype(str).tolist()

    # REC1 / REC2 / REC6: shared random_node damage seeds
    rand_node_damages: list[list[str]] = []
    for rep in range(repetitions):
        rng = random.Random(seed_base + rep * 100003 + 7)
        damaged = sample_removed(node_pool, damage_fraction, rng)
        rand_node_damages.append(sorted(damaged))  # sorted for reproducibility

    # REC5: random_transfer damage seeds (independent from REC1's seeds)
    rand_transfer_damages: list[list[str]] = []
    for rep in range(repetitions):
        rng = random.Random(seed_base + rep * 100003 + 11)
        damaged = sample_removed(transfer_pool, damage_fraction, rng)
        rand_transfer_damages.append(sorted(damaged))

    # REC3 / REC4: deterministic damage from T2 / T4 orderings, top f_dmg
    t2_dmg = orders["T2_node_betweenness"][: int(round(len(orders["T2_node_betweenness"]) * damage_fraction))]
    t4_dmg = orders["T4_edge_size"][: int(round(len(orders["T4_edge_size"]) * damage_fraction))]

    # ----- build recovery orders per strategy -----
    # REC1
    rec1_orders = [order_random(rand_node_damages[r], seed_base + r) for r in range(repetitions)]
    # REC2
    rec2_orders = [order_by_node_score(rand_node_damages[r], scores["node_hyperdegree"], rctx["node_id2idx"])
                   for r in range(repetitions)]
    # REC3 — single deterministic curve
    rec3_orders = [order_by_node_score(t2_dmg, scores["node_betweenness_rank"], rctx["node_id2idx"])]
    rec3_damages = [t2_dmg]
    # REC4 — single deterministic curve
    rec4_orders = [order_by_edge_score(t4_dmg, scores["hyperedge_size"], rctx["edge_id2idx"])]
    rec4_damages = [t4_dmg]
    # REC5
    rec5_orders = [order_by_transfer_score(rand_transfer_damages[r], scores["transfer_endpoint_hd"],
                                            rctx["transfer_id2idx"])
                   for r in range(repetitions)]
    # REC6 — paired with REC1/REC2's damaged states; recovery via marginal-LWCC
    rec6_orders = [order_by_marginal_lwcc(rand_node_damages[r], idx, A_out, A_in)
                   for r in range(repetitions)]

    runs = [
        ("REC1_rand_dmg_rand_rec",          "node",      rand_node_damages,     rec1_orders, repetitions),
        ("REC2_rand_dmg_hyperdegree_rec",   "node",      rand_node_damages,     rec2_orders, repetitions),
        ("REC3_T2_dmg_betweenness_rec",     "node",      rec3_damages,          rec3_orders, 1),
        ("REC4_T4_dmg_edgesize_rec",        "hyperedge", rec4_damages,          rec4_orders, 1),
        ("REC5_R7_dmg_transferendpoint_rec","transfer",  rand_transfer_damages, rec5_orders, repetitions),
        ("REC6_rand_dmg_marginal_rec",      "node",      rand_node_damages,     rec6_orders, repetitions),
    ]

    curves: dict[str, pd.DataFrame] = {}
    for label, damage_kind, damages_list, orders_list, reps in runs:
        curves[label] = run_recovery_curve(
            idx, rctx, base_n, damage_kind,
            damages_list, orders_list, step, source_indices,
        )

    summaries = []
    for label, damage_kind, damages_list, orders_list, reps in runs:
        curve_df = curves[label]
        if "rand" in label and "rand_rec" in label:
            strategy = "random"
        elif "hyperdegree" in label:
            strategy = "hyperdegree"
        elif "betweenness" in label:
            strategy = "betweenness"
        elif "edgesize" in label:
            strategy = "edge_size"
        elif "transferendpoint" in label:
            strategy = "transfer_endpoint_hd"
        elif "marginal" in label:
            strategy = "marginal_lwcc"
        else:
            strategy = "?"
        summaries.append({
            "city": loaded["city"],
            "network_version": network_version,
            "attack_type": label,
            "damage_kind": damage_kind,
            "recovery_strategy": strategy,
            "repetitions": reps,
            "fraction_step": step,
            "damage_fraction": damage_fraction,
            "source_samples": effective_K,
            "baseline_lwcc": baseline_lwcc,
            "baseline_reachable": baseline_reach,
            "recovery_auc_lwcc": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                              "largest_weakly_connected_ratio"),
            "recovery_auc_lscc": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                              "largest_strongly_connected_ratio"),
            "recovery_auc_reachable_pair_ratio": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                                              "reachable_ordered_pair_ratio"),
            "recovery_auc_avg_directed_efficiency": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                                                 "avg_directed_efficiency"),
            "recovery_auc_nrr": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                             "node_retention_rate"),
            "recovery_auc_her": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                             "hyperedge_retention_rate"),
            "recovery_auc_clr": compute_auc(curve_df.rename(columns={"recovery_fraction": "attack_fraction"}),
                                             "cross_layer_retention_rate"),
            "t90_lwcc": _t90(curve_df, "largest_weakly_connected_ratio", baseline_lwcc),
            "t90_reachable_ordered_pair_ratio": _t90(curve_df, "reachable_ordered_pair_ratio", baseline_reach),
        })

    return summaries, curves


# ---------- I/O + CLI ----------

def write_city_outputs_recovery(
    results_root: Path,
    network_version: str,
    output_folder: str,
    curves: dict[str, pd.DataFrame],
    summaries: list[dict],
) -> None:
    city_dir = results_root / network_version / output_folder
    city_dir.mkdir(parents=True, exist_ok=True)
    for label, curve_df in curves.items():
        curve_df.to_csv(city_dir / f"{label}_recovery_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS_RECOVERY).to_csv(
        city_dir / "resilience_summary_recovery.csv", index=False, encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 recovery experiments (REC1–REC6).")
    parser.add_argument("--city")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--network-version", default="walk_200m",
                        choices=["exact_name", "walk_100m", "walk_200m", "walk_300m"])
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--fraction-step", type=float, default=0.02)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--damage-fraction", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--source-samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args()


def _process_city(args_tuple):
    (city_row_dict, build_root, analysis_root, network_version, step, repetitions,
     damage_fraction, seed_base, source_samples) = args_tuple
    city_row = pd.Series(city_row_dict)
    summaries, curves = run_city_recovery(
        city_row,
        build_root=build_root,
        analysis_root=analysis_root,
        network_version=network_version,
        step=step,
        repetitions=repetitions,
        damage_fraction=damage_fraction,
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
    for i, (_, city_row) in enumerate(selected.iterrows()):
        job_args.append((
            city_row.to_dict(),
            args.build_root,
            args.analysis_root,
            args.network_version,
            args.fraction_step,
            args.repetitions,
            args.damage_fraction,
            args.seed_base + i * 1000,
            args.source_samples,
        ))

    n_workers = max(1, min(args.workers, len(job_args)))
    print(f"Phase 6 (recovery): {len(job_args)} cities on {n_workers} worker(s) "
          f"| version={args.network_version} step={args.fraction_step} "
          f"reps={args.repetitions} f_dmg={args.damage_fraction} K={args.source_samples}")

    if n_workers == 1:
        for task in job_args:
            city, output_folder, summaries, curves = _process_city(task)
            write_city_outputs_recovery(args.results_root, args.network_version, output_folder, curves, summaries)
            all_summaries.extend(summaries)
            print(f"[OK] {city} -> {output_folder} curves={len(curves)}")
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
                write_city_outputs_recovery(args.results_root, args.network_version, output_folder, curves, summaries)
                all_summaries.extend(summaries)
                print(f"[OK {done}/{len(job_args)}] {city} -> {output_folder} curves={len(curves)}")

    out_dir = args.results_root / args.network_version
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries, columns=SUMMARY_COLUMNS_RECOVERY).sort_values(
        ["city", "attack_type"]
    ).to_csv(out_dir / "all_cities_resilience_summary_recovery.csv", index=False, encoding="utf-8-sig")
    print(f"Finished Phase 6: cities={len(selected)} version={args.network_version} "
          f"results_root={args.results_root}")


if __name__ == "__main__":
    main()
