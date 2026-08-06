#!/usr/bin/env python3
"""Phase 5 — Cascade failures C1–C5 on transit hypergraph projection graphs.

C1–C3: random_node initial attack + hyperedge-loss cascade rule (τ=0.2/0.4/0.6, 100 reps)
C4:    T2-betweenness ordered node attack + hyperedge-loss cascade (τ=0.2/0.4/0.6, deterministic)
C5:    random_transfer initial attack + transfer-loss cascade rule (τ=0.2/0.4/0.6, 100 reps)

Cascade rule for C1–C4 (hyperedge-loss, synchronous Motter–Lai style):
    A node fails when ≥ τ fraction of its initial hyperedges are no longer alive
    (a hyperedge is alive iff ≥ 2 of its members are alive). Iterate until fixed point.

Cascade rule for C5 (transfer-loss, symmetric form):
    A transfer-endpoint node fails when ≥ τ fraction of its initial transfers are dead.
    A transfer is dead iff (a) directly removed by initial attack, or (b) either endpoint
    has died. Iterate until fixed point.

Engine — CSR adjacency, retention indices, BFS-based metrics — is reused from
run_resilience_experiments. T2 ordering is reused from run_resilience_targeted.
"""
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from run_resilience_experiments import (
    DEFAULT_CITY_CSV,
    DEFAULT_BUILD_ROOT,
    DEFAULT_ANALYSIS_ROOT,
    GRAPH_METRIC_COLUMNS,
    RETENTION_METRIC_COLUMNS,
    load_city_inventory,
    select_cities,
    load_city_artifacts,
    build_csr_index,
    compute_graph_metrics_csr,
    build_retention_index,
    attack_fraction_grid,
    sample_removed,
    compute_auc,
)
from run_resilience_targeted import compute_importance_orders


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results_run_resilience_cascade"

# Two extra dynamics metrics beyond the standard 7 (4 graph + 3 retention)
DYNAMICS_COLUMNS = ["collapse_ratio", "cascade_depth"]
CASCADE_METRIC_COLUMNS = GRAPH_METRIC_COLUMNS + RETENTION_METRIC_COLUMNS + DYNAMICS_COLUMNS

SUMMARY_COLUMNS_CASCADE = [
    "city", "network_version", "attack_type", "cascade_kind", "tau",
    "repetitions", "fraction_step", "source_samples",
    "auc_lwcc", "auc_lscc", "auc_reachable_pair_ratio", "auc_avg_directed_efficiency",
    "auc_nrr", "auc_her", "auc_clr",
    "auc_collapse_ratio", "mean_cascade_depth",
]


# ---------- one-time per-city precomputation ----------

def build_node_initial_hyperdegree(rctx: dict) -> np.ndarray:
    """Per-node count of hyperedges they participate in (initial state)."""
    return np.bincount(rctx["hen_nidx"], minlength=rctx["base_node_count"]).astype(np.int32)


def build_node_initial_transfer_count(rctx: dict) -> np.ndarray:
    """Per-node count of transfers they're an endpoint of (initial state)."""
    n = rctx["base_node_count"]
    cnt = np.zeros(n, dtype=np.int32)
    np.add.at(cnt, rctx["transfer_bus_idx"], 1)
    np.add.at(cnt, rctx["transfer_metro_idx"], 1)
    return cnt


def build_transfer_edge_lookup(idx: dict, rctx: dict) -> np.ndarray:
    """For each edge in idx, the index into rctx transfer arrays (or -1 for non-transfer edges)."""
    n_e = len(idx["src_idx"])
    lookup = np.full(n_e, -1, dtype=np.int32)
    is_t = idx["is_transfer"]
    if not is_t.any():
        return lookup
    eids_t = idx["edge_id"][is_t]
    t_idx = np.fromiter(
        (rctx["transfer_id2idx"].get(eid, -1) for eid in eids_t),
        dtype=np.int32,
        count=len(eids_t),
    )
    lookup[is_t] = t_idx
    return lookup


# ---------- cascade simulators ----------

def simulate_hyperedge_cascade(
    rctx: dict,
    initial_node_dead: np.ndarray,
    initial_hd: np.ndarray,
    tau: float,
    max_rounds: int = 50,
) -> tuple[np.ndarray, int]:
    """Synchronous hyperedge-loss cascade. All numpy bincount; no Python loop over nodes."""
    n = rctx["base_node_count"]
    n_h = rctx["base_hyperedge_count"]
    hen_e = rctx["hen_eidx"]
    hen_n = rctx["hen_nidx"]
    initial_hd_safe = np.maximum(initial_hd, 1)

    node_dead = initial_node_dead.copy()
    rounds = 0
    while True:
        rounds += 1
        alive_per_edge = np.bincount(
            hen_e,
            weights=(~node_dead[hen_n]).astype(np.int32),
            minlength=n_h,
        )
        edge_alive = alive_per_edge >= 2
        node_alive_h = np.bincount(
            hen_n,
            weights=edge_alive[hen_e].astype(np.int32),
            minlength=n,
        )
        lost_ratio = 1.0 - node_alive_h / initial_hd_safe
        new_dead = (lost_ratio >= tau) & (initial_hd > 0) & (~node_dead)
        if not new_dead.any() or rounds >= max_rounds:
            break
        node_dead |= new_dead
    return node_dead, rounds


def simulate_transfer_cascade(
    rctx: dict,
    initial_transfer_dead: np.ndarray,
    initial_tcount: np.ndarray,
    tau: float,
    max_rounds: int = 50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Synchronous transfer-loss cascade. Returns (node_dead, transfer_dead, rounds)."""
    n = rctx["base_node_count"]
    bus_idx = rctx["transfer_bus_idx"]
    metro_idx = rctx["transfer_metro_idx"]
    has_t = initial_tcount > 0
    safe_init = np.maximum(initial_tcount, 1)

    node_dead = np.zeros(n, dtype=bool)
    transfer_dead = initial_transfer_dead.copy()
    rounds = 0
    while True:
        rounds += 1
        # A transfer is dead if directly killed OR either endpoint already dead.
        # Recompute from cumulative state each round for correctness.
        effective_dead = transfer_dead | node_dead[bus_idx] | node_dead[metro_idx]
        node_alive_t = np.zeros(n, dtype=np.int32)
        np.add.at(node_alive_t, bus_idx, (~effective_dead).astype(np.int32))
        np.add.at(node_alive_t, metro_idx, (~effective_dead).astype(np.int32))
        lost_ratio = 1.0 - node_alive_t / safe_init
        new_dead = (lost_ratio >= tau) & has_t & (~node_dead)
        if not new_dead.any() or rounds >= max_rounds:
            break
        node_dead |= new_dead
    # Final transfer_dead state: also include endpoint-killed
    transfer_dead = transfer_dead | node_dead[bus_idx] | node_dead[metro_idx]
    return node_dead, transfer_dead, rounds


# ---------- post-cascade edge mask + retention ----------

def build_post_cascade_edge_mask(
    idx: dict,
    node_dead: np.ndarray,
    transfer_dead: np.ndarray | None,
    transfer_edge_lookup: np.ndarray,
) -> np.ndarray:
    """Edges to keep after the full attack + cascade."""
    keep = ~(node_dead[idx["src_idx"]] | node_dead[idx["tgt_idx"]])
    if transfer_dead is not None:
        # Look up each transfer edge's transfer_idx, then test against transfer_dead.
        # For non-transfer edges, lookup is -1 → mask out of consideration.
        t_lookup = transfer_edge_lookup
        is_killed_transfer = (t_lookup >= 0) & transfer_dead[np.maximum(t_lookup, 0)]
        keep &= ~is_killed_transfer
    return keep


def compute_post_cascade_retention(
    rctx: dict,
    node_dead: np.ndarray,
    transfer_dead: np.ndarray | None,
) -> dict[str, float]:
    """NRR / HER / CLR consistent with the post-cascade state.

    For C1–C4 (transfer_dead is None): HER and CLR are derived from node_dead only.
    For C5: NRR/HER from node_dead, CLR from transfer_dead which already incorporates endpoint deaths.
    """
    base_n = rctx["base_node_count"]
    base_h = rctx["base_hyperedge_count"]
    base_t = rctx["base_transfer_count"]

    nrr = float((~node_dead).sum()) / base_n if base_n else 0.0

    if base_h == 0:
        her = 0.0
    else:
        alive_in_edge = ~node_dead[rctx["hen_nidx"]]
        survivors = np.bincount(
            rctx["hen_eidx"], weights=alive_in_edge.astype(np.int32), minlength=base_h,
        )
        her = float((survivors >= 2).sum()) / base_h

    if base_t == 0:
        clr = 0.0
    elif transfer_dead is not None:
        clr = float((~transfer_dead).sum()) / base_t
    else:
        bus_alive = ~node_dead[rctx["transfer_bus_idx"]]
        metro_alive = ~node_dead[rctx["transfer_metro_idx"]]
        clr = float((bus_alive & metro_alive).sum()) / base_t

    return {
        "node_retention_rate": nrr,
        "hyperedge_retention_rate": her,
        "cross_layer_retention_rate": clr,
    }


# ---------- attack curve runner ----------

def _empirical_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def run_cascade_attack_curve(
    idx: dict,
    rctx: dict,
    base_n: int,
    attack_pool: list[str],
    tau: float,
    step: float,
    repetitions: int,
    seed_base: int,
    source_indices: np.ndarray | None,
    cascade_kind: str,
    initial_node_hd: np.ndarray,
    initial_tcount: np.ndarray,
    transfer_edge_lookup: np.ndarray,
    deterministic_order: list[str] | None = None,
) -> pd.DataFrame:
    """One curve = one (initial-attack source, tau) combination.

    cascade_kind = 'hyperedge' (C1–C4) or 'transfer' (C5).
    If deterministic_order is set, repetitions is forced to 1 and removal goes by prefix.
    """
    fractions = attack_fraction_grid(step)
    is_deterministic = deterministic_order is not None
    pool = deterministic_order if is_deterministic else attack_pool
    n_h = rctx["base_hyperedge_count"]  # noqa: F841 -- kept for clarity
    n_t = rctx["base_transfer_count"]   # noqa: F841

    local_reps = 1 if is_deterministic else repetitions
    rows = []
    for fraction in fractions:
        buf = {col: np.empty(local_reps, dtype=np.float64) for col in CASCADE_METRIC_COLUMNS}

        for rep in range(local_reps):
            # 1. Initial removal set
            if is_deterministic:
                n_rm = min(len(pool), int(round(len(pool) * fraction)))
                removed = set(pool[:n_rm])
            else:
                rng = random.Random(seed_base + rep * 100003 + int(fraction * 10000))
                removed = sample_removed(pool, fraction, rng)

            # 2. Cascade
            if cascade_kind == "hyperedge":
                initial_node_dead = np.zeros(base_n, dtype=bool)
                if removed:
                    rm_idx = np.fromiter(
                        (rctx["node_id2idx"][r] for r in removed if r in rctx["node_id2idx"]),
                        dtype=np.int32,
                    )
                    initial_node_dead[rm_idx] = True
                node_dead, depth = simulate_hyperedge_cascade(
                    rctx, initial_node_dead, initial_node_hd, tau,
                )
                transfer_dead = None
            else:  # transfer
                initial_transfer_dead = np.zeros(rctx["base_transfer_count"], dtype=bool)
                if removed:
                    rm_idx = np.fromiter(
                        (rctx["transfer_id2idx"][t] for t in removed if t in rctx["transfer_id2idx"]),
                        dtype=np.int32,
                    )
                    initial_transfer_dead[rm_idx] = True
                node_dead, transfer_dead, depth = simulate_transfer_cascade(
                    rctx, initial_transfer_dead, initial_tcount, tau,
                )

            # 3. Metric on the post-cascade graph
            keep_mask = build_post_cascade_edge_mask(idx, node_dead, transfer_dead, transfer_edge_lookup)
            gm = compute_graph_metrics_csr(idx, keep_mask, base_n, source_indices)
            rm_metrics = compute_post_cascade_retention(rctx, node_dead, transfer_dead)
            collapse_ratio = float(node_dead.mean())

            buf["collapse_ratio"][rep] = collapse_ratio
            buf["cascade_depth"][rep] = depth
            for col in GRAPH_METRIC_COLUMNS:
                buf[col][rep] = gm[col]
            for col in RETENTION_METRIC_COLUMNS:
                buf[col][rep] = rm_metrics[col]

        # Aggregate row
        row = {"attack_fraction": fraction, "repetitions": local_reps, "tau": tau}
        for col in CASCADE_METRIC_COLUMNS:
            values = buf[col]
            mean = float(values.mean())
            row[col] = mean
            if local_reps >= 2:
                row[f"std_{col}"] = float(values.std(ddof=0))
                lo, hi = _empirical_ci(values)
                row[f"ci_lo_{col}"] = lo
                row[f"ci_hi_{col}"] = hi
            else:
                # Deterministic: no spread. Fill with zero/mean for plotting convenience.
                row[f"std_{col}"] = 0.0
                row[f"ci_lo_{col}"] = mean
                row[f"ci_hi_{col}"] = mean
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- city runner ----------

def run_city_cascade(
    city_row: pd.Series,
    build_root: Path,
    analysis_root: Path,
    network_version: str,
    step: float,
    repetitions: int,
    seed_base: int,
    source_samples: int,
    taus: list[float],
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

    # City-level fixed source sample (same policy as Phase 1)
    if source_samples and source_samples < base_n:
        sample_rng = np.random.default_rng(seed_base + 977)
        source_indices = np.sort(
            sample_rng.choice(base_n, size=source_samples, replace=False)
        ).astype(np.int32)
    else:
        source_indices = None
    effective_K = int(len(source_indices)) if source_indices is not None else base_n

    # One-time precomputation
    initial_hd = build_node_initial_hyperdegree(rctx)
    initial_tcount = build_node_initial_transfer_count(rctx)
    transfer_edge_lookup = build_transfer_edge_lookup(idx, rctx)

    # T2 ordering reused for C4
    orders = compute_importance_orders(idx, rctx)
    t2_order = orders["T2_node_betweenness"]

    node_pool = nodes_df["node_id"].astype(str).tolist()
    transfer_pool = transfers_df["transfer_id"].astype(str).tolist()

    # 9 curves: (C1–C3, C4, C5) × {0.2, 0.4, 0.6}
    curve_specs = []
    for tau in taus:
        c_id = {0.2: 1, 0.4: 2, 0.6: 3}.get(tau, None)
        label = f"C{c_id}_random_node_tau{tau}" if c_id else f"Crand_node_tau{tau}"
        curve_specs.append((label, node_pool, tau, "hyperedge", None, 11))
    for tau in taus:
        label = f"C4_targeted_node_tau{tau}"
        curve_specs.append((label, None, tau, "hyperedge", t2_order, 29))
    for tau in taus:
        label = f"C5_random_transfer_tau{tau}"
        curve_specs.append((label, transfer_pool, tau, "transfer", None, 47))

    curves: dict[str, pd.DataFrame] = {}
    for label, pool, tau, cascade_kind, deterministic_order, seed_offset in curve_specs:
        curves[label] = run_cascade_attack_curve(
            idx, rctx, base_n,
            pool or [], tau, step, repetitions, seed_base + seed_offset,
            source_indices, cascade_kind,
            initial_hd, initial_tcount, transfer_edge_lookup,
            deterministic_order=deterministic_order,
        )

    summaries = []
    for label, curve_df in curves.items():
        # Parse cascade_kind and tau from label
        if "transfer" in label:
            cascade_kind_s = "transfer"
        else:
            cascade_kind_s = "hyperedge"
        tau_val = float(label.split("tau")[-1])
        rep_for_row = int(curve_df["repetitions"].iloc[0])
        summaries.append({
            "city": loaded["city"],
            "network_version": network_version,
            "attack_type": label,
            "cascade_kind": cascade_kind_s,
            "tau": tau_val,
            "repetitions": rep_for_row,
            "fraction_step": step,
            "source_samples": effective_K,
            "auc_lwcc": compute_auc(curve_df, "largest_weakly_connected_ratio"),
            "auc_lscc": compute_auc(curve_df, "largest_strongly_connected_ratio"),
            "auc_reachable_pair_ratio": compute_auc(curve_df, "reachable_ordered_pair_ratio"),
            "auc_avg_directed_efficiency": compute_auc(curve_df, "avg_directed_efficiency"),
            "auc_nrr": compute_auc(curve_df, "node_retention_rate"),
            "auc_her": compute_auc(curve_df, "hyperedge_retention_rate"),
            "auc_clr": compute_auc(curve_df, "cross_layer_retention_rate"),
            "auc_collapse_ratio": compute_auc(curve_df, "collapse_ratio"),
            "mean_cascade_depth": float(curve_df["cascade_depth"].mean()),
        })

    return summaries, curves


# ---------- I/O + CLI ----------

def write_city_outputs_cascade(
    results_root: Path,
    network_version: str,
    output_folder: str,
    curves: dict[str, pd.DataFrame],
    summaries: list[dict],
) -> None:
    city_dir = results_root / network_version / output_folder
    city_dir.mkdir(parents=True, exist_ok=True)
    for label, curve_df in curves.items():
        curve_df.to_csv(city_dir / f"{label}_attack_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS_CASCADE).to_csv(
        city_dir / "resilience_summary_cascade.csv", index=False, encoding="utf-8-sig",
    )


def parse_taus(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 cascade failure experiments (C1–C5).")
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
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--source-samples", type=int, default=500)
    parser.add_argument("--taus", type=parse_taus, default=[0.2, 0.4, 0.6],
                        help="Comma-separated cascade thresholds. Default: 0.2,0.4,0.6")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args()


def _process_city(args_tuple):
    (city_row_dict, build_root, analysis_root, network_version, step, repetitions,
     seed_base, source_samples, taus) = args_tuple
    city_row = pd.Series(city_row_dict)
    summaries, curves = run_city_cascade(
        city_row,
        build_root=build_root,
        analysis_root=analysis_root,
        network_version=network_version,
        step=step,
        repetitions=repetitions,
        seed_base=seed_base,
        source_samples=source_samples,
        taus=taus,
    )
    return str(city_row["城市中文"]), str(city_row["公交文件夹"]), summaries, curves


def main() -> None:
    args = parse_args()
    city_df = load_city_inventory(args.city_csv)
    selected = select_cities(city_df, args.city, args.all)
    all_summaries: list[dict] = []

    job_args = []
    for idx_, (_, city_row) in enumerate(selected.iterrows()):
        job_args.append((
            city_row.to_dict(),
            args.build_root,
            args.analysis_root,
            args.network_version,
            args.fraction_step,
            args.repetitions,
            args.seed_base + idx_ * 1000,
            args.source_samples,
            args.taus,
        ))

    n_workers = max(1, min(args.workers, len(job_args)))
    print(f"Phase 5 (cascade): {len(job_args)} cities on {n_workers} worker(s) "
          f"| version={args.network_version} step={args.fraction_step} "
          f"reps={args.repetitions} K={args.source_samples} taus={args.taus}")

    if n_workers == 1:
        for task in job_args:
            city, output_folder, summaries, curves = _process_city(task)
            write_city_outputs_cascade(args.results_root, args.network_version, output_folder, curves, summaries)
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
                write_city_outputs_cascade(args.results_root, args.network_version, output_folder, curves, summaries)
                all_summaries.extend(summaries)
                print(f"[OK {done}/{len(job_args)}] {city} -> {output_folder} curves={len(curves)}")

    out_dir = args.results_root / args.network_version
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries, columns=SUMMARY_COLUMNS_CASCADE).sort_values(
        ["city", "attack_type"]
    ).to_csv(out_dir / "all_cities_resilience_summary_cascade.csv", index=False, encoding="utf-8-sig")
    print(f"Finished Phase 5: cities={len(selected)} version={args.network_version} "
          f"results_root={args.results_root}")


if __name__ == "__main__":
    main()
