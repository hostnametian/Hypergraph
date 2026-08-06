#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITY_CSV = PROJECT_ROOT / "metadata/cities_with_bus_and_metro.csv"
DEFAULT_BUILD_ROOT = PROJECT_ROOT / "results_build_transit_hypergraphs"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "results_analyze_transit_hypergraphs"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_run_resilience_experiments"
SUMMARY_COLUMNS = [
    "city",
    "network_version",
    "attack_type",
    "repetitions",
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

ATTACK_TYPES = [
    "random_node_all",
    "random_node_bus",
    "random_node_metro",
    "random_hyperedge_all",
    "random_hyperedge_bus",
    "random_hyperedge_metro",
    "random_transfer",
]

GRAPH_METRIC_COLUMNS = [
    "largest_weakly_connected_ratio",
    "largest_strongly_connected_ratio",
    "reachable_ordered_pair_ratio",
    "avg_directed_efficiency",
]

RETENTION_METRIC_COLUMNS = [
    "node_retention_rate",
    "hyperedge_retention_rate",
    "cross_layer_retention_rate",
]

ALL_METRIC_COLUMNS = GRAPH_METRIC_COLUMNS + RETENTION_METRIC_COLUMNS


def load_city_inventory(city_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(city_csv_path, encoding="utf-8-sig")


def select_cities(city_df: pd.DataFrame, city: str | None, run_all: bool) -> pd.DataFrame:
    if run_all:
        return city_df.copy()
    if not city:
        raise ValueError("Provide --city <name> or use --all")
    mask = (
        city_df["城市中文"].astype(str).eq(city)
        | city_df["公交文件夹"].astype(str).eq(city)
        | city_df["地铁文件夹"].astype(str).eq(city)
    )
    selected = city_df[mask].copy()
    if selected.empty:
        raise ValueError(f"City not found in inventory: {city}")
    return selected


def build_version_path(root: Path, network_version: str) -> Path:
    return root / network_version


def load_city_artifacts(city_row: pd.Series, build_root: Path, analysis_root: Path, network_version: str) -> dict[str, pd.DataFrame | str]:
    output_folder = str(city_row["公交文件夹"])
    build_dir = build_version_path(build_root, network_version) / output_folder
    analysis_dir = build_version_path(analysis_root, network_version) / output_folder
    required = {
        "nodes": build_dir / "nodes.csv",
        "hyperedges": build_dir / "hyperedges.csv",
        "hyperedge_nodes": build_dir / "hyperedge_nodes.csv",
        "transfers": build_dir / "transfers.csv",
        "projection": analysis_dir / "directed_edges_projected_with_transfers.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing resilience input files:\n" + "\n".join(missing))
    return {
        "city": str(city_row["城市中文"]),
        "output_folder": output_folder,
        "nodes": pd.read_csv(required["nodes"]),
        "hyperedges": pd.read_csv(required["hyperedges"]),
        "hyperedge_nodes": pd.read_csv(required["hyperedge_nodes"]),
        "transfers": pd.read_csv(required["transfers"]),
        "projection": pd.read_csv(required["projection"]),
    }


def build_csr_index(edge_df: pd.DataFrame, base_nodes: pd.Series) -> dict:
    """Pre-index nodes and edges to integers. Done ONCE per city.

    Returns vectors that let an attack be implemented as a boolean mask over edges
    and a boolean mask over nodes, so each (fraction, rep) is O(E) integer work
    plus one scipy call — no DataFrame copies, no Python dicts in the hot loop.
    """
    node_ids = base_nodes.astype(str).to_numpy()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    src = edge_df["source_node_id"].astype(str).to_numpy()
    tgt = edge_df["target_node_id"].astype(str).to_numpy()
    n_nodes = len(node_ids)

    # Map edge endpoints to integer indices. Drop any edge whose endpoint
    # is missing from nodes.csv (defensive — shouldn't normally happen).
    src_idx = np.fromiter((id2idx.get(s, -1) for s in src), dtype=np.int32, count=len(src))
    tgt_idx = np.fromiter((id2idx.get(t, -1) for t in tgt), dtype=np.int32, count=len(tgt))
    valid = (src_idx >= 0) & (tgt_idx >= 0)
    src_idx = src_idx[valid]
    tgt_idx = tgt_idx[valid]

    edge_kind = edge_df["edge_kind"].to_numpy()[valid]
    edge_id_arr = edge_df["edge_id"].astype(str).to_numpy()[valid]

    return {
        "n": n_nodes,
        "node_ids": node_ids,
        "id2idx": id2idx,
        "src_idx": src_idx,
        "tgt_idx": tgt_idx,
        "edge_kind": edge_kind,
        "edge_id": edge_id_arr,
        "is_intra": (edge_kind == "intra_route"),
        "is_transfer": (edge_kind == "transfer"),
    }


def build_edge_mask(idx: dict, kind: str, removed: set[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (edge_keep_mask, node_remove_mask).

    For node attacks the node mask is also used to zero out reachable_count / efficiency
    for vanished sources (their out-row in the distance matrix is irrelevant).
    """
    n_edges = len(idx["src_idx"])
    if not removed:
        return np.ones(n_edges, dtype=bool), None

    if kind == "node":
        removed_idx = np.array(
            [idx["id2idx"][r] for r in removed if r in idx["id2idx"]],
            dtype=np.int32,
        )
        node_removed = np.zeros(idx["n"], dtype=bool)
        node_removed[removed_idx] = True
        keep = ~(node_removed[idx["src_idx"]] | node_removed[idx["tgt_idx"]])
        return keep, node_removed

    if kind == "hyperedge":
        target_kind = idx["is_intra"]
    elif kind == "transfer":
        target_kind = idx["is_transfer"]
    else:
        raise ValueError(f"Unknown attack kind: {kind}")

    removed_mask = np.isin(idx["edge_id"], np.fromiter(removed, dtype=idx["edge_id"].dtype))
    keep = ~(target_kind & removed_mask)
    return keep, None


def compute_graph_metrics_csr(
    idx: dict,
    edge_keep_mask: np.ndarray,
    base_node_count: int,
    source_indices: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute the four projection-graph metrics in C via scipy.

    SCC/WCC are always exact (scipy is O(N+E)).
    Reachability + efficiency are exact when source_indices is None; otherwise
    they are unbiased estimators using the supplied source subset. The denominator
    uses (K, base_node_count-1) for the sampled case so the estimator is unbiased
    regardless of K.
    """
    if base_node_count <= 1:
        return {col: 0.0 for col in GRAPH_METRIC_COLUMNS}

    src = idx["src_idx"][edge_keep_mask]
    tgt = idx["tgt_idx"][edge_keep_mask]
    n = idx["n"]

    if len(src) == 0:
        return {col: 0.0 for col in GRAPH_METRIC_COLUMNS}

    data = np.ones(len(src), dtype=np.int8)
    A = csr_matrix((data, (src, tgt)), shape=(n, n))

    # Restrict component analysis to nodes that actually appear in the graph.
    # Isolated nodes (no in/out edges remaining) are excluded so that LWCC/LSCC
    # reflect what the projection-graph attack actually leaves connected.
    active = np.zeros(n, dtype=bool)
    active[src] = True
    active[tgt] = True
    n_active = int(active.sum())
    if n_active == 0:
        return {col: 0.0 for col in GRAPH_METRIC_COLUMNS}

    _, scc_lab = connected_components(A, directed=True, connection="strong")
    _, wcc_lab = connected_components(A, directed=True, connection="weak")
    lwcc = int(np.bincount(wcc_lab[active]).max())
    lscc = int(np.bincount(scc_lab[active]).max())

    # All-pairs (or sampled) shortest path lengths via BFS — scipy C kernel.
    if source_indices is None:
        D = dijkstra(A, directed=True, unweighted=True)
        np.fill_diagonal(D, np.inf)
        denom = base_node_count * (base_node_count - 1)
    else:
        K = len(source_indices)
        D = dijkstra(A, directed=True, unweighted=True, indices=source_indices)
        # Self distance for each sampled source is 0 — exclude it from the sum.
        D[np.arange(K), source_indices] = np.inf
        denom = K * (base_node_count - 1)

    finite = np.isfinite(D)
    reachable = int(finite.sum())
    inv = np.zeros_like(D)
    np.reciprocal(D, out=inv, where=finite)
    eff_sum = float(inv.sum())

    return {
        "largest_weakly_connected_ratio": lwcc / base_node_count,
        "largest_strongly_connected_ratio": lscc / base_node_count,
        "reachable_ordered_pair_ratio": reachable / denom,
        "avg_directed_efficiency": eff_sum / denom,
    }


def attack_kind(attack_type: str) -> str:
    if attack_type.startswith("random_node"):
        return "node"
    if attack_type.startswith("random_hyperedge"):
        return "hyperedge"
    if attack_type == "random_transfer":
        return "transfer"
    raise ValueError(f"Unknown attack_type: {attack_type}")


def build_retention_index(
    nodes_df: pd.DataFrame,
    hyperedges_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    hyperedge_nodes_df: pd.DataFrame,
) -> dict:
    """Pre-index retention inputs to integer arrays (done ONCE per city).

    Lets HER/CLR be computed as boolean mask reductions on numpy arrays instead of
    per-rep pandas groupby calls.
    """
    base_nodes = nodes_df["node_id"].astype(str).to_numpy()
    node_id2idx = {nid: i for i, nid in enumerate(base_nodes)}

    base_hyperedges = hyperedges_df["edge_id"].astype(str).to_numpy()
    edge_id2idx = {eid: i for i, eid in enumerate(base_hyperedges)}

    hen_eid = hyperedge_nodes_df["edge_id"].astype(str).to_numpy()
    hen_nid = hyperedge_nodes_df["node_id"].astype(str).to_numpy()
    # Map each participation row to integer edge/node indices.
    hen_eidx = np.fromiter(
        (edge_id2idx.get(e, -1) for e in hen_eid), dtype=np.int32, count=len(hen_eid)
    )
    hen_nidx = np.fromiter(
        (node_id2idx.get(n, -1) for n in hen_nid), dtype=np.int32, count=len(hen_nid)
    )
    valid = (hen_eidx >= 0) & (hen_nidx >= 0)
    hen_eidx = hen_eidx[valid]
    hen_nidx = hen_nidx[valid]

    transfer_bus = transfers_df["bus_node_id"].astype(str).to_numpy()
    transfer_metro = transfers_df["metro_node_id"].astype(str).to_numpy()
    transfer_bus_idx = np.fromiter(
        (node_id2idx.get(n, -1) for n in transfer_bus), dtype=np.int32, count=len(transfer_bus)
    )
    transfer_metro_idx = np.fromiter(
        (node_id2idx.get(n, -1) for n in transfer_metro),
        dtype=np.int32,
        count=len(transfer_metro),
    )

    base_transfer_ids = transfers_df["transfer_id"].astype(str).to_numpy()
    transfer_id2idx = {tid: i for i, tid in enumerate(base_transfer_ids)}

    return {
        "node_id2idx": node_id2idx,
        "edge_id2idx": edge_id2idx,
        "transfer_id2idx": transfer_id2idx,
        "base_node_count": len(base_nodes),
        "base_hyperedge_count": len(base_hyperedges),
        "base_transfer_count": len(base_transfer_ids),
        "hen_eidx": hen_eidx,
        "hen_nidx": hen_nidx,
        "transfer_bus_idx": transfer_bus_idx,
        "transfer_metro_idx": transfer_metro_idx,
    }


def compute_retention_metrics(
    kind: str,
    removed: set[str],
    rctx: dict,
) -> Dict[str, float]:
    base_n = rctx["base_node_count"]
    base_h = rctx["base_hyperedge_count"]
    base_t = rctx["base_transfer_count"]

    if kind == "node":
        nrr = (base_n - len(removed)) / base_n if base_n else 0.0
        if not removed:
            return {
                "node_retention_rate": 1.0,
                "hyperedge_retention_rate": 1.0,
                "cross_layer_retention_rate": 1.0,
            }
        rm_idx = np.fromiter(
            (rctx["node_id2idx"][r] for r in removed if r in rctx["node_id2idx"]),
            dtype=np.int32,
        )
        node_killed = np.zeros(base_n, dtype=bool)
        node_killed[rm_idx] = True

        if base_h == 0:
            her = 0.0
        else:
            # Surviving nodes per hyperedge — bincount of (node not killed) within each edge group.
            alive_in_edge = ~node_killed[rctx["hen_nidx"]]
            survivors = np.bincount(
                rctx["hen_eidx"], weights=alive_in_edge.astype(np.int32), minlength=base_h
            )
            her = float((survivors >= 2).sum()) / base_h

        if base_t == 0:
            clr = 0.0
        else:
            bus_alive = ~node_killed[rctx["transfer_bus_idx"]]
            metro_alive = ~node_killed[rctx["transfer_metro_idx"]]
            clr = float((bus_alive & metro_alive).sum()) / base_t
        return {
            "node_retention_rate": nrr,
            "hyperedge_retention_rate": her,
            "cross_layer_retention_rate": clr,
        }

    if kind == "hyperedge":
        return {
            "node_retention_rate": 1.0,
            "hyperedge_retention_rate": ((base_h - len(removed)) / base_h) if base_h else 0.0,
            "cross_layer_retention_rate": 1.0,
        }

    if kind == "transfer":
        return {
            "node_retention_rate": 1.0,
            "hyperedge_retention_rate": 1.0,
            "cross_layer_retention_rate": ((base_t - len(removed)) / base_t) if base_t else 0.0,
        }

    raise ValueError(f"Unknown attack kind: {kind}")


def attack_fraction_grid(step: float) -> list[float]:
    fractions = np.arange(0.0, 1.0 - 1e-9, step)
    return [round(float(x), 4) for x in fractions]


def sample_removed(items: list[str], fraction: float, rng: random.Random) -> set[str]:
    if not items:
        return set()
    remove_count = min(len(items), int(round(len(items) * fraction)))
    if remove_count == 0:
        return set()
    return set(rng.sample(items, remove_count))


def compute_auc(curve_df: pd.DataFrame, metric: str) -> float:
    x = curve_df["attack_fraction"].to_numpy(dtype=float)
    y = curve_df[metric].to_numpy(dtype=float)
    return float(np.trapezoid(y, x))


CI_METRIC_COLUMNS = [
    "largest_weakly_connected_ratio",
    "largest_strongly_connected_ratio",
    "reachable_ordered_pair_ratio",
    "avg_directed_efficiency",
    "node_retention_rate",
    "hyperedge_retention_rate",
    "cross_layer_retention_rate",
]


def run_random_attack_curve(
    idx: dict,
    rctx: dict,
    base_node_count: int,
    attack_items: list[str],
    attack_type: str,
    step: float,
    repetitions: int,
    seed_base: int,
    source_indices: np.ndarray | None,
    progress_label: str | None = None,
) -> pd.DataFrame:
    """Sweep `attack_fraction` × `repetitions`, computing all 7 metrics per draw.

    `source_indices` is a fixed (city-level) subset of node indices used as BFS
    sources for the sampled reachability / efficiency estimator. Holding it
    fixed across reps and fractions means the across-rep variance reflects
    only the attack randomness, not measurement noise.
    """
    fractions = attack_fraction_grid(step)
    kind = attack_kind(attack_type)
    rows = []
    frac_iter = fractions
    if progress_label:
        frac_iter = tqdm(fractions, desc=progress_label, unit="fraction")
    for fraction in frac_iter:
        metrics_buffer = {col: np.empty(repetitions, dtype=np.float64) for col in ALL_METRIC_COLUMNS}
        for rep in range(repetitions):
            rng = random.Random(seed_base + rep * 100003 + int(fraction * 10000))
            removed = sample_removed(attack_items, fraction, rng)
            edge_keep, _ = build_edge_mask(idx, kind, removed)
            graph_metrics = compute_graph_metrics_csr(idx, edge_keep, base_node_count, source_indices)
            retention = compute_retention_metrics(kind, removed, rctx)
            for col in GRAPH_METRIC_COLUMNS:
                metrics_buffer[col][rep] = graph_metrics[col]
            for col in RETENTION_METRIC_COLUMNS:
                metrics_buffer[col][rep] = retention[col]

        row = {
            "attack_fraction": fraction,
            "repetitions": repetitions,
        }
        for col in ALL_METRIC_COLUMNS:
            values = metrics_buffer[col]
            row[col] = float(values.mean())
            row[f"std_{col}"] = float(values.std(ddof=0))
            if col in CI_METRIC_COLUMNS and repetitions >= 2:
                lo, hi = np.quantile(values, [0.025, 0.975])
                row[f"ci_lo_{col}"] = float(lo)
                row[f"ci_hi_{col}"] = float(hi)
        rows.append(row)
    return pd.DataFrame(rows)


def run_city_resilience(
    city_row: pd.Series,
    build_root: Path,
    analysis_root: Path,
    network_version: str,
    step: float,
    repetitions: int,
    seed_base: int,
    source_samples: int = 500,
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

    # City-level source sample for reachability/efficiency estimator.
    # source_samples == 0 (or >= N) means use ALL sources (exact, no sampling).
    if source_samples and source_samples < base_node_count:
        sample_rng = np.random.default_rng(seed_base + 977)
        source_indices = np.sort(
            sample_rng.choice(base_node_count, size=source_samples, replace=False)
        ).astype(np.int32)
    else:
        source_indices = None

    node_pools = {
        "all": nodes_df["node_id"].astype(str).tolist(),
        "bus": nodes_df.loc[nodes_df["mode"] == "bus", "node_id"].astype(str).tolist(),
        "metro": nodes_df.loc[nodes_df["mode"] == "metro", "node_id"].astype(str).tolist(),
    }
    hyperedge_pools = {
        "all": hyperedges_df["edge_id"].astype(str).tolist(),
        "bus": hyperedges_df.loc[hyperedges_df["mode"] == "bus", "edge_id"].astype(str).tolist(),
        "metro": hyperedges_df.loc[hyperedges_df["mode"] == "metro", "edge_id"].astype(str).tolist(),
    }
    transfer_ids = transfers_df["transfer_id"].astype(str).tolist()

    attack_specs = [
        ("random_node_all",        node_pools["all"],        11),
        ("random_node_bus",        node_pools["bus"],        13),
        ("random_node_metro",      node_pools["metro"],      17),
        ("random_hyperedge_all",   hyperedge_pools["all"],   29),
        ("random_hyperedge_bus",   hyperedge_pools["bus"],   31),
        ("random_hyperedge_metro", hyperedge_pools["metro"], 37),
        ("random_transfer",        transfer_ids,             47),
    ]

    curves: dict[str, pd.DataFrame] = {}
    for attack_type, pool, seed_offset in attack_specs:
        curves[attack_type] = run_random_attack_curve(
            idx,
            rctx,
            base_node_count,
            pool,
            attack_type,
            step,
            repetitions,
            seed_base + seed_offset,
            source_indices,
        )

    summaries = []
    effective_K = int(len(source_indices)) if source_indices is not None else base_node_count
    for attack_type, curve_df in curves.items():
        summaries.append({
            "city": loaded["city"],
            "network_version": network_version,
            "attack_type": attack_type,
            "repetitions": repetitions,
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


def write_city_outputs(results_root: Path, network_version: str, output_folder: str, curves: dict[str, pd.DataFrame], summaries: list[dict]) -> None:
    city_dir = results_root / network_version / output_folder
    city_dir.mkdir(parents=True, exist_ok=True)
    for attack_type, curve_df in curves.items():
        curve_df.to_csv(city_dir / f"{attack_type}_attack_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(city_dir / "resilience_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 random-attack resilience for transit hypergraphs (R1–R7).")
    parser.add_argument("--city", help="Chinese city name or folder name from the city inventory")
    parser.add_argument("--all", action="store_true", help="Run all cities from the city inventory")
    parser.add_argument("--network-version", default="walk_200m", choices=["exact_name", "walk_100m", "walk_200m", "walk_300m"], help="Input network version to evaluate")
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--fraction-step", type=float, default=0.02)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--source-samples",
        type=int,
        default=500,
        help="K BFS sources per city for sampled reachability/efficiency. "
             "0 (or >= N) = use ALL sources (exact). Default 500 keeps "
             "estimator |error| < 0.5%% (mean) on cities up to N≈18k, while "
             "letting the largest city finish in ~6 h on one core.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="Parallel worker processes (one city per worker). Default = cpu-2.")
    return parser.parse_args()


def _process_city(args_tuple):
    """Worker entry point. Returns (city, output_folder, summaries, curves) or raises."""
    (city_row_dict, build_root, analysis_root, network_version, step, repetitions, seed_base, source_samples) = args_tuple
    city_row = pd.Series(city_row_dict)
    summaries, curves = run_city_resilience(
        city_row,
        build_root=build_root,
        analysis_root=analysis_root,
        network_version=network_version,
        step=step,
        repetitions=repetitions,
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
            args.repetitions,
            args.seed_base + idx * 1000,
            args.source_samples,
        ))

    n_workers = max(1, min(args.workers, len(job_args)))
    print(f"Running {len(job_args)} cities on {n_workers} worker(s) | version={args.network_version} "
          f"step={args.fraction_step} reps={args.repetitions} source_samples={args.source_samples}")

    if n_workers == 1:
        for task in job_args:
            city, output_folder, summaries, curves = _process_city(task)
            write_city_outputs(args.results_root, args.network_version, output_folder, curves, summaries)
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
                write_city_outputs(args.results_root, args.network_version, output_folder, curves, summaries)
                all_summaries.extend(summaries)
                print(f"[OK {done}/{len(job_args)}] {city} -> {output_folder} summaries={len(summaries)}")

    out_dir = args.results_root / args.network_version
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sort by city for a deterministic combined summary (workers complete out of order).
    pd.DataFrame(all_summaries, columns=SUMMARY_COLUMNS).sort_values(
        ["city", "attack_type"]
    ).to_csv(out_dir / "all_cities_resilience_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Finished: cities={len(selected)} version={args.network_version} results_root={args.results_root}")


if __name__ == "__main__":
    main()
