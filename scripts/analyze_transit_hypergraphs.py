#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITY_CSV = PROJECT_ROOT / "metadata/cities_with_bus_and_metro.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results_build_transit_hypergraphs/default_run"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "results_analyze_transit_hypergraphs/default_run"
METRIC_COLUMNS = [
    "city",
    "output_folder",
    "n_nodes_total",
    "n_nodes_bus",
    "n_nodes_metro",
    "n_hyperedges_total",
    "n_hyperedges_bus",
    "n_hyperedges_metro",
    "n_transfers",
    "avg_hyperedge_size_total",
    "avg_hyperedge_size_bus",
    "avg_hyperedge_size_metro",
    "max_hyperedge_size_total",
    "avg_node_hyperdegree_total",
    "avg_node_hyperdegree_bus_nodes",
    "avg_node_hyperdegree_metro_nodes",
    "max_node_hyperdegree_total",
    "distinct_transfer_nodes_total",
    "distinct_transfer_node_ratio",
    "bus_nodes_with_transfer_ratio",
    "metro_nodes_with_transfer_ratio",
    "max_sequence",
    "avg_terminal_span",
    "n_projected_edges_total",
    "n_projected_edges_with_transfers",
    "avg_projected_out_degree",
    "avg_projected_in_degree",
    "avg_projected_out_degree_with_transfers",
    "avg_projected_in_degree_with_transfers",
]


@dataclass
class CityHypergraph:
    city: str
    output_folder: str
    nodes: pd.DataFrame
    hyperedges: pd.DataFrame
    hyperedge_nodes: pd.DataFrame
    transfers: pd.DataFrame
    metadata: Dict[str, int | str | float]
    node_to_edges: Dict[str, list[str]]
    edge_to_nodes: Dict[str, list[str]]
    transfer_pairs: set[tuple[str, str]]


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


def load_city_outputs(city_row: pd.Series, output_root: Path) -> dict[str, pd.DataFrame | str]:
    city = str(city_row["城市中文"])
    output_folder = str(city_row["公交文件夹"])
    city_dir = output_root / output_folder
    required = {
        "nodes": city_dir / "nodes.csv",
        "hyperedges": city_dir / "hyperedges.csv",
        "hyperedge_nodes": city_dir / "hyperedge_nodes.csv",
        "transfers": city_dir / "transfers.csv",
        "city_summary": city_dir / "city_summary.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing city output files:\n" + "\n".join(missing))
    return {
        "city": city,
        "output_folder": output_folder,
        "nodes": pd.read_csv(required["nodes"]),
        "hyperedges": pd.read_csv(required["hyperedges"]),
        "hyperedge_nodes": pd.read_csv(required["hyperedge_nodes"]),
        "transfers": pd.read_csv(required["transfers"]),
        "city_summary": pd.read_csv(required["city_summary"]),
    }


def build_city_hypergraph(city_row: pd.Series, output_root: Path) -> CityHypergraph:
    loaded = load_city_outputs(city_row, output_root)
    nodes = loaded["nodes"]
    hyperedges = loaded["hyperedges"]
    hyperedge_nodes = loaded["hyperedge_nodes"]
    transfers = loaded["transfers"]
    city_summary = loaded["city_summary"]

    memberships = hyperedge_nodes[["node_id", "edge_id"]].drop_duplicates()
    node_to_edges = memberships.groupby("node_id")["edge_id"].apply(list).to_dict()
    edge_to_nodes = (
        hyperedge_nodes.sort_values(["edge_id", "sequence", "node_id"], kind="stable")
        .groupby("edge_id")["node_id"]
        .apply(list)
        .to_dict()
    )
    transfer_pairs = set(zip(transfers["bus_node_id"], transfers["metro_node_id"]))
    metadata = city_summary.iloc[0].to_dict()

    return CityHypergraph(
        city=str(loaded["city"]),
        output_folder=str(loaded["output_folder"]),
        nodes=nodes,
        hyperedges=hyperedges,
        hyperedge_nodes=hyperedge_nodes,
        transfers=transfers,
        metadata=metadata,
        node_to_edges=node_to_edges,
        edge_to_nodes=edge_to_nodes,
        transfer_pairs=transfer_pairs,
    )


def compute_hyperedge_metrics(hg: CityHypergraph) -> Dict[str, float | int]:
    edge_sizes = hg.hyperedge_nodes.groupby(["edge_id", "mode"]).size().reset_index(name="edge_size")
    total = edge_sizes["edge_size"]
    bus = edge_sizes.loc[edge_sizes["mode"] == "bus", "edge_size"]
    metro = edge_sizes.loc[edge_sizes["mode"] == "metro", "edge_size"]
    return {
        "avg_hyperedge_size_total": float(total.mean()),
        "avg_hyperedge_size_bus": float(bus.mean()) if not bus.empty else 0.0,
        "avg_hyperedge_size_metro": float(metro.mean()) if not metro.empty else 0.0,
        "max_hyperedge_size_total": int(total.max()),
    }


def compute_node_participation_metrics(hg: CityHypergraph) -> Dict[str, float | int]:
    node_participation = (
        hg.hyperedge_nodes[["node_id", "edge_id"]]
        .drop_duplicates()
        .groupby("node_id")
        .size()
        .rename("hyperdegree")
        .reset_index()
    )
    node_participation = node_participation.merge(hg.nodes[["node_id", "mode"]], on="node_id", how="left")
    total = node_participation["hyperdegree"]
    bus = node_participation.loc[node_participation["mode"] == "bus", "hyperdegree"]
    metro = node_participation.loc[node_participation["mode"] == "metro", "hyperdegree"]
    return {
        "avg_node_hyperdegree_total": float(total.mean()),
        "avg_node_hyperdegree_bus_nodes": float(bus.mean()) if not bus.empty else 0.0,
        "avg_node_hyperdegree_metro_nodes": float(metro.mean()) if not metro.empty else 0.0,
        "max_node_hyperdegree_total": int(total.max()),
    }


def compute_transfer_metrics(hg: CityHypergraph) -> Dict[str, float | int]:
    transfer_nodes = set(hg.transfers["bus_node_id"]).union(set(hg.transfers["metro_node_id"]))
    bus_transfer_nodes = set(hg.transfers["bus_node_id"])
    metro_transfer_nodes = set(hg.transfers["metro_node_id"])
    n_nodes_total = len(hg.nodes)
    n_bus_nodes = int((hg.nodes["mode"] == "bus").sum())
    n_metro_nodes = int((hg.nodes["mode"] == "metro").sum())
    return {
        "distinct_transfer_nodes_total": len(transfer_nodes),
        "distinct_transfer_node_ratio": len(transfer_nodes) / n_nodes_total if n_nodes_total else 0.0,
        "bus_nodes_with_transfer_ratio": len(bus_transfer_nodes) / n_bus_nodes if n_bus_nodes else 0.0,
        "metro_nodes_with_transfer_ratio": len(metro_transfer_nodes) / n_metro_nodes if n_metro_nodes else 0.0,
    }


def compute_direction_metrics(hg: CityHypergraph) -> Dict[str, float | int]:
    edge_terminal_span = hg.hyperedge_nodes.groupby("edge_id")["sequence"].max()
    return {
        "max_sequence": int(hg.hyperedge_nodes["sequence"].max()),
        "avg_terminal_span": float(edge_terminal_span.mean()),
    }


def build_directed_projection(hg: CityHypergraph, include_transfers: bool = False) -> pd.DataFrame:
    records = []
    grouped = hg.hyperedge_nodes.sort_values(["edge_id", "sequence", "node_id"], kind="stable").groupby("edge_id")
    edge_modes = hg.hyperedges.set_index("edge_id")["mode"].to_dict()
    for edge_id, group in grouped:
        nodes = group["node_id"].tolist()
        city = group["city"].iloc[0]
        mode = edge_modes[edge_id]
        for idx in range(len(nodes) - 1):
            records.append(
                {
                    "source_node_id": nodes[idx],
                    "target_node_id": nodes[idx + 1],
                    "edge_id": edge_id,
                    "mode": mode,
                    "city": city,
                    "step_sequence": idx + 1,
                    "edge_kind": "intra_route",
                }
            )
    if include_transfers and not hg.transfers.empty:
        for _, row in hg.transfers.iterrows():
            records.append(
                {
                    "source_node_id": row["bus_node_id"],
                    "target_node_id": row["metro_node_id"],
                    "edge_id": row["transfer_id"],
                    "mode": "transfer",
                    "city": row["city"],
                    "step_sequence": 1,
                    "edge_kind": "transfer",
                }
            )
            records.append(
                {
                    "source_node_id": row["metro_node_id"],
                    "target_node_id": row["bus_node_id"],
                    "edge_id": row["transfer_id"],
                    "mode": "transfer",
                    "city": row["city"],
                    "step_sequence": 1,
                    "edge_kind": "transfer",
                }
            )
    return pd.DataFrame(records)


def compute_projection_metrics(hg: CityHypergraph, projected_edges: pd.DataFrame, projected_edges_with_transfers: pd.DataFrame) -> Dict[str, float | int]:
    def one_projection_metrics(df: pd.DataFrame) -> tuple[int, float, float]:
        if df.empty:
            return 0, 0.0, 0.0
        out_degree = df.groupby("source_node_id").size().reindex(hg.nodes["node_id"], fill_value=0)
        in_degree = df.groupby("target_node_id").size().reindex(hg.nodes["node_id"], fill_value=0)
        return len(df), float(out_degree.mean()), float(in_degree.mean())

    n_edges, avg_out, avg_in = one_projection_metrics(projected_edges)
    n_edges_t, avg_out_t, avg_in_t = one_projection_metrics(projected_edges_with_transfers)
    return {
        "n_projected_edges_total": n_edges,
        "n_projected_edges_with_transfers": n_edges_t,
        "avg_projected_out_degree": avg_out,
        "avg_projected_in_degree": avg_in,
        "avg_projected_out_degree_with_transfers": avg_out_t,
        "avg_projected_in_degree_with_transfers": avg_in_t,
    }


def analyze_city(city_row: pd.Series, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hg = build_city_hypergraph(city_row, output_root)
    projected_edges = build_directed_projection(hg, include_transfers=False)
    projected_edges_with_transfers = build_directed_projection(hg, include_transfers=True)
    metrics = {
        "city": hg.city,
        "output_folder": hg.output_folder,
        "n_nodes_total": len(hg.nodes),
        "n_nodes_bus": int((hg.nodes["mode"] == "bus").sum()),
        "n_nodes_metro": int((hg.nodes["mode"] == "metro").sum()),
        "n_hyperedges_total": len(hg.hyperedges),
        "n_hyperedges_bus": int((hg.hyperedges["mode"] == "bus").sum()),
        "n_hyperedges_metro": int((hg.hyperedges["mode"] == "metro").sum()),
        "n_transfers": len(hg.transfers),
    }
    metrics.update(compute_hyperedge_metrics(hg))
    metrics.update(compute_node_participation_metrics(hg))
    metrics.update(compute_transfer_metrics(hg))
    metrics.update(compute_direction_metrics(hg))
    metrics.update(compute_projection_metrics(hg, projected_edges, projected_edges_with_transfers))
    metrics_df = pd.DataFrame([{col: metrics[col] for col in METRIC_COLUMNS}])
    return metrics_df, projected_edges, projected_edges_with_transfers


def write_city_analysis(analysis_root: Path, output_folder: str, metrics_df: pd.DataFrame, projected_edges: pd.DataFrame, projected_edges_with_transfers: pd.DataFrame) -> None:
    city_dir = analysis_root / output_folder
    city_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(city_dir / "basic_metrics.csv", index=False, encoding="utf-8-sig")
    projected_edges.to_csv(city_dir / "directed_edges_projected.csv", index=False, encoding="utf-8-sig")
    projected_edges_with_transfers.to_csv(city_dir / "directed_edges_projected_with_transfers.csv", index=False, encoding="utf-8-sig")


def compare_transfer_rules(exact_root: Path, layered_root: Path, city_df: pd.DataFrame) -> pd.DataFrame:
    exact_metrics = []
    layered_metrics = []
    exact_meta = []
    layered_meta = []
    for _, city_row in city_df.iterrows():
        exact_metrics_df, _, _ = analyze_city(city_row, exact_root)
        layered_metrics_df, _, _ = analyze_city(city_row, layered_root)
        exact_metrics.append(exact_metrics_df)
        layered_metrics.append(layered_metrics_df)

        exact_summary = pd.read_csv(exact_root / str(city_row["公交文件夹"]) / "city_summary.csv")
        layered_summary = pd.read_csv(layered_root / str(city_row["公交文件夹"]) / "city_summary.csv")
        exact_meta.append(exact_summary)
        layered_meta.append(layered_summary)

    exact_all = pd.concat(exact_metrics, ignore_index=True)
    layered_all = pd.concat(layered_metrics, ignore_index=True)
    exact_meta_all = pd.concat(exact_meta, ignore_index=True)
    layered_meta_all = pd.concat(layered_meta, ignore_index=True)

    merged = exact_all.merge(
        layered_all,
        on=["city", "output_folder"],
        suffixes=("_exact", "_layered"),
        how="inner",
    )
    meta_cols = [
        "transfer_count_exact",
        "transfer_count_layered",
        "transfer_normalized_added_count",
        "transfer_spatial_added_count",
        "transfer_growth_abs",
        "transfer_growth_ratio",
        "spatial_threshold_m",
        "transfer_rule",
    ]
    if "transfer_walk_added_count" in exact_meta_all.columns and "transfer_walk_added_count" in layered_meta_all.columns:
        meta_cols.append("transfer_walk_added_count")

    merged = merged.merge(
        exact_meta_all[["city", "output_folder", *meta_cols]].rename(columns=lambda c: c if c in {"city", "output_folder"} else f"{c}_exact"),
        on=["city", "output_folder"],
        how="left",
    )
    merged = merged.merge(
        layered_meta_all[["city", "output_folder", *meta_cols]].rename(columns=lambda c: c if c in {"city", "output_folder"} else f"{c}_layered"),
        on=["city", "output_folder"],
        how="left",
    )

    for col in [
        "n_transfers",
        "distinct_transfer_nodes_total",
        "distinct_transfer_node_ratio",
        "bus_nodes_with_transfer_ratio",
        "metro_nodes_with_transfer_ratio",
        "n_projected_edges_total",
        "n_projected_edges_with_transfers",
        "avg_projected_out_degree_with_transfers",
        "avg_projected_in_degree_with_transfers",
        "transfer_count_layered",
        "transfer_normalized_added_count",
        "transfer_spatial_added_count",
        "transfer_growth_abs",
        "transfer_growth_ratio",
    ]:
        exact_col = f"{col}_exact"
        layered_col = f"{col}_layered"
        if exact_col in merged.columns and layered_col in merged.columns:
            merged[f"delta_{col}"] = merged[layered_col] - merged[exact_col]
    if "transfer_walk_added_count_exact" in merged.columns and "transfer_walk_added_count_layered" in merged.columns:
        merged["delta_transfer_walk_added_count"] = merged["transfer_walk_added_count_layered"] - merged["transfer_walk_added_count_exact"]
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze city hypergraph outputs and compute basic metrics.")
    parser.add_argument("--city", help="Chinese city name or folder name from the city inventory")
    parser.add_argument("--all", action="store_true", help="Analyze all cities from the city inventory")
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV, help="Path to cities_with_bus_and_metro.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Path to raw hypergraph output root")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT, help="Path to analysis output root")
    parser.add_argument("--compare-output-root", type=Path, help="Optional second output root for transfer-rule robustness comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_df = load_city_inventory(args.city_csv)
    selected = select_cities(city_df, args.city, args.all)
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    metrics_frames = []
    for _, city_row in selected.iterrows():
        metrics_df, projected_edges, projected_edges_with_transfers = analyze_city(city_row, args.output_root)
        output_folder = str(city_row["公交文件夹"])
        write_city_analysis(args.analysis_root, output_folder, metrics_df, projected_edges, projected_edges_with_transfers)
        metrics_frames.append(metrics_df)
        row = metrics_df.iloc[0]
        print(
            f"[OK] {row['city']} -> {output_folder} | "
            f"nodes={row['n_nodes_total']} edges={row['n_hyperedges_total']} "
            f"transfers={row['n_transfers']} projected_edges={row['n_projected_edges_total']} "
            f"projected_with_transfers={row['n_projected_edges_with_transfers']}"
        )

    all_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame(columns=METRIC_COLUMNS)
    all_metrics.to_csv(args.analysis_root / "all_cities_basic_metrics.csv", index=False, encoding="utf-8-sig")

    if args.compare_output_root:
        robustness = compare_transfer_rules(args.compare_output_root, args.output_root, selected)
        robustness.to_csv(args.analysis_root / "transfer_rule_robustness.csv", index=False, encoding="utf-8-sig")
        print(f"Wrote transfer-rule robustness comparison to {args.analysis_root / 'transfer_rule_robustness.csv'}")

    print(f"Finished analysis: cities={len(all_metrics)} analysis_root={args.analysis_root}")


if __name__ == "__main__":
    main()
