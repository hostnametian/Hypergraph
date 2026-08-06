#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd


BUILD_RULE_VERSION = "v4_layered_transfer_with_walk_nearby"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITY_CSV = PROJECT_ROOT / "metadata/cities_with_bus_and_metro.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results_build_transit_hypergraphs/default_run"

NODE_COLUMNS = [
    "node_id",
    "city",
    "mode",
    "stop_id",
    "name_cn",
    "name_en",
    "city_code",
    "city_cn",
    "city_en",
]

HYPEREDGE_COLUMNS = [
    "edge_id",
    "city",
    "mode",
    "route_key",
    "route_id",
    "route_cn",
    "route_en",
    "route_type",
    "s_stop_cn",
    "e_stop_cn",
    "total_stop",
    "loop",
    "status",
    "start_time",
    "end_time",
    "basic_prc",
    "total_prc",
    "distance",
    "length",
    "company_cn",
    "company_en",
]

HYPEREDGE_NODE_COLUMNS = [
    "edge_id",
    "node_id",
    "sequence",
    "route_key",
    "route_id",
    "route_cn",
    "stop_id",
    "mode",
    "city",
]

TRANSFER_COLUMNS = [
    "transfer_id",
    "city",
    "bus_node_id",
    "metro_node_id",
    "bus_stop_id",
    "metro_stop_id",
    "bus_name_cn",
    "metro_name_cn",
    "bus_name_norm",
    "metro_name_norm",
    "distance_m",
    "match_rule",
    "match_confidence",
]

SEGMENT_COLUMNS = [
    "segment_id",
    "mode",
    "s_stopid",
    "e_stopid",
    "s_stop_cn",
    "e_stop_cn",
    "distance",
    "num",
    "city_cn",
    "city_en",
]

SUMMARY_COLUMNS = [
    "city",
    "output_folder",
    "bus_node_count",
    "metro_node_count",
    "bus_edge_count",
    "metro_edge_count",
    "transfer_count",
    "transfer_count_exact",
    "transfer_count_layered",
    "transfer_normalized_added_count",
    "transfer_spatial_added_count",
    "transfer_walk_added_count",
    "transfer_growth_abs",
    "transfer_growth_ratio",
    "bus_segment_count",
    "metro_segment_count",
    "build_rule_version",
    "transfer_rule",
    "spatial_threshold_m",
    "source_bus_folder",
    "source_metro_folder",
]

FAILURE_COLUMNS = ["city", "output_folder", "stage", "error"]


def folder_slug(folder_name: str) -> str:
    return folder_name.lower().replace(" ", "_")


def build_city_config(city_row: pd.Series) -> Dict[str, Path | str]:
    city_cn = str(city_row["城市中文"])
    bus_folder = str(city_row["公交文件夹"])
    metro_folder = str(city_row["地铁文件夹"])
    bus_dir = Path(str(city_row["公交文件路径"]))
    metro_dir = Path(str(city_row["地铁文件路径"]))
    if not bus_dir.is_absolute():
        bus_dir = PROJECT_ROOT / bus_dir
    if not metro_dir.is_absolute():
        metro_dir = PROJECT_ROOT / metro_dir
    bus_slug = folder_slug(bus_folder)
    metro_slug = folder_slug(metro_folder)
    return {
        "city": city_cn,
        "output_folder": bus_folder,
        "bus_dir": bus_dir,
        "metro_dir": metro_dir,
        "bus_stops_path": bus_dir / f"{bus_slug}_bus_stops.shp",
        "bus_routes_path": bus_dir / f"{bus_slug}_bus_routes.shp",
        "bus_segments_path": bus_dir / f"{bus_slug}_bus_segments.shp",
        "metro_stops_path": metro_dir / f"{metro_slug}_metro_stops.shp",
        "metro_routes_path": metro_dir / f"{metro_slug}_metro_routes.shp",
        "metro_segments_path": metro_dir / f"{metro_slug}_metro_segments.shp",
    }


def require_paths(config: Dict[str, Path | str]) -> None:
    required = [
        config["bus_stops_path"],
        config["bus_routes_path"],
        config["bus_segments_path"],
        config["metro_stops_path"],
        config["metro_routes_path"],
        config["metro_segments_path"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))


def read_city_layers(config: Dict[str, Path | str]) -> Dict[str, gpd.GeoDataFrame]:
    require_paths(config)
    return {
        "bus_stops": gpd.read_file(config["bus_stops_path"]),
        "bus_routes": gpd.read_file(config["bus_routes_path"]),
        "bus_segments": gpd.read_file(config["bus_segments_path"]),
        "metro_stops": gpd.read_file(config["metro_stops_path"]),
        "metro_routes": gpd.read_file(config["metro_routes_path"]),
        "metro_segments": gpd.read_file(config["metro_segments_path"]),
    }


def build_nodes(bus_stops: gpd.GeoDataFrame, metro_stops: gpd.GeoDataFrame, city_label: str) -> pd.DataFrame:
    def one_mode_nodes(stops: gpd.GeoDataFrame, mode: str) -> pd.DataFrame:
        nodes = stops[["stop_id", "name_cn", "name_en", "city_code", "city_cn", "city_en"]].copy()
        nodes = nodes.drop_duplicates(subset=["stop_id"]).copy()
        nodes["city"] = city_label
        nodes["mode"] = mode
        nodes["node_id"] = mode + "_" + nodes["stop_id"].astype(str)
        return nodes[NODE_COLUMNS].sort_values(["mode", "stop_id"], kind="stable").reset_index(drop=True)

    nodes = pd.concat([
        one_mode_nodes(bus_stops, "bus"),
        one_mode_nodes(metro_stops, "metro"),
    ], ignore_index=True)
    if nodes[["mode", "stop_id"]].duplicated().any():
        raise ValueError("Duplicate (mode, stop_id) found in nodes table")
    if nodes["node_id"].duplicated().any():
        raise ValueError("Duplicate node_id found in nodes table")
    if nodes["stop_id"].isna().any():
        raise ValueError("Null stop_id found in nodes table")
    return nodes


def build_hyperedges(bus_routes: gpd.GeoDataFrame, metro_routes: gpd.GeoDataFrame, bus_stops: gpd.GeoDataFrame, metro_stops: gpd.GeoDataFrame, city_label: str) -> pd.DataFrame:
    def one_mode_edges(routes: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, mode: str) -> pd.DataFrame:
        route_keys = stops[["route_id", "route_cn"]].drop_duplicates().copy()
        route_keys["route_key"] = route_keys["route_id"].astype(str) + "||" + route_keys["route_cn"].astype(str)
        route_attrs = routes[
            [
                "route_cn", "route_en", "route_type", "s_stop_cn", "e_stop_cn", "total_stop",
                "loop", "status", "start_time", "end_time", "basic_prc", "total_prc",
                "distance", "length", "company_cn", "company_en",
            ]
        ].drop_duplicates(subset=["route_cn"]).copy()
        edges = route_keys.merge(route_attrs, on="route_cn", how="left", validate="many_to_one")
        edges["city"] = city_label
        edges["mode"] = mode
        edges["edge_id"] = mode + "_route_" + edges["route_key"].str.replace("||", "__", regex=False)
        return edges[HYPEREDGE_COLUMNS].sort_values(["mode", "route_key"], kind="stable").reset_index(drop=True)

    hyperedges = pd.concat([
        one_mode_edges(bus_routes, bus_stops, "bus"),
        one_mode_edges(metro_routes, metro_stops, "metro"),
    ], ignore_index=True)
    if hyperedges[["mode", "route_key"]].duplicated().any():
        raise ValueError("Duplicate (mode, route_key) found in hyperedges table")
    if hyperedges["edge_id"].duplicated().any():
        raise ValueError("Duplicate edge_id found in hyperedges table")
    return hyperedges


def build_hyperedge_nodes(bus_stops: gpd.GeoDataFrame, metro_stops: gpd.GeoDataFrame, nodes: pd.DataFrame, hyperedges: pd.DataFrame, city_label: str) -> pd.DataFrame:
    node_map = dict(zip(nodes["mode"] + "|" + nodes["stop_id"].astype(str), nodes["node_id"]))
    edge_map = dict(zip(hyperedges["mode"] + "|" + hyperedges["route_key"].astype(str), hyperedges["edge_id"]))

    def one_mode_memberships(stops: gpd.GeoDataFrame, mode: str) -> pd.DataFrame:
        memberships = stops[["route_id", "route_cn", "stop_id", "sequence"]].copy()
        if memberships["sequence"].isna().any():
            raise ValueError(f"Null sequence found in {mode} memberships")
        memberships["route_key"] = memberships["route_id"].astype(str) + "||" + memberships["route_cn"].astype(str)
        memberships["city"] = city_label
        memberships["mode"] = mode
        memberships["node_id"] = (mode + "|" + memberships["stop_id"].astype(str)).map(node_map)
        memberships["edge_id"] = (mode + "|" + memberships["route_key"].astype(str)).map(edge_map)
        if memberships["node_id"].isna().any():
            raise ValueError(f"Unmapped node_id in {mode} memberships")
        if memberships["edge_id"].isna().any():
            raise ValueError(f"Unmapped edge_id in {mode} memberships")
        return memberships[HYPEREDGE_NODE_COLUMNS].sort_values(["mode", "route_key", "sequence", "stop_id"], kind="stable").reset_index(drop=True)

    hyperedge_nodes = pd.concat([
        one_mode_memberships(bus_stops, "bus"),
        one_mode_memberships(metro_stops, "metro"),
    ], ignore_index=True)
    counts = hyperedge_nodes.groupby("edge_id").size()
    short_edges = counts[counts < 2]
    if not short_edges.empty:
        raise ValueError(f"Hyperedges with fewer than 2 nodes found: {short_edges.index.tolist()[:10]}")
    return hyperedge_nodes


def normalize_station_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\([^)]*\)", "", text)
    for suffix in ["地铁站", "公交站", "站"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_transfers(nodes: pd.DataFrame, bus_stops: gpd.GeoDataFrame, metro_stops: gpd.GeoDataFrame, city_label: str, transfer_rule: str, spatial_threshold_m: float) -> tuple[pd.DataFrame, dict]:
    bus_nodes = nodes[nodes["mode"] == "bus"][["node_id", "stop_id", "name_cn", "city"]].copy()
    metro_nodes = nodes[nodes["mode"] == "metro"][["node_id", "stop_id", "name_cn", "city"]].copy()

    bus_geo = bus_stops[["stop_id", "name_cn", "geometry"]].drop_duplicates(subset=["stop_id"]).copy()
    metro_geo = metro_stops[["stop_id", "name_cn", "geometry"]].drop_duplicates(subset=["stop_id"]).copy()
    bus_geo["bus_name_norm"] = bus_geo["name_cn"].map(normalize_station_name)
    metro_geo["metro_name_norm"] = metro_geo["name_cn"].map(normalize_station_name)
    bus_geo["bus_lon"] = bus_geo.geometry.x
    bus_geo["bus_lat"] = bus_geo.geometry.y
    metro_geo["metro_lon"] = metro_geo.geometry.x
    metro_geo["metro_lat"] = metro_geo.geometry.y

    bus_base = bus_nodes.merge(bus_geo[["stop_id", "bus_name_norm", "bus_lon", "bus_lat"]], on="stop_id", how="left")
    metro_base = metro_nodes.merge(metro_geo[["stop_id", "metro_name_norm", "metro_lon", "metro_lat"]], on="stop_id", how="left")

    exact = bus_base.merge(metro_base, on="city", how="inner", suffixes=("_bus", "_metro"))
    exact = exact[exact["name_cn_bus"] == exact["name_cn_metro"]].copy()
    exact["distance_m"] = pd.NA
    exact["match_rule"] = "exact_name"
    exact["match_confidence"] = "high"
    exact["pair_key"] = exact["node_id_bus"].astype(str) + "||" + exact["node_id_metro"].astype(str)
    exact_pairs = set(exact["pair_key"].drop_duplicates())

    pair_cols = ["node_id_bus", "node_id_metro"]
    normalized = pd.DataFrame(columns=exact.columns)
    spatial = pd.DataFrame(columns=exact.columns)
    walk = pd.DataFrame(columns=exact.columns)
    spatial_pairs: set[str] = set()

    if transfer_rule == "layered":
        normalized = bus_base.merge(metro_base, on="city", how="inner", suffixes=("_bus", "_metro"))
        normalized = normalized[(normalized["bus_name_norm"] != "") & (normalized["bus_name_norm"] == normalized["metro_name_norm"])].copy()
        normalized["pair_key"] = normalized["node_id_bus"].astype(str) + "||" + normalized["node_id_metro"].astype(str)
        normalized = normalized[~normalized["pair_key"].isin(exact_pairs)].copy()
        normalized["distance_m"] = pd.NA
        normalized["match_rule"] = "normalized_name"
        normalized["match_confidence"] = "medium"

        spatial = normalized.drop_duplicates(subset=pair_cols).copy()
        spatial["distance_m"] = spatial.apply(lambda r: haversine_m(r["bus_lon"], r["bus_lat"], r["metro_lon"], r["metro_lat"]), axis=1)
        spatial = spatial[spatial["distance_m"] <= spatial_threshold_m].copy()
        spatial = spatial[~spatial["pair_key"].isin(exact_pairs)].copy()
        spatial["match_rule"] = "spatial_name"
        spatial["match_confidence"] = "medium"
        spatial_pairs = set(spatial["pair_key"].drop_duplicates())
        normalized = normalized[~normalized["pair_key"].isin(spatial_pairs)].copy()

        walk = bus_base.merge(metro_base, on="city", how="inner", suffixes=("_bus", "_metro"))
        walk["pair_key"] = walk["node_id_bus"].astype(str) + "||" + walk["node_id_metro"].astype(str)
        walk = walk.drop_duplicates(subset=pair_cols).copy()
        walk["distance_m"] = walk.apply(lambda r: haversine_m(r["bus_lon"], r["bus_lat"], r["metro_lon"], r["metro_lat"]), axis=1)
        excluded_pairs = exact_pairs.union(spatial_pairs).union(set(normalized["pair_key"].drop_duplicates()))
        walk = walk[(walk["distance_m"] <= spatial_threshold_m) & (~walk["pair_key"].isin(excluded_pairs))].copy()
        walk["match_rule"] = "walk_nearby"
        walk["match_confidence"] = "low"

    combined = pd.concat([exact, normalized, spatial, walk], ignore_index=True)
    combined = combined.drop_duplicates(subset=pair_cols, keep="first").copy()
    combined = combined.rename(columns={
        "node_id_bus": "bus_node_id",
        "node_id_metro": "metro_node_id",
        "stop_id_bus": "bus_stop_id",
        "stop_id_metro": "metro_stop_id",
        "name_cn_bus": "bus_name_cn",
        "name_cn_metro": "metro_name_cn",
    })
    combined["city"] = city_label
    combined["transfer_id"] = [f"transfer_{i + 1:04d}" for i in range(len(combined))]
    combined = combined[[
        "transfer_id", "city", "bus_node_id", "metro_node_id", "bus_stop_id", "metro_stop_id",
        "bus_name_cn", "metro_name_cn", "bus_name_norm", "metro_name_norm", "distance_m", "match_rule", "match_confidence"
    ]].copy()
    combined = combined[TRANSFER_COLUMNS].sort_values(["match_rule", "bus_name_cn", "bus_node_id", "metro_node_id"], kind="stable").reset_index(drop=True)

    diagnostics = {
        "transfer_count_exact": len(exact.drop_duplicates(subset=pair_cols)),
        "transfer_count_layered": len(combined),
        "transfer_normalized_added_count": len(normalized.drop_duplicates(subset=pair_cols)),
        "transfer_spatial_added_count": len(spatial.drop_duplicates(subset=pair_cols)),
        "transfer_walk_added_count": len(walk.drop_duplicates(subset=pair_cols)),
    }
    diagnostics["transfer_growth_abs"] = diagnostics["transfer_count_layered"] - diagnostics["transfer_count_exact"]
    diagnostics["transfer_growth_ratio"] = diagnostics["transfer_growth_abs"] / diagnostics["transfer_count_exact"] if diagnostics["transfer_count_exact"] else 0.0
    return combined, diagnostics


def build_segments(segments: gpd.GeoDataFrame, mode: str) -> pd.DataFrame:
    out = segments[["s_stopid", "e_stopid", "s_stop_cn", "e_stop_cn", "distance", "num", "city_cn", "city_en"]].copy()
    out["mode"] = mode
    out["segment_id"] = [f"{mode}_segment_{i + 1:06d}" for i in range(len(out))]
    return out[SEGMENT_COLUMNS]


def build_summary(nodes: pd.DataFrame, hyperedges: pd.DataFrame, transfers: pd.DataFrame, transfer_diagnostics: dict, bus_segments: pd.DataFrame, metro_segments: pd.DataFrame, city_label: str, output_folder: str, config: Dict[str, Path | str], transfer_rule: str, spatial_threshold_m: float) -> pd.DataFrame:
    summary = pd.DataFrame([{
        "city": city_label,
        "output_folder": output_folder,
        "bus_node_count": int((nodes["mode"] == "bus").sum()),
        "metro_node_count": int((nodes["mode"] == "metro").sum()),
        "bus_edge_count": int((hyperedges["mode"] == "bus").sum()),
        "metro_edge_count": int((hyperedges["mode"] == "metro").sum()),
        "transfer_count": len(transfers),
        "transfer_count_exact": transfer_diagnostics["transfer_count_exact"],
        "transfer_count_layered": transfer_diagnostics["transfer_count_layered"],
        "transfer_normalized_added_count": transfer_diagnostics["transfer_normalized_added_count"],
        "transfer_spatial_added_count": transfer_diagnostics["transfer_spatial_added_count"],
        "transfer_walk_added_count": transfer_diagnostics["transfer_walk_added_count"],
        "transfer_growth_abs": transfer_diagnostics["transfer_growth_abs"],
        "transfer_growth_ratio": transfer_diagnostics["transfer_growth_ratio"],
        "bus_segment_count": len(bus_segments),
        "metro_segment_count": len(metro_segments),
        "build_rule_version": BUILD_RULE_VERSION,
        "transfer_rule": transfer_rule,
        "spatial_threshold_m": spatial_threshold_m,
        "source_bus_folder": str(config["bus_dir"]),
        "source_metro_folder": str(config["metro_dir"]),
    }])
    return summary[SUMMARY_COLUMNS]


def validate_outputs(nodes: pd.DataFrame, hyperedges: pd.DataFrame, hyperedge_nodes: pd.DataFrame, transfers: pd.DataFrame) -> None:
    node_ids = set(nodes["node_id"])
    edge_ids = set(hyperedges["edge_id"])
    if set(hyperedge_nodes["node_id"]) - node_ids:
        raise ValueError("Membership table references unknown node_ids")
    if set(hyperedge_nodes["edge_id"]) - edge_ids:
        raise ValueError("Membership table references unknown edge_ids")
    if hyperedge_nodes["sequence"].isna().any():
        raise ValueError("Membership table contains null sequence values")
    if not transfers.empty:
        if not transfers["bus_node_id"].str.startswith("bus_").all():
            raise ValueError("Transfers contain non-bus bus_node_id values")
        if not transfers["metro_node_id"].str.startswith("metro_").all():
            raise ValueError("Transfers contain non-metro metro_node_id values")


def write_outputs(output_dir: Path, outputs: Dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def read_city_inventory(city_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(city_csv_path, encoding="utf-8-sig")


def select_cities(city_df: pd.DataFrame, city: str | None, run_all: bool) -> pd.DataFrame:
    if run_all:
        return city_df.copy()
    if not city:
        raise ValueError("Provide --city <name> or use --all")
    mask = city_df["城市中文"].astype(str).eq(city) | city_df["公交文件夹"].astype(str).eq(city) | city_df["地铁文件夹"].astype(str).eq(city)
    selected = city_df[mask].copy()
    if selected.empty:
        raise ValueError(f"City not found in inventory: {city}")
    return selected


def build_city(city_row: pd.Series, output_root: Path, transfer_rule: str, spatial_threshold_m: float) -> pd.DataFrame:
    config = build_city_config(city_row)
    city_label = str(config["city"])
    output_folder = str(config["output_folder"])
    output_dir = output_root / output_folder
    layers = read_city_layers(config)
    nodes = build_nodes(layers["bus_stops"], layers["metro_stops"], city_label)
    hyperedges = build_hyperedges(layers["bus_routes"], layers["metro_routes"], layers["bus_stops"], layers["metro_stops"], city_label)
    hyperedge_nodes = build_hyperedge_nodes(layers["bus_stops"], layers["metro_stops"], nodes, hyperedges, city_label)
    transfers, transfer_diagnostics = build_transfers(nodes, layers["bus_stops"], layers["metro_stops"], city_label, transfer_rule, spatial_threshold_m)
    bus_segments = build_segments(layers["bus_segments"], "bus")
    metro_segments = build_segments(layers["metro_segments"], "metro")
    summary = build_summary(nodes, hyperedges, transfers, transfer_diagnostics, bus_segments, metro_segments, city_label, output_folder, config, transfer_rule, spatial_threshold_m)
    validate_outputs(nodes, hyperedges, hyperedge_nodes, transfers)
    write_outputs(output_dir, {
        "nodes.csv": nodes,
        "hyperedges.csv": hyperedges,
        "hyperedge_nodes.csv": hyperedge_nodes,
        "transfers.csv": transfers,
        "city_summary.csv": summary,
        "bus_segments.csv": bus_segments,
        "metro_segments.csv": metro_segments,
    })
    return summary


def write_aggregate_reports(output_root: Path, summaries: list[pd.DataFrame], failures: list[dict]) -> None:
    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=SUMMARY_COLUMNS)
    all_summary.to_csv(output_root / "all_cities_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(output_root / "build_failures.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build directed unweighted transit hypergraph tables for one or more cities.")
    parser.add_argument("--city", help="Chinese city name or folder name from the city inventory")
    parser.add_argument("--all", action="store_true", help="Build all cities from the city inventory")
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV, help="Path to cities_with_bus_and_metro.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--transfer-rule", choices=["exact", "layered"], default="exact")
    parser.add_argument("--spatial-threshold", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_df = read_city_inventory(args.city_csv)
    selected = select_cities(city_df, args.city, args.all)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    failures: list[dict] = []
    for _, city_row in selected.iterrows():
        city_label = str(city_row["城市中文"])
        output_folder = str(city_row["公交文件夹"])
        try:
            summary = build_city(city_row, args.output_root, args.transfer_rule, args.spatial_threshold)
            summaries.append(summary)
            row = summary.iloc[0]
            print(
                f"[OK] {city_label} -> {output_folder} | "
                f"bus_nodes={row['bus_node_count']} metro_nodes={row['metro_node_count']} "
                f"bus_edges={row['bus_edge_count']} metro_edges={row['metro_edge_count']} "
                f"transfers={row['transfer_count']} rule={row['transfer_rule']}"
            )
        except Exception as exc:
            failures.append({"city": city_label, "output_folder": output_folder, "stage": "build_city", "error": str(exc)})
            print(f"[FAIL] {city_label} -> {output_folder} | {exc}")
    write_aggregate_reports(args.output_root, summaries, failures)
    print(f"Finished builds: success={len(summaries)} failure={len(failures)} output_root={args.output_root}")


if __name__ == "__main__":
    main()
