# Transfer-matching audit trail

The complete city-level transfer matching tables are the derived `transfers.csv` files
under `results_build_transit_hypergraphs/{exact_name,walk_100m,walk_200m,walk_300m}/<city>/`.
Each row records the bus node, metro node, distance and matching rule. The 45-city count
summary is `results_analyze_transit_hypergraphs/walk_200m/all_cities_basic_metrics.csv`
(columns `n_transfers`, `distinct_transfer_nodes_total` and
`distinct_transfer_node_ratio`). These files are generated directly by
`build_transit_hypergraphs.py`; no manual matching or post-processing is used.

The anonymous deposit should include these derived CSVs (or a compressed archive of them)
alongside this audit file. The raw CPTOND-2025 geometries remain available from the cited
public DOI and are not copied into the code repository.
