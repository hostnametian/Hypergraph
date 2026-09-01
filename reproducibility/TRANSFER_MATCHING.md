# Transfer-matching audit trail

The complete city-level transfer-matching tables are generated as `transfers.csv` under
`results_build_transit_hypergraphs/{exact_name,walk_100m,walk_200m,walk_300m}/<city>/`.
Each row records the bus node, metro node, distance, and matching rule. The 45-city count
summary is generated as
`results_analyze_transit_hypergraphs/walk_200m/all_cities_basic_metrics.csv` with columns
`n_transfers`, `distinct_transfer_nodes_total`, and `distinct_transfer_node_ratio`.

These files are produced directly by `scripts/build_transit_hypergraphs.py` and
`scripts/analyze_transit_hypergraphs.py`; no manual matching or post-processing is used.
The derived CSV files are not tracked in this lightweight repository and can be regenerated
from the CPTOND-2025 source data identified in `DATA_VERSION.md`.
