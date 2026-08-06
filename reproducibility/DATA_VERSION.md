# Data and preprocessing version

- Source dataset: CPTOND-2025, Wang, Wei, Guan et al., Scientific Data 13, 188 (2026).
- Local source directory: `CPTOND-2025/` (not redistributed in the anonymous code bundle).
- City inventory: `cities_with_bus_and_metro.csv` (45 cities with both modes).
- Network version used for the manuscript: `walk_200m`.
- Build rule: `v4_layered_transfer_with_walk_nearby` (declared in `build_transit_hypergraphs.py`).
- Transfer inference: exact stop-name matching plus geometric proximity threshold 200 m;
  inferred links are labelled potential proximity coupling (PPCR), not observed transfers.
- Alternative preprocessing versions: `exact_name`, `walk_100m`, and `walk_300m`.

Before release, the repository maintainer should add SHA-256 hashes for the downloaded
CPTOND-2025 archive and each source file in this document. Raw files remain subject to the
dataset licence; derived tables and all code are released with the anonymous repository.
