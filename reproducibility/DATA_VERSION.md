# Data and preprocessing version

- Source dataset: CPTOND-2025, Wang, Wei, Guan et al., *Scientific Data* 13, 188 (2026).
- Public archive: https://doi.org/10.6084/m9.figshare.29377427
- Expected local directory: `CPTOND-2025/` at the repository root (not redistributed).
- City inventory: `metadata/cities_with_bus_and_metro.csv` (45 cities with both modes).
- Network version used for the manuscript: `walk_200m`.
- Build rule: `v4_layered_transfer_with_walk_nearby` (declared in
  `scripts/build_transit_hypergraphs.py`).
- Transfer inference: exact stop-name matching plus a 200 m geometric-proximity threshold;
  inferred links represent potential proximity coupling (PPCR), not observed transfers.
- Alternative preprocessing versions: `exact_name`, `walk_100m`, and `walk_300m`.

The DOI and edition above identify the source release used by the workflow. Raw data and
locally generated file hashes are not tracked in this code repository. Source files remain
subject to the dataset publisher's license; the original repository code and documentation
are covered by the repository `LICENSE`.
