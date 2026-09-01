# Reproducibility package

This directory specifies the public computational workflow for the article “Contrasting
Resilience Diagnostics in Route-Preserving Multimodal Transit Hypergraphs”. The workflow
uses the CPTOND-2025 release cited in the manuscript, build rule
`v4_layered_transfer_with_walk_nearby`, and the frozen `walk_200m` network version.

The repository contains the analysis scripts, parameter record, random-seed ledger,
environment specification, transfer-matching definition, and an analysis-to-output map.
The raw CPTOND-2025 files are not duplicated because they are distributed separately. The
exact release and 45-city inventory are recorded in `DATA_VERSION.md` and `../metadata/`.
Derived node, route, transfer, simulation, and summary tables can be regenerated from the
public source data and are intentionally not tracked in this repository.

Run the stages in `WORKFLOW.md` in order. Stochastic stages expose a `--seed-base`
argument or use deterministic seed rules recorded in `SEEDS.csv`.
