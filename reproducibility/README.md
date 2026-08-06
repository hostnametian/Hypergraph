# Anonymous reproducibility package

This directory is the reviewer-facing reproducibility specification for the manuscript
“Contrasting Resilience Diagnostics in Route-Preserving Multimodal Transit Hypergraphs”.
It is intended to be deposited as an anonymous repository before peer review. The complete
workflow uses the CPTOND-2025 release cited in the manuscript, build rule `v4_layered_transfer_with_walk_nearby`,
and the frozen `walk_200m` network version.

The repository contains the analysis scripts, parameter manifest, random-seed ledger,
environment specification, transfer-matching definition, and a map from every manuscript
table/figure to the script and output used to generate it. The raw CPTOND-2025 files are
not duplicated here because they are a separately distributed dataset; the exact release,
city inventory and file hashes are recorded in `DATA_VERSION.md`. The derived node, route,
transfer and summary tables can be regenerated from those files.
The transfer-matching audit and paths to the complete derived matching tables are given in
`TRANSFER_MATCHING.md`.

Run the stages in the order given in `WORKFLOW.md`. All stochastic stages expose a
`--seed-base` argument and use deterministic seed offsets recorded in `SEEDS.csv`.
