# Route-Preserving Multimodal Transit Hypergraphs

Public reproducibility repository accompanying the article “Contrasting Resilience
Diagnostics in Route-Preserving Multimodal Transit Hypergraphs”. This repository contains
the manuscript-related analysis code and lightweight metadata. Raw CPTOND-2025 files,
intermediate networks, simulation outputs, journal artwork, manuscript PDFs, and personal
workspace files are intentionally excluded.

## Layout

- `scripts/`: network construction, attacks, cascades, recovery, representation checks,
  clustering, and statistical analyses.
- `metadata/`: the 45-city inventory with repository-relative CPTOND-2025 paths.
- `reproducibility/`: environment, data-version, seed, transfer-matching, workflow, and
  analysis-to-output records.

## Quick start

Clone the repository, place the public CPTOND-2025 archive in `CPTOND-2025/` at the
repository root, and follow `reproducibility/WORKFLOW.md`.

```bash
git clone https://github.com/hostnametian/Hypergraph.git
cd Hypergraph
conda env create -f reproducibility/environment.yml
conda activate transit-hypergraph-repro
bash reproducibility/run_smoke_test.sh
```

The source data release is identified in `reproducibility/DATA_VERSION.md`. Derived node,
route, transfer, simulation, and summary tables are generated locally by the workflow and
are not tracked in this repository.

## License

The original code and repository documentation are released under the MIT License; see
`LICENSE`. CPTOND-2025 remains subject to the license stated by its data publisher.
