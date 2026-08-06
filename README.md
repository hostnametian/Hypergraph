# Route-Preserving Multimodal Transit Hypergraphs

Anonymous code release accompanying the manuscript. This repository contains only the
manuscript-related analysis code and lightweight metadata; raw CPTOND-2025 files,
intermediate networks, simulation outputs, manuscript PDFs, and personal workspace files
are intentionally excluded.

## Layout

- `scripts/`: network construction, attacks, cascades, recovery, representation checks,
  clustering, and reviewer-requested statistical analyses.
- `metadata/`: 45-city inventory with repository-relative CPTOND-2025 paths.
- `reproducibility/`: environment, data-version, seed, transfer-audit and workflow records.

All paths are repository-relative. Place the public CPTOND-2025 archive at
`CPTOND-2025/` in the repository root, then follow `reproducibility/WORKFLOW.md`.
Create the environment with:

```bash
conda env create -f reproducibility/environment.yml
conda activate transit-hypergraph-repro
cd github
bash reproducibility/run_smoke_test.sh
```

The raw data are available from the public DOI stated in `reproducibility/DATA_VERSION.md`.
