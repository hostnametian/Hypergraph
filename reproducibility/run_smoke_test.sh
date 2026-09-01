#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# Lightweight release check: verify imports, metadata, and Python syntax.
PYTHON_BIN="${PYTHON:-python3}"
"$PYTHON_BIN" - <<'PY'
import numpy, pandas, scipy, sklearn
print('Core Python dependencies import successfully')
PY
"$PYTHON_BIN" scripts/build_transit_hypergraphs.py --help >/dev/null
"$PYTHON_BIN" -m compileall -q scripts
test -s metadata/cities_with_bus_and_metro.csv
echo 'Smoke test completed'
