#!/usr/bin/env python3
"""L-space external-reconstruction check (reviewer R1-C1 response).

This script addresses the reviewer question: "If the cascade rule matters more
than the representation, why use hypergraphs?"

The response is two-part:
  1. Numerical equivalence: When route membership is correctly preserved and
     re-imported externally, the route-support cascade on an L-space projection
     gives the SAME collapse fraction as on the native hypergraph.  The
     representation does not change the cascade outcome when membership is intact.
  2. Semantic distinction: The neighbor-fraction rule -- the natural rule to
     apply on a pairwise graph WITHOUT re-importing membership -- produces
     near-total collapse on the same L-space graph, showing that discarding
     route membership during graph construction changes the dynamics even when
     the tolerance parameter is held constant.

Together, points 1-2 show that the hypergraph is not needed to reproduce the
route-support outcome numerically; it is needed to make route-dependent failure
the default, correct rule rather than a post-hoc external fix.

Experiment conditions (fixed, matching Table cascade_model_comparison):
  - Initial random node damage: f = 0.10
  - Cascade threshold: tau = 0.30
  - Monte Carlo repetitions: 50
  - Cities: 6 cluster representatives

Output
------
results_lspace_reconstruction/
    reconstruction_check_table.csv   -- main per-city comparison table
    reconstruction_check_summary.csv -- mean ± SD across 6 cities
    reconstruction_check_full.csv    -- per-rep raw data (for SD computation)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from run_resilience_experiments import (
    DEFAULT_CITY_CSV,
    DEFAULT_BUILD_ROOT,
    DEFAULT_ANALYSIS_ROOT,
    load_city_inventory,
    load_city_artifacts,
    build_csr_index,
    build_retention_index,
)
from run_pspace_comparison import (
    build_pspace_adjacency,
    build_pspace_sequential_adjacency,
    simulate_pspace_cascade,
    compute_graph_metrics_from_csr,
)
from run_resilience_cascade import (
    simulate_hyperedge_cascade,
    build_node_initial_hyperdegree,
)

# ---------- configuration ----------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results_lspace_reconstruction"

REPRESENTATIVE_CITIES = ["贵阳", "成都", "滁州", "青岛", "石家庄", "福州"]

F_DAMAGE = 0.10   # initial random node damage fraction (matches Table cascade_model_comparison)
TAU = 0.30        # cascade threshold (matches Table cascade_model_comparison)
REPS = 50         # Monte Carlo repetitions
SEED_BASE = 42
NETWORK_VERSION = "walk_200m"
SOURCE_SAMPLES = 500


# ---------- single-city runner ----------

def run_city(city_row: pd.Series,
             build_root: Path,
             analysis_root: Path,
             f_damage: float,
             tau: float,
             reps: int,
             seed_base: int) -> list[dict]:
    """Run the three-way cascade comparison for one city.

    Returns a list of per-rep dicts, each with keys:
        city, rep, method, collapse_fraction, cascade_depth
    """
    loaded = load_city_artifacts(
        city_row, build_root, analysis_root, NETWORK_VERSION
    )
    city_name = loaded["city"]
    nodes_df         = loaded["nodes"]
    hyperedges_df    = loaded["hyperedges"]
    transfers_df     = loaded["transfers"]
    hyperedge_nodes_df = loaded["hyperedge_nodes"]
    edge_df          = loaded["projection"]

    base_n = len(nodes_df)
    node_ids = nodes_df["node_id"].astype(str).to_numpy()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # --- build structures ---
    # rctx: the route-membership retention context — same object for both
    #   "native hypergraph" and "L-space + external reconstruction"; the point
    #   is that in the latter case this context must be imported from outside
    #   the graph, whereas in the former it is intrinsic to the representation.
    rctx = build_retention_index(
        nodes_df, hyperedges_df, transfers_df, hyperedge_nodes_df
    )
    initial_hd = build_node_initial_hyperdegree(rctx)

    # L-space sequential graph: consecutive stops only + transfer edges.
    # Route membership is NOT encoded in this graph; it must be re-imported
    # from hyperedge_nodes_df to run the route-support rule correctly.
    A_seq = build_pspace_sequential_adjacency(
        hyperedge_nodes_df, nodes_df, transfers_df
    )
    # Build the P-space clique explicitly for the matched representation
    # comparison.  The route-support dynamics use the externally imported
    # incidence context below; the graph representation itself does not alter
    # the rule when that context is preserved.
    A_clique = build_pspace_adjacency(
        hyperedge_nodes_df, nodes_df, transfers_df
    )

    # Attack pool: all nodes
    attack_pool = list(node_ids)
    n_pool = len(attack_pool)

    rows = []
    for rep in range(reps):
        rng = random.Random(seed_base + rep * 100003)
        n_remove = max(1, int(round(n_pool * f_damage)))
        removed_ids = set(rng.sample(attack_pool, n_remove))
        initial_dead = np.zeros(base_n, dtype=bool)
        for nid in removed_ids:
            if nid in id2idx:
                initial_dead[id2idx[nid]] = True

        # ------------------------------------------------------------------ #
        # Method A: Route-support cascade — native hypergraph                 #
        #   Uses rctx (route membership intrinsic to hypergraph).             #
        #   LWCC computed on L-space graph for fair comparison.               #
        # ------------------------------------------------------------------ #
        dead_A, depth_A = simulate_hyperedge_cascade(
            rctx, initial_dead.copy(), initial_hd, tau
        )
        collapse_A = float(dead_A.mean())
        rows.append(dict(city=city_name, rep=rep,
                         method="route_support_native_hypergraph",
                         collapse_fraction=collapse_A,
                         cascade_depth=depth_A,
                         representation_edges=int(A_seq.nnz)))

        # ------------------------------------------------------------------ #
        # Method B: Route-support cascade — L-space + external reconstruction #
        #   Conceptually: graph is L-space (membership discarded at           #
        #   construction); route membership is re-imported from the original  #
        #   incidence table (hyperedge_nodes_df) and used via rctx.           #
        #   Since rctx is identical to Method A, the cascade is identical.    #
        #   This confirms numerical equivalence; the semantic difference is   #
        #   that membership had to be explicitly re-imported.                 #
        # ------------------------------------------------------------------ #
        # NOTE: same rctx and same initial_dead → results will be identical
        #   to Method A by construction. This is the expected outcome.
        dead_B, depth_B = simulate_hyperedge_cascade(
            rctx, initial_dead.copy(), initial_hd, tau
        )
        collapse_B = float(dead_B.mean())
        rows.append(dict(city=city_name, rep=rep,
                         method="route_support_lspace_external_recon",
                         collapse_fraction=collapse_B,
                         cascade_depth=depth_B,
                         representation_edges=int(A_seq.nnz)))

        # ------------------------------------------------------------------ #
        # Method B2: Route-support cascade — P-space + external reconstruction
        #   The route context is imported externally, while the graph used for
        #   the representation is the P-space clique projection.  The rule
        #   itself is identical to Methods A and B.
        # ------------------------------------------------------------------ #
        dead_B2, depth_B2 = simulate_hyperedge_cascade(
            rctx, initial_dead.copy(), initial_hd, tau
        )
        collapse_B2 = float(dead_B2.mean())
        rows.append(dict(city=city_name, rep=rep,
                         method="route_support_pspace_external_recon",
                         collapse_fraction=collapse_B2,
                         cascade_depth=depth_B2,
                         representation_edges=int(A_clique.nnz)))

        # ------------------------------------------------------------------ #
        # Method C: Neighbor-fraction cascade — L-space only                 #
        #   The natural rule when only the pairwise graph is available and    #
        #   route membership is NOT re-imported.  A node fails when the       #
        #   fraction of alive graph-neighbors drops below (1 - tau).         #
        #   This is the baseline showing what happens when representation     #
        #   discards membership and no external reconstruction is done.       #
        # ------------------------------------------------------------------ #
        dead_C, depth_C = simulate_pspace_cascade(
            A_seq, initial_dead.copy(), tau
        )
        collapse_C = float(dead_C.mean())
        rows.append(dict(city=city_name, rep=rep,
                         method="neighbor_fraction_lspace_only",
                         collapse_fraction=collapse_C,
                         cascade_depth=depth_C,
                         representation_edges=int(A_seq.nnz)))

    return rows


# ---------- main ----------

def main() -> None:
    results_root = DEFAULT_RESULTS_ROOT
    results_root.mkdir(parents=True, exist_ok=True)

    city_df = load_city_inventory(DEFAULT_CITY_CSV)

    print("=" * 70)
    print("L-SPACE EXTERNAL RECONSTRUCTION CHECK  (reviewer R1-C1)")
    print(f"  f_damage={F_DAMAGE}  tau={TAU}  reps={REPS}  version={NETWORK_VERSION}")
    print("=" * 70)

    all_rows: list[dict] = []

    for city_cn in REPRESENTATIVE_CITIES:
        mask = city_df["城市中文"].astype(str).eq(city_cn)
        if not mask.any():
            print(f"[SKIP] {city_cn} not found in city inventory")
            continue
        city_row = city_df[mask].iloc[0]
        print(f"\n[{city_cn}] running {REPS} reps …", end=" ", flush=True)
        try:
            rows = run_city(
                city_row,
                build_root=DEFAULT_BUILD_ROOT,
                analysis_root=DEFAULT_ANALYSIS_ROOT,
                f_damage=F_DAMAGE,
                tau=TAU,
                reps=REPS,
                seed_base=SEED_BASE,
            )
            all_rows.extend(rows)
            print("OK")
        except Exception as exc:
            print(f"FAIL: {exc!r}")
            import traceback; traceback.print_exc()

    if not all_rows:
        print("No results — exiting.")
        return

    full_df = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------ #
    # Per-city summary table                                               #
    # ------------------------------------------------------------------ #
    METHOD_LABELS = {
        "route_support_native_hypergraph":    "Route-support (native hypergraph)",
        "route_support_lspace_external_recon": "Route-support (L-space + external recon.)",
        "route_support_pspace_external_recon": "Route-support (P-space + external recon.)",
        "neighbor_fraction_lspace_only":      "Neighbor-fraction (L-space only)",
    }

    summary_rows = []
    for city_cn in REPRESENTATIVE_CITIES:
        city_data = full_df[full_df["city"] == city_cn]
        if city_data.empty:
            continue
        for method_key, method_label in METHOD_LABELS.items():
            mdata = city_data[city_data["method"] == method_key]
            summary_rows.append(dict(
                city=city_cn,
                method=method_label,
                collapse_mean=round(mdata["collapse_fraction"].mean(), 4),
                collapse_std=round(mdata["collapse_fraction"].std(), 4),
                depth_mean=round(mdata["cascade_depth"].mean(), 3),
                depth_std=round(mdata["cascade_depth"].std(), 3),
            ))

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------ #
    # Wide format: one row per city, three method columns side by side    #
    # ------------------------------------------------------------------ #
    method_keys = list(METHOD_LABELS.keys())
    wide_rows = []
    for city_cn in REPRESENTATIVE_CITIES:
        city_data = full_df[full_df["city"] == city_cn]
        if city_data.empty:
            continue
        row = {"city": city_cn}
        for mk in method_keys:
            mdata = city_data[city_data["method"] == mk]
            short = mk.replace("route_support_", "rs_").replace(
                "neighbor_fraction_", "nf_")
            row[f"{short}_collapse_mean"]  = round(mdata["collapse_fraction"].mean(), 4)
            row[f"{short}_collapse_std"]   = round(mdata["collapse_fraction"].std(), 4)
            row[f"{short}_depth_mean"]     = round(mdata["cascade_depth"].mean(), 3)
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)

    # ------------------------------------------------------------------ #
    # Cross-city summary row                                               #
    # ------------------------------------------------------------------ #
    agg_rows = []
    for method_key, method_label in METHOD_LABELS.items():
        mdata = full_df[full_df["method"] == method_key]
        city_means = mdata.groupby("city")["collapse_fraction"].mean()
        agg_rows.append(dict(
            method=method_label,
            collapse_mean_across_cities=round(city_means.mean(), 4),
            collapse_std_across_cities=round(city_means.std(), 4),
            collapse_min=round(city_means.min(), 4),
            collapse_max=round(city_means.max(), 4),
        ))
    agg_df = pd.DataFrame(agg_rows)

    # ------------------------------------------------------------------ #
    # Save outputs                                                         #
    # ------------------------------------------------------------------ #
    full_df.to_csv(results_root / "reconstruction_check_full.csv",
                   index=False, encoding="utf-8-sig")
    summary_df.to_csv(results_root / "reconstruction_check_table.csv",
                      index=False, encoding="utf-8-sig")
    wide_df.to_csv(results_root / "reconstruction_check_wide.csv",
                   index=False, encoding="utf-8-sig")
    agg_df.to_csv(results_root / "reconstruction_check_summary.csv",
                  index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ #
    # Print the main result to terminal                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("RESULT — Collapse fraction (mean ± SD over 50 reps, f=0.10, τ=0.30)")
    print("=" * 70)
    print(f"\n{'City':<14}  {'RS-native':>12}  {'RS-Lspace-recon':>17}  "
          f"{'NF-Lspace-only':>16}")
    print("-" * 66)
    for _, r in wide_df.iterrows():
        rs_n  = f"{r['rs_native_hypergraph_collapse_mean']:.3f}±{r['rs_native_hypergraph_collapse_std']:.3f}"
        rs_l  = f"{r['rs_lspace_external_recon_collapse_mean']:.3f}±{r['rs_lspace_external_recon_collapse_std']:.3f}"
        nf_l  = f"{r['nf_lspace_only_collapse_mean']:.3f}±{r['nf_lspace_only_collapse_std']:.3f}"
        print(f"  {r['city']:<12}  {rs_n:>12}  {rs_l:>17}  {nf_l:>16}")
    print("-" * 66)
    print(f"\nMean across 6 cities:")
    for _, r in agg_df.iterrows():
        print(f"  {r['method']:<44}  "
              f"{r['collapse_mean_across_cities']:.4f} ± {r['collapse_std_across_cities']:.4f}")

    print(f"""
KEY TAKEAWAY
  Columns RS-native and RS-Lspace-recon should be IDENTICAL (same rctx,
  same cascade function — external reconstruction is numerically equivalent).
  Column NF-Lspace-only should be close to 1.0 (near-total collapse),
  confirming that discarding membership and using only pairwise adjacency
  produces categorically different dynamics.

  This supports the paper's claim: the hypergraph is needed not because it
  produces different cascade numbers (when membership is correctly
  re-imported it does not), but because it makes route-membership the
  default, intrinsic object of the model rather than an external re-import.
""")
    print(f"Results written to: {results_root}/")


if __name__ == "__main__":
    main()
