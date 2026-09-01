# End-to-end workflow

1. `scripts/build_transit_hypergraphs.py` — construct route-preserving nodes, hyperedges and
   inferred transfer edges for `exact_name`, `walk_100m`, `walk_200m`, and `walk_300m`.
2. `scripts/analyze_transit_hypergraphs.py` — compute structural descriptors and transfer-rule checks.
3. `scripts/run_resilience_experiments.py` — R1–R7 random disruption (default: 100 repetitions,
   fraction step 0.02, source samples 500, seed base 42).
4. `scripts/run_resilience_targeted.py` — T1–T3 targeted disruption (seed base 42).
5. `scripts/run_resilience_cascade.py` — C1–C5 route-support cascades (tau = 0.2, 0.4, 0.6;
   default 100 repetitions; seed base 42).
6. `scripts/run_resilience_recovery.py` — REC1–REC6 recovery heuristics (100 repetitions;
   damage fraction 0.10; seed base 42).
7. `scripts/run_phase7_clustering.py` — frozen structure-only representative selection and
   separate outcome-informed descriptive clustering.
8. `scripts/run_pspace_comparison.py`, `scripts/run_representation_comparison.py`,
   `scripts/run_lspace_reconstruction_check.py`, and `scripts/run_cross_mode_r3.py` — representation and
   cross-mode checks.
9. `scripts/run_reviewer1_comment5_stats.py` — bootstrap hypotheses and model-selection uncertainty.
10. Run the relevant analysis and plotting scripts listed in `FIGURE_TABLE_MAP.csv` to
    regenerate numerical and graphical outputs. Final journal artwork and typeset manuscript
    files are not tracked in this repository.
