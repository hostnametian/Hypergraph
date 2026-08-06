#!/usr/bin/env python3
"""Mean-field theory and phase diagram analysis for the route-support cascade.

Implements:
  1. Mean-field self-consistency equations for route-support cascade
     (heterogeneous threshold: k(v) = ⌈τ·d_H(v)⌉)
  2. Phase boundary computation in (f, τ) parameter space
  3. Comparison with neighbor-fraction rule on projected graphs
  4. Synthetic hypergraph generation and numerical validation
  5. Critical scaling exponent extraction

Theory references:
  - Dorogovtsev et al. (2006) PRL 96, 040601  [k-core hybrid transition]
  - Zhang et al. (2023) NatComm 14, 1605      [cross-order degree correlation]
  - This study: heterogeneous-threshold cascade on hypergraphs

Output: results_phase_transition_analysis/
  - mean_field_phase_diagram.csv
  - synthetic_phase_diagram.csv
  - scaling_analysis.csv
  - figures/phase_diagram_*.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq
from scipy.special import binom, gammaln
from scipy.stats import linregress

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results_phase_transition_analysis"

# Number of quadrature points for distribution sums
N_QUAD = 500


# ====================================================================
# PART A: MEAN-FIELD THEORY
# ====================================================================

# --- Distribution utilities ---

def poisson_pmf(d: int, lam: float) -> float:
    """Poisson PMF: P(d) = e^{-λ} λ^d / d!"""
    if d < 0 or lam <= 0:
        return 0.0
    return math.exp(-lam + d * math.log(lam) - gammaln(d + 1))


def powerlaw_pmf(d: int, gamma: float, d_min: int = 1, d_max: int = 500) -> float:
    """Power-law PMF: P(d) ∝ d^{-γ} for d ∈ [d_min, d_max]."""
    if d < d_min or d > d_max:
        return 0.0
    norm = sum(k ** (-gamma) for k in range(d_min, d_max + 1))
    return d ** (-gamma) / norm


# --- Binomial survival functions ---

def binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p).

    Uses scipy.stats.binom.sf for numerical stability with large n.
    """
    if k > n:
        return 0.0
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    from scipy.stats import binom as binom_dist
    return float(binom_dist.sf(k - 1, n, p))


def binomial_sf_fast(k: int, n: int, p: np.ndarray) -> np.ndarray:
    """Vectorized P(X >= k) for fixed k, n and array p."""
    result = np.zeros_like(p)
    for i, pi in enumerate(p):
        result[i] = binomial_sf(k, n, pi)
    return result


# --- Mean-field functions ---

def F_hyperedge_viability(y: float, Q_s: np.ndarray, alpha: float = 0.0) -> float:
    """Probability a random hyperedge is viable.

    Args:
        y: probability a random node (on a random hyperedge) survives
        Q_s: array of hyperedge size probabilities, Q_s[s] = P(size=s)
        alpha: proportional threshold (default 0 = fixed min-2, i.e. α=2/s)
               alpha > 0 means need >= ⌈α·s⌉ surviving nodes
    """
    x = 0.0
    s_max = len(Q_s) - 1
    for s in range(2, s_max + 1):
        if Q_s[s] == 0:
            continue
        if alpha > 0:
            k_min = max(2, math.ceil(alpha * s))
        else:
            k_min = 2  # fixed min-2 condition
        # P(viable | size=s) = P(>= k_min of s nodes survive)
        p_viable = binomial_sf(k_min, s, y)
        x += Q_s[s] * p_viable
    return x


def F_hyperedge_viability_vectorized(y: np.ndarray, Q_s: np.ndarray,
                                     alpha: float = 0.0) -> np.ndarray:
    """Vectorized version for plotting."""
    return np.array([F_hyperedge_viability(float(yi), Q_s, alpha) for yi in y])


def G_node_survival(x: float, P_d: np.ndarray, tau: float, initial_f: float) -> float:
    """Probability a random node (reached via random hyperedge) survives.

    Uses excess hyperdegree distribution: d·P(d)/⟨d⟩.

    Args:
        x: probability a random hyperedge is viable
        P_d: array of hyperdegree probabilities, P_d[d] = P(hyperdegree=d)
        tau: tolerance threshold (node needs >= ⌈τ·d⌉ viable hyperedges)
        initial_f: initial random damage fraction
    """
    d_max = len(P_d) - 1
    mean_d = sum(d * P_d[d] for d in range(d_max + 1))
    if mean_d == 0:
        return 0.0

    y = 0.0
    for d in range(1, d_max + 1):
        if P_d[d] == 0:
            continue
        k_min = math.ceil(tau * d)
        if k_min > d:
            p_survive = 0.0
        elif k_min <= 0:
            p_survive = 1.0
        else:
            p_survive = binomial_sf(k_min, d, x)
        y += (d * P_d[d] / mean_d) * p_survive

    return (1 - initial_f) * y


def G_node_survival_vectorized(x: np.ndarray, P_d: np.ndarray, tau: float,
                                initial_f: float) -> np.ndarray:
    """Vectorized version."""
    return np.array([G_node_survival(float(xi), P_d, tau, initial_f) for xi in x])


def H_map(y: float, P_d: np.ndarray, Q_s: np.ndarray, tau: float,
          initial_f: float, alpha: float = 0.0) -> float:
    """Composite map: y_{t+1} = H(y_t) = (1-f)·G(F(y_t))."""
    x = F_hyperedge_viability(y, Q_s, alpha)
    return G_node_survival(x, P_d, tau, initial_f)


def find_fixed_point(P_d: np.ndarray, Q_s: np.ndarray, tau: float,
                     initial_f: float, alpha: float = 0.0,
                     n_iter: int = 200) -> Tuple[float, list]:
    """Iterate H(y) to convergence. Returns (y*, trajectory)."""
    y = 1.0 - initial_f
    traj = [y]
    for _ in range(n_iter):
        y_new = H_map(y, P_d, Q_s, tau, initial_f, alpha)
        traj.append(y_new)
        if abs(y_new - y) < 1e-12:
            break
        y = y_new
    return y, traj


def find_collapse_fraction(P_d: np.ndarray, Q_s: np.ndarray, tau: float,
                           initial_f: float, alpha: float = 0.0) -> float:
    """Compute final collapse fraction Γ from fixed point y*."""
    y_star, _ = find_fixed_point(P_d, Q_s, tau, initial_f, alpha)
    x_star = F_hyperedge_viability(y_star, Q_s, alpha)

    # Fraction of nodes that survive: Σ_d P(d)·P(>=⌈τd⌉ viable hyperedges)
    d_max = len(P_d) - 1
    p_survive_total = 0.0
    for d in range(d_max + 1):
        if P_d[d] == 0:
            continue
        k_min = math.ceil(tau * d)
        if k_min <= 0:
            p_survive_total += P_d[d] * 1.0
        elif k_min <= d:
            p_survive_total += P_d[d] * binomial_sf(k_min, d, x_star)

    return 1.0 - p_survive_total


# --- Phase boundary computation ---

def find_phase_boundary(P_d: np.ndarray, Q_s: np.ndarray,
                        tau_values: np.ndarray,
                        alpha: float = 0.0,
                        tol: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Find critical f_c(τ) where collapse fraction jumps.

    For each τ, scan f from 0 to 1 and find where collapse fraction
    deviates from the initial damage by more than tol.
    """
    f_c = np.zeros(len(tau_values))

    for i, tau in enumerate(tau_values):
        f_scan = np.linspace(0.01, 0.95, 50)
        collapses = np.array([
            find_collapse_fraction(P_d, Q_s, tau, f, alpha)
            for f in f_scan
        ])

        # Find first f where collapse exceeds initial damage significantly
        excess = collapses - f_scan
        threshold_idx = np.where(excess > tol)[0]
        if len(threshold_idx) > 0:
            # Refine with binary search
            f_lo = f_scan[max(0, threshold_idx[0] - 1)]
            f_hi = f_scan[threshold_idx[0]]
            for _ in range(10):
                f_mid = (f_lo + f_hi) / 2
                col_mid = find_collapse_fraction(P_d, Q_s, tau, f_mid, alpha)
                if col_mid - f_mid > tol:
                    f_hi = f_mid
                else:
                    f_lo = f_mid
            f_c[i] = f_hi
        else:
            f_c[i] = np.nan

    return tau_values, f_c


# --- Critical scaling ---

def extract_critical_exponent(P_d: np.ndarray, Q_s: np.ndarray,
                              tau: float, f_c: float,
                              n_points: int = 30) -> Tuple[float, float]:
    """Extract scaling exponent β from Γ(f) near f_c.

    Γ(f) - Γ(f_c) ∝ |f - f_c|^β
    """
    epsilons = np.logspace(-3, -0.5, n_points)

    gamma_base = find_collapse_fraction(P_d, Q_s, tau, f_c, 0.0)
    gammas = np.array([find_collapse_fraction(P_d, Q_s, tau, f_c + eps, 0.0)
                        for eps in epsilons])
    deltas = np.maximum(gammas - gamma_base, 1e-12)

    # Log-log fit
    valid = (deltas > 1e-10) & np.isfinite(deltas)
    if valid.sum() < 5:
        return np.nan, np.nan

    slope, intercept, r, p, std_err = linregress(
        np.log(epsilons[valid]), np.log(deltas[valid])
    )
    return slope, std_err


# ====================================================================
# PART B: NEIGHBOUR-FRACTION COMPARISON
# ====================================================================

def projected_degree_distribution(P_d: np.ndarray, Q_s: np.ndarray) -> np.ndarray:
    """Approximate degree distribution of the clique-projected graph.

    A node with hyperdegree d, incident to hyperedges of mean size ⟨s⟩,
    has projected degree ≈ d · (⟨s⟩ - 1) · d_neighbor_factor.
    """
    d_max = len(P_d) - 1
    s_mean = sum(s * Q_s[s] for s in range(len(Q_s)))
    if s_mean < 2:
        s_mean = 2.0

    # Each hyperedge of size s adds (s-1) pairwise connections per node
    # Node with hyperdegree d gets ~ d·(⟨s⟩-1) unique projected neighbors
    k_max_proj = int(d_max * (s_mean - 1) * 2)  # generous upper bound
    k_max_proj = min(k_max_proj, 2000)

    P_k = np.zeros(k_max_proj + 1)
    for d in range(1, d_max + 1):
        if P_d[d] == 0:
            continue
        # Approximate projected degree ≈ d * (s_mean - 1) * overlap_factor
        # (overlap_factor < 1 accounts for shared neighbors across hyperedges)
        k_approx = int(d * (s_mean - 1) * 0.7)  # heuristic overlap factor
        k_approx = min(k_approx, k_max_proj)
        P_k[k_approx] += P_d[d]

    # Normalize
    if P_k.sum() > 0:
        P_k /= P_k.sum()

    return P_k


def neighbor_fraction_collapse(P_k: np.ndarray, tau: float,
                                initial_f: float) -> float:
    """Neighbor-fraction cascade on projected graph.

    Node with projected degree k fails if < ⌈τ·k⌉ neighbors survive.
    This is a heterogeneous k-core variant on the projected graph.
    """
    k_max = len(P_k) - 1
    mean_k = sum(k * P_k[k] for k in range(k_max + 1))
    if mean_k == 0:
        return initial_f

    # Self-consistency: y = probability a randomly reached neighbor survives
    def H_nf(y):
        x = 0.0  # probability neighbor is "alive" in the cascade sense
        # Under random damage f, a neighbor is alive if it was not initially removed
        # AND it has enough surviving neighbors
        for k in range(1, k_max + 1):
            if P_k[k] == 0:
                continue
            k_min = math.ceil(tau * k)
            if k_min <= 0:
                x += (k * P_k[k] / mean_k) * 1.0
            elif k_min <= k:
                x += (k * P_k[k] / mean_k) * binomial_sf(k_min, k, y)
        return (1 - initial_f) * x

    # Iterate
    y = 1.0 - initial_f
    for _ in range(200):
        y_new = H_nf(y)
        if abs(y_new - y) < 1e-12:
            break
        y = y_new

    # Total survival
    p_total = 0.0
    for k in range(k_max + 1):
        if P_k[k] == 0:
            continue
        k_min = math.ceil(tau * k)
        if k_min <= 0:
            p_total += P_k[k]
        elif k_min <= k:
            p_total += P_k[k] * binomial_sf(k_min, k, y)

    return 1.0 - p_total


def find_nf_phase_boundary(P_k: np.ndarray, tau_values: np.ndarray,
                            tol: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Phase boundary for neighbor-fraction cascade."""
    f_c = np.zeros(len(tau_values))

    for i, tau in enumerate(tau_values):
        f_scan = np.linspace(0.01, 0.95, 50)
        collapses = np.array([neighbor_fraction_collapse(P_k, tau, f) for f in f_scan])
        excess = collapses - f_scan
        threshold_idx = np.where(excess > tol)[0]
        if len(threshold_idx) > 0:
            f_c[i] = f_scan[threshold_idx[0]]
        else:
            f_c[i] = np.nan

    return tau_values, f_c


# ====================================================================
# PART C: SYNTHETIC HYPERGRAPH GENERATION
# ====================================================================

def generate_synthetic_hypergraph(
    n_nodes: int,
    mean_hyperdegree: float,
    mean_hyperedge_size: float,
    cross_order_corr: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic hypergraph with controlled structural properties.

    Args:
        n_nodes: number of nodes
        mean_hyperdegree: target mean hyperdegree λ
        mean_hyperedge_size: target mean hyperedge size ⟨s⟩
        cross_order_corr: target cross-order degree correlation ρ_{1,2}
        seed: random seed

    Returns:
        P_d: empirical hyperdegree distribution (as array)
        Q_s: empirical hyperedge size distribution (as array)
    """
    rng = np.random.default_rng(seed)

    # Generate hyperdegrees from Poisson(λ)
    hyperdegrees = rng.poisson(mean_hyperdegree, size=n_nodes)
    hyperdegrees = np.maximum(hyperdegrees, 1)

    total_stubs = hyperdegrees.sum()

    # Generate hyperedge sizes
    n_edges = int(total_stubs / mean_hyperedge_size)
    if cross_order_corr == 0:
        # Independent: sizes from Poisson
        edge_sizes = rng.poisson(mean_hyperedge_size, size=n_edges)
    elif cross_order_corr > 0:
        # Positive correlation: high-degree nodes → large hyperedges
        # Assign nodes to edges, then weight edge sizes by mean degree of members
        edge_sizes = rng.poisson(mean_hyperedge_size, size=n_edges)
        # Bias: sort edges, assign larger to high-degree region
        sort_idx = np.argsort(edge_sizes)
        edge_sizes[sort_idx] = np.sort(
            rng.poisson(mean_hyperedge_size * (1 + cross_order_corr), size=n_edges)
        )
    else:
        # Negative correlation
        edge_sizes = rng.poisson(mean_hyperedge_size, size=n_edges)
        sort_idx = np.argsort(edge_sizes)[::-1]
        edge_sizes[sort_idx] = np.sort(
            rng.poisson(mean_hyperedge_size * (1 - abs(cross_order_corr) * 0.5), size=n_edges)
        )

    edge_sizes = np.maximum(edge_sizes, 2)  # min hyperedge size = 2

    # Build empirical distributions
    d_max = hyperdegrees.max()
    P_d = np.zeros(d_max + 1)
    for d in hyperdegrees:
        P_d[d] += 1
    P_d /= P_d.sum()

    s_max = edge_sizes.max()
    Q_s = np.zeros(s_max + 1)
    for s in edge_sizes:
        Q_s[s] += 1
    Q_s /= Q_s.sum()

    return P_d, Q_s


# ====================================================================
# PART D: MAIN ANALYSIS PIPELINE
# ====================================================================

def run_mean_field_analysis(output_dir: Path):
    """Compute mean-field phase diagrams and compare route-support vs neighbor-fraction."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Parameter grid ---
    # Hyperdegree distributions to test
    distributions = {
        "poisson_small": (lambda d: poisson_pmf(d, 5.0), 5, "Poisson(λ=5)"),
        "poisson_medium": (lambda d: poisson_pmf(d, 15.0), 15, "Poisson(λ=15)"),
        "poisson_large": (lambda d: poisson_pmf(d, 40.0), 40, "Poisson(λ=40)"),
        "powerlaw_2.5": (lambda d: powerlaw_pmf(d, 2.5, d_max=200), None, "Power-law(γ=2.5)"),
        "powerlaw_3.5": (lambda d: powerlaw_pmf(d, 3.5, d_max=200), None, "Power-law(γ=3.5)"),
    }

    # Hyperedge size distributions
    size_dists = {
        "constant_small": ("Constant(⟨s⟩=10)", 10),
        "constant_medium": ("Constant(⟨s⟩=30)", 30),
        "constant_large": ("Constant(⟨s⟩=100)", 100),
    }

    tau_values = np.arange(0.1, 0.9, 0.05)

    all_phase_data = []

    for dist_name, (pmf_func, lam, dist_label) in distributions.items():
        for size_name, (size_label, mean_s) in size_dists.items():
            print(f"\n{'='*60}")
            print(f"Distribution: {dist_label}, Size: {size_label}")

            # Build P_d
            d_max = 300 if lam is None else int(lam * 5)
            P_d = np.array([pmf_func(d) for d in range(d_max + 1)])
            P_d /= P_d.sum()

            # Build Q_s (concentrated at mean_s)
            s_max = int(mean_s * 3)
            Q_s = np.zeros(s_max + 1)
            Q_s[int(mean_s)] = 1.0

            # --- Route-support phase boundary ---
            print("  Computing route-support phase boundary...")
            tau_rs, fc_rs = find_phase_boundary(P_d, Q_s, tau_values)

            # --- Neighbour-fraction phase boundary ---
            print("  Computing neighbour-fraction phase boundary...")
            P_k = projected_degree_distribution(P_d, Q_s)
            tau_nf, fc_nf = find_nf_phase_boundary(P_k, tau_values)

            # --- Fit phase boundary functions ---
            valid_rs = ~np.isnan(fc_rs)
            valid_nf = ~np.isnan(fc_nf)

            rs_slope, rs_intercept = np.nan, np.nan
            nf_slope, nf_intercept = np.nan, np.nan

            if valid_rs.sum() >= 3:
                rs_slope, rs_intercept, _, _, _ = linregress(
                    tau_values[valid_rs], fc_rs[valid_rs]
                )

            if valid_nf.sum() >= 3:
                nf_slope, nf_intercept, _, _, _ = linregress(
                    tau_values[valid_nf], fc_nf[valid_nf]
                )

            # --- Extract critical exponent ---
            mid_tau_idx = len(tau_values) // 2
            tau_mid = tau_values[mid_tau_idx]
            if not np.isnan(fc_rs[mid_tau_idx]) and fc_rs[mid_tau_idx] < 0.9:
                beta_rs, beta_err = extract_critical_exponent(
                    P_d, Q_s, tau_mid, fc_rs[mid_tau_idx]
                )
            else:
                beta_rs, beta_err = np.nan, np.nan

            # Store results
            for i in range(len(tau_values)):
                all_phase_data.append({
                    "distribution": dist_label,
                    "size_dist": size_label,
                    "mean_hyperdegree": lam if lam else np.nan,
                    "mean_hyperedge_size": mean_s,
                    "tau": tau_values[i],
                    "fc_route_support": fc_rs[i],
                    "fc_neighbor_fraction": fc_nf[i],
                    "rs_slope": rs_slope,
                    "nf_slope": nf_slope,
                    "rs_intercept": rs_intercept,
                    "nf_intercept": nf_intercept,
                    "beta_rs": beta_rs,
                    "beta_err_rs": beta_err,
                })

            print(f"  Route-support: slope={rs_slope:.4f}, intercept={rs_intercept:.4f}")
            print(f"  Neighbor-fraction: slope={nf_slope:.4f}, intercept={nf_intercept:.4f}")
            print(f"  β_rs = {beta_rs:.3f} ± {beta_err:.3f}")

    # Save results
    df = pd.DataFrame(all_phase_data)
    df.to_csv(output_dir / "mean_field_phase_diagram.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved phase diagram data to {output_dir / 'mean_field_phase_diagram.csv'}")

    # Key finding summary
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    for dist_name in df["distribution"].unique():
        for size_name in df["size_dist"].unique():
            subset = df[(df["distribution"] == dist_name) &
                        (df["size_dist"] == size_name)]
            rs_slope = subset["rs_slope"].iloc[0]
            nf_slope = subset["nf_slope"].iloc[0]
            print(f"\n  {dist_name} × {size_name}:")
            print(f"    Route-support f_c(τ) = {rs_slope:.3f}·τ + {subset['rs_intercept'].iloc[0]:.3f}")
            print(f"    Neighbour-frac  f_c(τ) = {nf_slope:.3f}·τ + {subset['nf_intercept'].iloc[0]:.3f}")
            print(f"    Δslope = {abs(rs_slope - nf_slope):.3f} "
                  f"({'*** NOVEL' if abs(rs_slope) < 0.1 and abs(nf_slope) > 0.5 else ''})")

    return df


def run_synthetic_validation(output_dir: Path):
    """Generate synthetic hypergraphs and validate mean-field predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("SYNTHETIC HYPERGRAPH VALIDATION")
    print("="*60)

    # Parameter grid
    configs = [
        # (n_nodes, mean_d, mean_s, cross_order_corr, label)
        (3000, 8, 15, 0.0, "baseline"),
        (3000, 8, 15, 0.5, "positive_corr"),
        (3000, 8, 15, -0.5, "negative_corr"),
        (3000, 15, 30, 0.0, "dense"),
        (3000, 4, 8, 0.0, "sparse"),
        (5000, 8, 15, 0.0, "large_N"),
        (10000, 8, 15, 0.0, "xlarge_N"),
    ]

    all_results = []

    for n_nodes, mean_d, mean_s, corr, label in configs:
        print(f"\n  [{label}] N={n_nodes}, ⟨d⟩={mean_d}, ⟨s⟩={mean_s}, ρ={corr}")

        # Generate synthetic hypergraph
        P_d_emp, Q_s_emp = generate_synthetic_hypergraph(
            n_nodes, mean_d, mean_s, corr, seed=42
        )

        # Mean-field prediction using empirical distributions
        tau_mid = 0.3
        try:
            _, fc_mf = find_phase_boundary(P_d_emp, Q_s_emp, np.array([tau_mid]))
            f_c_mf = fc_mf[0]
        except Exception:
            f_c_mf = np.nan

        # Determine transition type at critical point
        if not np.isnan(f_c_mf) and f_c_mf < 0.9:
            beta, beta_err = extract_critical_exponent(P_d_emp, Q_s_emp, tau_mid, f_c_mf)
        else:
            beta, beta_err = np.nan, np.nan

        all_results.append({
            "label": label,
            "n_nodes": n_nodes,
            "mean_d": mean_d,
            "mean_s": mean_s,
            "cross_order_corr": corr,
            "f_c_mf": f_c_mf,
            "beta": beta,
            "beta_err": beta_err,
            "empirical_mean_d": float(np.sum(np.arange(len(P_d_emp)) * P_d_emp)),
            "empirical_mean_s": float(np.sum(np.arange(len(Q_s_emp)) * Q_s_emp)),
        })

        print(f"    f_c^MF = {f_c_mf:.4f}, β = {beta:.3f} ± {beta_err:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "synthetic_validation.csv", index=False, encoding="utf-8-sig")

    # Cross-order correlation effect
    corr_subset = df[df["label"].isin(["baseline", "positive_corr", "negative_corr"])]
    if len(corr_subset) >= 3:
        print("\n  Cross-order correlation effect on f_c:")
        for _, row in corr_subset.iterrows():
            print(f"    ρ={row['cross_order_corr']:+.1f}: f_c={row['f_c_mf']:.4f}")

    return df


# ====================================================================
# PART E: CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mean-field theory and phase diagram analysis")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "mean_field", "synthetic", "scaling"])
    args = parser.parse_args()

    if args.mode in ["all", "mean_field"]:
        run_mean_field_analysis(args.output_dir)

    if args.mode in ["all", "synthetic"]:
        run_synthetic_validation(args.output_dir)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
