#!/usr/bin/env python3
"""Phase 7 — City clustering, representative selection, city-level statistics.

Inputs: Phase 1/2/5/6 summary CSVs already produced under results_run_resilience_*.
Outputs: feature matrix, hierarchical-clustering result, K selected by silhouette,
representatives per cluster (closest-to-centroid), SCI-quality figures, regression
and ranking-correlation tables.

All figures are English-only (city names use the inventory's 'pinyin' folder name).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------- paths ----------

ROOT = Path(__file__).resolve().parents[1]
PHASE1_SUMMARY = ROOT / "results_run_resilience_experiments/walk_200m/all_cities_resilience_summary.csv"
PHASE2_SUMMARY = ROOT / "results_run_resilience_targeted/walk_200m/all_cities_resilience_summary_targeted.csv"
PHASE5_SUMMARY = ROOT / "results_run_resilience_cascade/walk_200m/all_cities_resilience_summary_cascade.csv"
PHASE6_SUMMARY = ROOT / "results_run_resilience_recovery/walk_200m/all_cities_resilience_summary_recovery.csv"
CITY_CSV = ROOT / "metadata/cities_with_bus_and_metro.csv"
BUILD_ROOT = ROOT / "results_build_transit_hypergraphs/walk_200m"

OUT_DIR = ROOT / "results_phase7_clustering"
FIG_DIR = OUT_DIR / "figures"


# ---------- publication-grade style ----------

PUB_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 13,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Categorical palette (Set2 = colorblind-friendly; up to 8 clusters)
CLUSTER_PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]


# ---------- feature engineering ----------

FEATURE_COLS = [
    "log10_n_nodes",
    "log10_n_hyperedges",
    "transfer_ratio",
    "metro_node_ratio",
    "avg_hyperdegree",
    "avg_hyperedge_size",
    "auc_lwcc_R3",
    "auc_lwcc_R6",
    "auc_clr_R7",
    "auc_lwcc_T2",
    "cascade_depth_C1",
    "auc_collapse_C4",
    "recovery_auc_REC2",
]


def load_inventory() -> pd.DataFrame:
    df = pd.read_csv(CITY_CSV, encoding="utf-8-sig")
    df = df.rename(columns={"城市中文": "city_cn", "公交文件夹": "city"})
    return df[["city_cn", "city"]]


def load_structural_features(city_folder: str) -> dict[str, float]:
    """Read per-city structural metrics from build outputs."""
    nodes_df = pd.read_csv(BUILD_ROOT / city_folder / "nodes.csv")
    hyperedges_df = pd.read_csv(BUILD_ROOT / city_folder / "hyperedges.csv")
    transfers_df = pd.read_csv(BUILD_ROOT / city_folder / "transfers.csv")
    hen_df = pd.read_csv(BUILD_ROOT / city_folder / "hyperedge_nodes.csv")

    n_nodes = len(nodes_df)
    n_hyperedges = len(hyperedges_df)
    n_transfers = len(transfers_df)
    n_metro = int((nodes_df["mode"] == "metro").sum())
    hd = hen_df.groupby("node_id").size().reindex(nodes_df["node_id"], fill_value=0)
    es = hen_df.groupby("edge_id").size().reindex(hyperedges_df["edge_id"], fill_value=0)
    return {
        "log10_n_nodes": np.log10(n_nodes),
        "log10_n_hyperedges": np.log10(max(n_hyperedges, 1)),
        "transfer_ratio": n_transfers / n_nodes,
        "metro_node_ratio": n_metro / n_nodes,
        "avg_hyperdegree": float(hd.mean()),
        "avg_hyperedge_size": float(es.mean()),
    }


def build_feature_matrix() -> pd.DataFrame:
    inv = load_inventory()

    p1 = pd.read_csv(PHASE1_SUMMARY)
    p2 = pd.read_csv(PHASE2_SUMMARY)
    p5 = pd.read_csv(PHASE5_SUMMARY)
    p6 = pd.read_csv(PHASE6_SUMMARY)

    rows = []
    for _, row in inv.iterrows():
        cn, folder = row["city_cn"], row["city"]
        if not (BUILD_ROOT / folder / "nodes.csv").exists():
            continue
        feats = load_structural_features(folder)
        feats["city_cn"] = cn
        feats["city"] = folder

        # Phase 1 features (by city_cn — Phase outputs use Chinese name)
        p1_r3 = p1[(p1["city"] == cn) & (p1["attack_type"] == "random_node_all")]
        p1_r6 = p1[(p1["city"] == cn) & (p1["attack_type"] == "random_hyperedge_all")]
        p1_r7 = p1[(p1["city"] == cn) & (p1["attack_type"] == "random_transfer")]
        feats["auc_lwcc_R3"] = float(p1_r3["auc_lwcc"].iloc[0])
        feats["auc_lwcc_R6"] = float(p1_r6["auc_lwcc"].iloc[0])
        feats["auc_clr_R7"] = float(p1_r7["auc_clr"].iloc[0])

        # Phase 2 features
        p2_t2 = p2[(p2["city"] == cn) & (p2["attack_type"] == "T2_node_betweenness")]
        feats["auc_lwcc_T2"] = float(p2_t2["auc_lwcc"].iloc[0])

        # Phase 5 features
        p5_c1 = p5[(p5["city"] == cn) & (p5["attack_type"] == "C1_random_node_tau0.2")]
        p5_c4 = p5[(p5["city"] == cn) & (p5["attack_type"] == "C4_targeted_node_tau0.2")]
        feats["cascade_depth_C1"] = float(p5_c1["mean_cascade_depth"].iloc[0])
        feats["auc_collapse_C4"] = float(p5_c4["auc_collapse_ratio"].iloc[0])

        # Phase 6 features
        p6_rec2 = p6[(p6["city"] == cn) & (p6["attack_type"] == "REC2_rand_dmg_hyperdegree_rec")]
        feats["recovery_auc_REC2"] = float(p6_rec2["recovery_auc_lwcc"].iloc[0])

        rows.append(feats)

    df = pd.DataFrame(rows).set_index("city")
    # Reorder
    return df[["city_cn"] + FEATURE_COLS]


# ---------- clustering ----------

def cluster_cities(features: pd.DataFrame) -> dict:
    X = features[FEATURE_COLS].to_numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    D = pdist(X_std, metric="euclidean")
    Z = linkage(D, method="ward")

    K_candidates = [3, 4, 5, 6]
    n_total = X_std.shape[0]
    silhouette_rows = []
    for K in K_candidates:
        labels = fcluster(Z, t=K, criterion="maxclust")
        score = silhouette_score(X_std, labels, metric="euclidean")
        sizes = np.bincount(labels)
        sizes = sizes[sizes > 0]
        silhouette_rows.append({
            "K": K,
            "silhouette": score,
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "max_size_share": float(sizes.max()) / n_total,
        })
    sil_df = pd.DataFrame(silhouette_rows)

    # Composite selection: prefer silhouette, but require max cluster ≤ 50% of N.
    # If no K satisfies, fall back to silhouette-max.
    eligible = sil_df[sil_df["max_size_share"] <= 0.50]
    if len(eligible):
        best_K = int(eligible.sort_values("silhouette", ascending=False)["K"].iloc[0])
        selection_rationale = (
            f"K={best_K} chosen as silhouette-maximum among K with max_cluster_share ≤ 0.50"
        )
    else:
        best_K = int(sil_df.sort_values("silhouette", ascending=False)["K"].iloc[0])
        selection_rationale = f"K={best_K} chosen as silhouette-maximum (no K satisfies balance constraint)"
    labels_best = fcluster(Z, t=best_K, criterion="maxclust")

    # Cluster-id relabeling: assign cluster ids in the dendrogram leaf order so that
    # cluster 1 = leftmost in the dendrogram. This keeps figure ordering meaningful.
    leaf_order = leaves_list(Z)
    seen = {}
    new_id = 0
    relabel = {}
    for leaf in leaf_order:
        old = labels_best[leaf]
        if old not in seen:
            new_id += 1
            seen[old] = new_id
        relabel[leaf] = seen[old]
    labels_relabeled = np.array([seen[old] for old in labels_best])

    return {
        "X_std": X_std,
        "scaler": scaler,
        "linkage": Z,
        "leaf_order": leaf_order,
        "labels": labels_relabeled,
        "K": best_K,
        "silhouette_table": sil_df,
        "selected_silhouette": float(sil_df.set_index("K").loc[best_K, "silhouette"]),
        "selection_rationale": selection_rationale,
    }


def pick_representatives(features: pd.DataFrame, X_std: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    cities = features.index.to_numpy()
    cities_cn = features["city_cn"].to_numpy()
    rows = []
    for k in sorted(set(labels)):
        mask = labels == k
        members = X_std[mask]
        centroid = members.mean(axis=0)
        dists = np.linalg.norm(members - centroid, axis=1)
        order = np.argsort(dists)
        rep_idx_in_cluster = order[0]
        rep_city = cities[mask][rep_idx_in_cluster]
        rep_city_cn = cities_cn[mask][rep_idx_in_cluster]
        rows.append({
            "cluster_id": k,
            "representative_city": rep_city,
            "representative_city_cn": rep_city_cn,
            "dist_to_centroid": float(dists[rep_idx_in_cluster]),
            "intra_max_dist": float(dists.max()),
            "intra_mean_dist": float(dists.mean()),
            "n_members": int(mask.sum()),
            "members": "|".join(sorted(cities[mask].tolist())),
        })
    return pd.DataFrame(rows)


# ---------- regression + ranking ----------

def run_regression(features: pd.DataFrame) -> pd.DataFrame:
    """OLS: each structural feature → each resilience outcome (univariate)."""
    structural = ["log10_n_nodes", "log10_n_hyperedges", "transfer_ratio",
                  "metro_node_ratio", "avg_hyperdegree", "avg_hyperedge_size"]
    outcomes = ["auc_lwcc_R3", "auc_lwcc_R6", "auc_clr_R7", "auc_lwcc_T2",
                "cascade_depth_C1", "auc_collapse_C4", "recovery_auc_REC2"]
    rows = []
    for s in structural:
        for o in outcomes:
            x = features[s].to_numpy()
            y = features[o].to_numpy()
            x_mean = x.mean(); y_mean = y.mean()
            cov = np.mean((x - x_mean) * (y - y_mean))
            var_x = np.var(x)
            slope = cov / var_x if var_x else float("nan")
            intercept = y_mean - slope * x_mean if var_x else float("nan")
            y_pred = intercept + slope * x
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
            # Pearson + p
            n = len(x)
            pearson = cov / np.sqrt(np.var(x) * np.var(y)) if (np.var(x) and np.var(y)) else float("nan")
            # t-stat for slope = pearson * sqrt(n-2) / sqrt(1-pearson^2)
            if n > 2 and -1 < pearson < 1:
                t_stat = pearson * np.sqrt((n - 2) / (1 - pearson ** 2))
                from scipy.stats import t
                p = 2 * (1 - t.cdf(abs(t_stat), df=n - 2))
            else:
                t_stat, p = float("nan"), float("nan")
            rows.append({
                "predictor": s, "outcome": o,
                "slope": slope, "intercept": intercept,
                "pearson_r": pearson, "r_squared": r2,
                "p_value": p, "n": n,
            })
    return pd.DataFrame(rows)


def ranking_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between city rankings on different resilience outcomes."""
    outcomes = ["auc_lwcc_R3", "auc_lwcc_R6", "auc_clr_R7", "auc_lwcc_T2",
                "cascade_depth_C1", "auc_collapse_C4", "recovery_auc_REC2"]
    rho = pd.DataFrame(index=outcomes, columns=outcomes, dtype=float)
    for a in outcomes:
        for b in outcomes:
            r, _ = spearmanr(features[a], features[b])
            rho.loc[a, b] = r
    return rho


# ---------- figures ----------

def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1_dendrogram(features: pd.DataFrame, result: dict) -> None:
    Z = result["linkage"]
    K = result["K"]
    labels = result["labels"]
    n = len(features)

    def leaf_cluster(node_id: int) -> int:
        if node_id < n:
            return int(labels[node_id])
        left = int(Z[node_id - n, 0])
        return leaf_cluster(left)

    cluster_at_link = {}
    def link_color(node_id: int) -> str:
        if node_id in cluster_at_link:
            return cluster_at_link[node_id]
        left = int(Z[node_id - n, 0])
        right = int(Z[node_id - n, 1])
        lc, rc = leaf_cluster(left), leaf_cluster(right)
        color = CLUSTER_PALETTE[(lc - 1) % len(CLUSTER_PALETTE)] if lc == rc else "0.55"
        cluster_at_link[node_id] = color
        return color

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    mpl.rcParams["lines.linewidth"] = 1.6
    dendrogram(
        Z,
        labels=features.index.tolist(),
        ax=ax,
        link_color_func=link_color,
        leaf_rotation=90,
    )
    ax.set_ylabel("Ward linkage distance", labelpad=6)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.tick_params(axis="y", labelsize=11)

    handles = [Patch(color=CLUSTER_PALETTE[(k - 1) % len(CLUSTER_PALETTE)], label=f"C{k}")
               for k in range(1, K + 1)]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=11, ncol=K,
              bbox_to_anchor=(1.0, 1.02))
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "fig1_dendrogram")


def fig2_cluster_heatmap(features: pd.DataFrame, result: dict) -> None:
    leaves = result["leaf_order"]
    X_std = result["X_std"]
    ordered_cities = features.index.to_numpy()[leaves]
    ordered_X = X_std[leaves]
    cluster_at_leaf = result["labels"][leaves]
    K = result["K"]

    fig = plt.figure(figsize=(9.8, 12.0))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[0.045, 1.0, 0.030, 0.18],
        left=0.16, right=0.96, top=0.97, bottom=0.13, wspace=0.04,
    )
    band_ax = fig.add_subplot(gs[0])
    main_ax = fig.add_subplot(gs[1])
    cbar_ax = fig.add_subplot(gs[2])
    legend_ax = fig.add_subplot(gs[3])

    for i, c in enumerate(cluster_at_leaf):
        band_ax.add_patch(plt.Rectangle((0, i), 1, 1,
                                         color=CLUSTER_PALETTE[(c - 1) % len(CLUSTER_PALETTE)]))
    band_ax.set_xlim(0, 1); band_ax.set_ylim(0, len(leaves))
    band_ax.invert_yaxis()
    band_ax.set_xticks([]); band_ax.set_yticks([])
    band_ax.spines[:].set_visible(False)

    vmax = float(np.abs(ordered_X).max())
    im = main_ax.imshow(ordered_X, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        interpolation="nearest")
    main_ax.set_xticks(range(len(FEATURE_COLS)))
    main_ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=10)
    main_ax.set_yticks(range(len(ordered_cities)))
    main_ax.set_yticklabels(ordered_cities, fontsize=10)
    main_ax.tick_params(axis="both", length=0)

    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("z-score", fontsize=12)
    cb.ax.tick_params(labelsize=11)

    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
    step = 1.0 / (K + 1)
    for k in range(K):
        y = 1.0 - (k + 1) * step
        legend_ax.add_patch(plt.Rectangle((0.05, y), 0.30, 0.7 * step, transform=legend_ax.transAxes,
                                           color=CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]))
        legend_ax.text(0.42, y + 0.35 * step, f"C{k + 1}", ha="left", va="center",
                       fontsize=13, fontweight="bold", transform=legend_ax.transAxes)
    _save(fig, "fig2_cluster_heatmap")


def fig3_pca_biplot(features: pd.DataFrame, result: dict, representatives: pd.DataFrame) -> None:
    X_std = result["X_std"]
    labels = result["labels"]
    K = result["K"]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.subplots_adjust(left=0.10, right=0.86, top=0.97, bottom=0.10)

    for k in range(1, K + 1):
        m = labels == k
        ax.scatter(coords[m, 0], coords[m, 1], s=90,
                   color=CLUSTER_PALETTE[(k - 1) % len(CLUSTER_PALETTE)],
                   edgecolor="0.2", linewidth=0.7, label=f"C{k}", zorder=3)

    rep_cities = set(representatives["representative_city"].tolist())
    for i, city in enumerate(features.index):
        if city in rep_cities:
            ax.scatter(coords[i, 0], coords[i, 1], s=240, facecolor="none",
                       edgecolor="black", linewidth=2.2, zorder=5)

    # Greedy label spacing — representatives win first, others get small grey
    placed = []
    def place(x, y, text, fontsize, weight, color):
        offsets = [(10, 8), (-12, 8), (10, -10), (-12, -10),
                   (14, 0), (-16, 0), (0, 12), (0, -14),
                   (18, 6), (-20, 6), (18, -8), (-20, -8)]
        min_dist = 0.55
        for dx, dy in offsets:
            cand = (x + dx * 0.06, y + dy * 0.06)
            if all(np.hypot(cand[0] - p[0], cand[1] - p[1]) > min_dist for p in placed):
                placed.append(cand)
                ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize, fontweight=weight,
                            color=color, zorder=6)
                return
        placed.append((x + offsets[0][0] * 0.06, y + offsets[0][1] * 0.06))
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=offsets[0], fontsize=fontsize, fontweight=weight,
                    color=color, zorder=6)

    for i, city in enumerate(features.index):
        if city in rep_cities:
            place(coords[i, 0], coords[i, 1], city, 12, "bold", "black")
    for i, city in enumerate(features.index):
        if city not in rep_cities:
            place(coords[i, 0], coords[i, 1], city, 8.5, "normal", "0.35")

    ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)", labelpad=6)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)", labelpad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=11,
              bbox_to_anchor=(1.01, 1.0), title="Cluster", title_fontsize=12)
    _save(fig, "fig3_pca_biplot")


def fig4_silhouette(result: dict) -> None:
    sil_df = result["silhouette_table"]
    X_std = result["X_std"]
    labels = result["labels"]
    K = result["K"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})

    axes[0].bar(sil_df["K"].astype(str), sil_df["silhouette"], color="0.45", width=0.55)
    axes[0].bar([str(K)], [sil_df.set_index("K").loc[K, "silhouette"]],
                color="#1f77b4", width=0.55, label=f"Selected K = {K}")
    axes[0].set_xlabel("Number of clusters K", labelpad=6)
    axes[0].set_ylabel("Mean silhouette", labelpad=6)
    axes[0].legend(frameon=False, fontsize=11)

    samp_sil = silhouette_samples(X_std, labels, metric="euclidean")
    y = 0
    for k in range(1, K + 1):
        vals = sorted(samp_sil[labels == k])
        axes[1].fill_betweenx(np.arange(y, y + len(vals)), 0, vals,
                              color=CLUSTER_PALETTE[(k - 1) % len(CLUSTER_PALETTE)],
                              alpha=0.9, edgecolor="0.2", linewidth=0.5)
        axes[1].text(-0.03, y + len(vals) / 2, f"C{k}", ha="right", va="center",
                     fontsize=11, fontweight="bold")
        y += len(vals) + 2
    mean_sil = samp_sil.mean()
    axes[1].axvline(mean_sil, color="red", linestyle="--", linewidth=1.0,
                    label=f"mean = {mean_sil:.3f}")
    axes[1].set_xlabel("Silhouette coefficient", labelpad=6)
    axes[1].set_yticks([])
    axes[1].legend(loc="lower right", frameon=False, fontsize=11)
    _save(fig, "fig4_silhouette")


def fig5_representative_radar(features: pd.DataFrame, result: dict, representatives: pd.DataFrame) -> None:
    X_std = result["X_std"]
    cities_idx = list(features.index)
    rep_rows = []
    for _, r in representatives.iterrows():
        i = cities_idx.index(r["representative_city"])
        clipped = np.clip(X_std[i], -3.0, 3.0)
        rep_rows.append((r["representative_city"], r["cluster_id"], clipped))

    n_axes = len(FEATURE_COLS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.2, 8.2), subplot_kw={"polar": True})
    for city, cid, vec in rep_rows:
        v = list(vec) + [vec[0]]
        ax.plot(angles, v, linewidth=2.0, color=CLUSTER_PALETTE[(cid - 1) % len(CLUSTER_PALETTE)],
                label=f"C{cid}: {city}")
        ax.fill(angles, v, alpha=0.12, color=CLUSTER_PALETTE[(cid - 1) % len(CLUSTER_PALETTE)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURE_COLS, fontsize=11)
    ax.set_ylim(-3.2, 3.2)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.tick_params(axis="y", labelsize=9, pad=2)
    ax.set_rlabel_position(135)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=12)
    _save(fig, "fig5_representative_radar")


def fig6_ranking_correlations(rho: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    im = ax.imshow(rho.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho)))
    ax.set_xticklabels(rho.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(rho)))
    ax.set_yticklabels(rho.index, fontsize=11)
    for i in range(len(rho)):
        for j in range(len(rho)):
            v = rho.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    _save(fig, "fig6_ranking_correlations")


# ---------- report ----------

def write_report(features: pd.DataFrame, result: dict, representatives: pd.DataFrame,
                 regression: pd.DataFrame, rho: pd.DataFrame) -> None:
    K = result["K"]
    lines = [
        "# Phase 7 — Clustering & Representative Selection",
        "",
        f"- **Cities**: {len(features)}",
        f"- **Features**: {len(FEATURE_COLS)} (structural + Phase 1/2/5/6 resilience metrics)",
        f"- **Clustering**: Ward linkage on z-scored Euclidean distances",
        f"- **Selected K**: {K} (silhouette = {result['selected_silhouette']:.3f})",
        "",
        "## Silhouette across K",
        result["silhouette_table"].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Representatives (closest-to-centroid)",
        representatives.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Cluster sizes",
        pd.Series(result["labels"]).value_counts().sort_index().to_frame("n_cities")
            .to_markdown(),
        "",
        "## Top regression effects (|pearson_r| ≥ 0.5)",
        (regression[regression["pearson_r"].abs() >= 0.5]
            .sort_values("pearson_r", key=abs, ascending=False)
            .head(15)
            .to_markdown(index=False, floatfmt=".3f")),
        "",
        "## Resilience-outcome ranking correlations (Spearman)",
        rho.round(3).to_markdown(),
        "",
        "## Outputs",
        "- `city_features.csv`, `city_features_zscored.csv`",
        "- `cluster_assignment.csv`, `representatives.csv`",
        "- `regression_results.csv`, `ranking_correlations.csv`",
        "- `figures/fig1_dendrogram.pdf`, `figures/fig2_cluster_heatmap.pdf`, ...",
    ]
    (OUT_DIR / "phase7_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(PUB_STYLE)

    print("Building feature matrix...")
    features = build_feature_matrix()
    print(f"  features shape: {features.shape}")
    assert not features[FEATURE_COLS].isna().any().any(), "NaN in features"

    features.to_csv(OUT_DIR / "city_features.csv", encoding="utf-8-sig")

    print("Clustering...")
    result = cluster_cities(features)
    K = result["K"]
    print(f"  selected K = {K}  silhouette = {result['selected_silhouette']:.3f}")
    print(f"  silhouette table:\n{result['silhouette_table'].to_string(index=False)}")

    # standardized features
    zdf = pd.DataFrame(result["X_std"], index=features.index, columns=FEATURE_COLS)
    zdf.insert(0, "city_cn", features["city_cn"])
    zdf.to_csv(OUT_DIR / "city_features_zscored.csv", encoding="utf-8-sig")

    np.save(OUT_DIR / "linkage_matrix.npy", result["linkage"])
    result["silhouette_table"].to_csv(OUT_DIR / "silhouette_scores.csv", index=False)
    (OUT_DIR / "selected_K.txt").write_text(
        f"K={K}\nsilhouette={result['selected_silhouette']:.6f}\n"
        f"rationale: {result['selection_rationale']}\n",
        encoding="utf-8",
    )

    assignment = pd.DataFrame({
        "city": features.index,
        "city_cn": features["city_cn"].values,
        "cluster_id": result["labels"],
    })
    assignment.to_csv(OUT_DIR / "cluster_assignment.csv", index=False, encoding="utf-8-sig")

    print("Picking representatives...")
    reps = pick_representatives(features, result["X_std"], result["labels"])
    print(reps.to_string(index=False))
    reps.to_csv(OUT_DIR / "representatives.csv", index=False, encoding="utf-8-sig")

    print("Running regressions...")
    regression = run_regression(features)
    regression.to_csv(OUT_DIR / "regression_results.csv", index=False)

    print("Computing ranking correlations...")
    rho = ranking_correlations(features)
    rho.to_csv(OUT_DIR / "ranking_correlations.csv")

    print("Drawing figures...")
    fig1_dendrogram(features, result)
    fig2_cluster_heatmap(features, result)
    fig3_pca_biplot(features, result, reps)
    fig4_silhouette(result)
    fig5_representative_radar(features, result, reps)
    fig6_ranking_correlations(rho)

    print("Writing report...")
    write_report(features, result, reps, regression, rho)
    print(f"Done. Outputs at: {OUT_DIR}")


if __name__ == "__main__":
    main()
