"""
Generate the 4 evidence-gap figures identified in the paper review:
  GAP2: T2 vs T3 -- metro-node ratio scatter (explaining 12-city T2>T3 exception)
  GAP3: Transfer ratio vs cascade depth & collapse breadth scatter plots
  GAP4: Outcome-matrix PCA (scree + biplot of 5 resilience measures)
  GAP5: 5×5 correlation matrix heatmap of resilience outcomes
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── paths ──────────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parents[1])
STRUCT = os.path.join(BASE, "results_phase7_clustering", "city_features.csv")
TARGETED = os.path.join(BASE, "results_run_resilience_targeted", "walk_200m")
CASCADE  = os.path.join(BASE, "results_run_resilience_cascade",  "walk_200m")
RANDOM   = os.path.join(BASE, "results_run_resilience_experiments", "walk_200m")
RECOVERY = os.path.join(BASE, "results_run_resilience_recovery",  "walk_200m")
OUTDIR   = os.path.join(BASE, "paper_fig_table", "gap_figures")
os.makedirs(OUTDIR, exist_ok=True)

# ── load structural features ────────────────────────────────────────────────
struct_df = pd.read_csv(STRUCT)
# normalise city name column
struct_df.rename(columns={"city": "city_en"}, inplace=True)

# helper: collect per-city summary rows matching attack_type filter
def collect_city_rows(result_dir, summary_file, attack_filter, value_cols):
    rows = []
    for city_path in glob.glob(os.path.join(result_dir, "*")):
        if not os.path.isdir(city_path):   # skip stray CSV files
            continue
        fp = os.path.join(city_path, summary_file)
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp)
        df.columns = df.columns.str.lstrip('﻿')
        # keep only walk_200m
        if "network_version" in df.columns:
            df = df[df["network_version"] == "walk_200m"]
        for filt_col, filt_val in attack_filter.items():
            df = df[df[filt_col] == filt_val]
        if df.empty:
            continue
        row = {"city_en": os.path.basename(city_path)}
        for col in value_cols:
            row[col] = df[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)

# ── collect T2 and T3 AUC (targeted) ───────────────────────────────────────
t2_df = collect_city_rows(TARGETED, "resilience_summary_targeted.csv",
                           {"attack_type": "T2_node_betweenness"}, ["auc_lwcc"])
t3_df = collect_city_rows(TARGETED, "resilience_summary_targeted.csv",
                           {"attack_type": "T3_node_transfer_first"}, ["auc_lwcc"])
t2_df.rename(columns={"auc_lwcc": "t2_auc"}, inplace=True)
t3_df.rename(columns={"auc_lwcc": "t3_auc"}, inplace=True)

# ── collect cascade depth C1 (random tau=0.2) and C4 collapse ──────────────
c1_df = collect_city_rows(CASCADE, "resilience_summary_cascade.csv",
                           {"attack_type": "C1_random_node_tau0.2"},
                           ["mean_cascade_depth", "auc_collapse_ratio"])
c4_df = collect_city_rows(CASCADE, "resilience_summary_cascade.csv",
                           {"attack_type": "C4_targeted_node_tau0.2"},
                           ["mean_cascade_depth", "auc_collapse_ratio"])
c1_df.rename(columns={"mean_cascade_depth": "depth_C1",
                       "auc_collapse_ratio": "collapse_C1"}, inplace=True)
c4_df.rename(columns={"mean_cascade_depth": "depth_C4",
                       "auc_collapse_ratio": "collapse_C4"}, inplace=True)

# ── collect random node AUC (R1) ────────────────────────────────────────────
r1_df = collect_city_rows(RANDOM, "resilience_summary.csv",
                           {"attack_type": "random_node_all"}, ["auc_lwcc"])
r1_df.rename(columns={"auc_lwcc": "r1_auc"}, inplace=True)

# ── collect recovery AUC REC2 ───────────────────────────────────────────────
rec2_df = collect_city_rows(RECOVERY, "resilience_summary_recovery.csv",
                             {"attack_type": "REC2_rand_dmg_hyperdegree_rec"},
                             ["recovery_auc_lwcc"])
rec2_df.rename(columns={"recovery_auc_lwcc": "rec2_auc"}, inplace=True)

# ── merge everything ────────────────────────────────────────────────────────
df = struct_df[["city_en", "transfer_ratio", "metro_node_ratio",
                "log10_n_nodes", "log10_n_hyperedges"]].copy()

for sub in [t2_df, t3_df, c1_df, c4_df, r1_df, rec2_df]:
    df = df.merge(sub, on="city_en", how="left")

df.dropna(inplace=True)
print(f"Cities after merge: {len(df)}")

# ── shared style ────────────────────────────────────────────────────────────
BLUE   = "#2166AC"
RED    = "#D6604D"
GRAY   = "#888888"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

def add_ols(ax, x, y, color, label=None):
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if len(xv) < 3:
        return
    r, p = stats.pearsonr(xv, yv)
    m, b = np.polyfit(xv, yv, 1)
    xl = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xl, m * xl + b, color=color, lw=1.4, zorder=3)
    pstr = f"$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"
    lbl = f"$r = {r:.3f}$, {pstr}" if label is None else label
    ax.text(0.97, 0.05, lbl, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=color)
    return r, p

# ══════════════════════════════════════════════════════════════════════════
# GAP 2 — T2 vs T3 by metro-node ratio
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.2, 3.6))

df["t3_minus_t2"] = df["t3_auc"] - df["t2_auc"]   # positive = T3 worse than T2
t2_worse = df["t3_minus_t2"] > 0   # T3 more damaging (lower AUC)

ax.scatter(df.loc[t2_worse, "metro_node_ratio"],
           df.loc[t2_worse, "t3_minus_t2"],
           color=RED, s=40, alpha=0.85, zorder=4, label="T3 more damaging (33 cities)")
ax.scatter(df.loc[~t2_worse, "metro_node_ratio"],
           df.loc[~t2_worse, "t3_minus_t2"],
           color=BLUE, s=40, marker="^", alpha=0.85, zorder=4,
           label="T2 more damaging (12 cities)")
ax.axhline(0, color="k", lw=0.8, ls="--", zorder=2)
add_ols(ax, df["metro_node_ratio"], df["t3_minus_t2"], GRAY)
ax.set_xlabel("Metro-node ratio")
ax.set_ylabel("LWCC AUC: T3 − T2\n(positive = T3 more damaging)")
ax.set_title("T3 vs T2 damage gap by metro-node ratio", pad=6)
ax.legend(loc="upper right", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "FigS_gap2_T2vsT3_metro.pdf"), dpi=300)
plt.savefig(os.path.join(OUTDIR, "FigS_gap2_T2vsT3_metro.png"), dpi=300)
plt.close()
print("GAP 2 saved")

# ══════════════════════════════════════════════════════════════════════════
# GAP 3 — Transfer ratio vs cascade depth & collapse breadth
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

for ax, (ycol, ylabel, title) in zip(
        axes,
        [("depth_C4", "Mean cascade depth (C4, targeted, τ = 0.2)",
          "Transfer ratio vs cascade depth"),
         ("collapse_C4", "Collapse fraction (C4, targeted, τ = 0.2)",
          "Transfer ratio vs collapse breadth")]):
    ax.scatter(df["transfer_ratio"], df[ycol],
               color=BLUE, s=32, alpha=0.75, zorder=3)
    add_ols(ax, df["transfer_ratio"], df[ycol], RED)
    ax.set_xlabel("Transfer ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "FigS_gap3_transfer_vs_cascade.pdf"), dpi=300)
plt.savefig(os.path.join(OUTDIR, "FigS_gap3_transfer_vs_cascade.png"), dpi=300)
plt.close()
print("GAP 3 saved")

# ══════════════════════════════════════════════════════════════════════════
# GAP 4 — Outcome PCA (5 resilience measures)
# ══════════════════════════════════════════════════════════════════════════
outcome_cols = ["r1_auc", "t2_auc", "depth_C4", "collapse_C4", "rec2_auc"]
outcome_labels = ["LWCC AUC\n(random)", "LWCC AUC\n(betweenness)",
                  "Cascade depth\n(targeted)", "Collapse\n(targeted)",
                  "Recovery AUC\n(REC2)"]

X = df[outcome_cols].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
pca = PCA()
scores = pca.fit_transform(Xs)
evr = pca.explained_variance_ratio_

fig = plt.figure(figsize=(7.8, 3.6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.35)

# scree plot
ax0 = fig.add_subplot(gs[0])
cumvar = np.cumsum(evr) * 100
bars = ax0.bar(range(1, 6), evr * 100, color=BLUE, alpha=0.8, width=0.55, zorder=3)
ax0.plot(range(1, 6), cumvar, "o-", color=RED, lw=1.4, ms=5, zorder=4,
         label="Cumulative")
ax0.axhline(90, color=GRAY, ls=":", lw=1, zorder=2)
ax0.text(4.7, 91.5, "90%", fontsize=7.5, color=GRAY)
for i, (b, v) in enumerate(zip(bars, evr * 100)):
    ax0.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=7)
ax0.set_xlabel("Principal component")
ax0.set_ylabel("Variance explained (%)")
ax0.set_title("Scree plot (5 outcome measures)", pad=5)
ax0.legend(loc="upper right", framealpha=0.9)
ax0.set_xticks(range(1, 6))
ax0.set_ylim(0, max(cumvar) + 5)
ax0.spines[["top", "right"]].set_visible(False)

# biplot (PC1 × PC2)
ax1 = fig.add_subplot(gs[1])
ax1.scatter(scores[:, 0], scores[:, 1], color=BLUE, s=28, alpha=0.65, zorder=3)
loadings = pca.components_[:2].T
scale = 2.5
for i, (lx, ly) in enumerate(loadings):
    ax1.annotate("", xy=(lx * scale, ly * scale), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    offset = 0.12
    ax1.text(lx * scale * (1 + offset), ly * scale * (1 + offset),
             outcome_labels[i], fontsize=7.5, color=RED, ha="center",
             va="center")
ax1.axhline(0, color=GRAY, lw=0.6, ls="--")
ax1.axvline(0, color=GRAY, lw=0.6, ls="--")
ax1.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
ax1.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
ax1.set_title("Outcome PCA biplot", pad=5)
ax1.spines[["top", "right"]].set_visible(False)

plt.savefig(os.path.join(OUTDIR, "FigS_gap4_outcome_pca.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUTDIR, "FigS_gap4_outcome_pca.png"), dpi=300, bbox_inches="tight")
plt.close()
print("GAP 4 saved")
print(f"  PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, "
      f"PC3={evr[2]*100:.1f}%  → cumulative 3-PC: {sum(evr[:3])*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# GAP 5 — 5×5 correlation matrix heatmap
# ══════════════════════════════════════════════════════════════════════════
corr = df[outcome_cols].corr(method="pearson")
n = len(df)

# compute p-values
pval = pd.DataFrame(np.ones((5, 5)), index=outcome_cols, columns=outcome_cols)
for i in range(5):
    for j in range(5):
        if i != j:
            r, p = stats.pearsonr(df[outcome_cols[i]], df[outcome_cols[j]])
            pval.iloc[i, j] = p

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(outcome_labels, rotation=30, ha="right", fontsize=8)
ax.set_yticklabels(outcome_labels, fontsize=8)

# annotate cells
for i in range(5):
    for j in range(5):
        r_val = corr.iloc[i, j]
        p_val = pval.iloc[i, j]
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
               ("*" if p_val < 0.05 else ""))
        color = "white" if abs(r_val) > 0.6 else "black"
        ax.text(j, i, f"{r_val:.2f}{star}", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold" if star else "normal")

mean_abs_r = (corr.abs().values[np.triu_indices(5, k=1)]).mean()
ax.set_title(f"Resilience outcome correlation matrix ($n={n}$ cities)\n"
             f"Mean $|r|$ (off-diagonal) = {mean_abs_r:.2f}  |  * $p<0.05$, ** $p<0.01$, *** $p<0.001$",
             pad=8, fontsize=8.5)
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "FigS_gap5_outcome_corrmatrix.pdf"), dpi=300)
plt.savefig(os.path.join(OUTDIR, "FigS_gap5_outcome_corrmatrix.png"), dpi=300)
plt.close()
print(f"GAP 5 saved  (mean |r| = {mean_abs_r:.3f})")

print("\nAll gap figures written to:", OUTDIR)
