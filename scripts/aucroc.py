"""
Perturbation Score + AUROC Calculator
======================================
Computes THREE scores for every .pt file in the scoring folder:

  1. JSD Score    — mean Jensen-Shannon Divergence between per-residue AA
                    probability distributions of query vs reference.
                    Higher = more perturbed = more decoy-like.

  2. NLL Score    — negative mean log-likelihood of the native sequence given
                    the query ligand.  Active → low NLL.  Decoy → high NLL.

  3. Combined Score — NLL as base, JSD as emphasis factor:

                    Combined = NLL × (1 + α × JSD_norm)

                    where JSD_norm = JSD / max(JSD) across the dataset.
                    α (default 1.0) controls JSD amplification strength.

                    Properties:
                      α = 0  → Combined = NLL (no emphasis)
                      α = 1  → JSD can at most double the NLL signal
                    Actives (low NLL, low JSD)  → low combined score
                    Decoys  (high NLL, high JSD) → doubly penalised

Labels:
  label = 1  if filename contains the active keyword  (default: "active")
  label = 0  if filename contains the decoy  keyword  (default: "decoy")

Duplicate compounds (same CHEMBL/ZINC ID) are automatically removed,
keeping the pose with the lowest JSD score.

Classification metrics are computed at the Youden-optimal threshold.
Bootstrap 95 % CI on AUROC (default 2000 resamples).

Usage
-----
    python aucroc.py --scoring-dir <DIR> --reference <FILE_OR_DIR> \\
                     --out-csv <CSV> --out-roc <PNG> [--combined-alpha 1.0]

    # Run with no args to use the ABL1 defaults hard-coded below.
    python aucroc.py
"""

import argparse
import re
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — always saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
)
import torch

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS  —  edit these to match your project
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SCORING_DIR = "/home/tedeschg/prj/protein-perturbation/experiments/dude_experiments/cdk2/output_cdk2_scoring"

DEFAULT_REFERENCE_PATH = (
    "/home/tedeschg/prj/protein-perturbation/experiments/dude_experiments/cdk2/output_cdk2_reference/1h00_protein_fap_1.pt"
)

DEFAULT_ACTIVE_KEYWORD   = "active"   # label = 1
DEFAULT_DECOY_KEYWORD    = "decoy"    # label = 0
DEFAULT_COMBINED_ALPHA   = 0.05        # JSD amplification strength

DEFAULT_OUT_CSV = "/home/tedeschg/prj/protein-perturbation/experiments/dude_experiments/cdk2/reports/perturbation_scores_all_abl1_focused.csv"
DEFAULT_OUT_ROC = "/home/tedeschg/prj/protein-perturbation/experiments/dude_experiments/cdk2/reports/roc_curve_all_abl1_focused.png"

DEFAULT_BOOTSTRAP_N_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_CI          = 0.95
DEFAULT_BOOTSTRAP_SEED        = 42

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────

C_ACTIVE   = "#2E86AB"   # steel blue
C_DECOY    = "#E84855"   # crimson
C_THRESH   = "#F4A261"   # amber
C_RANDOM   = "#AAAAAA"   # grey
C_COMBINED = "#6A0572"   # purple  (used for combined score panels)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "numpy"):
        return tensor_or_array.numpy()
    return np.array(tensor_or_array)


def load_pt(path: str) -> dict:
    """Load a .pt file and return a dict with numpy arrays averaged over replicas."""
    data = torch.load(path, map_location="cpu", weights_only=False)

    probs      = _to_numpy(data["probs"]).astype(float)         # (R, N, 21)
    log_probs  = _to_numpy(data["log_probs"]).astype(float)     # (R, N, 21)
    native_seq = _to_numpy(data["native_sequence"]).astype(int) # (N,)

    return {
        "probs":      probs.mean(axis=0),      # (N, 21)
        "log_probs":  log_probs.mean(axis=0),  # (N, 21)
        "native_seq": native_seq,              # (N,)
    }


def load_reference(ref_path: str) -> dict:
    """Load the reference .pt (or all .pt in a folder, averaged)."""
    p = pathlib.Path(ref_path)

    if p.is_file():
        ref = load_pt(str(p))
        ref["n_files"] = 1
        return ref

    if p.is_dir():
        pt_files = sorted(p.glob("*.pt"))
        if not pt_files:
            sys.exit(f"ERROR: no .pt files found in reference folder: {ref_path}")
        loaded = [load_pt(str(f)) for f in pt_files]
        return {
            "probs":      np.mean([d["probs"]     for d in loaded], axis=0),
            "log_probs":  np.mean([d["log_probs"] for d in loaded], axis=0),
            "native_seq": loaded[0]["native_seq"],
            "n_files":    len(pt_files),
        }

    sys.exit(f"ERROR: --reference path does not exist: {ref_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Score computation
# ──────────────────────────────────────────────────────────────────────────────

def jsd_per_residue(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.array([jensenshannon(P[i], Q[i], base=2) for i in range(len(P))])


def jsd_score(P_ref: np.ndarray, Q: np.ndarray) -> float:
    """mean JSD × 100  (range 0–100).  Higher = more decoy-like."""
    return float(jsd_per_residue(P_ref, Q).mean() * 100)


def nll_score(log_probs: np.ndarray, native_seq: np.ndarray) -> float:
    """−mean log P(native | ligand).  Higher = more decoy-like."""
    native_lls = log_probs[np.arange(len(native_seq)), native_seq]
    return float(-native_lls.mean())


def compute_combined_scores(
    jsd_scores: np.ndarray,
    nll_scores: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Combined score: NLL as base, JSD as emphasis factor.

        Combined = NLL × (1 + α × JSD_norm)

    where  JSD_norm = JSD / max(JSD)  across the dataset  →  JSD_norm ∈ [0, 1].

    Properties
    ----------
    α = 0  → Combined = NLL  (JSD has no effect)
    α = 1  → JSD can at most double the NLL contribution (default)
    α > 1  → JSD amplifies NLL even more strongly

    Both actives and decoys are ranked in the same direction:
    higher Combined score = more decoy-like.
    """
    jsd_max = jsd_scores.max()
    if jsd_max == 0:
        return nll_scores.copy()
    jsd_norm = jsd_scores / jsd_max          # [0, 1]
    return nll_scores * (1.0 + alpha * jsd_norm)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def assign_label(filename: str, active_kw: str, decoy_kw: str):
    name = filename.lower()
    if active_kw in name:
        return 1
    if decoy_kw in name:
        return 0
    return None


def find_missing_indices(files: list, keyword: str) -> list:
    group = [pathlib.Path(f).name for f in files if keyword in pathlib.Path(f).name.lower()]
    found = set()
    for name in group:
        m = re.search(rf"{keyword}s?_(\d+)_", name, re.IGNORECASE)
        if m:
            found.add(int(m.group(1)))
    if not found:
        return []
    return sorted(set(range(min(found), max(found) + 1)) - found)


def extract_compound_id(fname: str) -> str:
    m = re.search(r"(CHEMBL\d+|ZINC\d+)", fname, re.IGNORECASE)
    return m.group(1).upper() if m else fname


def bootstrap_auroc_ci(
    labels: list,
    scores: list,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    rng      = np.random.default_rng(seed)
    labels_a = np.array(labels)
    scores_a = np.array(scores)
    n        = len(labels_a)
    boot: list = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y, s = labels_a[idx], scores_a[idx]
        if len(np.unique(y)) < 2:
            continue
        boot.append(roc_auc_score(y, s))

    alpha = 1.0 - ci
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def classification_metrics(labels: list, scores: list, fpr, tpr, thresh) -> dict:
    youden_idx  = int(np.argmax(tpr - fpr))
    best_thresh = thresh[youden_idx]
    preds       = [1 if s >= best_thresh else 0 for s in scores]

    tn, fp_val, fn, tp = confusion_matrix(labels, preds).ravel()
    spec = tn / (tn + fp_val) if (tn + fp_val) > 0 else 0.0

    return dict(
        youden_idx  = youden_idx,
        best_thresh = best_thresh,
        acc         = accuracy_score(labels, preds),
        bacc        = balanced_accuracy_score(labels, preds),
        prec        = precision_score(labels, preds, zero_division=0),
        rec         = recall_score(labels, preds, zero_division=0),
        spec        = spec,
        f1          = f1_score(labels, preds, zero_division=0),
        mcc         = matthews_corrcoef(labels, preds),
        tp=int(tp), fp=int(fp_val), tn=int(tn), fn=int(fn),
    )

# ──────────────────────────────────────────────────────────────────────────────
# AUROC pipeline
# ──────────────────────────────────────────────────────────────────────────────

def compute_auroc_block(
    df_labeled: pd.DataFrame, score_col: str,
    higher_is_positive: bool,
    bootstrap_n: int, bootstrap_ci: float, bootstrap_seed: int,
) -> dict:
    labels     = df_labeled["label"].astype(int).tolist()
    scores_raw = df_labeled[score_col].tolist()
    # negate so that roc_auc_score interprets higher = active (label=1)
    scores_for_roc = [-s for s in scores_raw] if higher_is_positive else scores_raw

    auroc              = roc_auc_score(labels, scores_for_roc)
    fpr, tpr, thresh   = roc_curve(labels, scores_for_roc)
    ci_lower, ci_upper = bootstrap_auroc_ci(
        labels, scores_for_roc,
        n_resamples=bootstrap_n, ci=bootstrap_ci, seed=bootstrap_seed,
    )
    metrics = classification_metrics(labels, scores_for_roc, fpr, tpr, thresh)

    return dict(
        auroc=auroc, fpr=fpr, tpr=tpr, thresh=thresh,
        ci_lower=ci_lower, ci_upper=ci_upper, metrics=metrics,
    )

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute JSD, NLL, and Combined perturbation scores + AUROC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scoring-dir",    default=DEFAULT_SCORING_DIR)
    p.add_argument("--reference",      default=DEFAULT_REFERENCE_PATH)
    p.add_argument("--out-csv",        default=DEFAULT_OUT_CSV)
    p.add_argument("--out-roc",        default=DEFAULT_OUT_ROC)
    p.add_argument("--active-keyword", default=DEFAULT_ACTIVE_KEYWORD)
    p.add_argument("--decoy-keyword",  default=DEFAULT_DECOY_KEYWORD)
    p.add_argument("--combined-alpha", type=float, default=DEFAULT_COMBINED_ALPHA,
                   help="JSD amplification factor α in Combined = NLL × (1 + α × JSD_norm)")
    p.add_argument("--bootstrap-n",    type=int,   default=DEFAULT_BOOTSTRAP_N_RESAMPLES)
    p.add_argument("--bootstrap-ci",   type=float, default=DEFAULT_BOOTSTRAP_CI)
    p.add_argument("--bootstrap-seed", type=int,   default=DEFAULT_BOOTSTRAP_SEED)
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _violin_panel(
    ax, actives: np.ndarray, decoys: np.ndarray,
    youden_score: float, ylabel: str, title: str,
    active_color: str = C_ACTIVE, decoy_color: str = C_DECOY,
) -> None:
    rng = np.random.default_rng(0)

    for pos, data, color in [
        (1, actives, active_color),
        (2, decoys,  decoy_color),
    ]:
        vp = ax.violinplot(data, positions=[pos], widths=0.55,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.28)
            body.set_edgecolor(color)
            body.set_linewidth(1.0)

        ax.hlines(np.median(data), pos - 0.18, pos + 0.18,
                  colors=color, linewidth=2.2, zorder=4)

        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(np.full(len(data), pos) + jitter, data,
                   color=color, s=32, alpha=0.60, zorder=3,
                   edgecolors="white", linewidth=0.4)

    ax.axhline(youden_score, color=C_THRESH, linestyle="--",
               linewidth=1.5, zorder=5, label=f"Youden = {youden_score:.3f}")

    _, pval = mannwhitneyu(actives, decoys, alternative="two-sided")
    pval_str = f"p = {pval:.2e}" if pval >= 1e-4 else "p < 1e-4"
    y_top = max(actives.max(), decoys.max())
    ax.annotate("", xy=(2, y_top * 1.04), xytext=(1, y_top * 1.04),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0))
    ax.text(1.5, y_top * 1.05, pval_str, ha="center", va="bottom",
            fontsize=9, color="#555555")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Active\n(n={len(actives)})", f"Decoy\n(n={len(decoys)})"],
                       fontsize=11, color="#444444")
    ax.set_ylabel(ylabel, fontsize=11, color="#444444")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#222222", pad=8)
    ax.set_xlim([0.5, 2.5])
    ax.grid(True, alpha=0.22, color="#CCCCCC", axis="y")
    ax.tick_params(colors="#555555")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="#CCCCCC")


def _roc_panel(
    ax, fpr, tpr, auroc, ci_lower, ci_upper, ci_pct,
    youden_idx, score_name: str,
    line_color: str = C_ACTIVE,
) -> None:
    ax.fill_between(fpr, tpr, alpha=0.09, color=line_color)
    ax.plot(fpr, tpr, color=line_color, linewidth=2.2, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color=C_RANDOM, linewidth=1.2, zorder=2)
    ax.scatter(fpr[youden_idx], tpr[youden_idx],
               color=C_THRESH, s=100, zorder=5,
               edgecolors="white", linewidth=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=11, color="#444444")
    ax.set_ylabel("True Positive Rate",  fontsize=11, color="#444444")
    ax.set_title(f"ROC — {score_name}", fontsize=12, fontweight="bold",
                 color="#222222", pad=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.22, color="#CCCCCC")
    ax.tick_params(colors="#555555")
    ax.text(0.97, 0.08,
            f"AUROC = {auroc:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}] {ci_pct}% CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.88))


def _metrics_panel(
    ax, metrics: dict, auroc: float,
    ci_lower: float, ci_upper: float,
    ci_pct: int, score_name: str,
) -> None:
    ax.axis("off")
    ax.set_title(f"Metrics — {score_name}", fontsize=12, fontweight="bold",
                 color="#222222", pad=8)

    rows = [
        ("AUROC",              f"{auroc:.4f}"),
        (f"  CI {ci_pct}% lower", f"{ci_lower:.4f}"),
        (f"  CI {ci_pct}% upper", f"{ci_upper:.4f}"),
        ("─" * 24, "─" * 8),
        ("Youden threshold",   f"{metrics['best_thresh']:.4f}"),
        ("─" * 24, "─" * 8),
        ("Accuracy",           f"{metrics['acc']:.4f}"),
        ("Balanced Accuracy",  f"{metrics['bacc']:.4f}"),
        ("Precision (PPV)",    f"{metrics['prec']:.4f}"),
        ("Recall (TPR)",       f"{metrics['rec']:.4f}"),
        ("Specificity (TNR)",  f"{metrics['spec']:.4f}"),
        ("F1-score",           f"{metrics['f1']:.4f}"),
        ("MCC",                f"{metrics['mcc']:.4f}"),
        ("─" * 24, "─" * 8),
        ("TP / FP / TN / FN",
         f"{metrics['tp']} / {metrics['fp']} / {metrics['tn']} / {metrics['fn']}"),
    ]

    bold = {"AUROC", "MCC", "Balanced Accuracy"}
    y, dy = 0.96, 0.063

    for i, (lbl, val) in enumerate(rows):
        is_sep = lbl.startswith("─")
        color  = "#AAAAAA" if is_sep else "#333333"
        weight = "bold" if lbl in bold else "normal"
        y_pos  = y - i * dy
        ax.text(0.02, y_pos, lbl, transform=ax.transAxes,
                fontsize=9, color=color, fontweight=weight,
                va="top", fontfamily="monospace")
        ax.text(0.75, y_pos, val, transform=ax.transAxes,
                fontsize=9, color=color, fontweight=weight,
                va="top", ha="right", fontfamily="monospace")


def plot_results(
    df_labeled: pd.DataFrame,
    jsd_result: dict, nll_result: dict, combined_result: dict,
    ci_pct: int, alpha: float, out_path: str,
) -> None:
    """
    3×3 figure:
      Row 0: ROC (JSD)      | violin (JSD)      | metrics (JSD)
      Row 1: ROC (NLL)      | violin (NLL)      | metrics (NLL)
      Row 2: ROC (Combined) | violin (Combined) | metrics (Combined)
    """
    actives_jsd = df_labeled[df_labeled["label"] == 1]["jsd_score"].values
    decoys_jsd  = df_labeled[df_labeled["label"] == 0]["jsd_score"].values
    actives_nll = df_labeled[df_labeled["label"] == 1]["nll_score"].values
    decoys_nll  = df_labeled[df_labeled["label"] == 0]["nll_score"].values
    actives_com = df_labeled[df_labeled["label"] == 1]["combined_score"].values
    decoys_com  = df_labeled[df_labeled["label"] == 0]["combined_score"].values

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor("#F8F9FA")

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=0.91,  bottom=0.05,
                           wspace=0.32, hspace=0.50)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
    for row in axes:
        for ax in row:
            ax.set_facecolor("#F8F9FA")
            for spine in ax.spines.values():
                spine.set_edgecolor("#DDDDDD")

    # ── Row 0: JSD ────────────────────────────────────────────────────────────
    jsd_thresh = -jsd_result["metrics"]["best_thresh"]
    _roc_panel(axes[0][0],
               jsd_result["fpr"], jsd_result["tpr"],
               jsd_result["auroc"], jsd_result["ci_lower"], jsd_result["ci_upper"],
               ci_pct, jsd_result["metrics"]["youden_idx"], "JSD Score")
    _violin_panel(axes[0][1], actives_jsd, decoys_jsd,
                  jsd_thresh, "JSD Score (mean JSD × 100)",
                  "Score Distribution — JSD")
    _metrics_panel(axes[0][2], jsd_result["metrics"],
                   jsd_result["auroc"], jsd_result["ci_lower"], jsd_result["ci_upper"],
                   ci_pct, "JSD Score")

    # ── Row 1: NLL ────────────────────────────────────────────────────────────
    nll_thresh = -nll_result["metrics"]["best_thresh"]
    _roc_panel(axes[1][0],
               nll_result["fpr"], nll_result["tpr"],
               nll_result["auroc"], nll_result["ci_lower"], nll_result["ci_upper"],
               ci_pct, nll_result["metrics"]["youden_idx"], "NLL Score")
    _violin_panel(axes[1][1], actives_nll, decoys_nll,
                  nll_thresh, "NLL Score (−mean log P native)",
                  "Score Distribution — NLL")
    _metrics_panel(axes[1][2], nll_result["metrics"],
                   nll_result["auroc"], nll_result["ci_lower"], nll_result["ci_upper"],
                   ci_pct, "NLL Score")

    # ── Row 2: Combined ───────────────────────────────────────────────────────
    combined_thresh = -combined_result["metrics"]["best_thresh"]
    combined_label  = f"Combined (α={alpha})"
    _roc_panel(axes[2][0],
               combined_result["fpr"], combined_result["tpr"],
               combined_result["auroc"], combined_result["ci_lower"], combined_result["ci_upper"],
               ci_pct, combined_result["metrics"]["youden_idx"], combined_label,
               line_color=C_COMBINED)
    _violin_panel(axes[2][1], actives_com, decoys_com,
                  combined_thresh,
                  f"Combined Score  NLL×(1+{alpha}×JSD_norm)",
                  f"Score Distribution — {combined_label}",
                  active_color=C_COMBINED, decoy_color=C_DECOY)
    _metrics_panel(axes[2][2], combined_result["metrics"],
                   combined_result["auroc"], combined_result["ci_lower"], combined_result["ci_upper"],
                   ci_pct, combined_label)

    fig.suptitle(
        f"Protein Perturbation Analysis  ·  "
        f"JSD AUROC={jsd_result['auroc']:.3f}  ·  "
        f"NLL AUROC={nll_result['auroc']:.3f}  ·  "
        f"Combined AUROC={combined_result['auroc']:.3f}  (α={alpha})",
        fontsize=13, fontweight="bold", color="#111111", y=0.97,
    )

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  → Figure saved to: {out_path}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_auroc_block(result: dict, label: str, ci_pct: int) -> None:
    sep = "─" * 52
    m   = result["metrics"]
    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  {sep}")
    print(f"  {'AUROC':<34s}: {result['auroc']:.4f}")
    print(f"  {'Bootstrap CI':<34s}: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  ({ci_pct}%)")
    print(f"  {sep}")
    print(f"  {'Youden threshold':<34s}: {m['best_thresh']:.5f}")
    print(f"  {sep}")
    print(f"  {'Accuracy':<34s}: {m['acc']:.4f}")
    print(f"  {'Balanced Accuracy':<34s}: {m['bacc']:.4f}")
    print(f"  {'Precision (PPV)':<34s}: {m['prec']:.4f}")
    print(f"  {'Recall (Sensitivity)':<34s}: {m['rec']:.4f}")
    print(f"  {'Specificity':<34s}: {m['spec']:.4f}")
    print(f"  {'F1-score':<34s}: {m['f1']:.4f}")
    print(f"  {'MCC':<34s}: {m['mcc']:.4f}")
    print(f"  {sep}")
    print(f"  Confusion matrix  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"  {sep}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" Perturbation Score + AUROC Calculator  (JSD + NLL + Combined)")
    print("=" * 60)
    print(f"  Combined formula: NLL × (1 + {args.combined_alpha} × JSD_norm)")

    # ── 1. Load reference ────────────────────────────────────────────────────
    print(f"\n[1/4] Loading reference: {args.reference}")
    try:
        ref = load_reference(args.reference)
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit(f"ERROR loading reference: {exc}")

    print(f"      Reference source files : {ref['n_files']}")
    print(f"      Reference probs shape  : {ref['probs'].shape}")

    # ── 2. Find scoring files ────────────────────────────────────────────────
    pt_files = sorted(glob.glob(str(pathlib.Path(args.scoring_dir) / "*.pt")))
    if not pt_files:
        sys.exit(f"No .pt files found in: {args.scoring_dir}")
    print(f"\n[2/4] Found {len(pt_files)} .pt files in {args.scoring_dir}")

    missing_actives = find_missing_indices(pt_files, args.active_keyword)
    missing_decoys  = find_missing_indices(pt_files, args.decoy_keyword)

    if missing_actives or missing_decoys:
        print("\n" + "=" * 60)
        print("  !! WARNING — MISSING FILES DETECTED !!")
        print("=" * 60)
        if missing_actives:
            print(f"  [ACTIVE] {len(missing_actives)} missing — "
                  f"{', '.join(f'{i:02d}' for i in missing_actives)}")
        if missing_decoys:
            print(f"  [DECOY]  {len(missing_decoys)} missing — "
                  f"{', '.join(f'{i:02d}' for i in missing_decoys)}")
        print("=" * 60 + "\n")
    else:
        print("  [OK] No missing indices — all files present.")

    # ── 3. Compute JSD and NLL scores ────────────────────────────────────────
    print("\n[3/4] Computing JSD and NLL scores...")
    rows: list = []
    skipped: list = []

    for fp in pt_files:
        fname = pathlib.Path(fp).name
        label = assign_label(fname, args.active_keyword, args.decoy_keyword)

        try:
            q = load_pt(fp)
        except Exception as exc:
            print(f"  SKIP {fname}: {exc}")
            skipped.append(fname)
            continue

        if q["probs"].shape != ref["probs"].shape:
            print(f"  SKIP {fname}: shape {q['probs'].shape} != ref {ref['probs'].shape}")
            skipped.append(fname)
            continue

        s_jsd = jsd_score(ref["probs"], q["probs"])
        s_nll = nll_score(q["log_probs"], ref["native_seq"])

        label_str = "active" if label == 1 else "decoy" if label == 0 else "unknown"
        print(f"  ✓  {fname:<55s}  JSD={s_jsd:7.4f}  NLL={s_nll:7.4f}  [{label_str}]")

        rows.append({
            "file":       fname,
            "label":      label,
            "label_str":  label_str,
            "jsd_score":  s_jsd,
            "nll_score":  s_nll,
            "n_residues": q["probs"].shape[0],
        })

    if not rows:
        sys.exit("No files processed successfully.")

    df = pd.DataFrame(rows)

    # ── Duplicate removal ────────────────────────────────────────────────────
    df["compound_id"] = df["file"].apply(extract_compound_id)
    before = len(df)
    df = (df.sort_values("jsd_score")
            .drop_duplicates(subset="compound_id", keep="first")
            .sort_values("file")
            .reset_index(drop=True))
    removed = before - len(df)
    if removed > 0:
        print(f"\n  [DEDUP] {removed} duplicate(s) removed (kept lowest JSD per ID).")
    else:
        print("\n  [DEDUP] No duplicates found.")

    # ── Combined score ────────────────────────────────────────────────────────
    df["combined_score"] = compute_combined_scores(
        df["jsd_score"].values,
        df["nll_score"].values,
        alpha=args.combined_alpha,
    )
    jsd_max = df["jsd_score"].max()
    print(f"\n  Combined score formula: NLL × (1 + {args.combined_alpha} × JSD / {jsd_max:.4f})")

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"  → Scores saved to: {args.out_csv}")

    print("\n" + "─" * 80)
    print(df[["file", "label_str", "jsd_score", "nll_score", "combined_score"]].to_string(index=False))
    print("─" * 80)

    # ── 4. AUROC + metrics ───────────────────────────────────────────────────
    print("\n[4/4] Computing AUROC, bootstrap CI, and classification metrics...")

    df_labeled = df[df["label"].notna()].copy()
    if len(df_labeled) == 0:
        print("  No labelled files (active/decoy). AUROC cannot be computed.")
        return
    if df_labeled["label"].nunique() < 2:
        print("  Need at least one sample per class. AUROC cannot be computed.")
        return

    ci_pct = int(args.bootstrap_ci * 100)
    print(f"  Bootstrap CI: {args.bootstrap_n} resamples, {ci_pct}% CI…")

    jsd_result = compute_auroc_block(
        df_labeled, "jsd_score", higher_is_positive=True,
        bootstrap_n=args.bootstrap_n, bootstrap_ci=args.bootstrap_ci,
        bootstrap_seed=args.bootstrap_seed,
    )
    nll_result = compute_auroc_block(
        df_labeled, "nll_score", higher_is_positive=True,
        bootstrap_n=args.bootstrap_n, bootstrap_ci=args.bootstrap_ci,
        bootstrap_seed=args.bootstrap_seed,
    )
    combined_result = compute_auroc_block(
        df_labeled, "combined_score", higher_is_positive=True,
        bootstrap_n=args.bootstrap_n, bootstrap_ci=args.bootstrap_ci,
        bootstrap_seed=args.bootstrap_seed,
    )

    print_auroc_block(jsd_result,      "JSD Score",                       ci_pct)
    print_auroc_block(nll_result,      "NLL Score",                       ci_pct)
    print_auroc_block(combined_result, f"Combined Score  (α={args.combined_alpha})", ci_pct)

    total = len(df_labeled)
    n_act = (df_labeled["label"] == 1).sum()
    n_dec = (df_labeled["label"] == 0).sum()
    print(f"\n  Total labelled: {total}  (active={n_act}, decoy={n_dec})")

    if skipped:
        print(f"\n  Skipped files ({len(skipped)}): {skipped}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_results(
        df_labeled,
        jsd_result, nll_result, combined_result,
        ci_pct, args.combined_alpha, args.out_roc,
    )


if __name__ == "__main__":
    main()