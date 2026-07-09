"""
JSD Score + AUROC + Confusion Matrix Report
============================================
Computes the JSD (Jensen-Shannon Divergence) perturbation score for every
.pt file in the scoring folder and produces TWO independent reports
(each with its own CSV + PNG):

  1. FULL dataset report      -> uses every labelled file found.
  2. BALANCED dataset report  -> actives and decoys are forced to the same
                                 count. If one class has more samples than
                                 the other (typically decoys), a random
                                 subset of that class is drawn (without
                                 replacement) to match the size of the
                                 smaller class. Sampling is seeded for
                                 reproducibility (--balance-seed).

For each report, only two things are shown:
  - ROC curve + AUROC (with bootstrap 95% CI) for the JSD score
  - Confusion matrix at the Youden-optimal threshold

Labels:
  label = 1  if filename contains the active keyword  (default: "active")
  label = 0  if filename contains the decoy  keyword  (default: "decoy")

Duplicate compounds (same CHEMBL/ZINC ID) are automatically removed,
keeping the pose with the lowest JSD score.

Usage
-----
    python jsd_auroc_report.py --scoring-dir <DIR> --reference <FILE_OR_DIR> \\
        --out-prefix <PATH_PREFIX> [--balance-seed 42] [--bootstrap-n 2000]

    This produces:
        <PATH_PREFIX>_full.csv       <PATH_PREFIX>_full.png
        <PATH_PREFIX>_balanced.csv   <PATH_PREFIX>_balanced.png

    # Run with no args to use the hard-coded defaults below.
    python jsd_auroc_report.py
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
from scipy.spatial.distance import jensenshannon
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

DEFAULT_ACTIVE_KEYWORD = "active"   # label = 1
DEFAULT_DECOY_KEYWORD  = "decoy"    # label = 0

DEFAULT_OUT_PREFIX = "/home/tedeschg/prj/protein-perturbation/experiments/dude_experiments/cdk2/reports/jsd_report"

DEFAULT_BOOTSTRAP_N_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_CI          = 0.95
DEFAULT_BOOTSTRAP_SEED        = 42
DEFAULT_BALANCE_SEED          = 42

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────

C_ACTIVE  = "#2E86AB"   # steel blue
C_DECOY   = "#E84855"   # crimson
C_THRESH  = "#F4A261"   # amber
C_RANDOM  = "#AAAAAA"   # grey

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
    probs = _to_numpy(data["probs"]).astype(float)  # (R, N, 21)
    return {"probs": probs.mean(axis=0)}            # (N, 21)


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
            "probs":   np.mean([d["probs"] for d in loaded], axis=0),
            "n_files": len(pt_files),
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
    labels: list, scores: list,
    n_resamples: int = 2000, ci: float = 0.95, seed: int = 42,
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
        youden_idx=youden_idx, best_thresh=best_thresh,
        acc=accuracy_score(labels, preds),
        bacc=balanced_accuracy_score(labels, preds),
        prec=precision_score(labels, preds, zero_division=0),
        rec=recall_score(labels, preds, zero_division=0),
        spec=spec,
        f1=f1_score(labels, preds, zero_division=0),
        mcc=matthews_corrcoef(labels, preds),
        tp=int(tp), fp=int(fp_val), tn=int(tn), fn=int(fn),
    )


def compute_auroc_block(
    df_labeled: pd.DataFrame, score_col: str,
    bootstrap_n: int, bootstrap_ci: float, bootstrap_seed: int,
) -> dict:
    """Higher JSD score => more decoy-like, so we negate for roc_auc_score
    (which expects higher score => label 1 / active)."""
    labels     = df_labeled["label"].astype(int).tolist()
    scores_raw = df_labeled[score_col].tolist()
    scores_for_roc = [-s for s in scores_raw]

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


def balance_dataset(df_labeled: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a copy of df_labeled where actives and decoys have equal count.
    The majority class is downsampled by random sampling (no replacement)."""
    rng = np.random.default_rng(seed)

    actives = df_labeled[df_labeled["label"] == 1]
    decoys  = df_labeled[df_labeled["label"] == 0]
    n_act, n_dec = len(actives), len(decoys)
    n_target = min(n_act, n_dec)

    if n_act == 0 or n_dec == 0:
        return df_labeled.iloc[0:0].copy()  # empty, can't balance

    if n_act > n_target:
        idx = rng.choice(actives.index.to_numpy(), size=n_target, replace=False)
        actives = actives.loc[idx]
    if n_dec > n_target:
        idx = rng.choice(decoys.index.to_numpy(), size=n_target, replace=False)
        decoys = decoys.loc[idx]

    return (pd.concat([actives, decoys])
              .sort_values("file")
              .reset_index(drop=True))

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _roc_panel(ax, fpr, tpr, auroc, ci_lower, ci_upper, ci_pct, youden_idx):
    ax.fill_between(fpr, tpr, alpha=0.09, color=C_ACTIVE)
    ax.plot(fpr, tpr, color=C_ACTIVE, linewidth=2.2, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color=C_RANDOM, linewidth=1.2, zorder=2)
    ax.scatter(fpr[youden_idx], tpr[youden_idx],
               color=C_THRESH, s=110, zorder=5,
               edgecolors="white", linewidth=1.2, label="Youden point")
    ax.set_xlabel("False Positive Rate", fontsize=11, color="#444444")
    ax.set_ylabel("True Positive Rate",  fontsize=11, color="#444444")
    ax.set_title("ROC — JSD Score", fontsize=13, fontweight="bold", color="#222222", pad=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.22, color="#CCCCCC")
    ax.tick_params(colors="#555555")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85, edgecolor="#CCCCCC")
    ax.text(0.97, 0.08,
            f"AUROC = {auroc:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}] {ci_pct}% CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))


def _confusion_matrix_panel(ax, metrics: dict):
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])
    im = ax.imshow(cm, cmap="Purples", vmin=0)

    labels = ["Decoy (0)", "Active (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10, color="#444444")
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10, color="#444444")
    ax.set_xlabel("Predicted", fontsize=11, color="#444444")
    ax.set_ylabel("True", fontsize=11, color="#444444")
    ax.set_title("Confusion Matrix (Youden threshold)", fontsize=13,
                 fontweight="bold", color="#222222", pad=8)

    cm_max = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm_max * 0.55 else "#222222"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    txt = (
        f"Threshold (raw JSD) = {-metrics['best_thresh']:.4f}\n"
        f"Accuracy            = {metrics['acc']:.4f}\n"
        f"Balanced Accuracy    = {metrics['bacc']:.4f}\n"
        f"Precision (PPV)      = {metrics['prec']:.4f}\n"
        f"Recall (Sensitivity) = {metrics['rec']:.4f}\n"
        f"Specificity          = {metrics['spec']:.4f}\n"
        f"F1-score             = {metrics['f1']:.4f}\n"
        f"MCC                  = {metrics['mcc']:.4f}"
    )
    ax.text(1.35, 0.5, txt, transform=ax.transAxes, ha="left", va="center",
            fontsize=9.5, color="#222222", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA",
                      edgecolor="#CCCCCC"))


def plot_jsd_report(df_labeled: pd.DataFrame, result: dict, ci_pct: int,
                     title_suffix: str, out_path: str) -> None:
    n_act = int((df_labeled["label"] == 1).sum())
    n_dec = int((df_labeled["label"] == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#F8F9FA")
    for ax in axes:
        ax.set_facecolor("#F8F9FA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#DDDDDD")

    _roc_panel(axes[0], result["fpr"], result["tpr"], result["auroc"],
               result["ci_lower"], result["ci_upper"], ci_pct,
               result["metrics"]["youden_idx"])
    _confusion_matrix_panel(axes[1], result["metrics"])

    fig.suptitle(
        f"JSD Score — AUROC & Confusion Matrix  ({title_suffix})\n"
        f"n_active={n_act}  n_decoy={n_dec}  ·  AUROC={result['auroc']:.3f}",
        fontsize=13, fontweight="bold", color="#111111", y=1.02,
    )

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  -> Figure saved to: {out_path}")
    plt.close(fig)


def print_auroc_block(result: dict, label: str, ci_pct: int) -> None:
    sep = "-" * 52
    m   = result["metrics"]
    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  {sep}")
    print(f"  {'AUROC':<24s}: {result['auroc']:.4f}")
    print(f"  {'Bootstrap CI':<24s}: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  ({ci_pct}%)")
    print(f"  {'Youden threshold':<24s}: {-m['best_thresh']:.5f}")
    print(f"  {sep}")
    print(f"  {'Accuracy':<24s}: {m['acc']:.4f}")
    print(f"  {'Balanced Accuracy':<24s}: {m['bacc']:.4f}")
    print(f"  {'Precision (PPV)':<24s}: {m['prec']:.4f}")
    print(f"  {'Recall (Sensitivity)':<24s}: {m['rec']:.4f}")
    print(f"  {'Specificity':<24s}: {m['spec']:.4f}")
    print(f"  {'F1-score':<24s}: {m['f1']:.4f}")
    print(f"  {'MCC':<24s}: {m['mcc']:.4f}")
    print(f"  {sep}")
    print(f"  Confusion matrix  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"  {sep}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JSD-only AUROC + confusion matrix report, full & balanced datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scoring-dir",    default=DEFAULT_SCORING_DIR)
    p.add_argument("--reference",      default=DEFAULT_REFERENCE_PATH)
    p.add_argument("--out-prefix",     default=DEFAULT_OUT_PREFIX,
                   help="Prefix for output files: <prefix>_full.csv/.png and <prefix>_balanced.csv/.png")
    p.add_argument("--active-keyword", default=DEFAULT_ACTIVE_KEYWORD)
    p.add_argument("--decoy-keyword",  default=DEFAULT_DECOY_KEYWORD)
    p.add_argument("--bootstrap-n",    type=int,   default=DEFAULT_BOOTSTRAP_N_RESAMPLES)
    p.add_argument("--bootstrap-ci",   type=float, default=DEFAULT_BOOTSTRAP_CI)
    p.add_argument("--bootstrap-seed", type=int,   default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--balance-seed",   type=int,   default=DEFAULT_BALANCE_SEED,
                   help="Random seed used to subsample the majority class for the balanced report")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_report(df_labeled: pd.DataFrame, ci_pct: int, args, title_suffix: str,
               csv_path: str, png_path: str) -> None:
    if len(df_labeled) == 0 or df_labeled["label"].nunique() < 2:
        print(f"  [SKIP] '{title_suffix}' report: need at least one active and one decoy.")
        return

    pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(csv_path, index=False)
    print(f"  -> Scores saved to: {csv_path}")

    result = compute_auroc_block(
        df_labeled, "jsd_score",
        bootstrap_n=args.bootstrap_n, bootstrap_ci=args.bootstrap_ci,
        bootstrap_seed=args.bootstrap_seed,
    )
    print_auroc_block(result, f"JSD Score — {title_suffix}", ci_pct)
    plot_jsd_report(df_labeled, result, ci_pct, title_suffix, png_path)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" JSD Score — AUROC + Confusion Matrix Report (full + balanced)")
    print("=" * 60)

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

    # ── 3. Compute JSD scores ────────────────────────────────────────────────
    print("\n[3/4] Computing JSD scores...")
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
        label_str = "active" if label == 1 else "decoy" if label == 0 else "unknown"
        print(f"  OK  {fname:<55s}  JSD={s_jsd:7.4f}  [{label_str}]")

        rows.append({
            "file":       fname,
            "label":      label,
            "label_str":  label_str,
            "jsd_score":  s_jsd,
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

    df_labeled = df[df["label"].notna()].copy()

    ci_pct = int(args.bootstrap_ci * 100)

    # ── 4. FULL dataset report ───────────────────────────────────────────────
    print("\n[4/4] Building reports...")
    print("\n  === FULL DATASET ===")
    n_act_full = int((df_labeled["label"] == 1).sum())
    n_dec_full = int((df_labeled["label"] == 0).sum())
    print(f"  n_active={n_act_full}  n_decoy={n_dec_full}")
    run_report(
        df_labeled, ci_pct, args, "Full dataset",
        csv_path=f"{args.out_prefix}_full.csv",
        png_path=f"{args.out_prefix}_full.png",
    )

    # ── BALANCED dataset report ──────────────────────────────────────────────
    print("\n  === BALANCED DATASET (random subsample of majority class) ===")
    df_balanced = balance_dataset(df_labeled, seed=args.balance_seed)
    n_act_bal = int((df_balanced["label"] == 1).sum())
    n_dec_bal = int((df_balanced["label"] == 0).sum())
    print(f"  balance-seed={args.balance_seed}  ->  n_active={n_act_bal}  n_decoy={n_dec_bal}")
    run_report(
        df_balanced, ci_pct, args, "Balanced dataset",
        csv_path=f"{args.out_prefix}_balanced.csv",
        png_path=f"{args.out_prefix}_balanced.png",
    )

    if skipped:
        print(f"\n  Skipped files ({len(skipped)}): {skipped}")

    print("\nDone.")


if __name__ == "__main__":
    main()