"""
Perturbation Score + AUROC Calculator
======================================
Computes the Jensen-Shannon Divergence (JSD) perturbation score for every .pt file in the
specified folder, then calculates the AUROC using:
  - label = 1  if the filename contains "active"  (configurable keyword)
  - label = 0  if the filename contains "decoy"   (configurable keyword)

The reference (intact / real ligand) is set via REFERENCE_PATH (single .pt file
or a folder — all replicas and files are averaged). Duplicate compounds are
automatically removed, keeping the pose with the lowest perturbation score.

Classification metrics (Accuracy, Precision, Recall, F1, MCC, Balanced Accuracy)
are computed using the optimal Youden threshold (maximises TPR - FPR on the ROC curve).

Enrichment Factors at 1%, 5%, and 10% of the library are reported.
A bootstrap 95% confidence interval on the AUROC is computed (default 2000 resamples).

Usage:
    python auroc.py
"""

import re
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
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
# CONFIGURATION  ← edit here
# ──────────────────────────────────────────────────────────────────────────────

SCORING_DIR = "/home/tedeschg/prj/protein-perturbation/output_abl1_lmpnn_focused/scoring"

# REFERENCE_PATH can be:
#   - a single .pt file  → its replicas are averaged (mean over replica axis)
#   - a folder of .pt files → all files are loaded and their probs are averaged together
REFERENCE_PATH = (
    "/home/tedeschg/prj/protein-perturbation/experiments/output_abl1_reference/2hzi_clean_11.pt"
)

# Keywords to assign labels from filenames
ACTIVE_KEYWORD = "active"   # label = 1
DECOY_KEYWORD  = "decoy"    # label = 0

OUT_CSV = "/home/tedeschg/prj/protein-perturbation/perturbation_scores_all_abl1_focused.csv"
OUT_ROC = "/home/tedeschg/prj/protein-perturbation/roc_curve_all_abl1_focused.png"

# Bootstrap settings
BOOTSTRAP_N_RESAMPLES = 2000   # number of bootstrap iterations for AUROC CI
BOOTSTRAP_CI          = 0.95   # confidence level (e.g. 0.95 → 95% CI)
BOOTSTRAP_SEED        = 42     # random seed for reproducibility

# Enrichment Factor fractions
EF_FRACTIONS = [0.01, 0.05, 0.10]   # 1 %, 5 %, 10 %

# ──────────────────────────────────────────────────────────────────────────────


def load_probs(path: str) -> np.ndarray:
    """Load a single .pt file and return the mean probs across all replicas.

    Tensor shape: (n_replicas, n_residues, n_aa)
    Returns mean over replica axis → shape (n_residues, n_aa).
    """
    data  = torch.load(path, map_location="cpu", weights_only=False)
    probs = data["probs"]
    if hasattr(probs, "numpy"):
        probs = probs.numpy()
    probs = probs.astype(float)   # (n_replicas, n_residues, n_aa)
    return probs.mean(axis=0)     # (n_residues, n_aa)


def load_reference(ref_path: str) -> tuple[np.ndarray, int]:
    """Load the reference probs, supporting both a single .pt file and a folder.

    Single file → average over its replicas.
    Folder      → load every .pt inside, average each over replicas,
                  then average across files.
    Returns (probs array, number of source files).
    """
    p = pathlib.Path(ref_path)
    if p.is_file():
        return load_probs(str(p)), 1
    if p.is_dir():
        pt_files = sorted(p.glob("*.pt"))
        if not pt_files:
            sys.exit(f"ERROR: no .pt files found in reference folder: {ref_path}")
        all_probs = [load_probs(str(f)) for f in pt_files]
        return np.mean(all_probs, axis=0), len(pt_files)
    sys.exit(f"ERROR: REFERENCE_PATH does not exist: {ref_path}")


def jsd_per_residue(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon Divergence per residue (row-wise).

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5*(P+Q)

    Properties:
      - Symmetric:  JSD(P,Q) == JSD(Q,P)
      - Bounded:    always in [0, 1] (base-2 log)
      - No epsilon smoothing needed (M is never zero where P or Q > 0)

    Returns shape (n_residues,) with values in [0, 1].
    """
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.array([jensenshannon(P[i], Q[i], base=2) for i in range(len(P))])


def perturbation_score(jsd_values: np.ndarray) -> float:
    """Overall perturbation score = mean JSD * 100  (range: 0 – 100)."""
    return float(jsd_values.mean() * 100)


def assign_label(filename: str) -> int | None:
    """Returns 1 (active), 0 (decoy), or None if the file cannot be classified."""
    name = filename.lower()
    if ACTIVE_KEYWORD in name:
        return 1
    if DECOY_KEYWORD in name:
        return 0
    return None


def find_missing_indices(files: list[str], keyword: str) -> list[int]:
    """Return sorted list of integer indices absent in the file list for a given keyword."""
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
    """Extract compound ID (e.g. CHEMBL40557 or ZINC39482920) from a filename."""
    m = re.search(r"(CHEMBL\d+|ZINC\d+)", fname, re.IGNORECASE)
    return m.group(1).upper() if m else fname


# ──────────────────────────────────────────────────────────────────────────────
# NEW HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def enrichment_factor(
    labels: list[int],
    scores_neg: list[float],
    fraction: float,
) -> float:
    """Compute the Enrichment Factor at a given top fraction of the library.

    EF(x%) = (actives in top x% / total in top x%) / (total actives / total N)

    Parameters
    ----------
    labels      : ground-truth labels  (1 = active, 0 = decoy)
    scores_neg  : negated perturbation scores (higher = more likely active)
    fraction    : top fraction to consider, e.g. 0.01 for 1 %

    Returns
    -------
    EF value; 1.0 = random, higher = better.
    """
    n_total   = len(labels)
    n_top     = max(1, int(np.floor(fraction * n_total)))
    n_actives = sum(labels)

    if n_actives == 0 or n_actives == n_total:
        return float("nan")

    # Sort by descending score (highest score = predicted active)
    order     = np.argsort(scores_neg)[::-1]
    top_labels = np.array(labels)[order[:n_top]]

    actives_in_top   = top_labels.sum()
    random_expectation = fraction * n_actives   # = fraction * n_actives (same as n_top * base_rate)

    return float(actives_in_top / random_expectation) if random_expectation > 0 else float("nan")


def bootstrap_auroc_ci(
    labels: list[int],
    scores_neg: list[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate bootstrap confidence interval for the AUROC.

    Uses the percentile bootstrap (no bias-correction).

    Parameters
    ----------
    labels      : ground-truth labels  (1 = active, 0 = decoy)
    scores_neg  : negated perturbation scores
    n_resamples : number of bootstrap iterations
    ci          : desired confidence level (e.g. 0.95)
    seed        : random seed for reproducibility

    Returns
    -------
    (lower_bound, upper_bound) at the requested confidence level.
    """
    rng      = np.random.default_rng(seed)
    labels_a = np.array(labels)
    scores_a = np.array(scores_neg)
    n        = len(labels_a)
    boot_aurocs: list[float] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y   = labels_a[idx]
        s   = scores_a[idx]
        # Skip resamples where only one class is present
        if len(np.unique(y)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y, s))

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_aurocs, 100 * alpha / 2))
    upper = float(np.percentile(boot_aurocs, 100 * (1 - alpha / 2)))
    return lower, upper


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(" Perturbation Score + AUROC Calculator")
    print("=" * 60)

    # ── 1. Load reference ────────────────────────────────────────────────────
    print(f"\n[1/4] Loading reference: {REFERENCE_PATH}")
    try:
        P_ref, n_ref_files = load_reference(REFERENCE_PATH)
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit(f"ERROR loading reference: {exc}")
    print(f"      Reference source files : {n_ref_files}")
    print(f"      Reference probs shape  : {P_ref.shape}  (replicas already averaged)")

    # ── 2. Find scoring files ────────────────────────────────────────────────
    pt_files = sorted(glob.glob(str(pathlib.Path(SCORING_DIR) / "*.pt")))
    if not pt_files:
        sys.exit(f"No .pt files found in: {SCORING_DIR}")
    print(f"\n[2/4] Found {len(pt_files)} .pt files in {SCORING_DIR}")

    missing_actives = find_missing_indices(pt_files, ACTIVE_KEYWORD)
    missing_decoys  = find_missing_indices(pt_files, DECOY_KEYWORD)

    if missing_actives or missing_decoys:
        print("\n" + "=" * 60)
        print("  !! WARNING — MISSING FILES DETECTED !!")
        print("=" * 60)
        if missing_actives:
            print(f"  [ACTIVE] {len(missing_actives)} missing — indices: "
                  f"{', '.join(f'{i:02d}' for i in missing_actives)}")
        if missing_decoys:
            print(f"  [DECOY]  {len(missing_decoys)} missing — indices: "
                  f"{', '.join(f'{i:02d}' for i in missing_decoys)}")
        print("  -> Re-run LigandMPNN on these structures to recover them.")
        print("=" * 60 + "\n")
    else:
        print("  [OK] No missing indices — all files present.")

    # ── 3. Compute perturbation scores ───────────────────────────────────────
    print("\n[3/4] Computing perturbation scores...")
    rows: list[dict] = []
    skipped: list[str] = []

    for fp in pt_files:
        fname = pathlib.Path(fp).name
        label = assign_label(fname)

        try:
            Q = load_probs(fp)
        except Exception as exc:
            print(f"  SKIP {fname}: {exc}")
            skipped.append(fname)
            continue

        if Q.shape != P_ref.shape:
            print(f"  SKIP {fname}: shape {Q.shape} != ref {P_ref.shape}")
            skipped.append(fname)
            continue

        jsd    = jsd_per_residue(P_ref, Q)
        score  = perturbation_score(jsd)
        label_str = "active" if label == 1 else "decoy" if label == 0 else "unknown"

        print(f"  {'✓':2s} {fname:<55s}  score={score:8.5f}  label={label_str}")
        rows.append({
            "file":        fname,
            "score":       score,
            "label":       label,
            "label_str":   label_str,
            "mean_jsd":    float(jsd.mean()),
            "sum_jsd":     float(jsd.sum()),
            "n_residues":  len(jsd),
        })

    if not rows:
        sys.exit("No files processed successfully.")

    df = pd.DataFrame(rows)

    # ── Duplicate removal ────────────────────────────────────────────────────
    df["compound_id"] = df["file"].apply(extract_compound_id)
    before = len(df)
    df = df.sort_values("score").drop_duplicates(subset="compound_id", keep="first")
    df = df.sort_values("file").reset_index(drop=True)
    removed = before - len(df)

    if removed > 0:
        print(f"\n  [DEDUP] {removed} duplicate compound(s) removed (kept lowest score per ID).")
        all_df = pd.DataFrame(rows)
        all_df["compound_id"] = all_df["file"].apply(extract_compound_id)
        dup_df = all_df[all_df.duplicated("compound_id", keep=False)]
        kept   = set(df["file"])
        for _, row in dup_df[~dup_df["file"].isin(kept)].iterrows():
            print(f"    ✗ REMOVED  {row['file']:<55s}  "
                  f"(compound: {row['compound_id']}, score={row['score']:.5f})")
        for _, row in dup_df[dup_df["file"].isin(kept)].iterrows():
            print(f"    ✓ KEPT     {row['file']:<55s}  "
                  f"(compound: {row['compound_id']}, score={row['score']:.5f})")
    else:
        print("\n  [DEDUP] No duplicates found.")

    # Save CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  → Scores saved to: {OUT_CSV}")

    # Summary table
    print("\n" + "─" * 60)
    print(df[["file", "label_str", "score", "mean_jsd"]].to_string(index=False))
    print("─" * 60)

    # ── 4. AUROC + Classification + EF + Bootstrap CI ───────────────────────
    print("\n[4/4] Computing AUROC, EF, bootstrap CI, and classification metrics...")

    df_labeled = df[df["label"].notna()].copy()
    if len(df_labeled) == 0:
        print("  No labelled files (active/decoy). AUROC cannot be computed.")
        return
    if df_labeled["label"].nunique() < 2:
        print("  Need at least one sample per class. AUROC cannot be computed.")
        return

    labels     = df_labeled["label"].astype(int).tolist()
    scores_raw = df_labeled["score"].tolist()

    # Actives have LOWER score → negate so that higher = more active for ROC
    scores_neg = [-s for s in scores_raw]

    auroc            = roc_auc_score(labels, scores_neg)
    fpr, tpr, thresh = roc_curve(labels, scores_neg)

    # ── Bootstrap CI on AUROC ────────────────────────────────────────────────
    print(f"  Computing bootstrap CI  "
          f"({BOOTSTRAP_N_RESAMPLES} resamples, {int(BOOTSTRAP_CI*100)}% CI)…")
    ci_lower, ci_upper = bootstrap_auroc_ci(
        labels, scores_neg,
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        ci=BOOTSTRAP_CI,
        seed=BOOTSTRAP_SEED,
    )

    # ── Enrichment Factors ───────────────────────────────────────────────────
    ef_results: dict[float, float] = {}
    for frac in EF_FRACTIONS:
        ef_results[frac] = enrichment_factor(labels, scores_neg, frac)

    # Max theoretical EF (limited by library composition and fraction size)
    n_total  = len(labels)
    n_act    = sum(labels)
    base_rate = n_act / n_total

    # ── Optimal threshold via Youden index (maximises TPR - FPR) ────────────
    youden_idx   = int(np.argmax(tpr - fpr))
    best_thresh  = thresh[youden_idx]          # threshold on scores_neg
    preds        = [1 if s >= best_thresh else 0 for s in scores_neg]

    tn, fp_val, fn, tp = confusion_matrix(labels, preds).ravel()

    acc  = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    mcc  = matthews_corrcoef(labels, preds)
    spec = tn / (tn + fp_val) if (tn + fp_val) > 0 else 0.0   # specificity

    sep = "─" * 52
    ci_pct = int(BOOTSTRAP_CI * 100)
    print(f"\n  {sep}")
    print(f"  {'AUROC':<34s}: {auroc:.4f}")
    print(f"  {'AUROC bootstrap CI':<34s}: [{ci_lower:.4f}, {ci_upper:.4f}]  ({ci_pct}%)")
    print(f"  {sep}")

    # Enrichment Factor block
    for frac, ef_val in ef_results.items():
        pct_label = f"EF{int(frac*100)}%"
        # Theoretical maximum EF
        n_top_k   = max(1, int(np.floor(frac * n_total)))
        ef_max    = min(1.0, n_act / n_top_k) / base_rate if base_rate > 0 else float("nan")
        ef_str    = f"{ef_val:.3f}" if not np.isnan(ef_val) else "n/a"
        max_str   = f"{ef_max:.3f}" if not np.isnan(ef_max) else "n/a"
        print(f"  {pct_label:<34s}: {ef_str:<10s}  (max achievable: {max_str})")
    print(f"  {sep}")

    print(f"  {'Optimal threshold (Youden)':<34s}: {-best_thresh:.5f}")
    print(f"  {sep}")
    print(f"  {'Accuracy':<34s}: {acc:.4f}")
    print(f"  {'Balanced Accuracy':<34s}: {bacc:.4f}")
    print(f"  {'Precision (PPV)':<34s}: {prec:.4f}")
    print(f"  {'Recall (Sensitivity)':<34s}: {rec:.4f}")
    print(f"  {'Specificity':<34s}: {spec:.4f}")
    print(f"  {'F1-score':<34s}: {f1:.4f}")
    print(f"  {'MCC':<34s}: {mcc:.4f}")
    print(f"  {sep}")
    print(f"  Confusion matrix  TP={tp}  FP={fp_val}  TN={tn}  FN={fn}")
    print(f"  Total samples: {len(df_labeled)}  "
          f"(active={labels.count(1)}, decoy={labels.count(0)})")
    print(f"  {sep}")

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1 – ROC curve with bootstrap CI band
    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"AUROC = {auroc:.3f}  [{ci_lower:.3f}, {ci_upper:.3f}] {ci_pct}% CI")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random")
    ax.scatter(fpr[youden_idx], tpr[youden_idx],
               color="darkorange", zorder=5, s=100,
               label=f"Youden threshold = {-best_thresh:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color="steelblue")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — JSD Perturbation Score", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.3)

    # Panel 2 – Score distribution scatter
    ax2 = axes[1]
    actives_scores = df_labeled[df_labeled["label"] == 1]["score"].values
    decoys_scores  = df_labeled[df_labeled["label"] == 0]["score"].values

    ax2.scatter(range(len(actives_scores)), actives_scores,
                color="steelblue", s=80, zorder=3,
                label=f"Active (n={len(actives_scores)})")
    ax2.scatter(range(len(decoys_scores)), decoys_scores,
                color="crimson", s=80, zorder=3, marker="^",
                label=f"Decoy (n={len(decoys_scores)})")
    ax2.axhline(np.mean(actives_scores), color="steelblue", linestyle="--",
                linewidth=1, alpha=0.7,
                label=f"Active mean = {np.mean(actives_scores):.3f}")
    ax2.axhline(np.mean(decoys_scores), color="crimson", linestyle="--",
                linewidth=1, alpha=0.7,
                label=f"Decoy mean  = {np.mean(decoys_scores):.3f}")
    youden_score = -best_thresh
    ax2.axhline(youden_score, color="darkorange", linestyle=":",
                linewidth=1.5, alpha=0.9,
                label=f"Youden threshold = {youden_score:.3f}")
    ax2.set_xlabel("Sample index", fontsize=12)
    ax2.set_ylabel("Perturbation Score", fontsize=12)
    ax2.set_title("JSD Score Distribution — Active vs Decoy", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3 – Enrichment Factor bar chart
    ax3 = axes[2]
    ef_labels = [f"EF{int(f*100)}%" for f in EF_FRACTIONS]
    ef_vals   = [ef_results[f] for f in EF_FRACTIONS]
    ef_maxes  = []
    for frac in EF_FRACTIONS:
        n_top_k = max(1, int(np.floor(frac * n_total)))
        ef_max  = min(1.0, n_act / n_top_k) / base_rate if base_rate > 0 else float("nan")
        ef_maxes.append(ef_max)

    x      = np.arange(len(EF_FRACTIONS))
    width  = 0.35
    bars1  = ax3.bar(x - width / 2, ef_vals, width, label="Observed EF",
                     color="steelblue", alpha=0.85, zorder=3)
    bars2  = ax3.bar(x + width / 2, ef_maxes, width, label="Max EF",
                     color="lightsteelblue", alpha=0.7, zorder=3,
                     edgecolor="steelblue", linewidth=0.8, linestyle="--")
    ax3.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Random (EF = 1)")

    for bar, val in zip(bars1, ef_vals):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10, color="steelblue")
    for bar, val in zip(bars2, ef_maxes):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10, color="grey")

    ax3.set_xticks(x)
    ax3.set_xticklabels(ef_labels, fontsize=11)
    ax3.set_ylabel("Enrichment Factor", fontsize=12)
    ax3.set_title("Enrichment Factors (1%, 5%, 10%)", fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(bottom=0)

    plt.suptitle(
        f"Protein Perturbation Analysis  —  "
        f"AUROC={auroc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]  |  "
        f"F1={f1:.3f}  |  MCC={mcc:.3f}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=150, bbox_inches="tight")
    print(f"\n  → ROC curve saved to: {OUT_ROC}")
    plt.show()

    if skipped:
        print(f"\n  Skipped files ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()