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

Usage:
    python perturbation_auroc.py
"""

import sys
import glob
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, roc_curve
import torch

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit here
# ──────────────────────────────────────────────────────────────────────────────

SCORING_DIR    = "/home/tedeschg/prj/protein-perturbation/output_scoring/"

# REFERENCE_PATH can be:
#   - a single .pt file  → its replicas are averaged (mean over replica axis)
#   - a folder of .pt files → all files are loaded and their probs are averaged together
REFERENCE_PATH = "/home/tedeschg/prj/protein-perturbation/output_TEST/score_1/abl-imatinib_clean_1.pt"

# Keywords to assign labels from filenames
ACTIVE_KEYWORD = "active"   # label = 1
DECOY_KEYWORD  = "decoy"    # label = 0


OUT_CSV = "/home/tedeschg/prj/protein-perturbation/perturbation_scores_all.csv"
OUT_ROC = "/home/tedeschg/prj/protein-perturbation/roc_curve_all.png"

# ──────────────────────────────────────────────────────────────────────────────


def load_probs(path: str) -> np.ndarray:
    """Load a single .pt file and return the mean probs across all replicas.

    probs tensor shape: (n_replicas, n_residues, n_aa)
    Returns mean over replica axis -> shape (n_residues, n_aa).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    probs = data["probs"]
    if hasattr(probs, "numpy"):
        probs = probs.numpy()
    probs = probs.astype(float)   # (n_replicas, n_residues, n_aa)
    return probs.mean(axis=0)     # (n_residues, n_aa)


def load_reference(ref_path: str) -> np.ndarray:
    """Load the reference probs, supporting both a single .pt file and a folder.

    Single file  → average over its replicas.
    Folder       → load every .pt inside, average each over replicas,
                   then average across files.
    This gives the most stable consensus reference distribution.
    """
    p = pathlib.Path(ref_path)
    if p.is_file():
        probs = load_probs(str(p))
        n_files = 1
    elif p.is_dir():
        pt_files = sorted(p.glob("*.pt"))
        if not pt_files:
            sys.exit(f"ERROR: no .pt files found in reference folder: {ref_path}")
        all_probs = [load_probs(str(f)) for f in pt_files]
        probs = np.mean(all_probs, axis=0)
        n_files = len(pt_files)
    else:
        sys.exit(f"ERROR: REFERENCE_PATH does not exist: {ref_path}")
    return probs, n_files


def jsd_per_residue(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon Divergence per residue (row-wise).

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5*(P+Q)

    Properties vs plain KL divergence:
      - Symmetric: JSD(P,Q) == JSD(Q,P)
      - Bounded:   always in [0, 1] (base-2 log used by scipy)
      - No need for epsilon smoothing (M is never zero where P or Q > 0)

    Returns shape (n_residues,) with values in [0, 1].
    """
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    # scipy jensenshannon works on 1-D arrays → apply row-wise
    return np.array([jensenshannon(P[i], Q[i], base=2) for i in range(len(P))])


def perturbation_score(jsd_values: np.ndarray) -> float:
    """Overall perturbation score = mean JSD * 100.

    JSD is in [0,1] so the score is in [0, 100].
    """
    return float(jsd_values.mean() * 100)


def assign_label(filename: str) -> int | None:
    """Returns 1 (active), 0 (decoy), or None if the file cannot be classified."""
    name = filename.lower()
    if ACTIVE_KEYWORD in name:
        return 1
    if DECOY_KEYWORD in name:
        return 0
    return None


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Perturbation Score + AUROC Calculator")
    print("=" * 60)

    # 1. Load the reference (single file or folder → averaged)
    print(f"\n[1/4] Loading reference: {REFERENCE_PATH}")
    try:
        P_ref, n_ref_files = load_reference(REFERENCE_PATH)
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"ERROR loading reference: {e}")
    print(f"      Reference source files : {n_ref_files}")
    print(f"      Reference probs shape  : {P_ref.shape}  (replicas already averaged)")

    # 2. Find all .pt files in the scoring folder
    pt_files = sorted(glob.glob(str(pathlib.Path(SCORING_DIR) / "*.pt")))
    if not pt_files:
        sys.exit(f"No .pt files found in: {SCORING_DIR}")
    print(f"\n[2/4] Found {len(pt_files)} .pt files in {SCORING_DIR}")

    # Detect missing indices for actives and decoys
    # Expects filenames like: actives_02_CHEMBL..._1.pt  /  decoys_10_ZINC..._1.pt
    import re

    def find_missing_indices(files, keyword):
        """Return sorted list of integer indices absent in the file list."""
        group = [pathlib.Path(f).name for f in files if keyword in pathlib.Path(f).name.lower()]
        found = set()
        for name in group:
            m = re.search(rf"{keyword}s?_(\d+)_", name, re.IGNORECASE)
            if m:
                found.add(int(m.group(1)))
        if not found:
            return []
        return sorted(set(range(min(found), max(found) + 1)) - found)

    missing_actives = find_missing_indices(pt_files, ACTIVE_KEYWORD)
    missing_decoys  = find_missing_indices(pt_files, DECOY_KEYWORD)

    if missing_actives or missing_decoys:
        print("\n" + "=" * 60)
        print("  !! WARNING — MISSING FILES DETECTED !!")
        print("=" * 60)
        if missing_actives:
            idx_str = ", ".join(f"{i:02d}" for i in missing_actives)
            print(f"  [ACTIVE] {len(missing_actives)} missing — indices: {idx_str}")
        if missing_decoys:
            idx_str = ", ".join(f"{i:02d}" for i in missing_decoys)
            print(f"  [DECOY]  {len(missing_decoys)} missing — indices: {idx_str}")
        print("  -> Re-run LigandMPNN on these structures to recover them.")
        print("=" * 60 + "\n")
    else:
        print("  [OK] No missing indices — all files present.")

    # 3. Compute perturbation score for each file
    print("\n[3/4] Computing perturbation scores...")
    rows = []
    skipped = []
    for fp in pt_files:
        fname = pathlib.Path(fp).name
        label = assign_label(fname)

        try:
            Q = load_probs(fp)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            skipped.append(fname)
            continue

        # Skip if shapes don't match
        if Q.shape != P_ref.shape:
            print(f"  SKIP {fname}: shape {Q.shape} != ref {P_ref.shape}")
            skipped.append(fname)
            continue

        kl    = jsd_per_residue(P_ref, Q)
        score = perturbation_score(kl)

        label_str = ("active" if label == 1 else "decoy" if label == 0 else "unknown")
        print(f"  {'✓':2s} {fname:<55s}  score={score:8.5f}  label={label_str}")

        rows.append({
            "file":       fname,
            "score":      score,
            "label":      label,
            "label_str":  label_str,
            "mean_jsd":   float(kl.mean()),
            "sum_jsd":    float(kl.sum()),
            "n_residues": len(kl),
        })

    if not rows:
        sys.exit("No files processed successfully.")

    df = pd.DataFrame(rows)

    # ── Duplicate removal ────────────────────────────────────────────────────
    # Extract compound ID (e.g. CHEMBL40557 or ZINC39482920) from filename.
    # If the same ID appears more than once (different poses/conformers),
    # keep only the one with the LOWEST perturbation score (best pose).
    import re as _re
    def extract_compound_id(fname):
        m = _re.search(r"(CHEMBL\d+|ZINC\d+)", fname, _re.IGNORECASE)
        return m.group(1).upper() if m else fname

    df["compound_id"] = df["file"].apply(extract_compound_id)
    before = len(df)
    # Among duplicates keep the row with the lowest score (most similar to ref)
    df = df.sort_values("score").drop_duplicates(subset="compound_id", keep="first")
    df = df.sort_values("file").reset_index(drop=True)
    after = len(df)

    removed = before - after
    if removed > 0:
        print(f"\n  [DEDUP] {removed} duplicate compound(s) removed (kept lowest score per ID).")
        # Show which ones were removed
        all_ids   = pd.DataFrame(rows)
        all_ids["compound_id"] = all_ids["file"].apply(extract_compound_id)
        dup_ids   = all_ids[all_ids.duplicated("compound_id", keep=False)]
        kept_files = set(df["file"])
        dropped   = dup_ids[~dup_ids["file"].isin(kept_files)]
        for _, row in dropped.iterrows():
            print(f"    ✗ REMOVED  {row['file']:<55s}  (compound: {row['compound_id']}, score={row['score']:.5f})")
        kept_dups = dup_ids[dup_ids["file"].isin(kept_files)]
        for _, row in kept_dups.iterrows():
            print(f"    ✓ KEPT     {row['file']:<55s}  (compound: {row['compound_id']}, score={row['score']:.5f})")
    else:
        print("\n  [DEDUP] No duplicates found.")
    # ────────────────────────────────────────────────────────────────────────

    # Save CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  → Scores saved to: {OUT_CSV}")

    # Print summary table
    print("\n" + "─" * 60)
    print(df[["file", "label_str", "score", "mean_jsd"]].to_string(index=False))
    print("─" * 60)

    # 4. AUROC
    print("\n[4/4] Computing AUROC...")
    df_labeled = df[df["label"].notna()].copy()
    if len(df_labeled) == 0:
        print("  No files with a label (active/decoy). AUROC cannot be computed.")
        return
    if df_labeled["label"].nunique() < 2:
        print("  Need at least one sample per class. AUROC cannot be computed.")
        return

    labels     = df_labeled["label"].astype(int).tolist()
    scores     = df_labeled["score"].tolist()

    # Actives have LOWER score → negate for ROC
    scores_neg = [-s for s in scores]

    auroc = roc_auc_score(labels, scores_neg)
    fpr, tpr, thresholds = roc_curve(labels, scores_neg)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │   AUROC = {auroc:.4f}               │")
    print(f"  └─────────────────────────────────┘")
    print(f"  Total samples: {len(df_labeled)}  "
          f"(active={labels.count(1)}, decoy={labels.count(0)})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 – ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.12, color="steelblue")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — JSD Perturbation Score", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.3)

    # Panel 2 – Score distribution scatter
    # Each dot = one compound; actives (blue circles) vs decoys (red triangles).
    # A good model pushes actives DOWN and decoys UP on the Y axis.
    # The dashed lines show the group means for quick visual comparison.
    ax2 = axes[1]
    actives = df_labeled[df_labeled["label"] == 1]["score"]
    decoys  = df_labeled[df_labeled["label"] == 0]["score"]
    ax2.scatter(range(len(actives)), actives.values,
                color="steelblue", s=80, zorder=3, label=f"Active (n={len(actives)})")
    ax2.scatter(range(len(decoys)), decoys.values,
                color="crimson", s=80, zorder=3, label=f"Decoy (n={len(decoys)})", marker="^")
    ax2.axhline(np.mean(actives), color="steelblue", linestyle="--", linewidth=1, alpha=0.7,
                label=f"Active mean = {np.mean(actives):.3f}")
    ax2.axhline(np.mean(decoys),  color="crimson",   linestyle="--", linewidth=1, alpha=0.7,
                label=f"Decoy mean  = {np.mean(decoys):.3f}")
    ax2.set_xlabel("Sample index", fontsize=12)
    ax2.set_ylabel("Perturbation Score", fontsize=12)
    ax2.set_title("JSD Score Distribution — Active vs Decoy", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Protein Perturbation Analysis  —  AUROC={auroc:.3f}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=150, bbox_inches="tight")
    print(f"  → ROC curve saved to: {OUT_ROC}")
    plt.show()

    if skipped:
        print(f"\n  Skipped files ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()