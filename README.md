# protein-perturbation

A single-shot, MD-free virtual-screening method that asks
**"how much does a docked ligand perturb the LigandMPNN per-residue
amino-acid probability distribution of the binding pocket, compared to the
reference complex?"** and uses that perturbation as a discrimination signal
between actives and decoys.

> **Hypothesis** — a binder that fits the pocket "as the protein expects"
> leaves the inverse-folding distribution close to the reference; a poor
> binder (decoy) drives the distribution further away. The Jensen–Shannon
> divergence (JSD) of the per-residue distributions, averaged across the
> pocket, is the per-pose **perturbation score**. Lower score → more
> active-like.

Full methodological detail and equations are in [`METHODS.tex`](METHODS.tex).

---

## 1. Pipeline at a glance

```
                     ┌────────────────────────────────────────────────┐
                     │              reference complex                 │
                     │  (e.g. crystal of target + native ligand)      │
                     └───────────────┬────────────────────────────────┘
                                     │
                ┌────────────────────┴──────────────────────┐
                │                                            │
                ▼                                            ▼
       (1) GNINA docking                            (R) Reference pass
       gnina_docking.py                             run_lmpnn.sh + run_score.sh
       — actives + decoys                           — reference .pt files
                │
                ▼
       (2) Merge poses + protein
       combine_sdf_protein.py
                │
                ▼
       (3) Pose QC          ──────────►   (4) Pocket residue selection
       pose_qc.py                          residues_selection.py
       (clash + RDKit geometry)            (atoms within autobox sphere)
                │
                ▼
       (5) LigandMPNN design + per-pose scoring
       run_pipeline.sh  →  run_score.sh
       (.pt: per-replica, per-residue, 21 AA probs)
                │
                ▼
       (6) JSD perturbation score + AUROC
       aucroc.py
       (JSD per residue → mean → AUROC, EF, bootstrap CI, Youden)
```

The reference pass (R) and the screening pass (1–5) produce **two sets of
`.pt` files with identical shape**. `aucroc.py` computes the JSD between each
screening file and the (averaged) reference, and reports the AUROC on the
`active`/`decoy` labels parsed from filenames.

---

## 2. Repository layout

```
protein-perturbation/
├── README.md                        ← this file
├── METHODS.tex                      ← full LaTeX methodology
├── environment-analysis.yml         ← conda env for QC, docking, analysis
├── environment-ligandmpnn.yml       ← conda env for LigandMPNN itself
├── analysis.ipynb                   ← exploratory notebook (single intact/decoy pair)
├── data/                            ← input PDBs, ligand libraries  (git-ignored)
├── experiments/                     ← per-target run outputs        (git-ignored)
├── output_*/                        ← scratch outputs               (git-ignored)
└── scripts/
    ├── config.yaml                  ← single source of truth for the bash pipeline
    ├── parse_config.py              ← YAML reader used by run_pipeline.sh
    ├── gnina_docking.py             ← GNINA docking with autobox from reference
    ├── combine_sdf_protein.py       ← merge GNINA SDF poses with protein PDB
    ├── pose_qc.py                   ← clash + RDKit geometry QC
    ├── residues_selection.py        ← pocket residue selection (autobox sphere)
    ├── run_pipeline.sh              ← QC → selection → LigandMPNN design (multi-PDB)
    ├── run_score.sh                 ← LigandMPNN per-pose scoring → .pt files
    ├── run_lmpnn.sh                 ← legacy one-shot design (kept for reference)
    └── aucroc.py                    ← JSD score + AUROC + EF + bootstrap CI
```

---

## 3. Installation

The pipeline uses **two conda environments** to avoid library conflicts
between RDKit/BioPython (analysis side) and the pinned PyTorch / numpy stack
LigandMPNN ships with.

```bash
# Analysis / QC / docking
conda env create -f environment-analysis.yml
conda activate protein-perturbation-analysis

# LigandMPNN
conda env create -f environment-ligandmpnn.yml
conda activate ligandmpnn_env
```

External binaries you also need:

| Tool       | How to install                                                       |
|------------|----------------------------------------------------------------------|
| GNINA      | `conda install -c conda-forge gnina`  or use the released binaries  |
| LigandMPNN | clone https://github.com/dauparas/LigandMPNN into `$LMPNN_PATH`     |
|            | download the checkpoint into `model_params/`                        |

Edit `scripts/config.yaml` so `lmpnn_path` points to the cloned LigandMPNN
directory.

---

## 4. Configuration — `scripts/config.yaml`

A single YAML file controls the whole bash pipeline. It is parsed via
`scripts/parse_config.py` (PyYAML), so quoting, comments and nesting all
work as expected.

```yaml
# Environment
conda_profile: "/home/tedeschg/miniforge3/etc/profile.d/conda.sh"
project_root:  "/home/tedeschg/prj/protein-perturbation"
lmpnn_path:    "/software/lmpnn/LigandMPNN"
env_dockndesign: "base"
env_ligandmpnn:  "ligandmpnn_env"
output_dir: "output_REFERENCE_abl1_lmpnn_focused"

# Pocket residue selection (docking-box based)
residues_selection:
  reference_complex: ".../2hzi_clean.pdb"   # crystal with native ligand
  autobox_add: 8.0                          # Å, sphere radius from ligand centroid
  output_file: "selected_residues.txt"
  output_format: "space_separated"          # list | space_separated | json
  include_hetatm_residues: false

# Pose QC
pose_qc:
  enabled: true
  clash_dist: 1.5          # Å (element-aware: max(clash_dist, 0.75*Σr_cov))
  bond_length_tol: 0.0     # extra tolerance (Å) for diffusion outputs
  strict: false            # if true, geometry warnings → failure
  out_json: true
  on_clash: "warn"         # warn | skip | fail

# LigandMPNN
ligandmpnn:
  model_type: "ligand_mpnn"
  checkpoint: "model_params/ligandmpnn_v_32_010_25.pt"
  temperature: 0.1
  seed: 42
  number_of_batches: 1
```

To switch target / reference, change only `output_dir`,
`residues_selection.reference_complex`, and the input PDB list passed to
`run_pipeline.sh`.

---

## 5. Running it end-to-end

### 5.1 One-shot reference pass

The reference complex (apo + native ligand) is run through LigandMPNN once
to produce a reference probability distribution `.pt` file:

```bash
bash scripts/run_lmpnn.sh                 # design pass (writes backbones)
bash scripts/run_score.sh                 # scoring pass (writes .pt files)
```

Both scripts have a couple of hard-coded paths near the top — edit them once
per target.

### 5.2 Library pass

For each docked pose:

```bash
# (1) Dock the whole library against the protein, using the crystal ligand
#     to define the box.
python scripts/gnina_docking.py \
  -p data/abl1/protein.pdb \
  -c data/abl1/crystal_structure/2hzi_clean.pdb \
  -L data/abl1/library_3d/ \
  -o experiments/output_abl1_docking

# (2) Pick best pose per ligand and write a protein+ligand complex PDB.
python scripts/combine_sdf_protein.py \
  -p data/abl1/protein.pdb \
  -s experiments/output_abl1_docking \
  -o experiments/output_abl1_complexes \
  --best_only --rank_by CNNaffinity

# (3+4+5) QC + pocket selection + LigandMPNN design, batched over all PDBs.
bash scripts/run_pipeline.sh \
  experiments/output_abl1_complexes/*.pdb

# (5b) Per-pose scoring (produces the .pt files used by aucroc.py).
bash scripts/run_score.sh
```

### 5.3 Score & evaluate

```bash
python scripts/aucroc.py \
  --scoring-dir experiments/output_abl1_lmpnn_focused/scoring \
  --reference   experiments/output_REFERENCE_abl1_lmpnn_focused/scoring/2hzi_clean_1.pt \
  --out-csv     experiments/abl1_scores.csv \
  --out-roc     experiments/abl1_roc.png
```

Outputs:

* a CSV with one row per pose: `file, score, label, label_str, mean_jsd,
  sum_jsd, n_residues, compound_id`
* a 3-panel PNG: ROC, score distribution, enrichment factors
* a printed report with AUROC + bootstrap CI, EF@1/5/10%, Youden threshold,
  confusion matrix, accuracy, balanced accuracy, precision, recall,
  specificity, F1, MCC.

`aucroc.py --help` lists all flags. Defaults are the ABL1 setup, so
running it with no args reproduces the historical ABL1 result.

---

## 6. The hypothesis, in one paragraph

LigandMPNN is an **inverse-folding** model: given backbone + ligand, it
outputs a per-residue distribution over the 20 canonical amino acids.
When the ligand is the native one for which the binding site evolved, the
distribution is concentrated on the wild-type residue at most positions.
When the ligand is a stranger (a decoy, or a poorly-fitting active), the
network is less sure — entropy goes up at pocket residues, the
distribution drifts. The further it drifts from the reference distribution
**at the same set of pocket residues**, the worse the binder. We summarise
that drift as the mean Jensen–Shannon divergence across the pocket — the
"perturbation score" — and the AUROC against active/decoy labels measures
whether that single scalar separates the two groups.

This is in the same family of ideas as "the protein's own inverse-folding
distribution is a structure-aware affinity proxy", but the perturbation
framing has the practical benefit that it cancels everything that is
target-specific (sequence, fold, baseline confidence) by working in
*differences* with respect to the reference.

---

## 7. Why "small perturbation = active" (and not the other way around)

The temptation is to read the sign the other way: a *good* ligand should
"signal" more than a bad one. In this framework it's the opposite, and the
reason is structural:

* The reference complex is the **co-evolved equilibrium** — sidechains, the
  backbone, and the native ligand are mutually adapted.
* LigandMPNN was trained on the PDB, i.e. on co-evolved complexes, so its
  high-confidence output (low entropy, peaky distribution) reflects that
  equilibrium.
* Plugging a different ligand into the same pocket perturbs the
  equilibrium. If the new ligand is chemically "neighbour-like" (a real
  binder), the perturbation is small. If the new ligand is geometrically /
  electrostatically alien (a decoy), perturbation is large.

The sign convention is implemented as `scores_neg = [-s for s in scores]`
at `scripts/aucroc.py` before computing the ROC, so that "higher = more
active" downstream of the negation.

---

## 8. Reproducibility notes

* Random seeds: `42` for LigandMPNN design, `111` for LigandMPNN scoring,
  `42` for the bootstrap. Pinned in `config.yaml` and `aucroc.py`
  respectively.
* GNINA poses are sensitive to `--exhaustiveness`; the default is 8. Bump
  this if you see large run-to-run drift on the same ligand.
* LigandMPNN `temperature` is `0.1` by default. Higher temperatures broaden
  the per-residue distribution and inflate JSD across the board (i.e. the
  AUROC is preserved but absolute scores are not comparable across
  temperatures).
* The reference is **averaged across all replicas** in its `.pt` file
  before JSD is computed (`load_probs` in `aucroc.py`). To average over
  *multiple reference structures*, pass a folder as `--reference`.

---

## 9. Known limitations

* The autobox sphere may pick up residues that are spatially close but not
  functionally relevant to binding (e.g. a flexible loop pointing into the
  cleft). This adds noise but does not bias direction.
* The "missing indices" warning in `aucroc.py` assumes the integer suffix
  in filenames is consecutive (`active_00_…`, `active_01_…`, …). If your
  naming scheme deviates, the warning is misleading but harmless.
* `aucroc.py` deduplicates by compound id, keeping the **lowest-score**
  (most active-like) pose per compound. This is the right thing if you
  trust a single best pose per compound; it is wrong if you want
  pose-level statistics.

---

## 10. Quick troubleshooting

| Symptom                                          | Likely cause                                                                                  |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `aucroc.py` says "shape mismatch, skipped"       | Reference and scoring `.pt` are from different LigandMPNN runs with different pocket masks.   |
| AUROC is 0.5 ± noise                             | Reference and screening pass were not run with identical `temperature` and residue selection. |
| "All PDBs failed residue selection"              | `reference_complex` in `config.yaml` has no HETATM ligand (or the ligand chain is filtered).  |
| `pose_qc.py` flags every pose as FAIL            | `clash_dist` too generous; try `1.2`, or set `on_clash: warn` to inspect manually.            |
| `run_pipeline.sh` errors at the YAML read step   | PyYAML missing from the active env; `pip install pyyaml`.                                      |

---

## 11. Contact / authorship

Author: **Guglielmo Tedeschi**, tedeschg@vscht.cz. 

Work-in-progress
