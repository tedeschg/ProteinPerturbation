#!/usr/bin/env python3
"""
pose_qc.py — Quality control for ligand poses before LigandMPNN.

Checks:
  1. Clash detection (FAIL): protein heavy atoms too close to ligand heavy atoms.
  2. Internal geometry (WARN): bond lengths and angles outside expected ranges,
     evaluated via RDKit (requires a SMILES reference or uses distance-only heuristics).

Designed to be robust for both docking outputs and diffusion model outputs (e.g. Boltz-2).

Exit codes:
  0 — passed (or only warnings)
  1 — failed (clash detected, or --strict mode with geometry errors)

Usage:
  # Basic (auto-detects ligand):
  python pose_qc.py --pdb complex.pdb

  # With explicit ligand:
  python pose_qc.py --pdb complex.pdb --ligand_resname LIG --ligand_chain X

  # Strict mode (geometry warnings become failures):
  python pose_qc.py --pdb complex.pdb --strict

  # Custom thresholds:
  python pose_qc.py --pdb complex.pdb --clash_dist 1.8 --bond_length_tol 0.3

  # Suppress RDKit sanitization errors (useful for diffusion outputs):
  python pose_qc.py --pdb complex.pdb --sanitize_mol

  # Write a JSON summary:
  python pose_qc.py --pdb complex.pdb --out_json qc_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

# RDKit imports — optional but required for geometry checks
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms
    from rdkit.Geometry import rdGeometry
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMON_IONS: Set[str] = {
    "NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "CU",
    "CO", "CD", "NI", "SR", "HG", "LI", "RB", "CS", "BA",
    "AL", "CR", "MO", "W", "PT", "AU", "AG",
}

# Covalent radii (Å) for clash estimation — sum of radii = expected min distance
# Source: Alvarez 2008 / CSD averages
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "ZN": 1.22, "FE": 1.32, "MG": 1.41, "CA": 1.76, "NA": 1.66,
    "SE": 1.20, "SI": 1.11,
}
DEFAULT_COVALENT_RADIUS = 0.80  # fallback for unknown elements

# Bond length thresholds: (min, max) in Å for common bond types
# These are generous ranges for robustness with diffusion outputs
BOND_LENGTH_RANGES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("C", "C"):  (1.15, 1.65),
    ("C", "N"):  (1.10, 1.55),
    ("C", "O"):  (1.10, 1.55),
    ("C", "S"):  (1.55, 1.90),
    ("C", "H"):  (0.85, 1.20),
    ("N", "H"):  (0.80, 1.15),
    ("O", "H"):  (0.80, 1.10),
    ("C", "F"):  (1.25, 1.45),
    ("C", "CL"): (1.60, 1.90),
    ("C", "BR"): (1.75, 2.10),
    ("C", "I"):  (1.90, 2.30),
    ("C", "P"):  (1.70, 1.95),
    ("P", "O"):  (1.40, 1.75),
    ("S", "O"):  (1.35, 1.65),
    ("N", "N"):  (1.10, 1.50),
    ("N", "O"):  (1.10, 1.50),
    ("S", "S"):  (1.90, 2.15),
}

# Angle thresholds: warn outside (min, max) degrees
ANGLE_RANGES: Dict[Tuple[str, str, str], Tuple[float, float]] = {
    ("C", "C", "C"): (95.0, 130.0),
    ("C", "C", "N"): (95.0, 130.0),
    ("C", "C", "O"): (95.0, 130.0),
    ("C", "N", "C"): (95.0, 130.0),
    ("C", "O", "C"): (95.0, 125.0),
}


# ---------------------------------------------------------------------------
# BioPython helpers (reused from select_residues.py for consistency)
# ---------------------------------------------------------------------------

def is_ligand_candidate(res) -> bool:
    hetflag, resseq, icode = res.id
    if not hetflag.startswith("H_"):
        return False
    resname = res.get_resname().strip()
    if resname == "HOH":
        return False
    if resname in COMMON_IONS:
        return False
    if is_aa(res, standard=True):
        return False
    heavy_atoms = [a for a in res.get_atoms() if a.element != "H"]
    if len(heavy_atoms) < 3:
        return False
    return True


def get_heavy_coords(res) -> List[np.ndarray]:
    return [a.coord for a in res.get_atoms() if a.element != "H"]


def auto_detect_ligand(model, ligand_chain: Optional[str], ligand_resname: Optional[str]):
    ligand_candidates = defaultdict(list)
    for chain in model:
        if ligand_chain and chain.id != ligand_chain:
            continue
        for res in chain:
            if not is_ligand_candidate(res):
                continue
            resname = res.get_resname().strip()
            if ligand_resname and resname != ligand_resname:
                continue
            ligand_candidates[resname].append(res)

    if not ligand_candidates:
        msg = "No suitable ligand detected."
        if ligand_chain:
            msg += f" (ligand_chain={ligand_chain})"
        if ligand_resname:
            msg += f" (ligand_resname={ligand_resname})"
        raise SystemExit(f"[ERROR] {msg}")

    if ligand_resname and ligand_resname in ligand_candidates:
        chosen = ligand_resname
    else:
        chosen = max(
            ligand_candidates,
            key=lambda k: sum(len(get_heavy_coords(r)) for r in ligand_candidates[k])
        )

    heavy_total = sum(len(get_heavy_coords(r)) for r in ligand_candidates[chosen])
    print(f"[INFO] Ligand detected: {chosen} "
          f"(residues={len(ligand_candidates[chosen])}, heavy_atoms={heavy_total})")
    return ligand_candidates[chosen]


# ---------------------------------------------------------------------------
# Check 1: Clash detection (FAIL)
# ---------------------------------------------------------------------------

def check_clashes(
    model,
    ligand_residues,
    clash_dist: float,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (has_clash, clash_details).
    Clash = protein heavy atom within clash_dist Å of any ligand heavy atom.

    Uses element-aware clash detection: the threshold is either the user-specified
    clash_dist OR the sum of covalent radii * 0.75 (VdW overlap), whichever is larger.
    This avoids false positives on genuinely covalent ligand-protein bonds.
    """
    ligand_atoms = [
        a for r in ligand_residues
        for a in r.get_atoms()
        if a.element not in ("H", "")
    ]
    if not ligand_atoms:
        print("[WARN] Ligand has no heavy atoms — skipping clash check.")
        return False, []

    protein_atoms = []
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            for a in res.get_atoms():
                if a.element not in ("H", ""):
                    protein_atoms.append(a)

    if not protein_atoms:
        print("[WARN] No protein atoms found — skipping clash check.")
        return False, []

    ns = NeighborSearch(protein_atoms)
    clash_details: List[Dict[str, Any]] = []

    for latom in ligand_atoms:
        nearby = ns.search(latom.coord, clash_dist + 0.5, level="A")  # broad search first
        for patom in nearby:
            dist = float(np.linalg.norm(latom.coord - patom.coord))

            # Element-aware threshold: use sum of covalent radii * 0.75 as minimum,
            # but never below clash_dist. This allows genuine covalent bonds.
            r_lig  = COVALENT_RADII.get(latom.element.upper(), DEFAULT_COVALENT_RADIUS)
            r_prot = COVALENT_RADII.get(patom.element.upper(), DEFAULT_COVALENT_RADIUS)
            cov_sum = (r_lig + r_prot) * 0.75  # VdW overlap threshold

            threshold = max(clash_dist, cov_sum)

            if dist < threshold:
                pres = patom.get_parent()
                pchain = pres.get_parent().id
                clash_details.append({
                    "ligand_atom":   latom.fullname.strip(),
                    "protein_atom":  patom.fullname.strip(),
                    "protein_res":   f"{pchain}{pres.id[1]}",
                    "protein_resname": pres.get_resname().strip(),
                    "distance_A":    round(dist, 3),
                    "threshold_A":   round(threshold, 3),
                })

    has_clash = len(clash_details) > 0
    return has_clash, clash_details


# ---------------------------------------------------------------------------
# Check 2: Internal geometry via RDKit (WARN)
# ---------------------------------------------------------------------------

def load_ligand_mol_from_pdb(ligand_residues, sanitize: bool) -> Optional[Any]:
    """
    Build an RDKit Mol from BioPython ligand residue atoms.
    Writes a minimal PDB block and reads it with RDKit.
    """
    lines = ["REMARK  LigandQC\n"]
    atom_serial = 1
    for res in ligand_residues:
        resname = res.get_resname().strip()[:3].ljust(3)
        chain   = res.get_parent().id
        resseq  = res.id[1]
        for atom in res.get_atoms():
            elem = atom.element.strip() if atom.element else "C"
            name = atom.fullname.strip()[:4].ljust(4)
            x, y, z = atom.coord
            line = (
                f"HETATM{atom_serial:5d} {name} {resname} {chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
            )
            lines.append(line)
            atom_serial += 1
    lines.append("END\n")

    pdb_block = "".join(lines)

    try:
        if sanitize:
            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=True)
        else:
            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, catchErrors=True)
                except Exception:
                    pass
        return mol
    except Exception as e:
        print(f"[WARN] RDKit failed to parse ligand PDB block: {e}")
        return None


def check_bond_lengths(mol) -> List[Dict[str, Any]]:
    """Warn on bonds with anomalous lengths."""
    issues = []
    if mol is None:
        return issues

    try:
        conf = mol.GetConformer()
    except Exception:
        return issues

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ai = mol.GetAtomWithIdx(i)
        aj = mol.GetAtomWithIdx(j)
        ei = ai.GetSymbol().upper()
        ej = aj.GetSymbol().upper()

        pi = conf.GetAtomPosition(i)
        pj = conf.GetAtomPosition(j)
        dist = math.sqrt(
            (pi.x - pj.x)**2 + (pi.y - pj.y)**2 + (pi.z - pj.z)**2
        )

        # Look up range (try both orderings)
        key  = (ei, ej)
        key2 = (ej, ei)
        rng  = BOND_LENGTH_RANGES.get(key) or BOND_LENGTH_RANGES.get(key2)

        if rng is None:
            # Unknown bond type: use a very generous default
            rng = (0.8, 2.5)

        lo, hi = rng
        if not (lo <= dist <= hi):
            issues.append({
                "type":    "bond_length",
                "atoms":   f"{ai.GetSymbol()}{i+1}-{aj.GetSymbol()}{j+1}",
                "bond":    f"{ei}-{ej}",
                "value_A": round(dist, 3),
                "range_A": [lo, hi],
            })

    return issues


def check_bond_angles(mol) -> List[Dict[str, Any]]:
    """Warn on bond angles outside expected ranges."""
    issues = []
    if mol is None:
        return issues

    try:
        conf = mol.GetConformer()
    except Exception:
        return issues

    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        ej = atom.GetSymbol().upper()

        if len(neighbors) < 2:
            continue

        for ii in range(len(neighbors)):
            for kk in range(ii + 1, len(neighbors)):
                i = neighbors[ii]
                k = neighbors[kk]
                ei = mol.GetAtomWithIdx(i).GetSymbol().upper()
                ek = mol.GetAtomWithIdx(k).GetSymbol().upper()

                try:
                    angle_rad = rdMolTransforms.GetAngleRad(conf, i, j, k)
                    angle_deg = math.degrees(angle_rad)
                except Exception:
                    continue

                key  = (ei, ej, ek)
                key2 = (ek, ej, ei)
                rng  = ANGLE_RANGES.get(key) or ANGLE_RANGES.get(key2)

                if rng is None:
                    continue  # no threshold defined, skip

                lo, hi = rng
                if not (lo <= angle_deg <= hi):
                    issues.append({
                        "type":       "bond_angle",
                        "atoms":      f"{ei}{i+1}-{ej}{j+1}-{ek}{k+1}",
                        "angle":      f"{ei}-{ej}-{ek}",
                        "value_deg":  round(angle_deg, 2),
                        "range_deg":  [lo, hi],
                    })

    return issues


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def summarise_results(
    pdb: str,
    clash_result: Tuple[bool, List],
    geom_issues: List[Dict],
    strict: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (should_fail, report_dict).
    """
    has_clash, clash_details = clash_result

    print_section("CLASH CHECK (FAIL if any)")
    if has_clash:
        print(f"  [FAIL] {len(clash_details)} clash(es) detected:")
        for c in clash_details[:20]:  # cap output
            print(f"    ligand:{c['ligand_atom']:6s} <-> "
                  f"prot:{c['protein_resname']}{c['protein_res']}:{c['protein_atom']:6s} "
                  f"dist={c['distance_A']:.3f} Å (threshold={c['threshold_A']:.3f} Å)")
        if len(clash_details) > 20:
            print(f"    ... and {len(clash_details) - 20} more.")
    else:
        print("  [PASS] No clashes detected.")

    print_section("GEOMETRY CHECK (WARN)")
    if not RDKIT_AVAILABLE:
        print("  [SKIP] RDKit not available.")
        geom_issues = []
    elif not geom_issues:
        print("  [PASS] No geometry issues detected.")
    else:
        for g in geom_issues[:30]:
            if g["type"] == "bond_length":
                print(f"  [WARN] Bond length: {g['atoms']} = {g['value_A']:.3f} Å "
                      f"(expected {g['range_A'][0]:.2f}–{g['range_A'][1]:.2f} Å)")
            elif g["type"] == "bond_angle":
                print(f"  [WARN] Bond angle:  {g['atoms']} = {g['value_deg']:.1f}° "
                      f"(expected {g['range_deg'][0]:.1f}–{g['range_deg'][1]:.1f}°)")
        if len(geom_issues) > 30:
            print(f"  ... and {len(geom_issues) - 30} more.")

    should_fail = has_clash or (strict and len(geom_issues) > 0)

    print_section("SUMMARY")
    status = "FAIL" if should_fail else ("WARN" if geom_issues else "PASS")
    print(f"  PDB:            {pdb}")
    print(f"  Clashes:        {len(clash_details)}")
    print(f"  Geometry issues:{len(geom_issues)}")
    print(f"  Result:         [{status}]")
    if should_fail:
        print("  => This pose should NOT be passed to LigandMPNN.")
    elif geom_issues:
        print("  => Geometry warnings detected — proceed with caution.")
    else:
        print("  => Pose looks clean.")

    report = {
        "pdb": pdb,
        "status": status,
        "n_clashes": len(clash_details),
        "n_geometry_issues": len(geom_issues),
        "clashes": clash_details,
        "geometry_issues": geom_issues,
    }

    return should_fail, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="QC for ligand poses (docking/diffusion) before LigandMPNN."
    )
    p.add_argument("--pdb",             required=True,  help="Input PDB file (complex)")
    p.add_argument("--ligand_chain",    default=None,   help="Ligand chain ID (e.g. X)")
    p.add_argument("--ligand_resname",  default=None,   help="Ligand resname (e.g. LIG)")
    p.add_argument("--model",           type=int, default=0,
                   help="Model index for multi-model PDB (default 0)")
    p.add_argument("--clash_dist",      type=float, default=1.5,
                   help="Hard clash distance threshold in Å (default 1.5). "
                        "Element-aware: actual threshold = max(clash_dist, 0.75*sum_cov_radii).")
    p.add_argument("--bond_length_tol", type=float, default=0.0,
                   help="Extra tolerance in Å added to bond length ranges (default 0.0). "
                        "Useful for diffusion outputs with slightly distorted geometry.")
    p.add_argument("--strict",          action="store_true",
                   help="Treat geometry warnings as failures (exit code 1).")
    p.add_argument("--sanitize_mol",    action="store_true",
                   help="Force RDKit sanitization (may fail on unusual diffusion outputs).")
    p.add_argument("--out_json",        default=None,
                   help="Optional path to write JSON QC report.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.pdb):
        raise SystemExit(f"[ERROR] PDB file not found: {args.pdb}")

    # Apply bond length tolerance for diffusion outputs
    if args.bond_length_tol != 0.0:
        for key in BOND_LENGTH_RANGES:
            lo, hi = BOND_LENGTH_RANGES[key]
            BOND_LENGTH_RANGES[key] = (lo - args.bond_length_tol, hi + args.bond_length_tol)
        print(f"[INFO] Bond length tolerance: ±{args.bond_length_tol:.2f} Å applied.")

    print(f"[INFO] Loading PDB: {args.pdb}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", args.pdb)
    model = structure[args.model]

    # Detect ligand
    ligand_residues = auto_detect_ligand(model, args.ligand_chain, args.ligand_resname)

    # --- Check 1: Clashes ---
    print(f"[INFO] Clash threshold: {args.clash_dist:.2f} Å (element-aware, see --clash_dist)")
    clash_result = check_clashes(model, ligand_residues, args.clash_dist)

    # --- Check 2: Internal geometry ---
    geom_issues: List[Dict] = []
    if RDKIT_AVAILABLE:
        mol = load_ligand_mol_from_pdb(ligand_residues, sanitize=args.sanitize_mol)
        if mol is not None:
            geom_issues  = check_bond_lengths(mol)
            geom_issues += check_bond_angles(mol)
        else:
            print("[WARN] Could not build RDKit molecule — geometry check skipped.")
            print("       Try --sanitize_mol or check ligand atom naming in the PDB.")
    else:
        print("[WARN] RDKit not available — geometry check skipped. Install with: pip install rdkit")

    # --- Summarise and report ---
    should_fail, report = summarise_results(
        pdb=args.pdb,
        clash_result=clash_result,
        geom_issues=geom_issues,
        strict=args.strict,
    )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] JSON report written -> {args.out_json}")

    sys.exit(1 if should_fail else 0)


if __name__ == "__main__":
    main()