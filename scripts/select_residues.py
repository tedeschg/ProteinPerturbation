#!/usr/bin/env python3
"""
select_residues.py — Ligand-centric residue selection for LigandMPNN.
 
Select protein residues within a cutoff distance from the ligand atoms.
 
Ligand detection:
  - HETATM residues
  - Not water (HOH / WAT)
  - Not standard amino acids
  - Not common mono/polyatomic ions (extended list, aligned with pose_qc.py)
  - Optionally restrict by --ligand_resname and/or --ligand_chain
  - If not specified, auto-detects the ligand as the non-AA, non-ion HETATM group
    with the largest heavy-atom count
 
Residue selection:
  - All protein residues with at least one heavy atom within --dist Å
    of any ligand heavy atom
  - Optionally includes HETATM residues (cofactors, modified AA) via
    --include_hetatm_residues
 
Consistency checks:
  - Prints protein checksum (n_residues, first/last) so you can verify
    all PDBs in a screening campaign share the same scaffold
  - Warns if n_selected_residues is unusually low (< --min_expected) or
    high (> --max_expected)
  - Warns if any selected residue is suspiciously close to the ligand
    (possible clash — run pose_qc.py for a full clash report)
 
Output:
  - Writes residues as tokens "A10 A11 A12 ..." (1 line, space-separated)
  - Optionally writes a JSON summary (--out_json)
 
Usage:
  python select_residues.py --pdb complex.pdb --out selected_residues.txt
  python select_residues.py --pdb complex.pdb --out sel.txt --dist 8.0
  python select_residues.py --pdb complex.pdb --out sel.txt \\
      --ligand_resname LIG --ligand_chain X
  python select_residues.py --pdb complex.pdb --out sel.txt \\
      --include_hetatm_residues --out_json summary.json
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
 
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
 
 
# ---------------------------------------------------------------------------
# Constants (keep in sync with pose_qc.py)
# ---------------------------------------------------------------------------
 
COMMON_IONS: Set[str] = {
    "NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "CU",
    "CO", "CD", "NI", "SR", "HG", "LI", "RB", "CS", "BA",
    "AL", "CR", "MO", "W",  "PT", "AU", "AG",
    # polyatomic ions / solvent additives
    "SO4", "PO4", "NO3", "ACT", "EDO", "GOL", "PEG", "DMS",
    "MSE",  # selenomethionine — sometimes HETATM but is an AA analogue
}
 
WATER_NAMES: Set[str] = {"HOH", "WAT", "H2O", "DOD"}
 
# Covalent radii for clash heuristic (Å) — same source as pose_qc.py
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "ZN": 1.22, "FE": 1.32, "MG": 1.41, "CA": 1.76, "NA": 1.66,
    "SE": 1.20, "SI": 1.11,
}
DEFAULT_COVALENT_RADIUS = 0.80
 
 
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ligand-centric residue selection for LigandMPNN."
    )
    p.add_argument("--pdb",             required=True,
                   help="Input PDB file (protein-ligand complex)")
    p.add_argument("--out",             required=True,
                   help="Output TXT path for selected residue tokens")
    p.add_argument("--dist",            type=float, default=6.0,
                   help="Cutoff distance in Å (default 6.0)")
    p.add_argument("--ligand_chain",    default=None,
                   help="Restrict ligand search to this chain ID (e.g. X)")
    p.add_argument("--ligand_resname",  default=None,
                   help="Restrict ligand search to this resname (e.g. LIG)")
    p.add_argument("--model",           type=int, default=0,
                   help="Model index for multi-model PDB (default 0)")
    p.add_argument("--min_heavy_atoms", type=int, default=3,
                   help="Minimum heavy atoms for ligand candidate (default 3). "
                        "Increase to exclude small cofactors.")
    p.add_argument("--include_hetatm_residues", action="store_true",
                   help="Also include non-AA residues in the selection output "
                        "(e.g. cofactors, modified residues)")
    p.add_argument("--clash_warn_dist", type=float, default=1.5,
                   help="Distance Å below which a selected residue atom is "
                        "flagged as a possible clash (default 1.5). "
                        "For a full clash report use pose_qc.py.")
    p.add_argument("--min_expected",    type=int, default=3,
                   help="Warn if fewer than this many residues are selected (default 3)")
    p.add_argument("--max_expected",    type=int, default=60,
                   help="Warn if more than this many residues are selected (default 60)")
    p.add_argument("--out_json",        default=None,
                   help="Optional path to write a JSON summary of the selection")
    p.add_argument("--quiet",           action="store_true",
                   help="Suppress non-essential output")
    return p.parse_args()
 
 
# ---------------------------------------------------------------------------
# Ligand detection
# ---------------------------------------------------------------------------
 
def is_ligand_candidate(res, min_heavy_atoms: int) -> bool:
    """Return True if this residue could be a small-molecule ligand."""
    hetflag, resseq, icode = res.id
    if not hetflag.startswith("H_"):
        return False
 
    resname = res.get_resname().strip()
    if resname in WATER_NAMES:
        return False
    if resname in COMMON_IONS:
        return False
    if is_aa(res, standard=True):
        return False
 
    heavy_atoms = [a for a in res.get_atoms() if a.element not in ("H", "")]
    return len(heavy_atoms) >= min_heavy_atoms
 
 
def get_heavy_atoms(res) -> List:
    return [a for a in res.get_atoms() if a.element not in ("H", "")]
 
 
def auto_detect_ligand(
    model,
    ligand_chain: Optional[str],
    ligand_resname: Optional[str],
    min_heavy_atoms: int,
    quiet: bool,
) -> List:
    """
    Returns list of residue objects for the selected ligand.
    Groups residues by resname; picks the group with the largest heavy-atom count
    unless ligand_resname is specified.
    """
    ligand_candidates: Dict[str, list] = defaultdict(list)
 
    for chain in model:
        if ligand_chain and chain.id != ligand_chain:
            continue
        for res in chain:
            if not is_ligand_candidate(res, min_heavy_atoms):
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
        msg += f" (min_heavy_atoms={min_heavy_atoms})"
        raise SystemExit(f"[ERROR] {msg}")
 
    if ligand_resname and ligand_resname in ligand_candidates:
        chosen = ligand_resname
    else:
        chosen = max(
            ligand_candidates,
            key=lambda k: sum(len(get_heavy_atoms(r)) for r in ligand_candidates[k])
        )
 
    chosen_residues = ligand_candidates[chosen]
    heavy_total = sum(len(get_heavy_atoms(r)) for r in chosen_residues)
 
    if not quiet:
        print(f"[INFO] Ligand selected: {chosen} "
              f"(n_residues={len(chosen_residues)}, heavy_atoms={heavy_total})")
 
    if len(ligand_candidates) > 1 and not ligand_resname:
        others = [k for k in ligand_candidates if k != chosen]
        if not quiet:
            print(f"[INFO] Other HETATM groups ignored: {others}. "
                  f"Use --ligand_resname to select a specific one.")
 
    return chosen_residues
 
 
# ---------------------------------------------------------------------------
# Residue selection
# ---------------------------------------------------------------------------
 
def residue_key(res) -> Tuple[str, int, str]:
    chain_id = res.get_parent().id
    hetflag, resseq, icode = res.id
    return chain_id, int(resseq), str(icode).strip()
 
 
def format_token(chain_id: str, resseq: int, icode: str) -> str:
    """Format residue as LigandMPNN token, e.g. 'A10' or 'A10A' with icode."""
    return f"{chain_id}{resseq}{icode}" if icode else f"{chain_id}{resseq}"
 
 
def select_residues_by_distance(
    model,
    ligand_residues: List,
    dist: float,
    include_hetatm_residues: bool,
    clash_warn_dist: float,
    quiet: bool,
) -> Tuple[Set[Tuple[str, int, str]], List[Dict[str, Any]]]:
    """
    Returns:
        near_res_keys : set of residue keys within dist Å of any ligand heavy atom
        clash_warnings: list of dicts for residues suspiciously close to ligand
    """
    ligand_atoms = [a for r in ligand_residues for a in get_heavy_atoms(r)]
    if not ligand_atoms:
        raise SystemExit("[ERROR] Ligand has no heavy atoms.")
 
    # Collect candidate protein/residue atoms
    candidate_atoms = []
    for chain in model:
        for res in chain:
            if not include_hetatm_residues and not is_aa(res, standard=True):
                continue
            for a in res.get_atoms():
                if a.element not in ("H", ""):
                    candidate_atoms.append(a)
 
    if not candidate_atoms:
        raise SystemExit(
            "[ERROR] No protein atoms found. "
            "Check --include_hetatm_residues if you expect HETATM residues."
        )
 
    ns = NeighborSearch(candidate_atoms)
 
    near_res_keys: Set[Tuple[str, int, str]] = set()
    clash_atom_pairs: List[Dict[str, Any]] = []
 
    for latom in ligand_atoms:
        # Search with full dist for selection
        close_atoms = ns.search(latom.coord, dist, level="A")
        for atom in close_atoms:
            res = atom.get_parent()
            if not include_hetatm_residues and not is_aa(res, standard=True):
                continue
            key = residue_key(res)
            near_res_keys.add(key)
 
            # Inline clash heuristic: element-aware minimum distance
            atom_dist = float(np.linalg.norm(latom.coord - atom.coord))
            r_lig  = COVALENT_RADII.get(latom.element.upper(), DEFAULT_COVALENT_RADIUS)
            r_prot = COVALENT_RADII.get(atom.element.upper(),  DEFAULT_COVALENT_RADIUS)
            threshold = max(clash_warn_dist, (r_lig + r_prot) * 0.75)
 
            if atom_dist < threshold:
                chain_id = res.get_parent().id
                clash_atom_pairs.append({
                    "residue":      format_token(chain_id, res.id[1], str(res.id[2]).strip()),
                    "resname":      res.get_resname().strip(),
                    "protein_atom": atom.fullname.strip(),
                    "ligand_atom":  latom.fullname.strip(),
                    "distance_A":   round(atom_dist, 3),
                    "threshold_A":  round(threshold, 3),
                })
 
    return near_res_keys, clash_atom_pairs
 
 
# ---------------------------------------------------------------------------
# Protein checksum
# ---------------------------------------------------------------------------
 
def protein_checksum(model) -> Dict[str, Any]:
    """
    Returns summary of protein residues for cross-run consistency verification.
    All PDBs in a screening campaign should produce identical checksums.
    """
    aa_residues = [
        res
        for chain in model
        for res in chain
        if is_aa(res, standard=True)
    ]
    if not aa_residues:
        return {"n_residues": 0, "first": None, "last": None, "chains": []}
 
    def key_str(res):
        c, n, i = residue_key(res)
        return format_token(c, n, i)
 
    chains = sorted({res.get_parent().id for res in aa_residues})
    return {
        "n_residues": len(aa_residues),
        "first":      key_str(aa_residues[0]),
        "last":       key_str(aa_residues[-1]),
        "chains":     chains,
    }
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    args = parse_args()
 
    if not os.path.isfile(args.pdb):
        raise SystemExit(f"[ERROR] PDB file not found: {args.pdb}")
 
    if not args.quiet:
        print(f"[INFO] Loading: {args.pdb}")
 
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", args.pdb)
    model = structure[args.model]
 
    # --- Protein checksum (consistency across campaign) ---
    chk = protein_checksum(model)
    if not args.quiet:
        print(f"[CHECK] Protein scaffold: n_residues={chk['n_residues']}, "
              f"chains={chk['chains']}, "
              f"first={chk['first']}, last={chk['last']}")
        print(f"        => This should be identical for all PDBs in the campaign.")
 
    # --- Ligand detection ---
    ligand_residues = auto_detect_ligand(
        model,
        ligand_chain=args.ligand_chain,
        ligand_resname=args.ligand_resname,
        min_heavy_atoms=args.min_heavy_atoms,
        quiet=args.quiet,
    )
 
    # --- Distance-based residue selection ---
    if not args.quiet:
        print(f"[INFO] Selecting residues within {args.dist:.2f} Å of ligand heavy atoms...")
 
    near_res_keys, clash_warnings = select_residues_by_distance(
        model=model,
        ligand_residues=ligand_residues,
        dist=args.dist,
        include_hetatm_residues=args.include_hetatm_residues,
        clash_warn_dist=args.clash_warn_dist,
        quiet=args.quiet,
    )
 
    # --- Sort and format tokens ---
    near_sorted = sorted(near_res_keys, key=lambda x: (x[0], x[1], x[2]))
    tokens = [format_token(c, n, i) for (c, n, i) in near_sorted]
    n_selected = len(tokens)
 
    # --- Sanity checks ---
    if n_selected < args.min_expected:
        print(f"[WARN] Only {n_selected} residue(s) selected — fewer than "
              f"--min_expected={args.min_expected}. "
              f"Check ligand detection or increase --dist.")
 
    if n_selected > args.max_expected:
        print(f"[WARN] {n_selected} residues selected — more than "
              f"--max_expected={args.max_expected}. "
              f"Consider reducing --dist.")
 
    if clash_warnings:
        print(f"[WARN] {len(clash_warnings)} atom pair(s) below clash threshold "
              f"({args.clash_warn_dist:.2f} Å):")
        for cw in clash_warnings[:10]:
            print(f"  residue={cw['residue']} ({cw['resname']}) "
                  f"atom={cw['protein_atom']} <-> ligand={cw['ligand_atom']} "
                  f"dist={cw['distance_A']:.3f} Å")
        if len(clash_warnings) > 10:
            print(f"  ... and {len(clash_warnings) - 10} more.")
        print(f"       => Run pose_qc.py for a full clash report.")
 
    # --- Write output ---
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_line = " ".join(tokens)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_line + "\n")
 
    if not args.quiet:
        print(f"[OK] {n_selected} residues selected -> {args.out}")
        print(f"     {out_line}")
        print(f"[INFO] n_positions={n_selected} — should be equal (or very close) "
              f"for all ligands in this campaign (same scaffold, same site).")
 
    # --- Optional JSON summary ---
    if args.out_json:
        summary = {
            "pdb":             args.pdb,
            "dist_A":          args.dist,
            "n_selected":      n_selected,
            "tokens":          tokens,
            "protein_checksum": chk,
            "n_clash_warnings": len(clash_warnings),
            "clash_warnings":  clash_warnings,
            "ligand_resname":  ligand_residues[0].get_resname().strip() if ligand_residues else None,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if not args.quiet:
            print(f"[OK] JSON summary -> {args.out_json}")
 
    # Exit with code 1 if clashes detected (so pipeline can catch it)
    if clash_warnings:
        sys.exit(2)   # 2 = soft warning (clashes), distinct from 1 = error
 
 
if __name__ == "__main__":
    main()
 