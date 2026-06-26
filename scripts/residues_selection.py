#!/usr/bin/env python3
"""
Extract residues within the GNINA docking box from a protein structure.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List, Set, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFMCS
except ImportError:
    print("ERROR: RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def get_ligand_coords_from_pdb(pdb_file: Path) -> List[Tuple[float, float, float]]:
    coords = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except (ValueError, IndexError):
                    continue
    return coords

def get_center_from_ligand(ligand_coords: List[Tuple]) -> Tuple[float, float, float]:
    if not ligand_coords:
        raise ValueError("No ligand coordinates found")
    xs, ys, zs = zip(*ligand_coords)
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))

def get_residues_in_box(pdb_file: Path, center: Tuple[float, float, float], box_radius: float) -> Set[str]:
    residues = set()
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                dist_sq = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
                if dist_sq <= box_radius**2:
                    chain = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_id = f"{chain}{res_num}"
                    residues.add(res_id)
            except (ValueError, IndexError):
                continue
    return residues

def filter_protein_residues(residues: Set[str], protein_pdb: Path) -> Set[str]:
    """
    Filter to keep only standard amino acid residues (ATOM records only).

    Reads the PDB line by line and collects residue IDs that appear on ATOM
    records.  HETATM records (ligands, waters, ions, co-factors) are ignored,
    so the ligand itself is never included in the returned set even when its
    coordinates happen to fall within the docking box.
    """
    atom_residues: Set[str] = set()

    with open(protein_pdb) as fh:
        for line in fh:
            # Only standard protein/nucleic-acid atoms — skip HETATM entirely
            if not line.startswith("ATOM"):
                continue
            try:
                chain   = line[21:22].strip()
                res_num = int(line[22:26].strip())
                if chain:
                    atom_residues.add(f"{chain}{res_num}")
            except (ValueError, IndexError):
                continue

    return residues.intersection(atom_residues)

def main():
    parser = argparse.ArgumentParser(description="Extract residues within the GNINA docking box")
    parser.add_argument("--protein", "-p", required=True, type=Path)
    parser.add_argument("--reference_complex", "-c", required=True, type=Path)
    parser.add_argument("--autobox_add", type=float, default=8.0)
    parser.add_argument("--output", "-o", type=Path, default=Path("selected_residues.txt"))
    parser.add_argument("--format", choices=["list", "space_separated", "json"], default="list")
    parser.add_argument("--include_hetatm", action="store_true")
    args = parser.parse_args()

    log = setup_logging()

    for f in [args.protein, args.reference_complex]:
        if not f.exists():
            log.error(f"File not found: {f}")
            sys.exit(1)

    log.info("=" * 60)
    log.info("Extracting residues from GNINA docking box")
    log.info("=" * 60)
    log.info(f"Protein:           {args.protein}")
    log.info(f"Reference complex: {args.reference_complex}")
    log.info(f"Box radius:        {args.autobox_add} Å")
    log.info(f"Output:            {args.output}")

    log.info("\n[1/3] Extracting ligand coordinates from reference complex...")
    ligand_coords = get_ligand_coords_from_pdb(args.reference_complex)
    if not ligand_coords:
        log.error("No ligand coordinates found in reference complex")
        sys.exit(1)
    log.info(f"  Found {len(ligand_coords)} ligand atoms")

    center = get_center_from_ligand(ligand_coords)
    log.info(f"  Box center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

    log.info("\n[2/3] Finding residues within {:.1f} Å of box center...".format(args.autobox_add))
    all_residues = get_residues_in_box(args.protein, center, args.autobox_add)
    log.info(f"  Found {len(all_residues)} residues within box")

    log.info("\n[3/3] Filtering residues...")
    if not args.include_hetatm:
        residues = filter_protein_residues(all_residues, args.protein)
        log.info(f"  Filtered to {len(residues)} protein residues (excluding HETATM)")
    else:
        residues = all_residues
        log.info(f"  Keeping all {len(residues)} residues (including HETATM)")

    residues_sorted = sorted(residues, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    with open(args.output, "w") as f:
        if args.format == "list":
            f.write("\n".join(residues_sorted))
        elif args.format == "space_separated":
            f.write(" ".join(residues_sorted))
        elif args.format == "json":
            import json
            json.dump(residues_sorted, f, indent=2)

    log.info(f"\n  Residue list saved to: {args.output}")
    log.info(f"  Total residues: {len(residues_sorted)}")
    log.info(f"  First 10 residues: {', '.join(residues_sorted[:10])}")
    if len(residues_sorted) > 10:
        log.info(f"  ... and {len(residues_sorted) - 10} more")
    log.info("\nAll done!")

if __name__ == "__main__":
    main()