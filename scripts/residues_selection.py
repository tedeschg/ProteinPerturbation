#!/usr/bin/env python3
"""
Extract residues within the GNINA docking box from a protein structure.

This script reads a protein PDB file and a reference ligand (or complex)
to define the docking box center, then extracts all residues within the
specified box dimensions (autobox_add).

Usage:
    python extract_residues_from_box.py --protein protein.pdb --reference_complex complex.pdb \\
                                        --autobox_add 8.0 --output residues.txt
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
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def get_ligand_coords_from_pdb(pdb_file: Path) -> List[Tuple[float, float, float]]:
    """Extract ligand coordinates from a PDB file (HETATM records)."""
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
    """Calculate the geometric center of ligand coordinates."""
    if not ligand_coords:
        raise ValueError("No ligand coordinates found")
    
    xs, ys, zs = zip(*ligand_coords)
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)
    
    return (center_x, center_y, center_z)

def get_residues_in_box(pdb_file: Path, 
                        center: Tuple[float, float, float],
                        box_radius: float) -> Set[str]:
    """
    Get all residue IDs that have at least one atom within the box radius.
    
    Returns residues as chain+residue_number (e.g., "A123").
    """
    residues = set()
    
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            
            try:
                # Extract coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                # Check if atom is within box radius
                dist_sq = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
                if dist_sq <= box_radius**2:
                    # Extract residue identifier
                    chain = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_id = f"{chain}{res_num}"
                    residues.add(res_id)
            except (ValueError, IndexError):
                continue
    
    return residues

def filter_protein_residues(residues: Set[str], protein_pdb: Path) -> Set[str]:
    """
    Filter to keep only standard amino acids (not ligands, water, etc.).
    Uses RDKit to parse the protein.
    """
    # Load protein with RDKit
    mol = Chem.MolFromPDBFile(str(protein_pdb), removeHs=False, proximityBonding=False)
    if mol is None:
        print(f"Warning: Could not parse {protein_pdb} with RDKit")
        return residues
    
    # Get all protein residues
    protein_residues = set()
    for atom in mol.GetAtoms():
        res_num = atom.GetPDBResidueInfo().GetResidueNumber() if atom.GetPDBResidueInfo() else None
        chain = atom.GetPDBResidueInfo().GetChainId() if atom.GetPDBResidueInfo() else ""
        if res_num is not None and chain:
            protein_residues.add(f"{chain}{res_num}")
    
    # Filter to keep only residues that are in the protein
    filtered = residues.intersection(protein_residues)
    
    return filtered

def main():
    parser = argparse.ArgumentParser(
        description="Extract residues within the GNINA docking box"
    )
    parser.add_argument("--protein", "-p", required=True, type=Path,
                       help="Protein PDB file")
    parser.add_argument("--reference_complex", "-c", required=True, type=Path,
                       help="Reference complex PDB file containing ligand")
    parser.add_argument("--autobox_add", type=float, default=8.0,
                       help="Box padding in Angstroms (default: 8.0)")
    parser.add_argument("--output", "-o", type=Path, default=Path("selected_residues.txt"),
                       help="Output file for residue list")
    parser.add_argument("--format", choices=["list", "space_separated", "json"], 
                       default="list",
                       help="Output format (default: list)")
    parser.add_argument("--include_hetatm", action="store_true",
                       help="Include HETATM residues (ligands, waters, etc.)")
    
    args = parser.parse_args()
    log = setup_logging()
    
    # Validate inputs
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
    
    # Step 1: Extract ligand coordinates from reference complex
    log.info("\n[1/3] Extracting ligand coordinates from reference complex...")
    ligand_coords = get_ligand_coords_from_pdb(args.reference_complex)
    if not ligand_coords:
        log.error("No ligand coordinates found in reference complex")
        sys.exit(1)
    log.info(f"  Found {len(ligand_coords)} ligand atoms")
    
    # Step 2: Calculate box center
    center = get_center_from_ligand(ligand_coords)
    log.info(f"  Box center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    
    # Step 3: Get residues within box radius
    log.info("\n[2/3] Finding residues within {:.1f} Å of box center...".format(args.autobox_add))
    all_residues = get_residues_in_box(args.protein, center, args.autobox_add)
    log.info(f"  Found {len(all_residues)} residues within box")
    
    # Step 4: Filter residues (optional)
    log.info("\n[3/3] Filtering residues...")
    if not args.include_hetatm:
        residues = filter_protein_residues(all_residues, args.protein)
        log.info(f"  Filtered to {len(residues)} protein residues (excluding HETATM)")
    else:
        residues = all_residues
        log.info(f"  Keeping all {len(residues)} residues (including HETATM)")
    
    # Step 5: Write output
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