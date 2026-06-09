#!/usr/bin/env python3
"""
Combine docked SDF poses with protein to create protein-ligand complexes.

Takes GNINA output SDF files and combines them with the protein PDB to create
complete protein-ligand complex PDB files.

Usage:
    python combine_sdf_protein.py --protein protein.pdb --sdf_dir gnina_results --output_dir complexes
"""

import argparse
import sys
from pathlib import Path
from typing import List

try:
    from rdkit import Chem
except ImportError:
    print("ERROR: RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


# Supported ranking criteria and whether higher or lower is better
RANK_CRITERIA = {
    "CNNscore":          "higher",   # CNN pose quality,       higher = better
    "CNNaffinity":       "higher",   # CNN binding affinity,   higher = better
    "minimizedAffinity": "lower",    # Vina-like affinity,     lower  = better (more negative)
}


def read_protein_pdb(pdb_file: Path) -> List[str]:
    """Read protein PDB file and return ATOM/HETATM lines (excluding ligands)."""
    protein_lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("HETATM"):
                pass  # skip — replaced by docked pose
            elif line.startswith(("MODEL", "ENDMDL")):
                continue
            elif line.startswith(("HEADER", "TITLE", "REMARK", "CRYST1")):
                protein_lines.append(line)
    return protein_lines


def sdf_to_pdb_block(mol, chain_id: str = "L", res_name: str = "LIG", res_num: int = 1) -> List[str]:
    """Convert RDKit molecule to PDB HETATM lines."""
    pdb_lines = []
    conf = mol.GetConformer()
    atom_idx = 1
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        element = atom.GetSymbol()
        line = f"HETATM{atom_idx:5d}  {element:<3s} {res_name} {chain_id}{res_num:4d}    "
        line += f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
        line += f"  1.00  0.00          {element:>2s}\n"
        pdb_lines.append(line)
        atom_idx += 1
    return pdb_lines


def get_score(mol, prop: str):
    """Return float score for a property, or None if not present."""
    try:
        return float(mol.GetProp(prop))
    except KeyError:
        return None


def select_best_pose(suppl, rank_by: str):
    """
    Iterate over all poses in a supplier and return (best_mol, best_pose_idx, best_score).
    rank_by must be a key in RANK_CRITERIA.
    """
    direction = RANK_CRITERIA[rank_by]   # "higher" or "lower"
    best_mol   = None
    best_score = None
    best_idx   = None

    for pose_idx, mol in enumerate(suppl):
        if mol is None:
            continue
        score = get_score(mol, rank_by)
        if score is None:
            # Property missing — fall back to first pose
            if best_mol is None:
                best_mol, best_score, best_idx = mol, score, pose_idx
            continue

        if best_score is None:
            best_mol, best_score, best_idx = mol, score, pose_idx
        else:
            if direction == "higher" and score > best_score:
                best_mol, best_score, best_idx = mol, score, pose_idx
            elif direction == "lower" and score < best_score:
                best_mol, best_score, best_idx = mol, score, pose_idx

    return best_mol, best_idx, best_score


def process_sdf_file(sdf_file: Path, protein_lines: List[str], output_dir: Path,
                     ligand_name: str = None, best_only: bool = True,
                     rank_by: str = "CNNscore") -> int:
    """
    Process a single SDF file with multiple poses and write complex PDB file(s).

    Args:
        sdf_file:      Path to GNINA output SDF
        protein_lines: Protein PDB lines
        output_dir:    Output directory
        ligand_name:   Name prefix for output files
        best_only:     If True save only the best pose; otherwise save all poses
        rank_by:       Score property to use when selecting the best pose
                       ("CNNscore" | "CNNaffinity" | "minimizedAffinity")

    Returns number of complexes written.
    """
    if ligand_name is None:
        ligand_name = sdf_file.stem.replace("_docked", "")

    direction = RANK_CRITERIA.get(rank_by, "higher")

    if best_only:
        # Need to scan all poses to find the best → read twice
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        if suppl is None:
            print(f"  ERROR: Could not read SDF: {sdf_file}")
            return 0

        best_mol, best_idx, best_score = select_best_pose(suppl, rank_by)

        if best_mol is None:
            print(f"  ERROR: No valid poses found in {sdf_file.name}")
            return 0

        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        print(f"  Selected pose {best_idx + 1}  |  {rank_by} = {score_str}"
              f"  ({'higher is better' if direction == 'higher' else 'lower is better'})")

        # Print all poses for transparency
        suppl2 = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        print(f"  {'Pose':>4}  {'CNNscore':>10}  {'CNNaffinity':>12}  {'minimizedAffinity':>18}  {'selected':>8}")
        print("  " + "-" * 60)
        for idx, mol in enumerate(suppl2):
            if mol is None:
                continue
            cnn  = get_score(mol, "CNNscore")
            caff = get_score(mol, "CNNaffinity")
            maff = get_score(mol, "minimizedAffinity")
            sel  = "  ★" if idx == best_idx else ""
            cnn_s  = f"{cnn:>10.4f}"  if cnn  is not None else "       N/A"
            caff_s = f"{caff:>+12.4f}" if caff is not None else "         N/A"
            maff_s = f"{maff:>+18.4f}" if maff is not None else "               N/A"
            print(f"  {idx+1:>4}  {cnn_s}  {caff_s}  {maff_s}{sel}")

        output_file = output_dir / f"{ligand_name}_best_{rank_by}.pdb"
        with open(output_file, "w") as f:
            f.writelines(protein_lines)
            f.writelines(sdf_to_pdb_block(best_mol))
            f.write("END\n")
        print(f"  → Written: {output_file.name}")
        return 1

    else:
        # Save every pose
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        if suppl is None:
            print(f"  ERROR: Could not read SDF: {sdf_file}")
            return 0

        count = 0
        print(f"  {'Pose':>4}  {'CNNscore':>10}  {'CNNaffinity':>12}  {'minimizedAffinity':>18}")
        print("  " + "-" * 50)
        for pose_idx, mol in enumerate(suppl):
            if mol is None:
                continue
            cnn  = get_score(mol, "CNNscore")
            caff = get_score(mol, "CNNaffinity")
            maff = get_score(mol, "minimizedAffinity")
            cnn_s  = f"{cnn:>10.4f}"   if cnn  is not None else "       N/A"
            caff_s = f"{caff:>+12.4f}" if caff is not None else "         N/A"
            maff_s = f"{maff:>+18.4f}" if maff is not None else "               N/A"
            print(f"  {pose_idx+1:>4}  {cnn_s}  {caff_s}  {maff_s}")

            output_file = output_dir / f"{ligand_name}_pose_{pose_idx + 1}.pdb"
            with open(output_file, "w") as f:
                f.writelines(protein_lines)
                f.writelines(sdf_to_pdb_block(mol))
                f.write("END\n")
            count += 1

        print(f"  → Written {count} pose(s)")
        return count


def main():
    parser = argparse.ArgumentParser(
        description="Combine docked SDF poses with protein to create complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ranking criteria (--rank_by):
  CNNscore           CNN pose quality score      (higher = better)
  CNNaffinity        CNN predicted affinity       (higher = better)
  minimizedAffinity  Vina-like affinity (kcal/mol)(lower  = better, more negative)

Examples:
  # Best pose by CNNscore (default)
  python combine_sdf_protein.py -p protein.pdb -s gnina_results -o complexes --best_only

  # Best pose by minimizedAffinity (most negative = strongest binder)
  python combine_sdf_protein.py -p protein.pdb -s gnina_results -o complexes --best_only --rank_by minimizedAffinity

  # All poses for a single file
  python combine_sdf_protein.py -p protein.pdb -f ligand_docked.sdf -o complexes

  # Best pose by CNNaffinity, single file
  python combine_sdf_protein.py -p protein.pdb -f ligand_docked.sdf -o complexes --best_only --rank_by CNNaffinity
        """
    )

    parser.add_argument("-p", "--protein",    required=True, type=Path,
                        help="Protein PDB file (without ligand)")
    parser.add_argument("-s", "--sdf_dir",    type=Path,
                        help="Directory containing docked SDF files")
    parser.add_argument("-f", "--sdf_file",   type=Path,
                        help="Single SDF file to process")
    parser.add_argument("-o", "--output_dir", required=True, type=Path,
                        help="Output directory for complex PDB files")
    parser.add_argument("--pattern",  default="*_docked.sdf",
                        help="Glob pattern to match SDF files in sdf_dir (default: *_docked.sdf)")
    parser.add_argument("--best_only", action="store_true",
                        help="Save only the best scoring pose per ligand (requires --rank_by)")
    parser.add_argument("--rank_by",
                        choices=list(RANK_CRITERIA.keys()),
                        default="CNNscore",
                        help="Score used to select the best pose when --best_only is set "
                             "(default: CNNscore)")

    args = parser.parse_args()

    # Validate inputs
    if not args.protein.exists():
        print(f"ERROR: Protein file not found: {args.protein}")
        sys.exit(1)
    if args.sdf_file and args.sdf_dir:
        print("ERROR: Specify either --sdf_dir or --sdf_file, not both")
        sys.exit(1)
    if not args.sdf_file and not args.sdf_dir:
        print("ERROR: Must specify either --sdf_dir or --sdf_file")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combine SDF + Protein → Complex PDB")
    print("=" * 60)
    print(f"Protein:     {args.protein}")
    print(f"Output dir:  {args.output_dir}")
    if args.best_only:
        direction = RANK_CRITERIA[args.rank_by]
        print(f"Mode:        best pose only  |  ranked by: {args.rank_by} ({direction} is better)")
    else:
        print("Mode:        all poses")
    print("=" * 60)

    # Read protein
    print("\n[1/2] Reading protein structure…")
    protein_lines = read_protein_pdb(args.protein)
    print(f"  Read {len(protein_lines)} lines from {args.protein.name}")

    # Collect SDF files
    if args.sdf_file:
        if not args.sdf_file.exists():
            print(f"ERROR: SDF file not found: {args.sdf_file}")
            sys.exit(1)
        sdf_files = [args.sdf_file]
    else:
        sdf_files = sorted(args.sdf_dir.glob(args.pattern))
        if not sdf_files:
            print(f"ERROR: No SDF files matching '{args.pattern}' in {args.sdf_dir}")
            sys.exit(1)

    print(f"\n[2/2] Processing {len(sdf_files)} SDF file(s)…\n")

    total_complexes = 0
    for i, sdf_file in enumerate(sdf_files, 1):
        print(f"[{i}/{len(sdf_files)}] {sdf_file.name}")
        count = process_sdf_file(
            sdf_file, protein_lines, args.output_dir,
            best_only=args.best_only, rank_by=args.rank_by
        )
        total_complexes += count
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SDF files processed : {len(sdf_files)}")
    print(f"Complexes created   : {total_complexes}")
    print(f"Output directory    : {args.output_dir}")

    pdb_files = sorted(args.output_dir.glob("*.pdb"))
    if pdb_files:
        print("\nOutput files:")
        for pdb in pdb_files[:8]:
            print(f"  {pdb.name}")
        if len(pdb_files) > 8:
            print(f"  … and {len(pdb_files) - 8} more")

    print("\nAll done!")


if __name__ == "__main__":
    main()