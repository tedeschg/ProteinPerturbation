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


def read_protein_pdb(pdb_file: Path) -> List[str]:
    """
    Read protein PDB file and return ATOM/HETATM lines (excluding ligands).
    """
    protein_lines = []
    with open(pdb_file) as f:
        for line in f:
            # Keep ATOM lines and selected HETATM (waters, ions, etc.)
            # Skip ligand HETATM which will be replaced by docked poses
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("HETATM"):
                # You can add logic here to keep specific HETATMs like waters
                # For now, skip all HETATM to avoid duplicating ligands
                pass
            elif line.startswith(("CRYST1", "MODEL", "ENDMDL")):
                # Skip MODEL/ENDMDL from multi-model PDBs
                continue
            elif line.startswith(("HEADER", "TITLE", "REMARK", "CRYST1")):
                protein_lines.append(line)

    return protein_lines


def sdf_to_pdb_block(mol, chain_id: str = "L", res_name: str = "LIG", res_num: int = 1) -> List[str]:
    """
    Convert RDKit molecule to PDB HETATM lines.
    """
    pdb_lines = []

    conf = mol.GetConformer()
    atom_idx = 1

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        element = atom.GetSymbol()

        # PDB HETATM format
        line = f"HETATM{atom_idx:5d}  {element:<3s} {res_name} {chain_id}{res_num:4d}    "
        line += f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
        line += f"  1.00  0.00          {element:>2s}\n"

        pdb_lines.append(line)
        atom_idx += 1

    return pdb_lines


def process_sdf_file(sdf_file: Path, protein_lines: List[str], output_dir: Path,
                     ligand_name: str = None, best_only: bool = True) -> int:
    """
    Process single SDF file with multiple poses and create complex PDB files.

    Args:
        sdf_file: Path to SDF file with docked poses
        protein_lines: Protein PDB lines
        output_dir: Output directory
        ligand_name: Name for output files
        best_only: If True, only save the best scoring pose (first pose)

    Returns number of complexes created.
    """
    if ligand_name is None:
        ligand_name = sdf_file.stem.replace("_docked", "")

    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)

    if suppl is None:
        print(f"  ERROR: Could not read SDF file: {sdf_file}")
        return 0

    count = 0
    for pose_idx, mol in enumerate(suppl):
        if mol is None:
            continue

        # Get CNNscore or CNNaffinity if available
        score = None
        if mol.HasProp("CNNscore"):
            score = float(mol.GetProp("CNNscore"))
        elif mol.HasProp("CNNaffinity"):
            score = float(mol.GetProp("CNNaffinity"))
        elif mol.HasProp("minimizedAffinity"):
            score = float(mol.GetProp("minimizedAffinity"))

        # Create output filename
        if best_only:
            # Only save first (best) pose
            output_file = output_dir / f"{ligand_name}_best.pdb"
        else:
            output_file = output_dir / f"{ligand_name}_pose_{pose_idx + 1}.pdb"

        # Combine protein + ligand
        with open(output_file, 'w') as f:
            # Write protein
            f.writelines(protein_lines)

            # Write ligand as HETATM
            ligand_lines = sdf_to_pdb_block(mol, chain_id="L", res_name="LIG", res_num=1)
            f.writelines(ligand_lines)

            # End of file
            f.write("END\n")

        count += 1

        # Print score if available
        if score is not None:
            print(f"    Pose {pose_idx + 1}: score = {score:.3f}")

        # If best_only, stop after first pose
        if best_only:
            break

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Combine docked SDF poses with protein to create complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all SDF files from GNINA output
  python combine_sdf_protein.py -p protein.pdb -s gnina_results -o complexes

  # Process single SDF file
  python combine_sdf_protein.py -p protein.pdb --sdf_file ligand_docked.sdf -o complexes
        """
    )

    parser.add_argument("-p", "--protein", required=True, type=Path,
                        help="Protein PDB file (without ligand)")
    parser.add_argument("-s", "--sdf_dir", type=Path,
                        help="Directory containing docked SDF files")
    parser.add_argument("-f", "--sdf_file", type=Path,
                        help="Single SDF file to process")
    parser.add_argument("-o", "--output_dir", required=True, type=Path,
                        help="Output directory for complex PDB files")
    parser.add_argument("--pattern", default="*_docked.sdf",
                        help="Pattern to match SDF files (default: *_docked.sdf)")
    parser.add_argument("--best_only", action="store_true",
                        help="Only save the best scoring pose (first pose) for each ligand")

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

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combine SDF + Protein to Create Complexes")
    print("=" * 60)
    print(f"Protein:     {args.protein}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 60)
    print()

    # Read protein
    print("[1/2] Reading protein structure...")
    protein_lines = read_protein_pdb(args.protein)
    print(f"  Read {len(protein_lines)} lines from protein PDB")
    print()

    # Get SDF files to process
    if args.sdf_file:
        sdf_files = [args.sdf_file]
    else:
        sdf_files = sorted(args.sdf_dir.glob(args.pattern))

    if not sdf_files:
        print(f"ERROR: No SDF files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"[2/2] Processing {len(sdf_files)} SDF file(s)...")
    print()

    # Process each SDF file
    total_complexes = 0
    for i, sdf_file in enumerate(sdf_files, 1):
        print(f"[{i}/{len(sdf_files)}] Processing: {sdf_file.name}")

        count = process_sdf_file(sdf_file, protein_lines, args.output_dir, best_only=args.best_only)
        if args.best_only:
            print(f"  Created best pose complex")
        else:
            print(f"  Created {count} complex(es)")
        total_complexes += count

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total SDF files processed:  {len(sdf_files)}")
    print(f"Total complexes created:    {total_complexes}")
    print(f"Output directory:           {args.output_dir}")
    print()

    # List some examples
    pdb_files = sorted(args.output_dir.glob("*.pdb"))
    if pdb_files:
        print("Example complexes:")
        for pdb in pdb_files[:5]:
            print(f"  - {pdb.name}")
        if len(pdb_files) > 5:
            print(f"  ... and {len(pdb_files) - 5} more")

    print()
    print("All done!")


if __name__ == "__main__":
    main()
