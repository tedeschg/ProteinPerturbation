#!/usr/bin/env python3
"""
GNINA Automated Docking Script

Performs automated molecular docking using GNINA with a reference ligand
to define the binding site.

Usage:
    # Single 3D ligand
    python gnina_docking.py --protein protein.pdb --reference_complex complex.pdb \
                            --ligand ligand.sdf --output_dir results

    # Multiple 3D ligands (folder)
    python gnina_docking.py --protein protein.pdb --reference_complex complex.pdb \
                            --ligands_dir ./my_ligands/ --output_dir results

    # SMILES file (original mode, RDKit generates 3D)
    python gnina_docking.py --protein protein.pdb --reference_complex complex.pdb \
                            --smiles ligands.smi --output_dir results
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import shutil

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_smiles(smiles_file: Path) -> List[Tuple[str, str, str]]:
    """
    Read SMILES file and return list of (SMILES, ID, name) tuples.
    Format: SMILES ID name (space/tab separated)
    """
    ligands = []
    with open(smiles_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                smiles = parts[0]
                lig_id = parts[1] if len(parts) >= 2 else f"lig_{line_num}"
                name   = parts[2] if len(parts) >= 3 else lig_id
                ligands.append((smiles, lig_id, name))
    return ligands


def smiles_to_sdf(smiles: str, output_file: Path, lig_id: str = "ligand") -> bool:
    """Convert SMILES to 3D SDF using RDKit. Returns True on success."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ERROR: Invalid SMILES: {smiles}")
            return False

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            print(f"  WARNING: Retrying 3D embedding with random coords…")
            result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if result != 0:
                print(f"  ERROR: Failed to generate 3D coordinates for {smiles}")
                return False

        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        writer = Chem.SDWriter(str(output_file))
        mol.SetProp("_Name", lig_id)
        writer.write(mol)
        writer.close()
        return True

    except Exception as e:
        print(f"  ERROR converting SMILES to SDF: {e}")
        return False


def mol2_or_pdb_to_sdf(input_file: Path, output_file: Path) -> bool:
    """
    Convert a single .mol2 or .pdb ligand file to SDF using RDKit.
    Returns True on success.
    """
    suffix = input_file.suffix.lower()
    try:
        if suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(input_file), removeHs=False)
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(input_file), removeHs=False)
        elif suffix in (".sdf", ".mol"):
            # Already SDF-compatible — just copy
            shutil.copy(input_file, output_file)
            return True
        else:
            print(f"  ERROR: Unsupported format: {suffix}")
            return False

        if mol is None:
            print(f"  ERROR: RDKit could not read {input_file.name}")
            return False

        writer = Chem.SDWriter(str(output_file))
        mol.SetProp("_Name", input_file.stem)
        writer.write(mol)
        writer.close()
        return True

    except Exception as e:
        print(f"  ERROR converting {input_file.name} to SDF: {e}")
        return False


def extract_ligand_from_complex(complex_pdb: Path, output_sdf: Path) -> bool:
    """Extract HETATM ligand from a complex PDB and save as SDF."""
    try:
        hetero_atoms = []
        with open(complex_pdb) as f:
            for line in f:
                if line.startswith("HETATM"):
                    hetero_atoms.append(line)

        if not hetero_atoms:
            print(f"  ERROR: No HETATM found in {complex_pdb}")
            return False

        temp_pdb = output_sdf.with_suffix(".temp.pdb")
        with open(temp_pdb, "w") as f:
            f.writelines(hetero_atoms)
            f.write("END\n")

        mol = Chem.MolFromPDBFile(str(temp_pdb), removeHs=False)
        if mol is None:
            print(f"  ERROR: Could not read ligand from {temp_pdb}")
            temp_pdb.unlink()
            return False

        writer = Chem.SDWriter(str(output_sdf))
        writer.write(mol)
        writer.close()
        temp_pdb.unlink()
        return True

    except Exception as e:
        print(f"  ERROR extracting ligand: {e}")
        return False


def run_gnina(protein_pdb: Path, ligand_sdf: Path, reference_sdf: Path,
              output_sdf: Path, autobox_add: float = 8.0,
              exhaustiveness: int = 8, num_modes: int = 9) -> bool:
    """Run GNINA docking with autobox centred on the reference ligand."""
    cmd = [
        "gnina",
        "-r", str(protein_pdb),
        "-l", str(ligand_sdf),
        "--autobox_ligand", str(reference_sdf),
        "--autobox_add",    str(autobox_add),
        "-o",               str(output_sdf),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes",      str(num_modes),
        "--cpu", "1",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: GNINA failed (code {result.returncode})")
            print(f"  STDERR: {result.stderr}")
            return False
        if not output_sdf.exists():
            print(f"  ERROR: Output file not created: {output_sdf}")
            return False
        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: GNINA timed out after 600 s")
        return False
    except FileNotFoundError:
        print("  ERROR: 'gnina' not found. Please install GNINA.")
        return False
    except Exception as e:
        print(f"  ERROR running GNINA: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated GNINA docking — SMILES, single 3D file, or folder of 3D files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes (choose exactly one):
  --smiles      File with SMILES strings  →  RDKit generates 3D
  --ligand      Single 3D file (.sdf / .mol2 / .pdb)
  --ligands_dir Folder of 3D files        →  all .sdf/.mol2/.pdb inside

Examples:
  python gnina_docking.py -p protein.pdb -c complex.pdb -s ligands.smi -o results
  python gnina_docking.py -p protein.pdb -c complex.pdb --ligand mol.sdf -o results
  python gnina_docking.py -p protein.pdb -c complex.pdb --ligands_dir ./3d/ -o results
        """
    )

    parser.add_argument("-p", "--protein",          required=True,  type=Path,
                        help="Protein receptor PDB (apo structure)")
    parser.add_argument("-c", "--reference_complex", required=True,  type=Path,
                        help="Reference protein-ligand complex PDB (defines binding site)")
    parser.add_argument("-o", "--output_dir",        required=True,  type=Path,
                        help="Output directory")

    # --- input modes (mutually exclusive) ---
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("-s", "--smiles",      type=Path,
                     help="SMILES file (SMILES ID name, one per line)")
    inp.add_argument("-l", "--ligand",      type=Path,
                     help="Single 3D ligand file (.sdf / .mol2 / .pdb)")
    inp.add_argument("-L", "--ligands_dir", type=Path,
                     help="Folder containing 3D ligand files (.sdf / .mol2 / .pdb)")

    # --- docking parameters ---
    parser.add_argument("--autobox_add",    type=float, default=8.0,
                        help="Box padding around reference ligand in Å (default: 8.0)")
    parser.add_argument("--exhaustiveness", type=int,   default=8,
                        help="GNINA exhaustiveness (default: 8)")
    parser.add_argument("--num_modes",      type=int,   default=9,
                        help="Poses per ligand (default: 9)")
    parser.add_argument("--keep_temp",      action="store_true",
                        help="Keep temporary files")

    args = parser.parse_args()

    # --- validate common inputs ---
    for label, path in [("Protein", args.protein), ("Reference complex", args.reference_complex)]:
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GNINA Automated Docking")
    print("=" * 60)
    print(f"Protein:           {args.protein}")
    print(f"Reference complex: {args.reference_complex}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Autobox padding:   {args.autobox_add} Å")
    print(f"Exhaustiveness:    {args.exhaustiveness}")
    print(f"Poses per ligand:  {args.num_modes}")
    print("=" * 60)

    # --- Step 1: extract reference ligand ---
    print("\n[1/3] Extracting reference ligand from complex…")
    reference_sdf = temp_dir / "reference_ligand.sdf"
    if not extract_ligand_from_complex(args.reference_complex, reference_sdf):
        print("ERROR: Failed to extract reference ligand")
        sys.exit(1)
    print(f"  Reference ligand saved: {reference_sdf}")

    # --- Step 2: build ligand list ---
    print("\n[2/3] Collecting ligands…")

    SUPPORTED = {".sdf", ".mol", ".mol2", ".pdb"}

    # Each entry: (ligand_path_ready_for_gnina, display_name)
    # ligand_path may point to temp_dir if conversion was needed
    ligand_queue: List[Tuple[Path, str]] = []

    if args.smiles:
        if not args.smiles.exists():
            print(f"ERROR: SMILES file not found: {args.smiles}")
            sys.exit(1)
        entries = read_smiles(args.smiles)
        print(f"  Found {len(entries)} SMILES entries")
        for smiles, lig_id, name in entries:
            sdf = temp_dir / f"{lig_id}_input.sdf"
            print(f"  Converting {name} to 3D…", end=" ")
            if smiles_to_sdf(smiles, sdf, lig_id):
                print("OK")
                ligand_queue.append((sdf, name))
            else:
                print("FAILED — skipped")

    elif args.ligand:
        if not args.ligand.exists():
            print(f"ERROR: Ligand file not found: {args.ligand}")
            sys.exit(1)
        suffix = args.ligand.suffix.lower()
        if suffix not in SUPPORTED:
            print(f"ERROR: Unsupported format {suffix}. Use: {SUPPORTED}")
            sys.exit(1)
        if suffix in (".sdf", ".mol"):
            ligand_queue.append((args.ligand, args.ligand.stem))
        else:
            sdf = temp_dir / f"{args.ligand.stem}.sdf"
            print(f"  Converting {args.ligand.name} → SDF…", end=" ")
            if mol2_or_pdb_to_sdf(args.ligand, sdf):
                print("OK")
                ligand_queue.append((sdf, args.ligand.stem))
            else:
                print("FAILED")
                sys.exit(1)

    elif args.ligands_dir:
        if not args.ligands_dir.is_dir():
            print(f"ERROR: Folder not found: {args.ligands_dir}")
            sys.exit(1)
        files = sorted(f for f in args.ligands_dir.iterdir()
                       if f.suffix.lower() in SUPPORTED)
        if not files:
            print(f"ERROR: No supported files found in {args.ligands_dir}")
            sys.exit(1)
        print(f"  Found {len(files)} files")
        for f in files:
            suffix = f.suffix.lower()
            if suffix in (".sdf", ".mol"):
                ligand_queue.append((f, f.stem))
            else:
                sdf = temp_dir / f"{f.stem}.sdf"
                print(f"  Converting {f.name} → SDF…", end=" ")
                if mol2_or_pdb_to_sdf(f, sdf):
                    print("OK")
                    ligand_queue.append((sdf, f.stem))
                else:
                    print("FAILED — skipped")

    print(f"  Ligands ready for docking: {len(ligand_queue)}")

    # --- Step 3: dock ---
    print("\n[3/3] Running GNINA docking…")
    results = []

    for i, (ligand_path, name) in enumerate(ligand_queue, 1):
        print(f"\n[{i}/{len(ligand_queue)}] {name}")
        output_sdf = args.output_dir / f"{name}_docked.sdf"

        if run_gnina(args.protein, ligand_path, reference_sdf, output_sdf,
                     args.autobox_add, args.exhaustiveness, args.num_modes):
            print(f"  ✓ {output_sdf}")
            results.append((name, "SUCCESS", str(output_sdf)))
        else:
            print("  ✗ Failed")
            results.append((name, "FAILED", "GNINA error"))

    # --- Summary ---
    print()
    print("=" * 60)
    print("DOCKING SUMMARY")
    print("=" * 60)
    success = sum(1 for r in results if r[1] == "SUCCESS")
    print(f"Total:      {len(results)}")
    print(f"Successful: {success}")
    print(f"Failed:     {len(results) - success}")

    summary_file = args.output_dir / "docking_summary.csv"
    with open(summary_file, "w") as f:
        f.write("name,status,output\n")
        for name, status, output in results:
            f.write(f"{name},{status},{output}\n")
    print(f"\nSummary saved: {summary_file}")

    # --- Cleanup ---
    if not args.keep_temp:
        shutil.rmtree(temp_dir)
        print("Temporary files removed.")
    else:
        print(f"Temporary files kept in: {temp_dir}")

    print("\nAll done!")


if __name__ == "__main__":
    main()