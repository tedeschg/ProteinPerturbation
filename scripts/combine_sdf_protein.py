"""
Combine docked poses (SDF, MOL2, or PDB) with protein (PDB or MOL2) to create
protein-ligand complexes.

Takes GNINA output SDF/MOL2/PDB files and combines them with the protein
structure to create complete protein-ligand complex PDB files.

Usage:
    # SDF ligand + PDB protein (original behaviour)
    python combine_docked_protein.py -p protein.pdb -s gnina_results -o complexes --best_only

    # MOL2 ligand + PDB protein
    python combine_docked_protein.py -p protein.pdb -s docked_mol2 -o complexes --pattern "*.mol2" --best_only

    # PDB ligand + PDB protein
    python combine_docked_protein.py -p protein.pdb -s docked_pdb -o complexes --pattern "*.pdb" --best_only

    # SDF ligand + MOL2 protein
    python combine_docked_protein.py -p protein.mol2 -s gnina_results -o complexes --best_only

    # MOL2 ligand + MOL2 protein, single file, all poses
    python combine_docked_protein.py -p protein.mol2 -f ligand_docked.mol2 -o complexes

    # Best pose by minimizedAffinity
    python combine_docked_protein.py -p protein.pdb -s gnina_results -o complexes \\
        --best_only --rank_by minimizedAffinity
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from rdkit import Chem
except ImportError:
    print("ERROR: RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Ranking criteria
# ---------------------------------------------------------------------------

RANK_CRITERIA = {
    "CNNscore":          "higher",   # CNN pose quality,       higher = better
    "CNNaffinity":       "higher",   # CNN binding affinity,   higher = better
    "minimizedAffinity": "lower",    # Vina-like affinity,     lower  = better (more negative)
}

# All supported ligand extensions
LIGAND_EXTENSIONS = (".sdf", ".mol2", ".pdb")


# ---------------------------------------------------------------------------
# Protein readers
# ---------------------------------------------------------------------------

def read_protein_pdb(pdb_file: Path) -> List[str]:
    """Read ATOM lines from a PDB protein file (skip existing HETATM ligands).

    TER records are preserved so multi-chain proteins keep their chain breaks —
    LigandMPNN's parse_PDB relies on TER to delimit chains.
    """
    protein_lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith(("ATOM", "TER")):
                protein_lines.append(line)
            elif line.startswith(("HEADER", "TITLE", "REMARK", "CRYST1")):
                protein_lines.append(line)
            # Skip HETATM, MODEL, ENDMDL — replaced by docked pose
    return protein_lines


def read_protein_mol2(mol2_file: Path) -> List[str]:
    """
    Convert a MOL2 protein file to PDB-style ATOM lines via RDKit.
    Falls back to a raw-atom parser if RDKit cannot load the molecule
    (common for large proteins).
    """
    mol = Chem.MolFromMol2File(str(mol2_file), removeHs=False, sanitize=False)
    if mol is not None:
        # RDKit succeeded — convert via PDB block
        pdb_block = Chem.MolToPDBBlock(mol)
        if pdb_block:
            lines = []
            for line in pdb_block.splitlines(keepends=True):
                if line.startswith(("ATOM", "HETATM", "HEADER", "REMARK", "CRYST1")):
                    lines.append(line)
            print(f"  (MOL2 protein read via RDKit: {len(lines)} atom lines)")
            return lines

    # Fallback: parse MOL2 @<TRIPOS>ATOM section manually
    print("  (MOL2 protein: RDKit failed — using raw MOL2 parser)")
    return _mol2_atom_section_to_pdb(mol2_file, record="ATOM")


def _mol2_atom_section_to_pdb(mol2_file: Path, record: str = "ATOM") -> List[str]:
    """
    Minimal MOL2 → PDB converter that reads the @<TRIPOS>ATOM block.
    record: "ATOM" for protein, "HETATM" for ligand.
    """
    lines = []
    in_atom = False
    with open(mol2_file) as f:
        for raw in f:
            stripped = raw.strip()
            if stripped == "@<TRIPOS>ATOM":
                in_atom = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_atom:
                break  # left ATOM section
            if not in_atom or not stripped:
                continue

            parts = stripped.split()
            # MOL2 ATOM line: idx name x y z type [subst_id subst_name charge]
            if len(parts) < 6:
                continue
            try:
                atom_idx = int(parts[0])
                atom_name = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                atom_type = parts[5]          # e.g. C.3, N.am, O.2
                element = atom_type.split(".")[0].capitalize()
                res_name = parts[7] if len(parts) > 7 else "UNK"
                # subst_name can be "ALA1" or "1"  → strip digits for res_name
                res_name = ''.join(c for c in res_name if c.isalpha())[:3] or "UNK"
                res_num  = int(''.join(c for c in (parts[6] if len(parts) > 6 else "1")
                                       if c.isdigit()) or 1)
            except (ValueError, IndexError):
                continue

            pdb_line = (
                f"{record:<6}{atom_idx:5d}  {atom_name:<3s} {res_name:>3s}  "
                f"    {res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
            lines.append(pdb_line)

    return lines


def read_protein(protein_file: Path) -> List[str]:
    """Dispatch to the correct reader based on file extension."""
    ext = protein_file.suffix.lower()
    if ext == ".pdb":
        return read_protein_pdb(protein_file)
    elif ext == ".mol2":
        return read_protein_mol2(protein_file)
    else:
        print(f"ERROR: Unsupported protein format '{ext}'. Use .pdb or .mol2")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Ligand readers
# ---------------------------------------------------------------------------

def load_ligand_supplier(ligand_file: Path):
    """
    Return a list of RDKit molecules from an SDF, MOL2, or PDB file.

    SDF files may contain multiple poses; MOL2 files from GNINA also can
    (each pose is a separate @<TRIPOS>MOLECULE block).  Multi-model PDB files
    (MODEL/ENDMDL records) are split into individual pose molecules.
    Returns a list so callers can iterate over poses uniformly.
    """
    ext = ligand_file.suffix.lower()

    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
        if suppl is None:
            return []
        return list(suppl)

    elif ext == ".mol2":
        return _load_mol2_poses(ligand_file)

    elif ext == ".pdb":
        return _load_pdb_poses(ligand_file)

    else:
        print(f"ERROR: Unsupported ligand format '{ext}'. Use .sdf, .mol2, or .pdb")
        sys.exit(1)


def _load_pdb_poses(pdb_file: Path) -> List:
    """
    Load ligand poses from a PDB file.

    Handles two common layouts produced by docking programs:
      1. Multi-model PDB  — poses separated by MODEL / ENDMDL records.
      2. Single-model PDB — the whole file is one pose (HETATM / ATOM lines).

    Score properties embedded as REMARK lines are parsed for each model:
        REMARK CNNscore 0.9234
        REMARK CNNaffinity -8.12
        REMARK minimizedAffinity -7.45

    Only HETATM and ATOM lines are kept per pose so that protein context
    accidentally present in the file does not pollute the ligand molecule.
    Hydrogens are preserved (removeHs=False).
    """
    raw_blocks = _split_pdb_models(pdb_file)
    mols = []
    for block in raw_blocks:
        # Parse optional GNINA score REMARKs embedded in the block
        scores = {}
        for prop in RANK_CRITERIA:
            for rline in block.splitlines():
                stripped = rline.strip()
                # Support both "REMARK CNNscore 0.9" and "# CNNscore 0.9" styles
                if stripped.startswith(f"REMARK {prop}") or stripped.startswith(f"# {prop}"):
                    try:
                        scores[prop] = float(stripped.split()[2])
                    except (IndexError, ValueError):
                        pass

        mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        if mol is None:
            continue
        for prop, val in scores.items():
            mol.SetDoubleProp(prop, val)
        mols.append(mol)

    if not mols:
        print(f"  WARNING: RDKit could not parse any molecule from {pdb_file.name}. "
              "Check that the file contains valid HETATM/ATOM records.")
    return mols


def _split_pdb_models(pdb_file: Path) -> List[str]:
    """
    Split a PDB file into per-model blocks.

    • If MODEL/ENDMDL records are present each model becomes one block, keeping
      only ATOM, HETATM, and REMARK lines (so protein backbone accidentally
      included does not clutter the ligand mol).
    • If no MODEL records exist the whole file is treated as a single pose;
      only HETATM lines are kept (most docking outputs store ligands as HETATM).
      Falls back to including ATOM lines as well when no HETATM are found.
    """
    with open(pdb_file) as f:
        raw_lines = f.readlines()

    # Check whether the file uses MODEL/ENDMDL splitting
    model_indices = [i for i, l in enumerate(raw_lines) if l.startswith("MODEL")]

    if model_indices:
        blocks = []
        current: List[str] = []
        in_model = False
        for line in raw_lines:
            if line.startswith("MODEL"):
                in_model = True
                current = []
            elif line.startswith("ENDMDL"):
                in_model = False
                # Keep only chemically relevant lines for RDKit
                kept = [l for l in current
                        if l.startswith(("ATOM", "HETATM", "REMARK", "CONECT"))]
                if kept:
                    blocks.append("".join(kept) + "END\n")
                current = []
            elif in_model:
                current.append(line)
        # Handle file that has MODEL but no final ENDMDL
        if current:
            kept = [l for l in current
                    if l.startswith(("ATOM", "HETATM", "REMARK", "CONECT"))]
            if kept:
                blocks.append("".join(kept) + "END\n")
        return blocks

    # Single-model file — prefer HETATM; fall back to ATOM if nothing found
    hetatm = [l for l in raw_lines if l.startswith("HETATM")]
    remark = [l for l in raw_lines if l.startswith("REMARK")]
    conect = [l for l in raw_lines if l.startswith("CONECT")]

    if hetatm:
        block_lines = remark + hetatm + conect + ["END\n"]
    else:
        # Ligand stored as ATOM records (less common but valid)
        atom = [l for l in raw_lines if l.startswith("ATOM")]
        block_lines = remark + atom + conect + ["END\n"]

    return ["".join(block_lines)] if block_lines else []


def _load_mol2_poses(mol2_file: Path) -> List:
    """
    Split a multi-pose MOL2 file into individual molecules.
    Each @<TRIPOS>MOLECULE block is treated as one pose.
    Score properties (CNNscore, CNNaffinity, minimizedAffinity) are parsed
    from GNINA comment lines inside each block:
        # CNNscore 0.9234
        # CNNaffinity -8.12
        # minimizedAffinity -7.45
    """
    blocks = []
    current = []
    with open(mol2_file) as f:
        for line in f:
            if line.strip() == "@<TRIPOS>MOLECULE" and current:
                blocks.append("".join(current))
                current = []
            current.append(line)
    if current:
        blocks.append("".join(current))

    mols = []
    for block in blocks:
        # Parse GNINA score comments from the block
        scores = {}
        for prop in RANK_CRITERIA:
            for comment_line in block.splitlines():
                stripped = comment_line.strip()
                if stripped.startswith(f"# {prop}"):
                    try:
                        scores[prop] = float(stripped.split()[2])
                    except (IndexError, ValueError):
                        pass

        mol = Chem.MolFromMol2Block(block, removeHs=False, sanitize=False)
        if mol is None:
            continue
        for prop, val in scores.items():
            mol.SetDoubleProp(prop, val)
        mols.append(mol)

    return mols


# ---------------------------------------------------------------------------
# Ligand → PDB HETATM lines
# ---------------------------------------------------------------------------

def mol_to_hetatm_lines(mol, chain_id: str = "L",
                         res_name: str = "LIG", res_num: int = 1) -> List[str]:
    """Convert an RDKit molecule to PDB HETATM lines."""
    pdb_lines = []
    conf = mol.GetConformer()
    for atom_idx, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        element = atom.GetSymbol()
        line = (
            f"HETATM{atom_idx:5d}  {element:<3s} {res_name} {chain_id}{res_num:4d}    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00          {element:>2s}\n"
        )
        pdb_lines.append(line)
    return pdb_lines


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def get_score(mol, prop: str) -> Optional[float]:
    """Return float score for a property, or None if absent."""
    try:
        return float(mol.GetProp(prop))
    except KeyError:
        return None


def select_best_pose(mols: List, rank_by: str) -> Tuple:
    """Return (best_mol, best_pose_idx, best_score)."""
    direction = RANK_CRITERIA[rank_by]
    best_mol, best_score, best_idx = None, None, None

    for pose_idx, mol in enumerate(mols):
        if mol is None:
            continue
        score = get_score(mol, rank_by)
        if score is None:
            if best_mol is None:
                best_mol, best_score, best_idx = mol, score, pose_idx
            continue

        if best_score is None:
            best_mol, best_score, best_idx = mol, score, pose_idx
        elif direction == "higher" and score > best_score:
            best_mol, best_score, best_idx = mol, score, pose_idx
        elif direction == "lower"  and score < best_score:
            best_mol, best_score, best_idx = mol, score, pose_idx

    return best_mol, best_idx, best_score


# ---------------------------------------------------------------------------
# Score table printer
# ---------------------------------------------------------------------------

def _print_score_table(mols: List, best_idx: Optional[int] = None):
    print(f"  {'Pose':>4}  {'CNNscore':>10}  {'CNNaffinity':>12}  {'minimizedAffinity':>18}  {'':>4}")
    print("  " + "-" * 58)
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        cnn  = get_score(mol, "CNNscore")
        caff = get_score(mol, "CNNaffinity")
        maff = get_score(mol, "minimizedAffinity")
        sel  = "★" if idx == best_idx else ""
        cnn_s  = f"{cnn:>10.4f}"   if cnn  is not None else "       N/A"
        caff_s = f"{caff:>+12.4f}" if caff is not None else "         N/A"
        maff_s = f"{maff:>+18.4f}" if maff is not None else "               N/A"
        print(f"  {idx+1:>4}  {cnn_s}  {caff_s}  {maff_s}  {sel}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_ligand_file(
    ligand_file: Path,
    protein_lines: List[str],
    output_dir: Path,
    ligand_name: str = None,
    best_only: bool = True,
    rank_by: str = "CNNscore",
) -> int:
    """
    Process a single SDF, MOL2, or PDB ligand file and write complex PDB file(s).
    Returns the number of complexes written.
    """
    if ligand_name is None:
        ligand_name = ligand_file.stem.replace("_docked", "")

    direction = RANK_CRITERIA.get(rank_by, "higher")

    mols = load_ligand_supplier(ligand_file)
    valid_mols = [m for m in mols if m is not None]

    if not valid_mols:
        print(f"  ERROR: No valid poses found in {ligand_file.name}")
        return 0

    if best_only:
        best_mol, best_idx, best_score = select_best_pose(valid_mols, rank_by)
        if best_mol is None:
            print(f"  ERROR: Could not select best pose from {ligand_file.name}")
            return 0

        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        print(f"  Selected pose {best_idx + 1}  |  {rank_by} = {score_str}"
              f"  ({'higher is better' if direction == 'higher' else 'lower is better'})")
        _print_score_table(valid_mols, best_idx)

        output_file = output_dir / f"{ligand_name}_best_{rank_by}.pdb"
        with open(output_file, "w") as f:
            f.writelines(protein_lines)
            f.writelines(mol_to_hetatm_lines(best_mol))
            f.write("END\n")
        print(f"  → Written: {output_file.name}")
        return 1

    else:
        _print_score_table(valid_mols)
        count = 0
        for pose_idx, mol in enumerate(valid_mols):
            output_file = output_dir / f"{ligand_name}_pose_{pose_idx + 1}.pdb"
            with open(output_file, "w") as f:
                f.writelines(protein_lines)
                f.writelines(mol_to_hetatm_lines(mol))
                f.write("END\n")
            count += 1
        print(f"  → Written {count} pose(s)")
        return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combine docked SDF/MOL2/PDB poses with PDB/MOL2 protein → complex PDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  Protein : .pdb  .mol2
  Ligand  : .sdf  .mol2  .pdb  (multi-pose files supported for all three)

  PDB ligand notes:
    • Multi-model files (MODEL/ENDMDL records) are split into one pose per model.
    • Single-model files are treated as one pose (HETATM preferred, falls back to ATOM).
    • Score REMARKs are parsed from lines like:  REMARK CNNscore 0.9234

Ranking criteria (--rank_by):
  CNNscore           CNN pose quality score       (higher = better)
  CNNaffinity        CNN predicted affinity        (higher = better)
  minimizedAffinity  Vina-like affinity (kcal/mol) (lower  = better)

Examples:
  # SDF ligands + PDB protein, best pose by CNNscore (default)
  python combine_docked_protein.py -p protein.pdb -s gnina_results -o complexes --best_only

  # PDB ligands + PDB protein, best pose by CNNscore
  python combine_docked_protein.py -p protein.pdb -s docked_pdb -o complexes \\
      --pattern "*.pdb" --best_only

  # Single PDB ligand + PDB protein, all poses
  python combine_docked_protein.py -p protein.pdb -f ligand_docked.pdb -o complexes

  # MOL2 ligands + PDB protein, best by minimizedAffinity
  python combine_docked_protein.py -p protein.pdb -s docked_mol2 -o complexes \\
      --pattern "*.mol2" --best_only --rank_by minimizedAffinity

  # SDF ligands + MOL2 protein, all poses
  python combine_docked_protein.py -p protein.mol2 -s gnina_results -o complexes

  # Single MOL2 ligand + MOL2 protein, best pose
  python combine_docked_protein.py -p protein.mol2 -f ligand_docked.mol2 -o complexes --best_only
        """
    )

    parser.add_argument("-p", "--protein",    required=True, type=Path,
                        help="Protein structure file (.pdb or .mol2)")
    parser.add_argument("-s", "--sdf_dir",    type=Path,
                        help="Directory containing docked ligand files (SDF, MOL2, or PDB)")
    parser.add_argument("-f", "--sdf_file",   type=Path,
                        help="Single ligand file to process (.sdf, .mol2, or .pdb)")
    parser.add_argument("-o", "--output_dir", required=True, type=Path,
                        help="Output directory for complex PDB files")
    parser.add_argument("--pattern",  default=None,
                        help="Optional glob pattern to filter ligand files in sdf_dir "
                             "(e.g. '*.sdf', '*.mol2', '*.pdb', 'lig_*.pdb'). "
                             "If omitted, all .sdf, .mol2, and .pdb files are used.")
    parser.add_argument("--best_only", action="store_true",
                        help="Save only the best scoring pose per ligand")
    parser.add_argument("--rank_by",
                        choices=list(RANK_CRITERIA.keys()),
                        default="CNNscore",
                        help="Score used to select the best pose (default: CNNscore)")

    args = parser.parse_args()

    # --- Validate --------------------------------------------------------
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

    # --- Header ----------------------------------------------------------
    print("=" * 62)
    print("Combine Docked Ligand + Protein → Complex PDB")
    print("=" * 62)
    print(f"Protein    : {args.protein}  [{args.protein.suffix.lower()}]")
    print(f"Output dir : {args.output_dir}")
    if args.best_only:
        direction = RANK_CRITERIA[args.rank_by]
        print(f"Mode       : best pose only  |  ranked by: {args.rank_by} ({direction} is better)")
    else:
        print("Mode       : all poses")
    print("=" * 62)

    # --- Read protein ----------------------------------------------------
    print("\n[1/2] Reading protein structure…")
    protein_lines = read_protein(args.protein)
    print(f"  Read {len(protein_lines)} lines from {args.protein.name}")

    # --- Collect ligand files --------------------------------------------
    if args.sdf_file:
        if not args.sdf_file.exists():
            print(f"ERROR: Ligand file not found: {args.sdf_file}")
            sys.exit(1)
        ligand_files = [args.sdf_file]
    else:
        if args.pattern:
            # User supplied an explicit pattern
            ligand_files = sorted(args.sdf_dir.glob(args.pattern))
            if not ligand_files:
                print(f"ERROR: No files matching '{args.pattern}' in {args.sdf_dir}")
                sys.exit(1)
        else:
            # Auto-discover all SDF, MOL2, and PDB files
            ligand_files = sorted(
                f for f in args.sdf_dir.iterdir()
                if f.is_file() and f.suffix.lower() in LIGAND_EXTENSIONS
            )
            if not ligand_files:
                print(f"ERROR: No .sdf, .mol2, or .pdb files found in {args.sdf_dir}")
                print("       Use --pattern to specify a custom glob (e.g. '*.pdb')")
                sys.exit(1)
            print(f"  Auto-detected {len(ligand_files)} ligand file(s) "
                  f"({', '.join(sorted({f.suffix.lower() for f in ligand_files}))})")

    # Show detected format
    exts = {f.suffix.lower() for f in ligand_files}
    print(f"\n[2/2] Processing {len(ligand_files)} ligand file(s)  "
          f"[format(s): {', '.join(exts)}]\n")

    # --- Process ---------------------------------------------------------
    total = 0
    for i, lf in enumerate(ligand_files, 1):
        print(f"[{i}/{len(ligand_files)}] {lf.name}")
        total += process_ligand_file(
            lf, protein_lines, args.output_dir,
            best_only=args.best_only, rank_by=args.rank_by
        )
        print()

    # --- Summary ---------------------------------------------------------
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"Ligand files processed : {len(ligand_files)}")
    print(f"Complexes created      : {total}")
    print(f"Output directory       : {args.output_dir}")

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