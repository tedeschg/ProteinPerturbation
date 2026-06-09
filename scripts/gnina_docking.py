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
import logging
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
# Logging setup — writes to stdout AND to a logfile simultaneously
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure a logger that mirrors all output to both console and logfile."""
    log_file = output_dir / "docking.log"
    logger = logging.getLogger("gnina")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_smiles(smiles_file: Path) -> List[Tuple[str, str, str]]:
    """Read SMILES file and return list of (SMILES, ID, name) tuples."""
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


def smiles_to_sdf(smiles: str, output_file: Path, lig_id: str = "ligand",
                  log: logging.Logger = None) -> bool:
    """Convert SMILES to 3D SDF using RDKit. Returns True on success."""
    def msg(m): log.info(m) if log else print(m)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            msg(f"  ERROR: Invalid SMILES: {smiles}")
            return False
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            msg("  WARNING: Retrying 3D embedding with random coords…")
            result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if result != 0:
                msg(f"  ERROR: Failed to generate 3D coordinates for {smiles}")
                return False
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        writer = Chem.SDWriter(str(output_file))
        mol.SetProp("_Name", lig_id)
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        msg(f"  ERROR converting SMILES to SDF: {e}")
        return False


def mol2_or_pdb_to_sdf(input_file: Path, output_file: Path,
                        log: logging.Logger = None) -> bool:
    """Convert a single .mol2 or .pdb ligand file to SDF using RDKit."""
    def msg(m): log.info(m) if log else print(m)
    suffix = input_file.suffix.lower()
    try:
        if suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(input_file), removeHs=False)
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(input_file), removeHs=False)
        elif suffix in (".sdf", ".mol"):
            shutil.copy(input_file, output_file)
            return True
        else:
            msg(f"  ERROR: Unsupported format: {suffix}")
            return False
        if mol is None:
            msg(f"  ERROR: RDKit could not read {input_file.name}")
            return False
        writer = Chem.SDWriter(str(output_file))
        mol.SetProp("_Name", input_file.stem)
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        msg(f"  ERROR converting {input_file.name} to SDF: {e}")
        return False


def extract_ligand_from_complex(complex_pdb: Path, output_sdf: Path,
                                 log: logging.Logger = None) -> bool:
    """Extract HETATM ligand from a complex PDB and save as SDF."""
    def msg(m): log.info(m) if log else print(m)
    try:
        hetero_atoms = []
        with open(complex_pdb) as f:
            for line in f:
                if line.startswith("HETATM"):
                    hetero_atoms.append(line)
        if not hetero_atoms:
            msg(f"  ERROR: No HETATM found in {complex_pdb}")
            return False
        temp_pdb = output_sdf.with_suffix(".temp.pdb")
        with open(temp_pdb, "w") as f:
            f.writelines(hetero_atoms)
            f.write("END\n")
        mol = Chem.MolFromPDBFile(str(temp_pdb), removeHs=False)
        if mol is None:
            msg(f"  ERROR: Could not read ligand from {temp_pdb}")
            temp_pdb.unlink()
            return False
        writer = Chem.SDWriter(str(output_sdf))
        writer.write(mol)
        writer.close()
        temp_pdb.unlink()
        return True
    except Exception as e:
        msg(f"  ERROR extracting ligand: {e}")
        return False


def run_gnina(protein_pdb: Path, ligand_sdf: Path, reference_sdf: Path,
              output_sdf: Path, autobox_add: float = 8.0,
              exhaustiveness: int = 8, num_modes: int = 9,
              cpu: int = 1, log: logging.Logger = None) -> bool:
    """Run GNINA docking with autobox centred on the reference ligand."""
    def msg(m): log.info(m) if log else print(m)
    cmd = [
        "gnina",
        "-r", str(protein_pdb),
        "-l", str(ligand_sdf),
        "--autobox_ligand", str(reference_sdf),
        "--autobox_add",    str(autobox_add),
        "-o",               str(output_sdf),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes",      str(num_modes),
        "--cpu",            str(cpu),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            msg(f"  ERROR: GNINA failed (code {result.returncode})")
            msg(f"  STDERR: {result.stderr}")
            return False
        if not output_sdf.exists():
            msg(f"  ERROR: Output file not created: {output_sdf}")
            return False
        return True
    except subprocess.TimeoutExpired:
        msg("  ERROR: GNINA timed out after 600 s")
        return False
    except FileNotFoundError:
        msg("  ERROR: 'gnina' not found. Please install GNINA.")
        return False
    except Exception as e:
        msg(f"  ERROR running GNINA: {e}")
        return False


def parse_gnina_results(output_sdf: Path) -> List[dict]:
    """Parse a GNINA output SDF and extract per-pose scores."""
    poses = []
    if not output_sdf.exists():
        return poses
    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False, sanitize=False)
    for pose_idx, mol in enumerate(supplier, start=1):
        if mol is None:
            continue
        info = {"pose": pose_idx}
        for prop in ("minimizedAffinity", "CNNscore", "CNNaffinity"):
            try:
                info[prop] = float(mol.GetProp(prop))
            except KeyError:
                info[prop] = None
        poses.append(info)
    return poses


def check_score_consistency(best_pose: dict, threshold: float = 3.0) -> Tuple[bool, float]:
    """
    Check if CNNaffinity and minimizedAffinity are consistent for the best pose.

    GNINA v1.1 writes CNNaffinity with a POSITIVE sign in the SDF
    (e.g. +6.17 means -6.17 kcal/mol physically). We negate it before
    comparing with minimizedAffinity, which is already negative.

    Returns (is_inconsistent, gap) where:
      gap = |-CNNaffinity - minimizedAffinity|
      is_inconsistent = True if gap > threshold  ->  redo docking
    """
    cnn_aff = best_pose.get("CNNaffinity")
    min_aff = best_pose.get("minimizedAffinity")
    if cnn_aff is None or min_aff is None:
        return False, 0.0
    # Negate CNNaffinity to convert to physical kcal/mol (GNINA v1.1 convention)
    gap = abs(-cnn_aff - min_aff)
    return gap > threshold, gap


def print_pose_table(poses: List[dict], best_idx: int,
                     log: logging.Logger = None) -> None:
    """Print a formatted table of all poses, marking the best one."""
    def msg(m): log.info(m) if log else print(m)
    if not poses:
        msg("  No poses parsed from output SDF.")
        return
    msg(f"  {'Pose':>4}  {'minimizedAffinity':>18}  {'CNNscore':>10}  {'CNNaffinity':>12}  {'':>8}")
    msg("  " + "-" * 62)
    for p in poses:
        aff  = f"{p['minimizedAffinity']:>+.3f}" if p["minimizedAffinity"] is not None else "    N/A"
        cnn  = f"{p['CNNscore']:>10.4f}"          if p["CNNscore"]         is not None else "       N/A"
        caff = f"{p['CNNaffinity']:>+.3f}"         if p["CNNaffinity"]      is not None else "    N/A"
        flag = "  ★ best" if p["pose"] == best_idx else ""
        msg(f"  {p['pose']:>4}  {aff:>18}  {cnn:>10}  {caff:>12}{flag}")


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

    parser.add_argument("-p", "--protein",           required=True, type=Path)
    parser.add_argument("-c", "--reference_complex", required=True, type=Path)
    parser.add_argument("-o", "--output_dir",        required=True, type=Path)

    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("-s", "--smiles",      type=Path)
    inp.add_argument("-l", "--ligand",      type=Path)
    inp.add_argument("-L", "--ligands_dir", type=Path)

    parser.add_argument("--autobox_add",         type=float, default=8.0)
    parser.add_argument("--exhaustiveness",       type=int,   default=8)
    parser.add_argument("--exhaustiveness_redo",  type=int,   default=32,
                        help="exhaustiveness used when redoing inconsistent dockings (default: 32)")
    parser.add_argument("--num_modes",            type=int,   default=9)
    parser.add_argument("--cpu",                  type=int,   default=1,
                        help="Number of CPU cores to use (default: 1)")
    parser.add_argument("--consistency_threshold", type=float, default=3.0,
                        help="|CNNaffinity - minimizedAffinity| threshold to trigger redo (default: 3.0 kcal/mol)")
    parser.add_argument("--keep_temp",            action="store_true")

    args = parser.parse_args()

    for label, path in [("Protein", args.protein),
                         ("Reference complex", args.reference_complex)]:
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    log = setup_logging(args.output_dir)

    log.info("=" * 60)
    log.info("GNINA Automated Docking")
    log.info("=" * 60)
    log.info(f"Protein:               {args.protein}")
    log.info(f"Reference complex:     {args.reference_complex}")
    log.info(f"Output directory:      {args.output_dir}")
    log.info(f"Autobox padding:       {args.autobox_add} Å")
    log.info(f"Exhaustiveness:        {args.exhaustiveness}")
    log.info(f"Exhaustiveness (redo): {args.exhaustiveness_redo}")
    log.info(f"Poses per ligand:      {args.num_modes}")
    log.info(f"CPUs:                  {args.cpu}")
    log.info(f"Consistency threshold: {args.consistency_threshold} kcal/mol")
    log.info(f"Log file:              {args.output_dir / 'docking.log'}")
    log.info("=" * 60)

    # --- Step 1: extract reference ligand ---
    log.info("\n[1/3] Extracting reference ligand from complex…")
    reference_sdf = temp_dir / "reference_ligand.sdf"
    if not extract_ligand_from_complex(args.reference_complex, reference_sdf, log):
        log.error("ERROR: Failed to extract reference ligand")
        sys.exit(1)
    log.info(f"  Reference ligand saved: {reference_sdf}")

    # --- Step 2: build ligand list ---
    log.info("\n[2/3] Collecting ligands…")
    SUPPORTED = {".sdf", ".mol", ".mol2", ".pdb"}
    ligand_queue: List[Tuple[Path, str]] = []

    if args.smiles:
        if not args.smiles.exists():
            log.error(f"ERROR: SMILES file not found: {args.smiles}")
            sys.exit(1)
        entries = read_smiles(args.smiles)
        log.info(f"  Found {len(entries)} SMILES entries")
        for smiles, lig_id, name in entries:
            sdf = temp_dir / f"{lig_id}_input.sdf"
            log.info(f"  Converting {name} to 3D…")
            if smiles_to_sdf(smiles, sdf, lig_id, log):
                ligand_queue.append((sdf, name))
            else:
                log.warning(f"  FAILED — {name} skipped")

    elif args.ligand:
        if not args.ligand.exists():
            log.error(f"ERROR: Ligand file not found: {args.ligand}")
            sys.exit(1)
        suffix = args.ligand.suffix.lower()
        if suffix not in SUPPORTED:
            log.error(f"ERROR: Unsupported format {suffix}.")
            sys.exit(1)
        if suffix in (".sdf", ".mol"):
            ligand_queue.append((args.ligand, args.ligand.stem))
        else:
            sdf = temp_dir / f"{args.ligand.stem}.sdf"
            if mol2_or_pdb_to_sdf(args.ligand, sdf, log):
                ligand_queue.append((sdf, args.ligand.stem))
            else:
                sys.exit(1)

    elif args.ligands_dir:
        if not args.ligands_dir.is_dir():
            log.error(f"ERROR: Folder not found: {args.ligands_dir}")
            sys.exit(1)
        files = sorted(f for f in args.ligands_dir.iterdir()
                       if f.suffix.lower() in SUPPORTED)
        if not files:
            log.error(f"ERROR: No supported files found in {args.ligands_dir}")
            sys.exit(1)
        log.info(f"  Found {len(files)} files")
        for f in files:
            if f.suffix.lower() in (".sdf", ".mol"):
                ligand_queue.append((f, f.stem))
            else:
                sdf = temp_dir / f"{f.stem}.sdf"
                if mol2_or_pdb_to_sdf(f, sdf, log):
                    ligand_queue.append((sdf, f.stem))
                else:
                    log.warning(f"  FAILED conversion — {f.name} skipped")

    log.info(f"  Ligands ready for docking: {len(ligand_queue)}")

    # --- Step 3: dock ---
    log.info("\n[3/3] Running GNINA docking…")
    results      = []
    all_best_poses = []
    redo_list    = []   # names of ligands that triggered a redo

    for i, (ligand_path, name) in enumerate(ligand_queue, 1):
        log.info(f"\n{'─' * 60}")
        log.info(f"[{i}/{len(ligand_queue)}] {name}")
        log.info(f"{'─' * 60}")
        output_sdf = args.output_dir / f"{name}_docked.sdf"

        # ── First docking attempt ─────────────────────────────────────────
        success = run_gnina(args.protein, ligand_path, reference_sdf, output_sdf,
                            args.autobox_add, args.exhaustiveness, args.num_modes,
                            args.cpu, log)

        if not success:
            log.warning("  ✗ Docking failed.")
            results.append((name, "FAILED", "GNINA error", None))
            continue

        poses = parse_gnina_results(output_sdf)
        if not poses:
            log.warning("  ✗ No poses parsed.")
            results.append((name, "FAILED", "no poses", None))
            continue

        best = max(poses, key=lambda p: p["CNNscore"] if p["CNNscore"] is not None else -999)
        print_pose_table(poses, best["pose"], log)

        # ── Consistency check ─────────────────────────────────────────────
        inconsistent, gap = check_score_consistency(best, args.consistency_threshold)

        if inconsistent:
            log.warning("")
            log.warning(f"  !! INCONSISTENCY DETECTED for {name} !!")
            log.warning(f"     CNNaffinity (raw SDF)    = {best['CNNaffinity']:+.3f}  (GNINA v1.1: stored positive)")
            log.warning(f"     CNNaffinity (physical)   = {-best['CNNaffinity']:+.3f} kcal/mol")
            log.warning(f"     minimizedAffinity        = {best['minimizedAffinity']:+.3f} kcal/mol")
            log.warning(f"     |gap|                    = {gap:.3f} kcal/mol  (threshold: {args.consistency_threshold})")
            log.warning(f"     -> Deleting result and redoing with exhaustiveness={args.exhaustiveness_redo}...")

            output_sdf.unlink(missing_ok=True)
            redo_list.append(name)

            # ── Redo with higher exhaustiveness ───────────────────────────
            success_redo = run_gnina(
                args.protein, ligand_path, reference_sdf, output_sdf,
                args.autobox_add, args.exhaustiveness_redo, args.num_modes,
                args.cpu, log
            )

            if not success_redo:
                log.warning("  ✗ Redo also failed.")
                results.append((name, "FAILED", "redo failed", None))
                continue

            poses = parse_gnina_results(output_sdf)
            if not poses:
                log.warning("  ✗ No poses after redo.")
                results.append((name, "FAILED", "no poses after redo", None))
                continue

            best = max(poses, key=lambda p: p["CNNscore"] if p["CNNscore"] is not None else -999)
            log.info("  Results after redo:")
            print_pose_table(poses, best["pose"], log)

            _, gap_redo = check_score_consistency(best, args.consistency_threshold)
            if gap_redo > args.consistency_threshold:
                log.warning(f"  !! Still inconsistent after redo (gap={gap_redo:.3f}) — keeping result anyway.")
            else:
                log.info(f"  ✓ Consistency restored after redo (gap={gap_redo:.3f})")
        else:
            log.info(f"\n  ✓ Scores consistent (|CNNaffinity - minimizedAffinity| = {gap:.3f} kcal/mol)")

        log.info(f"\n  ★ Best pose: {best['pose']}"
                 f"  |  CNNscore={best['CNNscore']:.4f}"
                 f"  |  CNNaffinity={best['CNNaffinity']:+.3f}"
                 f"  |  minimizedAffinity={best['minimizedAffinity']:+.3f}")

        results.append((name, "SUCCESS", str(output_sdf), best))
        all_best_poses.append({
            "name":              name,
            "best_pose":         best["pose"],
            "CNNscore":          best["CNNscore"],
            "CNNaffinity":       best["CNNaffinity"],
            "minimizedAffinity": best["minimizedAffinity"],
            "redone":            name in redo_list,
            "output":            str(output_sdf),
        })

    # --- Final ranked summary ---
    log.info("")
    log.info("=" * 60)
    log.info("DOCKING SUMMARY — ranked by CNNscore")
    log.info("=" * 60)

    if all_best_poses:
        ranked = sorted(all_best_poses,
                        key=lambda x: x["CNNscore"] if x["CNNscore"] is not None else -999,
                        reverse=True)
        log.info(f"  {'Rank':>4}  {'Ligand':<35}  {'CNNscore':>10}  {'CNNaff':>8}  {'minAff':>8}  {'Redo':>5}")
        log.info("  " + "-" * 80)
        for rank, r in enumerate(ranked, 1):
            cnn  = f"{r['CNNscore']:>10.4f}"        if r["CNNscore"]        is not None else "       N/A"
            caff = f"{r['CNNaffinity']:>+8.3f}"      if r["CNNaffinity"]     is not None else "     N/A"
            maff = f"{r['minimizedAffinity']:>+8.3f}" if r["minimizedAffinity"] is not None else "     N/A"
            redo = "  YES" if r["redone"] else "   no"
            log.info(f"  {rank:>4}  {r['name']:<35}  {cnn}  {caff}  {maff}  {redo}")

    if redo_list:
        log.info("")
        log.info(f"  Ligands that triggered a redo ({len(redo_list)}):")
        for name in redo_list:
            log.info(f"    ⚠  {name}")

    failed = [r for r in results if r[1] == "FAILED"]
    log.info(f"\n  Total: {len(results)}  |  Success: {len(all_best_poses)}"
             f"  |  Redone: {len(redo_list)}  |  Failed: {len(failed)}")
    if failed:
        log.info("  Failed ligands:")
        for name, _, reason, _ in failed:
            log.info(f"    ✗ {name}  ({reason})")

    # --- Save CSV ---
    summary_file = args.output_dir / "docking_summary.csv"
    with open(summary_file, "w") as f:
        f.write("rank,name,best_pose,CNNscore,CNNaffinity,minimizedAffinity,redone,output\n")
        for rank, r in enumerate(ranked if all_best_poses else [], 1):
            f.write(f"{rank},{r['name']},{r['best_pose']},"
                    f"{r['CNNscore']},{r['CNNaffinity']},{r['minimizedAffinity']},"
                    f"{r['redone']},{r['output']}\n")
        for name, status, out, _ in results:
            if status == "FAILED":
                f.write(f"N/A,{name},N/A,N/A,N/A,N/A,N/A,FAILED\n")
    log.info(f"\n  Summary CSV: {summary_file}")
    log.info(f"  Log file:    {args.output_dir / 'docking.log'}")

    # --- Cleanup ---
    if not args.keep_temp:
        shutil.rmtree(temp_dir)
        log.info("  Temporary files removed.")
    else:
        log.info(f"  Temporary files kept in: {temp_dir}")

    log.info("\nAll done!")


if __name__ == "__main__":
    main()