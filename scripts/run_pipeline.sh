#!/bin/bash
###############################################################################
# LigandMPNN Pipeline - Multi-PDB (docking‑box residue selection)
#
# USAGE
# -----
#   bash run_pipeline.sh file1.pdb file2.pdb [file3.pdb ...]
#
# PIPELINE STEPS
# --------------
#   1. Pose QC        – clash + geometry check          (pose_qc.py)
#   2. Residues Selection – docking‑box based residues   (residues_selection.py)
#                         Residues are selected ONCE from the reference complex.
#                         Each input PDB is then checked to verify that those
#                         residues still fall within the docking box; any that
#                         do not are reported as a NOTE (informational only –
#                         the residue set is NOT modified).
#   3. LigandMPNN     – sequence design in batch mode
#   4. Final Report   – summary table printed to terminal + written to TSV
#
# NOTES
# -----
#   - The residues selection step extracts all residues that fall within
#     the GNINA‑style docking box (sphere of radius `autobox_add` around
#     the ligand) using the REFERENCE complex only.
#   - Each input PDB is checked for residues that are out-of-box relative to
#     its own structure; these are logged as notes but do not alter the set
#     passed to LigandMPNN.
#   - Re-running the script on the same output_dir will overwrite results.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()      { echo "[$(date '+%H:%M:%S')] $*"; }
log_warn() { echo "[$(date '+%H:%M:%S')] [WARN] $*" >&2; }
log_err()  { echo "[$(date '+%H:%M:%S')] [ERROR] $*" >&2; }

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║       LigandMPNN Pipeline -- Multi-PDB        ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
}

# ---------------------------------------------------------------------------
# YAML parsing — delegated to scripts/parse_config.py (PyYAML-based)
# ---------------------------------------------------------------------------

read_yaml() {
    local key="$1"
    python "$SCRIPT_DIR/parse_config.py" "$CONFIG_FILE" "$key"
}

read_yaml_nested() {
    local section="$1"
    local key="$2"
    python "$SCRIPT_DIR/parse_config.py" "$CONFIG_FILE" "${section}.${key}"
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

if [ ! -f "$CONFIG_FILE" ]; then
    log_err "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

CONDA_PROFILE=$(read_yaml "conda_profile")
PROJECT_ROOT="$SCRIPT_DIR"
LMPNN_PATH=$(read_yaml "lmpnn_path")
ENV_DOCKNDESIGN=$(read_yaml "env_dockndesign")
ENV_LIGANDMPNN=$(read_yaml "env_ligandmpnn")
BASE_OUTPUT_DIR=$(read_yaml "output_dir")

# residues_selection
RESIDUE_OUTPUT_FILE=$(read_yaml_nested "residues_selection" "output_file")
REF_COMPLEX_RAW=$(read_yaml_nested "residues_selection" "reference_complex")
AUTOBOX_ADD=$(read_yaml_nested "residues_selection" "autobox_add")
OUT_FORMAT=$(read_yaml_nested "residues_selection" "output_format")
INCL_HETATM=$(read_yaml_nested "residues_selection" "include_hetatm_residues")

# Fallback for missing values
if [ -z "$AUTOBOX_ADD" ]; then
    log_warn "autobox_add not found in config, using default 8.0"
    AUTOBOX_ADD=8.0
fi
if [ -z "$OUT_FORMAT" ]; then
    OUT_FORMAT="space_separated"
fi
if [ -z "$INCL_HETATM" ]; then
    INCL_HETATM="false"
fi
if [ -z "$REF_COMPLEX_RAW" ]; then
    log_err "reference_complex not defined in config.yaml under residues_selection"
    exit 1
fi

# Resolve relative paths
if [[ ! "$REF_COMPLEX_RAW" = /* ]]; then
    REF_COMPLEX="$PROJECT_ROOT/$REF_COMPLEX_RAW"
else
    REF_COMPLEX="$REF_COMPLEX_RAW"
fi

if [ ! -f "$REF_COMPLEX" ]; then
    log_err "Reference complex file not found: $REF_COMPLEX"
    exit 1
fi

# pose_qc
QC_ENABLED=$(read_yaml_nested "pose_qc" "enabled")
QC_CLASH_DIST=$(read_yaml_nested "pose_qc" "clash_dist")
QC_BOND_TOL=$(read_yaml_nested "pose_qc" "bond_length_tol")
QC_STRICT=$(read_yaml_nested "pose_qc" "strict")
QC_OUT_JSON=$(read_yaml_nested "pose_qc" "out_json")
QC_ON_CLASH=$(read_yaml_nested "pose_qc" "on_clash")

# ligandmpnn
MODEL_TYPE=$(read_yaml_nested "ligandmpnn" "model_type")
CHECKPOINT=$(read_yaml_nested "ligandmpnn" "checkpoint")
TEMPERATURE=$(read_yaml_nested "ligandmpnn" "temperature")
SEED=$(read_yaml_nested "ligandmpnn" "seed")
NUM_BATCHES=$(read_yaml_nested "ligandmpnn" "number_of_batches")

# ---------------------------------------------------------------------------
# Input PDBs
# ---------------------------------------------------------------------------

print_banner

if [ $# -eq 0 ]; then
    log_err "No PDB files specified."
    echo "  Usage: $0 file1.pdb file2.pdb ..."
    exit 1
fi

PDB_FILES=("$@")
VALID_PDBS=()

for pdb in "${PDB_FILES[@]}"; do
    if [ -f "$pdb" ]; then
        VALID_PDBS+=("$pdb")
    else
        log_warn "File not found: $pdb -- skipping."
    fi
done

if [ ${#VALID_PDBS[@]} -eq 0 ]; then
    log_err "No valid PDB files found."
    exit 1
fi

log "Found ${#VALID_PDBS[@]} valid PDB(s)."
mkdir -p "$BASE_OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Tracking arrays for final report
# ---------------------------------------------------------------------------

declare -A PDB_QC_STATUS
declare -A PDB_QC_N_CLASHES
declare -A PDB_QC_N_GEOM
declare -A PDB_N_POSITIONS           # always equal to REF_N_POSITIONS for active PDBs
declare -A PDB_OOB_RESIDUES          # residues from reference that are out-of-box in this PDB
declare -A PDB_SEL_STATUS
declare -A SKIPPED_SET
SKIPPED_PDBS=()
ACTIVE_PDBS=()

mark_skipped() {
    local pdb="$1"
    if [ -z "${SKIPPED_SET[$pdb]+_}" ]; then
        SKIPPED_SET["$pdb"]=1
        SKIPPED_PDBS+=("$pdb")
    fi
}

# ---------------------------------------------------------------------------
# Activate dockndesign env (used for QC + selection)
# ---------------------------------------------------------------------------

# shellcheck disable=SC1090
source "$CONDA_PROFILE"
conda activate "$ENV_DOCKNDESIGN"

###############################################################################
# STEP 1: Pose QC
###############################################################################
echo ""
echo "----------------------------------------------"
log "[1/3] Pose QC"
echo "----------------------------------------------"

for pdb in "${VALID_PDBS[@]}"; do
    PDB_BASENAME=$(basename "$pdb" .pdb)
    PDB_DIR="$BASE_OUTPUT_DIR/$PDB_BASENAME"
    mkdir -p "$PDB_DIR"

    if [ "$QC_ENABLED" != "true" ]; then
        log "  [$PDB_BASENAME] QC disabled -- skipping."
        PDB_QC_STATUS["$pdb"]="skipped"
        PDB_QC_N_CLASHES["$pdb"]=0
        PDB_QC_N_GEOM["$pdb"]=0
        ACTIVE_PDBS+=("$pdb")
        continue
    fi

    QC_ARGS=(
        --pdb "$pdb"
        --clash_dist "$QC_CLASH_DIST"
        --bond_length_tol "$QC_BOND_TOL"
    )
    [ "$QC_STRICT"   = "true" ] && QC_ARGS+=(--strict)
    [ "$QC_OUT_JSON" = "true" ] && QC_ARGS+=(--out_json "$PDB_DIR/qc_report.json")

    log "  [$PDB_BASENAME] Running pose_qc.py..."

    set +e
    QC_OUTPUT=$(python "$PROJECT_ROOT/pose_qc.py" "${QC_ARGS[@]}" 2>&1)
    QC_EXIT=$?
    set -e

    N_CLASHES=$(echo "$QC_OUTPUT" | { grep -oP 'Clashes:\s+\K[0-9]+' || true; } | head -1)
    N_GEOM=$(echo "$QC_OUTPUT"    | { grep -oP 'Geometry issues:\s+\K[0-9]+' || true; } | head -1)
    N_CLASHES="${N_CLASHES:-0}"
    N_GEOM="${N_GEOM:-0}"
    PDB_QC_N_CLASHES["$pdb"]="$N_CLASHES"
    PDB_QC_N_GEOM["$pdb"]="$N_GEOM"

    if [ "$QC_EXIT" -eq 0 ]; then
        log "  [$PDB_BASENAME] QC PASS"
        PDB_QC_STATUS["$pdb"]="pass"
        ACTIVE_PDBS+=("$pdb")
    elif [ "$QC_EXIT" -eq 1 ]; then
        log_warn "[$PDB_BASENAME] QC FAIL (clashes=${N_CLASHES}, geom=${N_GEOM})"
        echo "$QC_OUTPUT" | { grep -E "\[FAIL\]|\[WARN\]" || true; } | sed 's/^/    /'
        case "$QC_ON_CLASH" in
            fail)
                log_err "on_clash=fail -- aborting pipeline."
                exit 1
                ;;
            skip)
                log_warn "[$PDB_BASENAME] Skipping this PDB (on_clash=skip)."
                PDB_QC_STATUS["$pdb"]="skipped"
                mark_skipped "$pdb"
                ;;
            warn|*)
                log_warn "[$PDB_BASENAME] Continuing despite clash (on_clash=warn)."
                PDB_QC_STATUS["$pdb"]="warn_clash"
                ACTIVE_PDBS+=("$pdb")
                ;;
        esac
    else
        log_warn "[$PDB_BASENAME] pose_qc.py crashed (exit=$QC_EXIT). Continuing without QC."
        echo "$QC_OUTPUT" | sed 's/^/    /'
        PDB_QC_STATUS["$pdb"]="qc_crashed"
        ACTIVE_PDBS+=("$pdb")
    fi
done

if [ ${#ACTIVE_PDBS[@]} -eq 0 ]; then
    log_err "All PDBs were skipped after QC. Nothing to process."
    exit 1
fi

log "  QC done: ${#ACTIVE_PDBS[@]} active, ${#SKIPPED_PDBS[@]} skipped."

###############################################################################
# STEP 2: Residue Selection (reference-based)
#
# 2a. Select residues ONCE from the reference complex.
# 2b. For each input PDB, check whether those residues are still within the
#     docking box. Residues that fall outside are reported as a NOTE but the
#     set passed to LigandMPNN is never altered.
###############################################################################
echo ""
echo "----------------------------------------------"
log "[2/3] Residue Selection (reference complex → fixed set)"
echo "----------------------------------------------"

# ---- 2a. Run selection on the reference complex ----------------------------

REF_RESIDUES_FILE="$BASE_OUTPUT_DIR/reference_selected_residues.txt"

REF_SEL_ARGS=(
    --protein "$REF_COMPLEX"
    --reference_complex "$REF_COMPLEX"
    --autobox_add "$AUTOBOX_ADD"
    --output "$REF_RESIDUES_FILE"
    --format "$OUT_FORMAT"
)
[ "$INCL_HETATM" = "true" ] && REF_SEL_ARGS+=(--include_hetatm)

log "  Selecting residues from reference complex (radius=${AUTOBOX_ADD} Å)..."

set +e
REF_SEL_OUTPUT=$(python "$PROJECT_ROOT/residues_selection.py" "${REF_SEL_ARGS[@]}" 2>&1)
REF_SEL_EXIT=$?
set -e

echo "$REF_SEL_OUTPUT" | sed 's/^/    /'

if [ "$REF_SEL_EXIT" -ne 0 ] || [ ! -f "$REF_RESIDUES_FILE" ]; then
    log_err "residues_selection.py failed on the reference complex (exit=$REF_SEL_EXIT). Aborting."
    exit 1
fi

REFERENCE_RESIDUES=$(cat "$REF_RESIDUES_FILE")
REF_N_POS=$(echo "$REFERENCE_RESIDUES" | tr -s ' \t\n' '\n' | grep -c '[^[:space:]]' || true)
REF_N_POS="${REF_N_POS:-0}"

log "  Reference residues selected: $REF_N_POS positions."
log "  Residue set: $REFERENCE_RESIDUES"

# ---- 2b. Per-PDB out-of-box check -----------------------------------------
# residues_selection.py is called on each input PDB using the same reference
# to obtain the box centre, but with --check_only_residues to verify which of
# the reference residues are still within the box.
# If your residues_selection.py does not yet support --check_only_residues,
# the fallback block below calls it normally and compares the output set to
# the reference set to identify missing residues.

for pdb in "${ACTIVE_PDBS[@]}"; do
    PDB_BASENAME=$(basename "$pdb" .pdb)
    PDB_DIR="$BASE_OUTPUT_DIR/$PDB_BASENAME"
    PDB_CHECK_FILE="$PDB_DIR/residues_in_box.txt"
    OOB_NOTES_FILE="$PDB_DIR/out_of_box_residues.txt"

    log "  [$PDB_BASENAME] Checking reference residues against this structure..."

    # ------------------------------------------------------------------
    # Strategy: run residues_selection.py on the input PDB (same box
    # centre from the reference ligand) and compare the resulting set
    # to the reference set. Residues present in the reference set but
    # absent from the per-PDB set are considered out-of-box.
    # ------------------------------------------------------------------

    PDB_CHK_ARGS=(
        --protein "$pdb"
        --reference_complex "$REF_COMPLEX"
        --autobox_add "$AUTOBOX_ADD"
        --output "$PDB_CHECK_FILE"
        --format "$OUT_FORMAT"
    )
    [ "$INCL_HETATM" = "true" ] && PDB_CHK_ARGS+=(--include_hetatm)

    set +e
    CHK_OUTPUT=$(python "$PROJECT_ROOT/residues_selection.py" "${PDB_CHK_ARGS[@]}" 2>&1)
    CHK_EXIT=$?
    set -e

    if [ "$CHK_EXIT" -ne 0 ] || [ ! -f "$PDB_CHECK_FILE" ]; then
        log_warn "[$PDB_BASENAME] Could not run box-check (exit=$CHK_EXIT). Skipping out-of-box verification for this PDB."
        PDB_OOB_RESIDUES["$pdb"]="check_failed"
        PDB_N_POSITIONS["$pdb"]="$REF_N_POS"
        PDB_SEL_STATUS["$pdb"]="ok_no_check"
        continue
    fi

    PDB_IN_BOX=$(cat "$PDB_CHECK_FILE")

    # Identify reference residues that are absent from this PDB's in-box set.
    # Both sets use the same format (space/newline separated residue tokens).
    OOB_LIST=""
    for res in $REFERENCE_RESIDUES; do
        if ! echo "$PDB_IN_BOX" | tr -s ' \t\n' '\n' | grep -qxF "$res"; then
            OOB_LIST="${OOB_LIST:+$OOB_LIST }$res"
        fi
    done

    if [ -n "$OOB_LIST" ]; then
        log_warn "[$PDB_BASENAME] NOTE: the following residue(s) from the reference selection are outside the docking box in this structure: $OOB_LIST"
        log_warn "[$PDB_BASENAME] These residues are still included in the LigandMPNN input (reference set is used unchanged)."
        echo "$OOB_LIST" > "$OOB_NOTES_FILE"
        PDB_OOB_RESIDUES["$pdb"]="$OOB_LIST"
        PDB_SEL_STATUS["$pdb"]="ok_with_notes"
    else
        log "  [$PDB_BASENAME] All reference residues are within the docking box."
        PDB_OOB_RESIDUES["$pdb"]=""
        PDB_SEL_STATUS["$pdb"]="ok"
    fi

    # The number of design positions is always the reference count.
    PDB_N_POSITIONS["$pdb"]="$REF_N_POS"
done

# Remove PDBs where the box-check hard-failed (flagged as skip-worthy)
ACTIVE_PDBS_NEW=()
for pdb in "${ACTIVE_PDBS[@]}"; do
    if [ -z "${SKIPPED_SET[$pdb]+_}" ]; then
        ACTIVE_PDBS_NEW+=("$pdb")
    fi
done
ACTIVE_PDBS=("${ACTIVE_PDBS_NEW[@]}")

if [ ${#ACTIVE_PDBS[@]} -eq 0 ]; then
    log_err "All PDBs were skipped. Nothing to process."
    exit 1
fi

log "  Residue selection complete. All active PDBs will use $REF_N_POS reference position(s)."

###############################################################################
# STEP 3: LigandMPNN
###############################################################################
echo ""
echo "----------------------------------------------"
log "[3/3] LigandMPNN"
echo "----------------------------------------------"

conda activate "$ENV_LIGANDMPNN"

PDB_MULTI_JSON="$BASE_OUTPUT_DIR/pdb_ids.json"
REDESIGNED_JSON="$BASE_OUTPUT_DIR/redesigned_residues_multi.json"

# Build pdb_path_multi JSON
{
    echo "{"
    first=true
    for pdb in "${ACTIVE_PDBS[@]}"; do
        [ "$first" = false ] && echo ","
        first=false
        abs_pdb=$(realpath "$pdb")
        printf '  "%s": ""' "$abs_pdb"
    done
    echo ""
    echo "}"
} > "$PDB_MULTI_JSON"

# Build redesigned_residues_multi JSON.
# All PDBs receive the same REFERENCE_RESIDUES set.
{
    echo "{"
    first=true
    for pdb in "${ACTIVE_PDBS[@]}"; do
        [ "$first" = false ] && echo ","
        first=false
        abs_pdb=$(realpath "$pdb")
        printf '  "%s": "%s"' "$abs_pdb" "$REFERENCE_RESIDUES"
    done
    echo ""
    echo "}"
} > "$REDESIGNED_JSON"

log "  Launching LigandMPNN on ${#ACTIVE_PDBS[@]} PDB(s) with reference residue set..."

python "$LMPNN_PATH/run.py" \
    --model_type           "$MODEL_TYPE" \
    --checkpoint_ligand_mpnn "$LMPNN_PATH/$CHECKPOINT" \
    --pdb_path_multi       "$PDB_MULTI_JSON" \
    --out_folder           "$BASE_OUTPUT_DIR" \
    --temperature          "$TEMPERATURE" \
    --seed                 "$SEED" \
    --number_of_batches    "$NUM_BATCHES" \
    --redesigned_residues_multi "$REDESIGNED_JSON" \
    --save_stats 1

log "  LigandMPNN done."

###############################################################################
# STEP 4: Final Report
###############################################################################
echo ""
echo "----------------------------------------------"
echo "  PIPELINE COMPLETE -- FINAL REPORT"
echo "----------------------------------------------"

REPORT_TSV="$BASE_OUTPUT_DIR/pipeline_report.tsv"

printf "%-30s  %-12s  %-8s  %-8s  %-10s  %-s\n" \
    "PDB" "QC_STATUS" "CLASHES" "GEOM" "N_POS" "OUT_OF_BOX_RESIDUES"
printf "%-30s  %-12s  %-8s  %-8s  %-10s  %-s\n" \
    "------------------------------" "------------" "--------" "--------" "----------" "-------------------"

echo -e "PDB\tQC_STATUS\tCLASHES\tGEOM_ISSUES\tN_POSITIONS\tOUT_OF_BOX_RESIDUES" > "$REPORT_TSV"

ALL_PDBS=("${ACTIVE_PDBS[@]}" "${SKIPPED_PDBS[@]}")

for pdb in "${ALL_PDBS[@]}"; do
    PDB_BASENAME=$(basename "$pdb" .pdb)
    QC_ST="${PDB_QC_STATUS[$pdb]:-n/a}"
    N_CLASH="${PDB_QC_N_CLASHES[$pdb]:-0}"
    N_GEOM="${PDB_QC_N_GEOM[$pdb]:-0}"
    N_POS="${PDB_N_POSITIONS[$pdb]:-$REF_N_POS}"
    OOB="${PDB_OOB_RESIDUES[$pdb]:-}"
    [ -z "$OOB" ] && OOB="none"

    printf "%-30s  %-12s  %-8s  %-8s  %-10s  %-s\n" \
        "$PDB_BASENAME" "$QC_ST" "$N_CLASH" "$N_GEOM" "$N_POS" "$OOB"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$PDB_BASENAME" "$QC_ST" "$N_CLASH" "$N_GEOM" "$N_POS" "$OOB" \
        >> "$REPORT_TSV"
done

echo ""
log "Reference residues file -> $REF_RESIDUES_FILE"
log "Report TSV              -> $REPORT_TSV"
log "All done. Results in: $BASE_OUTPUT_DIR"