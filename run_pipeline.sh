#!/bin/bash
###############################################################################
# LigandMPNN Pipeline - Multi-PDB
#
# USAGE
# -----
#   bash run_pipeline.sh file1.pdb file2.pdb [file3.pdb ...]
#
#
# PIPELINE STEPS
# --------------
#   1. Pose QC        – clash + geometry check  (scripts/pose_qc.py)
#   2. Residue Selection – ligand-centric interface residues (scripts/select_residues.py)
#   3. LigandMPNN     – sequence design in batch mode
#   4. Final Report   – summary table printed to terminal + written to TSV
#
# NOTES
# -----
#   - Commented-out flags (e.g. --redesigned_residues_multi) are intentional
#     placeholders; do not uncomment unless the corresponding feature is needed.
#   - Re-running the script on the same output_dir will overwrite previous results.
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
# YAML parsing
# ---------------------------------------------------------------------------

# Read a top-level scalar value: read_yaml "key"
read_yaml() {
    local key="$1"
    grep "^${key}:" "$CONFIG_FILE" \
        | sed "s/^${key}:[[:space:]]*//" \
        | sed 's/#.*//' \
        | sed 's/[[:space:]]*$//' \
        | sed 's/"//g' \
        | sed "s/'//g" \
        | envsubst
}

# Read a scalar nested one level deep: read_yaml_nested "section" "key"
read_yaml_nested() {
    local section="$1"
    local key="$2"
    awk "/^${section}:/{found=1; next} found && /^[^ ]/{found=0} found && /^[[:space:]]+${key}:/{print}" \
        "$CONFIG_FILE" \
        | sed "s/.*${key}:[[:space:]]*//" \
        | sed 's/#.*//' \
        | sed 's/[[:space:]]*$//' \
        | sed 's/"//g' \
        | sed "s/'//g"
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

# residue_selection
CUTOFF_DISTANCE=$(read_yaml_nested "residue_selection" "cutoff_distance")
RESIDUE_OUTPUT_FILE=$(read_yaml_nested "residue_selection" "output_file")
SEL_MIN_HEAVY=$(read_yaml_nested "residue_selection" "min_heavy_atoms")
SEL_MIN_EXPECTED=$(read_yaml_nested "residue_selection" "min_expected")
SEL_MAX_EXPECTED=$(read_yaml_nested "residue_selection" "max_expected")
SEL_INCLUDE_HETATM=$(read_yaml_nested "residue_selection" "include_hetatm_residues")

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

declare -A PDB_QC_STATUS        # pass / warn_clash / qc_crashed / skipped
declare -A PDB_QC_N_CLASHES
declare -A PDB_QC_N_GEOM
declare -A PDB_RESIDUES_MAP
declare -A PDB_N_POSITIONS
declare -A PDB_SEL_STATUS       # ok / warn_clash / error

# Use associative array as a set to prevent duplicate entries in SKIPPED_PDBS
declare -A SKIPPED_SET
SKIPPED_PDBS=()
ACTIVE_PDBS=()

# Mark a PDB as skipped (deduplicates automatically)
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

    # Build pose_qc.py argument list
    QC_ARGS=(
        --pdb "$pdb"
        --clash_dist "$QC_CLASH_DIST"
        --bond_length_tol "$QC_BOND_TOL"
    )
    [ "$QC_STRICT"   = "true" ] && QC_ARGS+=(--strict)
    [ "$QC_OUT_JSON" = "true" ] && QC_ARGS+=(--out_json "$PDB_DIR/qc_report.json")

    log "  [$PDB_BASENAME] Running pose_qc.py..."

    set +e
    QC_OUTPUT=$(python "$PROJECT_ROOT/scripts/pose_qc.py" "${QC_ARGS[@]}" 2>&1)
    QC_EXIT=$?
    set -e

    # Parse clash/geometry counts from output; default to 0 if not found
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
        # Hard QC failure (clash or strict geometry violation)
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
        # pose_qc.py crashed (Python exception, missing dependencies, etc.)
        log_warn "[$PDB_BASENAME] pose_qc.py crashed (exit=$QC_EXIT). Output:"
        echo "$QC_OUTPUT" | sed 's/^/    /'
        log_warn "[$PDB_BASENAME] Continuing without QC validation."
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
# STEP 2: Residue Selection
###############################################################################
echo ""
echo "----------------------------------------------"
log "[2/3] Residue Selection"
echo "----------------------------------------------"

for pdb in "${ACTIVE_PDBS[@]}"; do
    PDB_BASENAME=$(basename "$pdb" .pdb)
    PDB_DIR="$BASE_OUTPUT_DIR/$PDB_BASENAME"
    SELECTED_RESIDUES_FILE="$PDB_DIR/$RESIDUE_OUTPUT_FILE"

    SEL_ARGS=(
        --pdb "$pdb"
        --dist "$CUTOFF_DISTANCE"
        --out "$SELECTED_RESIDUES_FILE"
        --min_heavy_atoms "$SEL_MIN_HEAVY"
        --min_expected "$SEL_MIN_EXPECTED"
        --max_expected "$SEL_MAX_EXPECTED"
    )
    [ "$SEL_INCLUDE_HETATM" = "true" ] && SEL_ARGS+=(--include_hetatm_residues)
    [ "$QC_OUT_JSON"         = "true" ] && SEL_ARGS+=(--out_json "$PDB_DIR/selection_summary.json")

    log "  [$PDB_BASENAME] Selecting residues (dist=${CUTOFF_DISTANCE} Å)..."

    set +e
    SEL_OUTPUT=$(python "$PROJECT_ROOT/scripts/select_residues.py" "${SEL_ARGS[@]}" 2>&1)
    SEL_EXIT=$?
    set -e

    # Always print full output so Python errors are visible
    echo "$SEL_OUTPUT" | sed 's/^/    /'

    # Exit codes: 0 = clean, 1 = hard error, 2 = soft clash warning
    if [ "$SEL_EXIT" -eq 1 ]; then
        log_err "[$PDB_BASENAME] select_residues.py failed (exit=1). Skipping."
        PDB_SEL_STATUS["$pdb"]="error"
        mark_skipped "$pdb"
        continue
    elif [ "$SEL_EXIT" -eq 2 ]; then
        log_warn "[$PDB_BASENAME] Clash warning from residue selection -- continuing."
        PDB_SEL_STATUS["$pdb"]="warn_clash"
    else
        PDB_SEL_STATUS["$pdb"]="ok"
    fi

    if [ ! -f "$SELECTED_RESIDUES_FILE" ]; then
        log_err "[$PDB_BASENAME] $RESIDUE_OUTPUT_FILE not created despite exit=$SEL_EXIT."
        log_err "            Check the output above for the actual Python error."
        PDB_SEL_STATUS["$pdb"]="error"
        mark_skipped "$pdb"
        continue
    fi

    RESIDUES=$(cat "$SELECTED_RESIDUES_FILE")
    # Count non-empty whitespace-separated tokens robustly
    N_POS=$(echo "$RESIDUES" | tr -s ' \t\n' '\n' | grep -c '[^[:space:]]' || true)
    N_POS="${N_POS:-0}"
    PDB_RESIDUES_MAP["$pdb"]="$RESIDUES"
    PDB_N_POSITIONS["$pdb"]="$N_POS"
    log "  [$PDB_BASENAME] Selected $N_POS positions."
done

# Rebuild ACTIVE_PDBS excluding any that failed selection
ACTIVE_PDBS_NEW=()
for pdb in "${ACTIVE_PDBS[@]}"; do
    if [ -z "${SKIPPED_SET[$pdb]+_}" ]; then
        ACTIVE_PDBS_NEW+=("$pdb")
    fi
done
ACTIVE_PDBS=("${ACTIVE_PDBS_NEW[@]}")

if [ ${#ACTIVE_PDBS[@]} -eq 0 ]; then
    log_err "All PDBs failed residue selection. Nothing to process."
    exit 1
fi

# Warn if position counts differ across PDBs (may affect scoring fairness)
UNIQUE_N_POS=$(for pdb in "${ACTIVE_PDBS[@]}"; do echo "${PDB_N_POSITIONS[$pdb]:-0}"; done | sort -u)
N_UNIQUE=$(echo "$UNIQUE_N_POS" | grep -c '[^[:space:]]' || true)
if [ "${N_UNIQUE:-1}" -gt 1 ]; then
    log_warn "n_positions_used differs across PDBs: $(echo "$UNIQUE_N_POS" | tr '\n' ' ')"
    log_warn "This may affect scoring fairness. Check scaffold consistency."
else
    log "  All PDBs selected ${UNIQUE_N_POS} positions."
fi

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

# Build pdb_path_multi JSON  {"<abs_path>": "", ...}
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

# Build redesigned_residues_multi JSON  {"<abs_path>": "<residue list>", ...}
{
    echo "{"
    first=true
    for pdb in "${ACTIVE_PDBS[@]}"; do
        [ "$first" = false ] && echo ","
        first=false
        abs_pdb=$(realpath "$pdb")
        RESIDUES="${PDB_RESIDUES_MAP[$pdb]}"
        printf '  "%s": "%s"' "$abs_pdb" "$RESIDUES"
    done
    echo ""
    echo "}"
} > "$REDESIGNED_JSON"

log "  Launching LigandMPNN on ${#ACTIVE_PDBS[@]} PDB(s)..."

python "$LMPNN_PATH/run.py" \
    --model_type           "$MODEL_TYPE" \
    --checkpoint_ligand_mpnn "$LMPNN_PATH/$CHECKPOINT" \
    --pdb_path_multi       "$PDB_MULTI_JSON" \
    --out_folder           "$BASE_OUTPUT_DIR" \
    --temperature          "$TEMPERATURE" \
    --seed                 "$SEED" \
    --number_of_batches    "$NUM_BATCHES" \
    --save_stats 1

#--redesigned_residues_multi "$REDESIGNED_JSON" \

log "  LigandMPNN done."

###############################################################################
# STEP 4: Final Report
###############################################################################
echo ""
echo "----------------------------------------------"
echo "  PIPELINE COMPLETE -- FINAL REPORT"
echo "----------------------------------------------"

REPORT_TSV="$BASE_OUTPUT_DIR/pipeline_report.tsv"

# Print header
printf "%-30s  %-12s  %-8s  %-8s  %-10s\n" \
    "PDB" "QC_STATUS" "CLASHES" "GEOM" "N_POS"
printf "%-30s  %-12s  %-8s  %-8s  %-10s\n" \
    "------------------------------" "------------" "--------" "--------" "----------"

# TSV header
echo -e "PDB\tQC_STATUS\tCLASHES\tGEOM_ISSUES\tN_POSITIONS" > "$REPORT_TSV"

# Combine active + skipped, preserving insertion order; deduplication via SKIPPED_SET
# (SKIPPED_PDBS already has no duplicates thanks to mark_skipped)
ALL_PDBS=("${ACTIVE_PDBS[@]}" "${SKIPPED_PDBS[@]}")

for pdb in "${ALL_PDBS[@]}"; do
    PDB_BASENAME=$(basename "$pdb" .pdb)
    QC_ST="${PDB_QC_STATUS[$pdb]:-n/a}"
    N_CLASH="${PDB_QC_N_CLASHES[$pdb]:-0}"
    N_GEOM="${PDB_QC_N_GEOM[$pdb]:-0}"
    N_POS="${PDB_N_POSITIONS[$pdb]:-0}"

    printf "%-30s  %-12s  %-8s  %-8s  %-10s\n" \
        "$PDB_BASENAME" "$QC_ST" "$N_CLASH" "$N_GEOM" "$N_POS"

    printf "%s\t%s\t%s\t%s\t%s\n" \
        "$PDB_BASENAME" "$QC_ST" "$N_CLASH" "$N_GEOM" "$N_POS" \
        >> "$REPORT_TSV"
done

echo ""
log "Report TSV -> $REPORT_TSV"
log "All done. Results in: $BASE_OUTPUT_DIR"