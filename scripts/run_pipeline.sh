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
#                         Uses the reference ligand to define the box center
#                         and the `autobox_add` radius (from config.yaml).
#   3. LigandMPNN     – sequence design in batch mode
#   4. Final Report   – summary table printed to terminal + written to TSV
#
# NOTES
# -----
#   - The residues selection step extracts all residues that fall within
#     the GNINA‑style docking box (sphere of radius `autobox_add` around
#     the ligand).
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
# YAML parsing (FIXED: robust nested key reading)
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
# FIXED: more robust parsing with awk
read_yaml_nested() {
    local section="$1"
    local key="$2"
    awk -v s="$section" -v k="$key" '
        $0 ~ "^" s ":" { in_section=1; next }
        in_section && /^[^ ]/ { in_section=0 }
        in_section && $0 ~ "^[[:space:]]+" k ":" {
            # Estrai il valore: rimuovi spazi, la chiave, i due punti,
            # commenti finali e virgolette
            sub("^[[:space:]]*" k ":[[:space:]]*", "")
            sub("[[:space:]]*#.*$", "")
            gsub(/^"|"$/, "")
            gsub(/^'"'"'|'"'"'$/, "")
            print
            exit
        }
    ' "$CONFIG_FILE"
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

# residues_selection (NEW: docking‑box based)
RESIDUE_OUTPUT_FILE=$(read_yaml_nested "residues_selection" "output_file")
REF_COMPLEX_RAW=$(read_yaml_nested "residues_selection" "reference_complex")
AUTOBOX_ADD=$(read_yaml_nested "residues_selection" "autobox_add")
OUT_FORMAT=$(read_yaml_nested "residues_selection" "output_format")
INCL_HETATM=$(read_yaml_nested "residues_selection" "include_hetatm_residues")

# Fallback per valori mancanti
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

# Rendi il percorso assoluto se relativo
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
declare -A PDB_RESIDUES_MAP
declare -A PDB_N_POSITIONS
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
# STEP 2: Residue Selection (NEW: docking‑box based)
###############################################################################
echo ""
echo "----------------------------------------------"
log "[2/3] Residue Selection (docking box)"
echo "----------------------------------------------"

for pdb in "${ACTIVE_PDBS[@]}"; do
    PDB_BASENAME=$(basename "$pdb" .pdb)
    PDB_DIR="$BASE_OUTPUT_DIR/$PDB_BASENAME"
    SELECTED_RESIDUES_FILE="$PDB_DIR/$RESIDUE_OUTPUT_FILE"

    SEL_ARGS=(
        --protein "$pdb"
        --reference_complex "$REF_COMPLEX"
        --autobox_add "$AUTOBOX_ADD"
        --output "$SELECTED_RESIDUES_FILE"
        --format "$OUT_FORMAT"
    )
    [ "$INCL_HETATM" = "true" ] && SEL_ARGS+=(--include_hetatm)

    log "  [$PDB_BASENAME] Selecting residues within box (radius=${AUTOBOX_ADD} Å)..."

    set +e
    SEL_OUTPUT=$(python "$PROJECT_ROOT/residues_selection.py" "${SEL_ARGS[@]}" 2>&1)
    SEL_EXIT=$?
    set -e

    echo "$SEL_OUTPUT" | sed 's/^/    /'

    if [ "$SEL_EXIT" -ne 0 ]; then
        log_err "[$PDB_BASENAME] residues_selection.py failed (exit=$SEL_EXIT). Skipping."
        PDB_SEL_STATUS["$pdb"]="error"
        mark_skipped "$pdb"
        continue
    else
        PDB_SEL_STATUS["$pdb"]="ok"
    fi

    if [ ! -f "$SELECTED_RESIDUES_FILE" ]; then
        log_err "[$PDB_BASENAME] $RESIDUE_OUTPUT_FILE not created despite exit=0."
        PDB_SEL_STATUS["$pdb"]="error"
        mark_skipped "$pdb"
        continue
    fi

    RESIDUES=$(cat "$SELECTED_RESIDUES_FILE")
    N_POS=$(echo "$RESIDUES" | tr -s ' \t\n' '\n' | grep -c '[^[:space:]]' || true)
    N_POS="${N_POS:-0}"
    PDB_RESIDUES_MAP["$pdb"]="$RESIDUES"
    PDB_N_POSITIONS["$pdb"]="$N_POS"
    log "  [$PDB_BASENAME] Selected $N_POS positions."
done

# Ricostruisci ACTIVE_PDBS escludendo quelli saltati
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

UNIQUE_N_POS=$(for pdb in "${ACTIVE_PDBS[@]}"; do echo "${PDB_N_POSITIONS[$pdb]:-0}"; done | sort -u)
N_UNIQUE=$(echo "$UNIQUE_N_POS" | grep -c '[^[:space:]]' || true)
if [ "${N_UNIQUE:-1}" -gt 1 ]; then
    log_warn "n_positions_used differs across PDBs: $(echo "$UNIQUE_N_POS" | tr '\n' ' ')"
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

# Build redesigned_residues_multi JSON (usando la MAPPA)
{
    echo "{"
    first=true
    for pdb in "${ACTIVE_PDBS[@]}"; do
        [ "$first" = false ] && echo ","
        first=false
        abs_pdb=$(realpath "$pdb")
        RESIDUES="${PDB_RESIDUES_MAP[$pdb]:-}"
        # Se per qualche motivo RESIDUES è vuoto, scrivi stringa vuota
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

printf "%-30s  %-12s  %-8s  %-8s  %-10s\n" \
    "PDB" "QC_STATUS" "CLASHES" "GEOM" "N_POS"
printf "%-30s  %-12s  %-8s  %-8s  %-10s\n" \
    "------------------------------" "------------" "--------" "--------" "----------"

echo -e "PDB\tQC_STATUS\tCLASHES\tGEOM_ISSUES\tN_POSITIONS" > "$REPORT_TSV"

# Combina attivi + saltati
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