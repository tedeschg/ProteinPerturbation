#!/bin/bash
# Exit on error
set -e

# Activate conda environment.
# On Metacentrum the wrapper does `module add mambaforge` first, so the local
# conda.sh may not exist — source it only if present.
CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_PROFILE" ] && source "$CONDA_PROFILE"

# LMPNN_ENV is either an env name (local) or a full path (Metacentrum).
# Default keeps the original local behaviour.
LMPNN_ENV="${LMPNN_ENV:-ligandmpnn_env}"
conda activate "$LMPNN_ENV"

# Paths are overridable via env vars so the same script works locally and as a
# PBS job-array slot (the wrapper exports these to point at /storage/brno2/...).
LMPNN_PATH="${LMPNN_PATH:-/software/lmpnn/LigandMPNN}"
BACKBONES_DIR="${BACKBONES_DIR:-/home/tedeschg/prj/protein-perturbation/experiments/dude-z_experiments/reference/output_reference_lmpnn/backbones}"
OUT_FOLDER="${OUT_FOLDER:-/home/tedeschg/prj/protein-perturbation/experiments/dude-z_experiments/reference/output_reference_score/}"

# Job-array chunking. When submitted as a PBS array, the wrapper sets
# CHUNK_SIZE and each slot processes PDBs in [array_idx*CHUNK_SIZE, +CHUNK_SIZE).
# When run locally (no PBS_ARRAY_INDEX), CHUNK_SIZE is ignored and ALL PDBs
# in BACKBONES_DIR are scored.
CHUNK_SIZE="${CHUNK_SIZE:-0}"
ARRAY_INDEX="${PBS_ARRAY_INDEX:-}"

# Format integer seconds as HH:MM:SS
fmt_duration() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# eta_str DONE TOTAL START_EPOCH -> "elapsed=hh:mm:ss, ETA=hh:mm:ss, finish≈HH:MM:SS"
eta_str() {
    local done=$1 total=$2 start=$3
    local now=$(date +%s)
    local elapsed=$((now - start))
    if [ "$done" -le 0 ] || [ "$total" -le 0 ]; then
        echo "elapsed=$(fmt_duration $elapsed)"
        return
    fi
    local remaining=$(( (elapsed * (total - done)) / done ))
    local finish=$(date -d "+${remaining} seconds" '+%H:%M:%S' 2>/dev/null || echo "?")
    echo "elapsed=$(fmt_duration $elapsed), ETA=$(fmt_duration $remaining), finish≈${finish}"
}

mkdir -p "$OUT_FOLDER"

ALL_PDB_FILES=("$BACKBONES_DIR"/*.pdb)
TOTAL_ALL=${#ALL_PDB_FILES[@]}

# Decide which slice of PDBs this run will process.
if [ -n "$ARRAY_INDEX" ] && [ "$CHUNK_SIZE" -gt 0 ]; then
    START=$(( ARRAY_INDEX * CHUNK_SIZE ))
    END=$(( START + CHUNK_SIZE ))
    [ "$END" -gt "$TOTAL_ALL" ] && END=$TOTAL_ALL
    if [ "$START" -ge "$TOTAL_ALL" ]; then
        echo "Array slot $ARRAY_INDEX has no work (START=$START >= TOTAL=$TOTAL_ALL). Exiting."
        exit 0
    fi
    PDB_FILES=("${ALL_PDB_FILES[@]:$START:$((END - START))}")
    echo "Array slot $ARRAY_INDEX: scoring PDBs [$START, $END) of $TOTAL_ALL"
else
    PDB_FILES=("${ALL_PDB_FILES[@]}")
    echo "Scoring all $TOTAL_ALL PDBs (no chunking)"
fi
TOTAL=${#PDB_FILES[@]}

echo "=================================="

SCORE_START=$(date +%s)

for i in "${!PDB_FILES[@]}"; do
    pdb="${PDB_FILES[$i]}"
    echo "[$(( i + 1 ))/$TOTAL] Scoring: $(basename "$pdb")"

    python "$LMPNN_PATH/score.py" \
        --model_type "ligand_mpnn" \
        --checkpoint_ligand_mpnn "$LMPNN_PATH/model_params/ligandmpnn_v_32_010_25.pt" \
        --pdb_path "$pdb" \
        --out_folder "$OUT_FOLDER" \
        --seed 111 \
        --batch_size 1 \
        --number_of_batches 10 \
        --single_aa_score 1 \
        --use_sequence 1

    echo "  Done: $(basename "$pdb")  [$(eta_str "$(( i + 1 ))" "$TOTAL" "$SCORE_START")]"
done

echo "=================================="
echo "All $TOTAL PDB files scored."
echo "Total runtime: $(fmt_duration $(( $(date +%s) - SCORE_START )) )"
echo "Output saved to: $OUT_FOLDER"