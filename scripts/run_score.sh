#!/bin/bash
# Exit on error
set -e

# Activate conda environment
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ligandmpnn_env

LMPNN_PATH="/software/lmpnn/LigandMPNN"
BACKBONES_DIR="/home/tedeschg/prj/protein-perturbation/output_lmpnn/backbones"
OUT_FOLDER="/home/tedeschg/prj/protein-perturbation/output_scoring/"

### Only for the reference ###
#BACKBONES_DIR="/home/tedeschg/prj/protein-perturbation/output_abl/lmpnn_crystal_reference/backbones/2hzi_clean_11.pdb"
#OUT_FOLDER="/home/tedeschg/prj/protein-perturbation/output_abl/lmpnn_crystal_reference"
###

mkdir -p "$OUT_FOLDER"


### Only for the reference ###
#PDB_FILES=("$BACKBONES_DIR" )
###

PDB_FILES=("$BACKBONES_DIR"/*.pdb)
TOTAL=${#PDB_FILES[@]}

echo "Found $TOTAL PDB files to process"
echo "=================================="

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

    echo "  Done: $(basename "$pdb")"
done

echo "=================================="
echo "All $TOTAL PDB files scored."
echo "Output saved to: $OUT_FOLDER"