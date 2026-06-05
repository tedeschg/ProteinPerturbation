#!/bin/bash
# Exit on error
set -e

# Attiva environment conda ligandmpnn_env
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ligandmpnn_env

# Vai nella cartella LigandMPNN
LMPNN_PATH="/software/lmpnn/LigandMPNN"

python "$LMPNN_PATH/score.py" \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "$LMPNN_PATH/model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "/home/tedeschg/prj/protein-perturbation/output_decoy2/backbones/decoy_random_2_pose_1_1.pdb" \
    --out_folder "/home/tedeschg/prj/protein-perturbation/output_scoring/decoy" \
    --seed 111 \
    --batch_size 1 \
    --number_of_batches 10 \
    --single_aa_score 1 \
    --use_sequence 1

