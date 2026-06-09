#!/bin/bash
# Exit on error
set -e

# Attiva environment conda ligandmpnn_env
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ligandmpnn_env

# Vai nella cartella LigandMPNN
LMPNN_PATH="/software/lmpnn/LigandMPNN"


#REDESIGN_RESIDUES=$(cat ./output/selected_residues.txt) #(i.g. C1 C2 C3 C4 C5 C6 C7 C8 C9 C10)
#FIXE_RESIDUES= #(i.gC1 C2 C3 C4 C5 C6 C7 C8 C9 C10)
#BIAS_RESIDUES= #(ig W:3.0,P:3.0,C:3.0,A:-3.0)

#multi .json
#{
#"./inputs/1BC8.pdb": "",
#"./inputs/4GYT.pdb": ""
#}

#fixmulti .json
#{
#"./inputs/1BC8.pdb": "C1 C2 C3 C4 C5 C10 C22",
#"./inputs/4GYT.pdb": "A7 A8 A9 A10 A11 A12 A13 B38"
#}

# Esegui LigandMPNN
python "$LMPNN_PATH/run.py" \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "$LMPNN_PATH/model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "/home/tedeschg/prj/protein-perturbation/data/abl1/crystal_structure/2hzi_clean.pdb" \
    --out_folder "/home/tedeschg/prj/protein-perturbation/output_abl/lmpnn_crystal_reference" \
    --temperature 0.05 \
    --seed 111 \
    --batch_size 3 \
    --number_of_batches 5

#    --redesigned_residues "$REDESIGN_RESIDUES"\
#    --ligand_mpnn_cutoff_for_score "6.0" \
#This sets the cutoff distance in angstroms to select residues that are considered to be close to ligand atoms. This flag only affects the num_ligand_res and ligand_confidence in the output fasta files.

#    --fixed_residues "$FIX_RESIDUES"\
#    --bias_AA "$BIAS_RESIDUES"\
#    --pdb_path_multi "./inputs/pdb_ids.json"\
#    --fixed_residues_multi "./inputs/fix_residues_multi.json"
