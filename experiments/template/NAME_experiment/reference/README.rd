cd output_reference_lmpnn

/home/tedeschg/prj/protein-perturbation/data/dude-z_dataset/complexes/ABL1/reference/abl1_crystal.pdb

nano ../../../../../scripts/config.yaml

pwd

nano ../../../../../scripts/config.yaml

nohup bash ../../../../../scripts/run_pipeline.sh ../../../../../data/dude-z_dataset/complexes/ABL1/reference/abl1_crystal.pdb > output_reference_lmpnn.log 2>&1 & 

nano ../../../../scripts/run_score.sh 

bash ../../../../scripts/run_score.sh
