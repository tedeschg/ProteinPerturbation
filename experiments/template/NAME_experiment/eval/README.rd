cd output_eval_lmpnn

nano ../../../../../scripts/config.yaml

nohup bash ../../../../../scripts/run_pipeline.sh ../../../../../data/dude-z_dataset/complexes/ABL1/eval/* > output_eval_lmpnn.log 2>&1 & 

scp output_eval_lmpnn skirit.metacentrum.cz:/storage/brno2/home/tedeschg/prj/protein-pertrubation/experiments/dude-z/ABL1/

scp -r skirit.metacentrum.cz:/storage/brno2/home/tedeschg/prj/protein-pertrubation/experiments/dude-z/ABL1/output_eval_score .
