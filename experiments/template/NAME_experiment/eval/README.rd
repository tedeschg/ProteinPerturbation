cd output_eval_lmpnn

nano ../../../../../scripts/config.yaml

nohup bash ../../../../../scripts/run_pipeline.sh ../../../../../data/dude-z_dataset/complexes/ABL1/eval/* > output_eval_lmpnn.log 2>&1 & 

python ../../../../../scripts/aucroc.py --scoring-dir /home/tedeschg/prj/protein-perturbation/experiments/dude-z_experiments/AA2AR/ABL1/output_eval_score  --reference /home/tedeschg/prj/protein-perturbation/experiments/dude-z_experiments/ABL1/reference/output_reference_score/ --out-csv report/AA2AR.csv --out-roc report/AA2AR.png

zip -r eval.zip eval/
