export model_name=microsoft/graphcodebert-base
export tokenizer_name=microsoft/graphcodebert-base
export model_type=roberta
export out_suffix=splits
export pos_weight=1.0
export epoch=5

seeds=(123456 789012 345678)
datasets=(icvul vdsc_mixed devign vuldeepecker cvefixes juliet reveal)

echo seeds: "${seeds}"
echo datasets: "${datasets}"
echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seed"


for name in "${datasets[@]}"; do
	export dataset_name=${name}
	for seed in "${seeds[@]}"; do
		export seed=${seed}
		sbatch $(sbatch_args ${name} ${seed} aoraki_gpu 20) train_split.sh
	done
done

big_datasets=(diversevul draper) 

for name in "${big_datasets[@]}"; do
	export dataset_name=${name}
	for seed in "${seeds[@]}"; do
		export seed=${seed}
		sbatch $(sbatch_args ${name} ${seed} aoraki_gpu_H100 50) train_split.sh
	done
done
