export model_name=../models/pretrained/natgen
export tokenizer_name=Salesforce/codet5-base
export model_type=natgen
export pos_weight=1.0
export out_suffix=test_splits
export epoch=1
#export seed=123456

seeds=(123456 789012 345678)
datasets=(draper)

echo seeds: "${seeds[@]}"
echo datasets: "${datasets[@]}"
echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seed"

sbatch_args() {
    local job_name=$1
    echo "--gpus-per-node=1 --partition=aoraki_gpu --mem=64gb --job-name=${job_name}_${out_suffix} --time=4:00:00 --output=/projects/sciences/computing/norja159/research/scripts/${job_name}_out/${model_type}_%j.out"
}

for name in "${datasets[@]}"; do
	export dir=${name}
	for seed in "${seeds[@]}"; do
		export seed=${seed}
		sbatch $(sbatch_args ${name}) train_split.sh
	done
done

