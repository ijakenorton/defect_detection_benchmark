export model_name=microsoft/graphcodebert-base
export tokenizer_name=microsoft/graphcodebert-base
export model_type=roberta
export out_suffix=splits
export pos_weight=1.0
export epoch=5
#export seed=123456

seeds=(123456 789012 345678)
datasets=(icvul vdsc_mixed devign vuldeepecker cvefixes juliet reveal)

echo seeds: "${seeds}"
echo datasets: "${datasets}"
echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seed"

sbatch_args() {
    local job_name=$1
    local seed=$2
    local gpu=$3
    local time=$4
    name=${job_name}_${out_suffix}_${seed}
    echo "--gpus-per-node=1 --partition=${gpu} --mem=64gb --job-name=${name} --time=${time}:00:00 --output=/projects/sciences/computing/norja159/research/scripts/${job_name}_out/${name}_%j.out"
}



for name in "${datasets[@]}"; do
	export dir=${name}
	for seed in "${seeds[@]}"; do
		export seed=${seed}
		sbatch $(sbatch_args ${name} ${seed} aoraki_gpu 20) test_split.sh
	done
done

big_datasets=(diversevul draper) 

for name in "${big_datasets[@]}"; do
	export dir=${name}
	for seed in "${seeds[@]}"; do
		export seed=${seed}
		sbatch $(sbatch_args ${name} ${seed} aoraki_gpu_H100 50) test_split.sh
	done
done
