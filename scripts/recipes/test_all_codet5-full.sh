#!/bin/bash

sbatch_args() {
    local job_name=$1
    local seed=$2
    local gpu=$3
    local time=$4
    name=${job_name}_${out_suffix}_${seed}
    echo "--gpus-per-node=1 --partition=${gpu} --mem=64gb --job-name=${name} --time=${time}:00:00 --output=/projects/sciences/computing/norja159/research/scripts/${job_name}_out/${name}_%j.out"
}

test_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2
    local time=$3

	echo seeds: "${seed_array[@]}"
	echo datasets: "${dataset_array[@]}"
	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
		export dir=${name}
		for seed in "${seed_array[@]}"; do
			export seed=${seed}
			sbatch $(sbatch_args ${name} ${seed} aoraki_gpu ${time}) test_split.sh
		done
	done
}

export out_suffix=splits
export pos_weight=1.0
export epoch=5
seeds=(123456 789012 345678)
datasets=(icvul mvdsc_mixed devign vuldeepecker cvefixes juliet reveal)
big_datasets=(diversevul draper) 

#codet5 encoder decoder
export model_name=Salesforce/codet5-base
export tokenizer_name=Salesforce/codet5-base
export model_type=codet5_full
train_split_all datasets seeds 20
train_split_all big_datasets seeds 50
