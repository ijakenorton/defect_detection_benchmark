#!/bin/bash

setup_paths() {
    if [[ -z "$PROJECT_ROOT" ]]; then
        SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
        export PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    fi

    export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output}"
    export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
    export MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"
    export SCRIPTS_DIR="${SCRIPTS_DIR:-$PROJECT_ROOT/scripts}"
    export CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/Defect-detection/code}"
    export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${SCRIPTS_DIR}/model_configs}"
}

sbatch_args() {
    local job_name=${vulndetection:-$1}
    local seed=${$2:-123456}
    local gpu=${$3:-aoraki_gpu}
    local time=${$4:-5}
    name=${job_name}_${out_suffix}_${seed}
    echo "--gpus-per-node=1 --partition=${gpu} --mem=64gb --job-name=${name} --time=${time}:00:00 --output=${SCRIPTS_DIR}/${job_name}_out/${name}_%j.out"
}

sbatch_test_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2
    local gpu=${$3:-aoraki_gpu}
    local time=${$4:-5}

	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
        (
		export dataset_name=${name}
		for seed in "${seed_array[@]}"; do
            (
			export seed=${seed}
			sbatch $(sbatch_args ${name} ${seed} ${gpu} ${time}) test_split.sh 
        )
		done
    )
	done
}


sbatch_train_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2
    local gpu=${$3:-aoraki_gpu}
    local time=${$4:-5}

	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
        (
		export dataset_name=${name}
		for seed in "${seed_array[@]}"; do
            (
			export seed=${seed}
			sbatch $(sbatch_args ${name} ${seed} ${gpu} ${time}) train_split.sh 
        )
		done
    )
	done
}

# For running batch jobs async without sbatch
async_test_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2

	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
        (
		export dataset_name=${name}
		for seed in "${seed_array[@]}"; do
            (
			export seed=${seed}
			./test_split.sh &
        )
		done
    )
	done
}


# For running batch jobs async without sbatch
async_train_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2

	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
        (
		export dataset_name=${name}
		for seed in "${seed_array[@]}"; do
            (
			export seed=${seed}
			./train_split.sh &
        )
		done
    )
	done
}


# For running batch jobs sync without sbatch
sync_test_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2

	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
        (
		export dataset_name=${name}
		for seed in "${seed_array[@]}"; do
            (
			export seed=${seed}
			./test_split.sh
        )
		done
    )
	done
}


# For running batch jobs sync without sbatch
sync_train_split_all() {
    local -n dataset_array=$1
    local -n seed_array=$2

	echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix: $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name seed: $seeds"
	for name in "${dataset_array[@]}"; do
        (
		export dataset_name=${name}
		for seed in "${seed_array[@]}"; do
            (
			export seed=${seed}
			./train_split.sh
        )
		done
    )
	done
}

