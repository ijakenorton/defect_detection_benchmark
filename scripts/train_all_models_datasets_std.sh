#!/bin/bash

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Not in a git repository. Cannot determine PROJECT_ROOT." >&2
    exit 1
fi

source "${PROJECT_ROOT}/scripts/utils/utils.sh"
setup_paths

#defaults
export out_suffix=splits
export pos_weight=1.0
export epoch=5
seeds=(123456 789012 345678)
datasets=(icvul mvdsc_mixed devign vuldeepecker cvefixes juliet reveal)
big_datasets=(diversevul draper) 

#Run
echo "=========================Config==========================="
ls -1 ${MODEL_CONFIG_DIR}/*.sh
echo seeds: "${seeds[@]}"
echo datasets: "${datasets[@]}"
echo big_datasets: "${big_datasets[@]}"
for model_config in ${model_config_dir}/*.sh; do
    (
    source "$model_config"

    sbatch_train_split_all datasets seeds aoraki_gpu 20
    sbatch_train_split_all big_datasets seeds aoraki_gpu_H100 50
)
done


