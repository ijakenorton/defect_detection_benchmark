#!/bin/bash

THIS_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
source ${THIS_DIR}/utils/utils.sh
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

    sbatch_test_split_all datasets seeds 20
    sbatch_test_split_all big_datasets seeds 50
)
done


