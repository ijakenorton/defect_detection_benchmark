#!/bin/bash

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Not in a git repository. Cannot determine PROJECT_ROOT." >&2
    exit 1
fi

source "${PROJECT_ROOT}/scripts/utils/utils.sh"
setup_paths

# export model_name=microsoft/graphcodebert-base
# export tokenizer_name=microsoft/graphcodebert-base
# export model_type=roberta
# Below is equivalent to the above exports
source ${MODEL_CONFIG_DIR}/model_configs/graphcodebert.sh
export out_suffix=splits
export pos_weight=1.0
export epoch=5

seeds=(123456 789012 345678)
datasets=(icvul mvdsc_mixed devign vuldeepecker cvefixes juliet reveal)
big_datasets=(diversevul draper) 

sbatch_test_split_all datasets seeds aoraki_gpu 20
sbatch_test_split_all big_datasets seeds aoraki_gpu_H100 50


