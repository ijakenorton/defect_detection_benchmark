#!/bin/bash

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Not in a git repository. Cannot determine PROJECT_ROOT." >&2
    exit 1
fi

source "${PROJECT_ROOT}/scripts/utils/utils.sh"
setup_paths

source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 
conda activate ensemble

source ${MODEL_CONFIG_DIR}/linevul.sh
export out_suffix=splits
export pos_weight=1.0
export epoch=1
seeds=(123456)
datasets=(juliet)

#echo $(sbatch_args juliet 123456 aoraki_gpu 20)

#sbatch $(sbatch_args juliet 123456 aoraki_gpu 20) ${SCRIPTS_DIR}/test_split.sh 
sbatch_test_split_all datasets seeds aoraki_gpu 20
