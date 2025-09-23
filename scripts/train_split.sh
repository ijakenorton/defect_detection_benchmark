#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 

source ${SCRIPTS_DIR}/utils/utils.sh
setup_paths

dataset_name=${dataset_name:-draper}
pos_weight=${pos_weight:-1.0}
epoch=${epoch:-5}
model_name=${model_name:-microsoft/codebert-base}
tokenizer_name=${tokenizer_name:-microsoft/codebert-base}
model_type=${model_type:-"roberta"}
out_suffix=${out_suffix:-""}
seed=${seed:-"123456"}

conda activate ensemble

python ${CODE_DIR}/run.py \
    --output_dir=${MODELS_DIR}/${model_name##*/}/${dataset_name}_${out_suffix}_seed${seed} \
    --model_type=${model_type} \
    --tokenizer_name=${tokenizer_name} \
    --model_name_or_path=${model_name} \
    --do_train \
    --do_eval \
    --do_test \
    --one_data_file=${DATA_DIR}/${dataset_name}/${dataset_name}_full_dataset.jsonl \
    --epoch $epoch \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --pos_weight $pos_weight \
    --evaluate_during_training \
    --dropout_probability 0.2 \
    --seed ${seed}  \
    --use_wandb \
    --wandb_project "vulnerability-benchmark" \
    --wandb_run_name "${model_type}_${dataset_name}_pos${pos_weight}_${out_suffix}" \
    2>&1 | tee ${SCRIPTS_DIR}/train.log

