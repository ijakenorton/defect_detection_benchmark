#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 

dir=${dir:-draper}
pos_weight=${pos_weight:-1.0}
epoch=${epoch:-10}
model_name=${model_name:-microsoft/codebert-base}
tokenizer_name=${tokenizer_name:-microsoft/codebert-base}
model_type=${model_type:-"roberta"}
out_suffix=${out_suffix:-""}
seed=${seed:-"123456"}

##conda activate ensemble

python ../Defect-detection/code/run.py \
    --output_dir=../models/${model_name##*/}/${dir}_${out_suffix}_seed${seed} \
    --model_type=${model_type} \
    --tokenizer_name=${tokenizer_name} \
    --model_name_or_path=${model_name} \
    --do_eval \
    --do_test \
    --one_data_file=../data/$dir/${dir}_full_dataset.jsonl \
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
    --wandb_run_name "${model_type}_${dir}_pos${pos_weight}_${out_suffix}" \
    2>&1 | tee train.log

