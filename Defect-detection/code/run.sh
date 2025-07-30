#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --job-name=test
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 

conda activate ensemble

python run.py \
    --output_dir=./saved_models_seed_42069 \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 42069  2>&1 | tee train.log
