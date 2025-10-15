export model_name=Salesforce/codet5-base
export tokenizer_name=Salesforce/codet5-base
export pos_weight=1.0
export out_suffix=codet5_full-test
export epoch=1
export model_type=codet5_full

echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name"

sbatch_args() {
    local job_name=$1
    echo "--gpus-per-node=1 --partition=aoraki_gpu --mem=64gb --job-name=${job_name}_${out_suffix} --time=70:00:00 --output=/projects/sciences/computing/norja159/research/scripts/${job_name}_out/${model_type}_%j.out"
}

export dir=icvul
sbatch $(sbatch_args icvul) test_split.sh
