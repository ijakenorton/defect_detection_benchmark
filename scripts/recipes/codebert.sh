
# Slurm batch recipe for graphcodebert
export model_name=microsoft/codebert-base
export tokenizer_name=microsoft/codebert-base
export model_type=roberta
export out_suffix=default
export pos_weight=1.0
export epoch=10

echo "Submitted jobs with model_name: $model_name model_type: $model_type pos: $pos_weight out_suffix $out_suffix epoch: $epoch tokenizer_name: $tokenizer_name"

sbatch_args() {
    local job_name=$1
    echo "--gpus-per-node=1 --partition=aoraki_gpu_H100 --mem=64gb --job-name=${job_name}_${out_suffix} --time=70:00:00 --output=/projects/sciences/computing/norja159/research/scripts/${job_name}_out/%j.out"
}

export dir=icvul
sbatch $(sbatch_args icvul) train.sh
export dir=diversevul
sbatch $(sbatch_args diversevul) train.sh
export dir=mvdsc_mixed
sbatch $(sbatch_args mvdsc_mixed) train.sh
export dir=devign
sbatch $(sbatch_args devign) train.sh
export dir=vuldeepecker
sbatch $(sbatch_args vuldeepecker) train.sh
export dir=cvefixes
sbatch $(sbatch_args cvefixes) train.sh
export dir=draper
sbatch $(sbatch_args draper) train.sh
export dir=juliet
sbatch $(sbatch_args juliet) train.sh
export dir=reveal
sbatch $(sbatch_args reveal) train.sh
