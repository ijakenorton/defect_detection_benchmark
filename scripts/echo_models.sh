for model_config in ./model_configs/*.sh; do
    source "$model_config"

    echo model_name: "$model_name" 
    echo tokenizer_name: "$tokenizer_name" 
    echo model_type: "$model_type"
done
