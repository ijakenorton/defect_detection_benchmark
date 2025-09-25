#!/bin/bash

# Check if hf command exists
if ! command -v hf >/dev/null 2>&1; then
    echo "Hugging Face CLI not found."
    echo "it can be installed now with:"
    echo "    pip install -U \"huggingface_hub[cli]\""
    read -p "Install it now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -U "huggingface_hub[cli]"
    else
        echo "Cannot proceed without HF CLI. Exiting."
        exit 1
    fi
fi

datasets=(
    "ijakenorton/cvefixes_for_ml"
    "ijakenorton/devign_for_ml"
    "ijakenorton/diversevul_for_ml"
    "ijakenorton/draper_for_ml"
    "ijakenorton/icvul_for_ml"
    "ijakenorton/juliet_for_ml"
    "ijakenorton/mvdsc_mixed_for_ml"
    "ijakenorton/reveal_for_ml"
    "ijakenorton/vuldeepecker_for_ml"
)

for dataset in "${datasets[@]}"; do
    echo "Downloading $dataset..."
    hf download "$dataset" --repo-type=dataset --force-download --local-dir .
done
