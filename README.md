# Defect Detection Benchmark

Framework for benchmarking vulnerability/defect detection models. In my travels around this area of research, there are many different datasets and models used. Some have done benchmarking on several datasets and models. However, it adds a bunch of wasted time to every research project in the space if we all have to duplicate this work.

This project glues together 9 different datasets of different formats and specifications into one consistent jsonl format ready for use with model training, testing and inference code. The base of that code is from `https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection` though it has been edited and expanded to use as a more flexible framework.

This is designed to be forked and extended by the user. It is a base to work from. If you need something, generally you will need to edit the code itself. The hope is that as it is small and mostly contained this will be easy.

## Datasets

All datasets are now available on Hugging Face and will be automatically downloaded:

- **CVEFixes** - Real-world vulnerability fixes from CVE-linked GitHub commits
- **Devign** - Qemu and ffmpeg functions with errors, difficult real-world code
- **DiverseVul** - Real-world vulnerability data from git commits, C/C++
- **Draper** - Large aggregation of C/C++ real-world code from GitHub scraping
- **ICVul** - Collection of vulnerability contributing commits from CVEs linked to GitHub commits
- **Juliet** - NIST's synthetic test suite for C/C++ vulnerabilities, very comprehensive but synthetic
- **MVDSC Mixed** - Mixed formats, some in AST/Graph form, includes Juliet samples
- **Reveal** - Cleaned real-world dataset from Chrome and Debian, JSON format
- **VulDeepecker** - From SARD dataset, C/C++, mix of synthetic and real vulnerabilities

The datasets range from highly synthetic (Juliet) to completely real-world (CVEFixes, DiverseVul) with various complexity levels.

## Models

Currently supports these pre-trained models:
- **CodeBERT** - Microsoft's code-understanding model
- **CodeT5** - Salesforce's text-to-code generation model
- **GraphCodeBERT** - Microsoft's graph-based code model
- **NatGen** - Natural language to code generation model

# Usage

```bash
# Clone or fork this repo
git clone https://github.com/ijakenorton/defect_detection_benchmark
cd defect_detection_benchmark

# Setup environment. Currently using conda.
conda env create -f environment.yml
# There may need to be some messing around with the environment depending on versions.
# The environment has been tested on Rocky Linux 9.2 (Blue Onyx) & Pop-os

# Download all datasets (now from Hugging Face!)
cd data
./download.sh
cd ..

# Train models on all datasets with multiple seeds
cd scripts
./train_all_models_datasets_std.sh

# Or train individual models/datasets
./train.sh <model_config> <dataset> <seed>

# Aggregate results across all experiments
python aggregate_results.py --results_dir ../models --output results_summary.csv
```

## What's Actually Implemented

This framework is now pretty mature and includes:

- **Automated dataset downloading** from Hugging Face
- **Multi-seed training** for robust statistical results
- **Model configuration system** - easy to add new models or tweak hyperparameters
- **Batch job support** - works with SLURM for cluster training
- **Results aggregation** - automatically combines results across seeds and experiments
- **Flexible training scripts** - train individual models/datasets or run the full suite

## Project Structure

```
├── data/                    # Dataset handling and transformation scripts
│   ├── download.sh         # Downloads all datasets from Hugging Face
│   └── */                  # Individual dataset processing scripts
├── scripts/                # Training and evaluation scripts
│   ├── train_all_models_datasets_std.sh  # Train everything
│   ├── train.sh           # Train individual experiments
│   ├── aggregate_results.py  # Combine results across seeds
│   └── model_configs/     # Model hyperparameter configs
├── models/                 # Pre-trained models and outputs
└── Defect-detection/      # Core training/inference code
```

## Advanced Usage

```bash
# Train specific model on specific dataset with custom seed
./scripts/train.sh ./scripts/model_configs/codebert.sh devign 42

# Test trained models
./scripts/test_all_models_datasets_std.sh

# Get detailed results breakdown
python scripts/aggregate_results.py --results_dir models --output detailed_results.csv
```

# Hardware Requirements

Currently I run all the models on H100s with 64gb of RAM. I believe most of the datasets will not need such a heavy duty setup. Draper and DiverseVul are very large datasets and will most likely be more difficult to run on smaller GPUs. Modifying the batch sizes in the model configs may help this though.

The framework automatically handles smaller vs. larger datasets differently - smaller datasets get 20GB GPU jobs while the big ones (DiverseVul, Draper) get 50GB H100 jobs.

I intend to do some testing on what are the minimal specs required for each of the default training options.
