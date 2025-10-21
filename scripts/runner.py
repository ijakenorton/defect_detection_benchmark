#!/usr/bin/env python3
"""
Main orchestrator for vulnerability detection experiments.

This script replaces the bash-based recipe system with a Python-based
configuration system that's easier to maintain and extend.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from schemas import ConfigLoader, ExperimentConfig, ModelConfig, DatasetConfig
from find_missing_experiments import (
    get_expected_experiments,
    get_actual_experiments,
    find_missing_experiments
)


class ExperimentRunner:
    """Runs vulnerability detection experiments."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "scripts"
        self.config_dir = self.scripts_dir / "config"
        self.output_dir = self.project_root / "output"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.code_dir = self.project_root / "Defect-detection" / "code"

        self.config_loader = ConfigLoader(self.config_dir)

    def _setup_env(self, model: ModelConfig, dataset: DatasetConfig,
                   experiment: ExperimentConfig, seed: int) -> dict:
        """Setup environment variables for a run."""
        env = os.environ.copy()

        # Path setup
        env["PROJECT_ROOT"] = str(self.project_root)
        env["SCRIPTS_DIR"] = str(self.scripts_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["DATA_DIR"] = str(self.data_dir)
        env["MODELS_DIR"] = str(self.models_dir)
        env["CODE_DIR"] = str(self.code_dir)

        # Model config
        env["model_name"] = model.model_name
        env["tokenizer_name"] = model.tokenizer_name
        env["model_type"] = model.model_type

        # Dataset config
        env["dataset_name"] = dataset.name

        # Experiment config
        env["seed"] = str(seed)
        env["pos_weight"] = str(experiment.pos_weight)
        env["epoch"] = str(experiment.epoch)
        env["out_suffix"] = experiment.out_suffix

        return env

    def _build_python_command(self, model: ModelConfig, dataset: DatasetConfig,
                             experiment: ExperimentConfig, seed: int, model_config_name: str, anonymized: bool = False) -> List[str]:
        """Build the Python command to run."""
        # Use model_config_name (e.g., "codet5" or "codet5-full") to avoid collisions
        # when multiple configs share the same HuggingFace model

        # Include pos_weight in directory name if it's not the default (1.0)
        if experiment.pos_weight != 1.0:
            dir_name = f"{dataset.name}_pos{experiment.pos_weight}_{experiment.out_suffix}_seed{seed}"
        else:
            dir_name = f"{dataset.name}_{experiment.out_suffix}_seed{seed}"

        output_dir = self.models_dir / model_config_name / dir_name

        cmd = [
            "python", str(self.code_dir / "run.py"),
            f"--output_dir={output_dir}",
            f"--model_type={model.model_type}",
            f"--tokenizer_name={model.tokenizer_name}",
            f"--model_name_or_path={model.model_name}",
        ]

        # Add mode flags
        if experiment.mode == "train":
            cmd.extend(["--do_train", "--do_eval", "--do_test"])
        else:
            cmd.append("--do_test")

        dataset_suffix = "full_dataset.jsonl" 

        if anonymized:
            dataset_suffix = "full_dataset_anonymized.jsonl" 
        

        # Add dataset
        data_file = self.data_dir / dataset.name / f"{dataset.name}_{dataset_suffix}"

        cmd.append(f"--one_data_file={data_file}")

        # Add hyperparameters
        cmd.extend([
            f"--epoch={experiment.epoch}",
            f"--block_size={experiment.block_size}",
            f"--train_batch_size={experiment.train_batch_size}",
            f"--eval_batch_size={experiment.eval_batch_size}",
            f"--learning_rate={experiment.learning_rate}",
            f"--max_grad_norm={experiment.max_grad_norm}",
            f"--pos_weight={experiment.pos_weight}",
            f"--dropout_probability={experiment.dropout_probability}",
            f"--seed={seed}",
        ])

        # Evaluation
        if experiment.mode == "train":
            cmd.append("--evaluate_during_training")

        # Wandb
        if experiment.use_wandb:
            wandb_run_name = f"{model.model_type}_{dataset.name}_pos{experiment.pos_weight}_{experiment.out_suffix}"
            cmd.extend([
                "--use_wandb",
                f"--wandb_project={experiment.wandb_project}",
                f"--wandb_run_name={wandb_run_name}",
            ])

        return cmd

    def _build_sbatch_command(self, dataset: DatasetConfig, model_name: str,
                             experiment: ExperimentConfig, seed: int,
                             legacy_mode: bool = False) -> List[str]:
        """Build the sbatch command."""
        job_name = f"{dataset.name}_{experiment.out_suffix}_{seed}"
        output_file = self.scripts_dir / f"{dataset.name}_out" / f"{job_name}_%j.out"

        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True)

        sbatch_args = [
            "sbatch",
            "--gpus-per-node=1",
            f"--partition={dataset.gpu}",
            "--mem=64gb",
            f"--job-name={job_name}",
            f"--time={dataset.time_hours}:00:00",
            f"--output={output_file}",
        ]

        return sbatch_args

    def _build_sbatch_wrap_command(self, model: ModelConfig, dataset: DatasetConfig,
                                   experiment: ExperimentConfig, seed: int, model_config_name: str, anonymized: False = False) -> str:
        """Build a complete sbatch --wrap command for direct Python execution."""
        python_cmd = self._build_python_command(model, dataset, experiment, seed, model_config_name, anonymized)

        # Build the full command with conda activation
        conda_activate = "source ~/miniconda3/etc/profile.d/conda.sh"
        conda_env = "conda activate ensemble"
        python_exec = " ".join(python_cmd)

        # Combine into a single shell command
        wrap_cmd = f"{conda_activate} && {conda_env} && {python_exec}"

        return wrap_cmd

    def run_experiment(self, experiment: ExperimentConfig,
                      use_sbatch: bool = True,
                      legacy_mode: bool = False,
                      dry_run: bool = False,
                      fix_missing: bool = False,
                      anonymized: bool = False) -> None:
        """Run an experiment.

        Args:
            experiment: Experiment configuration
            use_sbatch: Whether to use sbatch for job submission
            legacy_mode: If True (with use_sbatch), use old bash scripts with env vars.
                        If False (with use_sbatch), use sbatch --wrap with direct Python calls.
            dry_run: If True, print commands without executing
            fix_missing: If True, only run experiments that are missing from results directory
            anonymized: If True, run anonymized versions of the datasets. e.g. juliet_full_dataset_anonymized.jsonl
        """
        # Validate
        errors = self.config_loader.validate_experiment(experiment)
        if errors:
            print("Experiment validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

        # Load configs
        models = self.config_loader.load_models()
        datasets = self.config_loader.load_datasets()

        # Determine what to run
        if fix_missing:
            # Load models config for directory mapping
            from find_missing_experiments import load_config_files
            models_config, _ = load_config_files(self.config_dir)

            # Convert ExperimentConfig to dict format for compatibility
            exp_dict = {
                "models": experiment.models,
                "datasets": experiment.datasets,
                "seeds": experiment.seeds,
                "out_suffix": experiment.out_suffix
            }

            # Get expected and actual experiments
            expected = get_expected_experiments(exp_dict)
            print(f"Expected {len(expected)} experiment combinations")

            print(f"Scanning results directory: {self.models_dir}")
            actual = get_actual_experiments(self.models_dir, models_config, experiment.out_suffix)
            print(f"Found {len(actual)} completed experiments")

            # Find missing
            missing = find_missing_experiments(expected, actual)

            if not missing:
                print("\n✓ All experiments already completed!")
                return

            print(f"\nFound {len(missing)} missing experiments")
            runs_to_execute = missing
        else:
            # Generate Cartesian product
            runs_to_execute = [
                (model_name, dataset_name, seed)
                for model_name in experiment.models
                for dataset_name in experiment.datasets
                for seed in experiment.seeds
            ]
            print(f"Running experiment with {len(experiment.models)} model(s), "
                  f"{len(experiment.datasets)} dataset(s), {len(experiment.seeds)} seed(s)")

        mode_desc = "legacy (bash)" if legacy_mode else "direct (Python)"
        if use_sbatch:
            print(f"Execution mode: sbatch [{mode_desc}]")
        else:
            print(f"Execution mode: local")

        if dry_run:
            print("\n=== DRY RUN MODE ===\n")

        # Execute jobs
        job_count = 0
        for model_name, dataset_name, seed in runs_to_execute:
            model = models[model_name]
            dataset = datasets[dataset_name]
            job_count += 1

            if use_sbatch:

                if legacy_mode:
                    # Legacy mode: use bash scripts with env vars
                    env = self._setup_env(model, dataset, experiment, seed)
                    script_name = f"train_split.sh" if experiment.mode == "train" else "test_split.sh"
                    script_path = self.scripts_dir / script_name

                    sbatch_cmd = self._build_sbatch_command(dataset, model_name, experiment, seed, legacy_mode=True)
                    sbatch_cmd.append(str(script_path))

                    if dry_run:
                        print(f"Job {job_count}: {model_name} × {dataset_name} × seed={seed}")
                        print(f"  Command: {' '.join(sbatch_cmd)}")
                        print(f"  Script: {script_path}")
                        print(f"  Env vars: model_name={model.model_name}, dataset_name={dataset_name}, seed={seed}")
                        print()
                    else:
                        subprocess.run(sbatch_cmd, env=env, check=True)
                else:
                    # Direct mode: use sbatch --wrap with Python command
                    sbatch_cmd = self._build_sbatch_command(dataset, model_name, experiment, seed, legacy_mode=False)
                    wrap_cmd = self._build_sbatch_wrap_command(model, dataset, experiment, seed, model_name, anonymized)

                    sbatch_cmd.append("--wrap")
                    sbatch_cmd.append(wrap_cmd)

                    if dry_run:
                        print(f"Job {job_count}: {model_name} × {dataset_name} × seed={seed}")
                        print(f"  sbatch: {' '.join(sbatch_cmd[:-2])}")  # Print sbatch args
                        print(f"  --wrap: {wrap_cmd}")
                        print()
                    else:
                        subprocess.run(sbatch_cmd, check=True)
            else:
                # Run directly with Python (no sbatch)
                env = self._setup_env(model, dataset, experiment, seed)
                python_cmd = self._build_python_command(model, dataset, experiment, seed, model_name, anonymized)

                if dry_run:
                    print(f"Job {job_count}: {model_name} × {dataset_name} × seed={seed}")
                    print(f"  Command: {' '.join(python_cmd)}")
                    print()
                else:
                    # Activate conda environment and run
                    conda_activate = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ensemble"
                    full_cmd = f"{conda_activate} && {' '.join(python_cmd)}"
                    subprocess.run(full_cmd, shell=True, env=env, check=True)

        if not dry_run:
            print(f"Submitted {job_count} job(s)")
        else:
            print(f"=== Would submit {job_count} job(s) ===")


def find_project_root() -> Path:
    """Find the project root using git."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True
    )
    return Path(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Run vulnerability detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution modes:
  Default:      sbatch with direct Python calls (sbatch --wrap)
  --legacy:     sbatch with legacy bash scripts (train_split.sh/test_split.sh)
  --no-sbatch:  Run locally without sbatch

Examples:
  python runner.py train_linevul_all                    # Direct Python via sbatch
  python runner.py train_linevul_all --legacy           # Legacy bash scripts via sbatch
  python runner.py train_linevul_all --dry-run          # Preview commands
  python runner.py train_linevul_all --fix-missing      # Only run missing experiments
  python runner.py train_linevul_all --no-sbatch        # Run locally
        """
    )
    parser.add_argument("experiment", nargs="?", help="Path to experiment config JSON file")
    parser.add_argument("--no-sbatch", action="store_true",
                       help="Run directly without sbatch")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy bash scripts with sbatch (instead of direct Python)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--fix-missing", action="store_true",
                       help="Only run experiments that are missing from results directory")
    parser.add_argument("--anonymized", action="store_true",
                       help="Run anonymized versions of the datasets. e.g. juliet_full_dataset_anonymized.jsonl")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets")

    args = parser.parse_args()

    project_root = find_project_root()
    runner = ExperimentRunner(project_root)

    # List commands
    if args.list_models:
        models = runner.config_loader.load_models()
        print("Available models:")
        for name, config in models.items():
            print(f"  {name}: {config.model_name} ({config.model_type})")
        return

    if args.list_datasets:
        datasets = runner.config_loader.load_datasets()
        print("Available datasets:")
        for name, config in datasets.items():
            print(f"  {name}: {config.size}, {config.time_hours}h on {config.gpu}")
        print(f"\nDataset groups: small, big, all")
        return

    if not args.experiment:
        parser.print_help()
        sys.exit(1)

    # Load and run experiment
    experiment_file = Path(args.experiment)
    if not experiment_file.exists():
        # Try relative to config/experiments
        experiment_file = project_root / "scripts" / "config" / "experiments" / args.experiment
        if not experiment_file.suffix == ".json":
            experiment_file = experiment_file.with_suffix(".json")

    if not experiment_file.exists():
        print(f"Error: Experiment file not found: {args.experiment}", file=sys.stderr)
        sys.exit(1)

    experiment = runner.config_loader.load_experiment(experiment_file)
    runner.run_experiment(
        experiment,
        use_sbatch=not args.no_sbatch,
        legacy_mode=args.legacy,
        dry_run=args.dry_run,
        fix_missing=args.fix_missing,
        anonymized=args.anonymized
    )


if __name__ == "__main__":
    main()
