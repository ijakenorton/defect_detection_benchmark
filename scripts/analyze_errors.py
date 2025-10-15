#!/usr/bin/env python3
"""
Analyze model prediction errors using AST-based pattern detection.

This script integrates with the new configuration system to analyze model
performance patterns across different code structures and vulnerability types.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Add ast_parsing to path
sys.path.insert(0, str(Path(__file__).parent / "ast_parsing"))

from schemas import ConfigLoader
from ast_parsing.error_analysis_script import (
    load_predictions,
    load_ground_truth,
    load_code_samples,
    categorize_predictions,
    setup_treesitter,
    analyze_code_by_category,
)
from ast_parsing.advanced_analysis import (
    analyze_samples,
)


class ErrorAnalyzer:
    """Analyze prediction errors across experiments."""

    def __init__(self, models_dir: Path, data_dir: Path, config_dir: Path):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.config_loader = ConfigLoader(config_dir)
        self.parser = setup_treesitter()

    def find_experiment_results(
        self, model_name: str, dataset_name: str, seed: int, out_suffix: str = "splits"
    ) -> Optional[Path]:
        """
        Find experiment results directory.

        Supports both legacy (model-name-base) and new (model-name) directory structures.
        """
        # Try new structure first
        exp_dir = self.models_dir / model_name / f"{dataset_name}_{out_suffix}_seed{seed}"
        if exp_dir.exists():
            return exp_dir

        # Try legacy structure (using HuggingFace model name)
        models_config = self.config_loader.load_models()
        if model_name in models_config:
            legacy_name = models_config[model_name].model_name.split("/")[-1]
            exp_dir = self.models_dir / legacy_name / f"{dataset_name}_{out_suffix}_seed{seed}"
            if exp_dir.exists():
                return exp_dir

        return None

    def get_dataset_file(self, dataset_name: str) -> Optional[Path]:
        """Get the path to the dataset test file."""
        # Check for the full dataset file
        dataset_file = self.data_dir / dataset_name / f"{dataset_name}_full_dataset.jsonl"
        if dataset_file.exists():
            return dataset_file

        # Also check for anonymized version
        anon_file = self.data_dir / dataset_name / f"{dataset_name}_full_dataset_anonymized.jsonl"
        if anon_file.exists():
            return anon_file

        return None

    def analyze_experiment(
        self,
        model_name: str,
        dataset_name: str,
        seed: int,
        out_suffix: str = "splits",
        advanced: bool = False,
    ) -> Optional[Dict]:
        """
        Analyze a single experiment.

        Args:
            model_name: Model config name (e.g., "codebert", "codet5")
            dataset_name: Dataset name (e.g., "juliet", "devign")
            seed: Random seed used for the experiment
            out_suffix: Output suffix (default: "splits")
            advanced: If True, run advanced vulnerability pattern analysis

        Returns:
            Dictionary with analysis results or None if experiment not found
        """
        # Find experiment directory
        exp_dir = self.find_experiment_results(model_name, dataset_name, seed, out_suffix)
        if not exp_dir:
            print(f"Warning: Experiment not found: {model_name} × {dataset_name} × seed={seed}")
            return None

        # Check for predictions file
        predictions_file = exp_dir / "predictions.txt"
        if not predictions_file.exists():
            print(f"Warning: No predictions file in {exp_dir}")
            return None

        # Get dataset file
        dataset_file = self.get_dataset_file(dataset_name)
        if not dataset_file:
            print(f"Warning: Dataset file not found for {dataset_name}")
            return None

        print(f"\nAnalyzing: {model_name} × {dataset_name} × seed={seed}")
        print(f"  Predictions: {predictions_file}")
        print(f"  Dataset: {dataset_file}")

        # Load data
        predictions = load_predictions(str(predictions_file))
        ground_truth = load_ground_truth(str(dataset_file))
        code_samples = load_code_samples(str(dataset_file))

        # Categorize predictions
        categories = categorize_predictions(predictions, ground_truth)

        # Print category counts
        print(f"\n  Category counts:")
        for cat, indices in categories.items():
            print(f"    {cat}: {len(indices)}")

        # Run analysis
        if advanced:
            print(f"\n  Running advanced vulnerability pattern analysis...")
            results = analyze_samples(code_samples, categories, self.parser)
        else:
            print(f"\n  Running basic code structure analysis...")
            results = analyze_code_by_category(code_samples, categories, self.parser)

        # Add metadata
        results["_metadata"] = {
            "model": model_name,
            "dataset": dataset_name,
            "seed": seed,
            "experiment_dir": str(exp_dir),
            "categories": {cat: len(indices) for cat, indices in categories.items()},
        }

        return results

    def analyze_multiple_experiments(
        self,
        experiments: List[Tuple[str, str, int]],
        out_suffix: str = "splits",
        advanced: bool = False,
    ) -> Dict[str, Dict]:
        """
        Analyze multiple experiments.

        Args:
            experiments: List of (model_name, dataset_name, seed) tuples
            out_suffix: Output suffix
            advanced: If True, run advanced analysis

        Returns:
            Dictionary mapping experiment keys to analysis results
        """
        results = {}

        for model_name, dataset_name, seed in experiments:
            key = f"{model_name}_{dataset_name}_seed{seed}"
            result = self.analyze_experiment(
                model_name, dataset_name, seed, out_suffix, advanced
            )
            if result:
                results[key] = result

        return results

    def compare_models_on_dataset(
        self,
        model_names: List[str],
        dataset_name: str,
        seeds: List[int],
        out_suffix: str = "splits",
        advanced: bool = False,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Returns a summary DataFrame showing average metrics for each model.
        """
        results = []

        for model_name in model_names:
            for seed in seeds:
                analysis = self.analyze_experiment(
                    model_name, dataset_name, seed, out_suffix, advanced
                )
                if analysis:
                    results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        **analysis.get("_metadata", {}).get("categories", {}),
                    })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model prediction errors using AST-based analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single experiment
  python analyze_errors.py --model codebert --dataset juliet --seed 123456

  # Analyze with advanced vulnerability pattern detection
  python analyze_errors.py --model codet5 --dataset devign --seed 123456 --advanced

  # Analyze all experiments for a model/dataset combination
  python analyze_errors.py --model linevul --dataset bigvul --all-seeds

  # Save results to JSON
  python analyze_errors.py --model codebert --dataset juliet --seed 123456 --output results.json
        """,
    )

    # Required arguments
    parser.add_argument("--model", help="Model config name (e.g., codebert, codet5)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., juliet, devign)")

    # Seed options
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, help="Random seed for specific experiment")
    seed_group.add_argument(
        "--all-seeds",
        action="store_true",
        help="Analyze all seeds (123456, 345678, 789012)",
    )

    # Analysis options
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced vulnerability pattern analysis",
    )
    parser.add_argument(
        "--out-suffix",
        default="splits",
        help="Output suffix for experiment directories (default: splits)",
    )

    # Output options
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument(
        "--compare-models",
        nargs="+",
        help="Compare multiple models (space-separated list)",
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    config_dir = Path(__file__).parent / "config"

    # Create analyzer
    analyzer = ErrorAnalyzer(models_dir, data_dir, config_dir)

    # Determine seeds to analyze
    if args.all_seeds:
        seeds = [123456, 345678, 789012]
    else:
        seeds = [args.seed]

    # Validate arguments
    if args.compare_models and args.model:
        parser.error("Cannot use both --model and --compare-models")
    if not args.compare_models and not args.model:
        parser.error("Must provide either --model or --compare-models")

    # Run analysis
    if args.compare_models:
        # Compare multiple models
        print(f"Comparing models: {', '.join(args.compare_models)}")
        df = analyzer.compare_models_on_dataset(
            args.compare_models,
            args.dataset,
            seeds,
            args.out_suffix,
            args.advanced,
        )
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
    else:
        # Analyze single model
        results = {}
        for seed in seeds:
            result = analyzer.analyze_experiment(
                args.model, args.dataset, seed, args.out_suffix, args.advanced
            )
            if result:
                results[f"seed_{seed}"] = result

        if not results:
            print("\nNo results found!")
            return 1

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
