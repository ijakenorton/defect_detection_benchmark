#!/usr/bin/env python3
"""
Test suite for the vulnerability detection pipeline.

Tests model loading, dataset loading, and minimal training runs to catch
configuration errors before running full experiments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import tempfile
import shutil

from schemas import ConfigLoader


class PipelineTest:
    """Test runner for the vulnerability detection pipeline."""

    def __init__(self, config_dir: Path, data_dir: Path, code_dir: Path):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.code_dir = code_dir
        self.config_loader = ConfigLoader(config_dir)
        self.results = {
            "config_validation": {},
            "dataset_loading": {},
            "model_loading": {},
            "smoke_training": {}
        }

    def test_config_validation(self) -> bool:
        """Test that all config files are valid."""
        print("\n" + "="*80)
        print("TEST: Configuration Validation")
        print("="*80)

        all_passed = True

        # Test models.json
        print("\nTesting models.json...")
        try:
            models = self.config_loader.load_models()
            print(f"  ✓ Loaded {len(models)} models")
            self.results["config_validation"]["models"] = {
                "status": "pass",
                "count": len(models),
                "models": list(models.keys())
            }
        except Exception as e:
            print(f"  ✗ Failed to load models.json: {e}")
            self.results["config_validation"]["models"] = {
                "status": "fail",
                "error": str(e)
            }
            all_passed = False

        # Test datasets.json
        print("\nTesting datasets.json...")
        try:
            datasets = self.config_loader.load_datasets()
            print(f"  ✓ Loaded {len(datasets)} datasets")
            self.results["config_validation"]["datasets"] = {
                "status": "pass",
                "count": len(datasets),
                "datasets": list(datasets.keys())
            }
        except Exception as e:
            print(f"  ✗ Failed to load datasets.json: {e}")
            self.results["config_validation"]["datasets"] = {
                "status": "fail",
                "error": str(e)
            }
            all_passed = False

        # Test experiment configs
        print("\nTesting experiment configs...")
        experiment_files = list((self.config_dir / "experiments").glob("*.json"))
        passed = 0
        failed = 0
        for exp_file in experiment_files:
            try:
                exp = self.config_loader.load_experiment(exp_file)
                print(f"  ✓ {exp_file.name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {exp_file.name}: {e}")
                failed += 1
                all_passed = False

        self.results["config_validation"]["experiments"] = {
            "status": "pass" if failed == 0 else "partial" if passed > 0 else "fail",
            "passed": passed,
            "failed": failed,
            "total": len(experiment_files)
        }

        return all_passed

    def test_dataset_loading(self, datasets: Optional[List[str]] = None) -> bool:
        """Test that all datasets can be loaded."""
        print("\n" + "="*80)
        print("TEST: Dataset Loading")
        print("="*80)

        all_datasets = self.config_loader.load_datasets()

        if datasets:
            datasets_to_test = {k: v for k, v in all_datasets.items() if k in datasets}
        else:
            datasets_to_test = all_datasets

        all_passed = True
        for dataset_name, dataset_config in datasets_to_test.items():
            print(f"\nTesting {dataset_name}...")

            # Check if dataset file exists
            dataset_file = self.data_dir / dataset_name / f"{dataset_name}_full_dataset.jsonl"

            if not dataset_file.exists():
                print(f"  ✗ Dataset file not found: {dataset_file}")
                self.results["dataset_loading"][dataset_name] = {
                    "status": "fail",
                    "error": "file_not_found",
                    "path": str(dataset_file)
                }
                all_passed = False
                continue

            # Try to load first few lines
            try:
                with open(dataset_file, 'r') as f:
                    lines_checked = 0
                    for i, line in enumerate(f):
                        if i >= 10:  # Check first 10 lines
                            break
                        data = json.loads(line.strip())

                        # Check for required fields
                        if 'func' not in data:
                            raise ValueError(f"Missing 'func' field on line {i+1}")
                        if 'target' not in data:
                            raise ValueError(f"Missing 'target' field on line {i+1}")

                        lines_checked += 1

                print(f"  ✓ Valid (checked {lines_checked} lines)")
                self.results["dataset_loading"][dataset_name] = {
                    "status": "pass",
                    "path": str(dataset_file),
                    "lines_checked": lines_checked
                }

            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
                self.results["dataset_loading"][dataset_name] = {
                    "status": "fail",
                    "error": str(e),
                    "path": str(dataset_file)
                }
                all_passed = False

        return all_passed

    def test_model_loading(self, models: Optional[List[str]] = None) -> bool:
        """Test that all models can be initialized."""
        print("\n" + "="*80)
        print("TEST: Model Loading (Dry Run)")
        print("="*80)

        all_models = self.config_loader.load_models()

        if models:
            models_to_test = {k: v for k, v in all_models.items() if k in models}
        else:
            models_to_test = all_models

        print("\nNote: This tests that model configs are valid and can be loaded by run.py")
        print("It does NOT download or initialize the actual models.\n")

        all_passed = True
        for model_name, model_config in models_to_test.items():
            print(f"Testing {model_name}...")
            print(f"  Model type: {model_config.model_type}")
            print(f"  Model path: {model_config.model_name}")
            print(f"  Tokenizer: {model_config.tokenizer_name}")

            # Validate required fields
            try:
                if not model_config.model_type:
                    raise ValueError("Missing model_type")
                if not model_config.model_name:
                    raise ValueError("Missing model_name")
                if not model_config.tokenizer_name:
                    raise ValueError("Missing tokenizer_name")

                print(f"  ✓ Configuration valid")
                self.results["model_loading"][model_name] = {
                    "status": "pass",
                    "model_type": model_config.model_type,
                    "model_name": model_config.model_name
                }

            except Exception as e:
                print(f"  ✗ Configuration invalid: {e}")
                self.results["model_loading"][model_name] = {
                    "status": "fail",
                    "error": str(e)
                }
                all_passed = False

        return all_passed

    def test_smoke_training(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        use_sbatch: bool = False,
        dry_run: bool = True
    ) -> bool:
        """Run minimal training test (1 epoch) for each model/dataset combo."""
        print("\n" + "="*80)
        print("TEST: Smoke Training (1 epoch)")
        print("="*80)

        if dry_run:
            print("\n⚠ DRY RUN MODE - will not actually train\n")

        # Create a temporary test config
        all_models = self.config_loader.load_models()
        all_datasets = self.config_loader.load_datasets()

        test_models = models if models else list(all_models.keys())
        test_datasets = datasets if datasets else list(all_datasets.keys())

        test_config = {
            "models": test_models,
            "datasets": test_datasets,
            "seeds": [123456],
            "pos_weight": 1.0,
            "epoch": 1,
            "out_suffix": "smoke_test",
            "mode": "train"
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f, indent=2)
            temp_config_path = f.name

        try:
            # Run with runner.py
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "runner.py"),
                temp_config_path
            ]

            if dry_run:
                cmd.append("--dry-run")
            if not use_sbatch:
                cmd.append("--no-sbatch")

            print(f"\nRunning: {' '.join(cmd)}\n")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✓ Smoke test command succeeded")
                print(f"\n{result.stdout}")

                self.results["smoke_training"] = {
                    "status": "pass",
                    "models": test_models,
                    "datasets": test_datasets,
                    "command": ' '.join(cmd)
                }
                return True
            else:
                print("✗ Smoke test command failed")
                print(f"\nSTDOUT:\n{result.stdout}")
                print(f"\nSTDERR:\n{result.stderr}")

                self.results["smoke_training"] = {
                    "status": "fail",
                    "error": result.stderr,
                    "command": ' '.join(cmd)
                }
                return False

        except subprocess.TimeoutExpired:
            print("✗ Smoke test timed out")
            self.results["smoke_training"] = {
                "status": "fail",
                "error": "timeout"
            }
            return False

        except Exception as e:
            print(f"✗ Smoke test failed: {e}")
            self.results["smoke_training"] = {
                "status": "fail",
                "error": str(e)
            }
            return False

        finally:
            # Cleanup temp file
            Path(temp_config_path).unlink()

    def generate_report(self, output_file: Optional[Path] = None) -> None:
        """Generate a test report."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        # Count results
        total_tests = 0
        passed_tests = 0

        for category, tests in self.results.items():
            if not tests:
                continue

            print(f"\n{category.upper().replace('_', ' ')}:")

            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    if isinstance(result, dict) and 'status' in result:
                        total_tests += 1
                        status = result['status']

                        if status == 'pass':
                            print(f"  ✓ {test_name}")
                            passed_tests += 1
                        elif status == 'partial':
                            print(f"  ⚠ {test_name} (partial)")
                            passed_tests += 0.5
                        else:
                            print(f"  ✗ {test_name}")
                            if 'error' in result:
                                print(f"    Error: {result['error']}")

        print("\n" + "="*80)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        print("="*80)

        # Save detailed results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test suite for vulnerability detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests (dry run)
  python test_runner.py --all

  # Test specific models and datasets
  python test_runner.py --models codebert codet5 --datasets juliet devign

  # Run actual smoke training (1 epoch)
  python test_runner.py --smoke-train --no-dry-run

  # Full test suite with output
  python test_runner.py --all --output test_results.json
        """
    )

    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--config", action="store_true", help="Test config validation")
    parser.add_argument("--datasets-test", action="store_true", help="Test dataset loading")
    parser.add_argument("--models-test", action="store_true", help="Test model configs")
    parser.add_argument("--smoke-train", action="store_true", help="Run smoke training test")

    # Filters
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to test")

    # Options
    parser.add_argument("--no-dry-run", action="store_true", help="Actually run training (not just dry run)")
    parser.add_argument("--use-sbatch", action="store_true", help="Use sbatch for smoke tests")
    parser.add_argument("--output", type=Path, help="Save detailed results to JSON file")

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    config_dir = Path(__file__).parent / "config"
    data_dir = project_root / "data"
    code_dir = project_root / "Defect-detection" / "code"

    # Create tester
    tester = PipelineTest(config_dir, data_dir, code_dir)

    # Determine which tests to run
    if args.all:
        run_all = True
    else:
        run_all = not any([args.config, args.datasets_test, args.models_test, args.smoke_train])

    # Run tests
    all_passed = True

    if run_all or args.config:
        passed = tester.test_config_validation()
        all_passed = all_passed and passed

    if run_all or args.datasets_test:
        passed = tester.test_dataset_loading(args.datasets)
        all_passed = all_passed and passed

    if run_all or args.models_test:
        passed = tester.test_model_loading(args.models)
        all_passed = all_passed and passed

    if run_all or args.smoke_train:
        passed = tester.test_smoke_training(
            models=args.models,
            datasets=args.datasets,
            use_sbatch=args.use_sbatch,
            dry_run=not args.no_dry_run
        )
        all_passed = all_passed and passed

    # Generate report
    tester.generate_report(args.output)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
