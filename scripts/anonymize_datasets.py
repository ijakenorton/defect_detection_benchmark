#!/usr/bin/env python3
"""
Anonymize all datasets in the data directory.
Processes each dataset's full_dataset.jsonl file and creates an anonymized version.

Now uses the config system to automatically discover all datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add ast_parsing to path
sys.path.insert(0, str(Path(__file__).parent / "ast_parsing"))

from anonymize_code import anonymize_code
from schemas import ConfigLoader


# Default paths
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_DIR = Path(__file__).parent / "config"


def anonymize_dataset(dataset_name, code_field="func", position=0):
    """
    Anonymize a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "icvul")
        code_field: Field name containing the code (default: "func")
        position: Progress bar position for parallel processing

    Returns:
        Tuple of (dataset_name, success, anonymized_count, error_count, error_messages, output_file)
    """
    input_file = DATA_DIR / dataset_name / f"{dataset_name}_full_dataset.jsonl"
    output_file = DATA_DIR / dataset_name / f"{dataset_name}_full_dataset_anonymized.jsonl"

    if not input_file.exists():
        return (dataset_name, False, 0, 0, [f"Input file not found: {input_file}"], None)

    anonymized_count = 0
    error_count = 0
    error_messages = []

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(tqdm(infile, desc=f"{dataset_name}", position=position), 1):
            try:
                data = json.loads(line.strip())

                # Check if the code field exists
                if code_field not in data:
                    error_messages.append(f"Line {line_num}: '{code_field}' field not found")
                    error_count += 1
                    continue

                original_code = data[code_field]

                # Anonymize the code
                try:
                    anonymized = anonymize_code(
                        original_code,
                        remove_comments=True,
                        anonymize_functions=True,
                        anonymize_variables=False,
                        language_type="cpp"
                    )
                except (AttributeError, TypeError, RecursionError) as e:
                    # Parser failed or recursion limit hit - skip this entry
                    error_type = type(e).__name__
                    error_messages.append(f"Line {line_num}: {error_type}")
                    error_count += 1
                    continue

                # Update the data with anonymized code
                data[code_field] = anonymized

                # Write to output file
                outfile.write(json.dumps(data) + '\n')
                anonymized_count += 1

            except json.JSONDecodeError as e:
                error_messages.append(f"Line {line_num}: JSON decode error")
                error_count += 1
            except Exception as e:
                error_messages.append(f"Line {line_num}: {type(e).__name__}")
                error_count += 1

    return (dataset_name, True, anonymized_count, error_count, error_messages, str(output_file))


def anonymize_dataset_wrapper(args):
    """Wrapper for parallel processing."""
    dataset_name, position = args
    return anonymize_dataset(dataset_name, position=position)


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize datasets using Tree-sitter",
        epilog="By default, processes all datasets from config/datasets.json"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to anonymize (default: all from config)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Anonymization Script")
    print("=" * 60)
    print()

    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return 1

    # Load datasets from config or use provided list
    if args.datasets:
        datasets = args.datasets
        print(f"Using specified datasets: {', '.join(datasets)}")
    else:
        try:
            config_loader = ConfigLoader(CONFIG_DIR)
            datasets_config = config_loader.load_datasets()
            datasets = list(datasets_config.keys())
            print(f"Loaded {len(datasets)} datasets from config/datasets.json")
        except Exception as e:
            print(f"Warning: Could not load datasets from config: {e}")
            print("Falling back to hardcoded list")
            datasets = ["icvul", "mvdsc_mixed", "devign", "vuldeepecker", "cvefixes", "juliet", "reveal"]

    print(f"Datasets to process: {', '.join(datasets)}\n")

    # Determine number of workers
    num_workers = args.workers if args.workers else min(cpu_count(), len(datasets))
    print(f"Processing {len(datasets)} datasets in parallel with {num_workers} workers\n")

    # Process datasets in parallel (with positions for progress bars)
    dataset_args = [(dataset, i) for i, dataset in enumerate(datasets)]
    with Pool(num_workers) as pool:
        results = pool.map(anonymize_dataset_wrapper, dataset_args)

    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60 + "\n")

    total_anonymized = 0
    total_errors = 0
    successful = 0
    failed = 0

    for dataset_name, success, anon_count, err_count, error_msgs, output_file in results:
        if not success:
            print(f"❌ {dataset_name}: FAILED")
            for msg in error_msgs:
                print(f"   {msg}")
            failed += 1
        else:
            print(f"✓ {dataset_name}: {anon_count} anonymized, {err_count} skipped")
            if output_file:
                print(f"  Output: {output_file}")

            # Show first 5 errors if any
            if error_msgs:
                length = len(error_msgs)
                if length < 5:
                    print(f"  Errors:")
                else:
                    print(f"  Errors (showing first 5 of {len(error_msgs)}):")

                for msg in error_msgs[:5]:
                    print(f"    • {msg}")

            total_anonymized += anon_count
            total_errors += err_count
            successful += 1
        print()

    print("=" * 60)
    print(f"TOTAL: {successful}/{len(datasets)} datasets processed successfully")
    print(f"       {total_anonymized} entries anonymized")
    print(f"       {total_errors} entries skipped due to errors")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
