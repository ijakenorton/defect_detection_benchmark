#!/usr/bin/env python3
"""
Error aggregation script for batch SLURM runs.

This script scans centralized error logs and SLURM output files to generate
a summary report of all errors that occurred during a batch run.

Usage:
    python aggregate_errors.py [--batch-id BATCH_ID] [--output OUTPUT_FILE]
    python aggregate_errors.py --scan-slurm [SLURM_OUTPUT_DIR]
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
import sys


class ErrorEntry:
    """Represents a single error entry."""

    def __init__(self, timestamp: str, job_id: str, job_name: str,
                 model: str, dataset: str, seed: str,
                 error_type: str, error_msg: str, traceback: str):
        self.timestamp = timestamp
        self.job_id = job_id
        self.job_name = job_name
        self.model = model
        self.dataset = dataset
        self.seed = seed
        self.error_type = error_type
        self.error_msg = error_msg
        self.traceback = traceback

    def __repr__(self):
        return f"ErrorEntry({self.error_type} in {self.model}/{self.dataset})"


def parse_centralized_log(log_file: Path) -> List[ErrorEntry]:
    """Parse a centralized error log file and extract error entries."""
    errors = []

    if not log_file.exists():
        return errors

    with open(log_file, 'r') as f:
        content = f.read()

    # Split by separator
    entries = content.split('=' * 80)

    for entry in entries:
        if not entry.strip():
            continue

        # Parse fields
        timestamp = re.search(r'TIMESTAMP: (.+)', entry)
        job_id = re.search(r'JOB_ID: (.+)', entry)
        job_name = re.search(r'JOB_NAME: (.+)', entry)
        model = re.search(r'MODEL: (.+)', entry)
        dataset = re.search(r'DATASET: (.+)', entry)
        seed = re.search(r'SEED: (.+)', entry)
        error_type = re.search(r'ERROR_TYPE: (.+)', entry)
        error_msg = re.search(r'ERROR_MESSAGE: (.+)', entry)

        # Extract traceback
        tb_match = re.search(r'TRACEBACK:\n(.+?)\n\nCOMAND_ARGS:', entry, re.DOTALL)
        if not tb_match:
            tb_match = re.search(r'TRACEBACK:\n(.+)', entry, re.DOTALL)

        traceback_str = tb_match.group(1).strip() if tb_match else "N/A"

        if all([timestamp, job_id, error_type]):
            error = ErrorEntry(
                timestamp=timestamp.group(1).strip(),
                job_id=job_id.group(1).strip(),
                job_name=job_name.group(1).strip() if job_name else "unknown",
                model=model.group(1).strip() if model else "unknown",
                dataset=dataset.group(1).strip() if dataset else "unknown",
                seed=seed.group(1).strip() if seed else "unknown",
                error_type=error_type.group(1).strip(),
                error_msg=error_msg.group(1).strip() if error_msg else "N/A",
                traceback=traceback_str
            )
            errors.append(error)

    return errors


def scan_slurm_outputs(slurm_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Scan SLURM output files for errors.

    Returns:
        List of tuples: (file_path, job_id, error_snippet)
    """
    errors = []
    error_patterns = [
        r'Error:',
        r'Exception:',
        r'Traceback \(most recent call last\):',
        r'CUDA out of memory',
        r'RuntimeError:',
        r'ValueError:',
        r'KeyError:',
        r'ImportError:',
        r'FileNotFoundError:',
    ]

    for file_path in slurm_dir.rglob("*.out"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for error patterns
            for pattern in error_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Extract job ID from filename
                    job_id_match = re.search(r'_(\d+)\.out$', file_path.name)
                    job_id = job_id_match.group(1) if job_id_match else "unknown"

                    # Get context around error (10 lines)
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line, re.IGNORECASE):
                            start = max(0, i - 3)
                            end = min(len(lines), i + 7)
                            snippet = '\n'.join(lines[start:end])
                            errors.append((file_path, job_id, snippet))
                            break  # Only report first error in file
                    break  # Only match one pattern per file

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

    return errors


def generate_summary(errors: List[ErrorEntry], output_file: Path):
    """Generate a summary report of errors."""
    if not errors:
        with open(output_file, 'w') as f:
            f.write("No errors found!\n")
        return

    # Group errors by type
    by_type = defaultdict(list)
    by_model = defaultdict(list)
    by_dataset = defaultdict(list)

    for error in errors:
        by_type[error.error_type].append(error)
        by_model[error.model].append(error)
        by_dataset[error.dataset].append(error)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ERROR SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Errors: {len(errors)}\n\n")

        # Summary by error type
        f.write("-" * 80 + "\n")
        f.write("ERRORS BY TYPE\n")
        f.write("-" * 80 + "\n")
        for error_type, error_list in sorted(by_type.items(), key=lambda x: -len(x[1])):
            f.write(f"\n{error_type}: {len(error_list)} occurrences\n")
            for err in error_list:
                f.write(f"  - Job {err.job_id}: {err.model}/{err.dataset} (seed {err.seed})\n")
                f.write(f"    Message: {err.error_msg[:100]}...\n" if len(err.error_msg) > 100
                       else f"    Message: {err.error_msg}\n")

        # Summary by model
        f.write("\n" + "-" * 80 + "\n")
        f.write("ERRORS BY MODEL\n")
        f.write("-" * 80 + "\n")
        for model, error_list in sorted(by_model.items(), key=lambda x: -len(x[1])):
            f.write(f"\n{model}: {len(error_list)} errors\n")
            error_types = defaultdict(int)
            for err in error_list:
                error_types[err.error_type] += 1
            for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
                f.write(f"  - {et}: {count}\n")

        # Summary by dataset
        f.write("\n" + "-" * 80 + "\n")
        f.write("ERRORS BY DATASET\n")
        f.write("-" * 80 + "\n")
        for dataset, error_list in sorted(by_dataset.items(), key=lambda x: -len(x[1])):
            f.write(f"\n{dataset}: {len(error_list)} errors\n")
            error_types = defaultdict(int)
            for err in error_list:
                error_types[err.error_type] += 1
            for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
                f.write(f"  - {et}: {count}\n")

        # Detailed errors
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED ERROR LOGS\n")
        f.write("=" * 80 + "\n\n")

        for i, error in enumerate(errors, 1):
            f.write(f"\n[{i}] {error.error_type} - Job {error.job_id}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Timestamp: {error.timestamp}\n")
            f.write(f"Job Name: {error.job_name}\n")
            f.write(f"Model: {error.model}\n")
            f.write(f"Dataset: {error.dataset}\n")
            f.write(f"Seed: {error.seed}\n")
            f.write(f"Error Type: {error.error_type}\n")
            f.write(f"Error Message: {error.error_msg}\n\n")
            f.write(f"Traceback:\n{error.traceback}\n")
            f.write("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate errors from batch SLURM runs")
    parser.add_argument(
        "--batch-id",
        type=str,
        help="Batch run ID to analyze (default: latest)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory containing centralized error logs (default: PROJECT_ROOT/logs/errors)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for summary report (default: error_summary_<batch_id>.txt)"
    )
    parser.add_argument(
        "--scan-slurm",
        action="store_true",
        help="Also scan SLURM output files for errors"
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        help="Directory containing SLURM output files (default: scripts/)"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Setup log directory
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = project_root / "logs" / "errors"

    if not log_dir.exists():
        print(f"Error log directory not found: {log_dir}")
        print("No centralized error logs available.")
        sys.exit(1)

    # Find batch ID
    if args.batch_id:
        batch_id = args.batch_id
        log_file = log_dir / f"batch_run_{batch_id}.log"
        if not log_file.exists():
            print(f"Error log file not found: {log_file}")
            sys.exit(1)
    else:
        # Find latest batch log
        log_files = list(log_dir.glob("batch_run_*.log"))
        if not log_files:
            print(f"No batch error logs found in {log_dir}")
            sys.exit(1)
        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        batch_id = log_file.stem.replace("batch_run_", "")
        print(f"Using latest batch log: {log_file.name}")

    # Parse centralized error log
    print(f"Parsing centralized error log: {log_file}")
    errors = parse_centralized_log(log_file)
    print(f"Found {len(errors)} errors in centralized log")

    # Optionally scan SLURM outputs
    if args.scan_slurm:
        slurm_dir = args.slurm_dir if args.slurm_dir else script_dir
        print(f"\nScanning SLURM output files in: {slurm_dir}")
        slurm_errors = scan_slurm_outputs(slurm_dir)
        print(f"Found {len(slurm_errors)} potential errors in SLURM output files")

        if slurm_errors:
            print("\nSLURM Output Errors:")
            for file_path, job_id, snippet in slurm_errors:
                print(f"\n  File: {file_path.name} (Job {job_id})")
                print(f"  Preview:\n{snippet[:200]}...")

    # Generate summary
    if args.output:
        output_file = args.output
    else:
        output_file = script_dir / f"error_summary_{batch_id}.txt"

    print(f"\nGenerating summary report: {output_file}")
    generate_summary(errors, output_file)
    print(f"\nSummary report written to: {output_file}")

    # Print quick summary to console
    if errors:
        print("\nQuick Summary:")
        print("-" * 40)
        error_types = defaultdict(int)
        for err in errors:
            error_types[err.error_type] += 1
        for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {et}: {count}")


if __name__ == "__main__":
    main()
