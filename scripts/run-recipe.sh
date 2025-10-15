#!/bin/bash
#
# Wrapper script for easy recipe-like usage of the Python runner.
# This maintains the simplicity of the old bash recipe system while
# using the new Python configuration backend.
#
# Usage:
#   ./run-recipe.sh train_linevul_all        # Run experiment from config/experiments/
#   ./run-recipe.sh train_linevul_all --dry-run  # Preview what would run
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Not in a git repository. Cannot determine PROJECT_ROOT." >&2
    exit 1
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ensemble

# Run the Python orchestrator
python "${SCRIPT_DIR}/runner.py" "$@"
