#!/bin/bash

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Not in a git repository. Cannot determine PROJECT_ROOT." >&2
    exit 1
fi

source "${PROJECT_ROOT}/scripts/utils/utils.sh"
setup_paths
