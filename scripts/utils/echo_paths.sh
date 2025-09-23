SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export SCRIPTS_DIR="$SCRIPT_DIR"
export OUTPUT_DIR="$PROJECT_ROOT/output"
export DATA_DIR="$PROJECT_ROOT/data"
export MODELS_DIR="$PROJECT_ROOT/models"

echo "--output=${OUTPUT_DIR}/${job_name}_out/${name}_%j.out"
