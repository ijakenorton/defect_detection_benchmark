SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export SCRIPTS_DIR="$SCRIPT_DIR"
export OUTPUT_DIR="$PROJECT_ROOT/output"
export DATA_DIR="$PROJECT_ROOT/data"
export MODELS_DIR="$PROJECT_ROOT/models"

echo $SCRIPTS_DIR
echo $OUTPUT_DIR
echo $DATA_DIR
echo $MODELS_DIR

