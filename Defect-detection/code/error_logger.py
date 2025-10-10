"""
Centralized error logging for batch SLURM jobs.

This module provides utilities to log errors from multiple SLURM jobs to a centralized
error log file with proper file locking to handle concurrent writes.
"""

import os
import sys
import traceback
import fcntl
from datetime import datetime
from contextlib import contextmanager
from typing import Optional
import socket


class CentralizedErrorLogger:
    """Centralized error logger with file locking for concurrent writes."""

    def __init__(self, log_dir: Optional[str] = None, batch_id: Optional[str] = None):
        """
        Initialize the centralized error logger.

        Args:
            log_dir: Directory to store error logs (default: PROJECT_ROOT/logs/errors)
            batch_id: Batch run identifier (default: timestamp)
        """
        # Find project root (go up from code/ to Defect-detection/ to research/)
        code_dir = os.path.dirname(os.path.abspath(__file__))  # Defect-detection/code
        defect_detection_dir = os.path.dirname(code_dir)  # Defect-detection
        self.project_root = os.path.dirname(defect_detection_dir)  # research

        # Setup log directory
        if log_dir is None:
            log_dir = os.path.join(self.project_root, "logs", "errors")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup batch ID and log file
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_id = batch_id
        self.log_file = os.path.join(self.log_dir, f"batch_run_{batch_id}.log")

        # Get job metadata from environment
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        self.slurm_job_name = os.environ.get("SLURM_JOB_NAME", "unknown")
        self.hostname = socket.gethostname()

    def _get_job_context(self):
        """Extract job context from environment and arguments."""
        context = {
            "slurm_job_id": self.slurm_job_id,
            "slurm_job_name": self.slurm_job_name,
            "hostname": self.hostname,
            "cwd": os.getcwd(),
        }

        # Try to extract model/dataset/seed from command line args
        if len(sys.argv) > 1:
            args_str = " ".join(sys.argv[1:])
            context["command_args"] = args_str

            # Parse common arguments
            for arg in sys.argv:
                if arg.startswith("--model_type"):
                    idx = sys.argv.index(arg)
                    if idx + 1 < len(sys.argv):
                        context["model"] = sys.argv[idx + 1]
                elif arg.startswith("--train_data_file") or arg.startswith("--one_data_file"):
                    idx = sys.argv.index(arg)
                    if idx + 1 < len(sys.argv):
                        # Extract dataset name from path
                        path = sys.argv[idx + 1]
                        dataset = path.split('/')[-2] if '/' in path else "unknown"
                        context["dataset"] = dataset
                elif arg.startswith("--seed"):
                    idx = sys.argv.index(arg)
                    if idx + 1 < len(sys.argv):
                        context["seed"] = sys.argv[idx + 1]

        return context

    def log_error(self, exception: Exception, context: Optional[dict] = None):
        """
        Log an error to the centralized error log with file locking.

        Args:
            exception: The exception that occurred
            context: Additional context information (optional)
        """
        # Get job context
        job_context = self._get_job_context()
        if context:
            job_context.update(context)

        # Format error entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_type = type(exception).__name__
        error_msg = str(exception)
        tb = traceback.format_exc()

        # Build log entry
        log_entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
JOB_ID: {job_context.get('slurm_job_id', 'unknown')}
JOB_NAME: {job_context.get('slurm_job_name', 'unknown')}
HOSTNAME: {job_context.get('hostname', 'unknown')}
MODEL: {job_context.get('model', 'unknown')}
DATASET: {job_context.get('dataset', 'unknown')}
SEED: {job_context.get('seed', 'unknown')}
CWD: {job_context.get('cwd', 'unknown')}
ERROR_TYPE: {error_type}
ERROR_MESSAGE: {error_msg}

TRACEBACK:
{tb}

COMMAND_ARGS:
{job_context.get('command_args', 'N/A')}
{'='*80}

"""

        # Write to file with exclusive lock
        with open(self.log_file, 'a') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(log_entry)
                f.flush()
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Also log to stderr for immediate visibility
        print(f"\n[ERROR LOGGED] Error logged to: {self.log_file}", file=sys.stderr)
        print(f"[ERROR LOGGED] Job ID: {job_context.get('slurm_job_id')}, "
              f"Model: {job_context.get('model')}, "
              f"Dataset: {job_context.get('dataset')}, "
              f"Error: {error_type}: {error_msg}", file=sys.stderr)


@contextmanager
def error_logging_context(log_dir: Optional[str] = None, batch_id: Optional[str] = None):
    """
    Context manager that catches and logs any exceptions.

    Usage:
        with error_logging_context():
            main()  # Your main function

    Args:
        log_dir: Directory to store error logs
        batch_id: Batch run identifier
    """
    logger = CentralizedErrorLogger(log_dir=log_dir, batch_id=batch_id)

    try:
        yield logger
    except Exception as e:
        logger.log_error(e)
        # Re-raise the exception so the program still exits with error
        raise


def get_batch_id_from_env():
    """
    Get batch ID from environment variable, with fallback to timestamp.

    Set BATCH_RUN_ID environment variable before running batch jobs to group them.
    """
    batch_id = os.environ.get("BATCH_RUN_ID")
    if batch_id:
        return batch_id
    # Fallback to timestamp
    return datetime.now().strftime("%Y%m%d_%H%M%S")
