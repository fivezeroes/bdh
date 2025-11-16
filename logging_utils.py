# Copyright Pathway Technology, Inc.

"""Logging utilities for capturing stdout to file."""

import sys
from datetime import datetime


class TeeLogger:
    """Logger that writes to both stdout/stderr and a file."""
    
    def __init__(self, log_file_path, original_stream):
        """
        Initialize the TeeLogger.
        
        Args:
            log_file_path: Path to the log file
            original_stream: Original stream (stdout or stderr)
        """
        self.terminal = original_stream
        self.log_file = open(log_file_path, 'a', buffering=1)  # Line buffered
    
    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        self.log_file.close()
    
    def isatty(self):
        """Return terminal's isatty status."""
        return self.terminal.isatty()


def setup_logging(run_dir):
    """
    Set up logging to capture all stdout and stderr to a file.
    
    Args:
        run_dir: Directory where the log file will be created
        
    Returns:
        Tuple of (stdout_logger, stderr_logger)
    """
    import os
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "training.log")
    
    # Write header to log file
    with open(log_path, 'a', buffering=1) as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
    
    # Create tee loggers for both stdout and stderr
    stdout_logger = TeeLogger(log_path, sys.stdout)
    stderr_logger = TeeLogger(log_path, sys.stderr)
    
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger
    
    return stdout_logger, stderr_logger


def teardown_logging(loggers):
    """
    Restore stdout/stderr and close the log files.
    
    Args:
        loggers: Tuple of (stdout_logger, stderr_logger)
    """
    if loggers is not None:
        stdout_logger, stderr_logger = loggers
        
        # Write footer to log file
        stdout_logger.log_file.write(f"\n{'='*80}\n")
        stdout_logger.log_file.write(f"Training session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        stdout_logger.log_file.write(f"{'='*80}\n")
        
        # Restore original streams
        sys.stdout = stdout_logger.terminal
        sys.stderr = stderr_logger.terminal
        
        # Close log files
        stdout_logger.close()
        stderr_logger.close()
