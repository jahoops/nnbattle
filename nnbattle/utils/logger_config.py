import logging
import sys

# Create logger
logger = logging.getLogger('alphazero')

def setup_logger(level=logging.INFO):
    """Configure the root logger to output only important messages to console."""
    # Create formatter for important messages only
    formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers
    root_logger.handlers = []

    # Add console handler only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors
    root_logger.addHandler(console_handler)

    return root_logger

# Create a global logger instance
logger = setup_logger()

def set_log_level(level=logging.WARNING):
    """Set the logging level for the logger."""
    logger.setLevel(level)

__all__ = ['logger', 'set_log_level']
