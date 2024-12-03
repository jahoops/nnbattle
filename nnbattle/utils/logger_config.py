import logging
import sys
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_logger(level=logging.INFO):
    """Configure the root logger with consistent settings."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional: Add file handler
    file_handler = RotatingFileHandler(
        'alphazero_training.log',
        maxBytes=10**6,       # 1MB
        backupCount=5,
        encoding='utf-8',
        delay=True
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger

# Create a global logger instance
logger = setup_logger()

def set_log_level(level=logging.INFO):
    """Set the logging level for the logger."""
    logger.setLevel(level)

# Ensure that the logger is exported
__all__ = ['logger', 'set_log_level']
