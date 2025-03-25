"""Logging configuration for the ImageAnalysis package."""

import logging
import os
from pathlib import Path
from typing import Optional

from ..config.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE, RESULTS_DIR

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files (defaults to RESULTS_DIR)
        level: Logging level (defaults to LOG_LEVEL from settings)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level or LOG_LEVEL)
    
    # Create formatters and handlers
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = log_dir or RESULTS_DIR
    log_path = Path(log_dir) / LOG_FILE
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger 