"""Logging utilities for image analysis."""

import logging
import os
from datetime import datetime

def setup_logger(name, log_dir=None, level=logging.INFO):
    """Set up and configure a logger.
    
    Args:
        name (str): Logger name
        log_dir (str, optional): Directory for log files. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_dir specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
