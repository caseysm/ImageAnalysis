"""Base pipeline class for image analysis pipelines."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

class Pipeline(ABC):
    """Abstract base class for all image analysis pipelines.
    
    This class defines the common interface and functionality that all pipeline
    implementations must provide. It handles basic setup like logging, input
    validation, and configuration management.
    
    Attributes:
        input_file: Path to the input ND2 file
        output_dir: Directory for pipeline outputs
        logger: Logger instance for this pipeline
        config: Optional configuration dictionary
    """
    
    def __init__(
        self,
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        config_file: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO
    ):
        """Initialize the pipeline.
        
        Args:
            input_file: Path to input ND2 file
            output_dir: Directory for pipeline outputs
            config_file: Optional path to JSON configuration file
            log_level: Logging level (default: INFO)
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.config = {}
        
        # Load configuration if provided
        if config_file is not None:
            self.load_config(config_file)
            
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_config(self, config_file: Union[str, Path]) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Raises:
            ValueError: If config file doesn't exist or is invalid
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
            
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Set up logging for the pipeline.
        
        Args:
            log_level: Logging level to use
            
        Returns:
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        log_file = self.output_dir / f"{self.__class__.__name__}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def validate_inputs(self) -> None:
        """Validate pipeline inputs.
        
        This method performs basic validation that applies to all pipelines.
        Subclasses should call super().validate_inputs() and then perform
        their own specific validation.
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Check input file exists
        if not self.input_file.exists():
            raise ValueError(f"Input file not found: {self.input_file}")
            
        # Check input file is readable
        if not os.access(self.input_file, os.R_OK):
            raise ValueError(f"Input file is not readable: {self.input_file}")
            
        # Check output directory is writable
        if not os.access(self.output_dir.parent, os.W_OK):
            raise ValueError(f"Output directory is not writable: {self.output_dir}")
            
    def save_config(self) -> None:
        """Save current configuration to output directory."""
        config_file = self.output_dir / "pipeline_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save pipeline metadata.
        
        Args:
            metadata: Dictionary of metadata to save
        """
        metadata_file = self.output_dir / "pipeline_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> None:
        """Run the pipeline.
        
        This method must be implemented by all pipeline subclasses.
        It should contain the main logic for executing the pipeline.
        
        Args:
            *args: Positional arguments for the pipeline
            **kwargs: Keyword arguments for the pipeline
        """
        pass
        
    def __enter__(self) -> 'Pipeline':
        """Context manager entry.
        
        This allows pipelines to be used with 'with' statements for
        automatic resource cleanup.
        
        Returns:
            Self for context manager
        """
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit.
        
        Performs cleanup operations when pipeline execution is complete.
        
        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        # Close log handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
            
        # Log any unhandled exceptions
        if exc_type is not None:
            self.logger.error(
                "Unhandled exception during pipeline execution",
                exc_info=(exc_type, exc_val, exc_tb)
            ) 