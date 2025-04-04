"""Base class for all pipeline components."""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from imageanalysis.utils.logging import setup_logger


class Pipeline(ABC):
    """Base class for all pipeline components.
    
    This abstract base class provides common functionality for all pipeline
    components, including input validation, logging, and configuration.
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
            input_file: Path to input file
            output_dir: Path to output directory
            config_file: Optional path to configuration file
            log_level: Logging level
        """
        # Convert paths to Path objects
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.config_file = Path(config_file) if config_file else None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(
            self.__class__.__name__,
            level=log_level,
            log_file=self.output_dir / f"{self.__class__.__name__}.log"
        )
        
        # Load configuration if provided
        self.config = {}
        if self.config_file and self.config_file.exists():
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file.
        
        Raises:
            FileNotFoundError: If config file does not exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        if not self.config_file:
            return
            
        self.logger.info(f"Loading configuration from {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                
            self.logger.debug(f"Loaded configuration: {self.config}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_file}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid configuration file: {self.config_file}")
            raise
    
    def validate_inputs(self) -> None:
        """Validate pipeline inputs.
        
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If inputs are invalid
        """
        # Check input file
        if not self.input_file.exists():
            self.logger.error(f"Input file does not exist: {self.input_file}")
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
    
    @abstractmethod
    def run(self) -> None:
        """Run the pipeline.
        
        This method must be implemented by subclasses.
        """
        pass
    
    def save_output(self, data: Dict[str, Any], file_name: str) -> Path:
        """Save pipeline output to a file.
        
        Args:
            data: Data to save
            file_name: Name of the output file
            
        Returns:
            Path to the saved file
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        output_path = self.output_dir / file_name
        
        # Save based on file extension
        extension = output_path.suffix.lower()
        
        if extension == '.json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            self.logger.warning(f"Unsupported output format: {extension}")
            raise ValueError(f"Unsupported output format: {extension}")
            
        self.logger.info(f"Saved output to {output_path}")
        
        return output_path