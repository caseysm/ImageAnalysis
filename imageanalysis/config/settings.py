"""Configuration settings for the imageanalysis package."""

import os
from pathlib import Path

# Default base directory for results
DEFAULT_OUTPUT_DIR = "results"

# Default configuration file name
DEFAULT_CONFIG_NAME = "config.json"

# Default segmentation parameters
DEFAULT_SEGMENTATION = {
    "nuclei_diameter": 30,
    "cell_diameter": 60,
    "threshold_std": 3.0,
    "min_cell_size": 100,
    "max_cell_size": 5000
}

# Default genotyping parameters
DEFAULT_GENOTYPING = {
    "peak_threshold": 0.5,
    "min_peak_distance": 3,
    "prominence": 0.2
}

# Default phenotyping parameters
DEFAULT_PHENOTYPING = {
    "intensity_features": True,
    "morphology_features": True,
    "texture_features": False
}

# Default channel mapping
DEFAULT_CHANNELS = {
    "nuclei": "DAPI",
    "cytoplasm": "mClov3",
    "marker": "TMR"
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "file_format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    "console_format": "[%(levelname)s] %(message)s"
}

# File extensions
SUPPORTED_IMAGE_FORMATS = [".nd2", ".tif", ".tiff"]
SUPPORTED_MASK_FORMATS = [".npy", ".tif", ".tiff"]
SUPPORTED_RESULTS_FORMATS = [".csv", ".json"]