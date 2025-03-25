"""Centralized configuration settings for the ImageAnalysis package."""

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv('IMAGEANALYSIS_DATA_DIR', str(BASE_DIR / 'data'))
RESULTS_DIR = os.getenv('IMAGEANALYSIS_RESULTS_DIR', str(BASE_DIR / 'results'))

# Segmentation parameters
NUCLEI_DIAMETER_10X = 30
CELL_DIAMETER_10X = 100
NUCLEI_DIAMETER_40X = 60
CELL_DIAMETER_40X = 200

# Quality control parameters
DEFAULT_THRESHOLD_STD = 3
DEFAULT_LIM_LOW = 0.1
DEFAULT_LIM_HIGH = 0.9

# File patterns
ND2_PATTERN = "*.nd2"
NPY_PATTERN = "*.npy"
CSV_PATTERN = "*.csv"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "image_analysis.log"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "segmentation"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "genotyping"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "phenotyping"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "visualization"), exist_ok=True)

# Segmentation settings
SEGMENTATION_10X = {
    # Nuclear segmentation parameters
    'nuclei_diameter': 30,  # pixels
    'nuclei_min_area': 50,  # pixels^2
    'nuclei_max_area': 400,  # pixels^2
    'nuclei_threshold_method': 'otsu',
    'nuclei_flow_threshold': 0.4,
    'nuclei_min_distance': 8,  # pixels
    
    # Cell segmentation parameters
    'cell_diameter': 100,  # pixels
    'cell_min_area': 200,  # pixels^2
    'cell_max_area': 3000,  # pixels^2
    'cell_flow_threshold': 0.2,
    'membrane_thickness': 2,  # pixels
    
    # Quality control
    'min_solidity': 0.8,
    'max_eccentricity': 0.9,
    
    # Processing
    'tile_overlap': 100,  # pixels
    'batch_size': 4,
    'n_jobs': -1  # Use all available cores
}

SEGMENTATION_40X = {
    # Nuclear segmentation parameters
    'nuclei_diameter': 60,  # pixels
    'nuclei_min_area': 500,  # pixels^2
    'nuclei_max_area': 4000,  # pixels^2
    'nuclei_threshold_method': 'otsu',
    'nuclei_flow_threshold': 0.4,
    'nuclei_min_distance': 30,  # pixels
    
    # Cell segmentation parameters
    'cell_diameter': 200,  # pixels
    'cell_min_area': 2000,  # pixels^2
    'cell_max_area': 30000,  # pixels^2
    'cell_flow_threshold': 0.2,
    'membrane_thickness': 5,  # pixels
    
    # Quality control
    'min_solidity': 0.8,
    'max_eccentricity': 0.9,
    
    # Processing
    'tile_overlap': 200,  # pixels
    'batch_size': 2,
    'n_jobs': -1  # Use all available cores
}

# Genotyping settings
GENOTYPING = {
    # Peak calling parameters
    'min_peak_height': 0.2,  # normalized intensity
    'min_peak_distance': 3,  # bases
    'peak_width_range': (1, 5),  # bases
    'peak_rel_height': 0.5,
    'threshold_std': 200,  # Added to match original
    
    # Base calling parameters
    'min_quality_score': 0.8,
    'intensity_threshold': 0.25,  # Updated to match original lim_low
    'lim_high': 0.5,  # Added to match original
    'background_subtraction': True,
    'normalize_intensities': True,
    
    # Barcode assignment
    'max_hamming_distance': 1,
    'min_read_count': 3,
    'max_reads_per_cell': 1000,
    
    # Processing
    'batch_size': 50,
    'n_jobs': -1
}

# Phenotyping settings
PHENOTYPING = {
    # Cell filtering
    'min_cell_size': 100,  # pixels^2
    'max_cell_size': 10000,  # pixels^2
    'min_nuclear_size': 50,  # pixels^2
    'max_nuclear_size': 2000,  # pixels^2
    
    # Intensity measurements
    'background_percentile': 1.0,
    'intensity_normalization': True,
    'background_subtraction': True,
    
    # Texture analysis
    'haralick_distances': [1, 2, 4],
    'haralick_angles': [0, 45, 90, 135],
    'gabor_frequencies': [0.1, 0.2, 0.3, 0.4],
    'gabor_angles': [0, 45, 90, 135],
    
    # Shape analysis
    'shape_metrics': [
        'area',
        'perimeter',
        'circularity',
        'eccentricity',
        'solidity',
        'extent'
    ],
    
    # Processing
    'batch_size': 100,
    'n_jobs': -1
}

# Album generation settings
ALBUMS = {
    # Image parameters
    'window_size': 128,  # pixels
    'grid_size': (5, 10),  # (rows, cols)
    'dpi': 300,
    
    # Channel colors (matplotlib color names)
    'channel_colors': {
        'DAPI': 'blue',
        'GFP': 'green',
        'RFP': 'red',
        'CY5': 'magenta'
    },
    
    # Contrast settings
    'contrast_limits': {
        'DAPI': (0.01, 0.99),  # percentiles
        'GFP': (0.01, 0.99),
        'RFP': (0.01, 0.99),
        'CY5': (0.01, 0.99)
    },
    
    # Scale bar
    'scale_bar_length': 20,  # microns
    'scale_bar_color': 'white',
    'scale_bar_position': 'lower right',
    
    # Text overlay
    'text_color': 'white',
    'font_size': 8,
    'show_metrics': True,
    'show_genotype': True
}

# Physical parameters
PHYSICAL = {
    # Pixel sizes in microns
    'pixel_size_10x': 0.65,
    'pixel_size_40x': 0.1625,
    
    # Well dimensions
    'well_diameter_mm': 6.4,
    'well_height_mm': 11.0,
    
    # Imaging parameters
    'z_step_size': 2.0,  # microns
    'exposure_times': {
        'DAPI': 100,  # ms
        'GFP': 200,
        'RFP': 200,
        'CY5': 200
    }
}

# Default configuration dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    'segmentation': {
        '10x': SEGMENTATION_10X,
        '40x': SEGMENTATION_40X
    },
    'genotyping': GENOTYPING,
    'phenotyping': PHENOTYPING,
    'albums': ALBUMS,
    'physical': PHYSICAL
}

def get_default_config() -> Dict[str, Any]:
    """Get a copy of the default configuration.
    
    Returns:
        Default configuration dictionary
    """
    from copy import deepcopy
    return deepcopy(DEFAULT_CONFIG)

def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values.
    
    Args:
        base_config: Base configuration dictionary
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    from copy import deepcopy
    config = deepcopy(base_config)
    
    def update_recursive(d: Dict[str, Any], u: Dict[str, Any]) -> None:
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                update_recursive(d[k], v)
            else:
                d[k] = v
                
    update_recursive(config, updates)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate segmentation settings
    for mag in ['10x', '40x']:
        seg_config = config.get('segmentation', {}).get(mag, {})
        if not seg_config:
            continue
            
        if seg_config.get('nuclei_diameter', 0) <= 0:
            raise ValueError(f"Invalid nuclei_diameter for {mag}")
            
        if seg_config.get('cell_diameter', 0) <= 0:
            raise ValueError(f"Invalid cell_diameter for {mag}")
            
    # Validate genotyping settings
    geno_config = config.get('genotyping', {})
    if geno_config:
        if not 0 <= geno_config.get('min_quality_score', 0) <= 1:
            raise ValueError("min_quality_score must be between 0 and 1")
            
        if geno_config.get('max_hamming_distance', 0) < 0:
            raise ValueError("max_hamming_distance must be non-negative")
            
    # Validate phenotyping settings
    pheno_config = config.get('phenotyping', {})
    if pheno_config:
        if pheno_config.get('min_cell_size', 0) <= 0:
            raise ValueError("min_cell_size must be positive")
            
        if pheno_config.get('max_cell_size', 0) <= 0:
            raise ValueError("max_cell_size must be positive")
            
    # Validate album settings
    album_config = config.get('albums', {})
    if album_config:
        if any(x <= 0 for x in album_config.get('grid_size', (0, 0))):
            raise ValueError("grid_size dimensions must be positive")
            
        if album_config.get('window_size', 0) <= 0:
            raise ValueError("window_size must be positive")
            
    # Validate physical parameters
    phys_config = config.get('physical', {})
    if phys_config:
        if phys_config.get('pixel_size_10x', 0) <= 0:
            raise ValueError("pixel_size_10x must be positive")
            
        if phys_config.get('pixel_size_40x', 0) <= 0:
            raise ValueError("pixel_size_40x must be positive")

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file and merge with defaults.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid
        ValueError: If configuration values are invalid
    """
    import json
    
    # Load user config
    with open(config_file, 'r') as f:
        user_config = json.load(f)
        
    # Merge with defaults
    config = update_config(get_default_config(), user_config)
    
    # Validate
    validate_config(config)
    
    return config 