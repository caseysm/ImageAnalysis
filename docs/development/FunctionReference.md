# Function Reference

## Core Modules

### Segmentation (`core/segmentation/`)

#### CellSegmentation (`cell_segmentation.py`)
```python
def segment_nuclei(
    image: np.ndarray,
    nuclei_diameter: int,
    min_area: int = 100
) -> np.ndarray:
    """Segment nuclei from DAPI channel image.
    
    Args:
        image: DAPI channel image
        nuclei_diameter: Expected diameter of nuclei in pixels
        min_area: Minimum area in pixels for nuclei filtering
        
    Returns:
        Binary mask of segmented nuclei
    """
```

```python
def segment_cells(
    image: np.ndarray,
    nuclei_mask: np.ndarray,
    cell_diameter: int,
    min_area: int = 400
) -> np.ndarray:
    """Segment cells using membrane channel and nuclear seeds.
    
    Args:
        image: Membrane channel image
        nuclei_mask: Binary mask of nuclei to use as seeds
        cell_diameter: Expected diameter of cells in pixels
        min_area: Minimum area in pixels for cell filtering
        
    Returns:
        Label mask of segmented cells
    """
```

#### Base Pipeline (`base.py`)
```python
def validate_inputs(
    self,
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any]
) -> bool:
    """Validate pipeline inputs and configuration.
    
    Args:
        input_dir: Path to input data directory
        output_dir: Path to output directory
        config: Pipeline configuration dictionary
        
    Returns:
        True if inputs are valid, raises exception otherwise
    """
```

### Genotyping (`core/genotyping/`)

#### Peak Calling (`peak_calling.py`)
```python
def find_peaks(
    image_stack: np.ndarray,
    min_distance: int = 3,
    threshold_rel: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks in sequencing cycles.
    
    Args:
        image_stack: 4D array of sequencing cycles (cycles x channels x height x width)
        min_distance: Minimum distance between peaks
        threshold_rel: Relative threshold for peak detection
        
    Returns:
        Tuple of (peak_coordinates, peak_intensities)
    """
```

```python
def call_bases(
    peak_intensities: np.ndarray,
    threshold_std: float = 3.0,
    lim_low: float = 0.25,
    lim_high: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Call bases from peak intensities.
    
    Args:
        peak_intensities: Array of intensities for each channel at each peak
        threshold_std: Standard deviation threshold for peak quality
        lim_low: Lower limit for base calling
        lim_high: Upper limit for base calling
        
    Returns:
        Tuple of (base_calls, quality_scores)
    """
```

### Phenotyping (`core/phenotyping/`)

#### Metrics (`metrics.py`)
```python
def calculate_area_metrics(
    mask: np.ndarray,
    pixel_size: float = 1.0
) -> Dict[str, float]:
    """Calculate area-based metrics for a cell.
    
    Args:
        mask: Binary mask of the cell
        pixel_size: Physical size of each pixel in microns
        
    Returns:
        Dictionary containing area metrics:
            - area: Total area in square microns
            - perimeter: Perimeter length in microns
            - equivalent_diameter: Diameter of circle with same area
            - major_axis_length: Length of major axis
            - minor_axis_length: Length of minor axis
    """
```

```python
def calculate_intensity_metrics(
    mask: np.ndarray,
    image: np.ndarray,
    background: Optional[float] = None
) -> Dict[str, float]:
    """Calculate intensity metrics for a cell in a given channel.
    
    Args:
        mask: Binary mask of the cell
        image: Intensity image (single channel)
        background: Optional background intensity to subtract
        
    Returns:
        Dictionary containing intensity metrics:
            - mean_intensity: Mean intensity within cell
            - integrated_intensity: Total intensity within cell
            - max_intensity: Maximum intensity within cell
            - min_intensity: Minimum intensity within cell
            - std_intensity: Standard deviation of intensity
    """
```

### Visualization (`core/visualization/`)

#### Albums (`albums.py`)
```python
def create_composite(
    channels: Dict[str, np.ndarray],
    channel_colors: Dict[str, str],
    contrast_limits: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """Create RGB composite from multiple channels.
    
    Args:
        channels: Dictionary mapping channel names to images
        channel_colors: Dictionary mapping channel names to colors
        contrast_limits: Optional dictionary of (min, max) limits per channel
        
    Returns:
        RGB composite image
    """
```

```python
def add_scale_bar(
    image: np.ndarray,
    pixel_size: float,
    bar_length_um: float = 10.0,
    location: str = 'lower right',
    color: str = 'white',
    thickness: int = 2
) -> np.ndarray:
    """Add scale bar to image.
    
    Args:
        image: Input image
        pixel_size: Size of pixels in microns
        bar_length_um: Length of scale bar in microns
        location: Location of scale bar
        color: Color of scale bar
        thickness: Thickness of scale bar in pixels
        
    Returns:
        Image with scale bar added
    """
```

## Utility Functions

### I/O (`utils/io.py`)
```python
def load_nd2_image(
    path: str,
    well: int,
    tile: int
) -> np.ndarray:
    """Load specific well and tile from ND2 file.
    
    Args:
        path: Path to ND2 file
        well: Well number
        tile: Tile number
        
    Returns:
        Image array
    """
```

```python
def save_results(
    data: Dict[str, Any],
    output_path: str,
    file_format: str = 'csv'
) -> None:
    """Save analysis results to file.
    
    Args:
        data: Dictionary of results to save
        output_path: Path to save results
        file_format: Format to save results in ('csv' or 'json')
    """
```

### Parallel Processing (`utils/parallel.py`)
```python
def process_batch(
    func: Callable,
    items: List[Any],
    batch_size: int = 10,
    n_workers: int = 4
) -> List[Any]:
    """Process items in parallel batches.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        batch_size: Number of items per batch
        n_workers: Number of parallel workers
        
    Returns:
        List of processed results
    """
```

### Statistics (`utils/stats.py`)
```python
def calculate_background(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = 'gaussian'
) -> float:
    """Calculate background intensity level.
    
    Args:
        image: Input image
        mask: Optional mask of foreground objects
        method: Method for background calculation
        
    Returns:
        Estimated background intensity
    """
```

## CLI Functions (`cli.py`)
```python
def parse_args(
    args: Optional[List[str]] = None
) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args: Optional list of arguments to parse
        
    Returns:
        Parsed argument namespace
    """
```

```python
def run_pipeline(
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """Run complete analysis pipeline.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance for output
    """
```

## Configuration Functions (`config/settings.py`)
```python
def load_config(
    config_path: str
) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
```

```python
def validate_config(
    config: Dict[str, Any]
) -> bool:
    """Validate configuration values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
```

## Notes

1. All functions include proper error handling and logging
2. Type hints are used throughout for better IDE support
3. Documentation follows Google style guide
4. Optional parameters have sensible defaults
5. Return types are clearly specified
6. Complex operations include progress tracking 