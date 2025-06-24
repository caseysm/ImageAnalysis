"""Pytest configuration and shared fixtures for the imageanalysis test suite."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Tuple
import pandas as pd

# Add the parent directory to the path to import imageanalysis
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Generate a synthetic microscopy-like image."""
    # Create a 512x512 grayscale image
    image = np.zeros((512, 512), dtype=np.uint16)
    
    # Add some synthetic cells (circular objects)
    np.random.seed(42)
    n_cells = 20
    
    for _ in range(n_cells):
        # Random position
        y, x = np.random.randint(50, 462, size=2)
        # Random radius
        radius = np.random.randint(15, 30)
        # Random intensity
        intensity = np.random.randint(1000, 3000)
        
        # Create circular mask
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = yy**2 + xx**2 <= radius**2
        
        # Add cell to image
        y_min, y_max = max(0, y-radius), min(512, y+radius+1)
        x_min, x_max = max(0, x-radius), min(512, x+radius+1)
        
        mask_y_min = max(0, radius-y)
        mask_y_max = mask.shape[0] - max(0, (y+radius+1) - 512)
        mask_x_min = max(0, radius-x)
        mask_x_max = mask.shape[1] - max(0, (x+radius+1) - 512)
        
        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] * intensity
        )
    
    # Add some noise
    noise = np.random.normal(100, 20, image.shape).astype(np.uint16)
    image = np.clip(image + noise, 0, 4095).astype(np.uint16)
    
    return image


@pytest.fixture
def synthetic_multichannel_image() -> np.ndarray:
    """Generate a synthetic multi-channel microscopy image."""
    # Create 3 channels (e.g., DAPI, GFP, RFP)
    channels = []
    
    for i in range(3):
        np.random.seed(42 + i)
        channel = np.zeros((512, 512), dtype=np.uint16)
        
        # Different number of objects per channel
        n_objects = [30, 15, 10][i]
        
        for _ in range(n_objects):
            y, x = np.random.randint(50, 462, size=2)
            radius = np.random.randint(10, 25)
            intensity = np.random.randint(500, 2000)
            
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = yy**2 + xx**2 <= radius**2
            
            y_min, y_max = max(0, y-radius), min(512, y+radius+1)
            x_min, x_max = max(0, x-radius), min(512, x+radius+1)
            
            mask_y_min = max(0, radius-y)
            mask_y_max = mask.shape[0] - max(0, (y+radius+1) - 512)
            mask_x_min = max(0, radius-x)
            mask_x_max = mask.shape[1] - max(0, (x+radius+1) - 512)
            
            channel[y_min:y_max, x_min:x_max] = np.maximum(
                channel[y_min:y_max, x_min:x_max],
                mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] * intensity
            )
        
        # Add channel-specific noise
        noise = np.random.normal(50, 10, channel.shape).astype(np.uint16)
        channel = np.clip(channel + noise, 0, 4095).astype(np.uint16)
        channels.append(channel)
    
    return np.stack(channels, axis=0)


@pytest.fixture
def sample_segmentation_mask() -> np.ndarray:
    """Generate a sample segmentation mask."""
    mask = np.zeros((512, 512), dtype=np.uint16)
    
    # Add some labeled regions
    np.random.seed(42)
    n_regions = 15
    
    for i in range(1, n_regions + 1):
        y, x = np.random.randint(50, 462, size=2)
        radius = np.random.randint(15, 30)
        
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = yy**2 + xx**2 <= radius**2
        
        y_min, y_max = max(0, y-radius), min(512, y+radius+1)
        x_min, x_max = max(0, x-radius), min(512, x+radius+1)
        
        mask_y_min = max(0, radius-y)
        mask_y_max = circle.shape[0] - max(0, (y+radius+1) - 512)
        mask_x_min = max(0, radius-x)
        mask_x_max = circle.shape[1] - max(0, (x+radius+1) - 512)
        
        # Only add if region doesn't overlap existing labels
        region = mask[y_min:y_max, x_min:x_max]
        circle_region = circle[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        
        if np.all(region[circle_region] == 0):
            region[circle_region] = i
    
    return mask


@pytest.fixture
def sample_barcode_library() -> pd.DataFrame:
    """Generate a sample barcode library DataFrame."""
    np.random.seed(42)
    n_barcodes = 100
    
    data = {
        'Name': [f'Gene_{i:03d}' for i in range(n_barcodes)],
        'Barcode': [''.join(np.random.choice(['A', 'T', 'G', 'C'], 10)) 
                   for _ in range(n_barcodes)],
        'Category': np.random.choice(['Control', 'Target', 'Reference'], n_barcodes)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_config() -> dict:
    """Generate a sample configuration dictionary."""
    return {
        'segmentation': {
            'cell_diameter': 30,
            'flow_threshold': 0.4,
            'cellprob_threshold': 0.0,
            'min_size': 100,
            'max_size': 10000
        },
        'genotyping': {
            'cycles': 10,
            'channels': ['G-ISS', 'T-ISS', 'A-ISS', 'C-ISS'],
            'peak_prominence': 0.1,
            'min_peak_distance': 5
        },
        'phenotyping': {
            'channels': ['DAPI', 'mClov3', 'TMR'],
            'metrics': ['mean_intensity', 'total_intensity', 'area']
        },
        'output': {
            'save_masks': True,
            'save_visualizations': True,
            'save_summary': True
        }
    }


@pytest.fixture
def mock_nd2_file(temp_dir: Path) -> Path:
    """Create a mock ND2 file structure for testing."""
    # For now, we'll create a TIFF file that mimics ND2 structure
    # In real implementation, you might want to use nd2reader mocking
    import tifffile
    
    # Create a multi-page TIFF that simulates ND2 channels
    filename = temp_dir / "test_image.nd2"
    
    # Create 3 channels of synthetic data
    data = np.random.randint(0, 4095, size=(3, 512, 512), dtype=np.uint16)
    
    # Save as TIFF (we'll pretend it's ND2 for testing)
    tifffile.imwrite(str(filename).replace('.nd2', '.tif'), data)
    
    return Path(str(filename).replace('.nd2', '.tif'))


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    

@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level('DEBUG'):
        yield caplog


# Performance testing fixtures
@pytest.fixture
def benchmark_data() -> Tuple[np.ndarray, dict]:
    """Generate larger data for performance benchmarking."""
    # Create a 2048x2048 image with many cells
    image = np.zeros((2048, 2048), dtype=np.uint16)
    
    np.random.seed(42)
    n_cells = 200
    
    positions = []
    for _ in range(n_cells):
        y, x = np.random.randint(50, 1998, size=2)
        radius = np.random.randint(10, 20)
        intensity = np.random.randint(1000, 3000)
        
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = yy**2 + xx**2 <= radius**2
        
        y_min, y_max = max(0, y-radius), min(2048, y+radius+1)
        x_min, x_max = max(0, x-radius), min(2048, x+radius+1)
        
        mask_y_min = max(0, radius-y)
        mask_y_max = mask.shape[0] - max(0, (y+radius+1) - 2048)
        mask_x_min = max(0, radius-x)
        mask_x_max = mask.shape[1] - max(0, (x+radius+1) - 2048)
        
        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] * intensity
        )
        
        positions.append((y, x, radius))
    
    metadata = {
        'n_cells': n_cells,
        'positions': positions,
        'image_shape': image.shape
    }
    
    return image, metadata