"""Input/output utilities for the imageanalysis package."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import nd2reader
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False
    
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImageLoader:
    """Class for loading and managing image data."""
    
    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """Initialize the image loader.
        
        Args:
            file_path: Optional path to the image file
        """
        self.file_path = None
        self.extension = None
        self.data = None
        self.metadata = {}
        
        if file_path:
            self.set_file(file_path)
    
    def set_file(self, file_path: Union[str, Path]) -> None:
        """Set the file to load.
        
        Args:
            file_path: Path to the image file
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.file_path}")
            
        self.extension = self.file_path.suffix.lower()
        
    def load(self) -> np.ndarray:
        """Load image data.
        
        Returns:
            Image data as numpy array
        """
        if self.file_path is None:
            raise ValueError("No file path set. Call set_file() first.")
            
        if self.extension == '.nd2':
            return self._load_nd2()
        elif self.extension in ['.tif', '.tiff']:
            return self._load_tiff()
        else:
            raise ValueError(f"Unsupported file format: {self.extension}")
            
    def _load_nd2(self) -> np.ndarray:
        """Load ND2 image data.
        
        Returns:
            Image data as numpy array
        """
        if not ND2_AVAILABLE:
            raise ImportError("nd2reader is required to load ND2 files")
            
        logger.debug(f"Loading ND2 file: {self.file_path}")
        
        # Actually load the ND2 file
        with nd2reader.ND2Reader(str(self.file_path)) as nd2_data:
            # Extract dimensions
            if 'z' in nd2_data.sizes:
                z_size = nd2_data.sizes['z']
            else:
                z_size = 1
                
            if 'c' in nd2_data.sizes:
                channels = nd2_data.sizes['c']
            else:
                channels = 1
                
            height = nd2_data.sizes['y']
            width = nd2_data.sizes['x']
            
            # Extract metadata
            self.metadata = {
                'channels': channels,
                'height': height,
                'width': width,
                'z_size': z_size
            }
            
            # Create array to hold data
            if z_size > 1:
                # 3D data (z-stack)
                data = np.empty((z_size, channels, height, width), dtype=np.uint16)
                
                # Load all frames
                for z in range(z_size):
                    for c in range(channels):
                        nd2_data.default_coords['z'] = z
                        nd2_data.default_coords['c'] = c
                        data[z, c] = nd2_data[0]
            else:
                # 2D data
                data = np.empty((channels, height, width), dtype=np.uint16)
                
                # Load all channels
                for c in range(channels):
                    nd2_data.default_coords['c'] = c
                    data[c] = nd2_data[0]
        
        return data
        
    def _load_tiff(self) -> np.ndarray:
        """Load TIFF image data.
        
        Returns:
            Image data as numpy array
        """
        if not TIFF_AVAILABLE:
            raise ImportError("tifffile is required to load TIFF files")
            
        logger.debug(f"Loading TIFF file: {self.file_path}")
        
        # Actually load the TIFF file
        data = tifffile.imread(str(self.file_path))
        
        # Extract metadata
        self.metadata = {
            'shape': data.shape
        }
        
        return data
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get image metadata.
        
        Returns:
            Metadata dictionary
        """
        return self.metadata
        
    def get_wells(self) -> List[str]:
        """Get list of wells in the file.
        
        Returns:
            List of well identifiers
        """
        if not ND2_AVAILABLE:
            raise ImportError("nd2reader is required to extract well information")
            
        wells = []
        
        with nd2reader.ND2Reader(str(self.file_path)) as nd2_data:
            # Check if metadata contains well information
            if hasattr(nd2_data, 'metadata') and 'wells' in nd2_data.metadata:
                wells = [f"Well{w}" for w in nd2_data.metadata['wells']]
            elif hasattr(nd2_data, 'metadata') and 'experiment' in nd2_data.metadata:
                # Try to extract well information from experiment metadata
                if 'wells' in nd2_data.metadata['experiment']:
                    wells = [f"Well{w}" for w in nd2_data.metadata['experiment']['wells']]
                    
        # If no wells found in metadata, try to extract from filename
        if not wells:
            filename = self.file_path.name
            parts = filename.split('_')
            for part in parts:
                if part.startswith('Well'):
                    wells.append(part)
                    break
                    
        # If still no wells found, return a default
        if not wells:
            return ["Well1"]
            
        return wells
        
    def get_wells_and_tiles(self) -> Dict[str, List[str]]:
        """Get wells and tiles from a file.
        
        Returns:
            Dictionary mapping well IDs to lists of tile IDs
        """
        if not ND2_AVAILABLE:
            raise ImportError("nd2reader is required to extract well and tile information")
            
        wells_tiles = {}
        
        with nd2reader.ND2Reader(str(self.file_path)) as nd2_data:
            # Check if metadata contains position information
            if hasattr(nd2_data, 'metadata') and 'points' in nd2_data.metadata:
                points = nd2_data.metadata['points']
                # Get wells from filename if available
                wells = self.get_wells()
                
                for well in wells:
                    wells_tiles[well] = [f"Point{i+1}" for i in range(len(points))]
            else:
                # Try to extract from filename
                filename = self.file_path.name
                parts = filename.split('_')
                well = None
                point = None
                
                for part in parts:
                    if part.startswith('Well'):
                        well = part
                    elif part.startswith('Point'):
                        point = part
                        
                if well and point:
                    wells_tiles[well] = [point]
                else:
                    # Default
                    wells_tiles["Well1"] = ["Point1"]
                    
        return wells_tiles
        
    def get_tiles_for_well(self, well_id: str) -> List[str]:
        """Get list of tiles for a well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            List of tile identifiers
        """
        wells_tiles = self.get_wells_and_tiles()
        
        if well_id in wells_tiles:
            return wells_tiles[well_id]
        else:
            logger.warning(f"Well {well_id} not found in file")
            return []
        
    def load_nd2_image(self, well_id: str, tile_id: str) -> np.ndarray:
        """Load an ND2 image for a specific well and tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Image data as numpy array
        """
        if not ND2_AVAILABLE:
            raise ImportError("nd2reader is required to load ND2 images")
            
        # Find ND2 files for this well and tile
        directory = self.file_path.parent
        pattern = f"{well_id}_{tile_id}_*.nd2"
        matching_files = list(directory.glob(pattern))
        
        if not matching_files:
            logger.warning(f"No ND2 files found for well {well_id}, tile {tile_id}")
            raise FileNotFoundError(f"No files found matching pattern {pattern}")
            
        # Load the first matching file
        image_loader = ImageLoader(matching_files[0])
        return image_loader.load()

def save_image(data: np.ndarray, file_path: Union[str, Path], 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save image data to disk.
    
    Args:
        data: Image data as numpy array
        file_path: Path where to save the image
        metadata: Optional metadata to include
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    extension = file_path.suffix.lower()
    
    if extension in ['.tif', '.tiff']:
        if not TIFF_AVAILABLE:
            raise ImportError("tifffile is required to save TIFF files")
            
        # Save as TIFF
        logger.debug(f"Saving TIFF file: {file_path}")
        tifffile.imwrite(str(file_path), data, metadata=metadata)
    elif extension in ['.npy']:
        # Save as NumPy array
        np.save(file_path, data)
    else:
        raise ValueError(f"Unsupported output format: {extension}")