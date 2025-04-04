"""Core cell segmentation algorithms."""

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology
from typing import Tuple, Optional

class CellSegmentation:
    """Class containing core cell segmentation algorithms."""
    
    def __init__(
        self,
        nuclei_diameter: int = 30,
        cell_diameter: int = 100,
        threshold_std: float = 3.0
    ):
        """Initialize the cell segmentation class.
        
        Args:
            nuclei_diameter: Expected diameter of nuclei in pixels
            cell_diameter: Expected diameter of cells in pixels
            threshold_std: Standard deviations for thresholding
        """
        self.nuclei_diameter = nuclei_diameter
        self.cell_diameter = cell_diameter
        self.threshold_std = threshold_std
    
    def segment_nuclei(
        self,
        image: np.ndarray,
        nuclear_channel: int = 0
    ) -> np.ndarray:
        """Segment nuclei from an image.
        
        Args:
            image: Input image array
            nuclear_channel: Channel index containing nuclear signal
            
        Returns:
            Binary mask of segmented nuclei
        """
        # Extract nuclear channel
        if len(image.shape) > 2:
            nuclear_image = image[..., nuclear_channel]
        else:
            nuclear_image = image
            
        # Apply Gaussian filter
        sigma = self.nuclei_diameter / 4
        filtered = filters.gaussian(nuclear_image, sigma=sigma)
        
        # Threshold
        threshold = filters.threshold_local(filtered)
        binary = filtered > threshold
        
        # Clean up
        binary = morphology.remove_small_objects(binary, min_size=self.nuclei_diameter**2)
        binary = morphology.remove_small_holes(binary, area_threshold=self.nuclei_diameter**2)
        
        return binary
    
    def segment_cells(
        self,
        image: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        cell_channel: int = 1
    ) -> np.ndarray:
        """Segment cells from an image, optionally using nuclear guidance.
        
        Args:
            image: Input image array
            nuclear_mask: Optional binary mask of nuclei for guidance
            cell_channel: Channel index containing cell signal
            
        Returns:
            Binary mask of segmented cells
        """
        # Extract cell channel
        if len(image.shape) > 2:
            cell_image = image[..., cell_channel]
        else:
            cell_image = image
            
        # Apply Gaussian filter
        sigma = self.cell_diameter / 4
        filtered = filters.gaussian(cell_image, sigma=sigma)
        
        # Threshold
        threshold = filters.threshold_local(filtered)
        binary = filtered > threshold
        
        # Clean up
        binary = morphology.remove_small_objects(binary, min_size=self.cell_diameter**2)
        binary = morphology.remove_small_holes(binary, area_threshold=self.cell_diameter**2)
        
        # Use nuclear guidance if provided
        if nuclear_mask is not None:
            # Expand nuclear mask
            expanded = morphology.dilation(nuclear_mask, morphology.disk(self.cell_diameter//2))
            binary = binary & expanded
            
        return binary
    
    def _extract_properties(
        self,
        mask: np.ndarray,
        image: np.ndarray
    ) -> dict:
        """Extract properties from a segmented region.
        
        Args:
            mask: Binary mask of the region
            image: Original image
            
        Returns:
            Dictionary of region properties
        """
        props = measure.regionprops_table(
            mask,
            image,
            properties=['label', 'area', 'perimeter', 'mean_intensity']
        )
        return props
    
    def clean_and_label(
        self,
        mask: np.ndarray,
        min_size: Optional[int] = None
    ) -> np.ndarray:
        """Clean and label a binary mask.
        
        Args:
            mask: Binary mask to clean
            min_size: Minimum size for objects (defaults to nuclei_diameter**2)
            
        Returns:
            Labeled mask
        """
        if min_size is None:
            min_size = self.nuclei_diameter**2
            
        # Clean up
        cleaned = morphology.remove_small_objects(mask, min_size=min_size)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
        
        # Label
        labeled = measure.label(cleaned)
        
        return labeled
    
    def _plot_segmentation(
        self,
        image: np.ndarray,
        nuclei_mask: np.ndarray,
        cell_mask: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot segmentation results for debugging.
        
        Args:
            image: Original image
            nuclei_mask: Binary mask of nuclei
            cell_mask: Binary mask of cells
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(nuclei_mask, cmap='gray')
        axes[1].set_title('Nuclei Mask')
        axes[1].axis('off')
        
        axes[2].imshow(cell_mask, cmap='gray')
        axes[2].set_title('Cell Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 