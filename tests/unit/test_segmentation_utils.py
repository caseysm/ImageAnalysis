"""Unit tests for segmentation utility functions."""

import pytest
import numpy as np
from scipy import ndimage
from skimage import measure

# Assuming these utilities exist in the segmentation module
# You may need to adjust imports based on actual module structure
from imageanalysis.core.segmentation.base import (
    apply_threshold,
    watershed_segmentation,
    remove_small_objects,
    fill_holes,
    calculate_cell_properties,
    expand_labels
)


class TestThresholding:
    """Test thresholding functions."""
    
    @pytest.mark.unit
    def test_apply_threshold_otsu(self):
        """Test Otsu thresholding."""
        # Create bimodal image
        image = np.concatenate([
            np.random.normal(50, 10, 5000),
            np.random.normal(150, 10, 5000)
        ]).reshape(100, 100).astype(np.uint8)
        
        # Apply threshold
        binary = apply_threshold(image, method='otsu')
        
        # Check output
        assert binary.dtype == bool
        assert binary.shape == image.shape
        assert np.sum(binary) > 0  # Some pixels should be True
        assert np.sum(~binary) > 0  # Some pixels should be False
    
    @pytest.mark.unit
    def test_apply_threshold_manual(self):
        """Test manual thresholding."""
        image = np.arange(256).reshape(16, 16).astype(np.uint8)
        
        # Apply manual threshold
        binary = apply_threshold(image, method='manual', threshold=128)
        
        # Check threshold was applied correctly
        assert np.all(binary[image >= 128])
        assert np.all(~binary[image < 128])
    
    @pytest.mark.unit
    @pytest.mark.parametrize("method", ['otsu', 'li', 'yen'])
    def test_threshold_methods(self, method):
        """Test different threshold methods."""
        image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        binary = apply_threshold(image, method=method)
        
        assert binary.dtype == bool
        assert binary.shape == image.shape


class TestWatershed:
    """Test watershed segmentation."""
    
    @pytest.mark.unit
    def test_watershed_basic(self):
        """Test basic watershed segmentation."""
        # Create image with two touching circles
        image = np.zeros((100, 100), dtype=np.uint8)
        
        # Circle 1
        rr, cc = np.ogrid[30:70, 20:60]
        mask1 = (rr - 50)**2 + (cc - 40)**2 <= 20**2
        image[mask1] = 255
        
        # Circle 2 (overlapping)
        rr, cc = np.ogrid[30:70, 40:80]
        mask2 = (rr - 50)**2 + (cc - 60)**2 <= 20**2
        image[mask2] = 255
        
        # Apply watershed
        labels = watershed_segmentation(image)
        
        # Should separate the two circles
        assert labels.max() >= 2
        assert np.unique(labels).size >= 3  # Background + 2 objects
    
    @pytest.mark.unit
    def test_watershed_with_markers(self):
        """Test watershed with seed markers."""
        # Create test image
        image = np.zeros((100, 100))
        image[40:60, 40:60] = 1
        
        # Create markers
        markers = np.zeros_like(image, dtype=int)
        markers[50, 50] = 1  # Center marker
        
        labels = watershed_segmentation(image, markers=markers)
        
        # Check marker was used
        assert labels[50, 50] == 1
        assert labels.max() == 1


class TestMorphologicalOperations:
    """Test morphological operations."""
    
    @pytest.mark.unit
    def test_remove_small_objects(self):
        """Test removing small objects from binary image."""
        # Create image with objects of different sizes
        image = np.zeros((100, 100), dtype=bool)
        
        # Large object
        image[10:40, 10:40] = True  # 900 pixels
        
        # Small object
        image[50:55, 50:55] = True  # 25 pixels
        
        # Remove small objects
        cleaned = remove_small_objects(image, min_size=100)
        
        # Large object should remain
        assert np.any(cleaned[10:40, 10:40])
        
        # Small object should be removed
        assert not np.any(cleaned[50:55, 50:55])
    
    @pytest.mark.unit
    def test_fill_holes(self):
        """Test filling holes in binary objects."""
        # Create object with hole
        image = np.zeros((50, 50), dtype=bool)
        image[10:40, 10:40] = True  # Square
        image[20:30, 20:30] = False  # Hole in middle
        
        # Fill holes
        filled = fill_holes(image)
        
        # Hole should be filled
        assert np.all(filled[20:30, 20:30])
        
        # Original object area should remain
        assert np.all(filled[10:40, 10:40])
    
    @pytest.mark.unit
    def test_expand_labels(self):
        """Test label expansion (dilation)."""
        # Create labeled image
        labels = np.zeros((50, 50), dtype=int)
        labels[20:30, 20:30] = 1  # 10x10 square
        
        # Expand by 5 pixels
        expanded = expand_labels(labels, distance=5)
        
        # Check expansion
        labeled_area_original = np.sum(labels > 0)
        labeled_area_expanded = np.sum(expanded > 0)
        
        assert labeled_area_expanded > labeled_area_original
        assert expanded[25, 25] == 1  # Center should keep label
        assert expanded[15, 25] == 1  # Should expand to this point


class TestCellProperties:
    """Test cell property calculations."""
    
    @pytest.mark.unit
    def test_calculate_basic_properties(self):
        """Test calculation of basic cell properties."""
        # Create simple labeled image
        labels = np.zeros((100, 100), dtype=int)
        
        # Add two cells
        labels[20:40, 20:40] = 1  # Cell 1: 20x20 = 400 pixels
        labels[60:75, 60:75] = 2  # Cell 2: 15x15 = 225 pixels
        
        # Create intensity image
        intensity = np.ones_like(labels) * 100
        intensity[labels == 1] = 200
        intensity[labels == 2] = 150
        
        # Calculate properties
        props = calculate_cell_properties(labels, intensity)
        
        # Check properties
        assert len(props) == 2
        
        # Cell 1 properties
        cell1 = props[0]
        assert cell1['label'] == 1
        assert cell1['area'] == 400
        assert cell1['mean_intensity'] == 200
        assert cell1['centroid'] == pytest.approx((29.5, 29.5))
        
        # Cell 2 properties
        cell2 = props[1]
        assert cell2['label'] == 2
        assert cell2['area'] == 225
        assert cell2['mean_intensity'] == 150
    
    @pytest.mark.unit
    def test_calculate_shape_properties(self):
        """Test calculation of shape properties."""
        # Create circular object
        labels = np.zeros((100, 100), dtype=int)
        rr, cc = np.ogrid[40:60, 40:60]
        mask = (rr - 50)**2 + (cc - 50)**2 <= 10**2
        labels[mask] = 1
        
        props = calculate_cell_properties(labels, include_shape=True)
        
        # Check shape properties
        assert 'eccentricity' in props[0]
        assert 'solidity' in props[0]
        assert 'perimeter' in props[0]
        
        # Circle should have low eccentricity
        assert props[0]['eccentricity'] < 0.3
        
        # Circle should have high solidity
        assert props[0]['solidity'] > 0.9
    
    @pytest.mark.unit
    def test_properties_empty_image(self):
        """Test property calculation on empty image."""
        # Empty labeled image
        labels = np.zeros((100, 100), dtype=int)
        
        props = calculate_cell_properties(labels)
        
        # Should return empty list
        assert props == []
    
    @pytest.mark.unit
    def test_properties_with_background(self):
        """Test that background (label 0) is ignored."""
        # Create labeled image with explicit background
        labels = np.ones((100, 100), dtype=int) * 0  # All background
        labels[40:60, 40:60] = 1  # One object
        
        props = calculate_cell_properties(labels)
        
        # Should only have one object (not background)
        assert len(props) == 1
        assert props[0]['label'] == 1


class TestSegmentationValidation:
    """Test segmentation validation functions."""
    
    @pytest.mark.unit
    def test_validate_segmentation_masks(self):
        """Test validation of segmentation masks."""
        # Valid masks
        nuclei_mask = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 3]])
        cell_mask = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 3]])
        
        # Should pass validation
        from imageanalysis.core.segmentation.base import validate_segmentation
        assert validate_segmentation(nuclei_mask, cell_mask) == True
        
        # Invalid: mismatched shapes
        cell_mask_wrong = np.zeros((2, 2))
        assert validate_segmentation(nuclei_mask, cell_mask_wrong) == False
        
        # Invalid: cell without nucleus
        cell_mask_bad = cell_mask.copy()
        cell_mask_bad[2, 2] = 4  # Cell 4 with no nucleus
        
        with pytest.warns(UserWarning, match="Cell without nucleus"):
            validate_segmentation(nuclei_mask, cell_mask_bad)