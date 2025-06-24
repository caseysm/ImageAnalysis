"""Unit tests for image loading utilities."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tifffile

from imageanalysis.utils.io import ImageLoader


class TestImageLoader:
    """Test suite for ImageLoader class."""
    
    @pytest.mark.unit
    def test_load_tiff_image(self, temp_dir):
        """Test loading TIFF images."""
        # Create test TIFF
        test_image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        tiff_path = temp_dir / "test.tif"
        tifffile.imwrite(str(tiff_path), test_image)
        
        # Load image
        loader = ImageLoader()
        loaded_image = loader.load_image(str(tiff_path))
        
        # Assertions
        assert loaded_image is not None
        assert isinstance(loaded_image, np.ndarray)
        assert loaded_image.shape == test_image.shape
        assert np.array_equal(loaded_image, test_image)
    
    @pytest.mark.unit
    def test_load_multi_channel_tiff(self, temp_dir):
        """Test loading multi-channel TIFF images."""
        # Create multi-channel TIFF
        test_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
        tiff_path = temp_dir / "multi_channel.tif"
        tifffile.imwrite(str(tiff_path), test_image)
        
        # Load image
        loader = ImageLoader()
        loaded_image = loader.load_image(str(tiff_path))
        
        # Assertions
        assert loaded_image.shape == test_image.shape
        assert loaded_image.ndim == 3
        assert loaded_image.shape[0] == 3  # 3 channels
    
    @pytest.mark.unit
    @patch('nd2reader.ND2Reader')
    def test_load_nd2_image(self, mock_nd2reader):
        """Test loading ND2 images."""
        # Mock ND2Reader
        mock_reader = Mock()
        mock_reader.__enter__ = Mock(return_value=mock_reader)
        mock_reader.__exit__ = Mock(return_value=None)
        mock_reader.metadata = {
            'width': 512,
            'height': 512,
            'channels': ['DAPI', 'GFP']
        }
        mock_reader.__getitem__ = Mock(return_value=np.ones((512, 512), dtype=np.uint16))
        mock_reader.__len__ = Mock(return_value=2)
        mock_nd2reader.return_value = mock_reader
        
        # Load image
        loader = ImageLoader()
        loaded_image = loader.load_image("test.nd2")
        
        # Assertions
        assert loaded_image is not None
        assert isinstance(loaded_image, np.ndarray)
        mock_nd2reader.assert_called_once()
    
    @pytest.mark.unit
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = ImageLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_image("nonexistent.tif")
    
    @pytest.mark.unit
    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported file format."""
        # Create unsupported file
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("unsupported content")
        
        loader = ImageLoader()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_image(str(unsupported_file))
    
    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_load_different_dtypes(self, temp_dir, dtype):
        """Test loading images with different data types."""
        # Create test image with specific dtype
        if dtype == np.float32:
            test_image = np.random.random((100, 100)).astype(dtype)
        else:
            max_val = np.iinfo(dtype).max
            test_image = np.random.randint(0, max_val, size=(100, 100), dtype=dtype)
        
        tiff_path = temp_dir / f"test_{dtype.__name__}.tif"
        tifffile.imwrite(str(tiff_path), test_image)
        
        # Load image
        loader = ImageLoader()
        loaded_image = loader.load_image(str(tiff_path))
        
        # Check dtype is preserved
        assert loaded_image.dtype == dtype
    
    @pytest.mark.unit
    def test_get_image_info(self, temp_dir):
        """Test getting image metadata."""
        # Create test image
        test_image = np.random.randint(0, 255, size=(3, 256, 256), dtype=np.uint8)
        tiff_path = temp_dir / "info_test.tif"
        tifffile.imwrite(str(tiff_path), test_image)
        
        # Get info
        loader = ImageLoader()
        info = loader.get_image_info(str(tiff_path))
        
        # Assertions
        assert info is not None
        assert info['shape'] == (3, 256, 256)
        assert info['dtype'] == 'uint8'
        assert info['channels'] == 3
        assert info['width'] == 256
        assert info['height'] == 256
    
    @pytest.mark.unit
    def test_save_image(self, temp_dir):
        """Test saving images."""
        # Create test image
        test_image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        # Save image
        loader = ImageLoader()
        save_path = temp_dir / "saved.tif"
        loader.save_image(test_image, str(save_path))
        
        # Verify saved
        assert save_path.exists()
        
        # Load and compare
        loaded_image = loader.load_image(str(save_path))
        assert np.array_equal(loaded_image, test_image)
    
    @pytest.mark.unit
    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image
        test_image = np.array([[0, 128, 255]], dtype=np.uint8)
        
        loader = ImageLoader()
        normalized = loader.normalize_image(test_image)
        
        # Check normalization
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert normalized[0, 1] == pytest.approx(128/255)
    
    @pytest.mark.unit
    def test_resize_image(self):
        """Test image resizing."""
        # Create test image
        test_image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        loader = ImageLoader()
        
        # Test different resize operations
        resized = loader.resize_image(test_image, (50, 50))
        assert resized.shape == (50, 50)
        
        # Test with scale factor
        resized = loader.resize_image(test_image, scale=0.5)
        assert resized.shape == (50, 50)
        
        # Test upscaling
        resized = loader.resize_image(test_image, (200, 200))
        assert resized.shape == (200, 200)
    
    @pytest.mark.unit
    def test_convert_bit_depth(self):
        """Test bit depth conversion."""
        # Create 16-bit image
        test_image = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)
        
        loader = ImageLoader()
        
        # Convert to 8-bit
        converted = loader.convert_bit_depth(test_image, 8)
        assert converted.dtype == np.uint8
        assert converted.max() <= 255
        
        # Convert 8-bit to 16-bit
        test_image_8bit = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        converted = loader.convert_bit_depth(test_image_8bit, 16)
        assert converted.dtype == np.uint16