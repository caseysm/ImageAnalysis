"""Integration tests for segmentation pipelines."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from imageanalysis.core.segmentation.segmentation_10x import Segmentation10XPipeline
from imageanalysis.core.segmentation.segmentation_40x import Segmentation40XPipeline


class TestSegmentation10XPipeline:
    """Test suite for 10x segmentation pipeline."""
    
    @pytest.mark.integration
    def test_pipeline_with_synthetic_image(self, synthetic_image, temp_dir):
        """Test segmentation pipeline with synthetic image."""
        # Save synthetic image as mock input
        input_file = temp_dir / "test_image.tif"
        import tifffile
        tifffile.imwrite(str(input_file), synthetic_image)
        
        # Create pipeline
        pipeline = Segmentation10XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir)
        )
        
        # Run pipeline
        result = pipeline.run()
        
        # Assertions
        assert result is not None
        assert 'nuclei_mask' in result
        assert 'cell_mask' in result
        assert 'properties' in result
        
        # Check output files exist
        assert (temp_dir / "test_image_nuclei_mask.npy").exists()
        assert (temp_dir / "test_image_cell_mask.npy").exists()
        assert (temp_dir / "test_image_properties.json").exists()
    
    @pytest.mark.integration
    def test_pipeline_with_config(self, synthetic_image, sample_config, temp_dir):
        """Test segmentation pipeline with custom configuration."""
        # Save synthetic image
        input_file = temp_dir / "test_image.tif"
        import tifffile
        tifffile.imwrite(str(input_file), synthetic_image)
        
        # Create pipeline with config
        pipeline = Segmentation10XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir),
            config=sample_config['segmentation']
        )
        
        # Run pipeline
        result = pipeline.run()
        
        # Verify config was applied
        assert pipeline.cell_diameter == sample_config['segmentation']['cell_diameter']
        assert pipeline.flow_threshold == sample_config['segmentation']['flow_threshold']
    
    @pytest.mark.parametrize("image_size", [(256, 256), (512, 512), (1024, 1024)])
    def test_pipeline_different_image_sizes(self, image_size, temp_dir):
        """Test pipeline with different image sizes."""
        # Create image of specified size
        image = np.random.randint(0, 4095, size=image_size, dtype=np.uint16)
        
        # Save image
        input_file = temp_dir / f"test_{image_size[0]}x{image_size[1]}.tif"
        import tifffile
        tifffile.imwrite(str(input_file), image)
        
        # Create and run pipeline
        pipeline = Segmentation10XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir)
        )
        
        result = pipeline.run()
        
        # Check output dimensions match input
        assert result['nuclei_mask'].shape == image_size
        assert result['cell_mask'].shape == image_size
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_pipeline_performance(self, benchmark_data, temp_dir):
        """Test segmentation performance on larger images."""
        image, metadata = benchmark_data
        
        # Save image
        input_file = temp_dir / "benchmark_image.tif"
        import tifffile
        tifffile.imwrite(str(input_file), image)
        
        # Time the pipeline execution
        import time
        start = time.time()
        
        pipeline = Segmentation10XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir)
        )
        result = pipeline.run()
        
        end = time.time()
        
        # Performance assertions
        execution_time = end - start
        assert execution_time < 60  # Should complete within 60 seconds
        
        # Verify we found most cells
        n_detected = len(np.unique(result['nuclei_mask'])) - 1  # Subtract background
        expected_cells = metadata['n_cells']
        assert n_detected >= expected_cells * 0.7  # At least 70% detection rate


class TestSegmentation40XPipeline:
    """Test suite for 40x segmentation pipeline."""
    
    @pytest.mark.integration
    def test_pipeline_basic(self, synthetic_image, temp_dir):
        """Test basic 40x segmentation pipeline."""
        # Save synthetic image
        input_file = temp_dir / "test_40x.tif"
        import tifffile
        tifffile.imwrite(str(input_file), synthetic_image)
        
        # Create pipeline
        pipeline = Segmentation40XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir)
        )
        
        # Run pipeline
        result = pipeline.run()
        
        # Assertions
        assert result is not None
        assert 'nuclei_mask' in result
        assert 'cell_mask' in result
        
        # 40x should detect more detail
        n_nuclei = len(np.unique(result['nuclei_mask'])) - 1
        assert n_nuclei > 0
    
    @pytest.mark.integration
    def test_pipeline_multichannel(self, synthetic_multichannel_image, temp_dir):
        """Test 40x pipeline with multi-channel image."""
        # Save multi-channel image
        input_file = temp_dir / "test_multichannel.tif"
        import tifffile
        tifffile.imwrite(str(input_file), synthetic_multichannel_image)
        
        # Create pipeline (assuming it uses first channel for segmentation)
        pipeline = Segmentation40XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir),
            channel_index=0  # Use DAPI channel
        )
        
        # Run pipeline
        result = pipeline.run()
        
        # Check results
        assert result is not None
        assert result['nuclei_mask'].shape == synthetic_multichannel_image.shape[1:]


class TestSegmentationComparison:
    """Test comparing 10x and 40x segmentation results."""
    
    @pytest.mark.integration
    def test_compare_10x_40x_results(self, synthetic_image, temp_dir):
        """Compare results from 10x and 40x pipelines."""
        # Save image
        input_file = temp_dir / "test_comparison.tif"
        import tifffile
        tifffile.imwrite(str(input_file), synthetic_image)
        
        # Run both pipelines
        pipeline_10x = Segmentation10XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir / "10x")
        )
        result_10x = pipeline_10x.run()
        
        pipeline_40x = Segmentation40XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir / "40x")
        )
        result_40x = pipeline_40x.run()
        
        # Compare results
        n_cells_10x = len(np.unique(result_10x['nuclei_mask'])) - 1
        n_cells_40x = len(np.unique(result_40x['nuclei_mask'])) - 1
        
        # Results should be similar but not necessarily identical
        assert abs(n_cells_10x - n_cells_40x) / max(n_cells_10x, n_cells_40x) < 0.3


@pytest.mark.integration
class TestSegmentationErrorHandling:
    """Test error handling in segmentation pipelines."""
    
    def test_missing_input_file(self, temp_dir):
        """Test handling of missing input file."""
        with pytest.raises(FileNotFoundError):
            pipeline = Segmentation10XPipeline(
                input_file="nonexistent.tif",
                output_dir=str(temp_dir)
            )
            pipeline.run()
    
    def test_invalid_image_format(self, temp_dir):
        """Test handling of invalid image format."""
        # Create invalid file
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("This is not an image")
        
        with pytest.raises(ValueError):
            pipeline = Segmentation10XPipeline(
                input_file=str(invalid_file),
                output_dir=str(temp_dir)
            )
            pipeline.run()
    
    def test_empty_image(self, temp_dir):
        """Test handling of empty/black image."""
        # Create empty image
        empty_image = np.zeros((512, 512), dtype=np.uint16)
        input_file = temp_dir / "empty.tif"
        import tifffile
        tifffile.imwrite(str(input_file), empty_image)
        
        pipeline = Segmentation10XPipeline(
            input_file=str(input_file),
            output_dir=str(temp_dir)
        )
        result = pipeline.run()
        
        # Should handle gracefully
        assert result is not None
        assert np.all(result['nuclei_mask'] == 0)  # No cells detected