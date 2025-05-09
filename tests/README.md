# Tests for ImageAnalysis Package

This directory contains tests for the ImageAnalysis package.

## Test Categories

- **Basic Tests**: Simple tests to verify basic functionality
  - `test_basic.py`
  - `test_simplified.py`

- **Component Tests**: Tests for individual pipeline components
  - `test_genotyping/test_genotyping.py`
  - `test_phenotyping/test_phenotyping.py`
  - `test_segmentation/test_segmentation.py`
  - `test_segmentation_standalone.py`

- **Pipeline Tests**: Tests for the full pipeline
  - `test_full_pipeline.py`
  - `full_pipeline_tests/test_real_pipeline.py`
  - `synthetic_tests/test_synthetic_pipeline.py`

## Running Tests

You can run individual tests:

```bash
# Run a basic test
python -m tests.test_basic

# Run a component test
python -m tests.test_genotyping.test_genotyping

# Run a pipeline test
python -m tests.full_pipeline_tests.test_real_pipeline
```

Or run all tests:

```bash
python -m tests.run_all_tests
```

## Test Data

Test data is stored in the `data/` directory. This includes:
- Synthetic test images
- Sample barcode libraries
- Configuration files for testing

## Adding New Tests

When adding new tests:
1. Follow the existing pattern for test organization
2. Place component tests in their respective directories
3. Add the test to `run_all_tests.py` to include it in the full test suite