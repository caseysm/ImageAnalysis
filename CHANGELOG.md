# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-03-28

### Added
- Complete Python package structure in `imageanalysis/`
- Command-line tools with proper entry points
- Installation script and improved setup documentation
- Mapping module for image registration between 10X and 40X images
- Comprehensive test framework with synthetic and real data testing
- Original pipeline scripts preserved in `original_pipeline/` directory
- Real implementation of core pipeline functions replacing dummy placeholders
- Extraction and storage of nuclei centroids during segmentation for mapping
- RANSAC-based coordinate matching algorithm for robust image registration
- Production data testing for the mapping pipeline
- Diagnostic plots for mapping quality assessment
- Support for adjustable matching parameters through configuration

### Changed
- Converted from flat structure to proper Python package
- Standardized import statements to use absolute paths
- Updated naming conventions to follow Python standards
- Implemented setuptools properly for package management
- Updated documentation with detailed usage examples
- Updated segmentation to save nuclei centroids needed for mapping
- Improved parameter handling for 10X versus 40X segmentation

### Fixed
- Import errors with package structure
- Command-line argument handling
- Pipeline execution with real data
- File discovery issues in mapping pipeline
- Made matching algorithm more robust with different data distributions
- Reduced sample size requirements for RANSAC to work with sparse data
- Improved error handling and reporting for mapping failures
- Enhanced parameter bounds for transformation model to adapt to different data distributions
- Fixed nuclei centroid extraction and saving in segmentation module

## [0.1.0] - 2025-03-25

### Added
- Initial project structure
- Core pipeline framework
- Segmentation module for 10X and 40X images
- Basic genotyping and phenotyping functionality