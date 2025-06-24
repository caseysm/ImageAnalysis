# Directory Structure

```
ImageAnalysis/                           # Root package directory
├── __init__.py                      # Package initialization with version
├── cli.py                           # Command-line interface (main entry point)
├── config/                          # Configuration management
│   ├── __init__.py
│   └── settings.py                  # Centralized settings
├── core/                            # Core functionality
│   ├── __init__.py
│   ├── pipeline.py                  # Base pipeline class
│   ├── mapping.py                   # Coordinate mapping utilities
│   ├── segmentation/                # Segmentation components
│   │   ├── __init__.py
│   │   ├── base.py                  # Base segmentation pipeline
│   │   ├── segmentation_10x.py      # 10X segmentation implementation
│   │   ├── segmentation_40x.py      # 40X segmentation implementation
│   │   └── cell_segmentation.py     # Image segmentation algorithms
│   ├── genotyping/                  # Genotyping components
│   │   ├── __init__.py
│   │   ├── base.py                  # Genotyping pipeline
│   │   ├── barcode_assignment.py    # Barcode assignment algorithms
│   │   └── peak_calling.py          # Peak detection algorithms
│   ├── phenotyping/                 # Phenotyping components
│   │   ├── __init__.py
│   │   ├── base.py                  # Phenotyping pipeline
│   │   └── metrics.py               # Phenotype measurement algorithms
│   └── visualization/               # Visualization components
│       ├── __init__.py
│       ├── albums.py                # Album generation pipeline
│       └── plots.py                 # Plotting functions
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── io.py                        # Image and file I/O utilities
│   ├── logging.py                   # Logging setup
│   ├── parallel.py                  # Parallel processing utilities
│   └── stats.py                     # Statistical utilities
├── data/                            # Sample data and loaders
│   ├── __init__.py
│   └── sample_data.py               # Sample data loading utilities
├── bin/                             # Executable scripts
│   ├── run_full_pipeline.py         # Run complete pipeline
│   ├── run_segmentation.py          # Run only segmentation
│   ├── run_genotyping.py            # Run only genotyping
│   ├── run_phenotyping.py           # Run only phenotyping
│   └── create_albums.py             # Run only album generation
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Test configuration
│   ├── test_segmentation.py         # Segmentation tests
│   ├── test_genotyping.py           # Genotyping tests
│   ├── test_phenotyping.py          # Phenotyping tests
│   ├── test_visualization.py        # Visualization tests
│   ├── test_cli.py                  # CLI tests
│   └── data/                        # Test data
│       ├── __init__.py
│       └── test_images/             # Sample test images
├── setup.py                         # Package setup script
├── pyproject.toml                   # Modern Python packaging config
├── README.md                        # Project documentation
├── LICENSE                          # License information
├── CHANGELOG.md                     # Version history
└── CONTRIBUTING.md                  # Contribution guidelines
```

## Key Components

### Core Modules
- **segmentation/**: Cell and nuclei segmentation algorithms
- **genotyping/**: Barcode identification and assignment
- **phenotyping/**: Cell phenotype measurement
- **visualization/**: Data visualization and album generation

### Utility Modules
- **io.py**: Image and file handling
- **logging.py**: Logging configuration
- **parallel.py**: Parallel processing
- **stats.py**: Statistical analysis

### Configuration
- **settings.py**: Centralized configuration
- **pyproject.toml**: Package metadata and dependencies

### Documentation
- **README.md**: Project overview and setup
- **CHANGELOG.md**: Version history
- **CONTRIBUTING.md**: Development guidelines

### Testing
- Comprehensive test suite
- Sample test data
- Test configuration 