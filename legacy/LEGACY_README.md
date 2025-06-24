# Legacy Pipeline Code

This directory contains the original and transitional versions of the ImageAnalysis pipeline. These are preserved for reference and historical purposes.

## ⚠️ Important Notice

**For new projects, please use the modern `imageanalysis` Python package instead of these legacy scripts.**

The legacy code is maintained here for:
- Reference for users migrating from the old pipeline
- Historical documentation
- Comparison with the new implementation

## Directory Contents

### original_pipeline/
The original implementation of the image analysis pipeline:
- Individual Python scripts for each processing step
- SLURM batch scripts for HPC execution
- Jupyter notebooks for interactive analysis
- Original documentation and README

Key components:
- `Segment_10X.py` / `Segment_40X.py` - Cell segmentation scripts
- `Genotyping_Pipeline.py` - Barcode calling and genotyping
- `Phenotype_Cells.py` - Cell phenotyping and measurements
- `Mapping_*.ipynb` - Jupyter notebooks for image registration
- Various `.sh` scripts - SLURM job submission scripts

### modified_original_pipeline/
A transitional version with partial updates:
- Some refactoring of the original scripts
- Intermediate step between original and final package structure
- May contain bug fixes or improvements not in the original

## Migration Guide

To migrate from the legacy pipeline to the new package:

1. **Install the new package:**
   ```bash
   pip install -e .
   ```

2. **Replace script calls with package commands:**
   ```bash
   # Old way:
   python original_pipeline/Segment_10X.py --args
   
   # New way:
   run_segmentation --magnification 10x --args
   ```

3. **Use Python imports instead of scripts:**
   ```python
   # Old way:
   from In_Situ_Functions import function_name
   
   # New way:
   from imageanalysis.core.segmentation import function_name
   ```

4. **Configuration:**
   - Old: Command-line arguments and hardcoded values
   - New: Configuration files and Python API

## Key Differences

| Aspect | Legacy Pipeline | New Package |
|--------|----------------|-------------|
| Structure | Standalone scripts | Organized Python package |
| Installation | Copy scripts | `pip install` |
| Dependencies | Manual setup | Automatic with pip |
| Testing | None | Comprehensive test suite |
| Documentation | README only | Full API documentation |
| CLI | Direct script execution | Entry points (`run_*` commands) |
| Modularity | Limited | Fully modular |
| Error handling | Basic | Comprehensive |

## Using Legacy Code

If you must use the legacy code:

1. Review the original README in `original_pipeline/README.md`
2. Ensure all dependencies are manually installed
3. Run scripts from the appropriate directory
4. Be aware that paths may be hardcoded for specific systems

## Support

The legacy code is no longer actively maintained. For support:
- Check the migration guide in `/docs/MIGRATION_GUIDE.md`
- Use the new `imageanalysis` package for active development
- Refer to the main README for current documentation

## Historical Context

- **original_pipeline**: Developed 2016-2019 for HPC cluster use
- **modified_original_pipeline**: Transition work in 2019-2020
- **imageanalysis package**: Modern refactor completed 2020+

The evolution represents a transition from research scripts to production-ready software.