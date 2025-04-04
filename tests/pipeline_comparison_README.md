# Pipeline Comparison Testing

This directory contains tests to compare the modified original pipeline scripts in `modified_original_pipeline` with the refactored `imageanalysis` package.

## Test Modified Pipeline Script

The `test_modified_pipeline.py` script runs both implementations and compares their outputs to ensure functional equivalence.

### Basic Usage

```bash
python tests/test_modified_pipeline.py
```

This will run a simulated comparison test using the default settings, which:
- Uses the data in `/home/casey/Desktop/ShalemLab/ImageAnalysis/data`
- Outputs results to `/home/casey/Desktop/ShalemLab/ImageAnalysis/results/pipeline_comparison/`
- Tests all pipeline components (segmentation, genotyping, phenotyping, albums)
- Uses simulated data instead of actual execution

### Command-line Arguments

The script supports the following options:

```
--data-dir DIR               Data directory containing test images
--modified-output-dir DIR    Output directory for modified original pipeline
--refactored-output-dir DIR  Output directory for refactored pipeline
--log-dir DIR                Directory for log files
--modified-script-dir DIR    Directory containing the modified original pipeline scripts
--wells WELL [WELL ...]      Wells to process (default: Well1)
--channels CHANNELS          Comma-separated list of channel names
--timeout SECONDS            Timeout for commands in seconds (default: 600)
--skip-segmentation          Skip segmentation testing
--skip-genotyping            Skip genotyping testing
--skip-phenotyping           Skip phenotyping testing
--skip-albums                Skip album generation testing
--test-actual-execution      Run actual scripts instead of simulating (requires input data)
```

### Examples

1. Test with actual execution (requires real data):

```bash
python tests/test_modified_pipeline.py --test-actual-execution
```

2. Test specific components only:

```bash
python tests/test_modified_pipeline.py --skip-genotyping --skip-albums
```

3. Test with custom data and output directories:

```bash
python tests/test_modified_pipeline.py --data-dir /path/to/data --modified-output-dir /path/to/output/modified --refactored-output-dir /path/to/output/refactored
```

4. Test specific wells:

```bash
python tests/test_modified_pipeline.py --wells Well1 Well2
```

## Understanding the Results

After running the test, the script will:

1. Generate output files for both implementations in their respective output directories
2. Create a comparison report in the `report` subdirectory
3. Generate a detailed log file in the `logs` subdirectory

The comparison report (`comparison_report_YYYYMMDD_HHMMSS.json`) contains:
- Overall match status
- List of specific differences found
- Components tested
- Timestamp and test parameters

## Comparison Methodology

The script compares:

1. **Segmentation**: Nuclear and cell masks shapes and cell counts
2. **Genotyping**: Barcode assignments and quality scores
3. **Phenotyping**: Feature measurements and cell counts
4. **Albums**: Image arrays and cell counts

Small differences (within 10% tolerance) are allowed to account for implementation variations.

## Simulation Mode vs. Actual Execution

By default, the script runs in simulation mode, which:
- Creates synthetic data matching the expected output formats
- Uses the same random seed for both implementations to ensure comparability
- Runs much faster than actual execution

When using `--test-actual-execution`, the script:
- Executes the actual scripts from both implementations
- Requires real input data
- Takes longer to run but provides more realistic comparison