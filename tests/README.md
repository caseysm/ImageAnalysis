# ImageAnalysis Test Suite

This directory contains the test suite for the ImageAnalysis package. We use pytest as our testing framework.

## Table of Contents
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Test Organization](#test-organization)
- [Writing Tests](#writing-tests)
- [Fixtures](#fixtures)
- [Continuous Integration](#continuous-integration)
- [Coverage Reports](#coverage-reports)

## Installation

To install the test dependencies:

```bash
pip install -e ".[test]"
```

This will install pytest and related testing tools.

## Running Tests

### Run all tests:
```bash
pytest
```

### Run tests with coverage:
```bash
pytest --cov=imageanalysis --cov-report=html
```

### Run specific test categories:
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run all except slow tests
pytest -m "not slow"

# Run tests for a specific module
pytest tests/unit/test_image_loader.py

# Run with verbose output
pytest -v

# Run in parallel (faster)
pytest -n auto
```

### Run with different configurations:
```bash
# Run with specific Python version
python3.9 -m pytest

# Run with warnings displayed
pytest -W default

# Stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf
```

## Test Organization

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Unit tests (fast, isolated)
│   ├── test_image_loader.py
│   ├── test_segmentation_utils.py
│   └── ...
├── integration/          # Integration tests (slower, real workflows)
│   ├── test_segmentation_pipeline.py
│   ├── test_genotyping_pipeline.py
│   └── ...
├── fixtures/            # Test data and resources
│   ├── images/         # Sample images
│   ├── configs/        # Test configurations
│   └── expected/       # Expected outputs
└── performance/         # Performance benchmarks
    └── test_benchmarks.py
```

## Writing Tests

### Test File Naming
- Test files should start with `test_` or end with `_test.py`
- Test classes should start with `Test`
- Test functions should start with `test_`

### Basic Test Structure
```python
import pytest
from imageanalysis.module import function_to_test

class TestMyFeature:
    """Test suite for MyFeature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge cases."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
    
    @pytest.mark.slow
    def test_performance(self, benchmark_data):
        """Test performance on large data."""
        result = function_to_test(benchmark_data)
        assert result is not None
```

### Using Markers
Mark tests to categorize them:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Slower integration test."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Very slow test."""
    pass

@pytest.mark.requires_data
def test_with_real_data():
    """Test requiring data files."""
    pass
```

### Parametrized Tests
Test multiple inputs with one test function:

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input, expected):
    assert input * 2 == expected
```

## Fixtures

Fixtures provide reusable test data and setup. See `conftest.py` for available fixtures:

### Common Fixtures

```python
def test_with_synthetic_image(synthetic_image):
    """Test using synthetic image fixture."""
    assert synthetic_image.shape == (512, 512)

def test_with_temp_directory(temp_dir):
    """Test using temporary directory."""
    output_file = temp_dir / "output.txt"
    output_file.write_text("test")
    assert output_file.exists()

def test_with_sample_config(sample_config):
    """Test using sample configuration."""
    assert 'segmentation' in sample_config
```

### Creating New Fixtures

Add fixtures to `conftest.py` or test files:

```python
@pytest.fixture
def my_fixture():
    """Provide test data."""
    data = create_test_data()
    yield data  # Provide data to test
    cleanup_data()  # Cleanup after test
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- Every push to main/develop branches
- Every pull request
- Multiple Python versions (3.7-3.10)
- Multiple operating systems (Ubuntu, macOS)

See `.github/workflows/tests.yml` for configuration.

## Coverage Reports

### View Coverage Locally
After running tests with coverage:

```bash
# Generate HTML report
pytest --cov=imageanalysis --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Requirements
- Minimum coverage: 80% (configured in pytest.ini)
- Critical modules should have >90% coverage
- New code should include tests

### Excluding Code from Coverage
Mark code that shouldn't be covered:

```python
if TYPE_CHECKING:  # pragma: no cover
    import typing

def deprecated_function():  # pragma: no cover
    """This function is deprecated."""
    pass
```

## Best Practices

1. **Write tests first** (TDD) when adding new features
2. **Keep tests fast** - mock external dependencies
3. **Test edge cases** - empty inputs, large inputs, invalid inputs
4. **Use descriptive names** - test names should explain what they test
5. **One assertion per test** - makes failures clear
6. **Use fixtures** - avoid duplicating setup code
7. **Mock external dependencies** - tests should not depend on files/network
8. **Test the interface** - not the implementation details
9. **Keep tests maintainable** - refactor tests like production code
10. **Run tests locally** before pushing

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure package is installed
pip install -e .
```

**Missing dependencies:**
```bash
# Install all test dependencies
pip install -e ".[test,dev]"
```

**Tests hanging:**
- Check for infinite loops
- Use pytest-timeout: `pytest --timeout=300`

**Flaky tests:**
- Set random seeds in tests
- Mock time-dependent code
- Avoid depending on file system order

**Coverage not working:**
- Ensure .coveragerc is configured correctly
- Check that source paths are correct
- Use `pytest --cov=imageanalysis` not just `--cov`

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Scientific Code](https://katyhuff.github.io/python-testing/)