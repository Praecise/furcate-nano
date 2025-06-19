# Contributing to Furcate Nano

Thank you for your interest in contributing to Furcate Nano! This open-source environmental edge computing framework benefits from community contributions in code, documentation, testing, and educational resources.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that fosters an inclusive, respectful environment for all contributors. By participating, you agree to uphold these standards:

- **Be respectful**: Treat all contributors with respect and professionalism
- **Be inclusive**: Welcome contributors from all backgrounds and experience levels
- **Be collaborative**: Work together constructively and help others learn
- **Be professional**: Focus on technical merit and constructive feedback
- **Be educational**: Remember this project serves educational purposes

Report any unacceptable behavior to [opensource@praecise.com](mailto:opensource@praecise.com).

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Git**: Version control system
- **Hardware**: Raspberry Pi 4+ or NVIDIA Jetson (optional for simulation)
- **Experience**: Basic Python programming and environmental science concepts

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help identify and fix issues
2. **Feature Requests**: Suggest new capabilities
3. **Code Contributions**: Implement features and fixes
4. **Documentation**: Improve guides and tutorials
5. **Educational Content**: Create lesson plans and experiments
6. **Testing**: Expand test coverage and validation
7. **Hardware Support**: Add support for new sensors and devices

## Development Setup

### 1. Fork and Clone Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/furcate-nano.git
cd furcate-nano

# Add upstream remote
git remote add upstream https://github.com/praecise/furcate-nano.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv furcate-dev
source furcate-dev/bin/activate  # On Windows: furcate-dev\Scripts\activate

# Install development dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run basic tests
python -m pytest tests/unit/ -v

# Start in simulation mode
furcate-nano start --simulation --debug

# Verify development setup
python -c "import furcate_nano; print(f'Furcate Nano v{furcate_nano.__version__} ready for development')"
```

### 4. Development Dependencies

The development environment includes:

```
# Core development tools
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0

# Documentation tools
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0
myst-parser>=1.0.0

# Educational testing
jupyter>=1.0.0
matplotlib>=3.7.0
pandas>=2.0.0
```

## Contributing Guidelines

### Code Style

We follow Python PEP 8 with some project-specific conventions:

```python
# Use clear, descriptive variable names
sensor_reading = await hardware.read_temperature()

# Type hints for all public functions
async def process_sensor_data(data: Dict[str, float]) -> SensorReading:
    """Process raw sensor data into structured readings."""
    pass

# Educational code should be especially clear
class EnvironmentalMonitor:
    """
    Simple environmental monitoring class for educational use.
    
    This class demonstrates basic environmental data collection
    suitable for high school and undergraduate education.
    """
```

### Code Formatting

Use automated formatting tools:

```bash
# Format code
black furcate_nano/ tests/

# Check code style
flake8 furcate_nano/ tests/

# Type checking
mypy furcate_nano/
```

### Commit Messages

Use clear, descriptive commit messages:

```
# Good examples
feat(sensors): Add BME680 air quality sensor support
fix(classroom): Resolve dashboard WebSocket connection issue
docs(tutorial): Add advanced weather AI integration guide
test(hardware): Expand sensor calibration test coverage

# Message format
<type>(<scope>): <description>

# Types: feat, fix, docs, test, refactor, style, chore
# Scope: sensors, ml, mesh, classroom, hardware, etc.
```

### Branch Naming

Use descriptive branch names:

```bash
# Feature branches
git checkout -b feature/bme680-sensor-support
git checkout -b feature/classroom-dashboard-improvements

# Bug fix branches
git checkout -b fix/mqtt-connection-timeout
git checkout -b fix/raspberry-pi-hardware-detection

# Documentation branches
git checkout -b docs/advanced-ml-tutorial
git checkout -b docs/api-reference-update
```

## Testing

### Test Categories

#### 1. Unit Tests

Test individual components in isolation:

```python
# tests/unit/test_sensors.py
import pytest
from furcate_nano.hardware import HardwareManager

@pytest.mark.asyncio
async def test_sensor_reading_validation():
    """Test sensor reading data validation."""
    manager = HardwareManager(simulation=True)
    
    # Test valid reading
    reading = await manager.read_temperature()
    assert reading.value is not None
    assert reading.sensor_type == "temperature"
    assert reading.timestamp is not None

@pytest.mark.educational
async def test_classroom_sensor_setup():
    """Test classroom-specific sensor configurations."""
    # Educational test for classroom environments
    pass
```

#### 2. Integration Tests

Test component interactions:

```python
# tests/integration/test_ml_pipeline.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_ml_pipeline():
    """Test end-to-end ML processing pipeline."""
    config = NanoConfig(simulation=True)
    core = FurcateNanoCore(config)
    
    # Test sensor data -> ML analysis -> results
    await core.start()
    readings = await core.hardware.read_sensors()
    results = await core.ml.analyze(readings)
    
    assert results is not None
    assert "classification" in results
```

#### 3. Educational Tests

Test educational functionality:

```python
# tests/educational/test_classroom_features.py
@pytest.mark.educational
def test_student_dashboard_safety():
    """Ensure student dashboard has appropriate safety restrictions."""
    # Test educational safety features
    pass

@pytest.mark.educational 
def test_curriculum_alignment():
    """Test NGSS curriculum alignment features."""
    # Test educational standards compliance
    pass
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/educational/ -v

# Run with coverage
python -m pytest --cov=furcate_nano --cov-report=html

# Run educational tests only
python -m pytest -m educational

# Run tests requiring hardware
python -m pytest -m hardware --device=/dev/ttyUSB0
```

### Test Requirements

- **Unit tests**: Must run in simulation mode without hardware
- **Integration tests**: Can use simulation or require hardware flag
- **Educational tests**: Focus on classroom safety and usability
- **Coverage**: Aim for >80% code coverage
- **Documentation**: Test all public APIs and educational features

## Documentation

### Documentation Types

#### 1. Code Documentation

```python
class SensorManager:
    """
    Manages environmental sensors for Furcate Nano devices.
    
    This class provides a unified interface for reading from various
    environmental sensors commonly used in educational and research
    environments.
    
    Attributes:
        sensors: Dictionary of configured sensor instances
        simulation: Whether to use simulated sensor data
        
    Example:
        >>> manager = SensorManager(simulation=True)
        >>> reading = await manager.read_temperature()
        >>> print(f"Temperature: {reading.value}°C")
    """
```

#### 2. Tutorials

Create comprehensive, educational tutorials:

```markdown
# Tutorial Structure
## Overview
## Prerequisites  
## Step-by-Step Instructions
## Code Examples
## Educational Objectives
## Assessment Questions
## Troubleshooting
## Next Steps
```

#### 3. API Documentation

Document all public APIs:

```python
async def read_sensor_data(
    sensor_type: str,
    duration_seconds: int = 60,
    interval_seconds: int = 5
) -> List[SensorReading]:
    """
    Read sensor data over a specified duration.
    
    Args:
        sensor_type: Type of sensor to read ('temperature', 'humidity', etc.)
        duration_seconds: Total reading duration in seconds
        interval_seconds: Interval between readings in seconds
        
    Returns:
        List of sensor readings with timestamps and values
        
    Raises:
        SensorNotFoundError: If specified sensor type is not available
        HardwareError: If sensor hardware communication fails
        
    Example:
        >>> readings = await read_sensor_data('temperature', 300, 10)
        >>> avg_temp = sum(r.value for r in readings) / len(readings)
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### 1. Before Submitting

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
# ... make your changes ...

# Run tests
python -m pytest
black furcate_nano/ tests/
flake8 furcate_nano/ tests/
```

### 2. Pull Request Process

1. **Create Pull Request**: Open PR against `main` branch
2. **Fill Template**: Use the PR template to describe changes
3. **Pass Checks**: Ensure all CI checks pass
4. **Request Review**: Tag relevant maintainers
5. **Address Feedback**: Respond to review comments
6. **Merge**: Maintainer will merge when ready

### 3. Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Educational content
- [ ] Hardware support

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Educational tests pass
- [ ] Manual testing completed

## Educational Impact
- [ ] Safe for classroom use
- [ ] Includes educational documentation
- [ ] Aligns with curriculum standards
- [ ] Tested in educational environment

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or justified)
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Educational Forum**: Teaching and learning discussions
- **Developer Chat**: Real-time development coordination

### Maintainers

- **Core Team**: Praecise Ltd development team
- **Educational Lead**: Curriculum and classroom integration
- **Hardware Lead**: Sensor and device support
- **ML Lead**: Machine learning and AI features

### Recognition

Contributors are recognized through:

- **Contributor List**: Recognition in README and documentation
- **Educational Impact**: Special recognition for educational contributions
- **Release Notes**: Acknowledgment in version releases
- **Community Highlights**: Featured contributions in updates

## Getting Help

### Resources

- **Documentation**: Comprehensive guides and tutorials
- **Examples**: Sample code and configurations
- **Tests**: Reference implementations
- **Community**: Active contributor community

### Contact

- **General Questions**: [GitHub Discussions](https://github.com/praecise/furcate-nano/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/praecise/furcate-nano/issues)
- **Educational Support**: [education@praecise.com](mailto:education@praecise.com)
- **Security Issues**: [security@praecise.com](mailto:security@praecise.com)

## Development Roadmap

### Current Priorities

1. **Hardware Expansion**: Support for additional sensor types
2. **Educational Tools**: Enhanced classroom features
3. **ML Models**: Improved environmental classification
4. **Documentation**: Comprehensive tutorial expansion
5. **Testing**: Expanded test coverage and validation

### Getting Involved

Choose areas that match your interests and expertise:

- **Python Developers**: Core framework and ML features
- **Educators**: Curriculum alignment and classroom tools
- **Hardware Engineers**: Sensor integration and optimization
- **Data Scientists**: Environmental analysis and modeling
- **Technical Writers**: Documentation and tutorials
- **Students**: Testing, feedback, and educational content

---

**Thank you for contributing to Furcate Nano! Together, we're building the future of environmental education and research.**