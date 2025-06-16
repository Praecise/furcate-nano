# ============================================================================
# pyproject.toml - Modern Python packaging for Furcate Nano
# ============================================================================

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "furcate-nano"
version = "1.0.0"
description = "Open Source Environmental Edge Computing Framework"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Furcate Team", email = "opensource@furcate.earth"}
]
maintainers = [
    {name = "Furcate Team", email = "opensource@furcate.earth"}
]
keywords = [
    "environmental-monitoring", 
    "edge-computing", 
    "raspberry-pi", 
    "mesh-networking",
    "machine-learning",
    "iot",
    "environmental-ai"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Environmental Science",
    "Topic :: System :: Hardware",
    "Topic :: System :: Networking",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: POSIX :: Linux",
    "Environment :: No Input/Output (Daemon)"
]
requires-python = ">=3.8"
dependencies = [
    "pydantic>=1.8.0",
    "pyyaml>=6.0",
    "asyncio-mqtt>=0.11.0",
    "aiofiles>=0.8.0",
    "click>=8.0.0",
    "rich>=12.0.0",
    "psutil>=5.8.0",
    "numpy>=1.21.0,<1.24.0",
    "aiohttp>=3.8.0",
    "websockets>=10.0"
]

[project.optional-dependencies]
hardware = [
    "RPi.GPIO>=0.7.0",
    "gpiozero>=1.6.0", 
    "adafruit-circuitpython-dht>=3.7.0",
    "adafruit-circuitpython-bmp280>=3.2.0",
    "adafruit-circuitpython-ads1x15>=2.2.0",
    "pyserial>=3.5",
    "paho-mqtt>=1.6.0",
    "bleak>=0.19.0"
]
ml = [
    "tensorflow-lite>=2.10.0",
    "scikit-learn>=1.0.0"
]
storage = [
    "duckdb>=0.8.0",
    "python-rocksdb>=0.8.0"
]
dev = [
    "pytest>=6.0",
    "pytest-asyncio>=0.18.0",
    "pytest-cov>=3.0.0",
    "black>=22.0",
    "flake8>=4.0",
    "mypy>=0.950",
    "pre-commit>=2.15.0"
]
full = [
    "furcate-nano[hardware,ml,storage]"
]

[project.urls]
Homepage = "https://github.com/furcate-team/furcate-nano"
Documentation = "https://docs.furcate-nano.org"
Repository = "https://github.com/furcate-team/furcate-nano"
Issues = "https://github.com/furcate-team/furcate-nano/issues"
Changelog = "https://github.com/furcate-team/furcate-nano/blob/main/CHANGELOG.md"
Community = "https://discord.gg/furcate-nano"

[project.scripts]
furcate-nano = "furcate_nano.cli:main"

[tool.setuptools.packages.find]
include = ["furcate_nano*"]

[tool.setuptools.package-data]
furcate_nano = ["configs/*.yaml", "models/*.tflite", "scripts/*.sh"]

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
asyncio_mode = "auto"

# ============================================================================
# Makefile - Development and deployment commands
# ============================================================================

.PHONY: help install test lint format clean build deploy setup-pi

help:
	@echo "🌿 Furcate Nano Development Commands:"
	@echo "  install     - Install development dependencies"
	@echo "  test        - Run unit tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build distribution packages"
	@echo "  setup-pi    - Setup Raspberry Pi for deployment"
	@echo "  deploy      - Deploy to Raspberry Pi"

install:
	pip install -e .[dev,ml,full]
	pre-commit install

test:
	pytest tests/ -v --cov=furcate_nano --cov-report=html

test-fast:
	pytest tests/ -v -x --ff

lint:
	flake8 furcate_nano tests
	mypy furcate_nano

format:
	black furcate_nano tests
	isort furcate_nano tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Raspberry Pi deployment targets
setup-pi:
	@echo "Setting up Raspberry Pi 5 for Furcate Nano..."
	curl -sSL https://raw.githubusercontent.com/furcate-team/furcate-nano/main/scripts/setup-raspberry-pi.sh | bash

deploy:
	@echo "Deploying Furcate Nano to Raspberry Pi..."
	@read -p "Enter Pi IP address: " PI_IP && \
	scp -r . pi@$$PI_IP:/tmp/furcate-nano && \
	ssh pi@$$PI_IP "cd /tmp/furcate-nano && sudo python setup.py install"

# Docker targets
docker-build:
	docker build -t furcate-nano:latest .

docker-run:
	docker run --rm -it -p 8080:8080 furcate-nano:latest

docker-compose-up:
	docker-compose -f docker/docker-compose.yml up

docker-compose-down:
	docker-compose -f docker/docker-compose.yml down

# Development utilities
dev-server:
	furcate-nano start --config configs/development.yaml --verbose

simulate:
	furcate-nano start --config configs/simulation.yaml --daemon

monitor:
	tail -f /data/furcate-nano/logs/furcate-nano.log

# Release targets
tag-version:
	@echo "Current version: $$(grep __version__ furcate_nano/__init__.py)"
	@read -p "Enter new version: " VERSION && \
	sed -i "s/__version__ = \".*\"/__version__ = \"$$VERSION\"/" furcate_nano/__init__.py && \
	git add furcate_nano/__init__.py && \
	git commit -m "Bump version to $$VERSION" && \
	git tag "v$$VERSION"

release: clean build
	twine upload dist/*

# ============================================================================
# CONTRIBUTING.md - Contribution guidelines
# ============================================================================

# Contributing to Furcate Nano 🌿

Thank you for your interest in contributing to Furcate Nano! This document provides guidelines for contributing to our open-source environmental monitoring framework.

## 🎯 Project Vision

Furcate Nano democratizes environmental monitoring by making it affordable and accessible. We're building the foundation for a planetary-scale environmental intelligence network.

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Raspberry Pi 5 (for hardware testing)
- Git

### Local Development
```bash
# Clone the repository
git clone https://github.com/furcate-team/furcate-nano.git
cd furcate-nano

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,ml,full]

# Install pre-commit hooks
pre-commit install

# Run tests
make test
```

### Docker Development
```bash
# Build development container
make docker-build

# Run development environment
make docker-compose-up
```

## 🧪 Testing

We maintain high test coverage and quality standards:

```bash
# Run all tests
make test

# Run fast tests only
make test-fast

# Run specific test file
pytest tests/test_hardware.py -v

# Run tests with coverage
pytest tests/ --cov=furcate_nano --cov-report=html
```

## 📝 Code Style

We use automated code formatting and linting:

```bash
# Format code
make format

# Run linting
make lint

# Check types
mypy furcate_nano
```

### Style Guidelines
- **Python**: Follow PEP 8, use Black for formatting
- **Documentation**: Use Google-style docstrings
- **Commit Messages**: Use conventional commits format
- **Variable Names**: Descriptive names, avoid abbreviations

## 🏗️ Architecture Guidelines

### Core Principles
1. **Modularity**: Each component should be independently testable
2. **Performance**: Optimize for Raspberry Pi 5 constraints
3. **Reliability**: Graceful degradation and error handling
4. **Sustainability**: Power-efficient and solar-friendly

### Component Structure
```
furcate_nano/
├── hardware/     # Sensor interfaces and GPIO management
├── edge_ml/      # TensorFlow Lite inference
├── mesh/         # Bio-inspired networking
├── power/        # Solar power management
├── storage/      # DuckDB + RocksDB + SQLite
└── protocols/    # Communication protocols
```

## 🔧 Hardware Contributions

### Sensor Support
Adding new sensor support:

1. Add sensor type to `hardware.py` enum
2. Implement sensor interface in hardware manager
3. Add configuration schema
4. Write comprehensive tests
5. Update documentation

### Testing on Hardware
- Use simulation mode for initial development
- Test on actual Raspberry Pi 5 before submitting
- Verify power consumption characteristics
- Test in various environmental conditions

## 🌐 Network Protocol Contributions

### Mesh Networking
- Follow bio-inspired principles (mycelial networks)
- Optimize for low power consumption
- Handle network partitioning gracefully
- Maintain backward compatibility

### Message Protocols
- Use binary headers for efficiency
- Implement compression for large payloads
- Add checksums for data integrity
- Support message deduplication

## 🤖 Machine Learning Contributions

### Model Requirements
- Use TensorFlow Lite format (.tflite)
- Optimize for Raspberry Pi 5 inference
- Maximum model size: 50MB
- Inference time: <1 second per reading

### Model Integration
```python
# Example model integration
class CustomEnvironmentalClassifier:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
    
    async def predict(self, features: np.ndarray) -> Dict[str, float]:
        # Implementation here
        pass
```

## 📊 Storage Contributions

### Database Optimization
- DuckDB for analytics queries
- RocksDB for high-frequency writes
- SQLite for fallback compatibility
- Implement compression and retention policies

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues
2. Test with latest version
3. Reproduce in clean environment
4. Gather system information

### Bug Report Template
```markdown
**Environment:**
- OS: [e.g., Raspberry Pi OS]
- Python Version: [e.g., 3.9.2]
- Furcate Nano Version: [e.g., 1.0.0]
- Hardware: [e.g., Raspberry Pi 5 4GB]

**Description:**
[Clear description of the bug]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Logs:**
```
[Relevant log output]
```
```

## 💡 Feature Requests

### Before Requesting
- Check if feature aligns with project goals
- Consider implementation complexity
- Evaluate impact on performance and power usage

### Feature Request Template
```markdown
**Feature Description:**
[Clear description of the proposed feature]

**Use Case:**
[Why is this feature needed?]

**Implementation Ideas:**
[Any thoughts on how to implement this]

**Alternatives Considered:**
[Other solutions you've considered]
```

## 🚀 Pull Request Process

### Before Submitting
1. **Fork** the repository
2. **Create** a feature branch from `develop`
3. **Implement** your changes
4. **Write** tests for new functionality
5. **Update** documentation
6. **Run** the full test suite
7. **Commit** using conventional commit format

### PR Requirements
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Change log entry added
- [ ] PR description explains changes
- [ ] Linked to relevant issues

### Review Process
1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on target hardware
4. **Approval** and merge

## 📚 Documentation Contributions

### Types of Documentation
- **API Documentation**: Inline docstrings
- **User Guides**: Markdown files in `/docs`
- **Examples**: Working code examples
- **Tutorials**: Step-by-step guides

### Documentation Standards
- Use clear, concise language
- Include practical examples
- Test all code examples
- Keep documentation up-to-date

## 🏷️ Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes

### Release Checklist
- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Test on Raspberry Pi 5
- [ ] Create GitHub release
- [ ] Publish to PyPI
- [ ] Update Docker images

## 🎉 Recognition

Contributors are recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Discord community** contributor role

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/furcate-nano)
- **Issues**: [GitHub Issues](https://github.com/furcate-team/furcate-nano/issues)
- **Email**: opensource@furcate.earth

## 📄 License

By contributing to Furcate Nano, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the future of environmental monitoring! 🌍**