# Furcate Nano

Open Source Environmental Edge Computing Framework for Research and Education

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-ARM64%20%7C%20x86__64-lightgrey.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## Overview

Furcate Nano is the educational and research version of Furcate's distributed environmental intelligence platform. Transform embedded devices like Raspberry Pi and NVIDIA Jetson Nano into intelligent environmental monitoring nodes capable of:

- **Real-time Environmental Monitoring**: Temperature, humidity, air quality, and more
- **Edge AI and Machine Learning**: On-device intelligence for environmental analysis
- **Mesh Networking**: Self-organizing device networks for collaborative monitoring
- **Educational Integration**: Classroom-ready with curriculum alignment
- **Research Applications**: Flexible framework for environmental research

**For production environmental monitoring in vulnerable communities, see the commercial [Furcate platform](https://furcate.earth)**

## Quick Start

### Hardware Requirements

**Minimum Requirements:**
- **Raspberry Pi 4B** (4GB RAM) or **NVIDIA Jetson Nano** (4GB)
- **MicroSD Card**: 32GB+ (Class 10)
- **Power Supply**: Official power adapter
- **Network**: WiFi or Ethernet connectivity

**Recommended Hardware:**
- **Raspberry Pi 5** (8GB RAM) or **NVIDIA Jetson Orin Nano**
- **Environmental Sensors**: DHT22, BME280, SDS011, etc.
- **LoRa Module**: For long-range mesh networking
- **Solar Panel + Battery**: For remote deployment

### Software Installation

#### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+ and dependencies
sudo apt install python3 python3-pip python3-venv git -y

# Create virtual environment
python3 -m venv furcate-nano-env
source furcate-nano-env/bin/activate
```

#### 2. Install Furcate Nano

```bash
# Clone repository
git clone https://github.com/praecise/furcate-nano.git
cd furcate-nano

# Install core package
pip install -e .

# Install platform-specific dependencies
# For Raspberry Pi:
pip install -r requirements-raspberry-pi.txt

# For NVIDIA Jetson:
pip install -r requirements-jetson.txt

# For educational environments:
pip install -r requirements-educational.txt
```

#### 3. Initial Configuration

```bash
# Generate default configuration
furcate-nano init --device-type raspberry_pi

# Start with simulation mode (no hardware required)
furcate-nano start --simulation

# View dashboard at http://localhost:8080
```

## Features

### Core Capabilities

- **Environmental Monitoring**: Multi-sensor data collection with automatic calibration
- **Edge Machine Learning**: TensorFlow Lite and PyTorch models for environmental analysis
- **Mesh Networking**: LoRa, Bluetooth, and WiFi-based device collaboration
- **Power Management**: Solar charging and intelligent power optimization
- **Data Storage**: Local SQLite with cloud synchronization options
- **Protocol Support**: MQTT, HTTP, and custom Furcate protocols

### Educational Features

- **Classroom Integration**: Ready-made lesson plans and experiments
- **Real-time Dashboards**: Interactive visualization for learning
- **Student Projects**: Structured environmental monitoring projects
- **NGSS Alignment**: Supports Next Generation Science Standards
- **Safety Features**: Educational mode with restricted permissions

### Research Applications

- **Distributed Sensing**: Large-scale environmental monitoring networks
- **Custom ML Models**: Framework for developing environmental AI
- **Data Collection**: High-quality datasets for research
- **Blockchain Integration**: Immutable environmental data records
- **API Access**: Comprehensive REST and GraphQL APIs

## Tutorials

Comprehensive tutorials are available in the `/docs/tutorials/` directory:

1. **[Classroom Setup](docs/tutorials/classroom.md)**: Complete guide for educational environments
2. **[Custom ML Models](docs/tutorials/custom_ml_models.md)**: Develop environmental AI models
3. **[Multi-device Collaboration](docs/tutorials/multi_device_collaboration.md)**: Mesh networking and data sharing
4. **[Advanced Weather AI](docs/tutorials/advanced_weather_ai.md)**: Integration with global weather models
5. **[Blockchain Integration](docs/tutorials/blockchain_integration.md)**: Secure, verifiable environmental data

## Configuration

### Basic Configuration

```yaml
# config.yaml
device:
  id: "classroom-device-1"
  name: "Environmental Monitor 1"
  location:
    latitude: 40.7128
    longitude: -74.0060
    description: "Science Classroom"

hardware:
  simulation: false  # Set to true for development
  sensors:
    temperature_humidity:
      type: "DHT22"
      pin: 4
      enabled: true
    air_quality:
      type: "SDS011"
      port: "/dev/ttyUSB0"
      enabled: true

ml:
  simulation: false
  models:
    environmental_classifier:
      enabled: true
      model_path: "models/environmental_classifier.tflite"

monitoring:
  interval_seconds: 60
  alert_thresholds:
    temperature: [-10, 50]
    humidity: [0, 100]
    air_quality: [0, 200]
```

### Educational Configuration

```yaml
# classroom-config.yaml
device:
  educational_mode: true
  safety_restrictions: true

integrations:
  classroom_dashboard:
    enabled: true
    url: "https://dashboard.school.edu"
  google_classroom:
    enabled: true
    course_id: "12345"

mesh:
  educational_collaboration:
    experiment_sharing: true
    data_comparison: true
    peer_validation: true
```

## API Reference

### REST API

```python
# Start monitoring
POST /api/v1/monitoring/start

# Get sensor data
GET /api/v1/sensors/data?device_id=classroom-1&start=2024-01-01

# Environmental alerts
GET /api/v1/alerts?severity=warning

# ML predictions
POST /api/v1/ml/predict
{
  "sensor_data": {
    "temperature": 23.5,
    "humidity": 45.0,
    "air_quality": 75
  }
}
```

### Python SDK

```python
from furcate_nano import FurcateNanoCore, NanoConfig

# Initialize device
config = NanoConfig.from_file("config.yaml")
device = FurcateNanoCore(config)

# Start monitoring
await device.start()

# Get sensor readings
readings = await device.hardware.read_sensors()
print(f"Temperature: {readings['temperature']}°C")

# Run ML inference
results = await device.ml.analyze(readings)
print(f"Environment classification: {results['classification']}")
```

## Development

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone with development extras
git clone https://github.com/praecise/furcate-nano.git
cd furcate-nano

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run linting
flake8 furcate_nano/
black furcate_nano/
```

### Testing

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests (requires hardware or simulation)
python -m pytest tests/integration/

# Educational tests
python -m pytest tests/educational/
```

## Deployment

### Single Device

```bash
# Production deployment
furcate-nano deploy --config production-config.yaml --service

# Educational deployment
furcate-nano deploy --config classroom-config.yaml --educational
```

### Classroom Network

```bash
# Deploy classroom network (6 devices)
./scripts/deploy_classroom_network.sh

# Monitor classroom status
furcate-nano classroom status

# Update all classroom devices
furcate-nano classroom update
```

### Research Network

```bash
# Large-scale research deployment
furcate-nano research deploy --nodes 50 --region us-west

# Data synchronization
furcate-nano research sync --target research-cluster
```

## Supported Platforms

### Embedded Hardware

- **Raspberry Pi**: 4B, 5, Zero 2W (ARM64)
- **NVIDIA Jetson**: Nano, Xavier NX, Orin Nano
- **x86 Systems**: Intel NUC, mini PCs
- **Orange Pi**: 5, 5 Plus (experimental)

### Operating Systems

- **Raspberry Pi OS**: 64-bit (recommended)
- **Ubuntu**: 20.04+ LTS (ARM64/x86_64)
- **Debian**: 11+ (ARM64/x86_64)
- **NVIDIA JetPack**: 5.0+ (Jetson devices)

### Python Versions

- **Python**: 3.8, 3.9, 3.10, 3.11
- **Dependencies**: See requirements files for platform-specific versions

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

### Documentation

- **Tutorials**: Complete guides in `/docs/tutorials/`
- **API Reference**: `/docs/api/`
- **Configuration**: `/docs/configuration/`
- **Hardware Guides**: `/docs/hardware/`

### Community

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/praecise/furcate-nano/issues)
- **Discussions**: [Community forum](https://github.com/praecise/furcate-nano/discussions)
- **Educational Support**: [Contact Praecise Education](mailto:education@praecise.com)

### Commercial Support

For production environmental monitoring solutions, visit [Furcate.earth](https://furcate.earth) or contact [Praecise Ltd](https://praecise.com).

## Acknowledgments

Furcate Nano is developed by [Praecise Ltd](https://praecise.com) as part of the Furcate environmental intelligence ecosystem. This open-source version is designed specifically for educational and research use.

**Special thanks to:**
- Educational partners for classroom testing
- Research institutions for environmental data validation
- Open source community for contributions
- Environmental scientists for domain expertise

---

**Transform embedded devices into environmental intelligence. Start monitoring today.**