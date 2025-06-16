# Furcate Nano

**Open Source Environmental Edge Computing Framework**

Transform Raspberry Pi 5 into intelligent environmental monitoring nodes with bio-inspired mesh networking.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Raspberry Pi](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.org/)

## What is Furcate Nano?

Furcate Nano is an open-source framework that transforms Raspberry Pi 5 devices into intelligent environmental monitoring nodes. It enables the creation of self-organizing mesh networks for monitoring air quality, soil conditions, weather patterns, and ecosystem health.

**Key Features:**
- Edge AI with real-time environmental analysis using TensorFlow Lite
- Bio-inspired mesh networking with self-healing topology
- Multi-network integration (local P2P + global cloud synchronization)
- Solar-powered autonomous operation
- Multi-database storage (DuckDB + RocksDB + SQLite)

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/praecise/furcate-nano.git
cd furcate-nano

# Install with hardware support
pip install -e ".[hardware,ml,storage]"

# Test installation
python3 test_furcate_nano.py
```

### Start Monitoring
```bash
# Generate default configuration
furcate-nano init --output my-config.yaml

# Start environmental monitoring
furcate-nano start --config my-config.yaml
```

## Hardware Requirements

### Basic Kit (~$250-300)
| Component | Est. Cost | Purpose |
|-----------|-----------|---------|
| Raspberry Pi 5 (4GB) | $60 | Main computing unit |
| MicroSD Card (64GB) | $15 | Operating system + data |
| DHT22 Temperature/Humidity | $8-15 | Climate monitoring |
| BMP280 Pressure Sensor | $3-8 | Atmospheric pressure |
| MQ135 Air Quality Sensor | $5-12 | Gas detection (CO2, NH3, NOx) |
| Soil Moisture Sensor | $5-8 | Soil hydration levels |
| LoRa SX1276 Module | $15-25 | Long-range mesh networking |
| Solar Panel (20W) + Battery | $50-75 | Renewable power system |
| Weatherproof Enclosure | $20-35 | Environmental protection |
| Cables & Connectors | $15-25 | Assembly hardware |

**Note:** Raspberry Pi 5 8GB variant costs $80 and 16GB variant costs $120 for higher-performance applications.

### Supported Sensors

**Environmental Sensors:**
- **Temperature/Humidity**: DHT22, SHT30, AM2320
- **Atmospheric Pressure**: BMP280, BME680
- **Air Quality**: MQ135, MQ7, SGP30, CCS811
- **Soil Monitoring**: Capacitive moisture sensors, pH sensors
- **Light**: TSL2561, BH1750
- **Sound**: MEMS microphones

**Networking Hardware:**
- **LoRa**: SX1276, SX1262, RFM95W modules
- **WiFi**: Built-in Raspberry Pi 5 WiFi
- **Bluetooth**: Built-in Pi 5 Bluetooth
- **Cellular**: USB 4G/5G modems

## Architecture

### System Components
```
┌─────────────────────────────────────────────────────────────────┐
│                    FURCATE NANO NODE                            │
├─────────────────────────────────────────────────────────────────┤
│  Hardware Layer                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   Sensors    │ │   LoRa/WiFi  │ │ Solar+Battery│            │
│  │ DHT22,BMP280 │ │   Bluetooth  │ │   Management │            │
│  │ MQ135,Soil   │ │   Cellular   │ │   + Charging │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Edge Computing Layer                                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Sensor Fusion│ │   Edge ML    │ │ Data Storage │            │
│  │ Multi-sensor │ │ TensorFlow   │ │ DuckDB+Rocks │            │
│  │  Validation  │ │ Lite Models  │ │  DB+SQLite   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Networking Layer                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Mesh Network│ │ Tenzro P2P   │ │Web Integration│           │
│  │ Bio-inspired │ │Multi-Cloud   │ │REST/MQTT/WS  │            │
│  │ Self-healing │ │ Global Sync  │ │   Webhooks   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## AI & Machine Learning

### Edge AI Capabilities
- **Environmental Classification**: Normal, moderate, polluted, extreme conditions
- **Anomaly Detection**: Statistical and ML-based outlier identification
- **Collaborative Learning**: Peer-to-peer model sharing and validation
- **Real-time Inference**: Sub-second processing on Raspberry Pi 5

### Supported Models
- **TensorFlow Lite**: Optimized for Raspberry Pi 5's quad-core ARM Cortex-A76 processor @ 2.4GHz
- **Environmental Classifier**: Multi-class environmental condition detection
- **Anomaly Detector**: Unsupervised outlier detection

## Networking Protocols

### 1. Bio-Inspired Mesh Network
- Self-healing topology that adapts to node failures
- Organic growth patterns based on mycelial network principles
- Environmental zone clustering for relevant data sharing

### 2. Tenzro Network (Global P2P)
- Multi-cloud integration: GCP, AWS, Azure, Tenzro Cloud
- Encrypted P2P channels with PBKDF2 key derivation
- Global data synchronization with local peer discovery

### 3. Furcate Network (Local P2P)
- **WiFi Direct**: High-bandwidth device-to-device communication
- **Bluetooth LE**: Low-power proximity networking
- **LoRa Mesh**: Long-range, low-power networking (1-10km)
- **UDP Multicast**: Local network auto-discovery

### 4. Web Integrations
- **REST API**: Full-featured HTTP API with FastAPI
- **MQTT**: IoT platform integration
- **WebSockets**: Real-time data streaming
- **Webhooks**: External system notifications

## Power Management

### Solar-Powered Operation
- Intelligent power modes: Normal → Balanced → Low-Power → Emergency
- Optimized for Raspberry Pi 5's improved power efficiency
- CPU frequency scaling for power optimization
- 24/7 autonomous operation capability

## Configuration Examples

### Agricultural Monitoring
```yaml
# configs/agriculture.yaml
device:
  environmental_zone: "agricultural"
  
hardware:
  sensors:
    soil_moisture:
      enabled: true
      measurement_depth_cm: 20
    soil_ph:
      enabled: true
      
monitoring:
  interval_seconds: 900  # 15 minutes
  alert_thresholds:
    soil_moisture:
      moisture: [40, 80]  # Irrigation thresholds
```

### Urban Air Quality
```yaml
# configs/urban.yaml
device:
  environmental_zone: "urban_center"
  
hardware:
  sensors:
    air_quality:
      enabled: true
    particulate_matter:
      enabled: true
    sound_level:
      enabled: true
      
monitoring:
  interval_seconds: 180  # 3 minutes
  alert_thresholds:
    air_quality:
      aqi: [0, 150]
```

## Development

### Development Setup
```bash
# Clone repository
git clone https://github.com/praecise/furcate-nano.git
cd furcate-nano

# Create development environment
python3 -m venv furcate-dev
source furcate-dev/bin/activate

# Install development dependencies
pip install -e ".[dev,full]"

# Run test suite
pytest tests/
python3 test_furcate_nano.py
```

### Adding Custom Sensors
```python
# furcate_nano/sensors/custom_sensor.py
from furcate_nano.hardware import SensorReading, SensorType

class CustomSensor:
    def __init__(self, config):
        self.config = config
    
    async def read(self) -> SensorReading:
        # Your sensor reading logic
        value = await self.read_custom_hardware()
        
        return SensorReading(
            sensor_type=SensorType.CUSTOM,
            timestamp=time.time(),
            value={"measurement": value},
            unit="custom_unit",
            quality=0.95,
            confidence=0.90
        )
```

## Documentation & Support

### Documentation
The codebase includes comprehensive inline documentation and configuration examples in the `configs/` directory.

### Community & Support
- **GitHub Issues**: [Report bugs and request features](https://github.com/praecise/furcate-nano/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/praecise/furcate-nano/discussions)

### Contributing
We welcome contributions! Check the repository for contribution guidelines.

```bash
# Fork the repository
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python3 test_furcate_nano.py
pytest tests/

# Submit pull request
```



## License & Commercial Use

**Open Source License**: MIT License for research, education, and non-commercial use.

**Commercial Licensing**: 
- **Open Source**: Free for research, education, and non-commercial use under MIT License
- **Commercial**: Contact repository maintainers for commercial licensing inquiries

## Contact & Links

- **Repository**: [https://github.com/praecise/furcate-nano](https://github.com/praecise/furcate-nano)

---

*Built by the Furcate community. Democratizing environmental monitoring, one node at a time.*