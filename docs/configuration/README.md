# Furcate Nano Configuration Guide

Comprehensive configuration documentation for the Furcate Nano environmental edge computing framework.

## Table of Contents

- [Overview](#overview)
- [Configuration File Structure](#configuration-file-structure)
- [Device Configuration](#device-configuration)
- [Hardware Configuration](#hardware-configuration)
- [Machine Learning Configuration](#machine-learning-configuration)
- [Mesh Networking Configuration](#mesh-networking-configuration)
- [Power Management Configuration](#power-management-configuration)
- [Storage Configuration](#storage-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Protocol Configuration](#protocol-configuration)
- [Integration Configuration](#integration-configuration)
- [Educational Configuration](#educational-configuration)
- [Environment-Specific Configurations](#environment-specific-configurations)
- [Configuration Management](#configuration-management)

## Overview

Furcate Nano uses YAML configuration files to define device behavior, hardware settings, and operational parameters. Configuration follows a hierarchical structure with sensible defaults and environment-specific overrides.

### Configuration File Format

All configuration files use YAML format for human readability and ease of editing:

```yaml
# config.yaml - Main device configuration
device:
  id: "nano-device-001"
  name: "Environmental Monitor 1"

hardware:
  simulation: false
  sensors:
    temperature_humidity:
      enabled: true

# Additional sections...
```

### Configuration Loading

```python
from furcate_nano import NanoConfig, load_config

# Load from file
config = load_config("config.yaml")

# Create with defaults
config = NanoConfig()

# Save configuration
config.save_config(config, "saved-config.yaml")
```

## Configuration File Structure

### Complete Configuration Template

```yaml
# ============================================================================
# Furcate Nano Device Configuration
# ============================================================================

# Device identification and basic settings
device:
  id: "nano-device-001"
  name: "Environmental Monitor 1"
  location:
    latitude: 40.7128
    longitude: -74.0060
    description: "Science Classroom"
  educational_mode: false
  safety_restrictions: false

# Hardware configuration
hardware:
  simulation: false
  platform: "raspberry_pi"
  sensors:
    temperature_humidity:
      type: "DHT22"
      pin: 4
      enabled: true
      calibration:
        temperature_offset: 0.0
        humidity_offset: 0.0
    air_quality:
      type: "SDS011"
      port: "/dev/ttyUSB0"
      enabled: true
      calibration:
        pm25_factor: 1.0
        pm10_factor: 1.0
  gpio_pins:
    dht22_data: 4
    status_led: 25
    lora_reset: 22
    lora_dio0: 18
    moisture_power: 24

# Machine learning configuration
ml:
  simulation: false
  model_path: "/opt/furcate-nano/models"
  models:
    environmental_classifier:
      enabled: true
      confidence_threshold: 0.7
    anomaly_detector:
      enabled: true
      sensitivity: 0.8
  processing:
    enable_edge_inference: true
    batch_processing: false
    
# Mesh networking configuration
mesh:
  simulation: true
  max_connections: 8
  discovery_interval: 60
  environmental_zone: "classroom"
  protocols:
    lora:
      enabled: true
      frequency: 915
      spreading_factor: 7
      bandwidth: 125
    bluetooth:
      enabled: true
      max_range_m: 100

# Power management configuration
power:
  simulation: true
  battery:
    capacity_mah: 10000
    voltage_min: 3.0
    voltage_max: 4.2
  solar:
    panel_watts: 20
    voltage_max: 6.0
  thresholds:
    low_battery: 0.2
    critical_battery: 0.1
    emergency_battery: 0.05

# Storage configuration
storage:
  db_path: "/data/furcate_nano.db"
  data_path: "/data/sensor_data"
  retention_days: 30
  max_size_mb: 1000

# Monitoring configuration
monitoring:
  interval_seconds: 60
  default_interval_seconds: 60
  alert_thresholds:
    temperature_humidity:
      temperature: [-10, 50]
      humidity: [0, 100]
    air_quality:
      aqi: [0, 200]

# Protocol configuration
protocol:
  version: "1.0"
  master_nodes: []
  asset_creation_enabled: false
  compression_enabled: true

# Web integrations
integrations:
  rest_api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    cors_origins: ["*"]
  mqtt:
    enabled: false
    broker: "localhost"
    port: 1883
    username: ""
    password: ""
    base_topic: "furcate"
  websocket:
    enabled: false
    host: "localhost"
    port: 8765
  webhooks:
    enabled: false
    endpoints: []

# Global settings
debug: false
log_level: "INFO"
```

## Device Configuration

### Basic Device Settings

```yaml
device:
  id: "nano-device-001"                    # Unique device identifier
  name: "Environmental Monitor 1"          # Human-readable device name
  location:
    latitude: 40.7128                       # GPS latitude
    longitude: -74.0060                     # GPS longitude
    description: "Science Classroom"       # Location description
  educational_mode: false                   # Enable educational features
  safety_restrictions: false               # Enable safety restrictions
```

### Educational Mode Settings

```yaml
device:
  educational_mode: true
  safety_restrictions: true
  classroom_id: "science-room-101"
  teacher_contact: "teacher@school.edu"
  student_access_level: "read_only"        # Options: read_only, limited, full
```

### Location Configuration

```yaml
device:
  location:
    latitude: 40.7128
    longitude: -74.0060
    altitude_m: 10.5                        # Optional altitude in meters
    description: "Science Classroom"
    building: "Main Building"               # Optional building identifier
    room: "101"                            # Optional room number
    zone: "indoor"                         # Environment zone: indoor/outdoor
```

## Hardware Configuration

### Platform Settings

```yaml
hardware:
  simulation: false                         # Enable simulation mode
  platform: "raspberry_pi"                 # Platform: raspberry_pi, jetson_nano, x86
  board_revision: "5"                       # Board revision (for Raspberry Pi)
  cpu_governor: "ondemand"                  # CPU power management
```

### Sensor Configuration

#### Temperature and Humidity Sensors

```yaml
hardware:
  sensors:
    temperature_humidity:
      type: "DHT22"                         # Sensor type: DHT22, DHT11, SHT30, BME280
      pin: 4                               # GPIO pin number
      enabled: true                        # Enable this sensor
      read_interval: 2                     # Minimum seconds between readings
      calibration:
        temperature_offset: 0.0            # Temperature calibration offset
        humidity_offset: 0.0               # Humidity calibration offset
        temperature_scale: 1.0             # Temperature scaling factor
        humidity_scale: 1.0                # Humidity scaling factor
      validation:
        temperature_range: [-40, 80]       # Valid temperature range
        humidity_range: [0, 100]           # Valid humidity range
```

#### Air Quality Sensors

```yaml
hardware:
  sensors:
    air_quality:
      type: "SDS011"                        # Sensor type: SDS011, PMS5003, BME680
      port: "/dev/ttyUSB0"                 # Serial port
      baudrate: 9600                       # Serial baudrate
      enabled: true
      calibration:
        pm25_factor: 1.0                   # PM2.5 calibration factor
        pm10_factor: 1.0                   # PM10 calibration factor
      validation:
        pm25_max: 500                      # Maximum valid PM2.5 value
        pm10_max: 500                      # Maximum valid PM10 value
```

#### Pressure Sensors

```yaml
hardware:
  sensors:
    pressure:
      type: "BMP280"                        # Sensor type: BMP280, BME280
      i2c_address: 0x77                    # I2C address
      enabled: true
      calibration:
        pressure_offset: 0.0               # Pressure calibration offset
        altitude_correction: true          # Enable altitude correction
        reference_altitude: 0              # Reference altitude in meters
```

#### Light Sensors

```yaml
hardware:
  sensors:
    light:
      type: "TSL2561"                       # Sensor type: TSL2561, BH1750
      i2c_address: 0x39                    # I2C address
      enabled: true
      gain: "AUTO"                         # Gain setting: AUTO, 1X, 16X
      integration_time: "402MS"            # Integration time
```

### GPIO Pin Configuration

```yaml
hardware:
  gpio_pins:
    dht22_data: 4                          # DHT22 data pin
    status_led: 25                         # Status LED pin
    power_button: 3                        # Power button pin
    reset_button: 2                        # Reset button pin
    lora_reset: 22                         # LoRa reset pin
    lora_dio0: 18                         # LoRa DIO0 pin
    lora_cs: 8                            # LoRa chip select pin
    moisture_power: 24                     # Soil moisture power pin
    buzzer: 21                            # Buzzer pin
```

### Interface Configuration

```yaml
hardware:
  interfaces:
    i2c:
      enabled: true
      bus: 1                              # I2C bus number
      frequency: 100000                    # I2C frequency (Hz)
    spi:
      enabled: true
      bus: 0                              # SPI bus number
      device: 0                           # SPI device number
    uart:
      enabled: true
      port: "/dev/ttyS0"                  # UART port
      baudrate: 9600                      # UART baudrate
```

## Machine Learning Configuration

### Model Settings

```yaml
ml:
  simulation: false                         # Use simulated ML models
  model_path: "/opt/furcate-nano/models"   # Path to model files
  cache_size_mb: 512                      # Model cache size
  
  models:
    environmental_classifier:
      enabled: true                        # Enable this model
      model_file: "env_classifier.tflite" # Model filename
      confidence_threshold: 0.7           # Minimum confidence for predictions
      input_features:                     # Input feature configuration
        - "temperature"
        - "humidity" 
        - "air_quality"
        - "time_of_day"
      preprocessing:
        normalize: true                    # Normalize input features
        feature_scaling: "standard"       # Scaling method: standard, minmax
    
    anomaly_detector:
      enabled: true
      model_file: "anomaly_detector.pkl"
      sensitivity: 0.8                    # Anomaly detection sensitivity
      contamination: 0.05                 # Expected outlier fraction
      lookback_window: 24                 # Hours of data for context
```

### Processing Configuration

```yaml
ml:
  processing:
    enable_edge_inference: true           # Enable on-device inference
    batch_processing: false               # Process data in batches
    max_inference_time_ms: 1000          # Maximum inference time
    parallel_processing: true            # Enable parallel model execution
    model_warmup: true                   # Warm up models on startup
    
  optimization:
    quantization: true                    # Use quantized models
    pruning: false                       # Use pruned models
    acceleration: "auto"                 # Hardware acceleration: auto, cpu, gpu
```

### Training Configuration

```yaml
ml:
  training:
    enabled: false                       # Enable on-device training
    update_interval_hours: 24            # Model update interval
    min_samples: 1000                    # Minimum samples for training
    validation_split: 0.2                # Validation data split
    
  federated_learning:
    enabled: false                       # Enable federated learning
    aggregation_server: ""               # Aggregation server URL
    update_frequency: "daily"            # Update frequency
```

## Mesh Networking Configuration

### Basic Mesh Settings

```yaml
mesh:
  simulation: true                         # Use simulated mesh network
  max_connections: 8                      # Maximum peer connections
  discovery_interval: 60                  # Device discovery interval (seconds)
  heartbeat_interval: 30                  # Heartbeat interval (seconds)
  environmental_zone: "classroom"         # Environmental zone identifier
  
  network_topology: "peer_to_peer"       # Topology: peer_to_peer, star, mesh
  auto_discovery: true                    # Enable automatic peer discovery
  connection_timeout: 30                  # Connection timeout (seconds)
```

### Protocol Configuration

```yaml
mesh:
  protocols:
    lora:
      enabled: true                       # Enable LoRa protocol
      frequency: 915                      # Frequency (MHz): 433, 868, 915
      spreading_factor: 7                 # Spreading factor: 6-12
      bandwidth: 125                      # Bandwidth (kHz): 7.8, 10.4, 15.6, 20.8, 31.25, 41.7, 62.5, 125, 250, 500
      coding_rate: 5                      # Coding rate: 5-8
      tx_power: 17                        # Transmit power (dBm): 2-17
      sync_word: 0x12                     # Sync word
      preamble_length: 8                  # Preamble length
    
    bluetooth:
      enabled: true                       # Enable Bluetooth protocol
      max_range_m: 100                   # Maximum range in meters
      scan_interval: 10                   # Scan interval (seconds)
      advertising_interval: 5             # Advertising interval (seconds)
      connection_interval: 15             # Connection interval (ms)
    
    wifi_direct:
      enabled: false                      # Enable WiFi Direct
      ssid_prefix: "FurcateNano"         # SSID prefix for WiFi Direct
      passphrase: "environmental123"      # WiFi Direct passphrase
      channel: 6                         # WiFi channel
```

### Data Sharing Configuration

```yaml
mesh:
  data_sharing:
    enabled: true                        # Enable data sharing
    share_interval: 300                  # Data sharing interval (seconds)
    max_message_size: 1024              # Maximum message size (bytes)
    compression_enabled: true           # Enable data compression
    
    shared_data_types:                  # Types of data to share
      - "sensor_readings"
      - "ml_predictions"
      - "alerts"
      - "status_updates"
    
    privacy:
      anonymize_location: false         # Anonymize location data
      encrypt_data: true                # Encrypt shared data
      data_retention_hours: 24          # How long to retain shared data
```

## Power Management Configuration

### Battery Configuration

```yaml
power:
  simulation: true                        # Use simulated power management
  
  battery:
    type: "lithium_ion"                  # Battery type
    capacity_mah: 10000                  # Battery capacity (mAh)
    voltage_nominal: 3.7                 # Nominal voltage (V)
    voltage_min: 3.0                     # Minimum safe voltage (V)
    voltage_max: 4.2                     # Maximum voltage (V)
    
    charging:
      max_current_ma: 2000               # Maximum charging current (mA)
      termination_voltage: 4.15          # Charge termination voltage (V)
      trickle_threshold: 3.0             # Trickle charge threshold (V)
    
    monitoring:
      voltage_pin: 0                     # ADC pin for voltage monitoring
      current_pin: 1                     # ADC pin for current monitoring
      temperature_pin: 2                 # ADC pin for temperature monitoring
```

### Solar Panel Configuration

```yaml
power:
  solar:
    enabled: true                        # Enable solar charging
    panel_watts: 20                      # Panel power rating (W)
    voltage_max: 6.0                     # Maximum panel voltage (V)
    mppt_enabled: true                   # Enable MPPT charging
    
    tracking:
      voltage_pin: 3                     # ADC pin for panel voltage
      current_pin: 4                     # ADC pin for panel current
      irradiance_pin: 5                  # ADC pin for irradiance sensor
```

### Power Management Thresholds

```yaml
power:
  thresholds:
    low_battery: 0.2                     # Low battery threshold (20%)
    critical_battery: 0.1                # Critical battery threshold (10%)
    emergency_battery: 0.05              # Emergency battery threshold (5%)
    charging_complete: 0.95              # Charging complete threshold (95%)
    
  power_modes:
    normal:
      monitoring_interval: 60            # Normal monitoring interval (seconds)
      ml_processing: true                # Enable ML processing
      mesh_networking: true              # Enable mesh networking
    
    low_power:
      monitoring_interval: 300           # Extended monitoring interval (seconds)
      ml_processing: false               # Disable ML processing
      mesh_networking: false             # Disable mesh networking
      
    emergency:
      monitoring_interval: 3600          # Emergency monitoring interval (seconds)
      sensors_enabled: ["temperature"]   # Only essential sensors
      alert_only: true                   # Only send alerts
```

## Storage Configuration

### Database Configuration

```yaml
storage:
  db_path: "/data/furcate_nano.db"       # SQLite database path
  data_path: "/data/sensor_data"          # Raw data storage path
  backup_path: "/data/backups"            # Backup storage path
  
  retention:
    sensor_data_days: 30                 # Sensor data retention (days)
    ml_results_days: 7                   # ML results retention (days)
    alerts_days: 90                      # Alert retention (days)
    logs_days: 14                        # Log retention (days)
  
  limits:
    max_size_mb: 1000                    # Maximum storage size (MB)
    max_records_per_table: 100000       # Maximum records per table
    compression_enabled: true            # Enable data compression
```

### Backup Configuration

```yaml
storage:
  backup:
    enabled: true                        # Enable automatic backups
    interval_hours: 24                   # Backup interval (hours)
    max_backups: 7                       # Maximum backup files to keep
    compress_backups: true               # Compress backup files
    
    remote_backup:
      enabled: false                     # Enable remote backup
      type: "s3"                         # Remote backup type: s3, ftp, ssh
      endpoint: ""                       # Remote endpoint
      credentials_file: ""               # Credentials file path
```

## Monitoring Configuration

### Sensor Monitoring

```yaml
monitoring:
  interval_seconds: 60                   # Default monitoring interval
  default_interval_seconds: 60           # Fallback interval
  
  adaptive_intervals:
    enabled: true                        # Enable adaptive monitoring
    min_interval: 30                     # Minimum interval (seconds)
    max_interval: 300                    # Maximum interval (seconds)
    change_threshold: 0.1                # Change threshold for adaptation
  
  quality_control:
    enabled: true                        # Enable data quality control
    outlier_detection: true              # Enable outlier detection
    smoothing_window: 5                  # Data smoothing window size
    validation_rules: true               # Enable validation rules
```

### Alert Thresholds

```yaml
monitoring:
  alert_thresholds:
    temperature_humidity:
      temperature: [-10, 50]             # Temperature range (°C)
      humidity: [0, 100]                 # Humidity range (%)
      rate_of_change: 5                  # Maximum rate of change per hour
    
    air_quality:
      pm25: [0, 35]                      # PM2.5 range (μg/m³)
      pm10: [0, 150]                     # PM10 range (μg/m³)
      aqi: [0, 200]                      # AQI range
    
    pressure:
      atmospheric: [950, 1050]           # Pressure range (hPa)
      
    light:
      illuminance: [0, 100000]           # Light range (lux)
```

### Alert Configuration

```yaml
monitoring:
  alerts:
    enabled: true                        # Enable alerting
    cooldown_minutes: 15                 # Alert cooldown period
    escalation_enabled: true             # Enable alert escalation
    
    severity_levels:
      info:
        threshold_multiplier: 0.8        # 80% of threshold
        notification_delay: 300          # 5-minute delay
      warning:
        threshold_multiplier: 1.0        # 100% of threshold
        notification_delay: 60           # 1-minute delay
      critical:
        threshold_multiplier: 1.2        # 120% of threshold
        notification_delay: 0            # Immediate notification
    
    notifications:
      local_display: true                # Show on local display
      mqtt_publish: true                 # Publish via MQTT
      webhook_send: true                 # Send to webhooks
      mesh_broadcast: true               # Broadcast to mesh network
```

## Protocol Configuration

### Furcate Protocol Settings

```yaml
protocol:
  version: "1.0"                         # Protocol version
  compression_enabled: true              # Enable message compression
  encryption_enabled: true              # Enable message encryption
  
  message_format: "json"                 # Message format: json, msgpack, protobuf
  max_message_size: 4096                 # Maximum message size (bytes)
  message_timeout: 30                    # Message timeout (seconds)
  
  asset_creation_enabled: false          # Enable asset creation
  master_nodes: []                       # List of master node addresses
  
  reliability:
    enable_acknowledgments: true         # Enable message acknowledgments
    retry_attempts: 3                    # Number of retry attempts
    retry_delay_ms: 1000                 # Retry delay (milliseconds)
```

## Integration Configuration

### REST API Configuration

```yaml
integrations:
  rest_api:
    enabled: true                        # Enable REST API
    host: "0.0.0.0"                     # Bind host
    port: 8000                          # Bind port
    workers: 1                          # Number of worker processes
    
    cors:
      enabled: true                      # Enable CORS
      origins: ["*"]                     # Allowed origins
      methods: ["GET", "POST", "PUT"]    # Allowed methods
      headers: ["*"]                     # Allowed headers
    
    authentication:
      enabled: false                     # Enable authentication
      api_key_header: "X-API-Key"       # API key header name
      jwt_secret: ""                     # JWT secret key
    
    rate_limiting:
      enabled: false                     # Enable rate limiting
      requests_per_minute: 60            # Requests per minute limit
      burst_size: 10                     # Burst size allowance
```

### MQTT Configuration

```yaml
integrations:
  mqtt:
    enabled: false                       # Enable MQTT integration
    broker: "localhost"                  # MQTT broker address
    port: 1883                          # MQTT broker port
    keepalive: 60                       # Keep-alive interval (seconds)
    
    authentication:
      username: ""                       # MQTT username
      password: ""                       # MQTT password
      ca_cert: ""                        # CA certificate file
      client_cert: ""                    # Client certificate file
      client_key: ""                     # Client key file
    
    topics:
      base_topic: "furcate"              # Base topic prefix
      device_topic: "device"             # Device-specific topic
      sensor_topic: "sensors"            # Sensor data topic
      alert_topic: "alerts"              # Alert topic
    
    qos:
      sensor_data: 0                     # QoS for sensor data
      alerts: 2                          # QoS for alerts
      commands: 1                        # QoS for commands
    
    retain:
      device_status: true                # Retain device status messages
      last_will: true                    # Enable last will message
```

### WebSocket Configuration

```yaml
integrations:
  websocket:
    enabled: false                       # Enable WebSocket server
    host: "localhost"                    # WebSocket host
    port: 8765                          # WebSocket port
    
    connection_limits:
      max_connections: 100               # Maximum concurrent connections
      connection_timeout: 300            # Connection timeout (seconds)
      ping_interval: 30                  # Ping interval (seconds)
      ping_timeout: 10                   # Ping timeout (seconds)
    
    data_streaming:
      default_interval: 5                # Default streaming interval (seconds)
      max_interval: 60                   # Maximum streaming interval (seconds)
      min_interval: 1                    # Minimum streaming interval (seconds)
      buffer_size: 1000                  # Message buffer size
```

### Webhook Configuration

```yaml
integrations:
  webhooks:
    enabled: false                       # Enable webhook integration
    
    endpoints:
      - name: "data_webhook"             # Webhook name
        url: "https://api.example.com/data"  # Webhook URL
        enabled: true                    # Enable this webhook
        events: ["sensor_data", "alerts"]  # Events to send
        method: "POST"                   # HTTP method
        headers:                         # Custom headers
          Authorization: "Bearer token"
          Content-Type: "application/json"
        timeout: 30                      # Request timeout (seconds)
        retry_attempts: 3                # Retry attempts
        
    global_settings:
      max_retries: 3                     # Global max retries
      retry_delay_ms: 1000              # Retry delay (milliseconds)
      timeout_seconds: 30                # Global timeout (seconds)
      verify_ssl: true                   # Verify SSL certificates
```

## Educational Configuration

### Classroom Setup

```yaml
# classroom-config.yaml
device:
  educational_mode: true
  safety_restrictions: true
  classroom_id: "science-room-101"
  
educational:
  teacher_settings:
    contact_email: "teacher@school.edu"
    classroom_name: "Environmental Science"
    grade_level: [9, 10, 11]
    
  student_access:
    data_access_level: "read_only"       # read_only, limited, full
    configuration_access: false         # Allow students to modify config
    export_data: true                   # Allow data export
    real_time_view: true                # Allow real-time data viewing
    
  curriculum_alignment:
    standards: ["NGSS-MS-ESS3-3", "NGSS-HS-ESS3-3"]
    subjects: ["environmental_science", "physics", "chemistry"]
    
  safety_features:
    restricted_sensors: []               # Sensors to disable for safety
    max_experiment_duration: 480        # Maximum experiment duration (minutes)
    auto_shutdown_time: "17:00"         # Automatic shutdown time
    alert_teacher_on_errors: true       # Alert teacher on errors
```

### Student Project Configuration

```yaml
educational:
  projects:
    air_quality_study:
      title: "Indoor Air Quality Investigation"
      duration_hours: 48
      participating_devices: ["classroom-1", "classroom-2", "classroom-3"]
      variables: ["temperature", "humidity", "air_quality"]
      data_collection_interval: 300     # 5 minutes
      
      learning_objectives:
        - "Understand factors affecting indoor air quality"
        - "Learn data collection and analysis skills"
        - "Practice scientific method"
        
      assessment_criteria:
        data_quality: 25                 # Points for data quality
        analysis: 35                     # Points for analysis
        presentation: 25                 # Points for presentation
        collaboration: 15                # Points for collaboration
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# development-config.yaml
device:
  id: "dev-device"
  name: "Development Device"

hardware:
  simulation: true                       # Always use simulation in development

ml:
  simulation: true                       # Use simulated ML models

debug: true                             # Enable debug logging
log_level: "DEBUG"                      # Verbose logging

integrations:
  rest_api:
    enabled: true
    port: 8000
  mqtt:
    enabled: false                       # Disable MQTT in development
```

### Production Configuration

```yaml
# production-config.yaml
device:
  id: "prod-device-001"
  name: "Production Environmental Monitor"

hardware:
  simulation: false                      # Use real hardware
  
ml:
  simulation: false                      # Use real ML models
  
debug: false                            # Disable debug features
log_level: "INFO"                       # Standard logging

storage:
  retention:
    sensor_data_days: 90                # Longer retention for production
    
integrations:
  rest_api:
    enabled: true
    authentication:
      enabled: true                      # Enable authentication in production
  mqtt:
    enabled: true                        # Enable MQTT in production
```

### Research Configuration

```yaml
# research-config.yaml
device:
  id: "research-device-{node_id}"
  name: "Research Node {node_id}"
  
storage:
  retention:
    sensor_data_days: 365               # Long-term data retention for research
    
monitoring:
  interval_seconds: 30                  # High-frequency monitoring
  
integrations:
  webhooks:
    enabled: true                       # Enable data forwarding
    endpoints:
      - name: "research_data"
        url: "https://research.institution.edu/api/data"
        events: ["sensor_data"]
```

## Configuration Management

### Loading Configuration

```python
from furcate_nano import NanoConfig, load_config

# Load from specific file
config = load_config("config.yaml")

# Load with environment override
config = load_config("base-config.yaml")
if os.path.exists("local-config.yaml"):
    local_config = load_config("local-config.yaml")
    config.update(local_config)

# Create default configuration
config = NanoConfig()
```

### Creating Custom Configurations

```python
from furcate_nano import NanoConfig

# Create configuration programmatically
config = NanoConfig()
config.device.id = "custom-device-001"
config.device.name = "Custom Environmental Monitor"

# Configure sensors
config.hardware.sensors = {
    "temperature_humidity": {
        "type": "DHT22",
        "pin": 4,
        "enabled": True
    }
}

# Save configuration
config.save_config(config, "custom-config.yaml")
```

### Configuration Validation

```python
from furcate_nano import NanoConfig, validate_config

# Validate configuration
config = load_config("config.yaml")
is_valid, errors = validate_config(config)

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

### Environment Variables

Configuration values can be overridden using environment variables:

```bash
# Override device ID
export FURCATE_DEVICE_ID="env-device-001"

# Override simulation mode
export FURCATE_HARDWARE_SIMULATION="false"

# Override API port
export FURCATE_API_PORT="8080"

# Start device with environment overrides
furcate-nano start --config config.yaml
```

### Configuration Profiles

```bash
# Use different configuration profiles
furcate-nano start --profile development
furcate-nano start --profile production
furcate-nano start --profile classroom
furcate-nano start --profile research
```

Configuration profiles are stored in:
- `configs/development.yaml`
- `configs/production.yaml`
- `configs/classroom.yaml`
- `configs/research.yaml`

This comprehensive configuration guide covers all aspects of setting up and customizing Furcate Nano devices for various deployment scenarios.