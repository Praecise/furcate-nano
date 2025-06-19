# ============================================================================
# furcate_nano/config.py
"""Configuration management for Furcate Nano."""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)

class DeviceConfig(BaseModel):
    """Device configuration."""
    id: str = Field(default_factory=lambda: f"nano-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    name: str = "Furcate Nano Device"
    location: Dict[str, float] = Field(default_factory=lambda: {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0})
    environmental_zone: str = "default"

class HardwareConfig(BaseModel):
    """Hardware configuration for embedded devices."""
    platform: str = "auto_detect"  # auto_detect, arm64, x86_64, esp32
    simulation: bool = False
    gpio_pins: Dict[str, int] = Field(default_factory=lambda: {
        "dht22_data": 4,
        "status_led": 25,
        "lora_reset": 22,
        "lora_dio0": 18,
        "moisture_power": 24
    })
    sensors: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "temperature_humidity": {
            "type": "dht22",
            "pin": 4,
            "enabled": True
        },
        "air_quality": {
            "type": "mq135",
            "adc_channel": 0,
            "enabled": True
        },
        "soil_moisture": {
            "type": "moisture",
            "adc_channel": 1,
            "power_pin": 24,
            "enabled": True
        }
    })

class MLConfig(BaseModel):
    """Machine learning configuration."""
    simulation: bool = True
    model_path: str = "./models"
    models: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "environmental_classifier": {
            "file": "environmental_classifier.tflite",
            "enabled": True
        },
        "anomaly_detector": {
            "file": "anomaly_detector.tflite", 
            "enabled": True
        }
    })

class MeshConfig(BaseModel):
    """Mesh networking configuration."""
    simulation: bool = True
    max_connections: int = 8
    discovery_interval: int = 60
    environmental_zone: str = "default"
    protocols: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "lora": {
            "enabled": True,
            "frequency": 915,
            "spreading_factor": 7,
            "bandwidth": 125
        },
        "bluetooth": {
            "enabled": True,
            "max_range_m": 100
        }
    })

class PowerConfig(BaseModel):
    """Power management configuration."""
    simulation: bool = True
    battery: Dict[str, Any] = Field(default_factory=lambda: {
        "capacity_mah": 10000,
        "voltage_min": 3.0,
        "voltage_max": 4.2
    })
    solar: Dict[str, Any] = Field(default_factory=lambda: {
        "panel_watts": 20,
        "voltage_max": 6.0
    })
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "low_battery": 0.2,
        "critical_battery": 0.1,
        "emergency_battery": 0.05
    })

class StorageConfig(BaseModel):
    """Storage configuration."""
    db_path: str = "/data/furcate_nano.db"
    data_path: str = "/data/sensor_data"
    retention_days: int = 30
    max_size_mb: int = 1000

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    interval_seconds: int = 60
    default_interval_seconds: int = 60
    alert_thresholds: Dict[str, Dict[str, list]] = Field(default_factory=lambda: {
        "temperature_humidity": {
            "temperature": [-10, 50],
            "humidity": [0, 100]
        },
        "air_quality": {
            "aqi": [0, 200]
        }
    })

class ProtocolConfig(BaseModel):
    """Furcate protocol configuration."""
    version: str = "1.0"
    master_nodes: list = Field(default_factory=list)
    asset_creation_enabled: bool = False
    compression_enabled: bool = True

class NanoConfig(BaseModel):
    """Complete Furcate Nano configuration."""
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    
    # Global settings
    debug: bool = False
    log_level: str = "INFO"

def load_config(config_path: str) -> NanoConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return create_default_config()
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return NanoConfig(**config_data)
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return create_default_config()

def create_default_config(device_id: str = None) -> NanoConfig:
    """Create default configuration.
    
    Args:
        device_id: Optional device identifier
        
    Returns:
        Default configuration
    """
    config = NanoConfig()
    
    if device_id:
        config.device.id = device_id
    
    return config

def save_config(config: NanoConfig, config_path: str) -> bool:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
        
    Returns:
        Success status
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False