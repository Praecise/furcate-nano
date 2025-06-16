# ============================================================================
# furcate_nano/hardware.py
"""Hardware management for Raspberry Pi 5 environmental sensors."""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

try:
    import RPi.GPIO as GPIO
    import board
    import busio
    import adafruit_dht
    import adafruit_bmp280
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    import serial
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("Hardware libraries not available - running in simulation mode")

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Supported environmental sensor types."""
    TEMPERATURE_HUMIDITY = "dht22"
    PRESSURE_TEMPERATURE = "bmp280"
    AIR_QUALITY = "mq135"
    SOIL_MOISTURE = "moisture"
    LIGHT_INTENSITY = "tsl2561"
    SOUND_LEVEL = "microphone"
    PARTICULATE_MATTER = "pms5003"
    UV_INDEX = "veml6070"
    GPS = "neo_8m"

@dataclass
class SensorReading:
    """Standardized sensor reading format."""
    sensor_type: SensorType
    timestamp: float
    value: Union[float, Dict[str, float]]
    unit: str
    quality: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enum to string for JSON serialization
        result['sensor_type'] = self.sensor_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create SensorReading from dictionary."""
        data = data.copy()
        # Convert string back to enum
        if isinstance(data['sensor_type'], str):
            data['sensor_type'] = SensorType(data['sensor_type'])
        return cls(**data)

class HardwareManager:
    """Manages all hardware interfaces for Furcate Nano devices."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hardware manager.
        
        Args:
            config: Hardware configuration
        """
        self.config = config
        self.device_id = config.get("device_id", "nano-unknown")
        self.simulation_mode = not HARDWARE_AVAILABLE or config.get("simulation", False)
        
        # Hardware interfaces
        self.i2c = None
        self.adc = None
        self.uart = None
        
        # Active sensors
        self.sensors: Dict[str, Any] = {}
        self.sensor_configs: Dict[str, Dict] = config.get("sensors", {})
        
        # GPIO pin assignments
        self.pins = config.get("gpio_pins", {
            "dht22_data": 4,
            "status_led": 25,
            "lora_reset": 22,
            "lora_dio0": 18,
            "moisture_power": 24
        })
        
        # Sensor calibration data
        self.calibrations = {}
        self._load_calibrations()
        
        # Statistics
        self.stats = {
            "readings_taken": 0,
            "reading_errors": 0,
            "sensor_count": 0,
            "last_reading_time": None
        }
        
        logger.info(f"Hardware manager initialized (simulation: {self.simulation_mode})")
    
    def _load_calibrations(self):
        """Load sensor calibration data."""
        # Load calibration data from configuration or file
        for sensor_name, sensor_config in self.sensor_configs.items():
            calibration = sensor_config.get("calibration", {})
            if calibration:
                self.calibrations[sensor_name] = calibration
    
    async def initialize(self) -> bool:
        """Initialize hardware interfaces and sensors."""
        try:
            if self.simulation_mode:
                logger.info("🔧 Running in simulation mode")
                return await self._initialize_simulation()
            else:
                logger.info("🔧 Initializing Raspberry Pi 5 hardware")
                return await self._initialize_hardware()
                
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return False
    
    async def _initialize_hardware(self) -> bool:
        """Initialize real hardware."""
        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Initialize I2C
            self.i2c = busio.I2C(board.SCL, board.SDA)
            logger.info("✅ I2C bus initialized")
            
            # Initialize ADC for analog sensors
            self.adc = ADS.ADS1115(self.i2c)
            logger.info("✅ ADC initialized")
            
            # Initialize UART
            try:
                self.uart = serial.Serial('/dev/ttyS0', 9600, timeout=1)
                logger.info("✅ UART initialized")
            except Exception as e:
                logger.warning(f"UART initialization failed: {e}")
            
            # Initialize individual sensors
            success_count = 0
            for sensor_name, sensor_config in self.sensor_configs.items():
                if sensor_config.get("enabled", False):
                    success = await self._initialize_sensor(sensor_name, sensor_config)
                    if success:
                        success_count += 1
                        logger.info(f"✅ Initialized sensor: {sensor_name}")
                    else:
                        logger.warning(f"⚠️ Failed to initialize sensor: {sensor_name}")
            
            self.stats["sensor_count"] = success_count
            
            # Setup status LED
            if "status_led" in self.pins:
                GPIO.setup(self.pins["status_led"], GPIO.OUT)
                GPIO.output(self.pins["status_led"], GPIO.HIGH)
                logger.info("✅ Status LED initialized")
            
            logger.info(f"✅ Hardware initialization complete ({success_count} sensors)")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Real hardware initialization failed: {e}")
            return False
    
    async def _initialize_simulation(self) -> bool:
        """Initialize simulation mode."""
        # Create simulated sensors
        success_count = 0
        for sensor_name, sensor_config in self.sensor_configs.items():
            if sensor_config.get("enabled", False):
                sensor_type = SensorType(sensor_config["type"])
                self.sensors[sensor_name] = {
                    "type": sensor_type,
                    "config": sensor_config,
                    "simulation": True,
                    "last_reading": None,
                    "reading_count": 0
                }
                success_count += 1
                logger.info(f"🔧 Simulated sensor: {sensor_name} ({sensor_type.value})")
        
        self.stats["sensor_count"] = success_count
        return True
    
    async def _initialize_sensor(self, name: str, config: Dict) -> bool:
        """Initialize a single sensor."""
        try:
            sensor_type = SensorType(config["type"])
            
            if sensor_type == SensorType.TEMPERATURE_HUMIDITY:
                pin = config.get("pin", self.pins["dht22_data"])
                self.sensors[name] = {
                    "sensor": adafruit_dht.DHT22(getattr(board, f"D{pin}")),
                    "type": sensor_type,
                    "config": config
                }
                
            elif sensor_type == SensorType.PRESSURE_TEMPERATURE:
                i2c_addr = config.get("i2c_address", 0x76)
                self.sensors[name] = {
                    "sensor": adafruit_bmp280.Adafruit_BMP280_I2C(self.i2c, address=i2c_addr),
                    "type": sensor_type,
                    "config": config
                }
                
            elif sensor_type == SensorType.AIR_QUALITY:
                channel = config.get("adc_channel", 0)
                self.sensors[name] = {
                    "sensor": AnalogIn(self.adc, getattr(ADS, f"P{channel}")),
                    "type": sensor_type,
                    "config": config
                }
                
            elif sensor_type == SensorType.SOIL_MOISTURE:
                channel = config.get("adc_channel", 1)
                power_pin = config.get("power_pin", self.pins["moisture_power"])
                GPIO.setup(power_pin, GPIO.OUT)
                GPIO.output(power_pin, GPIO.LOW)  # Start with power off
                self.sensors[name] = {
                    "adc": AnalogIn(self.adc, getattr(ADS, f"P{channel}")),
                    "power_pin": power_pin,
                    "type": sensor_type,
                    "config": config
                }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            return False
    
    async def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read all configured sensors."""
        readings = {}
        current_time = time.time()
        
        for sensor_name, sensor_obj in self.sensors.items():
            try:
                reading = await self._read_sensor(sensor_name, sensor_obj, current_time)
                if reading:
                    readings[sensor_name] = reading
                    self.stats["readings_taken"] += 1
                    if sensor_obj.get("simulation", False):
                        sensor_obj["reading_count"] += 1
                        sensor_obj["last_reading"] = reading
                    
            except Exception as e:
                logger.warning(f"Failed to read {sensor_name}: {e}")
                self.stats["reading_errors"] += 1
        
        self.stats["last_reading_time"] = current_time
        return readings
    
    async def read_sensor(self, sensor_name: str) -> Optional[SensorReading]:
        """Read a specific sensor."""
        if sensor_name not in self.sensors:
            logger.error(f"Sensor not found: {sensor_name}")
            return None
        
        try:
            sensor_obj = self.sensors[sensor_name]
            reading = await self._read_sensor(sensor_name, sensor_obj, time.time())
            if reading:
                self.stats["readings_taken"] += 1
            return reading
            
        except Exception as e:
            logger.error(f"Failed to read sensor {sensor_name}: {e}")
            self.stats["reading_errors"] += 1
            return None
    
    async def _read_sensor(self, name: str, sensor_obj: Any, timestamp: float) -> Optional[SensorReading]:
        """Read a single sensor."""
        sensor_config = self.sensor_configs.get(name, {})
        sensor_type = sensor_obj.get("type") or SensorType(sensor_config["type"])
        
        if self.simulation_mode or sensor_config.get("simulation", False):
            return self._simulate_sensor_reading(name, sensor_type, timestamp)
        
        try:
            if sensor_type == SensorType.TEMPERATURE_HUMIDITY:
                sensor = sensor_obj["sensor"]
                temp = sensor.temperature
                humidity = sensor.humidity
                
                if temp is not None and humidity is not None:
                    # Apply calibration
                    temp_offset = self.calibrations.get(name, {}).get("temperature_offset", 0.0)
                    humidity_offset = self.calibrations.get(name, {}).get("humidity_offset", 0.0)
                    
                    temp += temp_offset
                    humidity += humidity_offset
                    humidity = max(0, min(100, humidity))  # Clamp humidity
                    
                    return SensorReading(
                        sensor_type=sensor_type,
                        timestamp=timestamp,
                        value={"temperature": round(temp, 1), "humidity": round(humidity, 1)},
                        unit="°C, %RH",
                        quality=0.95,
                        confidence=0.90,
                        metadata={"calibrated": True}
                    )
                    
            elif sensor_type == SensorType.PRESSURE_TEMPERATURE:
                sensor = sensor_obj["sensor"]
                pressure = sensor.pressure
                temp = sensor.temperature
                
                # Apply sea level correction
                sea_level_pressure = sensor_config.get("sea_level_pressure", 1013.25)
                altitude_correction = sensor_config.get("altitude_correction", 0.0)
                
                corrected_pressure = pressure + altitude_correction
                
                return SensorReading(
                    sensor_type=sensor_type,
                    timestamp=timestamp,
                    value={"pressure": round(corrected_pressure, 1), "temperature": round(temp, 1)},
                    unit="hPa, °C",
                    quality=0.98,
                    confidence=0.95,
                    metadata={"sea_level_corrected": True}
                )
                
            elif sensor_type == SensorType.AIR_QUALITY:
                sensor = sensor_obj["sensor"]
                voltage = sensor.voltage
                
                # Convert voltage to air quality index using calibration
                calibration_factor = self.calibrations.get(name, {}).get("calibration_factor", 1.0)
                baseline_voltage = self.calibrations.get(name, {}).get("baseline_voltage", 0.1)
                
                # Simplified AQI calculation (would be more complex in reality)
                raw_aqi = max(0, (voltage - baseline_voltage) * 1000 * calibration_factor)
                aqi = min(500, raw_aqi)  # Cap at hazardous level
                
                return SensorReading(
                    sensor_type=sensor_type,
                    timestamp=timestamp,
                    value={"aqi": round(aqi, 0), "voltage": round(voltage, 3)},
                    unit="AQI",
                    quality=0.85,
                    confidence=0.80,
                    metadata={"calibration_factor": calibration_factor}
                )
                
            elif sensor_type == SensorType.SOIL_MOISTURE:
                # Power on sensor
                GPIO.output(sensor_obj["power_pin"], GPIO.HIGH)
                await asyncio.sleep(0.1)  # Wait for sensor to stabilize
                
                voltage = sensor_obj["adc"].voltage
                
                # Convert to moisture percentage using calibration
                dry_value = sensor_config.get("dry_value", 3.3)
                wet_value = sensor_config.get("wet_value", 1.2)
                
                # Calculate moisture percentage
                if dry_value != wet_value:
                    moisture = ((dry_value - voltage) / (dry_value - wet_value)) * 100
                else:
                    moisture = 50.0  # Default if no calibration
                
                moisture = max(0, min(100, moisture))  # Clamp to valid range
                
                # Power off sensor to save energy
                GPIO.output(sensor_obj["power_pin"], GPIO.LOW)
                
                return SensorReading(
                    sensor_type=sensor_type,
                    timestamp=timestamp,
                    value={"moisture": round(moisture, 1), "voltage": round(voltage, 3)},
                    unit="%",
                    quality=0.90,
                    confidence=0.85,
                    metadata={"power_cycled": True}
                )
        
        except Exception as e:
            logger.warning(f"Sensor read error for {name}: {e}")
            return None
    
    def _simulate_sensor_reading(self, name: str, sensor_type: SensorType, timestamp: float) -> SensorReading:
        """Generate simulated sensor reading."""
        import random
        
        # Get previous reading for trend continuity
        sensor_obj = self.sensors.get(name, {})
        last_reading = sensor_obj.get("last_reading")
        
        # Generate realistic trending data
        if sensor_type == SensorType.TEMPERATURE_HUMIDITY:
            # Base values with diurnal variation
            hour = datetime.fromtimestamp(timestamp).hour
            temp_base = 20 + 5 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2 PM
            humidity_base = 60 - 20 * math.sin((hour - 6) * math.pi / 12)  # Inverse of temp
            
            # Add some random variation
            temp = temp_base + random.gauss(0, 3)
            humidity = humidity_base + random.gauss(0, 10)
            humidity = max(0, min(100, humidity))
            
            return SensorReading(
                sensor_type=sensor_type,
                timestamp=timestamp,
                value={"temperature": round(temp, 1), "humidity": round(humidity, 1)},
                unit="°C, %RH",
                quality=0.95,
                confidence=0.90,
                metadata={"simulated": True, "trend": "diurnal"}
            )
            
        elif sensor_type == SensorType.AIR_QUALITY:
            # Simulate pollution patterns
            base_aqi = 50
            if 7 <= datetime.fromtimestamp(timestamp).hour <= 9:  # Morning rush
                base_aqi = 80
            elif 17 <= datetime.fromtimestamp(timestamp).hour <= 19:  # Evening rush
                base_aqi = 90
            
            aqi = max(0, base_aqi + random.gauss(0, 15))
            
            return SensorReading(
                sensor_type=sensor_type,
                timestamp=timestamp,
                value={"aqi": round(aqi, 0)},
                unit="AQI",
                quality=0.85,
                confidence=0.80,
                metadata={"simulated": True, "pattern": "traffic"}
            )
            
        elif sensor_type == SensorType.SOIL_MOISTURE:
            # Simulate soil moisture with weather patterns
            base_moisture = 45
            
            # Simulate drying over time
            if last_reading:
                last_moisture = last_reading.value.get("moisture", 45)
                # Gradual drying with some random variation
                moisture = last_moisture - random.uniform(0, 2) + random.gauss(0, 5)
            else:
                moisture = base_moisture + random.gauss(0, 15)
            
            moisture = max(0, min(100, moisture))
            
            return SensorReading(
                sensor_type=sensor_type,
                timestamp=timestamp,
                value={"moisture": round(moisture, 1)},
                unit="%",
                quality=0.90,
                confidence=0.85,
                metadata={"simulated": True, "pattern": "drying"}
            )
        
        # Default for unknown sensor types
        return SensorReading(
            sensor_type=sensor_type,
            timestamp=timestamp,
            value={"value": random.random()},
            unit="simulated",
            quality=0.95,
            confidence=0.90,
            metadata={"simulated": True}
        )
    
    async def calibrate_sensor(self, sensor_name: str, calibration_data: Dict[str, float]) -> bool:
        """Calibrate a sensor with provided calibration data."""
        try:
            if sensor_name not in self.sensors:
                logger.error(f"Cannot calibrate unknown sensor: {sensor_name}")
                return False
            
            self.calibrations[sensor_name] = calibration_data
            logger.info(f"✅ Calibrated sensor {sensor_name}: {calibration_data}")
            return True
            
        except Exception as e:
            logger.error(f"Sensor calibration failed for {sensor_name}: {e}")
            return False
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get information about all sensors."""
        sensor_info = {}
        
        for sensor_name, sensor_obj in self.sensors.items():
            config = self.sensor_configs.get(sensor_name, {})
            sensor_info[sensor_name] = {
                "type": sensor_obj["type"].value,
                "enabled": config.get("enabled", False),
                "simulation": sensor_obj.get("simulation", False),
                "calibrated": sensor_name in self.calibrations,
                "reading_count": sensor_obj.get("reading_count", 0),
                "last_reading_time": sensor_obj.get("last_reading", {}).get("timestamp") if sensor_obj.get("last_reading") else None
            }
        
        return sensor_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hardware manager statistics."""
        return {
            **self.stats,
            "sensor_info": self.get_sensor_info(),
            "calibrations": len(self.calibrations),
            "simulation_mode": self.simulation_mode
        }
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive hardware diagnostics."""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "sensor_tests": {},
            "interface_tests": {},
            "summary": {}
        }
        
        try:
            # Test all sensors
            sensor_test_results = {}
            for sensor_name in self.sensors.keys():
                reading = await self.read_sensor(sensor_name)
                sensor_test_results[sensor_name] = {
                    "status": "pass" if reading else "fail",
                    "reading": reading.to_dict() if reading else None
                }
            
            diagnostics["sensor_tests"] = sensor_test_results
            
            # Test interfaces (if not in simulation)
            if not self.simulation_mode:
                interface_tests = {
                    "i2c": "pass" if self.i2c else "fail",
                    "adc": "pass" if self.adc else "fail",
                    "uart": "pass" if self.uart else "fail",
                    "gpio": "pass"  # Assume GPIO is working if we got this far
                }
                diagnostics["interface_tests"] = interface_tests
            
            # Calculate summary
            sensor_pass_count = sum(1 for test in sensor_test_results.values() if test["status"] == "pass")
            total_sensors = len(sensor_test_results)
            
            diagnostics["summary"] = {
                "sensors_passed": sensor_pass_count,
                "sensors_total": total_sensors,
                "sensor_success_rate": sensor_pass_count / total_sensors if total_sensors > 0 else 0,
                "simulation_mode": self.simulation_mode
            }
            
            # Overall status
            if sensor_pass_count == total_sensors:
                diagnostics["overall_status"] = "excellent"
            elif sensor_pass_count >= total_sensors * 0.8:
                diagnostics["overall_status"] = "good"
            elif sensor_pass_count >= total_sensors * 0.5:
                diagnostics["overall_status"] = "degraded"
            else:
                diagnostics["overall_status"] = "critical"
            
            logger.info(f"🔧 Hardware diagnostics complete: {diagnostics['overall_status']}")
            return diagnostics
            
        except Exception as e:
            logger.error(f"Hardware diagnostics failed: {e}")
            diagnostics["overall_status"] = "error"
            diagnostics["error"] = str(e)
            return diagnostics
    
    async def shutdown(self):
        """Shutdown hardware interfaces."""
        logger.info("🔧 Shutting down hardware manager...")
        
        try:
            # Turn off all sensor power pins
            for sensor_name, sensor_obj in self.sensors.items():
                if "power_pin" in sensor_obj and not self.simulation_mode:
                    GPIO.output(sensor_obj["power_pin"], GPIO.LOW)
            
            # Clean up GPIO
            if not self.simulation_mode and HARDWARE_AVAILABLE:
                GPIO.cleanup()
                logger.info("✅ GPIO cleanup complete")
            
            # Close serial connections
            if self.uart:
                self.uart.close()
            
            # Clear sensor references
            self.sensors.clear()
            
        except Exception as e:
            logger.error(f"Hardware shutdown error: {e}")
        
        logger.info("✅ Hardware manager shutdown complete")

# Import math for simulation calculations
import math