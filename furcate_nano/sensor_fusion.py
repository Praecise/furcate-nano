# ============================================================================
# furcate_nano/sensor_fusion.py
"""Sensor fusion and data validation for Furcate Nano."""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .hardware import SensorReading, SensorType

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"  # >95% confidence
    GOOD = "good"           # 80-95% confidence  
    FAIR = "fair"           # 60-80% confidence
    POOR = "poor"           # <60% confidence
    INVALID = "invalid"     # Failed validation

@dataclass
class FusedReading:
    """Fused sensor reading with validation."""
    timestamp: datetime
    environmental_conditions: Dict[str, float]
    data_quality: DataQuality
    confidence_score: float
    contributing_sensors: List[str]
    anomalies_detected: List[str]
    metadata: Dict[str, Any]

class NanoSensorFusion:
    """Sensor fusion engine for environmental data validation and enhancement."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize sensor fusion engine."""
        self.config = config
        self.validation_rules = config.get("validation", {})
        self.fusion_window_seconds = config.get("fusion_window", 60)
        self.min_sensors_required = config.get("min_sensors", 2)
        
        # Sensor reading history for temporal validation
        self.reading_history: Dict[str, List[SensorReading]] = {}
        self.max_history_size = 100
        
        # Cross-sensor validation rules
        self.validation_matrix = {
            # Temperature correlation rules
            (SensorType.TEMPERATURE_HUMIDITY, SensorType.PRESSURE_TEMPERATURE): {
                "correlation_threshold": 0.8,
                "max_difference": 3.0  # °C
            }
        }
        
        logger.info("Sensor fusion engine initialized")
    
    async def fuse_sensor_readings(self, readings: Dict[str, SensorReading]) -> FusedReading:
        """Fuse multiple sensor readings into validated environmental data."""
        try:
            timestamp = datetime.now()
            
            # Update reading history
            self._update_reading_history(readings)
            
            # Validate individual sensors
            validated_readings = await self._validate_individual_sensors(readings)
            
            # Cross-validate sensors
            cross_validation_results = await self._cross_validate_sensors(validated_readings)
            
            # Temporal validation
            temporal_validation = await self._temporal_validation(validated_readings)
            
            # Fuse into environmental conditions
            environmental_conditions = await self._fuse_environmental_data(validated_readings)
            
            # Calculate overall confidence
            confidence_score = self._calculate_fusion_confidence(
                validated_readings, cross_validation_results, temporal_validation
            )
            
            # Determine data quality
            data_quality = self._determine_data_quality(confidence_score, validated_readings)
            
            # Detect anomalies
            anomalies = await self._detect_fusion_anomalies(
                environmental_conditions, validated_readings
            )
            
            return FusedReading(
                timestamp=timestamp,
                environmental_conditions=environmental_conditions,
                data_quality=data_quality,
                confidence_score=confidence_score,
                contributing_sensors=list(validated_readings.keys()),
                anomalies_detected=anomalies,
                metadata={
                    "fusion_method": "weighted_average",
                    "sensor_count": len(validated_readings),
                    "cross_validation": cross_validation_results,
                    "temporal_validation": temporal_validation
                }
            )
            
        except Exception as e:
            logger.error(f"Sensor fusion failed: {e}")
            return self._create_error_reading(readings)
    
    def _update_reading_history(self, readings: Dict[str, SensorReading]):
        """Update sensor reading history for temporal analysis."""
        for sensor_name, reading in readings.items():
            if sensor_name not in self.reading_history:
                self.reading_history[sensor_name] = []
            
            self.reading_history[sensor_name].append(reading)
            
            # Limit history size
            if len(self.reading_history[sensor_name]) > self.max_history_size:
                self.reading_history[sensor_name] = self.reading_history[sensor_name][-self.max_history_size:]
    
    async def _validate_individual_sensors(self, readings: Dict[str, SensorReading]) -> Dict[str, SensorReading]:
        """Validate individual sensor readings."""
        validated = {}
        
        for sensor_name, reading in readings.items():
            try:
                # Check basic validity
                if reading.quality < 0.3 or reading.confidence < 0.3:
                    logger.warning(f"Low quality reading from {sensor_name}")
                    continue
                
                # Range validation
                if await self._validate_sensor_range(sensor_name, reading):
                    validated[sensor_name] = reading
                else:
                    logger.warning(f"Range validation failed for {sensor_name}")
                    
            except Exception as e:
                logger.error(f"Validation failed for {sensor_name}: {e}")
        
        return validated
    
    async def _validate_sensor_range(self, sensor_name: str, reading: SensorReading) -> bool:
        """Validate sensor reading is within expected ranges."""
        ranges = {
            SensorType.TEMPERATURE_HUMIDITY: {
                "temperature": (-40, 80),  # °C
                "humidity": (0, 100)       # %RH
            },
            SensorType.PRESSURE_TEMPERATURE: {
                "pressure": (800, 1200),   # hPa
                "temperature": (-40, 80)   # °C
            },
            SensorType.AIR_QUALITY: {
                "aqi": (0, 500)           # AQI
            },
            SensorType.SOIL_MOISTURE: {
                "moisture": (0, 100)      # %
            }
        }
        
        sensor_ranges = ranges.get(reading.sensor_type, {})
        
        if isinstance(reading.value, dict):
            for param, value in reading.value.items():
                if param in sensor_ranges:
                    min_val, max_val = sensor_ranges[param]
                    if not (min_val <= value <= max_val):
                        return False
        elif isinstance(reading.value, (int, float)):
            # Single value sensor - use first range
            if sensor_ranges:
                min_val, max_val = list(sensor_ranges.values())[0]
                if not (min_val <= reading.value <= max_val):
                    return False
        
        return True
    
    async def _cross_validate_sensors(self, readings: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Cross-validate sensors against each other."""
        validation_results = {}
        
        # Temperature cross-validation
        temp_sensors = []
        for sensor_name, reading in readings.items():
            if reading.sensor_type in [SensorType.TEMPERATURE_HUMIDITY, SensorType.PRESSURE_TEMPERATURE]:
                if isinstance(reading.value, dict) and "temperature" in reading.value:
                    temp_sensors.append((sensor_name, reading.value["temperature"]))
        
        if len(temp_sensors) >= 2:
            temp_values = [temp for _, temp in temp_sensors]
            temp_std = np.std(temp_values)
            temp_mean = np.mean(temp_values)
            
            validation_results["temperature_cross_validation"] = {
                "sensor_count": len(temp_sensors),
                "std_deviation": temp_std,
                "mean_value": temp_mean,
                "consistent": temp_std < 2.0  # Within 2°C
            }
        
        return validation_results
    
    async def _temporal_validation(self, readings: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Validate readings against historical trends."""
        temporal_results = {}
        
        for sensor_name, reading in readings.items():
            if sensor_name in self.reading_history:
                history = self.reading_history[sensor_name]
                
                if len(history) >= 5:  # Need at least 5 readings for trend analysis
                    recent_values = []
                    
                    # Extract comparable values
                    for hist_reading in history[-5:]:
                        if isinstance(reading.value, dict) and isinstance(hist_reading.value, dict):
                            # Multi-parameter sensor
                            for param in reading.value.keys():
                                if param in hist_reading.value:
                                    recent_values.append(hist_reading.value[param])
                                    break
                        elif isinstance(reading.value, (int, float)):
                            recent_values.append(hist_reading.value)
                    
                    if recent_values:
                        mean_val = np.mean(recent_values)
                        std_val = np.std(recent_values)
                        
                        # Check if current reading is within expected range
                        current_val = reading.value if isinstance(reading.value, (int, float)) else list(reading.value.values())[0]
                        z_score = abs(current_val - mean_val) / (std_val + 1e-6)
                        
                        temporal_results[sensor_name] = {
                            "z_score": z_score,
                            "within_normal_range": z_score < 3.0,
                            "trend_consistent": z_score < 2.0
                        }
        
        return temporal_results
    
    async def _fuse_environmental_data(self, readings: Dict[str, SensorReading]) -> Dict[str, float]:
        """Fuse sensor data into environmental conditions."""
        conditions = {}
        
        # Temperature fusion (weighted average if multiple sensors)
        temp_readings = []
        for reading in readings.values():
            if isinstance(reading.value, dict) and "temperature" in reading.value:
                temp_readings.append((reading.value["temperature"], reading.confidence))
        
        if temp_readings:
            weighted_temp = sum(temp * conf for temp, conf in temp_readings) / sum(conf for _, conf in temp_readings)
            conditions["temperature"] = round(weighted_temp, 2)
        
        # Humidity (usually single sensor)
        for reading in readings.values():
            if isinstance(reading.value, dict) and "humidity" in reading.value:
                conditions["humidity"] = round(reading.value["humidity"], 1)
                break
        
        # Pressure (usually single sensor)
        for reading in readings.values():
            if isinstance(reading.value, dict) and "pressure" in reading.value:
                conditions["pressure"] = round(reading.value["pressure"], 1)
                break
        
        # Air quality
        for reading in readings.values():
            if reading.sensor_type == SensorType.AIR_QUALITY:
                if isinstance(reading.value, dict) and "aqi" in reading.value:
                    conditions["air_quality_index"] = round(reading.value["aqi"], 0)
                break
        
        # Soil moisture
        for reading in readings.values():
            if reading.sensor_type == SensorType.SOIL_MOISTURE:
                if isinstance(reading.value, dict) and "moisture" in reading.value:
                    conditions["soil_moisture"] = round(reading.value["moisture"], 1)
                break
        
        return conditions
    
    def _calculate_fusion_confidence(self, readings: Dict[str, SensorReading], 
                                   cross_validation: Dict, temporal_validation: Dict) -> float:
        """Calculate overall confidence in fused data."""
        if not readings:
            return 0.0
        
        # Base confidence from sensor readings
        sensor_confidences = [reading.confidence for reading in readings.values()]
        base_confidence = np.mean(sensor_confidences)
        
        # Cross-validation bonus
        cross_validation_bonus = 0.0
        if "temperature_cross_validation" in cross_validation:
            if cross_validation["temperature_cross_validation"]["consistent"]:
                cross_validation_bonus = 0.1
        
        # Temporal validation bonus
        temporal_bonus = 0.0
        consistent_sensors = sum(1 for result in temporal_validation.values() 
                               if result.get("trend_consistent", False))
        if temporal_validation:
            temporal_bonus = (consistent_sensors / len(temporal_validation)) * 0.05
        
        # Sensor count bonus (more sensors = higher confidence)
        sensor_count_bonus = min(0.1, len(readings) * 0.02)
        
        final_confidence = min(1.0, base_confidence + cross_validation_bonus + temporal_bonus + sensor_count_bonus)
        return round(final_confidence, 3)
    
    def _determine_data_quality(self, confidence: float, readings: Dict[str, SensorReading]) -> DataQuality:
        """Determine overall data quality classification."""
        if confidence >= 0.95:
            return DataQuality.EXCELLENT
        elif confidence >= 0.80:
            return DataQuality.GOOD
        elif confidence >= 0.60:
            return DataQuality.FAIR
        elif confidence >= 0.30:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    async def _detect_fusion_anomalies(self, conditions: Dict[str, float], 
                                     readings: Dict[str, SensorReading]) -> List[str]:
        """Detect anomalies in fused environmental data."""
        anomalies = []
        
        # Environmental anomaly patterns
        if "temperature" in conditions and "humidity" in conditions:
            temp = conditions["temperature"]
            humidity = conditions["humidity"]
            
            # Impossible combinations
            if temp < 0 and humidity > 90:  # Very cold but very humid
                anomalies.append("impossible_temp_humidity_combination")
            
            if temp > 40 and humidity > 90:  # Very hot and very humid (possible but rare)
                anomalies.append("extreme_heat_humidity")
        
        # Air quality anomalies
        if "air_quality_index" in conditions:
            aqi = conditions["air_quality_index"]
            if aqi > 300:
                anomalies.append("hazardous_air_quality")
            elif aqi < 0:
                anomalies.append("invalid_air_quality")
        
        # Pressure anomalies
        if "pressure" in conditions:
            pressure = conditions["pressure"]
            if pressure < 950 or pressure > 1050:
                anomalies.append("extreme_pressure")
        
        return anomalies
    
    def _create_error_reading(self, original_readings: Dict[str, SensorReading]) -> FusedReading:
        """Create error reading when fusion fails."""
        return FusedReading(
            timestamp=datetime.now(),
            environmental_conditions={},
            data_quality=DataQuality.INVALID,
            confidence_score=0.0,
            contributing_sensors=list(original_readings.keys()),
            anomalies_detected=["fusion_error"],
            metadata={"error": "Sensor fusion failed"}
        )