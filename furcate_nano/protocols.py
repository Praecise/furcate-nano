# ============================================================================
# furcate_nano/protocols.py
"""Furcate communication protocols for nano devices."""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class MessageFormat(Enum):
    """Message format types."""
    SENSOR_DATA = "sensor_data"
    ENVIRONMENTAL_ALERT = "environmental_alert"
    MESH_HEARTBEAT = "mesh_heartbeat"
    DISCOVERY_BEACON = "discovery_beacon"
    ASSET_NOTIFICATION = "asset_notification"

class FurcateProtocol:
    """Furcate communication protocol handler."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize protocol handler.
        
        Args:
            config: Protocol configuration
        """
        self.config = config
        self.version = config.get("version", "1.0")
        self.compression_enabled = config.get("compression_enabled", True)
        self.master_nodes = config.get("master_nodes", [])
        
        # Message processing statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "compression_ratio": 0.0
        }
        
        logger.info(f"Furcate protocol initialized v{self.version}")
    
    def create_sensor_data_message(self, device_id: str, sensor_data: Dict[str, Any], 
                                 ml_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized sensor data message.
        
        Args:
            device_id: Source device identifier
            sensor_data: Raw sensor data
            ml_analysis: ML analysis results
            
        Returns:
            Formatted message
        """
        message = {
            "format": MessageFormat.SENSOR_DATA.value,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "sequence": int(time.time()),
            "data": {
                "sensors": self._compress_sensor_data(sensor_data),
                "ml": self._compress_ml_data(ml_analysis),
                "quality_score": self._calculate_data_quality(sensor_data)
            }
        }
        
        return message
    
    def create_alert_message(self, device_id: str, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Create environmental alert message.
        
        Args:
            device_id: Source device identifier
            alert: Alert information
            
        Returns:
            Formatted alert message
        """
        message = {
            "format": MessageFormat.ENVIRONMENTAL_ALERT.value,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "priority": alert.get("severity", "warning"),
            "alert": {
                "type": alert.get("type", "unknown"),
                "severity": alert.get("severity", "warning"),
                "description": alert.get("message", "Environmental alert"),
                "parameters": alert.get("parameters", {}),
                "confidence": alert.get("confidence", 0.8)
            }
        }
        
        return message
    
    def create_discovery_beacon(self, device_id: str, capabilities: List[str], 
                               environmental_zone: str) -> Dict[str, Any]:
        """Create discovery beacon message.
        
        Args:
            device_id: Source device identifier
            capabilities: Device capabilities
            environmental_zone: Environmental zone identifier
            
        Returns:
            Discovery beacon message
        """
        message = {
            "format": MessageFormat.DISCOVERY_BEACON.value,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "discovery": {
                "device_type": "furcate_nano",
                "capabilities": capabilities,
                "environmental_zone": environmental_zone,
                "protocol_version": self.version,
                "mesh_enabled": True
            }
        }
        
        return message
    
    def _compress_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress sensor data for efficient transmission."""
        if not self.compression_enabled:
            return sensor_data
        
        compressed = {}
        
        for sensor_name, sensor_info in sensor_data.get("sensors", {}).items():
            if isinstance(sensor_info, dict) and "value" in sensor_info:
                # Keep only essential data for transmission
                compressed[sensor_name] = {
                    "v": sensor_info["value"],  # Shortened key
                    "q": round(sensor_info.get("quality", 1.0), 2),  # Quality
                    "t": sensor_info.get("timestamp", time.time())  # Timestamp
                }
        
        return compressed
    
    def _compress_ml_data(self, ml_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compress ML analysis data."""
        if not self.compression_enabled:
            return ml_analysis
        
        return {
            "class": ml_analysis.get("environmental_class", "normal"),
            "anomaly": round(ml_analysis.get("anomaly_score", 0.0), 3),
            "conf": round(ml_analysis.get("confidence", 0.0), 3)
        }
    
    def _calculate_data_quality(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        qualities = []
        
        for sensor_info in sensor_data.get("sensors", {}).values():
            if isinstance(sensor_info, dict):
                quality = sensor_info.get("quality", 1.0)
                qualities.append(quality)
        
        return sum(qualities) / len(qualities) if qualities else 0.0
    
    def serialize_message(self, message: Dict[str, Any]) -> bytes:
        """Serialize message for transmission.
        
        Args:
            message: Message to serialize
            
        Returns:
            Serialized message bytes
        """
        try:
            json_str = json.dumps(message, separators=(',', ':'))  # Compact JSON
            serialized = json_str.encode('utf-8')
            
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Message serialization failed: {e}")
            return b""
    
    def deserialize_message(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Deserialize received message.
        
        Args:
            data: Serialized message data
            
        Returns:
            Deserialized message or None if failed
        """
        try:
            json_str = data.decode('utf-8')
            message = json.loads(json_str)
            
            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(data)
            
            return message
            
        except Exception as e:
            logger.error(f"Message deserialization failed: {e}")
            return None
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message format and content.
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["format", "version", "timestamp", "device_id"]
        
        for field in required_fields:
            if field not in message:
                logger.warning(f"Message missing required field: {field}")
                return False
        
        # Check version compatibility
        if message["version"] != self.version:
            logger.warning(f"Protocol version mismatch: {message['version']} != {self.version}")
            # Could handle version compatibility here
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        stats = self.stats.copy()
        
        # Calculate compression ratio
        if stats["bytes_received"] > 0:
            stats["compression_ratio"] = 1.0 - (stats["bytes_sent"] / stats["bytes_received"])
        
        return stats