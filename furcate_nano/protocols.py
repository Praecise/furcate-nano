# ============================================================================
# furcate_nano/protocols.py
"""
Complete protocol implementation with comprehensive error handling,
validation, and network resilience.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import hashlib
import gzip
import base64

logger = logging.getLogger(__name__)

class MessageFormat(Enum):
    """Supported message formats."""
    ENVIRONMENTAL_DATA = "environmental_data"
    ML_ANALYSIS = "ml_analysis"
    DEVICE_STATUS = "device_status"
    ENVIRONMENTAL_ALERT = "environmental_alert"
    DISCOVERY_BEACON = "discovery_beacon"
    NETWORK_SYNC = "network_sync"
    COMMAND = "command"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class CompressionLevel(Enum):
    """Compression levels for message optimization."""
    NONE = 0
    LOW = 1
    MEDIUM = 5
    HIGH = 9

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class MessageMetadata:
    """Enhanced message metadata."""
    message_id: str
    timestamp: datetime
    sender_id: str
    recipient_id: Optional[str]
    message_type: MessageFormat
    priority: MessagePriority
    ttl_seconds: int
    retry_count: int = 0
    max_retries: int = 3
    compression_level: CompressionLevel = CompressionLevel.NONE
    encrypted: bool = False
    checksum: Optional[str] = None
    sequence_number: int = 0
    correlation_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries

@dataclass
class ProtocolStats:
    """Protocol statistics for monitoring."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    compression_ratio: float = 1.0
    average_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    last_activity: Optional[datetime] = None
    connection_uptime: float = 0.0
    protocol_errors: int = 0

class MessageQueue:
    """Priority message queue with retry logic."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {
            MessagePriority.CRITICAL: asyncio.Queue(),
            MessagePriority.HIGH: asyncio.Queue(),
            MessagePriority.NORMAL: asyncio.Queue(),
            MessagePriority.LOW: asyncio.Queue()
        }
        self.retry_queue = asyncio.Queue()
        self.total_messages = 0
        
    async def enqueue(self, message: Dict[str, Any], metadata: MessageMetadata):
        """Add message to appropriate priority queue."""
        if self.total_messages >= self.max_size:
            # Drop lowest priority message
            await self._drop_low_priority_message()
        
        message_item = {"message": message, "metadata": metadata}
        
        await self.queues[metadata.priority].put(message_item)
        self.total_messages += 1
    
    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Get next message based on priority."""
        # Check retry queue first
        if not self.retry_queue.empty():
            try:
                item = self.retry_queue.get_nowait()
                self.total_messages -= 1
                return item
            except asyncio.QueueEmpty:
                pass
        
        # Check priority queues in order
        for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            try:
                item = self.queues[priority].get_nowait()
                self.total_messages -= 1
                return item
            except asyncio.QueueEmpty:
                continue
        
        return None
    
    async def requeue_for_retry(self, message: Dict[str, Any], metadata: MessageMetadata):
        """Add message back to retry queue."""
        if metadata.can_retry():
            metadata.retry_count += 1
            message_item = {"message": message, "metadata": metadata}
            await self.retry_queue.put(message_item)
            self.total_messages += 1
    
    async def _drop_low_priority_message(self):
        """Drop the oldest low priority message."""
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL]:
            try:
                self.queues[priority].get_nowait()
                self.total_messages -= 1
                logger.warning(f"Dropped {priority.name} priority message due to queue overflow")
                return
            except asyncio.QueueEmpty:
                continue
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            "critical": self.queues[MessagePriority.CRITICAL].qsize(),
            "high": self.queues[MessagePriority.HIGH].qsize(),
            "normal": self.queues[MessagePriority.NORMAL].qsize(),
            "low": self.queues[MessagePriority.LOW].qsize(),
            "retry": self.retry_queue.qsize(),
            "total": self.total_messages
        }

class MessageValidator:
    """Validates message integrity and format."""
    
    @staticmethod
    def validate_message_structure(message: Dict[str, Any]) -> bool:
        """Validate basic message structure."""
        required_fields = ["format", "version", "timestamp", "device_id"]
        
        for field in required_fields:
            if field not in message:
                logger.warning(f"Message missing required field: {field}")
                return False
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(message["timestamp"])
        except (ValueError, TypeError):
            logger.warning("Invalid timestamp format in message")
            return False
        
        return True
    
    @staticmethod
    def validate_message_content(message: Dict[str, Any], 
                                expected_format: MessageFormat) -> bool:
        """Validate message content based on expected format."""
        message_format = message.get("format")
        
        if message_format != expected_format.value:
            logger.warning(f"Message format mismatch: expected {expected_format.value}, got {message_format}")
            return False
        
        # Format-specific validation
        if expected_format == MessageFormat.ENVIRONMENTAL_DATA:
            return MessageValidator._validate_environmental_data(message)
        elif expected_format == MessageFormat.ML_ANALYSIS:
            return MessageValidator._validate_ml_analysis(message)
        elif expected_format == MessageFormat.DEVICE_STATUS:
            return MessageValidator._validate_device_status(message)
        elif expected_format == MessageFormat.ENVIRONMENTAL_ALERT:
            return MessageValidator._validate_environmental_alert(message)
        
        return True
    
    @staticmethod
    def _validate_environmental_data(message: Dict[str, Any]) -> bool:
        """Validate environmental data message."""
        if "sensor_data" not in message:
            logger.warning("Environmental data message missing sensor_data")
            return False
        
        sensor_data = message["sensor_data"]
        if not isinstance(sensor_data, dict):
            logger.warning("sensor_data must be a dictionary")
            return False
        
        # Validate sensor readings
        for sensor_name, reading in sensor_data.items():
            if not isinstance(reading, dict):
                continue
            
            if "value" not in reading:
                logger.warning(f"Sensor {sensor_name} missing value")
                return False
            
            # Validate numeric values
            try:
                float(reading["value"])
            except (ValueError, TypeError):
                logger.warning(f"Sensor {sensor_name} has invalid value")
                return False
        
        return True
    
    @staticmethod
    def _validate_ml_analysis(message: Dict[str, Any]) -> bool:
        """Validate ML analysis message."""
        if "analysis" not in message:
            logger.warning("ML analysis message missing analysis field")
            return False
        
        analysis = message["analysis"]
        required_fields = ["environmental_class", "confidence"]
        
        for field in required_fields:
            if field not in analysis:
                logger.warning(f"ML analysis missing {field}")
                return False
        
        # Validate confidence score
        try:
            confidence = float(analysis["confidence"])
            if not 0.0 <= confidence <= 1.0:
                logger.warning("ML confidence must be between 0.0 and 1.0")
                return False
        except (ValueError, TypeError):
            logger.warning("Invalid ML confidence value")
            return False
        
        return True
    
    @staticmethod
    def _validate_device_status(message: Dict[str, Any]) -> bool:
        """Validate device status message."""
        if "status" not in message:
            logger.warning("Device status message missing status field")
            return False
        
        status = message["status"]
        required_fields = ["system_health", "sensor_status"]
        
        for field in required_fields:
            if field not in status:
                logger.warning(f"Device status missing {field}")
                return False
        
        return True
    
    @staticmethod
    def _validate_environmental_alert(message: Dict[str, Any]) -> bool:
        """Validate environmental alert message."""
        if "alert" not in message:
            logger.warning("Environmental alert message missing alert field")
            return False
        
        alert = message["alert"]
        required_fields = ["type", "severity", "description"]
        
        for field in required_fields:
            if field not in alert:
                logger.warning(f"Environmental alert missing {field}")
                return False
        
        # Validate severity
        valid_severities = ["info", "warning", "error", "critical"]
        if alert["severity"] not in valid_severities:
            logger.warning(f"Invalid alert severity: {alert['severity']}")
            return False
        
        return True
    
    @staticmethod
    def calculate_checksum(message: Dict[str, Any]) -> str:
        """Calculate message checksum for integrity verification."""
        try:
            # Create deterministic JSON representation
            message_json = json.dumps(message, sort_keys=True, separators=(',', ':'))
            
            # Calculate SHA-256 hash
            return hashlib.sha256(message_json.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    @staticmethod
    def verify_checksum(message: Dict[str, Any], expected_checksum: str) -> bool:
        """Verify message checksum."""
        calculated_checksum = MessageValidator.calculate_checksum(message)
        return calculated_checksum == expected_checksum

class MessageCompressor:
    """Handles message compression for network optimization."""
    
    @staticmethod
    def compress_message(message: Dict[str, Any], 
                        level: CompressionLevel = CompressionLevel.MEDIUM) -> bytes:
        """Compress message using specified compression level."""
        try:
            # Convert to JSON
            message_json = json.dumps(message, separators=(',', ':'))
            message_bytes = message_json.encode('utf-8')
            
            if level == CompressionLevel.NONE:
                return message_bytes
            
            # Apply gzip compression
            compressed = gzip.compress(message_bytes, compresslevel=level.value)
            
            # Encode as base64 for safe transport
            return base64.b64encode(compressed)
            
        except Exception as e:
            logger.error(f"Message compression failed: {e}")
            return message_json.encode('utf-8')
    
    @staticmethod
    def decompress_message(compressed_data: bytes, 
                          level: CompressionLevel = CompressionLevel.MEDIUM) -> Dict[str, Any]:
        """Decompress message."""
        try:
            if level == CompressionLevel.NONE:
                message_json = compressed_data.decode('utf-8')
            else:
                # Decode from base64
                decoded_data = base64.b64decode(compressed_data)
                
                # Decompress
                decompressed = gzip.decompress(decoded_data)
                message_json = decompressed.decode('utf-8')
            
            return json.loads(message_json)
            
        except Exception as e:
            logger.error(f"Message decompression failed: {e}")
            return {}

class NetworkResilience:
    """Handles network resilience and fault tolerance."""
    
    def __init__(self):
        self.connection_health = {}
        self.backup_routes = {}
        self.network_latency = {}
        self.packet_loss_rates = {}
        
    def assess_connection_health(self, device_id: str) -> float:
        """Assess connection health score (0.0 to 1.0)."""
        if device_id not in self.connection_health:
            return 0.5  # Unknown connection, assume medium health
        
        return self.connection_health[device_id]
    
    def update_connection_metrics(self, device_id: str, latency_ms: float, 
                                 packet_loss: float, success: bool):
        """Update connection metrics for health assessment."""
        # Update latency
        if device_id not in self.network_latency:
            self.network_latency[device_id] = []
        
        self.network_latency[device_id].append(latency_ms)
        # Keep only recent measurements
        if len(self.network_latency[device_id]) > 100:
            self.network_latency[device_id] = self.network_latency[device_id][-100:]
        
        # Update packet loss
        if device_id not in self.packet_loss_rates:
            self.packet_loss_rates[device_id] = []
        
        self.packet_loss_rates[device_id].append(1.0 if not success else 0.0)
        if len(self.packet_loss_rates[device_id]) > 100:
            self.packet_loss_rates[device_id] = self.packet_loss_rates[device_id][-100:]
        
        # Calculate health score
        avg_latency = sum(self.network_latency[device_id]) / len(self.network_latency[device_id])
        avg_packet_loss = sum(self.packet_loss_rates[device_id]) / len(self.packet_loss_rates[device_id])
        
        # Health score based on latency and packet loss
        latency_score = max(0, 1.0 - (avg_latency / 1000.0))  # Normalize to 1 second
        loss_score = 1.0 - avg_packet_loss
        
        self.connection_health[device_id] = (latency_score + loss_score) / 2.0
    
    def select_best_route(self, target_device: str, available_routes: List[str]) -> Optional[str]:
        """Select the best route based on connection health."""
        if not available_routes:
            return None
        
        best_route = available_routes[0]
        best_health = self.assess_connection_health(best_route)
        
        for route in available_routes[1:]:
            health = self.assess_connection_health(route)
            if health > best_health:
                best_route = route
                best_health = health
        
        return best_route
    
    def should_use_backup_route(self, primary_route: str) -> bool:
        """Determine if backup route should be used."""
        health = self.assess_connection_health(primary_route)
        return health < 0.3  # Use backup if health is poor

class FurcateProtocol:
    """Enhanced Furcate protocol with comprehensive error handling and resilience."""
    
    def __init__(self, device_id: str, version: str = "1.0", 
                 enable_compression: bool = True, enable_encryption: bool = False):
        self.device_id = device_id
        self.version = version
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        # Core components
        self.message_queue = MessageQueue()
        self.validator = MessageValidator()
        self.compressor = MessageCompressor()
        self.resilience = NetworkResilience()
        
        # Statistics and monitoring
        self.stats = ProtocolStats()
        self.message_handlers: Dict[MessageFormat, Callable] = {}
        self.error_handlers: List[Callable] = []
        
        # Sequence tracking
        self.outbound_sequence = 0
        self.inbound_sequences: Dict[str, int] = {}
        
        # Active messages (for acknowledgment tracking)
        self.pending_acks: Dict[str, Dict[str, Any]] = {}
        
        # Protocol configuration
        self.default_ttl = 300  # 5 minutes
        self.max_retries = 3
        self.ack_timeout = 30  # seconds
        self.heartbeat_interval = 60  # seconds
        
        logger.info(f"🔗 Furcate Protocol initialized for device {device_id}")
    
    def register_message_handler(self, message_format: MessageFormat, 
                                handler: Callable[[Dict[str, Any]], None]):
        """Register handler for specific message format."""
        self.message_handlers[message_format] = handler
        logger.debug(f"Registered handler for {message_format.value}")
    
    def register_error_handler(self, handler: Callable[[Exception, Dict[str, Any]], None]):
        """Register error handler for protocol errors."""
        self.error_handlers.append(handler)
        logger.debug("Registered error handler")
    
    async def create_environmental_data_message(self, sensor_data: Dict[str, Any],
                                              ml_analysis: Dict[str, Any] = None,
                                              priority: MessagePriority = MessagePriority.NORMAL) -> Dict[str, Any]:
        """Create environmental data message with full validation."""
        try:
            # Compress sensor data if enabled
            compressed_sensor_data = self._compress_sensor_data(sensor_data)
            compressed_ml_data = self._compress_ml_data(ml_analysis) if ml_analysis else {}
            
            message = {
                "format": MessageFormat.ENVIRONMENTAL_DATA.value,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "sensor_data": compressed_sensor_data,
                "ml_analysis": compressed_ml_data,
                "data_quality": self._calculate_data_quality(sensor_data),
                "sequence_number": self._get_next_sequence()
            }
            
            # Add checksum
            message["checksum"] = self.validator.calculate_checksum(message)
            
            # Validate message
            if not self.validator.validate_message_structure(message):
                raise ValueError("Message validation failed")
            
            # Create metadata
            metadata = MessageMetadata(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sender_id=self.device_id,
                recipient_id=None,
                message_type=MessageFormat.ENVIRONMENTAL_DATA,
                priority=priority,
                ttl_seconds=self.default_ttl,
                compression_level=CompressionLevel.MEDIUM if self.enable_compression else CompressionLevel.NONE,
                sequence_number=message["sequence_number"]
            )
            
            # Queue message for sending
            await self.message_queue.enqueue(message, metadata)
            
            self.stats.messages_sent += 1
            self.stats.last_activity = datetime.now()
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to create environmental data message: {e}")
            await self._handle_protocol_error(e, {"operation": "create_environmental_data_message"})
            raise
    
    async def create_ml_analysis_message(self, analysis_results: Dict[str, Any],
                                       confidence: float,
                                       environmental_class: str,
                                       priority: MessagePriority = MessagePriority.HIGH) -> Dict[str, Any]:
        """Create ML analysis message."""
        try:
            message = {
                "format": MessageFormat.ML_ANALYSIS.value,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "analysis": {
                    "environmental_class": environmental_class,
                    "confidence": confidence,
                    "results": self._compress_ml_data(analysis_results),
                    "model_version": "1.0",  # Could be dynamic
                    "inference_time_ms": analysis_results.get("inference_time_ms", 0)
                },
                "sequence_number": self._get_next_sequence()
            }
            
            message["checksum"] = self.validator.calculate_checksum(message)
            
            if not self.validator.validate_message_structure(message):
                raise ValueError("ML analysis message validation failed")
            
            metadata = MessageMetadata(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sender_id=self.device_id,
                recipient_id=None,
                message_type=MessageFormat.ML_ANALYSIS,
                priority=priority,
                ttl_seconds=self.default_ttl,
                sequence_number=message["sequence_number"]
            )
            
            await self.message_queue.enqueue(message, metadata)
            
            self.stats.messages_sent += 1
            return message
            
        except Exception as e:
            logger.error(f"Failed to create ML analysis message: {e}")
            await self._handle_protocol_error(e, {"operation": "create_ml_analysis_message"})
            raise
    
    async def create_device_status_message(self, system_health: Dict[str, Any],
                                         sensor_status: Dict[str, Any],
                                         priority: MessagePriority = MessagePriority.NORMAL) -> Dict[str, Any]:
        """Create device status message."""
        try:
            message = {
                "format": MessageFormat.DEVICE_STATUS.value,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "status": {
                    "system_health": system_health,
                    "sensor_status": sensor_status,
                    "uptime_seconds": time.time() - getattr(self, 'start_time', time.time()),
                    "memory_usage_mb": self._get_memory_usage(),
                    "cpu_usage_percent": self._get_cpu_usage(),
                    "network_status": self._get_network_status()
                },
                "sequence_number": self._get_next_sequence()
            }
            
            message["checksum"] = self.validator.calculate_checksum(message)
            
            metadata = MessageMetadata(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sender_id=self.device_id,
                recipient_id=None,
                message_type=MessageFormat.DEVICE_STATUS,
                priority=priority,
                ttl_seconds=self.default_ttl,
                sequence_number=message["sequence_number"]
            )
            
            await self.message_queue.enqueue(message, metadata)
            
            self.stats.messages_sent += 1
            return message
            
        except Exception as e:
            logger.error(f"Failed to create device status message: {e}")
            await self._handle_protocol_error(e, {"operation": "create_device_status_message"})
            raise
    
    async def create_environmental_alert_message(self, alert: Dict[str, Any],
                                               priority: MessagePriority = MessagePriority.CRITICAL) -> Dict[str, Any]:
        """Create environmental alert message."""
        try:
            message = {
                "format": MessageFormat.ENVIRONMENTAL_ALERT.value,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "priority": priority.name.lower(),
                "alert": {
                    "type": alert.get("type", "unknown"),
                    "severity": alert.get("severity", "warning"),
                    "description": alert.get("message", "Environmental alert"),
                    "parameters": alert.get("parameters", {}),
                    "confidence": alert.get("confidence", 0.8),
                    "recommended_actions": alert.get("recommended_actions", []),
                    "affected_sensors": alert.get("affected_sensors", [])
                },
                "sequence_number": self._get_next_sequence()
            }
            
            message["checksum"] = self.validator.calculate_checksum(message)
            
            if not self.validator.validate_message_structure(message):
                raise ValueError("Environmental alert message validation failed")
            
            metadata = MessageMetadata(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sender_id=self.device_id,
                recipient_id=None,
                message_type=MessageFormat.ENVIRONMENTAL_ALERT,
                priority=priority,
                ttl_seconds=self.default_ttl * 2,  # Alerts have longer TTL
                max_retries=5,  # More retries for critical alerts
                sequence_number=message["sequence_number"]
            )
            
            await self.message_queue.enqueue(message, metadata)
            
            self.stats.messages_sent += 1
            return message
            
        except Exception as e:
            logger.error(f"Failed to create environmental alert message: {e}")
            await self._handle_protocol_error(e, {"operation": "create_environmental_alert_message"})
            raise
    
    async def create_discovery_beacon(self, capabilities: List[str], 
                                    environmental_zone: str,
                                    priority: MessagePriority = MessagePriority.LOW) -> Dict[str, Any]:
        """Create discovery beacon message."""
        try:
            message = {
                "format": MessageFormat.DISCOVERY_BEACON.value,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "discovery": {
                    "device_type": "furcate_nano",
                    "capabilities": capabilities,
                    "environmental_zone": environmental_zone,
                    "protocol_version": self.version,
                    "mesh_enabled": True,
                    "last_seen": datetime.now().isoformat(),
                    "device_info": {
                        "manufacturer": "Praecise Ltd",
                        "model": "Furcate Nano",
                        "firmware_version": "1.0.0"
                    }
                },
                "sequence_number": self._get_next_sequence()
            }
            
            message["checksum"] = self.validator.calculate_checksum(message)
            
            metadata = MessageMetadata(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sender_id=self.device_id,
                recipient_id=None,
                message_type=MessageFormat.DISCOVERY_BEACON,
                priority=priority,
                ttl_seconds=60,  # Short TTL for discovery
                max_retries=1,  # Limited retries for beacons
                sequence_number=message["sequence_number"]
            )
            
            await self.message_queue.enqueue(message, metadata)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to create discovery beacon: {e}")
            await self._handle_protocol_error(e, {"operation": "create_discovery_beacon"})
            raise
    
    async def process_incoming_message(self, raw_message: bytes, sender_id: str) -> bool:
        """Process incoming message with comprehensive validation."""
        try:
            # Update connection metrics
            start_time = time.time()
            
            # Decompress if needed
            if self.enable_compression:
                message = self.compressor.decompress_message(raw_message, CompressionLevel.MEDIUM)
            else:
                message_json = raw_message.decode('utf-8')
                message = json.loads(message_json)
            
            # Validate message structure
            if not self.validator.validate_message_structure(message):
                raise ValueError("Invalid message structure")
            
            # Verify checksum if present
            if "checksum" in message:
                expected_checksum = message.pop("checksum")
                if not self.validator.verify_checksum(message, expected_checksum):
                    raise ValueError("Checksum verification failed")
                message["checksum"] = expected_checksum  # Restore for processing
            
            # Check sequence number
            sequence_number = message.get("sequence_number", 0)
            if not self._validate_sequence_number(sender_id, sequence_number):
                logger.warning(f"Out-of-order message from {sender_id}: {sequence_number}")
            
            # Determine message format
            message_format_str = message.get("format", "unknown")
            try:
                message_format = MessageFormat(message_format_str)
            except ValueError:
                raise ValueError(f"Unknown message format: {message_format_str}")
            
            # Validate message content
            if not self.validator.validate_message_content(message, message_format):
                raise ValueError(f"Invalid content for {message_format.value}")
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.resilience.update_connection_metrics(sender_id, processing_time, 0.0, True)
            
            self.stats.messages_received += 1
            self.stats.bytes_received += len(raw_message)
            self.stats.last_activity = datetime.now()
            
            # Update average latency
            if self.stats.messages_received > 1:
                self.stats.average_latency_ms = (
                    (self.stats.average_latency_ms * (self.stats.messages_received - 1) + processing_time) 
                    / self.stats.messages_received
                )
            else:
                self.stats.average_latency_ms = processing_time
            
            # Route to appropriate handler
            if message_format in self.message_handlers:
                try:
                    await self.message_handlers[message_format](message)
                except Exception as e:
                    logger.error(f"Message handler failed for {message_format.value}: {e}")
                    await self._handle_protocol_error(e, {"message": message, "sender": sender_id})
            
            # Send acknowledgment for critical messages
            if message.get("priority") == "critical":
                await self._send_acknowledgment(message, sender_id)
            
            return True
            
        except Exception as e:
            # Update failure statistics
            self.stats.messages_failed += 1
            self.stats.protocol_errors += 1
            self.resilience.update_connection_metrics(sender_id, 0, 1.0, False)
            
            logger.error(f"Failed to process incoming message from {sender_id}: {e}")
            await self._handle_protocol_error(e, {"sender": sender_id, "raw_message_length": len(raw_message)})
            return False
    
    async def get_next_outbound_message(self) -> Optional[Tuple[bytes, MessageMetadata]]:
        """Get next message from queue for transmission."""
        try:
            item = await self.message_queue.dequeue()
            if not item:
                return None
            
            message = item["message"]
            metadata = item["metadata"]
            
            # Check if message has expired
            if metadata.is_expired():
                logger.warning(f"Dropping expired message: {metadata.message_id}")
                return await self.get_next_outbound_message()  # Get next message
            
            # Serialize message
            if self.enable_compression:
                serialized = self.compressor.compress_message(message, metadata.compression_level)
            else:
                serialized = json.dumps(message, separators=(',', ':')).encode('utf-8')
            
            # Update statistics
            self.stats.bytes_sent += len(serialized)
            
            # Track for acknowledgment if needed
            if metadata.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]:
                self.pending_acks[metadata.message_id] = {
                    "message": message,
                    "metadata": metadata,
                    "sent_at": datetime.now()
                }
            
            return serialized, metadata
            
        except Exception as e:
            logger.error(f"Failed to get outbound message: {e}")
            await self._handle_protocol_error(e, {"operation": "get_next_outbound_message"})
            return None
    
    async def handle_transmission_failure(self, metadata: MessageMetadata):
        """Handle failed message transmission."""
        try:
            if metadata.can_retry():
                # Create dummy message for requeue (actual message is in metadata context)
                message = {"retry": True, "original_id": metadata.message_id}
                await self.message_queue.requeue_for_retry(message, metadata)
                
                self.stats.messages_retried += 1
                logger.warning(f"Requeued message {metadata.message_id} for retry ({metadata.retry_count}/{metadata.max_retries})")
            else:
                logger.error(f"Message {metadata.message_id} failed after {metadata.max_retries} retries")
                self.stats.messages_failed += 1
                
        except Exception as e:
            logger.error(f"Failed to handle transmission failure: {e}")
            await self._handle_protocol_error(e, {"failed_message_id": metadata.message_id})
    
    def _compress_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress sensor data for efficient transmission."""
        if not self.enable_compression:
            return sensor_data
        
        compressed = {}
        
        for sensor_name, sensor_info in sensor_data.get("sensors", {}).items():
            if isinstance(sensor_info, dict) and "value" in sensor_info:
                # Keep only essential data for transmission
                compressed[sensor_name] = {
                    "v": sensor_info["value"],  # Shortened key
                    "q": round(sensor_info.get("quality", 1.0), 2),  # Quality
                    "t": sensor_info.get("timestamp", time.time()),  # Timestamp
                    "u": sensor_info.get("unit", "")[:10]  # Unit (truncated)
                }
        
        return {"sensors": compressed} if compressed else {}
    
    def _compress_ml_data(self, ml_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compress ML analysis data."""
        if not self.enable_compression or not ml_analysis:
            return ml_analysis or {}
        
        return {
            "class": ml_analysis.get("environmental_class", "normal"),
            "anomaly": round(ml_analysis.get("anomaly_score", 0.0), 3),
            "conf": round(ml_analysis.get("confidence", 0.0), 3),
            "features": ml_analysis.get("top_features", [])[:5]  # Top 5 features only
        }
    
    def _calculate_data_quality(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        qualities = []
        
        for sensor_info in sensor_data.get("sensors", {}).values():
            if isinstance(sensor_info, dict):
                quality = sensor_info.get("quality", 1.0)
                qualities.append(quality)
        
        return round(sum(qualities) / len(qualities), 3) if qualities else 0.0
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        self.outbound_sequence += 1
        return self.outbound_sequence
    
    def _validate_sequence_number(self, sender_id: str, sequence_number: int) -> bool:
        """Validate sequence number for ordering."""
        if sender_id not in self.inbound_sequences:
            self.inbound_sequences[sender_id] = sequence_number
            return True
        
        expected = self.inbound_sequences[sender_id] + 1
        if sequence_number >= expected:
            self.inbound_sequences[sender_id] = sequence_number
            return True
        
        return False  # Out of order
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return round(process.memory_info().rss / 1024 / 1024, 1)
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return round(psutil.cpu_percent(interval=0.1), 1)
        except ImportError:
            return 0.0
    
    def _get_network_status(self) -> Dict[str, Any]:
        """Get network status information."""
        return {
            "active_connections": len(self.resilience.connection_health),
            "average_health": round(
                sum(self.resilience.connection_health.values()) / 
                max(len(self.resilience.connection_health), 1), 2
            ),
            "total_messages_sent": self.stats.messages_sent,
            "total_messages_received": self.stats.messages_received,
            "error_rate": round(
                self.stats.messages_failed / max(self.stats.messages_sent, 1) * 100, 2
            )
        }
    
    async def _send_acknowledgment(self, original_message: Dict[str, Any], recipient_id: str):
        """Send acknowledgment for received message."""
        try:
            ack_message = {
                "format": MessageFormat.RESPONSE.value,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "response": {
                    "type": "acknowledgment",
                    "original_message_id": original_message.get("sequence_number"),
                    "status": "received",
                    "processing_time_ms": 0  # Could be calculated
                },
                "sequence_number": self._get_next_sequence()
            }
            
            ack_message["checksum"] = self.validator.calculate_checksum(ack_message)
            
            metadata = MessageMetadata(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                sender_id=self.device_id,
                recipient_id=recipient_id,
                message_type=MessageFormat.RESPONSE,
                priority=MessagePriority.HIGH,
                ttl_seconds=30,  # Short TTL for acks
                max_retries=1,
                sequence_number=ack_message["sequence_number"]
            )
            
            await self.message_queue.enqueue(ack_message, metadata)
            
        except Exception as e:
            logger.error(f"Failed to send acknowledgment: {e}")
    
    async def _handle_protocol_error(self, error: Exception, context: Dict[str, Any]):
        """Handle protocol errors with registered handlers."""
        try:
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            
            # Call registered error handlers
            for handler in self.error_handlers:
                try:
                    await handler(error, error_info)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            # Log error
            logger.error(f"Protocol error: {error_info}")
            
        except Exception as e:
            logger.critical(f"Failed to handle protocol error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive protocol statistics."""
        queue_sizes = self.message_queue.get_queue_sizes()
        
        # Calculate packet loss rate
        total_attempts = self.stats.messages_sent + self.stats.messages_failed
        packet_loss_rate = (
            self.stats.messages_failed / max(total_attempts, 1) * 100
            if total_attempts > 0 else 0.0
        )
        
        return {
            "protocol_version": self.version,
            "device_id": self.device_id,
            "statistics": {
                "messages_sent": self.stats.messages_sent,
                "messages_received": self.stats.messages_received,
                "messages_failed": self.stats.messages_failed,
                "messages_retried": self.stats.messages_retried,
                "bytes_sent": self.stats.bytes_sent,
                "bytes_received": self.stats.bytes_received,
                "average_latency_ms": round(self.stats.average_latency_ms, 2),
                "packet_loss_rate": round(packet_loss_rate, 2),
                "protocol_errors": self.stats.protocol_errors,
                "last_activity": self.stats.last_activity.isoformat() if self.stats.last_activity else None
            },
            "queue_status": queue_sizes,
            "compression": {
                "enabled": self.enable_compression,
                "ratio": round(self.stats.compression_ratio, 2)
            },
            "network_health": {
                "connections": len(self.resilience.connection_health),
                "average_health": round(
                    sum(self.resilience.connection_health.values()) / 
                    max(len(self.resilience.connection_health), 1), 2
                ),
                "pending_acknowledgments": len(self.pending_acks)
            },
            "configuration": {
                "default_ttl_seconds": self.default_ttl,
                "max_retries": self.max_retries,
                "heartbeat_interval_seconds": self.heartbeat_interval,
                "compression_enabled": self.enable_compression,
                "encryption_enabled": self.enable_encryption
            }
        }
    
    async def cleanup_expired_messages(self):
        """Clean up expired pending acknowledgments."""
        try:
            current_time = datetime.now()
            expired_acks = []
            
            for msg_id, ack_info in self.pending_acks.items():
                if (current_time - ack_info["sent_at"]).total_seconds() > self.ack_timeout:
                    expired_acks.append(msg_id)
            
            for msg_id in expired_acks:
                del self.pending_acks[msg_id]
                logger.warning(f"Acknowledgment timeout for message {msg_id}")
            
            if expired_acks:
                logger.info(f"Cleaned up {len(expired_acks)} expired acknowledgments")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired messages: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown protocol."""
        try:
            logger.info("🔗 Shutting down Furcate Protocol...")
            
            # Process remaining messages
            remaining_messages = 0
            while True:
                item = await self.message_queue.dequeue()
                if not item:
                    break
                remaining_messages += 1
                if remaining_messages > 100:  # Prevent infinite loop
                    break
            
            if remaining_messages > 0:
                logger.warning(f"Dropped {remaining_messages} unprocessed messages during shutdown")
            
            # Clear handlers
            self.message_handlers.clear()
            self.error_handlers.clear()
            
            logger.info("✅ Furcate Protocol shutdown complete")
            
        except Exception as e:
            logger.error(f"Protocol shutdown error: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_protocol():
        """Test the protocol implementation."""
        protocol = FurcateProtocol("test_device_001", enable_compression=True)
        
        print("Testing Furcate Protocol...")
        
        # Test environmental data message
        sensor_data = {
            "sensors": {
                "temperature": {"value": 23.5, "quality": 0.95, "unit": "°C"},
                "humidity": {"value": 67.2, "quality": 0.98, "unit": "%"},
                "pressure": {"value": 1013.2, "quality": 0.92, "unit": "hPa"}
            }
        }
        
        ml_analysis = {
            "environmental_class": "normal",
            "confidence": 0.85,
            "anomaly_score": 0.15,
            "top_features": ["temperature", "humidity"]
        }
        
        # Create messages
        env_msg = await protocol.create_environmental_data_message(sensor_data, ml_analysis)
        print(f"✅ Created environmental data message: {env_msg['format']}")
        
        alert = {
            "type": "temperature_high",
            "severity": "warning",
            "message": "Temperature above threshold",
            "parameters": {"threshold": 25.0, "current": 26.5}
        }
        
        alert_msg = await protocol.create_environmental_alert_message(alert)
        print(f"✅ Created alert message: {alert_msg['format']}")
        
        # Test message processing
        serialized_msg = json.dumps(env_msg).encode('utf-8')
        success = await protocol.process_incoming_message(serialized_msg, "remote_device")
        print(f"✅ Processed incoming message: {success}")
        
        # Get statistics
        stats = protocol.get_statistics()
        print(f"📊 Protocol statistics: {json.dumps(stats, indent=2, default=str)}")
        
        await protocol.shutdown()
        print("Protocol test completed!")
    
    # Run test
    asyncio.run(test_protocol())
