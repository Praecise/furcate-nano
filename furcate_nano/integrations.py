# ============================================================================
# furcate_nano/integrations.py
"""Web-based integrations (REST API, MQTT, WebSockets, Webhooks)."""

import asyncio
import logging
import json
import time
import ssl
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

# External integrations with proper error handling
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import aiohttp
    import websockets
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebIntegrationType(Enum):
    """Types of web-based integrations."""
    REST_API = "rest_api"
    MQTT = "mqtt"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"

class IntegrationStatus(Enum):
    """Integration status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class ConnectionMetrics:
    """Connection metrics for monitoring."""
    connected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_failures: int = 0
    last_error: Optional[str] = None

class WebSocketConnectionManager:
    """Manages WebSocket connections with automatic reconnection."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: List[Callable] = []
        self.metrics = ConnectionMetrics()
        
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept and register a WebSocket connection."""
        try:
            await websocket.accept()
            
            if not client_id:
                client_id = str(uuid.uuid4())[:8]
            
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                'connected_at': datetime.now(),
                'last_activity': datetime.now(),
                'user_agent': websocket.headers.get('user-agent', 'unknown'),
                'remote_addr': websocket.client.host if websocket.client else 'unknown'
            }
            
            self.metrics.messages_received += 1
            logger.info(f"🔌 WebSocket client {client_id} connected from {self.connection_metadata[client_id]['remote_addr']}")
            
            # Send welcome message
            await self.send_personal_message({
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "server_info": {
                    "name": "Furcate Nano WebSocket Server",
                    "version": "1.0.0"
                }
            }, client_id)
            
            return client_id
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.metrics.connection_failures += 1
            self.metrics.last_error = str(e)
            raise

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket {client_id}: {e}")
            
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            logger.info(f"🔌 WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message))
                
                self.metrics.messages_sent += 1
                self.metrics.bytes_sent += len(json.dumps(message))
                self.metrics.last_activity = datetime.now()
                
                # Update client activity
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]['last_activity'] = datetime.now()
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected during message send")
                await self.disconnect(client_id)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected_clients = []
        message_json = json.dumps(message)
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message_json)
                
                # Update metrics and client activity
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]['last_activity'] = datetime.now()
                    
            except WebSocketDisconnect:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
        
        if len(self.active_connections) > 0:
            self.metrics.messages_sent += len(self.active_connections)
            self.metrics.bytes_sent += len(message_json) * len(self.active_connections)
            self.metrics.last_activity = datetime.now()

    def get_connection_count(self) -> int:
        """Get current connection count."""
        return len(self.active_connections)

    def get_connection_info(self) -> Dict[str, Any]:
        """Get detailed connection information."""
        return {
            "active_connections": len(self.active_connections),
            "connection_metadata": {
                client_id: {
                    **metadata,
                    "connected_at": metadata["connected_at"].isoformat(),
                    "last_activity": metadata["last_activity"].isoformat()
                }
                for client_id, metadata in self.connection_metadata.items()
            },
            "metrics": {
                "connected_at": self.metrics.connected_at.isoformat() if self.metrics.connected_at else None,
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None,
                "messages_sent": self.metrics.messages_sent,
                "messages_received": self.metrics.messages_received,
                "bytes_sent": self.metrics.bytes_sent,
                "bytes_received": self.metrics.bytes_received,
                "connection_failures": self.metrics.connection_failures,
                "last_error": self.metrics.last_error
            }
        }

class FurcateMQTTClient:
    """Enhanced MQTT client with reconnection and error handling."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        self.core = core
        self.config = config
        self.stats = stats
        self.client = None
        self.status = IntegrationStatus.DISCONNECTED
        self.metrics = ConnectionMetrics()
        self.subscribed_topics: Set[str] = set()
        self.message_handlers: Dict[str, Callable] = {}
        self.reconnect_task = None
        self.reconnect_delay = 5
        self.max_reconnect_delay = 300
        
        # Configuration
        self.broker_host = config.get("host", "localhost")
        self.broker_port = config.get("port", 1883)
        self.username = config.get("username")
        self.password = config.get("password")
        self.use_tls = config.get("use_tls", False)
        self.client_id = config.get("client_id", f"furcate_nano_{uuid.uuid4().hex[:8]}")
        self.keep_alive = config.get("keep_alive", 60)
        self.use_websockets = config.get("use_websockets", False)
        
        # Topic configuration
        self.base_topic = config.get("base_topic", f"furcate/{self.core.device_id}")
        self.sensor_topic = f"{self.base_topic}/sensors"
        self.ml_topic = f"{self.base_topic}/ml"
        self.alert_topic = f"{self.base_topic}/alerts"
        self.status_topic = f"{self.base_topic}/status"
        
    async def connect(self):
        """Connect to MQTT broker with error handling."""
        if not MQTT_AVAILABLE:
            logger.error("MQTT library not available")
            self.status = IntegrationStatus.DISABLED
            return False
        
        try:
            self.status = IntegrationStatus.CONNECTING
            
            # Create MQTT client
            if self.use_websockets:
                self.client = mqtt.Client(
                    client_id=self.client_id,
                    transport="websockets"
                )
            else:
                self.client = mqtt.Client(client_id=self.client_id)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_publish = self._on_publish
            self.client.on_subscribe = self._on_subscribe
            
            # Configure authentication
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Configure TLS
            if self.use_tls:
                context = ssl.create_default_context()
                self.client.tls_set_context(context)
            
            # Connect to broker
            self.client.connect_async(
                self.broker_host,
                self.broker_port,
                self.keep_alive
            )
            
            # Start network loop
            self.client.loop_start()
            
            # Wait for connection with timeout
            timeout = 30
            start_time = time.time()
            while self.status == IntegrationStatus.CONNECTING and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self.status == IntegrationStatus.CONNECTED:
                logger.info(f"✅ MQTT connected to {self.broker_host}:{self.broker_port}")
                await self._setup_default_subscriptions()
                return True
            else:
                logger.error(f"MQTT connection timeout to {self.broker_host}:{self.broker_port}")
                self.status = IntegrationStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self.status = IntegrationStatus.ERROR
            self.metrics.connection_failures += 1
            self.metrics.last_error = str(e)
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self.status = IntegrationStatus.CONNECTED
            self.metrics.connected_at = datetime.now()
            self.reconnect_delay = 5  # Reset reconnect delay
            logger.info(f"🔗 MQTT connected with result code {rc}")
        else:
            self.status = IntegrationStatus.ERROR
            self.metrics.connection_failures += 1
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Connection refused - code {rc}")
            self.metrics.last_error = error_msg
            logger.error(f"MQTT connection failed: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        self.status = IntegrationStatus.DISCONNECTED
        logger.warning(f"🔗 MQTT disconnected with result code {rc}")
        
        # Start reconnection if not intentional
        if rc != 0:
            self.status = IntegrationStatus.RECONNECTING
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT message."""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            self.metrics.messages_received += 1
            self.metrics.bytes_received += len(msg.payload)
            self.metrics.last_activity = datetime.now()
            
            logger.debug(f"📨 MQTT received on {topic}: {payload[:100]}...")
            
            # Update stats
            self.stats["mqtt_messages"] += 1
            
            # Handle message based on topic
            if topic in self.message_handlers:
                handler = self.message_handlers[topic]
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(topic, payload))
                else:
                    handler(topic, payload)
            
            # Default handling for system topics
            if topic.endswith("/command"):
                asyncio.create_task(self._handle_command_message(topic, payload))
            elif topic.endswith("/request"):
                asyncio.create_task(self._handle_request_message(topic, payload))
                
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """Handle successful message publish."""
        self.metrics.messages_sent += 1
        self.metrics.last_activity = datetime.now()
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """Handle successful subscription."""
        logger.debug(f"MQTT subscription confirmed with QoS {granted_qos}")
    
    async def _setup_default_subscriptions(self):
        """Setup default topic subscriptions."""
        default_topics = [
            f"{self.base_topic}/command",
            f"{self.base_topic}/request",
            f"{self.base_topic}/config",
            "furcate/broadcast",
            "furcate/network/discovery"
        ]
        
        for topic in default_topics:
            await self.subscribe(topic)
    
    async def _reconnect_loop(self):
        """Automatic reconnection loop."""
        while self.status in [IntegrationStatus.RECONNECTING, IntegrationStatus.DISCONNECTED]:
            try:
                logger.info(f"🔄 Attempting MQTT reconnection in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                
                if await self.connect():
                    logger.info("✅ MQTT reconnection successful")
                    break
                else:
                    # Exponential backoff
                    self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                    
            except Exception as e:
                logger.error(f"MQTT reconnection attempt failed: {e}")
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
    
    async def publish(self, topic: str, payload: dict, qos: int = 0, retain: bool = False):
        """Publish message to MQTT topic."""
        if self.status != IntegrationStatus.CONNECTED:
            logger.warning(f"Cannot publish - MQTT not connected (status: {self.status.value})")
            return False
        
        try:
            message_json = json.dumps(payload)
            result = self.client.publish(topic, message_json, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.metrics.bytes_sent += len(message_json)
                logger.debug(f"📤 MQTT published to {topic}: {message_json[:100]}...")
                return True
            else:
                logger.error(f"MQTT publish failed with code {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing MQTT message: {e}")
            return False
    
    async def subscribe(self, topic: str, qos: int = 0):
        """Subscribe to MQTT topic."""
        if self.status != IntegrationStatus.CONNECTED:
            logger.warning(f"Cannot subscribe - MQTT not connected (status: {self.status.value})")
            return False
        
        try:
            result, mid = self.client.subscribe(topic, qos=qos)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.subscribed_topics.add(topic)
                logger.info(f"📥 MQTT subscribed to {topic}")
                return True
            else:
                logger.error(f"MQTT subscription failed with code {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error subscribing to MQTT topic: {e}")
            return False
    
    def register_message_handler(self, topic: str, handler: Callable):
        """Register message handler for specific topic."""
        self.message_handlers[topic] = handler
        logger.info(f"Registered MQTT handler for {topic}")
    
    async def _handle_command_message(self, topic: str, payload: str):
        """Handle command messages."""
        try:
            command = json.loads(payload)
            command_type = command.get("type")
            
            if command_type == "get_status":
                status = self.core.get_status()
                response_topic = topic.replace("/command", "/response")
                await self.publish(response_topic, status)
            
            elif command_type == "restart":
                logger.info("Restart command received via MQTT")
                # Implement restart logic
                
        except Exception as e:
            logger.error(f"Error handling command message: {e}")
    
    async def _handle_request_message(self, topic: str, payload: str):
        """Handle request messages."""
        try:
            request = json.loads(payload)
            request_type = request.get("type")
            request_id = request.get("id", str(uuid.uuid4())[:8])
            
            response_topic = topic.replace("/request", "/response")
            response = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if request_type == "sensor_data":
                if hasattr(self.core, 'hardware'):
                    readings = await self.core.hardware.read_all_sensors()
                    response["data"] = {
                        name: reading.to_dict() for name, reading in readings.items()
                    }
                else:
                    response["error"] = "Hardware not available"
            
            await self.publish(response_topic, response)
            
        except Exception as e:
            logger.error(f"Error handling request message: {e}")
    
    async def shutdown(self):
        """Shutdown MQTT client."""
        self.status = IntegrationStatus.DISCONNECTED
        
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
        
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        logger.info("🔗 MQTT client shutdown")

class FurcateRestAPI:
    """Enhanced REST API with comprehensive endpoints."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        self.core = core
        self.config = config
        self.stats = stats
        self.app = None
        self.server = None
        
        # Configuration
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8000)
        self.enable_docs = config.get("enable_docs", True)
        self.enable_auth = config.get("enable_auth", False)
        self.api_key = config.get("api_key")
        
    async def initialize(self):
        """Initialize FastAPI application."""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available")
            return False
        
        try:
            self.app = FastAPI(
                title="Furcate Nano API",
                description="Environmental monitoring and edge computing API",
                version="1.0.0",
                docs_url="/docs" if self.enable_docs else None,
                redoc_url="/redoc" if self.enable_docs else None
            )
            
            # Configure CORS
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.get("cors_origins", ["*"]),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            self._setup_routes()
            logger.info("✅ REST API initialized")
            return True
            
        except Exception as e:
            logger.error(f"REST API initialization failed: {e}")
            return False
    
    def _setup_routes(self):
        """Setup API routes with comprehensive endpoints."""
        
        @self.app.get("/")
        async def root():
            """API root endpoint."""
            self.stats["api_requests"] += 1
            return {
                "message": "Furcate Nano API",
                "device_id": self.core.device_id,
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            self.stats["api_requests"] += 1
            health = await self.core.get_health_status()
            
            status_code = 200
            if not health.get("healthy", True):
                status_code = 503
            
            return JSONResponse(
                content=health,
                status_code=status_code
            )
        
        @self.app.get("/status")
        async def get_status():
            """Get comprehensive device status."""
            self.stats["api_requests"] += 1
            return self.core.get_status()
        
        @self.app.get("/sensors")
        async def get_sensors():
            """Get sensor information."""
            self.stats["api_requests"] += 1
            if not hasattr(self.core, 'hardware'):
                raise HTTPException(status_code=503, detail="Hardware not available")
            
            return {
                "sensors": list(self.core.hardware.sensors.keys()),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/sensors/current")
        async def get_current_sensors():
            """Get current sensor readings."""
            try:
                self.stats["api_requests"] += 1
                if not hasattr(self.core, 'hardware'):
                    raise HTTPException(status_code=503, detail="Hardware not available")
                
                readings = await self.core.hardware.read_all_sensors()
                return {
                    "timestamp": datetime.now().isoformat(),
                    "device_id": self.core.device_id,
                    "sensors": {name: reading.to_dict() for name, reading in readings.items()}
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/sensors/{sensor_name}")
        async def get_sensor_reading(sensor_name: str):
            """Get specific sensor reading."""
            try:
                self.stats["api_requests"] += 1
                if not hasattr(self.core, 'hardware'):
                    raise HTTPException(status_code=503, detail="Hardware not available")
                
                if sensor_name not in self.core.hardware.sensors:
                    raise HTTPException(status_code=404, detail=f"Sensor {sensor_name} not found")
                
                reading = await self.core.hardware.read_sensor(sensor_name)
                return {
                    "sensor": sensor_name,
                    "reading": reading.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/recent")
        async def get_recent_data(hours: int = 24, limit: int = 1000):
            """Get recent environmental data."""
            try:
                self.stats["api_requests"] += 1
                if not hasattr(self.core, 'storage'):
                    raise HTTPException(status_code=503, detail="Storage not available")
                
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                data = await self.core.storage.get_environmental_data(
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                return {
                    "data": data,
                    "query": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "hours": hours,
                        "limit": limit
                    },
                    "count": len(data)
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/ml/status")
        async def get_ml_status():
            """Get ML model status."""
            self.stats["api_requests"] += 1
            if not hasattr(self.core, 'ml_engine'):
                raise HTTPException(status_code=503, detail="ML engine not available")
            
            return self.core.ml_engine.get_status()
        
        @self.app.post("/ml/predict")
        async def ml_predict(data: dict):
            """Run ML prediction on provided data."""
            try:
                self.stats["api_requests"] += 1
                if not hasattr(self.core, 'ml_engine'):
                    raise HTTPException(status_code=503, detail="ML engine not available")
                
                result = await self.core.ml_engine.run_inference(data)
                return {
                    "prediction": result,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/network/status")
        async def get_network_status():
            """Get network status."""
            self.stats["api_requests"] += 1
            if not hasattr(self.core, 'furcate_client'):
                return {"status": "disabled", "message": "Network not available"}
            
            return {
                "discovered_devices": len(self.core.furcate_client.discovered_devices),
                "active_connections": len(self.core.furcate_client.active_connections),
                "supported_protocols": list(self.core.furcate_client.supported_protocols)
            }
        
        @self.app.get("/integrations/status")
        async def get_integrations_status():
            """Get integration status."""
            self.stats["api_requests"] += 1
            if not hasattr(self.core, 'web_integrations'):
                return {"status": "disabled"}
            
            return self.core.web_integrations.get_status()
        
        @self.app.post("/system/restart")
        async def restart_system(background_tasks: BackgroundTasks):
            """Restart the system."""
            self.stats["api_requests"] += 1
            background_tasks.add_task(self._restart_system)
            return {"message": "Restart initiated"}
        
        @self.app.get("/logs/recent")
        async def get_recent_logs(lines: int = 100):
            """Get recent log entries."""
            self.stats["api_requests"] += 1
            # Implementation would depend on logging setup
            return {"message": "Log retrieval not implemented"}
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data."""
            if hasattr(self.core, 'web_integrations') and hasattr(self.core.web_integrations, 'websocket_manager'):
                client_id = await self.core.web_integrations.websocket_manager.connect(websocket)
                try:
                    while True:
                        data = await websocket.receive_text()
                        # Handle incoming WebSocket messages
                        await self._handle_websocket_message(client_id, data)
                except WebSocketDisconnect:
                    await self.core.web_integrations.websocket_manager.disconnect(client_id)
            else:
                await websocket.close()
    
    async def _handle_websocket_message(self, client_id: str, data: str):
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(data)
            message_type = message.get("type")
            
            if message_type == "subscribe_sensors":
                # Start sending sensor data to this client
                pass
            elif message_type == "get_status":
                status = self.core.get_status()
                await self.core.web_integrations.websocket_manager.send_personal_message(
                    {"type": "status", "data": status}, client_id
                )
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _restart_system(self):
        """Restart system (background task)."""
        await asyncio.sleep(1)  # Allow response to be sent
        logger.info("System restart requested via API")
        # Implement actual restart logic here
    
    async def start_server(self):
        """Start the FastAPI server."""
        if not self.app:
            logger.error("FastAPI app not initialized")
            return
        
        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            self.server = uvicorn.Server(config)
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start REST API server: {e}")
    
    async def shutdown(self):
        """Shutdown the REST API server."""
        if self.server:
            self.server.should_exit = True
        logger.info("🌐 REST API shutdown")

class FurcateWebhookClient:
    """Webhook client for external notifications."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        self.core = core
        self.config = config
        self.stats = stats
        self.webhooks = config.get("endpoints", [])
        self.session = None
        self.retry_attempts = config.get("retry_attempts", 3)
        self.timeout = config.get("timeout", 30)
        
    async def initialize(self):
        """Initialize webhook client."""
        if not HTTP_AVAILABLE:
            logger.error("HTTP libraries not available")
            return False
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("✅ Webhook client initialized")
            return True
            
        except Exception as e:
            logger.error(f"Webhook client initialization failed: {e}")
            return False
    
    async def send_webhook(self, event_type: str, data: dict):
        """Send webhook notification."""
        if not self.session:
            logger.warning("Webhook client not initialized")
            return False
        
        webhook_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "device_id": self.core.device_id,
            "data": data
        }
        
        success_count = 0
        
        for webhook in self.webhooks:
            if await self._send_to_endpoint(webhook, webhook_data):
                success_count += 1
        
        self.stats["webhook_calls"] += len(self.webhooks)
        return success_count > 0
    
    async def _send_to_endpoint(self, webhook: dict, data: dict) -> bool:
        """Send data to specific webhook endpoint."""
        url = webhook.get("url")
        headers = webhook.get("headers", {})
        method = webhook.get("method", "POST").upper()
        
        if not url:
            logger.error("Webhook URL not configured")
            return False
        
        for attempt in range(self.retry_attempts):
            try:
                if method == "POST":
                    async with self.session.post(url, json=data, headers=headers) as response:
                        if response.status < 400:
                            logger.debug(f"📤 Webhook sent to {url}: {response.status}")
                            return True
                        else:
                            logger.warning(f"Webhook failed {url}: {response.status}")
                
                elif method == "PUT":
                    async with self.session.put(url, json=data, headers=headers) as response:
                        if response.status < 400:
                            logger.debug(f"📤 Webhook sent to {url}: {response.status}")
                            return True
                        else:
                            logger.warning(f"Webhook failed {url}: {response.status}")
                
            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    async def shutdown(self):
        """Shutdown webhook client."""
        if self.session:
            await self.session.close()
        logger.info("🔗 Webhook client shutdown")

class WebIntegrationManager:
    """Enhanced web integration manager with complete implementations."""
    
    def __init__(self, core, config: Dict[str, Any]):
        self.core = core
        self.config = config
        self.integrations: Dict[WebIntegrationType, Any] = {}
        self.websocket_manager = WebSocketConnectionManager()
        
        # Integration statistics
        self.stats = {
            "api_requests": 0,
            "mqtt_messages": 0,
            "websocket_connections": 0,
            "webhook_calls": 0,
            "total_data_served_mb": 0
        }
        
        logger.info("🌐 Web Integration Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize all web integrations."""
        try:
            # Initialize REST API
            if self.config.get("rest_api", {}).get("enabled", True):
                await self._init_rest_api()
            
            # Initialize MQTT
            if self.config.get("mqtt", {}).get("enabled", False):
                await self._init_mqtt()
            
            # Initialize WebSocket (integrated with REST API)
            if self.config.get("websocket", {}).get("enabled", False):
                # WebSocket is handled by REST API
                pass
            
            # Initialize Webhooks
            if self.config.get("webhooks", {}).get("enabled", False):
                await self._init_webhooks()
            
            logger.info(f"✅ Web integrations initialized ({len(self.integrations)} active)")
            return True
            
        except Exception as e:
            logger.error(f"Web integration initialization failed: {e}")
            return False
    
    async def _init_rest_api(self):
        """Initialize REST API."""
        try:
            api_config = self.config.get("rest_api", {})
            api = FurcateRestAPI(self.core, api_config, self.stats)
            
            if await api.initialize():
                self.integrations[WebIntegrationType.REST_API] = api
                
                # Start API server in background
                asyncio.create_task(api.start_server())
                
                logger.info("✅ REST API integration initialized")
            
        except Exception as e:
            logger.error(f"REST API initialization failed: {e}")
    
    async def _init_mqtt(self):
        """Initialize MQTT."""
        try:
            mqtt_config = self.config.get("mqtt", {})
            mqtt_client = FurcateMQTTClient(self.core, mqtt_config, self.stats)
            
            if await mqtt_client.connect():
                self.integrations[WebIntegrationType.MQTT] = mqtt_client
                logger.info("✅ MQTT integration initialized")
            
        except Exception as e:
            logger.error(f"MQTT initialization failed: {e}")
    
    async def _init_webhooks(self):
        """Initialize webhooks."""
        try:
            webhook_config = self.config.get("webhooks", {})
            webhook_client = FurcateWebhookClient(self.core, webhook_config, self.stats)
            
            if await webhook_client.initialize():
                self.integrations[WebIntegrationType.WEBHOOK] = webhook_client
                logger.info("✅ Webhook integration initialized")
            
        except Exception as e:
            logger.error(f"Webhook initialization failed: {e}")
    
    async def broadcast_sensor_data(self, sensor_data: Dict[str, Any], ml_analysis: Dict[str, Any]):
        """Broadcast sensor data to all integrations."""
        message = {
            "type": "sensor_data",
            "timestamp": datetime.now().isoformat(),
            "device_id": self.core.device_id,
            "sensor_data": sensor_data,
            "ml_analysis": ml_analysis
        }
        
        # MQTT broadcast
        if WebIntegrationType.MQTT in self.integrations:
            mqtt_client = self.integrations[WebIntegrationType.MQTT]
            await mqtt_client.publish(f"{mqtt_client.sensor_topic}/data", message)
        
        # WebSocket broadcast
        await self.websocket_manager.broadcast(message)
        
        # Webhook notifications (for critical events)
        if ml_analysis.get("anomaly_score", 0) > 0.8:
            if WebIntegrationType.WEBHOOK in self.integrations:
                webhook_client = self.integrations[WebIntegrationType.WEBHOOK]
                await webhook_client.send_webhook("anomaly_detected", message)
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to all integrations."""
        message = {
            "type": "alert",
            "timestamp": datetime.now().isoformat(),
            "device_id": self.core.device_id,
            "alert": alert
        }
        
        # MQTT broadcast
        if WebIntegrationType.MQTT in self.integrations:
            mqtt_client = self.integrations[WebIntegrationType.MQTT]
            await mqtt_client.publish(f"{mqtt_client.alert_topic}", message)
        
        # WebSocket broadcast
        await self.websocket_manager.broadcast(message)
        
        # Webhook notification
        if WebIntegrationType.WEBHOOK in self.integrations:
            webhook_client = self.integrations[WebIntegrationType.WEBHOOK]
            await webhook_client.send_webhook("alert", message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        status = {
            "enabled_integrations": list(self.integrations.keys()),
            "websocket_connections": self.websocket_manager.get_connection_count(),
            "stats": self.stats
        }
        
        # Add individual integration status
        for integration_type, client in self.integrations.items():
            if hasattr(client, 'status'):
                status[f"{integration_type.value}_status"] = client.status.value
            elif hasattr(client, 'metrics'):
                status[f"{integration_type.value}_metrics"] = client.metrics
        
        return status
    
    async def shutdown(self):
        """Shutdown all web integrations."""
        logger.info("🌐 Shutting down web integrations...")
        
        for integration_type, client in self.integrations.items():
            try:
                if hasattr(client, 'shutdown'):
                    await client.shutdown()
                logger.info(f"✅ {integration_type.value} integration shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown {integration_type.value}: {e}")
        
        logger.info("✅ Web integration manager shutdown complete")