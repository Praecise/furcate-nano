# ============================================================================
# furcate_nano/integrations.py
"""Web-based integrations (REST API, MQTT, WebSockets, Webhooks)."""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

# External integrations
try:
    from fastapi import FastAPI, HTTPException, WebSocket
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

class WebIntegrationManager:
    """Manager for web-based integrations (APIs, MQTT, WebSockets, Webhooks)."""
    
    def __init__(self, core, config: Dict[str, Any]):
        """Initialize web integration manager."""
        self.core = core
        self.config = config
        self.integrations: Dict[WebIntegrationType, Any] = {}
        
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
            
            # Initialize WebSocket
            if self.config.get("websocket", {}).get("enabled", False):
                await self._init_websocket()
            
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
        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available - skipping REST API")
            return
        
        try:
            api_config = self.config.get("rest_api", {})
            api = FurcateRestAPI(self.core, api_config, self.stats)
            
            self.integrations[WebIntegrationType.REST_API] = api
            
            # Start API server in background
            asyncio.create_task(api.start_server())
            
            logger.info("✅ REST API integration initialized")
            
        except Exception as e:
            logger.error(f"REST API initialization failed: {e}")
    
    async def _init_mqtt(self):
        """Initialize MQTT."""
        if not MQTT_AVAILABLE:
            logger.warning("MQTT not available - skipping MQTT integration")
            return
        
        try:
            mqtt_config = self.config.get("mqtt", {})
            mqtt_client = FurcateMQTTClient(self.core, mqtt_config, self.stats)
            
            await mqtt_client.connect()
            self.integrations[WebIntegrationType.MQTT] = mqtt_client
            
            logger.info("✅ MQTT integration initialized")
            
        except Exception as e:
            logger.error(f"MQTT initialization failed: {e}")
    
    async def _init_websocket(self):
        """Initialize WebSocket."""
        try:
            ws_config = self.config.get("websocket", {})
            ws_server = FurcateWebSocketServer(self.core, ws_config, self.stats)
            
            self.integrations[WebIntegrationType.WEBSOCKET] = ws_server
            
            # Start WebSocket server
            asyncio.create_task(ws_server.start_server())
            
            logger.info("✅ WebSocket integration initialized")
            
        except Exception as e:
            logger.error(f"WebSocket initialization failed: {e}")
    
    async def _init_webhooks(self):
        """Initialize webhooks."""
        try:
            webhook_config = self.config.get("webhooks", {})
            webhook_client = FurcateWebhookClient(self.core, webhook_config, self.stats)
            
            self.integrations[WebIntegrationType.WEBHOOK] = webhook_client
            
            logger.info("✅ Webhook integration initialized")
            
        except Exception as e:
            logger.error(f"Webhook initialization failed: {e}")
    
    async def broadcast_sensor_data(self, sensor_data: Dict[str, Any], ml_analysis: Dict[str, Any]):
        """Broadcast sensor data to web integrations."""
        try:
            message = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.core.device_id,
                "sensor_data": sensor_data,
                "ml_analysis": ml_analysis,
                "message_type": "sensor_update"
            }
            
            # Send to MQTT
            if WebIntegrationType.MQTT in self.integrations:
                await self.integrations[WebIntegrationType.MQTT].send_sensor_data(message)
            
            # Send to WebSocket clients
            if WebIntegrationType.WEBSOCKET in self.integrations:
                await self.integrations[WebIntegrationType.WEBSOCKET].broadcast(message)
            
            # Send to Webhooks
            if WebIntegrationType.WEBHOOK in self.integrations:
                await self.integrations[WebIntegrationType.WEBHOOK].send_sensor_data(message)
            
        except Exception as e:
            logger.error(f"Web broadcast failed: {e}")
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to web integrations."""
        try:
            alert_message = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.core.device_id,
                "alert": alert,
                "message_type": "environmental_alert",
                "priority": alert.get("severity", "warning")
            }
            
            # Send to all web integrations
            for integration in self.integrations.values():
                if hasattr(integration, 'send_alert'):
                    await integration.send_alert(alert_message)
                elif hasattr(integration, 'broadcast'):
                    await integration.broadcast(alert_message)
            
        except Exception as e:
            logger.error(f"Web alert broadcast failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get web integration statistics."""
        return {
            **self.stats,
            "active_integrations": [integration_type.value for integration_type in self.integrations.keys()],
            "integration_count": len(self.integrations)
        }
    
    async def shutdown(self):
        """Shutdown web integrations."""
        logger.info("🌐 Shutting down web integrations...")
        
        for integration_type, client in self.integrations.items():
            try:
                if hasattr(client, 'shutdown'):
                    await client.shutdown()
                logger.info(f"✅ {integration_type.value} integration shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown {integration_type.value}: {e}")
        
        logger.info("✅ Web integration manager shutdown complete")


# ============================================================================
# REST API
# ============================================================================

class FurcateRestAPI:
    """REST API for Furcate Nano data access."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        """Initialize REST API."""
        self.core = core
        self.config = config
        self.stats = stats
        self.app = FastAPI(
            title="Furcate Nano API",
            description="Environmental monitoring data access",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            self.stats["api_requests"] += 1
            return {"message": "Furcate Nano API", "device_id": self.core.device_id}
        
        @self.app.get("/status")
        async def get_status():
            """Get device status."""
            self.stats["api_requests"] += 1
            return self.core.get_status()
        
        @self.app.get("/sensors/current")
        async def get_current_sensors():
            """Get current sensor readings."""
            try:
                self.stats["api_requests"] += 1
                readings = await self.core.hardware.read_all_sensors()
                return {
                    "timestamp": datetime.now().isoformat(),
                    "device_id": self.core.device_id,
                    "sensors": {name: reading.to_dict() for name, reading in readings.items()}
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/recent")
        async def get_recent_data(hours: int = 24):
            """Get recent environmental data."""
            try:
                self.stats["api_requests"] += 1
                data = await self.core.storage.get_recent_environmental_data(hours)
                return {
                    "records": len(data),
                    "hours": hours,
                    "data": data
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts")
        async def get_alerts(hours: int = 24, severity: str = None):
            """Get recent alerts."""
            try:
                self.stats["api_requests"] += 1
                alerts = await self.core.storage.get_alerts(hours, severity=severity)
                return {
                    "alerts": len(alerts),
                    "hours": hours,
                    "severity_filter": severity,
                    "data": alerts
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/networks/status")
        async def get_network_status():
            """Get network status for all networks."""
            try:
                self.stats["api_requests"] += 1
                status = {
                    "tenzro_network": self.core.tenzro_client.stats if hasattr(self.core, 'tenzro_client') else {},
                    "furcate_network": self.core.furcate_client.local_stats if hasattr(self.core, 'furcate_client') else {},
                    "mesh_network": self.core.mesh.get_status(),
                    "web_integrations": self.stats
                }
                return status
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws/live")
        async def websocket_live_data(websocket: WebSocket):
            """WebSocket endpoint for live data streaming."""
            await websocket.accept()
            try:
                while True:
                    # Send current sensor data
                    readings = await self.core.hardware.read_all_sensors()
                    ml_analysis = await self.core.edge_ml.process_environmental_data(
                        {name: reading.value for name, reading in readings.items()}
                    )
                    
                    message = {
                        "timestamp": datetime.now().isoformat(),
                        "device_id": self.core.device_id,
                        "sensors": {name: reading.to_dict() for name, reading in readings.items()},
                        "ml_analysis": ml_analysis
                    }
                    
                    await websocket.send_json(message)
                    await asyncio.sleep(5)  # Send every 5 seconds
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await websocket.close()
    
    async def start_server(self):
        """Start the API server."""
        try:
            host = self.config.get("host", "0.0.0.0")
            port = self.config.get("port", 8000)
            
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            logger.info(f"🚀 Starting REST API server on {host}:{port}")
            await server.serve()
            
        except Exception as e:
            logger.error(f"API server startup failed: {e}")


# ============================================================================
# MQTT CLIENT
# ============================================================================

class FurcateMQTTClient:
    """MQTT client for IoT platform integration."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        """Initialize MQTT client."""
        self.core = core
        self.config = config
        self.stats = stats
        self.client = mqtt.Client()
        self.connected = False
        
        # MQTT topics
        self.base_topic = config.get("base_topic", f"furcate/{core.device_id}")
        
        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
        # Authentication
        username = config.get("username")
        password = config.get("password")
        if username and password:
            self.client.username_pw_set(username, password)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self.connected = True
            logger.info(f"✅ Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        self.connected = False
        logger.warning("MQTT disconnected")
    
    async def connect(self):
        """Connect to MQTT broker."""
        try:
            broker = self.config.get("broker", "localhost")
            port = self.config.get("port", 1883)
            keepalive = self.config.get("keepalive", 60)
            
            self.client.connect(broker, port, keepalive)
            self.client.loop_start()
            
            # Wait for connection
            await asyncio.sleep(1)
            
            if not self.connected:
                logger.warning("MQTT connection failed - continuing in simulation mode")
            
        except Exception as e:
            logger.warning(f"MQTT connection failed: {e} - continuing in simulation mode")
    
    async def send_sensor_data(self, data: Dict[str, Any]):
        """Send sensor data via MQTT."""
        try:
            if self.connected:
                topic = f"{self.base_topic}/sensors/data"
                self.client.publish(topic, json.dumps(data))
            
            self.stats["mqtt_messages"] += 1
            
        except Exception as e:
            logger.warning(f"MQTT send failed: {e}")
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert via MQTT."""
        try:
            if self.connected:
                topic = f"{self.base_topic}/alerts/{alert.get('priority', 'normal')}"
                self.client.publish(topic, json.dumps(alert), qos=2)
            
            self.stats["mqtt_messages"] += 1
            
        except Exception as e:
            logger.warning(f"MQTT alert send failed: {e}")


# ============================================================================
# WEBSOCKET SERVER
# ============================================================================

class FurcateWebSocketServer:
    """WebSocket server for real-time data streaming."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        """Initialize WebSocket server."""
        self.core = core
        self.config = config
        self.stats = stats
        self.clients = set()
    
    async def start_server(self):
        """Start WebSocket server."""
        try:
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8765)
            
            if HTTP_AVAILABLE:
                async def handler(websocket, path):
                    """Handle WebSocket connections."""
                    self.clients.add(websocket)
                    self.stats["websocket_connections"] += 1
                    try:
                        await websocket.wait_closed()
                    finally:
                        self.clients.remove(websocket)
                
                import websockets
                server = await websockets.serve(handler, host, port)
                logger.info(f"🔌 WebSocket server started on {host}:{port}")
                await server.wait_closed()
            else:
                logger.warning("WebSocket server not available")
                
        except Exception as e:
            logger.error(f"WebSocket server failed: {e}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients],
                return_exceptions=True
            )


# ============================================================================
# WEBHOOK CLIENT
# ============================================================================

class FurcateWebhookClient:
    """Webhook client for HTTP-based integrations."""
    
    def __init__(self, core, config: Dict[str, Any], stats: Dict[str, Any]):
        """Initialize webhook client."""
        self.core = core
        self.config = config
        self.stats = stats
        self.webhooks = config.get("endpoints", [])
    
    async def send_sensor_data(self, data: Dict[str, Any]):
        """Send sensor data to webhooks."""
        for webhook_config in self.webhooks:
            if webhook_config.get("enabled", True):
                await self._send_webhook(webhook_config["url"], data, webhook_config.get("headers", {}))
        
        self.stats["webhook_calls"] += len(self.webhooks)
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert to webhooks."""
        await self.send_sensor_data(alert)
    
    async def _send_webhook(self, url: str, data: Dict[str, Any], headers: Dict[str, str]):
        """Send webhook request."""
        try:
            if HTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data, headers=headers, timeout=30) as response:
                        if response.status >= 400:
                            logger.warning(f"Webhook failed: {response.status}")
            else:
                logger.debug(f"Webhook simulated: {url}")
                
        except Exception as e:
            logger.warning(f"Webhook error: {e}")