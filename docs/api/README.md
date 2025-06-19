# Furcate Nano API Reference

Complete API documentation for the Furcate Nano environmental edge computing framework.

## Table of Contents

- [Overview](#overview)
- [REST API](#rest-api)
- [WebSocket API](#websocket-api)
- [MQTT Integration](#mqtt-integration)
- [Core Methods](#core-methods)
- [Health Monitoring](#health-monitoring)
- [Network Coordination](#network-coordination)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

The Furcate Nano API provides programmatic access to environmental monitoring data, device management, machine learning capabilities, and mesh networking features through FastAPI-based REST endpoints, WebSocket connections, and MQTT integration.

### Base URL

- **Local Device**: `http://localhost:8000` (default)
- **Custom Host/Port**: Configurable via device configuration

### Content Types

All API requests and responses use JSON format:
- Request: `Content-Type: application/json`
- Response: `Content-Type: application/json`

## REST API

The REST API is implemented using FastAPI and provides the following endpoints:

### Device Information

#### Get API Root
```http
GET /
```

**Response:**
```json
{
  "message": "Furcate Nano API",
  "device_id": "nano-device-001"
}
```

#### Get Device Status
```http
GET /status
```

Returns comprehensive device status including hardware, network, and operational metrics.

**Response:**
```json
{
  "device_id": "nano-device-001",
  "running": true,
  "cycles": 1547,
  "last_reading": "2025-06-19T14:30:00Z",
  "uptime_seconds": 3600.5,
  "health_status": {
    "overall": "excellent",
    "subsystems": {
      "hardware": {"healthy": true, "critical": false},
      "ml": {"healthy": true, "critical": false},
      "mesh": {"healthy": true, "critical": false},
      "power": {"healthy": true, "critical": false},
      "storage": {"healthy": true, "critical": false}
    },
    "networks": {
      "tenzro": {"healthy": true, "peers": 2, "clouds": 1},
      "furcate": {"healthy": true, "devices": 3, "protocols": 4},
      "mesh": {"healthy": true, "peers": 2},
      "web": {"healthy": true, "integrations": 1}
    },
    "failed_systems": [],
    "critical_systems": []
  },
  "performance_metrics": {
    "avg_cycle_time_ms": 250.5,
    "total_alerts": 12,
    "network_stats": {
      "tenzro": {"messages_sent": 156, "success_rate": 0.98},
      "furcate": {"connections": 3, "data_shared": true},
      "web": {"api_requests": 89, "mqtt_messages": 234},
      "mesh": {"peers": 2, "status": "active"}
    }
  },
  "network_coordination": {
    "data_sharing_enabled": true,
    "collaborative_learning": true,
    "emergency_broadcasting": true,
    "local_sync_enabled": true,
    "cloud_backup_enabled": true
  },
  "networks": {
    "tenzro": {
      "enabled": true,
      "peers": 2,
      "cloud_connections": 1,
      "stats": {"messages_sent": 156, "messages_failed": 3}
    },
    "furcate": {
      "enabled": true,
      "devices": 3,
      "protocols": 4,
      "stats": {"active_connections": 2, "data_synced": true}
    },
    "mesh": {
      "enabled": true,
      "peers": 2,
      "stats": {"status": "active", "network_health": "good"}
    },
    "web": {
      "enabled": true,
      "integrations": 1,
      "stats": {"api_requests": 89, "mqtt_messages": 234}
    }
  }
}
```

### Sensor Data

#### Get Current Sensor Readings
```http
GET /sensors/current
```

Returns the latest readings from all active sensors.

**Response:**
```json
{
  "timestamp": "2025-06-19T14:30:00Z",
  "device_id": "nano-device-001",
  "sensors": {
    "temperature": {
      "value": 23.5,
      "unit": "°C",
      "quality": 0.95,
      "timestamp": "2025-06-19T14:30:00Z"
    },
    "humidity": {
      "value": 65.2,
      "unit": "%",
      "quality": 0.98,
      "timestamp": "2025-06-19T14:30:00Z"
    },
    "air_pressure": {
      "value": 1013.25,
      "unit": "hPa",
      "quality": 0.92,
      "timestamp": "2025-06-19T14:30:00Z"
    }
  },
  "ml_analysis": {
    "anomaly_score": 0.15,
    "predictions": {
      "air_quality_forecast": "good",
      "comfort_index": 0.85
    },
    "alerts": []
  }
}
```

#### Get Historical Sensor Data
```http
GET /sensors/history?hours={hours}&sensor={sensor_name}
```

**Parameters:**
- `hours` (optional): Number of hours to retrieve (default: 24, max: 168)
- `sensor` (optional): Specific sensor name to filter by

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2025-06-19T13:30:00Z",
      "sensors": {
        "temperature": {"value": 22.8, "quality": 0.96},
        "humidity": {"value": 64.1, "quality": 0.97}
      }
    }
  ],
  "total_records": 24,
  "time_range": {
    "start": "2025-06-18T14:30:00Z",
    "end": "2025-06-19T14:30:00Z"
  }
}
```

### Alerts and Notifications

#### Get Recent Alerts
```http
GET /alerts?hours={hours}&priority={priority}
```

**Parameters:**
- `hours` (optional): Number of hours to retrieve (default: 24)
- `priority` (optional): Filter by priority (low, medium, high, critical)

**Response:**
```json
{
  "data": [
    {
      "id": "alert-001",
      "timestamp": "2025-06-19T14:25:00Z",
      "priority": "medium",
      "type": "environmental",
      "message": "Temperature trending upward",
      "details": {
        "sensor": "temperature",
        "current_value": 23.5,
        "threshold": 25.0,
        "trend": "increasing"
      },
      "resolved": false
    }
  ],
  "total_alerts": 1,
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 1,
    "low": 0
  }
}
```

### Machine Learning

#### Get ML Analysis
```http
GET /ml/analysis
```

Returns current machine learning analysis and predictions.

**Response:**
```json
{
  "timestamp": "2025-06-19T14:30:00Z",
  "environmental_analysis": {
    "anomaly_score": 0.15,
    "comfort_index": 0.85,
    "air_quality_forecast": "good",
    "trend_analysis": {
      "temperature": "stable",
      "humidity": "decreasing",
      "pressure": "stable"
    }
  },
  "predictions": {
    "next_hour": {
      "temperature": {"value": 23.8, "confidence": 0.89},
      "humidity": {"value": 63.5, "confidence": 0.92}
    },
    "risk_assessment": {
      "comfort_risk": "low",
      "equipment_risk": "low",
      "health_risk": "low"
    }
  },
  "model_info": {
    "version": "1.2.3",
    "last_trained": "2025-06-19T10:00:00Z",
    "confidence": 0.91
  }
}
```

### Network Operations

#### Get Network Status
```http
GET /network/status
```

Returns status of all network connections and protocols.

**Response:**
```json
{
  "networks": {
    "tenzro": {
      "status": "connected",
      "peers": 2,
      "cloud_connections": 1,
      "last_sync": "2025-06-19T14:29:00Z",
      "data_shared": true
    },
    "furcate": {
      "status": "active",
      "discovered_devices": 3,
      "active_connections": 2,
      "protocols": ["wifi_direct", "bluetooth", "lora", "cellular"],
      "last_discovery": "2025-06-19T14:28:00Z"
    },
    "mesh": {
      "status": "active",
      "peers": 2,
      "network_health": "good",
      "mesh_quality": 0.87
    },
    "web": {
      "status": "active",
      "integrations": 1,
      "api_requests_total": 89,
      "mqtt_messages_total": 234
    }
  },
  "coordination": {
    "data_sharing_enabled": true,
    "collaborative_learning": true,
    "emergency_broadcasting": true,
    "sync_status": {
      "last_sync": "2025-06-19T14:25:00Z",
      "records_synced": 5,
      "networks_synced": ["tenzro", "furcate"]
    }
  }
}
```

#### Trigger Network Coordination
```http
POST /network/coordinate
```

Manually triggers network coordination and optimization.

**Response:**
```json
{
  "success": true,
  "coordination_result": {
    "network_health": {
      "mesh": {"status": "good", "peers": 2},
      "tenzro": {"status": "good", "peers": 2, "clouds": 1},
      "furcate": {"status": "good", "devices": 3, "protocols": 4},
      "web": {"status": "good", "integrations": 1}
    },
    "optimization_applied": [],
    "sync_status": {
      "records_synced": 5,
      "sync_timestamp": "2025-06-19T14:30:00Z"
    }
  }
}
```

### System Diagnostics

#### Run Comprehensive Diagnostics
```http
POST /diagnostics/comprehensive
```

Runs full system diagnostics across all subsystems and networks.

**Response:**
```json
{
  "overall_health": "excellent",
  "diagnostics": {
    "hardware": {
      "status": "healthy",
      "sensors_operational": 3,
      "simulation_mode": false
    },
    "ml": {
      "status": "healthy",
      "model_loaded": true,
      "processing_time_ms": 45.2
    },
    "power": {
      "status": "healthy",
      "battery_level": 85.2,
      "charging": true
    },
    "storage": {
      "status": "healthy",
      "usage_percent": 23.5,
      "available_space_mb": 7650
    },
    "networks": {
      "all_operational": true,
      "total_connections": 7,
      "data_flow": "normal"
    }
  },
  "recommendations": [],
  "timestamp": "2025-06-19T14:30:00Z"
}
```

## WebSocket API

Real-time data streaming via WebSocket connections.

### Connection
```
ws://localhost:8765/ws
```

### Real-time Sensor Data Stream

Once connected, the WebSocket automatically sends sensor data every 5 seconds:

```json
{
  "timestamp": "2025-06-19T14:30:00Z",
  "device_id": "nano-device-001",
  "sensors": {
    "temperature": {
      "value": 23.5,
      "unit": "°C",
      "quality": 0.95,
      "timestamp": "2025-06-19T14:30:00Z"
    },
    "humidity": {
      "value": 65.2,
      "unit": "%",
      "quality": 0.98,
      "timestamp": "2025-06-19T14:30:00Z"
    }
  },
  "ml_analysis": {
    "anomaly_score": 0.15,
    "predictions": {
      "air_quality_forecast": "good",
      "comfort_index": 0.85
    }
  }
}
```

## MQTT Integration

MQTT topics and message formats for IoT platform integration.

### Topic Structure
```
{base_topic}/{device_id}/{data_type}/{sub_type}
```

### Default Topics

#### Sensor Data
**Topic:** `furcate/{device_id}/sensors/data`
**QoS:** 0
**Retain:** false

```json
{
  "timestamp": "2025-06-19T14:30:00Z",
  "device_id": "nano-device-001",
  "sensors": {
    "temperature": {"value": 23.5, "unit": "°C"},
    "humidity": {"value": 65.2, "unit": "%"}
  },
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060
  }
}
```

#### Alerts
**Topic:** `furcate/{device_id}/alerts/{priority}`
**QoS:** 2
**Retain:** false

```json
{
  "timestamp": "2025-06-19T14:25:00Z",
  "device_id": "nano-device-001",
  "alert_id": "alert-001",
  "priority": "medium",
  "type": "environmental",
  "message": "Temperature trending upward",
  "resolved": false
}
```

#### Device Status
**Topic:** `furcate/{device_id}/status`
**QoS:** 1
**Retain:** true

```json
{
  "timestamp": "2025-06-19T14:30:00Z",
  "device_id": "nano-device-001",
  "status": "online",
  "uptime_seconds": 3600,
  "battery_level": 85.2,
  "network_health": "excellent"
}
```

## Health Monitoring

The system continuously monitors health across all subsystems:

### Health Check Categories

1. **Hardware**: Sensor functionality, power status
2. **ML**: Model performance, processing times
3. **Networks**: Connectivity, data flow, peer status
4. **Storage**: Disk usage, database health
5. **Power**: Battery levels, charging status

### Health Status Levels

- **excellent**: All systems operational
- **warning**: Some non-critical issues
- **degraded**: Multiple system issues
- **critical**: Critical system failures

### Automatic Health Monitoring

The system performs automatic health checks every 5 minutes, updating the overall health status and triggering optimizations when needed.

## Network Coordination

Advanced network coordination manages multiple network types:

### Supported Networks

1. **Tenzro Network**: Peer-to-peer and cloud connectivity
2. **Furcate Network**: Multi-protocol local networking
3. **Mesh Network**: Local mesh networking
4. **Web Integrations**: API, MQTT, WebSocket, Webhooks

### Coordination Features

- **Automatic Optimization**: Detects degraded networks and applies fixes
- **Data Synchronization**: Syncs data across connected networks
- **Load Balancing**: Distributes traffic across available networks
- **Failover**: Automatically switches to backup networks

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Endpoint or resource not found
- `500`: Internal Server Error - System error
- `503`: Service Unavailable - System not ready

### Error Response Format

```json
{
  "error": {
    "code": "SENSOR_READ_FAILED",
    "message": "Unable to read sensor data",
    "details": {
      "sensor": "temperature",
      "reason": "Hardware connection lost"
    },
    "timestamp": "2025-06-19T14:30:00Z"
  }
}
```

## Examples

### Python REST API Client

```python
import asyncio
import aiohttp
import json

async def fetch_sensor_data():
    async with aiohttp.ClientSession() as session:
        # Get current sensor readings
        async with session.get('http://localhost:8000/sensors/current') as resp:
            data = await resp.json()
            print("Current readings:", data['sensors'])
        
        # Get device status with full network information
        async with session.get('http://localhost:8000/status') as resp:
            status = await resp.json()
            print("Device status:", status['health_status']['overall'])
            print("Network health:", status['networks'])
        
        # Get recent alerts
        async with session.get('http://localhost:8000/alerts?hours=1') as resp:
            alerts = await resp.json()
            print("Recent alerts:", len(alerts['data']))
        
        # Trigger network coordination
        async with session.post('http://localhost:8000/network/coordinate') as resp:
            result = await resp.json()
            print("Coordination result:", result['success'])

asyncio.run(fetch_sensor_data())
```

### MQTT Subscriber

```python
import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    # Subscribe to all device topics
    client.subscribe("furcate/+/sensors/data")
    client.subscribe("furcate/+/alerts/+")
    client.subscribe("furcate/+/status")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = json.loads(msg.payload.decode())
    
    if "/sensors/data" in topic:
        device_id = payload['device_id']
        temperature = payload['sensors'].get('temperature', {}).get('value', 'N/A')
        print(f"Device {device_id}: Temperature = {temperature}°C")
    
    elif "/alerts/" in topic:
        print(f"Alert [{payload['priority']}]: {payload['message']}")
    
    elif "/status" in topic:
        print(f"Device {payload['device_id']}: {payload['status']}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 60)
client.loop_forever()
```

### WebSocket Client

```python
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8765/ws"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        try:
            while True:
                # Receive real-time sensor data
                data = await websocket.recv()
                sensor_data = json.loads(data)
                
                timestamp = sensor_data['timestamp']
                temperature = sensor_data['sensors']['temperature']['value']
                ml_score = sensor_data['ml_analysis']['anomaly_score']
                
                print(f"[{timestamp}] Temp: {temperature}°C, Anomaly: {ml_score}")
                
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")

asyncio.run(websocket_client())
```

This documentation reflects the actual implementation in the Furcate Nano codebase, including all network coordination features, health monitoring systems, and multi-protocol networking capabilities.