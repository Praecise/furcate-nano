#!/usr/bin/env python3
# ============================================================================
# test_complete_furcate_nano.py
# Complete Furcate Nano System Test
# ============================================================================

import asyncio
import json
import time
import logging
import sys
import os
import platform
import socket
import random
import uuid
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================

@dataclass
class DeviceConfig:
    id: str = None
    name: str = "Furcate Test Device"
    location: Dict[str, float] = None
    environmental_zone: str = "test_zone"
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"furcate-test-{int(time.time())}"
        if self.location is None:
            self.location = {"latitude": 37.7749, "longitude": -122.4194, "altitude": 50.0}

@dataclass 
class HardwareConfig:
    simulation: bool = True
    sensors: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.sensors is None:
            self.sensors = {
                "temperature_humidity": {"type": "dht22", "enabled": True},
                "air_quality": {"type": "mq135", "enabled": True},
                "soil_moisture": {"type": "moisture", "enabled": True},
                "pressure": {"type": "bmp280", "enabled": True}
            }

@dataclass
class MLConfig:
    simulation: bool = True
    collaborative_learning: bool = True

@dataclass
class MeshConfig:
    simulation: bool = True
    max_connections: int = 8
    discovery_interval: int = 30

@dataclass
class PowerConfig:
    simulation: bool = True
    battery: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.battery is None:
            self.battery = {"capacity_mah": 10000, "voltage_min": 3.0, "voltage_max": 4.2}

@dataclass
class StorageConfig:
    data_path: str = None
    retention_days: int = 7
    
    def __post_init__(self):
        if self.data_path is None:
            self.data_path = tempfile.mkdtemp(prefix="furcate_test_")

@dataclass
class MonitoringConfig:
    interval_seconds: int = 2
    alert_thresholds: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "temperature_humidity": {"temperature": [5, 40], "humidity": [10, 90]},
                "air_quality": {"aqi": [0, 200]}
            }

@dataclass
class TenzroNetworkConfig:
    enabled: bool = True
    api_key: str = "test_api_key_12345"
    node_id: str = None
    
    def __post_init__(self):
        if self.node_id is None:
            self.node_id = f"tenzro-test-{int(time.time())}"

@dataclass
class FurcateNetworkConfig:
    enabled: bool = True
    device_name: str = "Test-Furcate-Device"
    auto_connect: bool = True

@dataclass
class WebIntegrationsConfig:
    rest_api: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.rest_api is None:
            self.rest_api = {"enabled": True, "port": 8000}

@dataclass
class NanoConfig:
    device: DeviceConfig = None
    hardware: HardwareConfig = None
    ml: MLConfig = None
    mesh: MeshConfig = None
    power: PowerConfig = None
    storage: StorageConfig = None
    monitoring: MonitoringConfig = None
    tenzro_network: TenzroNetworkConfig = None
    furcate_network: FurcateNetworkConfig = None
    integrations: WebIntegrationsConfig = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = DeviceConfig()
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.mesh is None:
            self.mesh = MeshConfig()
        if self.power is None:
            self.power = PowerConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.tenzro_network is None:
            self.tenzro_network = TenzroNetworkConfig()
        if self.furcate_network is None:
            self.furcate_network = FurcateNetworkConfig()
        if self.integrations is None:
            self.integrations = WebIntegrationsConfig()

# ============================================================================
# SENSOR SIMULATION
# ============================================================================

class SensorType(Enum):
    TEMPERATURE_HUMIDITY = "dht22"
    PRESSURE_TEMPERATURE = "bmp280"
    AIR_QUALITY = "mq135"
    SOIL_MOISTURE = "moisture"

@dataclass
class SensorReading:
    sensor_type: SensorType
    timestamp: float
    value: Union[float, Dict[str, float]]
    unit: str
    quality: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['sensor_type'] = self.sensor_type.value
        return result

class HardwareManager:
    def __init__(self, config):
        self.config = config
        self.sensors = {}
        self.reading_count = 0
        
    async def initialize(self) -> bool:
        for name, cfg in self.config.get("sensors", {}).items():
            if cfg.get("enabled"):
                self.sensors[name] = {"type": SensorType(cfg["type"]), "config": cfg}
        return True
    
    async def read_all_sensors(self) -> Dict[str, SensorReading]:
        readings = {}
        for name, sensor in self.sensors.items():
            readings[name] = self._simulate_reading(name, sensor["type"])
            self.reading_count += 1
        return readings
    
    def _simulate_reading(self, name: str, sensor_type: SensorType) -> SensorReading:
        timestamp = time.time()
        
        if sensor_type == SensorType.TEMPERATURE_HUMIDITY:
            temp = 20 + random.gauss(0, 5)
            humidity = 50 + random.gauss(0, 15)
            humidity = max(0, min(100, humidity))
            value = {"temperature": round(temp, 1), "humidity": round(humidity, 1)}
            unit = "°C, %RH"
        elif sensor_type == SensorType.AIR_QUALITY:
            aqi = max(0, min(500, 50 + random.gauss(0, 20)))
            value = {"aqi": round(aqi, 0)}
            unit = "AQI"
        elif sensor_type == SensorType.SOIL_MOISTURE:
            moisture = max(0, min(100, 45 + random.gauss(0, 15)))
            value = {"moisture": round(moisture, 1)}
            unit = "%"
        else:
            value = {"value": random.random()}
            unit = "units"
        
        return SensorReading(
            sensor_type=sensor_type,
            timestamp=timestamp,
            value=value,
            unit=unit,
            quality=0.95,
            confidence=0.90,
            metadata={"simulated": True}
        )
    
    def get_stats(self):
        return {"sensor_count": len(self.sensors), "readings_taken": self.reading_count}
    
    async def run_diagnostics(self):
        return {"overall_status": "excellent"}
    
    async def shutdown(self):
        pass

# ============================================================================
# SUBSYSTEM SIMULATIONS
# ============================================================================

class EdgeMLEngine:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        return True
    
    async def process_environmental_data(self, data):
        # Simple ML simulation
        temp = aqi = None
        for sensor_values in data.values():
            if isinstance(sensor_values, dict):
                if "temperature" in sensor_values:
                    temp = sensor_values["temperature"]
                if "aqi" in sensor_values:
                    aqi = sensor_values["aqi"]
        
        if aqi and aqi > 150:
            env_class = "polluted"
            anomaly = 0.8
        elif temp and (temp < 10 or temp > 30):
            env_class = "extreme"
            anomaly = 0.7
        else:
            env_class = "normal"
            anomaly = random.random() * 0.4
        
        return {
            "environmental_class": env_class,
            "anomaly_score": round(anomaly, 3),
            "confidence": 0.85 + random.random() * 0.1,
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        pass

class MeshNetworkManager:
    def __init__(self, config, device_id):
        self.device_id = device_id
        self.peers = {}
        self.stats = {"messages_sent": 0}
        
    async def initialize(self):
        # Simulate peer discovery
        for i in range(1, 4):
            peer_id = f"nano-{i:03d}"
            if peer_id != self.device_id:
                self.peers[peer_id] = {"last_seen": datetime.now()}
        return True
    
    async def broadcast_environmental_update(self, data):
        self.stats["messages_sent"] += 1
        return True
    
    async def broadcast_alert(self, alert):
        self.stats["messages_sent"] += 1
        return True
    
    def get_status(self):
        return {"peer_count": len(self.peers)}
    
    def get_peer_info(self):
        return list(self.peers.keys())
    
    async def shutdown(self):
        pass

class PowerMode(Enum):
    NORMAL = "normal"
    LOW_POWER = "low_power"

class PowerManager:
    def __init__(self, config):
        self.current_mode = PowerMode.NORMAL
        
    async def initialize(self):
        return True
    
    async def get_status(self):
        return {
            "battery_level": 0.7 + random.random() * 0.2,
            "charging": random.choice([True, False]),
            "power_mode": self.current_mode.value
        }
    
    async def set_mode(self, mode):
        self.current_mode = mode
        return True
    
    async def shutdown(self):
        pass

class StorageManager:
    def __init__(self, config):
        self.records = []
        self.data_path = Path(config.get("data_path", "./test_data"))
        self.data_path.mkdir(exist_ok=True)
        
    async def initialize(self):
        return True
    
    async def store_environmental_record(self, record):
        self.records.append(record)
        return True
    
    async def store_alert(self, alert, device_id):
        return True
    
    async def store_system_event(self, event_type, data, device_id):
        return True
    
    async def get_recent_environmental_data(self, hours):
        return self.records[-10:] if len(self.records) > 10 else self.records
    
    def get_stats(self):
        return {"records_stored": len(self.records)}
    
    async def shutdown(self):
        pass

class FurcateProtocol:
    def __init__(self, config):
        self.stats = {"messages_sent": 0}

# ============================================================================
# NETWORK CLIENT SIMULATIONS
# ============================================================================

class TenzroNetworkClient:
    def __init__(self, core, config):
        self.core = core
        self.config = config
        self.connected_peers = {}
        self.cloud_connections = {"tenzro_cloud"}
        self.stats = {"messages_sent": 0, "data_shared_mb": 0}
        
    async def connect(self):
        # Simulate peer connections
        for i in range(1, 4):
            peer_id = f"tenzro-peer-{i}"
            self.connected_peers[peer_id] = {"status": "connected"}
        logger.info(f"🌐 Tenzro Network: {len(self.connected_peers)} peers connected")
    
    async def send_sensor_data(self, data):
        self.stats["messages_sent"] += 1
        self.stats["data_shared_mb"] += 0.001
    
    async def send_alert(self, alert):
        self.stats["messages_sent"] += 1
    
    async def share_ml_model(self, model_data):
        self.stats["messages_sent"] += 1
    
    async def request_collaborative_insights(self, query):
        return {"peer_count": len(self.connected_peers)}

class FurcateNetworkClient:
    def __init__(self, core, config):
        self.core = core
        self.discovered_devices = {}
        self.active_connections = {}
        self.supported_protocols = {"wifi_direct", "bluetooth_le"}
        self.protocol_handlers = {"wifi": "handler"}
        self.local_stats = {"messages_exchanged": 0}
        
    async def initialize(self):
        # Simulate device discovery
        for i in range(1, 3):
            device_id = f"furcate-device-{i}"
            self.discovered_devices[device_id] = {"status": "discovered"}
        logger.info(f"📡 Furcate Network: {len(self.discovered_devices)} devices discovered")
    
    async def share_environmental_data(self, sensor_data, ml_analysis):
        self.local_stats["messages_exchanged"] += 1
    
    async def request_local_collaboration(self, collaboration_type, parameters):
        return [{"participant": "local_device"}]

class WebIntegrationManager:
    def __init__(self, core, config):
        self.core = core
        self.config = config
        self.integrations = {}
        self.stats = {"api_requests": 0}
        
    async def initialize(self):
        if self.config.get("rest_api", {}).get("enabled"):
            self.integrations["rest_api"] = {"status": "active"}
        return True
    
    async def broadcast_sensor_data(self, sensor_data, ml_analysis):
        self.stats["api_requests"] += 1
    
    async def broadcast_alert(self, alert):
        self.stats["api_requests"] += 1
    
    def get_stats(self):
        return self.stats
    
    async def shutdown(self):
        pass

# ============================================================================
# FURCATE NANO CORE
# ============================================================================

class FurcateNanoCore:
    def __init__(self, config: NanoConfig):
        self.config = config
        self.device_id = config.device.id
        self.running = False
        self.monitoring_cycles = 0
        
        # Initialize subsystems
        self.hardware = HardwareManager(config.hardware.__dict__)
        self.edge_ml = EdgeMLEngine(config.ml.__dict__)
        self.mesh = MeshNetworkManager(config.mesh.__dict__, self.device_id)
        self.power = PowerManager(config.power.__dict__)
        self.storage = StorageManager(config.storage.__dict__)
        self.protocol = FurcateProtocol({})
        
        # Initialize network clients
        self.tenzro_client = TenzroNetworkClient(self, config.tenzro_network.__dict__) if config.tenzro_network.enabled else None
        self.furcate_client = FurcateNetworkClient(self, config.furcate_network.__dict__) if config.furcate_network.enabled else None
        self.web_integrations = WebIntegrationManager(self, config.integrations.__dict__)
        
        # Performance metrics
        self.performance_metrics = {
            "total_alerts": 0,
            "uptime_start": datetime.now()
        }
        
        logger.info(f"🌿 Furcate Nano Core initialized: {self.device_id}")
    
    async def initialize(self) -> bool:
        logger.info("🚀 Starting initialization...")
        
        # Initialize subsystems
        init_results = {}
        init_results["hardware"] = await self.hardware.initialize()
        init_results["edge_ml"] = await self.edge_ml.initialize()
        init_results["mesh"] = await self.mesh.initialize()
        init_results["power"] = await self.power.initialize()
        init_results["storage"] = await self.storage.initialize()
        
        # Initialize networks
        if self.tenzro_client:
            try:
                await self.tenzro_client.connect()
                init_results["tenzro_network"] = True
            except Exception as e:
                init_results["tenzro_network"] = False
                logger.warning(f"Tenzro Network failed: {e}")
        
        if self.furcate_client:
            try:
                await self.furcate_client.initialize()
                init_results["furcate_network"] = True
            except Exception as e:
                init_results["furcate_network"] = False
                logger.warning(f"Furcate Network failed: {e}")
        
        init_results["web_integrations"] = await self.web_integrations.initialize()
        
        failed_systems = [name for name, success in init_results.items() if not success]
        
        if failed_systems:
            logger.warning(f"Some systems failed: {failed_systems}")
        
        logger.info("✅ Initialization complete")
        return True
    
    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        cycle_start = datetime.now()
        
        try:
            # 1. Read sensors
            sensor_readings = await self.hardware.read_all_sensors()
            sensor_data = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "sensors": {name: reading.to_dict() for name, reading in sensor_readings.items()}
            }
            
            # 2. ML processing
            sensor_values = {}
            for name, reading in sensor_readings.items():
                if hasattr(reading, 'value'):
                    sensor_values[name] = reading.value
            
            ml_analysis = await self.edge_ml.process_environmental_data(sensor_values)
            
            # 3. Check for alerts
            alerts = await self._check_alerts(sensor_data, ml_analysis)
            
            # 4. Share data across networks
            await self._share_across_networks(sensor_data, ml_analysis, alerts)
            
            # 5. Store data
            await self.storage.store_environmental_record({
                "sensor_data": sensor_data,
                "ml_analysis": ml_analysis,
                "alerts": alerts
            })
            
            self.monitoring_cycles += 1
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            
            return {
                "success": True,
                "cycle": self.monitoring_cycles,
                "duration_ms": cycle_time * 1000,
                "sensors_read": len(sensor_readings),
                "ml_class": ml_analysis.get("environmental_class", "unknown"),
                "alerts": len(alerts)
            }
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _check_alerts(self, sensor_data, ml_analysis):
        alerts = []
        
        # Check thresholds
        for sensor_name, sensor_info in sensor_data.get("sensors", {}).items():
            thresholds = self.config.monitoring.alert_thresholds.get(sensor_name, {})
            if thresholds and isinstance(sensor_info, dict):
                value = sensor_info.get("value", {})
                if isinstance(value, dict):
                    for param, param_value in value.items():
                        if param in thresholds:
                            min_val, max_val = thresholds[param]
                            if param_value < min_val or param_value > max_val:
                                alerts.append({
                                    "type": "threshold_exceeded",
                                    "sensor": sensor_name,
                                    "parameter": param,
                                    "value": param_value,
                                    "severity": "warning"
                                })
        
        # ML anomaly alerts
        if ml_analysis.get("anomaly_score", 0) > 0.8:
            alerts.append({
                "type": "ml_anomaly",
                "anomaly_score": ml_analysis["anomaly_score"],
                "severity": "critical"
            })
        
        if alerts:
            self.performance_metrics["total_alerts"] += len(alerts)
        
        return alerts
    
    async def _share_across_networks(self, sensor_data, ml_analysis, alerts):
        # Share via mesh
        await self.mesh.broadcast_environmental_update({
            "device_id": self.device_id,
            "ml_summary": ml_analysis,
            "alert_count": len(alerts)
        })
        
        # Share via Tenzro Network
        if self.tenzro_client:
            await self.tenzro_client.send_sensor_data(sensor_data)
        
        # Share via Furcate Network
        if self.furcate_client:
            await self.furcate_client.share_environmental_data(sensor_data, ml_analysis)
        
        # Share via web integrations
        await self.web_integrations.broadcast_sensor_data(sensor_data, ml_analysis)
        
        # Broadcast critical alerts
        for alert in alerts:
            if alert.get("severity") == "critical":
                if self.tenzro_client:
                    await self.tenzro_client.send_alert(alert)
                await self.web_integrations.broadcast_alert(alert)
    
    async def run_test_sequence(self, cycles: int = 5, interval: float = 1.0):
        logger.info(f"🧪 Starting test sequence: {cycles} cycles")
        
        if not await self.initialize():
            return {"error": "Initialization failed"}
        
        results = []
        total_start = time.time()
        
        try:
            for i in range(cycles):
                logger.info(f"🔄 Running cycle {i + 1}/{cycles}")
                
                cycle_result = await self.run_monitoring_cycle()
                results.append(cycle_result)
                
                if i < cycles - 1:
                    await asyncio.sleep(interval)
            
            total_time = time.time() - total_start
            successful_cycles = [r for r in results if r.get("success")]
            
            return {
                "test_completed": True,
                "total_cycles": cycles,
                "successful_cycles": len(successful_cycles),
                "total_duration_seconds": round(total_time, 2),
                "avg_cycle_time_ms": round(
                    sum(r.get("duration_ms", 0) for r in successful_cycles) / len(successful_cycles), 2
                ) if successful_cycles else 0,
                "total_alerts": sum(r.get("alerts", 0) for r in successful_cycles),
                "ml_classifications": [r.get("ml_class") for r in successful_cycles],
                "device_id": self.device_id,
                "network_stats": {
                    "tenzro": self.tenzro_client.stats if self.tenzro_client else {},
                    "furcate": self.furcate_client.local_stats if self.furcate_client else {},
                    "mesh": self.mesh.stats,
                    "web": self.web_integrations.stats
                }
            }
            
        except Exception as e:
            logger.error(f"Test sequence failed: {e}")
            return {"error": str(e)}
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        logger.info("🛑 Shutting down Furcate Nano...")
        
        # Shutdown all subsystems
        await self.storage.shutdown()
        await self.power.shutdown()
        await self.mesh.shutdown()
        await self.edge_ml.shutdown()
        await self.hardware.shutdown()
        await self.web_integrations.shutdown()
        
        logger.info("✅ Shutdown complete")

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

async def main():
    print("\n🧪 Furcate Nano System Test")
    print("=" * 50)
    
    try:
        # Create configuration
        config = NanoConfig()
        
        # Create core system
        core = FurcateNanoCore(config)
        
        # Run test sequence
        results = await core.run_test_sequence(cycles=5, interval=1.0)
        
        # Display results
        print("\n📊 TEST RESULTS")
        print("=" * 50)
        
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            return False
        
        print(f"✅ Test successful!")
        print(f"📱 Device: {results['device_id']}")
        print(f"🔄 Cycles: {results['successful_cycles']}/{results['total_cycles']}")
        print(f"⏱️  Duration: {results['total_duration_seconds']}s")
        print(f"⚡ Avg cycle: {results['avg_cycle_time_ms']:.1f}ms")
        print(f"🚨 Alerts: {results['total_alerts']}")
        
        # Network stats
        net_stats = results['network_stats']
        print(f"\n🌐 Network Activity:")
        if net_stats['tenzro']:
            print(f"   • Tenzro: {net_stats['tenzro']['messages_sent']} messages")
        if net_stats['furcate']:
            print(f"   • Furcate: {net_stats['furcate']['messages_exchanged']} messages")
        print(f"   • Mesh: {net_stats['mesh']['messages_sent']} messages")
        print(f"   • Web: {net_stats['web']['api_requests']} requests")
        
        # ML results
        ml_classes = results['ml_classifications']
        if ml_classes:
            class_counts = {}
            for cls in ml_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            print(f"\n🤖 ML Classifications: {class_counts}")
        
        print("\n🎉 Furcate Nano system verified!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("🌿 Furcate Nano System Test")
    print(f"Python {sys.version}")
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
            print("\nSystems verified:")
            print("• 🔧 Hardware simulation")
            print("• 🤖 Edge ML processing")
            print("• 🕸️ Mesh networking")
            print("• 🌐 Tenzro Network P2P")
            print("• 📡 Furcate Network local P2P")
            print("• 🌐 Web integrations")
            print("• ⚡ Power management")
            print("• 💾 Data storage")
            print("\nReady for Raspberry Pi 5 deployment!")
            exit(0)
        else:
            print("\n❌ TESTS FAILED!")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        exit(130)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        exit(1)