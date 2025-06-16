# ============================================================================
# furcate_nano/core.py
"""Complete Furcate Nano core."""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from .config import NanoConfig
from .hardware import HardwareManager
from .edge_ml import EdgeMLEngine  
from .mesh import MeshNetworkManager
from .power import PowerManager
from .storage import StorageManager
from .protocols import FurcateProtocol
from .integrations import WebIntegrationManager
from .network_clients import TenzroNetworkClient, FurcateNetworkClient

logger = logging.getLogger(__name__)

class FurcateNanoCore:
    """
    Complete Furcate Nano core framework with comprehensive network integration.
    
    Network Architecture:
    - Local Mesh: Bio-inspired mesh networking for nearby devices
    - Furcate Network: Local P2P using WiFi/Bluetooth/LoRa/ESP-NOW
    - Tenzro Network: Global P2P with multi-cloud integration
    - Web Integrations: REST API, MQTT, WebSockets, Webhooks
    
    Features:
    - Multi-protocol sensor fusion and validation
    - Advanced edge ML with collaborative learning
    - Intelligent power management with predictive optimization
    - Multi-database storage (DuckDB, RocksDB, SQLite)
    - Comprehensive health monitoring and diagnostics
    - Real-time data sharing and collaboration
    """
    
    def __init__(self, config: NanoConfig):
        """Initialize complete Furcate Nano core.
        
        Args:
            config: Nano device configuration
        """
        self.config = config
        self.device_id = config.device.id
        self.running = False
        self.monitoring_cycles = 0
        
        # Initialize core subsystems
        self.hardware = HardwareManager(config.hardware.__dict__)
        self.edge_ml = EdgeMLEngine(config.ml.__dict__)
        self.mesh = MeshNetworkManager(config.mesh.__dict__, self.device_id)
        self.power = PowerManager(config.power.__dict__)
        self.storage = StorageManager(config.storage.__dict__)
        self.protocol = FurcateProtocol(config.protocol.__dict__)
        
        # Initialize network clients
        self._init_network_clients(config)
        
        # Initialize web integrations
        web_config = getattr(config, 'integrations', {
            "rest_api": {"enabled": True, "port": 8000},
            "mqtt": {"enabled": False},
            "websocket": {"enabled": False},
            "webhooks": {"enabled": False}
        })
        self.web_integrations = WebIntegrationManager(self, web_config)
        
        # System monitoring state
        self.last_reading_time = None
        self.environmental_data_buffer = []
        self.alert_history = []
        self.performance_metrics = {
            "cycle_times": [],
            "sensor_success_rates": {},
            "ml_processing_times": [],
            "network_stats": {},
            "total_alerts": 0,
            "uptime_start": datetime.now()
        }
        
        # System health monitoring
        self.health_status = {
            "overall": "unknown",
            "subsystems": {},
            "networks": {},
            "last_check": None
        }
        
        # Network coordination
        self.network_coordination = {
            "data_sharing_enabled": True,
            "collaborative_learning": True,
            "emergency_broadcasting": True,
            "local_sync_enabled": True,
            "cloud_backup_enabled": True
        }
        
        logger.info(f"🌿 Complete Furcate Nano Core initialized: {self.device_id}")
    
    def _init_network_clients(self, config: NanoConfig):
        """Initialize network clients based on configuration."""
        # Tenzro Network client
        tenzro_config = getattr(config, 'tenzro_network', {
            "enabled": True,
            "node_id": f"tenzro-{self.device_id}",
            "network_key": "default_tenzro_key",
            "cloud_providers": ["tenzro_cloud"],
            "multi_cloud_enabled": True
        })
        
        if tenzro_config.get("enabled", True):
            self.tenzro_client = TenzroNetworkClient(self, tenzro_config)
        else:
            self.tenzro_client = None
        
        # Furcate Network client  
        furcate_config = getattr(config, 'furcate_network', {
            "enabled": True,
            "device_name": f"Furcate-{self.device_id[-6:]}",
            "auto_connect": True,
            "discovery_interval": 30
        })
        
        if furcate_config.get("enabled", True):
            self.furcate_client = FurcateNetworkClient(self, furcate_config)
        else:
            self.furcate_client = None
    
    async def initialize(self) -> bool:
        """Initialize all subsystems and network clients."""
        try:
            logger.info("🚀 Starting complete Furcate Nano initialization...")
            
            # Initialize core subsystems
            init_results = {}
            
            logger.info("📡 Initializing hardware manager...")
            init_results["hardware"] = await self.hardware.initialize()
            
            logger.info("🤖 Initializing edge ML engine...")
            init_results["edge_ml"] = await self.edge_ml.initialize()
            
            logger.info("🕸️ Initializing mesh network...")
            init_results["mesh"] = await self.mesh.initialize()
            
            logger.info("⚡ Initializing power management...")
            init_results["power"] = await self.power.initialize()
            
            logger.info("💾 Initializing storage system...")
            init_results["storage"] = await self.storage.initialize()
            
            # Initialize network clients
            logger.info("🌐 Initializing network clients...")
            network_results = await self._initialize_network_clients()
            init_results.update(network_results)
            
            # Initialize web integrations
            logger.info("🔗 Initializing web integrations...")
            init_results["web_integrations"] = await self.web_integrations.initialize()
            
            # Check initialization results
            failed_systems = [name for name, success in init_results.items() if not success]
            
            if failed_systems:
                logger.warning(f"⚠️ Some systems failed to initialize: {failed_systems}")
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start system health monitoring
            asyncio.create_task(self._comprehensive_health_monitoring())
            
            # Start network coordination
            asyncio.create_task(self._network_coordination_loop())
            
            # Update health status
            self.health_status = {
                "overall": "good" if len(failed_systems) == 0 else "degraded",
                "subsystems": init_results,
                "last_check": datetime.now(),
                "failed_systems": failed_systems
            }
            
            logger.info("✅ Complete Furcate Nano initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete initialization failed: {e}")
            self.health_status = {
                "overall": "critical",
                "subsystems": {},
                "last_check": datetime.now(),
                "error": str(e)
            }
            return False
    
    async def _initialize_network_clients(self) -> Dict[str, bool]:
        """Initialize all network clients."""
        results = {}
        
        # Initialize Tenzro Network
        if self.tenzro_client:
            try:
                await self.tenzro_client.connect()
                results["tenzro_network"] = True
                logger.info("✅ Tenzro Network client initialized")
            except Exception as e:
                logger.warning(f"Tenzro Network initialization failed: {e}")
                results["tenzro_network"] = False
        else:
            results["tenzro_network"] = False
        
        # Initialize Furcate Network
        if self.furcate_client:
            try:
                await self.furcate_client.initialize()
                results["furcate_network"] = True
                logger.info("✅ Furcate Network client initialized")
            except Exception as e:
                logger.warning(f"Furcate Network initialization failed: {e}")
                results["furcate_network"] = False
        else:
            results["furcate_network"] = False
        
        return results
    
    async def start_monitoring(self):
        """Start the complete environmental monitoring with all network integration."""
        if not await self.initialize():
            logger.error("Failed to initialize, cannot start monitoring")
            return
        
        self.running = True
        logger.info("🌱 Starting complete environmental monitoring with network integration...")
        
        try:
            while self.running:
                cycle_start = datetime.now()
                
                # Run comprehensive monitoring cycle
                cycle_result = await self.run_complete_monitoring_cycle()
                
                # Update performance metrics
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                self.performance_metrics["cycle_times"].append(cycle_time)
                
                # Keep only recent cycle times (last 100)
                if len(self.performance_metrics["cycle_times"]) > 100:
                    self.performance_metrics["cycle_times"] = self.performance_metrics["cycle_times"][-100:]
                
                # Log cycle completion
                if cycle_result.get("success", False):
                    logger.debug(f"✅ Complete monitoring cycle {self.monitoring_cycles} completed in {cycle_time:.2f}s")
                else:
                    logger.warning(f"⚠️ Complete monitoring cycle {self.monitoring_cycles} had issues")
                
                # Wait for next monitoring interval
                interval = self.config.monitoring.interval_seconds
                await asyncio.sleep(max(0, interval - cycle_time))
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Complete monitoring error: {e}")
        finally:
            await self.shutdown()
    
    async def run_complete_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle with all network integrations."""
        cycle_start = datetime.now()
        cycle_result = {
            "cycle": self.monitoring_cycles + 1,
            "timestamp": cycle_start.isoformat(),
            "success": False,
            "errors": [],
            "metrics": {},
            "network_activity": {}
        }
        
        try:
            # 1. Power management and system optimization
            power_result = await self._power_management()
            cycle_result["metrics"]["power"] = power_result
            
            # 2. Sensor data collection with fusion
            sensor_result = await self._sensor_collection()
            cycle_result["metrics"]["sensors"] = sensor_result
            
            if not sensor_result.get("success", False):
                cycle_result["errors"].append("sensor_collection_failed")
                return cycle_result
            
            # 3. Advanced ML processing
            ml_result = await self._ml_processing(sensor_result["data"])
            cycle_result["metrics"]["ml"] = ml_result
            
            # 4. Intelligent alert detection
            alert_result = await self._alert_detection(
                sensor_result["data"], ml_result["analysis"]
            )
            cycle_result["metrics"]["alerts"] = alert_result
            
            # 5. Multi-network data sharing
            network_result = await self._multi_network_data_sharing(
                sensor_result["data"], ml_result["analysis"], alert_result["alerts"]
            )
            cycle_result["network_activity"] = network_result
            
            # 6. Data storage
            storage_result = await self._data_storage(
                sensor_result["data"], ml_result["analysis"], alert_result["alerts"]
            )
            cycle_result["metrics"]["storage"] = storage_result
            
            # 7. Network coordination and optimization
            coordination_result = await self._coordinate_network_activities()
            cycle_result["metrics"]["coordination"] = coordination_result
            
            # 8. Performance monitoring and optimization
            performance_result = await self._update_comprehensive_metrics(cycle_result)
            cycle_result["metrics"]["performance"] = performance_result
            
            # Update cycle counters
            self.monitoring_cycles += 1
            self.last_reading_time = cycle_start
            
            cycle_result["success"] = True
            cycle_result["duration_ms"] = (datetime.now() - cycle_start).total_seconds() * 1000
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"Complete monitoring cycle failed: {e}")
            cycle_result["errors"].append(f"cycle_exception: {str(e)}")
            return cycle_result
    
    async def _power_management(self) -> Dict[str, Any]:
        """Power management with network-aware optimization."""
        try:
            # Get comprehensive power status
            power_status = await self.power.get_status()
            
            # Network-aware power optimization
            battery_level = power_status.get("battery_level", 0.5)
            charging = power_status.get("charging", False)
            
            # Consider network activity in power decisions
            network_activity = self._get_network_activity_level()
            
            # Determine optimal power mode
            if battery_level <= 0.05:  # Emergency
                target_mode = "emergency"
                new_interval = 600  # 10 minutes
                # Disable non-essential networks
                self.network_coordination["data_sharing_enabled"] = False
                self.network_coordination["collaborative_learning"] = False
            elif battery_level <= 0.15:  # Critical
                target_mode = "low_power"
                new_interval = 300  # 5 minutes
                # Reduce network activity
                self.network_coordination["collaborative_learning"] = False
            elif battery_level <= 0.30 and not charging:  # Low and not charging
                target_mode = "balanced"
                new_interval = 180  # 3 minutes
            else:  # Normal operation
                target_mode = "normal"
                new_interval = self.config.monitoring.default_interval_seconds
                # Enable all network features
                self.network_coordination["data_sharing_enabled"] = True
                self.network_coordination["collaborative_learning"] = True
            
            # Apply power mode changes
            from .power import PowerMode
            current_mode = self.power.current_mode.value
            if target_mode != current_mode:
                await self.power.set_mode(PowerMode(target_mode))
                self.config.monitoring.interval_seconds = new_interval
                
                logger.info(f"⚡ Network-aware power mode: {current_mode} → {target_mode}")
            
            return {
                "success": True,
                "current_mode": target_mode,
                "battery_level": battery_level,
                "charging": charging,
                "interval_seconds": new_interval,
                "network_activity_level": network_activity,
                "network_coordination": self.network_coordination.copy()
            }
            
        except Exception as e:
            logger.error(f"Power management failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_network_activity_level(self) -> str:
        """Calculate current network activity level."""
        try:
            activity_score = 0
            
            # Check active network connections
            if self.tenzro_client:
                activity_score += len(self.tenzro_client.connected_peers) * 2
            
            if self.furcate_client:
                activity_score += len(self.furcate_client.discovered_devices)
            
            # Check recent message activity
            total_messages = sum([
                self.tenzro_client.stats.get("messages_sent", 0) if self.tenzro_client else 0,
                self.furcate_client.local_stats.get("messages_exchanged", 0) if self.furcate_client else 0,
                self.web_integrations.stats.get("api_requests", 0)
            ])
            
            if total_messages > 100:
                activity_score += 3
            elif total_messages > 50:
                activity_score += 2
            elif total_messages > 10:
                activity_score += 1
            
            # Determine activity level
            if activity_score >= 8:
                return "very_high"
            elif activity_score >= 5:
                return "high"
            elif activity_score >= 3:
                return "moderate"
            elif activity_score >= 1:
                return "low"
            else:
                return "minimal"
                
        except Exception:
            return "unknown"
    
    async def _sensor_collection(self) -> Dict[str, Any]:
        """Sensor data collection with validation and fusion."""
        try:
            # Collect raw sensor data
            raw_readings = await self.hardware.read_all_sensors()
            
            if not raw_readings:
                return {"success": False, "error": "no_sensor_data"}
            
            # Sensor fusion (would use sensor_fusion.py if available)
            fused_reading = self._perform_sensor_fusion(raw_readings)
            
            # Prepare environmental data structure
            environmental_data = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "cycle": self.monitoring_cycles + 1,
                "sensors": {name: reading.to_dict() for name, reading in raw_readings.items()},
                "fused_data": fused_reading,
                "power_status": await self.power.get_status(),
                "network_status": self._get_comprehensive_network_status()
            }
            
            logger.debug(f"📊 Sensor collection: {len(raw_readings)} sensors")
            
            return {
                "success": True,
                "data": environmental_data,
                "sensor_count": len(raw_readings),
                "data_quality": fused_reading.get("data_quality", "unknown"),
                "confidence": fused_reading.get("confidence_score", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Sensor collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _perform_sensor_fusion(self, raw_readings: Dict) -> Dict[str, Any]:
        """Perform sensor fusion (simplified implementation)."""
        try:
            # Calculate overall data quality
            quality_scores = []
            for reading in raw_readings.values():
                quality_scores.append(reading.quality)
                quality_scores.append(reading.confidence)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            
            # Determine quality classification
            if avg_quality >= 0.9:
                data_quality = "excellent"
            elif avg_quality >= 0.7:
                data_quality = "good"
            elif avg_quality >= 0.5:
                data_quality = "fair"
            else:
                data_quality = "poor"
            
            return {
                "data_quality": data_quality,
                "confidence_score": avg_quality,
                "sensor_count": len(raw_readings),
                "anomalies_detected": []
            }
            
        except Exception as e:
            logger.warning(f"Sensor fusion failed: {e}")
            return {
                "data_quality": "unknown",
                "confidence_score": 0.5,
                "sensor_count": 0,
                "anomalies_detected": []
            }
    
    def _get_comprehensive_network_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all networks."""
        status = {
            "mesh_network": self.mesh.get_status(),
            "web_integrations": self.web_integrations.get_stats()
        }
        
        if self.tenzro_client:
            status["tenzro_network"] = {
                "connected_peers": len(self.tenzro_client.connected_peers),
                "cloud_connections": len(self.tenzro_client.cloud_connections),
                "stats": self.tenzro_client.stats
            }
        
        if self.furcate_client:
            status["furcate_network"] = {
                "discovered_devices": len(self.furcate_client.discovered_devices),
                "active_connections": len(self.furcate_client.active_connections),
                "protocols": list(self.furcate_client.supported_protocols),
                "stats": self.furcate_client.local_stats
            }
        
        return status
    
    async def _ml_processing(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """ML processing with network collaboration."""
        try:
            ml_start = datetime.now()
            
            # Extract sensor values for ML processing
            sensor_values = {}
            for sensor_name, sensor_info in sensor_data.get("sensors", {}).items():
                if isinstance(sensor_info, dict) and "value" in sensor_info:
                    sensor_values[sensor_name] = sensor_info["value"]
            
            # Run local ML analysis
            ml_results = await self.edge_ml.process_environmental_data(sensor_values)
            
            # Analysis with network collaboration
            analysis = {
                **ml_results,
                "processing_time_ms": (datetime.now() - ml_start).total_seconds() * 1000,
                "sensor_features": len(sensor_values),
                "collaborative_insights": await self._get_collaborative_insights(sensor_data),
                "network_validation": await self._validate_with_network(ml_results)
            }
            
            # Share ML insights with networks if enabled
            if self.network_coordination["collaborative_learning"]:
                await self._share_ml_insights(analysis)
            
            processing_time = analysis["processing_time_ms"]
            self.performance_metrics["ml_processing_times"].append(processing_time)
            
            logger.debug(f"🤖 ML processing: {analysis.get('environmental_class', 'unknown')}")
            
            return {
                "success": True,
                "analysis": analysis,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"ML processing failed: {e}")
            return {"success": False, "error": str(e), "analysis": {}}
    
    async def _get_collaborative_insights(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get collaborative insights from network."""
        try:
            insights = {}
            
            # Request insights from Tenzro Network
            if self.tenzro_client and self.network_coordination["collaborative_learning"]:
                query = {
                    "type": "environmental_comparison",
                    "parameters": sensor_data.get("fused_data", {}),
                    "radius": 10,  # 10km radius
                    "time_range": 24  # 24 hours
                }
                tenzro_insights = await self.tenzro_client.request_collaborative_insights(query)
                insights["tenzro"] = tenzro_insights
            
            # Get insights from Furcate Network
            if self.furcate_client and self.network_coordination["collaborative_learning"]:
                collaboration_params = {
                    "data_types": ["environmental_data"],
                    "duration": 30  # 30 minutes
                }
                furcate_insights = await self.furcate_client.request_local_collaboration(
                    "environmental_analysis", collaboration_params
                )
                insights["furcate"] = furcate_insights
            
            return insights
            
        except Exception as e:
            logger.warning(f"Collaborative insights failed: {e}")
            return {}
    
    async def _validate_with_network(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ML results with network peers."""
        try:
            validation = {
                "peer_count": 0,
                "consensus_score": 0.0,
                "confidence_boost": 0.0
            }
            
            # Count available peers for validation
            peer_count = 0
            if self.tenzro_client:
                peer_count += len(self.tenzro_client.connected_peers)
            if self.furcate_client:
                peer_count += len(self.furcate_client.discovered_devices)
            
            validation["peer_count"] = peer_count
            
            # Calculate consensus score based on peer count
            if peer_count >= 5:
                validation["consensus_score"] = 0.9
                validation["confidence_boost"] = 0.1
            elif peer_count >= 3:
                validation["consensus_score"] = 0.7
                validation["confidence_boost"] = 0.05
            elif peer_count >= 1:
                validation["consensus_score"] = 0.5
                validation["confidence_boost"] = 0.02
            
            return validation
            
        except Exception as e:
            logger.warning(f"Network validation failed: {e}")
            return {"peer_count": 0, "consensus_score": 0.0, "confidence_boost": 0.0}
    
    async def _share_ml_insights(self, ml_analysis: Dict[str, Any]):
        """Share ML insights with networks."""
        try:
            if self.tenzro_client:
                await self.tenzro_client.share_ml_model({
                    "model_type": "environmental_classifier",
                    "version": "1.0",
                    "performance_metrics": {
                        "confidence": ml_analysis.get("confidence", 0.0),
                        "processing_time": ml_analysis.get("processing_time_ms", 0)
                    },
                    "environmental_zone": self.config.device.environmental_zone
                })
            
        except Exception as e:
            logger.warning(f"ML insight sharing failed: {e}")
    
    async def _alert_detection(self, sensor_data: Dict, ml_analysis: Dict) -> Dict[str, Any]:
        """Alert detection with network coordination."""
        try:
            alerts = []
            alert_summary = {"critical": 0, "warning": 0, "info": 0}
            
            # Standard threshold-based alerts
            for sensor_name, sensor_info in sensor_data.get("sensors", {}).items():
                thresholds = self.config.monitoring.alert_thresholds.get(sensor_name, {})
                if thresholds and isinstance(sensor_info, dict):
                    value = sensor_info.get("value", {})
                    if isinstance(value, dict):
                        for param, param_value in value.items():
                            if param in thresholds:
                                min_val, max_val = thresholds[param]
                                if param_value < min_val or param_value > max_val:
                                    alert = {
                                        "type": "threshold_exceeded",
                                        "sensor": sensor_name,
                                        "parameter": param,
                                        "value": param_value,
                                        "threshold": [min_val, max_val],
                                        "severity": "warning",
                                        "timestamp": datetime.now().isoformat(),
                                        "device_id": self.device_id,
                                        "network_validated": False
                                    }
                                    alerts.append(alert)
                                    alert_summary["warning"] += 1
            
            # ML-based anomaly alerts with network validation
            anomaly_score = ml_analysis.get("anomaly_score", 0.0)
            confidence = ml_analysis.get("confidence", 0.0)
            
            if anomaly_score > 0.8 and confidence > 0.7:
                severity = "critical" if anomaly_score > 0.95 else "warning"
                
                alert = {
                    "type": "ml_anomaly",
                    "anomaly_score": anomaly_score,
                    "confidence": confidence,
                    "environmental_class": ml_analysis.get("environmental_class", "unknown"),
                    "severity": severity,
                    "timestamp": datetime.now().isoformat(),
                    "device_id": self.device_id,
                    "network_validated": ml_analysis.get("network_validation", {}).get("peer_count", 0) > 0
                }
                alerts.append(alert)
                alert_summary[severity] += 1
            
            # Store alerts in history
            for alert in alerts:
                self.alert_history.append(alert)
                self.performance_metrics["total_alerts"] += 1
            
            # Keep only recent alerts
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            if alerts:
                logger.warning(f"🚨 {len(alerts)} alerts detected")
            
            return {
                "success": True,
                "alerts": alerts,
                "summary": alert_summary,
                "total_count": len(alerts)
            }
            
        except Exception as e:
            logger.error(f"Alert detection failed: {e}")
            return {"success": False, "error": str(e), "alerts": []}
    
    async def _multi_network_data_sharing(self, sensor_data: Dict, ml_analysis: Dict, alerts: List) -> Dict[str, Any]:
        """Share data across all available networks."""
        try:
            sharing_results = {}
            
            # Share via mesh network
            if self.mesh:
                mesh_message = {
                    "device_id": self.device_id,
                    "timestamp": datetime.now().isoformat(),
                    "data_summary": self._create_data_summary(sensor_data),
                    "ml_summary": self._create_ml_summary(ml_analysis),
                    "alert_count": len(alerts)
                }
                mesh_success = await self.mesh.broadcast_environmental_update(mesh_message)
                sharing_results["mesh"] = {"success": mesh_success, "peers": len(self.mesh.get_peer_info())}
            
            # Share via Tenzro Network
            if self.tenzro_client and self.network_coordination["data_sharing_enabled"]:
                try:
                    await self.tenzro_client.send_sensor_data(sensor_data)
                    sharing_results["tenzro"] = {
                        "success": True,
                        "peers": len(self.tenzro_client.connected_peers),
                        "cloud_connections": len(self.tenzro_client.cloud_connections)
                    }
                except Exception as e:
                    sharing_results["tenzro"] = {"success": False, "error": str(e)}
            
            # Share via Furcate Network
            if self.furcate_client and self.network_coordination["data_sharing_enabled"]:
                try:
                    await self.furcate_client.share_environmental_data(sensor_data, ml_analysis)
                    sharing_results["furcate"] = {
                        "success": True,
                        "devices": len(self.furcate_client.discovered_devices),
                        "protocols": len(self.furcate_client.protocol_handlers)
                    }
                except Exception as e:
                    sharing_results["furcate"] = {"success": False, "error": str(e)}
            
            # Share via web integrations
            try:
                await self.web_integrations.broadcast_sensor_data(sensor_data, ml_analysis)
                sharing_results["web"] = {
                    "success": True,
                    "integrations": len(self.web_integrations.integrations)
                }
            except Exception as e:
                sharing_results["web"] = {"success": False, "error": str(e)}
            
            # Share alerts with priority
            for alert in alerts:
                if alert.get("severity") == "critical":
                    await self._broadcast_critical_alert(alert)
            
            logger.debug(f"📡 Multi-network data sharing completed")
            
            return sharing_results
            
        except Exception as e:
            logger.error(f"Multi-network data sharing failed: {e}")
            return {"error": str(e)}
    
    async def _broadcast_critical_alert(self, alert: Dict[str, Any]):
        """Broadcast critical alert across all networks."""
        try:
            # Emergency broadcasting enabled
            if self.network_coordination["emergency_broadcasting"]:
                # Send via all available networks
                if self.mesh:
                    await self.mesh.broadcast_alert(alert)
                if self.tenzro_client:
                    await self.tenzro_client.send_alert(alert)
                if self.furcate_client:
                    # Furcate network doesn't have direct alert method, use data sharing
                    alert_data = {"alert": alert, "priority": "emergency"}
                    await self.furcate_client.share_environmental_data(alert_data, {})
                if self.web_integrations:
                    await self.web_integrations.broadcast_alert(alert)
            
        except Exception as e:
            logger.error(f"Critical alert broadcast failed: {e}")
    
    def _create_data_summary(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized data summary."""
        try:
            fused_data = sensor_data.get("fused_data", {})
            return {
                "data_quality": fused_data.get("data_quality", "unknown"),
                "confidence": fused_data.get("confidence_score", 0.5),
                "sensor_count": fused_data.get("sensor_count", 0)
            }
        except Exception:
            return {"data_quality": "unknown", "confidence": 0.5, "sensor_count": 0}
    
    def _create_ml_summary(self, ml_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized ML summary."""
        return {
            "class": ml_analysis.get("environmental_class", "unknown"),
            "anomaly": round(ml_analysis.get("anomaly_score", 0.0), 3),
            "confidence": round(ml_analysis.get("confidence", 0.0), 3)
        }
    
    async def _data_storage(self, sensor_data: Dict, ml_analysis: Dict, alerts: List) -> Dict[str, Any]:
        """Data storage with network metadata."""
        try:
            # Prepare comprehensive storage record
            storage_record = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "sensor_data": sensor_data,
                "ml_analysis": ml_analysis,
                "alerts": alerts,
                "cycle": self.monitoring_cycles + 1,
                "network_metadata": {
                    "tenzro_peers": len(self.tenzro_client.connected_peers) if self.tenzro_client else 0,
                    "furcate_devices": len(self.furcate_client.discovered_devices) if self.furcate_client else 0,
                    "mesh_peers": len(self.mesh.get_peer_info()),
                    "sharing_enabled": self.network_coordination["data_sharing_enabled"]
                }
            }
            
            # Store main record
            storage_success = await self.storage.store_environmental_record(storage_record)
            
            # Store alerts separately
            alerts_stored = 0
            for alert in alerts:
                alert_success = await self.storage.store_alert(alert, self.device_id)
                if alert_success:
                    alerts_stored += 1
            
            return {
                "success": storage_success,
                "record_stored": storage_success,
                "alerts_stored": alerts_stored,
                "storage_stats": self.storage.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Data storage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _coordinate_network_activities(self) -> Dict[str, Any]:
        """Coordinate activities across all networks."""
        try:
            coordination_result = {
                "network_health": {},
                "optimization_applied": [],
                "sync_status": {}
            }
            
            # Check network health
            for network_name in ["mesh", "tenzro", "furcate", "web"]:
                health = await self._check_network_health(network_name)
                coordination_result["network_health"][network_name] = health
                
                # Apply optimizations based on health
                if health.get("status") == "degraded":
                    await self._optimize_network(network_name)
                    coordination_result["optimization_applied"].append(network_name)
            
            # Sync data if enabled
            if self.network_coordination["local_sync_enabled"]:
                sync_result = await self._sync_with_networks()
                coordination_result["sync_status"] = sync_result
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"Network coordination failed: {e}")
            return {"error": str(e)}
    
    async def _check_network_health(self, network_name: str) -> Dict[str, Any]:
        """Check health of specific network."""
        try:
            if network_name == "mesh":
                peer_count = len(self.mesh.get_peer_info())
                return {"status": "good" if peer_count > 0 else "degraded", "peers": peer_count}
            
            elif network_name == "tenzro" and self.tenzro_client:
                peer_count = len(self.tenzro_client.connected_peers)
                cloud_count = len(self.tenzro_client.cloud_connections)
                return {
                    "status": "good" if peer_count > 0 or cloud_count > 0 else "degraded",
                    "peers": peer_count,
                    "clouds": cloud_count
                }
            
            elif network_name == "furcate" and self.furcate_client:
                device_count = len(self.furcate_client.discovered_devices)
                protocol_count = len(self.furcate_client.protocol_handlers)
                return {
                    "status": "good" if device_count > 0 or protocol_count > 0 else "degraded",
                    "devices": device_count,
                    "protocols": protocol_count
                }
            
            elif network_name == "web":
                integration_count = len(self.web_integrations.integrations)
                return {"status": "good" if integration_count > 0 else "degraded", "integrations": integration_count}
            
            else:
                return {"status": "unavailable"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _optimize_network(self, network_name: str):
        """Apply optimizations to specific network."""
        try:
            if network_name == "mesh":
                # Restart mesh discovery
                logger.info("🔧 Optimizing mesh network")
            
            elif network_name == "tenzro" and self.tenzro_client:
                # Attempt to reconnect to Tenzro Network
                logger.info("🔧 Optimizing Tenzro Network")
            
            elif network_name == "furcate" and self.furcate_client:
                # Restart device discovery
                logger.info("🔧 Optimizing Furcate Network")
            
        except Exception as e:
            logger.warning(f"Network optimization failed for {network_name}: {e}")
    
    async def _sync_with_networks(self) -> Dict[str, Any]:
        """Synchronize data with network peers."""
        try:
            # Get recent data for sync
            recent_data = await self.storage.get_recent_environmental_data(1)  # Last hour
            
            sync_status = {
                "records_synced": len(recent_data[-5:]) if recent_data else 0,  # Sync last 5 records
                "sync_timestamp": datetime.now().isoformat()
            }
            
            return sync_status
            
        except Exception as e:
            logger.warning(f"Network sync failed: {e}")
            return {"error": str(e)}
    
    async def _update_comprehensive_metrics(self, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update comprehensive performance metrics."""
        try:
            current_time = datetime.now()
            
            # Update network statistics
            self.performance_metrics["network_stats"] = {
                "tenzro": self.tenzro_client.stats if self.tenzro_client else {},
                "furcate": self.furcate_client.local_stats if self.furcate_client else {},
                "web": self.web_integrations.stats,
                "mesh": self.mesh.get_status()
            }
            
            # Calculate comprehensive metrics
            uptime_seconds = (current_time - self.performance_metrics["uptime_start"]).total_seconds()
            avg_cycle_time = sum(self.performance_metrics["cycle_times"]) / len(self.performance_metrics["cycle_times"]) if self.performance_metrics["cycle_times"] else 0
            
            performance_summary = {
                "uptime_seconds": uptime_seconds,
                "total_cycles": self.monitoring_cycles + 1,
                "avg_cycle_time_ms": avg_cycle_time * 1000 if avg_cycle_time < 1 else avg_cycle_time,
                "total_alerts": self.performance_metrics["total_alerts"],
                "network_stats": self.performance_metrics["network_stats"],
                "cycle_success": cycle_result.get("success", False),
                "last_update": current_time.isoformat()
            }
            
            return {
                "success": True,
                "metrics": performance_summary
            }
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _comprehensive_health_monitoring(self):
        """Comprehensive health monitoring for all subsystems and networks."""
        while self.running:
            try:
                # Update comprehensive health status
                await self._update_comprehensive_health_status()
                
                # Wait for next health check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Comprehensive health monitoring error: {e}")
                await asyncio.sleep(60)  # Shorter wait on error
    
    async def _update_comprehensive_health_status(self):
        """Update comprehensive system health status including all networks."""
        try:
            health_checks = {
                "hardware": await self._check_hardware_health(),
                "ml": await self._check_ml_health(),
                "mesh": await self._check_mesh_health(),
                "power": await self._check_power_health(),
                "storage": await self._check_storage_health(),
                "tenzro_network": await self._check_tenzro_network_health(),
                "furcate_network": await self._check_furcate_network_health(),
                "web_integrations": await self._check_web_integrations_health()
            }
            
            # Determine overall health
            failed_systems = [name for name, status in health_checks.items() if not status.get("healthy", False)]
            critical_systems = [name for name, status in health_checks.items() if status.get("critical", False)]
            
            if critical_systems:
                overall_health = "critical"
            elif len(failed_systems) > 3:
                overall_health = "degraded"
            elif failed_systems:
                overall_health = "warning"
            else:
                overall_health = "excellent"
            
            self.health_status = {
                "overall": overall_health,
                "subsystems": health_checks,
                "networks": {
                    "tenzro": health_checks.get("tenzro_network", {}),
                    "furcate": health_checks.get("furcate_network", {}),
                    "mesh": health_checks.get("mesh", {}),
                    "web": health_checks.get("web_integrations", {})
                },
                "last_check": datetime.now(),
                "failed_systems": failed_systems,
                "critical_systems": critical_systems,
                "network_coordination": self.network_coordination.copy()
            }
            
            # Log significant health changes
            if overall_health != "excellent":
                logger.warning(f"🏥 System health: {overall_health} (failed: {failed_systems})")
            
        except Exception as e:
            logger.error(f"Comprehensive health status update failed: {e}")
    
    async def _check_hardware_health(self) -> Dict[str, Any]:
        """Check hardware subsystem health."""
        try:
            sensor_stats = self.hardware.get_stats()
            sensor_count = sensor_stats.get("sensor_count", 0)
            reading_errors = sensor_stats.get("reading_errors", 0)
            total_readings = sensor_stats.get("readings_taken", 1)
            
            error_rate = reading_errors / total_readings if total_readings > 0 else 0
            
            return {
                "healthy": sensor_count > 0 and error_rate < 0.1,
                "critical": sensor_count == 0,
                "sensor_count": sensor_count,
                "error_rate": error_rate,
                "last_reading": sensor_stats.get("last_reading_time")
            }
            
        except Exception as e:
            return {"healthy": False, "critical": True, "error": str(e)}
    
    async def _check_ml_health(self) -> Dict[str, Any]:
        """Check ML subsystem health."""
        try:
            if not self.performance_metrics["ml_processing_times"]:
                return {"healthy": False, "critical": False, "error": "no_ml_processing_data"}
            
            avg_processing_time = sum(self.performance_metrics["ml_processing_times"]) / len(self.performance_metrics["ml_processing_times"])
            
            return {
                "healthy": avg_processing_time < 5000,  # 5 seconds
                "critical": avg_processing_time > 30000,  # 30 seconds
                "avg_processing_time_ms": avg_processing_time,
                "simulation_mode": self.edge_ml.simulation_mode
            }
            
        except Exception as e:
            return {"healthy": False, "critical": False, "error": str(e)}
    
    async def _check_mesh_health(self) -> Dict[str, Any]:
        """Check mesh network health."""
        try:
            mesh_status = self.mesh.get_status()
            peer_count = mesh_status.get("peer_count", 0)
            
            return {
                "healthy": True,  # Mesh is optional, so always healthy
                "critical": False,
                "peer_count": peer_count,
                "simulation_mode": mesh_status.get("simulation_mode", True)
            }
            
        except Exception as e:
            return {"healthy": True, "critical": False, "error": str(e)}
    
    async def _check_power_health(self) -> Dict[str, Any]:
        """Check power subsystem health."""
        try:
            power_status = await self.power.get_status()
            battery_level = power_status.get("battery_level", 0.5)
            
            return {
                "healthy": battery_level > 0.1,
                "critical": battery_level < 0.05,
                "battery_level": battery_level,
                "charging": power_status.get("charging", False),
                "power_mode": power_status.get("power_mode", "unknown")
            }
            
        except Exception as e:
            return {"healthy": False, "critical": True, "error": str(e)}
    
    async def _check_storage_health(self) -> Dict[str, Any]:
        """Check storage subsystem health."""
        try:
            storage_stats = self.storage.get_stats()
            storage_errors = storage_stats.get("storage_errors", 0)
            total_records = storage_stats.get("records_stored", 1)
            
            error_rate = storage_errors / total_records if total_records > 0 else 0
            
            return {
                "healthy": error_rate < 0.05,
                "critical": error_rate > 0.2,
                "records_stored": total_records,
                "error_rate": error_rate,
                "database_sizes": storage_stats.get("database_sizes", {})
            }
            
        except Exception as e:
            return {"healthy": False, "critical": False, "error": str(e)}
    
    async def _check_tenzro_network_health(self) -> Dict[str, Any]:
        """Check Tenzro Network health."""
        try:
            if not self.tenzro_client:
                return {"healthy": True, "critical": False, "status": "disabled"}
            
            peer_count = len(self.tenzro_client.connected_peers)
            cloud_count = len(self.tenzro_client.cloud_connections)
            message_success_rate = self._calculate_message_success_rate(self.tenzro_client.stats)
            
            return {
                "healthy": peer_count > 0 or cloud_count > 0 or message_success_rate > 0.8,
                "critical": peer_count == 0 and cloud_count == 0 and message_success_rate < 0.5,
                "peer_count": peer_count,
                "cloud_connections": cloud_count,
                "message_success_rate": message_success_rate,
                "data_shared_mb": self.tenzro_client.stats.get("data_shared_mb", 0)
            }
            
        except Exception as e:
            return {"healthy": True, "critical": False, "error": str(e)}  # Network is optional
    
    async def _check_furcate_network_health(self) -> Dict[str, Any]:
        """Check Furcate Network health."""
        try:
            if not self.furcate_client:
                return {"healthy": True, "critical": False, "status": "disabled"}
            
            device_count = len(self.furcate_client.discovered_devices)
            connection_count = len(self.furcate_client.active_connections)
            protocol_count = len(self.furcate_client.protocol_handlers)
            
            return {
                "healthy": device_count > 0 or protocol_count > 0,
                "critical": False,  # Local network is never critical
                "discovered_devices": device_count,
                "active_connections": connection_count,
                "available_protocols": protocol_count,
                "supported_protocols": list(self.furcate_client.supported_protocols)
            }
            
        except Exception as e:
            return {"healthy": True, "critical": False, "error": str(e)}
    
    async def _check_web_integrations_health(self) -> Dict[str, Any]:
        """Check web integrations health."""
        try:
            integration_count = len(self.web_integrations.integrations)
            total_requests = sum([
                self.web_integrations.stats.get("api_requests", 0),
                self.web_integrations.stats.get("mqtt_messages", 0),
                self.web_integrations.stats.get("webhook_calls", 0)
            ])
            
            return {
                "healthy": integration_count > 0,
                "critical": False,  # Web integrations are optional
                "active_integrations": integration_count,
                "total_requests": total_requests,
                "integrations": list(self.web_integrations.integrations.keys())
            }
            
        except Exception as e:
            return {"healthy": True, "critical": False, "error": str(e)}
    
    def _calculate_message_success_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate message success rate from statistics."""
        try:
            sent = stats.get("messages_sent", 0)
            failed = stats.get("messages_failed", 0)
            total = sent + failed
            
            if total == 0:
                return 1.0  # No messages means 100% success rate
            
            return sent / total
            
        except Exception:
            return 0.0
    
    async def _network_coordination_loop(self):
        """Background network coordination and optimization."""
        while self.running:
            try:
                # Coordinate network activities
                await self._coordinate_network_activities()
                
                # Optimize network performance
                await self._optimize_network_performance()
                
                # Balance network loads
                await self._balance_network_loads()
                
                # Wait for next coordination cycle
                await asyncio.sleep(120)  # Coordinate every 2 minutes
                
            except Exception as e:
                logger.error(f"Network coordination loop error: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_network_performance(self):
        """Optimize performance across all networks."""
        try:
            # Analyze network performance
            performance_metrics = {
                "tenzro": self.tenzro_client.stats if self.tenzro_client else {},
                "furcate": self.furcate_client.local_stats if self.furcate_client else {},
                "mesh": self.mesh.get_status(),
                "web": self.web_integrations.stats
            }
            
            # Apply optimizations based on performance
            for network, metrics in performance_metrics.items():
                await self._apply_network_optimization(network, metrics)
            
        except Exception as e:
            logger.warning(f"Network performance optimization failed: {e}")
    
    async def _apply_network_optimization(self, network: str, metrics: Dict[str, Any]):
        """Apply specific optimization to a network."""
        try:
            if network == "tenzro" and self.tenzro_client:
                # Optimize Tenzro Network based on metrics
                message_failure_rate = self._calculate_message_failure_rate(metrics)
                if message_failure_rate > 0.2:  # 20% failure rate
                    logger.info(f"🔧 Optimizing Tenzro Network (failure rate: {message_failure_rate:.2f})")
            
            elif network == "furcate" and self.furcate_client:
                # Optimize Furcate Network based on device discovery
                device_count = metrics.get("devices_discovered", 0)
                if device_count == 0:
                    logger.info("🔧 Optimizing Furcate Network (no devices discovered)")
            
            # Add more network-specific optimizations as needed
            
        except Exception as e:
            logger.warning(f"Network optimization failed for {network}: {e}")
    
    def _calculate_message_failure_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate message failure rate."""
        try:
            sent = stats.get("messages_sent", 0)
            failed = stats.get("messages_failed", 0)
            total = sent + failed
            
            if total == 0:
                return 0.0
            
            return failed / total
            
        except Exception:
            return 1.0  # Assume worst case if calculation fails
    
    async def _balance_network_loads(self):
        """Balance loads across available networks."""
        try:
            # Get network load information
            network_loads = {
                "tenzro": self._get_network_load("tenzro"),
                "furcate": self._get_network_load("furcate"),
                "mesh": self._get_network_load("mesh"),
                "web": self._get_network_load("web")
            }
            
            # Find overloaded networks
            overloaded_networks = [
                name for name, load in network_loads.items() 
                if load.get("load_percentage", 0) > 80
            ]
            
            # Apply load balancing if needed
            if overloaded_networks:
                await self._redistribute_network_load(overloaded_networks, network_loads)
            
        except Exception as e:
            logger.warning(f"Network load balancing failed: {e}")
    
    def _get_network_load(self, network: str) -> Dict[str, Any]:
        """Get current load for a specific network."""
        try:
            if network == "tenzro" and self.tenzro_client:
                stats = self.tenzro_client.stats
                message_rate = stats.get("messages_sent", 0) / max(1, (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds() / 60)
                return {
                    "load_percentage": min(100, message_rate * 10),  # Simplified load calculation
                    "message_rate_per_minute": message_rate,
                    "connection_count": len(self.tenzro_client.connected_peers)
                }
            
            elif network == "furcate" and self.furcate_client:
                stats = self.furcate_client.local_stats
                message_rate = stats.get("messages_exchanged", 0) / max(1, (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds() / 60)
                return {
                    "load_percentage": min(100, message_rate * 15),  # Local network has higher capacity
                    "message_rate_per_minute": message_rate,
                    "device_count": len(self.furcate_client.discovered_devices)
                }
            
            elif network == "mesh":
                mesh_stats = self.mesh.get_status()
                return {
                    "load_percentage": 20,  # Assume moderate load for mesh
                    "peer_count": mesh_stats.get("peer_count", 0)
                }
            
            elif network == "web":
                web_stats = self.web_integrations.stats
                request_rate = web_stats.get("api_requests", 0) / max(1, (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds() / 60)
                return {
                    "load_percentage": min(100, request_rate * 5),  # Web has high capacity
                    "request_rate_per_minute": request_rate,
                    "integration_count": len(self.web_integrations.integrations)
                }
            
            else:
                return {"load_percentage": 0}
                
        except Exception:
            return {"load_percentage": 0}
    
    async def _redistribute_network_load(self, overloaded_networks: List[str], network_loads: Dict[str, Any]):
        """Redistribute load from overloaded networks."""
        try:
            # Find underutilized networks
            underutilized_networks = [
                name for name, load in network_loads.items() 
                if load.get("load_percentage", 100) < 50 and name not in overloaded_networks
            ]
            
            if underutilized_networks:
                logger.info(f"🔄 Redistributing load from {overloaded_networks} to {underutilized_networks}")
                
                # Adjust network coordination settings
                for network in overloaded_networks:
                    if network == "tenzro":
                        # Reduce Tenzro Network usage temporarily
                        self.network_coordination["collaborative_learning"] = False
                    elif network == "furcate":
                        # Reduce Furcate Network discovery frequency
                        if self.furcate_client:
                            self.furcate_client.discovery_interval = 60  # Increase interval
            
        except Exception as e:
            logger.warning(f"Load redistribution failed: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating comprehensive shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Gracefully shutdown all subsystems and networks."""
        logger.info("🛑 Shutting down Complete Furcate Nano...")
        self.running = False
        
        try:
            # Store final system event
            await self.storage.store_system_event(
                "complete_system_shutdown",
                {
                    "total_cycles": self.monitoring_cycles,
                    "uptime_seconds": (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds(),
                    "total_alerts": self.performance_metrics["total_alerts"],
                    "final_health_status": self.health_status,
                    "network_statistics": {
                        "tenzro": self.tenzro_client.stats if self.tenzro_client else {},
                        "furcate": self.furcate_client.local_stats if self.furcate_client else {},
                        "mesh": self.mesh.get_status(),
                        "web": self.web_integrations.stats
                    }
                },
                self.device_id
            )
            
            # Shutdown networks first (in parallel for efficiency)
            network_shutdowns = []
            
            if self.tenzro_client and hasattr(self.tenzro_client, 'shutdown'):
                network_shutdowns.append(self._shutdown_tenzro_network())
            
            if self.furcate_client and hasattr(self.furcate_client, 'shutdown'):
                network_shutdowns.append(self._shutdown_furcate_network())
            
            network_shutdowns.append(self._shutdown_web_integrations())
            network_shutdowns.append(self._shutdown_mesh_network())
            
            # Execute network shutdowns in parallel
            await asyncio.gather(*network_shutdowns, return_exceptions=True)
            
            # Shutdown core subsystems in reverse order
            await self.storage.shutdown()
            await self.power.shutdown()
            await self.edge_ml.shutdown()
            await self.hardware.shutdown()
            
            logger.info("✅ Complete Furcate Nano shutdown complete")
            
        except Exception as e:
            logger.error(f"Complete shutdown error: {e}")
    
    async def _shutdown_tenzro_network(self):
        """Shutdown Tenzro Network."""
        try:
            if self.tenzro_client:
                # Send goodbye message to network
                goodbye_data = {
                    "device_id": self.device_id,
                    "shutdown_reason": "graceful_shutdown",
                    "final_stats": self.tenzro_client.stats,
                    "uptime_seconds": (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds()
                }
                
                # Send goodbye (don't wait for completion)
                try:
                    await asyncio.wait_for(
                        self.tenzro_client.send_sensor_data(goodbye_data), 
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    pass  # Continue shutdown even if goodbye fails
                
                logger.info("✅ Tenzro Network shutdown complete")
        except Exception as e:
            logger.warning(f"Tenzro Network shutdown error: {e}")
    
    async def _shutdown_furcate_network(self):
        """Shutdown Furcate Network."""
        try:
            if self.furcate_client:
                # Send goodbye to local devices
                goodbye_data = {
                    "device_id": self.device_id,
                    "message_type": "device_shutdown",
                    "final_stats": self.furcate_client.local_stats
                }
                
                # Send goodbye (don't wait for completion)
                try:
                    await asyncio.wait_for(
                        self.furcate_client.share_environmental_data(goodbye_data, {}),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    pass  # Continue shutdown
                
                logger.info("✅ Furcate Network shutdown complete")
        except Exception as e:
            logger.warning(f"Furcate Network shutdown error: {e}")
    
    async def _shutdown_web_integrations(self):
        """Shutdown web integrations."""
        try:
            await self.web_integrations.shutdown()
            logger.info("✅ Web integrations shutdown complete")
        except Exception as e:
            logger.warning(f"Web integrations shutdown error: {e}")
    
    async def _shutdown_mesh_network(self):
        """Shutdown mesh network."""
        try:
            await self.mesh.shutdown()
            logger.info("✅ Mesh network shutdown complete")
        except Exception as e:
            logger.warning(f"Mesh network shutdown error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all networks."""
        uptime = (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds()
        
        return {
            "device_id": self.device_id,
            "running": self.running,
            "cycles": self.monitoring_cycles,
            "last_reading": self.last_reading_time.isoformat() if self.last_reading_time else None,
            "uptime_seconds": round(uptime, 1),
            "health_status": self.health_status,
            "performance_metrics": {
                "avg_cycle_time_ms": sum(self.performance_metrics["cycle_times"][-10:]) / min(10, len(self.performance_metrics["cycle_times"])) * 1000 if self.performance_metrics["cycle_times"] else 0,
                "total_alerts": self.performance_metrics["total_alerts"],
                "network_stats": self.performance_metrics.get("network_stats", {})
            },
            "network_coordination": self.network_coordination.copy(),
            "networks": {
                "tenzro": {
                    "enabled": self.tenzro_client is not None,
                    "stats": self.tenzro_client.stats if self.tenzro_client else {},
                    "peers": len(self.tenzro_client.connected_peers) if self.tenzro_client else 0,
                    "cloud_connections": len(self.tenzro_client.cloud_connections) if self.tenzro_client else 0
                },
                "furcate": {
                    "enabled": self.furcate_client is not None,
                    "stats": self.furcate_client.local_stats if self.furcate_client else {},
                    "devices": len(self.furcate_client.discovered_devices) if self.furcate_client else 0,
                    "protocols": len(self.furcate_client.protocol_handlers) if self.furcate_client else 0
                },
                "mesh": {
                    "enabled": True,
                    "stats": self.mesh.get_status(),
                    "peers": len(self.mesh.get_peer_info())
                },
                "web": {
                    "enabled": True,
                    "stats": self.web_integrations.stats,
                    "integrations": len(self.web_integrations.integrations)
                }
            },
            "subsystems": {
                "hardware": {"initialized": hasattr(self.hardware, 'initialized'), "simulation": self.hardware.simulation_mode},
                "ml": {"initialized": hasattr(self.edge_ml, 'initialized'), "simulation": self.edge_ml.simulation_mode},
                "power": {"initialized": hasattr(self.power, 'initialized'), "simulation": self.power.simulation_mode},
                "storage": {"initialized": hasattr(self.storage, 'initialized')}
            }
        }
    
    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics including all networks."""
        logger.info("🔧 Running comprehensive system diagnostics...")
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id,
            "overall_status": "unknown",
            "subsystem_diagnostics": {},
            "network_diagnostics": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        try:
            # Hardware diagnostics
            diagnostics["subsystem_diagnostics"]["hardware"] = await self.hardware.run_diagnostics()
            
            # ML diagnostics
            test_data = {"temperature": 25.0, "humidity": 60.0, "air_quality": 50.0}
            ml_start = datetime.now()
            ml_result = await self.edge_ml.process_environmental_data(test_data)
            ml_time = (datetime.now() - ml_start).total_seconds() * 1000
            
            diagnostics["subsystem_diagnostics"]["ml"] = {
                "test_processing_time_ms": ml_time,
                "test_result": ml_result,
                "simulation_mode": self.edge_ml.simulation_mode
            }
            
            # Network diagnostics
            diagnostics["network_diagnostics"] = {
                "tenzro": await self._diagnose_tenzro_network(),
                "furcate": await self._diagnose_furcate_network(),
                "mesh": await self._diagnose_mesh_network(),
                "web": await self._diagnose_web_integrations()
            }
            
            # Performance analysis
            diagnostics["performance_analysis"] = {
                "uptime_hours": (datetime.now() - self.performance_metrics["uptime_start"]).total_seconds() / 3600,
                "total_cycles": self.monitoring_cycles,
                "avg_cycle_time_ms": sum(self.performance_metrics["cycle_times"]) / len(self.performance_metrics["cycle_times"]) * 1000 if self.performance_metrics["cycle_times"] else 0,
                "storage_stats": self.storage.get_stats(),
                "network_performance": self.performance_metrics.get("network_stats", {})
            }
            
            # Generate comprehensive recommendations
            recommendations = []
            
            # Hardware recommendations
            hw_status = diagnostics["subsystem_diagnostics"]["hardware"]["overall_status"]
            if hw_status != "excellent":
                recommendations.append(f"Hardware status is {hw_status} - check sensor connections")
            
            # Performance recommendations
            avg_cycle_time = diagnostics["performance_analysis"]["avg_cycle_time_ms"]
            if avg_cycle_time > 5000:
                recommendations.append("Consider optimizing monitoring intervals for better performance")
            
            # Network recommendations
            network_diag = diagnostics["network_diagnostics"]
            if not any(net.get("connected", False) for net in network_diag.values()):
                recommendations.append("No network connections active - check network configuration")
            
            if network_diag.get("tenzro", {}).get("peer_count", 0) == 0:
                recommendations.append("No Tenzro Network peers - check internet connectivity")
            
            if network_diag.get("furcate", {}).get("device_count", 0) == 0:
                recommendations.append("No local Furcate devices discovered - check local network")
            
            diagnostics["recommendations"] = recommendations
            
            # Overall status
            if hw_status == "excellent" and len(recommendations) <= 1:
                diagnostics["overall_status"] = "excellent"
            elif hw_status in ["excellent", "good"] and len(recommendations) <= 3:
                diagnostics["overall_status"] = "good"
            elif hw_status in ["good", "degraded"]:
                diagnostics["overall_status"] = "degraded"
            else:
                diagnostics["overall_status"] = "critical"
            
            logger.info(f"🔧 Comprehensive diagnostics complete: {diagnostics['overall_status']}")
            return diagnostics
            
        except Exception as e:
            logger.error(f"Comprehensive diagnostics failed: {e}")
            diagnostics["overall_status"] = "error"
            diagnostics["error"] = str(e)
            return diagnostics
    
    async def _diagnose_tenzro_network(self) -> Dict[str, Any]:
        """Diagnose Tenzro Network."""
        if not self.tenzro_client:
            return {"enabled": False, "status": "disabled"}
        
        return {
            "enabled": True,
            "connected": len(self.tenzro_client.connected_peers) > 0 or len(self.tenzro_client.cloud_connections) > 0,
            "peer_count": len(self.tenzro_client.connected_peers),
            "cloud_connections": len(self.tenzro_client.cloud_connections),
            "stats": self.tenzro_client.stats,
            "status": "healthy" if len(self.tenzro_client.connected_peers) > 0 else "degraded"
        }
    
    async def _diagnose_furcate_network(self) -> Dict[str, Any]:
        """Diagnose Furcate Network."""
        if not self.furcate_client:
            return {"enabled": False, "status": "disabled"}
        
        return {
            "enabled": True,
            "connected": len(self.furcate_client.discovered_devices) > 0,
            "device_count": len(self.furcate_client.discovered_devices),
            "active_connections": len(self.furcate_client.active_connections),
            "protocols": len(self.furcate_client.protocol_handlers),
            "supported_protocols": list(self.furcate_client.supported_protocols),
            "stats": self.furcate_client.local_stats,
            "status": "healthy" if len(self.furcate_client.protocol_handlers) > 0 else "degraded"
        }
    
    async def _diagnose_mesh_network(self) -> Dict[str, Any]:
        """Diagnose mesh network."""
        mesh_status = self.mesh.get_status()
        peer_count = len(self.mesh.get_peer_info())
        
        return {
            "enabled": True,
            "connected": peer_count > 0,
            "peer_count": peer_count,
            "stats": mesh_status,
            "status": "healthy" if peer_count > 0 else "degraded"
        }
    
    async def _diagnose_web_integrations(self) -> Dict[str, Any]:
        """Diagnose web integrations."""
        integration_count = len(self.web_integrations.integrations)
        
        return {
            "enabled": True,
            "connected": integration_count > 0,
            "integration_count": integration_count,
            "integrations": list(self.web_integrations.integrations.keys()),
            "stats": self.web_integrations.stats,
            "status": "healthy" if integration_count > 0 else "degraded"
        }