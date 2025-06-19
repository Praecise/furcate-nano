        # Multi-Device Collaboration Tutorial

Learn how to create distributed environmental monitoring networks with automatic peer discovery, collaborative machine learning, and emergency response coordination.

## Overview

This tutorial covers building collaborative networks of Furcate Nano devices that can:

- Discover and connect to peer devices automatically
- Share environmental data and ML insights
- Coordinate emergency responses
- Perform distributed consensus and data fusion
- Implement resilient mesh networking

## Prerequisites

### Hardware Requirements
- 3+ Furcate Nano devices for meaningful collaboration
- Compatible wireless communication (WiFi, Bluetooth, LoRa)
- Sufficient network bandwidth for data sharing
- Optional: GPS modules for location-based coordination

### Network Requirements
- Local mesh network capability
- Internet connectivity for global coordination (optional)
- Network security and encryption support
- Quality of Service (QoS) management

## Part 1: Collaborative Network Architecture

### Core Collaboration Framework

```python
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from enum import Enum
import numpy as np

class CollaborationRole(Enum):
    """Device roles in collaborative network"""
    COORDINATOR = "coordinator"    # Network coordinator and consensus leader
    SPECIALIST = "specialist"      # Specialized environmental monitoring
    RELAY = "relay"               # Network relay and bridge
    WORKER = "worker"             # Standard environmental monitoring
    EMERGENCY = "emergency"       # Emergency response and alerting

class DataPriority(Enum):
    """Data transmission priority levels"""
    EMERGENCY = 5
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1

@dataclass
class CollaborativeDevice:
    """Represents a peer device in the collaborative network"""
    device_id: str
    device_name: str
    role: CollaborationRole
    capabilities: Set[str]
    location: Dict[str, float]
    last_seen: datetime
    trust_score: float
    processing_power: float
    battery_level: float
    network_quality: float
    specializations: List[str]

class CollaborativeNetworkManager:
    """Manages collaborative networking for Furcate Nano devices"""
    
    def __init__(self, core, role: CollaborationRole = CollaborationRole.WORKER):
        self.core = core
        self.device_id = core.device_id
        self.role = role
        self.peer_devices: Dict[str, CollaborativeDevice] = {}
        self.capabilities = self._determine_capabilities()
        self.specializations = self._determine_specializations()
        
        # Network state
        self.network_active = False
        self.discovery_interval = 60  # seconds
        self.trust_threshold = 0.7
        self.consensus_data = {}
        self.emergency_propagation_hops = 5
        
        # Communication channels
        self.mesh_network = None
        self.p2p_network = None
        
    def _determine_capabilities(self) -> Set[str]:
        """Determine device capabilities based on hardware and configuration"""
        capabilities = set()
        
        # Base capabilities
        capabilities.add("sensor_monitoring")
        capabilities.add("data_sharing")
        capabilities.add("mesh_networking")
        
        # Hardware-specific capabilities
        if hasattr(self.core, 'edge_ml') and self.core.edge_ml:
            capabilities.add("edge_ml")
            capabilities.add("anomaly_detection")
        
        if hasattr(self.core, 'hardware'):
            # Check for specific sensors
            if 'air_quality' in self.core.hardware.sensors:
                capabilities.add("air_quality_monitoring")
            if 'camera' in self.core.hardware.sensors:
                capabilities.add("visual_monitoring")
            if 'gps' in self.core.hardware.sensors:
                capabilities.add("location_tracking")
        
        # Role-specific capabilities
        if self.role == CollaborationRole.COORDINATOR:
            capabilities.update(["consensus_coordination", "network_management"])
        elif self.role == CollaborationRole.SPECIALIST:
            capabilities.update(["specialized_analysis", "expert_validation"])
        elif self.role == CollaborationRole.RELAY:
            capabilities.update(["long_range_communication", "network_bridging"])
        elif self.role == CollaborationRole.EMERGENCY:
            capabilities.update(["emergency_response", "alert_coordination"])
        
        return capabilities
    
    def _determine_specializations(self) -> List[str]:
        """Determine device specializations based on configuration and sensors"""
        specializations = []
        
        # Environmental specializations
        if 'air_quality_monitoring' in self.capabilities:
            specializations.append("air_quality_expert")
        
        if hasattr(self.core, 'hardware'):
            sensor_count = len(self.core.hardware.sensors)
            if sensor_count >= 5:
                specializations.append("multi_sensor_fusion")
        
        # ML specializations
        if 'edge_ml' in self.capabilities:
            specializations.append("environmental_ml")
            specializations.append("predictive_analytics")
        
        # Location specializations
        if 'location_tracking' in self.capabilities:
            specializations.append("spatial_analysis")
            specializations.append("mobility_tracking")
        
        return specializations
    
    async def initialize_collaboration(self):
        """Initialize collaborative networking system"""
        try:
            print("🌐 Initializing collaborative network...")
            
            # Initialize mesh networking
            await self._setup_mesh_network()
            
            # Initialize P2P networking
            await self._setup_p2p_network()
            
            # Start peer discovery
            asyncio.create_task(self._peer_discovery_loop())
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            # Start consensus coordination
            if self.role == CollaborationRole.COORDINATOR:
                asyncio.create_task(self._consensus_coordination_loop())
            
            self.network_active = True
            print("✅ Collaborative network initialized")
            print(f"   Role: {self.role.value}")
            print(f"   Capabilities: {', '.join(self.capabilities)}")
            print(f"   Specializations: {', '.join(self.specializations)}")
            
        except Exception as e:
            print(f"❌ Collaboration initialization failed: {e}")
    
    async def _setup_mesh_network(self):
        """Setup mesh networking for local device communication"""
        try:
            # Initialize mesh network configuration
            mesh_config = {
                "device_id": self.device_id,
                "role": self.role.value,
                "discovery_port": 8000,
                "data_port": 8001,
                "encryption": True,
                "max_peers": 20
            }
            
            # This would integrate with actual mesh networking library
            # For demonstration, we'll simulate the mesh network
            self.mesh_network = MeshNetworkSimulator(mesh_config)
            
            print("✅ Mesh network configured")
            
        except Exception as e:
            print(f"❌ Mesh network setup failed: {e}")
    
    async def _setup_p2p_network(self):
        """Setup P2P networking for wide-area communication"""
        try:
            # P2P network configuration
            p2p_config = {
                "device_id": self.device_id,
                "capabilities": list(self.capabilities),
                "location": self.core.config.device.location,
                "trust_enabled": True
            }
            
            # This would integrate with actual P2P networking
            self.p2p_network = P2PNetworkSimulator(p2p_config)
            
            print("✅ P2P network configured")
            
        except Exception as e:
            print(f"❌ P2P network setup failed: {e}")

class MeshNetworkSimulator:
    """Simulated mesh network for demonstration"""
    
    def __init__(self, config):
        self.config = config
        self.peers = {}
        self.message_handlers = {}
    
    async def broadcast_discovery(self, message):
        """Broadcast discovery message to mesh network"""
        # Simulate mesh broadcast
        print(f"📡 Broadcasting discovery: {message['device_id']}")
    
    async def send_to_peer(self, peer_id, message):
        """Send message to specific peer"""
        # Simulate peer-to-peer messaging
        print(f"📤 Sending to {peer_id}: {message['message_type']}")

class P2PNetworkSimulator:
    """Simulated P2P network for demonstration"""
    
    def __init__(self, config):
        self.config = config
        self.connected = False
    
    async def connect(self):
        """Connect to P2P network"""
        self.connected = True
        print("🔗 Connected to P2P network")
    
    async def share_data(self, data):
        """Share data through P2P network"""
        print(f"🌍 Sharing data globally: {data.get('data_type', 'unknown')}")

# Continue with peer discovery and health monitoring...

    async def _peer_discovery_loop(self):
        """Continuous peer discovery and network maintenance"""
        while self.network_active:
            try:
                # Broadcast discovery message
                discovery_message = {
                    "message_type": "collaboration_discovery",
                    "device_id": self.device_id,
                    "device_name": f"Furcate-{self.device_id[:8]}",
                    "role": self.role.value,
                    "capabilities": list(self.capabilities),
                    "specializations": self.specializations,
                    "location": self.core.config.device.location,
                    "trust_score": 1.0,  # Self-reported trust score
                    "processing_power": self._estimate_processing_power(),
                    "battery_level": self._get_battery_level(),
                    "network_quality": 0.9,  # Estimated network quality
                    "timestamp": datetime.now().isoformat()
                }
                
                # Broadcast through mesh network
                if self.mesh_network:
                    await self.mesh_network.broadcast_discovery(discovery_message)
                
                # Share through P2P network
                if self.p2p_network and self.p2p_network.connected:
                    await self.p2p_network.share_data(discovery_message)
                
                # Clean up stale peers
                await self._cleanup_stale_peers()
                
                # Log network status
                if len(self.peer_devices) > 0:
                    print(f"🔍 Network status: {len(self.peer_devices)} peers discovered")
                    for device_id, device in list(self.peer_devices.items())[:3]:
                        time_since_seen = (datetime.now() - device.last_seen).total_seconds()
                        print(f"   {device_id}: {device.role.value}, trust={device.trust_score:.2f}, seen={time_since_seen:.0f}s ago")
                
                await asyncio.sleep(self.discovery_interval)
                
            except Exception as e:
                print(f"Peer discovery error: {e}")
                await asyncio.sleep(30)  # Error recovery delay
    
    async def _health_monitoring_loop(self):
        """Monitor health of peer devices and network"""
        while self.network_active:
            try:
                current_time = datetime.now()
                
                # Update peer health scores
                for device_id, device in self.peer_devices.items():
                    try:
                        # Calculate time since last contact
                        time_since_contact = (current_time - device.last_seen).total_seconds()
                        
                        # Update trust score based on reliability
                        if time_since_contact < 300:  # Less than 5 minutes
                            device.trust_score = min(1.0, device.trust_score + 0.01)
                        elif time_since_contact < 3600:  # Less than 1 hour
                            device.trust_score = max(0.0, device.trust_score - 0.005)
                        else:  # More than 1 hour
                            device.trust_score = max(0.0, device.trust_score - 0.02)
                        
                        # Update network quality based on response times
                        device.network_quality = self._calculate_network_quality(device_id)
                        
                    except Exception as e:
                        print(f"Error updating peer {device_id} health: {e}")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                print(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _calculate_network_quality(self, device_id: str) -> float:
        """Calculate network quality score for a peer device"""
        # Simplified network quality calculation
        # In real implementation, this would measure actual network metrics
        base_quality = 0.8
        
        # Adjust based on device type and distance
        device = self.peer_devices.get(device_id)
        if device:
            # Consider distance (if location available)
            if device.location and hasattr(self.core.config.device, 'location'):
                distance = self._calculate_distance(
                    device.location, 
                    self.core.config.device.location
                )
                distance_factor = max(0.1, 1.0 - (distance / 1000))  # 1km = no penalty
                base_quality *= distance_factor
            
            # Consider device capabilities
            if 'long_range_communication' in device.capabilities:
                base_quality *= 1.2
        
        return min(1.0, base_quality)
    
    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate distance between two locations in meters"""
        try:
            lat1, lon1 = loc1.get('latitude', 0), loc1.get('longitude', 0)
            lat2, lon2 = loc2.get('latitude', 0), loc2.get('longitude', 0)
            
            # Haversine formula for distance
            from math import radians, sin, cos, sqrt, atan2
            
            R = 6371000  # Earth's radius in meters
            lat1_rad, lon1_rad = radians(lat1), radians(lon1)
            lat2_rad, lon2_rad = radians(lat2), radians(lon2)
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            return R * c
            
        except Exception:
            return 0.0
    
    async def _cleanup_stale_peers(self):
        """Remove peers that haven't been seen recently"""
        stale_threshold = timedelta(hours=2)
        current_time = datetime.now()
        
        stale_peers = [
            device_id for device_id, device in self.peer_devices.items()
            if (current_time - device.last_seen) > stale_threshold
        ]
        
        for device_id in stale_peers:
            del self.peer_devices[device_id]
            print(f"Removed stale peer: {device_id}")
    
    async def handle_peer_discovery(self, discovery_message: Dict[str, Any]):
        """Handle incoming peer discovery messages"""
        try:
            device_id = discovery_message.get("device_id")
            if device_id == self.device_id:
                return  # Ignore self-discovery
            
            # Create or update peer device
            peer_device = CollaborativeDevice(
                device_id=device_id,
                device_name=discovery_message.get("device_name", device_id),
                role=CollaborationRole(discovery_message.get("role", "worker")),
                capabilities=set(discovery_message.get("capabilities", [])),
                location=discovery_message.get("location", {}),
                last_seen=datetime.now(),
                trust_score=discovery_message.get("trust_score", 0.5),
                processing_power=discovery_message.get("processing_power", 1.0),
                battery_level=discovery_message.get("battery_level", 100.0),
                network_quality=discovery_message.get("network_quality", 0.8),
                specializations=discovery_message.get("specializations", [])
            )
            
            self.peer_devices[device_id] = peer_device
            
            # Send discovery response
            await self._send_discovery_response(device_id)
            
            print(f"Discovered new peer: {device_id} ({peer_device.role.value})")
            
        except Exception as e:
            print(f"Error handling peer discovery: {e}")
    
    async def _send_discovery_response(self, target_device_id: str):
        """Send discovery response to a specific peer"""
        response_message = {
            "message_type": "collaboration_discovery_response",
            "device_id": self.device_id,
            "role": self.role.value,
            "capabilities": list(self.capabilities),
            "specializations": self.specializations,
            "location": self.core.config.device.location,
            "trust_score": 1.0,
            "processing_power": self._estimate_processing_power(),
            "battery_level": self._get_battery_level(),
            "network_quality": 0.9,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send response through available networks
        await self._send_to_peer(target_device_id, response_message)
    
    def _estimate_processing_power(self) -> float:
        """Estimate relative processing power of this device"""
        # Simple heuristic based on platform
        if hasattr(self.core, 'hardware'):
            platform = self.core.hardware.platform_info.get("platform", "unknown")
            if "jetson" in platform.lower():
                return 2.0  # High processing power
            elif "pi5" in platform.lower():
                return 1.5  # Medium-high processing
            elif "pi4" in platform.lower():
                return 1.0  # Standard processing
            else:
                return 0.8  # Lower processing
        return 1.0
    
    def _get_battery_level(self) -> float:
        """Get current battery level (if available)"""
        if hasattr(self.core, 'power') and hasattr(self.core.power, 'battery_level'):
            return self.core.power.battery_level
        return 100.0  # Assume AC power
    
    async def _send_to_peer(self, peer_id: str, message: Dict):
        """Send message to specific peer through available networks"""
        try:
            # Try mesh network first
            if self.mesh_network:
                await self.mesh_network.send_to_peer(peer_id, message)
            
            # Fallback to P2P network
            elif self.p2p_network and self.p2p_network.connected:
                await self.p2p_network.share_data({
                    "target_peer": peer_id,
                    "message": message
                })
            
        except Exception as e:
            print(f"Failed to send message to peer {peer_id}: {e}")
```

## Part 2: Distributed Data Fusion

### Multi-Source Environmental Data Integration

```python
class DistributedDataFusion:
    """Fuse environmental data from multiple collaborative devices"""
    
    def __init__(self, collaboration_manager):
        self.collaboration_manager = collaboration_manager
        self.fusion_algorithms = {
            "weighted_average": self._weighted_average_fusion,
            "kalman_filter": self._kalman_filter_fusion,
            "consensus": self._consensus_fusion,
            "ml_ensemble": self._ml_ensemble_fusion
        }
        
        # Fusion state
        self.fusion_history = {}
        self.sensor_weights = {}
        self.quality_metrics = {}
        
    async def create_environmental_consensus(self, data_type: str, timeout_seconds: int = 30):
        """Create environmental consensus from multiple devices"""
        try:
            # Request data from all capable peers
            request_id = str(uuid.uuid4())
            data_request = {
                "message_type": "data_fusion_request",
                "request_id": request_id,
                "data_type": data_type,
                "requester_id": self.collaboration_manager.device_id,
                "timeout": timeout_seconds,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send request to all peers with relevant capabilities
            capable_peers = self._find_capable_peers(data_type)
            responses = []
            
            for peer_id in capable_peers:
                try:
                    await self.collaboration_manager._send_to_peer(peer_id, data_request)
                except Exception as e:
                    print(f"Failed to request data from {peer_id}: {e}")
            
            # Collect responses with timeout
            responses = await self._collect_data_responses(request_id, timeout_seconds)
            
            # Include local data
            local_data = await self._get_local_sensor_data(data_type)
            if local_data:
                responses.append({
                    "device_id": self.collaboration_manager.device_id,
                    "data": local_data,
                    "quality": self._assess_local_data_quality(local_data),
                    "trust_score": 1.0,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Perform data fusion
            if len(responses) >= 2:
                consensus_result = await self._fuse_environmental_data(responses, data_type)
                
                # Store consensus for future reference
                self.collaboration_manager.consensus_data[data_type] = consensus_result
                
                return consensus_result
            else:
                print(f"Insufficient responses for {data_type} consensus: {len(responses)}")
                return local_data
                
        except Exception as e:
            print(f"Environmental consensus creation failed: {e}")
            return None
    
    def _find_capable_peers(self, data_type: str) -> List[str]:
        """Find peers capable of providing specific data type"""
        capable_peers = []
        
        for device_id, device in self.collaboration_manager.peer_devices.items():
            # Check if peer has relevant capabilities
            if data_type == "air_quality" and "air_quality_monitoring" in device.specializations:
                capable_peers.append(device_id)
            elif data_type == "temperature" and "sensor_monitoring" in device.capabilities:
                capable_peers.append(device_id)
            elif data_type == "environmental_analysis" and "edge_ml" in device.capabilities:
                capable_peers.append(device_id)
            elif "multi_sensor_fusion" in device.capabilities:
                capable_peers.append(device_id)
        
        # Filter by trust score
        capable_peers = [
            peer_id for peer_id in capable_peers
            if self.collaboration_manager.peer_devices[peer_id].trust_score >= 
               self.collaboration_manager.trust_threshold
        ]
        
        return capable_peers
    
    async def _collect_data_responses(self, request_id: str, timeout_seconds: int) -> List[Dict]:
        """Collect data responses from peers"""
        responses = []
        start_time = time.time()
        
        # In a real implementation, this would listen for incoming responses
        # For this tutorial, we'll simulate responses
        await asyncio.sleep(min(timeout_seconds, 5))
        
        # Simulate responses from collaborative peers
        for device_id, device in list(self.collaboration_manager.peer_devices.items())[:3]:
            simulated_response = self._simulate_peer_response(device_id, device)
            if simulated_response:
                responses.append(simulated_response)
        
        return responses
    
    def _simulate_peer_response(self, device_id: str, device: CollaborativeDevice) -> Dict:
        """Simulate peer data response for demonstration"""
        # Generate realistic environmental data based on device specializations
        simulated_data = {}
        
        if "air_quality_monitoring" in device.specializations:
            simulated_data["air_quality"] = {
                "aqi": np.random.normal(75, 20),
                "pm2_5": np.random.normal(15, 5),
                "pm10": np.random.normal(25, 8)
            }
        
        if "sensor_monitoring" in device.capabilities:
            # Add location-based variation
            lat_variation = device.location.get("latitude", 0) * 0.1
            simulated_data.update({
                "temperature": np.random.normal(22 + lat_variation, 3),
                "humidity": np.random.normal(50, 10),
                "pressure": np.random.normal(1013, 15)
            })
        
        return {
            "device_id": device_id,
            "data": simulated_data,
            "quality": device.trust_score * np.random.uniform(0.8, 1.0),
            "trust_score": device.trust_score,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_local_sensor_data(self, data_type: str) -> Dict:
        """Get local sensor data for fusion"""
        try:
            if hasattr(self.collaboration_manager.core, 'hardware'):
                return await self.collaboration_manager.core.hardware.read_all_sensors()
            return {}
        except Exception as e:
            print(f"Error getting local sensor data: {e}")
            return {}
    
    def _assess_local_data_quality(self, data: Dict) -> float:
        """Assess quality of local sensor data"""
        if not data:
            return 0.0
        
        quality_scores = []
        for sensor_name, sensor_data in data.items():
            if isinstance(sensor_data, dict) and 'quality' in sensor_data:
                quality_scores.append(sensor_data['quality'])
            else:
                quality_scores.append(0.8)  # Default quality
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    async def _fuse_environmental_data(self, responses: List[Dict], data_type: str) -> Dict:
        """Fuse environmental data from multiple sources"""
        try:
            # Choose fusion algorithm based on data type and response count
            if len(responses) <= 3:
                algorithm = "weighted_average"
            elif data_type in ["air_quality", "temperature"]:
                algorithm = "kalman_filter"
            else:
                algorithm = "consensus"
            
            fusion_function = self.fusion_algorithms[algorithm]
            fused_result = fusion_function(responses, data_type)
            
            # Add metadata
            fused_result["fusion_metadata"] = {
                "algorithm": algorithm,
                "source_count": len(responses),
                "fusion_timestamp": datetime.now().isoformat(),
                "confidence": self._calculate_fusion_confidence(responses),
                "source_devices": [r["device_id"] for r in responses]
            }
            
            return fused_result
            
        except Exception as e:
            print(f"Data fusion failed: {e}")
            return responses[0]["data"] if responses else {}
    
    def _weighted_average_fusion(self, responses: List[Dict], data_type: str) -> Dict:
        """Weighted average fusion based on trust scores and data quality"""
        fused_data = {}
        
        # Extract all unique data keys
        all_keys = set()
        for response in responses:
            if isinstance(response["data"], dict):
                all_keys.update(response["data"].keys())
        
        for key in all_keys:
            values = []
            weights = []
            
            for response in responses:
                if key in response["data"]:
                    value = response["data"][key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        # Weight combines trust score and data quality
                        weight = response["trust_score"] * response["quality"]
                        weights.append(weight)
                    elif isinstance(value, dict):
                        # Handle nested data structures
                        if key not in fused_data:
                            fused_data[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                if subkey not in fused_data[key]:
                                    fused_data[key][subkey] = []
                                fused_data[key][subkey].append((subvalue, response["trust_score"] * response["quality"]))
            
            if values and weights:
                # Calculate weighted average
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                total_weight = sum(weights)
                fused_data[key] = weighted_sum / total_weight if total_weight > 0 else np.mean(values)
        
        # Process nested dictionaries
        for key, value in fused_data.items():
            if isinstance(value, dict):
                for subkey, subvalues in value.items():
                    if isinstance(subvalues, list):
                        values, weights = zip(*subvalues)
                        weighted_sum = sum(v * w for v, w in zip(values, weights))
                        total_weight = sum(weights)
                        fused_data[key][subkey] = weighted_sum / total_weight if total_weight > 0 else np.mean(values)
        
        return fused_data
    
    def _kalman_filter_fusion(self, responses: List[Dict], data_type: str) -> Dict:
        """Kalman filter-based fusion for time-series data"""
        # Simplified Kalman filter implementation
        fused_data = {}
        
        for key in ["temperature", "humidity", "pressure", "air_quality"]:
            values = []
            uncertainties = []
            
            for response in responses:
                if key in response["data"]:
                    value = response["data"][key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        # Uncertainty inversely related to trust and quality
                        uncertainty = 1.0 / (response["trust_score"] * response["quality"] + 0.1)
                        uncertainties.append(uncertainty)
            
            if len(values) >= 2:
                # Simple Kalman-like fusion
                weights = [1/u for u in uncertainties]
                total_weight = sum(weights)
                fused_value = sum(v * w for v, w in zip(values, weights)) / total_weight
                fused_uncertainty = 1.0 / total_weight
                
                fused_data[key] = {
                    "value": fused_value,
                    "uncertainty": fused_uncertainty,
                    "confidence": min(1.0, total_weight / len(values))
                }
            elif values:
                fused_data[key] = values[0]
        
        return fused_data
    
    def _consensus_fusion(self, responses: List[Dict], data_type: str) -> Dict:
        """Consensus-based fusion using majority agreement"""
        fused_data = {}
        
        # For each data key, find consensus among responses
        all_keys = set()
        for response in responses:
            if isinstance(response["data"], dict):
                all_keys.update(response["data"].keys())
        
        for key in all_keys:
            values = []
            for response in responses:
                if key in response["data"]:
                    value = response["data"][key]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if len(values) >= 3:
                # Remove outliers and calculate consensus
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                consensus_values = [v for v in values if lower_bound <= v <= upper_bound]
                if consensus_values:
                    fused_data[key] = {
                        "consensus_value": np.median(consensus_values),
                        "agreement_ratio": len(consensus_values) / len(values),
                        "range": [min(consensus_values), max(consensus_values)]
                    }
                else:
                    fused_data[key] = np.median(values)
            elif values:
                fused_data[key] = np.median(values)
        
        return fused_data
    
    def _ml_ensemble_fusion(self, responses: List[Dict], data_type: str) -> Dict:
        """ML ensemble-based fusion"""
        # This would use trained ensemble models to fuse data
        # For simplicity, falling back to weighted average
        return self._weighted_average_fusion(responses, data_type)
    
    def _calculate_fusion_confidence(self, responses: List[Dict]) -> float:
        """Calculate confidence in fused result"""
        if not responses:
            return 0.0
        
        # Confidence based on number of sources and their quality
        source_count_factor = min(1.0, len(responses) / 5.0)  # Optimal at 5 sources
        
        avg_trust = np.mean([r["trust_score"] for r in responses])
        avg_quality = np.mean([r["quality"] for r in responses])
        
        confidence = source_count_factor * 0.4 + avg_trust * 0.3 + avg_quality * 0.3
        return min(1.0, confidence)
```

## Part 3: Collaborative ML and Model Sharing

### Distributed Model Training and Inference

```python
class CollaborativeMLSystem:
    """Collaborative machine learning across multiple devices"""
    
    def __init__(self, collaboration_manager):
        self.collaboration_manager = collaboration_manager
        self.shared_models = {}
        self.model_versions = {}
        self.training_sessions = {}
        self.inference_cache = {}
        
    async def initiate_collaborative_training(self, model_type: str, training_config: Dict):
        """Initiate collaborative model training session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create training session
            training_session = {
                "session_id": session_id,
                "model_type": model_type,
                "coordinator": self.collaboration_manager.device_id,
                "participants": [],
                "status": "recruiting",
                "config": training_config,
                "started_at": datetime.now(),
                "data_requirements": training_config.get("data_requirements", {}),
                "min_participants": training_config.get("min_participants", 3),
                "max_participants": training_config.get("max_participants", 10)
            }
            
            self.training_sessions[session_id] = training_session
            
            # Send recruitment message to suitable peers
            recruitment_message = {
                "message_type": "collaborative_training_invitation",
                "session_id": session_id,
                "model_type": model_type,
                "coordinator": self.collaboration_manager.device_id,
                "requirements": training_config.get("requirements", {}),
                "estimated_duration": training_config.get("duration_hours", 2),
                "data_requirements": training_config.get("data_requirements", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Find suitable training partners
            suitable_peers = self._find_training_partners(training_config)
            
            for peer_id in suitable_peers:
                await self.collaboration_manager._send_to_peer(peer_id, recruitment_message)
            
            print(f"Initiated collaborative training session: {session_id}")
            print(f"Invited {len(suitable_peers)} suitable peers")
            
            # Wait for responses and start training
            await self._coordinate_training_session(session_id)
            
            return session_id
            
        except Exception as e:
            print(f"Failed to initiate collaborative training: {e}")
            return None
    
    def _find_training_partners(self, training_config: Dict) -> List[str]:
        """Find suitable partners for collaborative training"""
        suitable_peers = []
        
        required_capabilities = training_config.get("required_capabilities", ["edge_ml"])
        min_processing_power = training_config.get("min_processing_power", 1.0)
        min_trust_score = training_config.get("min_trust_score", 0.8)
        
        for device_id, device in self.collaboration_manager.peer_devices.items():
            # Check capabilities
            if not all(cap in device.capabilities for cap in required_capabilities):
                continue
            
            # Check processing power
            if device.processing_power < min_processing_power:
                continue
            
            # Check trust score
            if device.trust_score < min_trust_score:
                continue
            
            # Check battery level for long training sessions
            if training_config.get("duration_hours", 1) > 2 and device.battery_level < 50:
                continue
            
            suitable_peers.append(device_id)
        
        # Sort by processing power and trust score
        suitable_peers.sort(
            key=lambda peer_id: (
                self.collaboration_manager.peer_devices[peer_id].processing_power *
                self.collaboration_manager.peer_devices[peer_id].trust_score
            ),
            reverse=True
        )
        
        return suitable_peers
    
    async def _coordinate_training_session(self, session_id: str):
        """Coordinate collaborative training session"""
        session = self.training_sessions[session_id]
        
        # Wait for participant responses (simplified)
        await asyncio.sleep(30)  # Wait 30 seconds for responses
        
        # Check if we have enough participants
        if len(session["participants"]) >= session["min_participants"]:
            print(f"Starting training with {len(session['participants'])} participants")
            await self._execute_federated_training(session_id)
        else:
            print(f"Insufficient participants for training: {len(session['participants'])}")
            session["status"] = "cancelled"
    
    async def _execute_federated_training(self, session_id: str):
        """Execute federated learning training"""
        session = self.training_sessions[session_id]
        
        try:
            # Phase 1: Data preparation
            await self._coordinate_data_preparation(session_id)
            
            # Phase 2: Distributed training rounds
            num_rounds = session["config"].get("training_rounds", 5)
            
            for round_num in range(num_rounds):
                print(f"Starting training round {round_num + 1}/{num_rounds}")
                
                # Send current global model to participants
                await self._distribute_global_model(session_id)
                
                # Coordinate local training
                await self._coordinate_local_training(session_id, round_num)
                
                # Aggregate model updates
                await self._aggregate_model_updates(session_id)
                
                # Evaluate global model
                global_performance = await self._evaluate_global_model(session_id)
                print(f"Round {round_num + 1} performance: {global_performance}")
            
            # Phase 3: Finalize and distribute trained model
            await self._finalize_collaborative_model(session_id)
            
            session["status"] = "completed"
            print(f"Collaborative training completed for session {session_id}")
            
        except Exception as e:
            print(f"Federated training failed: {e}")
            session["status"] = "failed"
    
    async def handle_training_invitation(self, invitation: Dict):
        """Handle incoming collaborative training invitation"""
        try:
            session_id = invitation["session_id"]
            model_type = invitation["model_type"]
            requirements = invitation.get("requirements", {})
            
            # Evaluate if we can participate
            can_participate = self._evaluate_training_participation(requirements)
            
            response_message = {
                "message_type": "collaborative_training_response",
                "session_id": session_id,
                "participant_id": self.collaboration_manager.device_id,
                "accepted": can_participate,
                "capabilities": list(self.collaboration_manager.capabilities),
                "processing_power": self.collaboration_manager._estimate_processing_power(),
                "available_data": self._estimate_available_data(model_type),
                "timestamp": datetime.now().isoformat()
            }
            
            # Send response to coordinator
            coordinator_id = invitation["coordinator"]
            await self.collaboration_manager._send_to_peer(coordinator_id, response_message)
            
            if can_participate:
                print(f"Accepted training invitation for session {session_id}")
            else:
                print(f"Declined training invitation for session {session_id}")
            
        except Exception as e:
            print(f"Error handling training invitation: {e}")
    
    def _evaluate_training_participation(self, requirements: Dict) -> bool:
        """Evaluate whether device can participate in training"""
        # Check processing power
        min_processing = requirements.get("min_processing_power", 1.0)
        if self.collaboration_manager._estimate_processing_power() < min_processing:
            return False
        
        # Check battery level
        min_battery = requirements.get("min_battery_level", 30.0)
        if self.collaboration_manager._get_battery_level() < min_battery:
            return False
        
        # Check available capabilities
        required_caps = requirements.get("required_capabilities", [])
        if not all(cap in self.collaboration_manager.capabilities for cap in required_caps):
            return False
        
        # Check data availability
        data_requirements = requirements.get("data_requirements", {})
        if not self._check_data_availability(data_requirements):
            return False
        
        return True
    
    def _check_data_availability(self, data_requirements: Dict) -> bool:
        """Check if device has required data for training"""
        # Simple check - would be more sophisticated in real implementation
        required_sensors = data_requirements.get("sensors", [])
        
        if hasattr(self.collaboration_manager.core, 'hardware'):
            available_sensors = list(self.collaboration_manager.core.hardware.sensors.keys())
            return all(sensor in available_sensors for sensor in required_sensors)
        
        return True  # Assume data is available
    
    def _estimate_available_data(self, model_type: str) -> Dict:
        """Estimate amount of data available for training"""
        # Simplified estimation
        return {
            "records": 1000,  # Number of data records
            "time_range_days": 30,  # Days of data
            "sensors": ["temperature", "humidity", "air_quality"],
            "quality_score": 0.8
        }
    
    async def share_model_insights(self, model_results: Dict, target_peers: List[str] = None):
        """Share ML model insights with collaborative peers"""
        try:
            insight_message = {
                "message_type": "model_insights_sharing",
                "sender_id": self.collaboration_manager.device_id,
                "model_results": model_results,
                "confidence": model_results.get("confidence", 0.8),
                "model_version": self._get_model_version(),
                "applicable_conditions": self._get_applicable_conditions(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Determine target peers
            if target_peers is None:
                target_peers = list(self.collaboration_manager.peer_devices.keys())
            
            # Send insights to peers
            for peer_id in target_peers:
                try:
                    await self.collaboration_manager._send_to_peer(peer_id, insight_message)
                except Exception as e:
                    print(f"Failed to share insights with {peer_id}: {e}")
            
            print(f"Shared model insights with {len(target_peers)} peers")
            
        except Exception as e:
            print(f"Model insight sharing failed: {e}")
    
    async def handle_model_insights(self, insights_message: Dict):
        """Handle incoming model insights from peers"""
        try:
            sender_id = insights_message["sender_id"]
            model_results = insights_message["model_results"]
            confidence = insights_message["confidence"]
            
            # Validate sender trust
            sender_device = self.collaboration_manager.peer_devices.get(sender_id)
            if not sender_device or sender_device.trust_score < 0.7:
                print(f"Ignored insights from untrusted peer: {sender_id}")
                return
            
            # Store insights for ensemble predictions
            insight_key = f"{sender_id}_{datetime.now().strftime('%Y%m%d%H')}"
            self.inference_cache[insight_key] = {
                "sender_id": sender_id,
                "results": model_results,
                "confidence": confidence,
                "received_at": datetime.now(),
                "trust_score": sender_device.trust_score
            }
            
            # Clean old insights
            await self._cleanup_old_insights()
            
            print(f"Received model insights from {sender_id} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"Error handling model insights: {e}")
    
    async def _cleanup_old_insights(self):
        """Clean up old insights from cache"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        old_keys = [
            key for key, insight in self.inference_cache.items()
            if insight["received_at"] < cutoff_time
        ]
        
        for key in old_keys:
            del self.inference_cache[key]
    
    async def ensemble_prediction(self, sensor_data: Dict) -> Dict:
        """Create ensemble prediction using insights from multiple devices"""
        try:
            # Get local prediction
            local_prediction = await self._get_local_prediction(sensor_data)
            
            # Collect relevant peer insights
            peer_predictions = []
            for insight in self.inference_cache.values():
                # Check if insight is relevant to current conditions
                if self._is_insight_relevant(insight, sensor_data):
                    peer_predictions.append(insight)
            
            if not peer_predictions:
                return local_prediction
            
            # Combine predictions using weighted ensemble
            ensemble_result = self._create_ensemble_prediction(
                local_prediction, peer_predictions
            )
            
            return ensemble_result
            
        except Exception as e:
            print(f"Ensemble prediction failed: {e}")
            return await self._get_local_prediction(sensor_data)
    
    def _is_insight_relevant(self, insight: Dict, current_data: Dict) -> bool:
        """Check if peer insight is relevant to current conditions"""
        # Simple relevance check - could be more sophisticated
        insight_age = (datetime.now() - insight["received_at"]).total_seconds()
        
        # Insights older than 1 hour are less relevant
        if insight_age > 3600:
            return False
        
        # Trust score must be above threshold
        if insight["trust_score"] < 0.7:
            return False
        
        return True
    
    def _create_ensemble_prediction(self, local_prediction: Dict, peer_predictions: List[Dict]) -> Dict:
        """Create ensemble prediction from local and peer predictions"""
        all_predictions = [local_prediction] + [p["results"] for p in peer_predictions]
        all_confidences = [local_prediction.get("confidence", 0.8)] + [p["confidence"] for p in peer_predictions]
        all_weights = [1.0] + [p["trust_score"] for p in peer_predictions]
        
        # Weighted ensemble for classification
        if "environmental_class" in local_prediction:
            class_votes = {}
            total_weight = 0
            
            for pred, conf, weight in zip(all_predictions, all_confidences, all_weights):
                effective_weight = weight * conf
                env_class = pred.get("environmental_class", "unknown")
                
                if env_class not in class_votes:
                    class_votes[env_class] = 0
                class_votes[env_class] += effective_weight
                total_weight += effective_weight
            
            # Normalize votes
            for class_name in class_votes:
                class_votes[class_name] /= total_weight
            
            # Select class with highest vote
            best_class = max(class_votes, key=class_votes.get)
            ensemble_confidence = class_votes[best_class]
            
            ensemble_result = {
                "environmental_class": best_class,
                "confidence": ensemble_confidence,
                "ensemble_votes": class_votes,
                "source_count": len(all_predictions),
                "ensemble_method": "weighted_voting"
            }
        else:
            # Weighted average for regression
            ensemble_result = local_prediction.copy()
            ensemble_result["ensemble_method"] = "weighted_average"
            ensemble_result["source_count"] = len(all_predictions)
        
        return ensemble_result
    
    async def _get_local_prediction(self, sensor_data: Dict) -> Dict:
        """Get prediction from local ML model"""
        try:
            if hasattr(self.collaboration_manager.core, 'edge_ml'):
                return await self.collaboration_manager.core.edge_ml.process_sensor_data(sensor_data)
            else:
                # Fallback prediction
                return {
                    "environmental_class": "moderate",
                    "confidence": 0.6,
                    "source": "fallback"
                }
        except Exception as e:
            print(f"Local prediction failed: {e}")
            return {"environmental_class": "unknown", "confidence": 0.0}
    
    def _get_model_version(self) -> str:
        """Get current model version"""
        if hasattr(self.collaboration_manager.core, 'edge_ml'):
            return getattr(self.collaboration_manager.core.edge_ml, 'model_version', '1.0.0')
        return '1.0.0'
    
    def _get_applicable_conditions(self) -> Dict:
        """Get conditions where current model is applicable"""
        return {
            "temperature_range": [-10, 50],
            "humidity_range": [0, 100],
            "location_type": "general",
            "season": "all"
        }

    # Placeholder methods for federated learning implementation
    async def _coordinate_data_preparation(self, session_id: str):
        """Coordinate data preparation across participants"""
        print(f"Coordinating data preparation for session {session_id}")
    
    async def _distribute_global_model(self, session_id: str):
        """Distribute global model to participants"""
        print(f"Distributing global model for session {session_id}")
    
    async def _coordinate_local_training(self, session_id: str, round_num: int):
        """Coordinate local training round"""
        print(f"Coordinating local training round {round_num} for session {session_id}")
    
    async def _aggregate_model_updates(self, session_id: str):
        """Aggregate model updates from participants"""
        print(f"Aggregating model updates for session {session_id}")
    
    async def _evaluate_global_model(self, session_id: str) -> Dict:
        """Evaluate global model performance"""
        return {"accuracy": 0.85, "loss": 0.15}
    
    async def _finalize_collaborative_model(self, session_id: str):
        """Finalize and distribute trained model"""
        print(f"Finalizing collaborative model for session {session_id}")
```

## Part 4: Emergency Response and Alert Propagation

### Distributed Emergency Response System

```python
class CollaborativeEmergencySystem:
    """Collaborative emergency response and alert propagation"""
    
    def __init__(self, collaboration_manager):
        self.collaboration_manager = collaboration_manager
        self.emergency_alerts = {}
        self.response_protocols = {}
        self.evacuation_routes = {}
        self.emergency_contacts = {}
        
        # Emergency severity levels
        self.severity_levels = {
            "INFO": 1,
            "WARNING": 2,
            "ALERT": 3,
            "EMERGENCY": 4,
            "CRITICAL": 5
        }
        
    async def detect_emergency_condition(self, sensor_data: Dict, ml_results: Dict):
        """Detect emergency conditions from sensor data and ML results"""
        try:
            emergency_conditions = []
            
            # Environmental threshold emergencies
            emergency_conditions.extend(self._check_environmental_thresholds(sensor_data))
            
            # ML-based anomaly emergencies
            emergency_conditions.extend(self._check_ml_anomalies(ml_results))
            
            # Multi-sensor correlation emergencies
            emergency_conditions.extend(self._check_correlation_emergencies(sensor_data))
            
            # Process detected emergencies
            for condition in emergency_conditions:
                await self._handle_emergency_condition(condition)
            
            return emergency_conditions
            
        except Exception as e:
            print(f"Emergency detection failed: {e}")
            return []
    
    def _check_environmental_thresholds(self, sensor_data: Dict) -> List[Dict]:
        """Check for environmental threshold emergencies"""
        emergencies = []
        
        # Air quality emergencies
        air_quality = sensor_data.get("air_quality", {})
        if isinstance(air_quality, dict):
            aqi = air_quality.get("aqi", 0)
            if aqi > 300:
                emergencies.append({
                    "type": "air_quality_critical",
                    "severity": "CRITICAL",
                    "value": aqi,
                    "threshold": 300,
                    "message": f"Hazardous air quality detected: AQI {aqi}",
                    "immediate_action": "Seek indoor shelter immediately"
                })
            elif aqi > 200:
                emergencies.append({
                    "type": "air_quality_emergency",
                    "severity": "EMERGENCY",
                    "value": aqi,
                    "threshold": 200,
                    "message": f"Very unhealthy air quality: AQI {aqi}",
                    "immediate_action": "Limit outdoor activities"
                })
        
        # Temperature emergencies
        temperature = sensor_data.get("temperature", 20)
        if temperature > 45:
            emergencies.append({
                "type": "extreme_heat",
                "severity": "EMERGENCY",
                "value": temperature,
                "threshold": 45,
                "message": f"Extreme heat warning: {temperature}°C",
                "immediate_action": "Seek air conditioning, avoid outdoor activities"
            })
        elif temperature < -20:
            emergencies.append({
                "type": "extreme_cold",
                "severity": "EMERGENCY",
                "value": temperature,
                "threshold": -20,
                "message": f"Extreme cold warning: {temperature}°C",
                "immediate_action": "Seek warm shelter immediately"
            })
        
        return emergencies
    
    def _check_ml_anomalies(self, ml_results: Dict) -> List[Dict]:
        """Check for ML-detected anomalies that indicate emergencies"""
        emergencies = []
        
        # High confidence anomalies
        if ml_results.get("is_anomaly", False):
            anomaly_prob = ml_results.get("anomaly_probability", 0)
            if anomaly_prob > 0.9:
                emergencies.append({
                    "type": "environmental_anomaly",
                    "severity": "ALERT",
                    "value": anomaly_prob,
                    "threshold": 0.9,
                    "message": f"High confidence environmental anomaly detected: {anomaly_prob:.2f}",
                    "immediate_action": "Monitor conditions closely, prepare for potential emergency"
                })
        
        return emergencies
    
    def _check_correlation_emergencies(self, sensor_data: Dict) -> List[Dict]:
        """Check for emergencies based on sensor correlation patterns"""
        emergencies = []
        
        # Rapid temperature rise with low humidity (possible fire)
        temperature = sensor_data.get("temperature", 20)
        humidity = sensor_data.get("humidity", 50)
        
        if temperature > 35 and humidity < 20:
            emergencies.append({
                "type": "fire_risk",
                "severity": "WARNING",
                "message": "Fire risk conditions: High temperature, low humidity",
                "immediate_action": "Monitor for smoke, prepare evacuation plan"
            })
        
        return emergencies
    
    async def _handle_emergency_condition(self, condition: Dict):
        """Handle detected emergency condition"""
        try:
            emergency_id = str(uuid.uuid4())
            
            # Create emergency alert
            emergency_alert = {
                "emergency_id": emergency_id,
                "type": condition["type"],
                "severity": condition["severity"],
                "message": condition["message"],
                "immediate_action": condition.get("immediate_action", ""),
                "detected_by": self.collaboration_manager.device_id,
                "location": self.collaboration_manager.core.config.device.location,
                "detected_at": datetime.now().isoformat(),
                "status": "active",
                "propagation_hops": 0,
                "responding_devices": []
            }
            
            self.emergency_alerts[emergency_id] = emergency_alert
            
            # Propagate emergency alert
            await self._propagate_emergency_alert(emergency_alert)
            
            # Execute local emergency response
            await self._execute_local_emergency_response(emergency_alert)
            
            print(f"Emergency detected and handled: {condition['type']} - {condition['message']}")
            
        except Exception as e:
            print(f"Emergency handling failed: {e}")
    
    async def _propagate_emergency_alert(self, emergency_alert: Dict):
        """Propagate emergency alert to collaborative network"""
        try:
            alert_message = {
                "message_type": "emergency_alert",
                "priority": DataPriority.EMERGENCY.value,
                "emergency_alert": emergency_alert,
                "propagation_path": [self.collaboration_manager.device_id],
                "max_hops": self.collaboration_manager.emergency_propagation_hops,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all trusted peers
            trusted_peers = [
                device_id for device_id, device in self.collaboration_manager.peer_devices.items()
                if device.trust_score >= 0.8
            ]
            
            for peer_id in trusted_peers:
                try:
                    await self.collaboration_manager._send_to_peer(peer_id, alert_message)
                except Exception as e:
                    print(f"Failed to send emergency alert to {peer_id}: {e}")
            
            # Also send through global networks
            if hasattr(self.collaboration_manager.core, 'tenzro_client'):
                await self.collaboration_manager.core.tenzro_client.send_alert(emergency_alert)
            
            print(f"Emergency alert propagated to {len(trusted_peers)} peers")
            
        except Exception as e:
            print(f"Emergency alert propagation failed: {e}")
    
    async def handle_emergency_alert(self, alert_message: Dict):
        """Handle incoming emergency alert from peer"""
        try:
            emergency_alert = alert_message["emergency_alert"]
            propagation_path = alert_message.get("propagation_path", [])
            max_hops = alert_message.get("max_hops", 5)
            
            emergency_id = emergency_alert["emergency_id"]
            
            # Check if we've already seen this alert
            if emergency_id in self.emergency_alerts:
                return
            
            # Store the alert
            self.emergency_alerts[emergency_id] = emergency_alert
            
            # Calculate distance from emergency location
            emergency_location = emergency_alert.get("location", {})
            device_location = self.collaboration_manager.core.config.device.location
            distance = self._calculate_distance(emergency_location, device_location)
            
            print(f"Received emergency alert: {emergency_alert['type']} - {emergency_alert['message']}")
            print(f"Distance from emergency: {distance:.1f} km")
            
            # Execute appropriate response based on distance and severity
            await self._execute_emergency_response(emergency_alert, distance)
            
            # Re-propagate if within hop limit and relevant
            if len(propagation_path) < max_hops and distance < 50:  # 50km radius
                await self._re_propagate_alert(alert_message, propagation_path)
            
        except Exception as e:
            print(f"Error handling emergency alert: {e}")
    
    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate distance between two locations in kilometers"""
        try:
            lat1 = loc1.get("latitude", 0)
            lon1 = loc1.get("longitude", 0)
            lat2 = loc2.get("latitude", 0)
            lon2 = loc2.get("longitude", 0)
            
            # Haversine formula for distance calculation
            from math import radians, sin, cos, sqrt, atan2
            
            R = 6371  # Earth's radius in kilometers
            
            lat1_rad = radians(lat1)
            lon1_rad = radians(lon1)
            lat2_rad = radians(lat2)
            lon2_rad = radians(lon2)
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            return distance
            
        except Exception as e:
            print(f"Distance calculation failed: {e}")
            return 0.0
    
    async def _execute_local_emergency_response(self, emergency_alert: Dict):
        """Execute local emergency response actions"""
        try:
            severity = emergency_alert["severity"]
            emergency_type = emergency_alert["type"]
            
            # Increase monitoring frequency
            if hasattr(self.collaboration_manager.core, 'hardware'):
                await self._increase_monitoring_frequency(severity)
            
            # Activate emergency sensors
            await self._activate_emergency_sensors(emergency_type)
            
            # Log emergency for authorities
            await self._log_emergency_for_authorities(emergency_alert)
            
            # Send notifications
            await self._send_emergency_notifications(emergency_alert)
            
        except Exception as e:
            print(f"Local emergency response failed: {e}")
    
    async def _execute_emergency_response(self, emergency_alert: Dict, distance: float):
        """Execute emergency response based on alert and distance"""
        try:
            severity_level = self.severity_levels.get(emergency_alert["severity"], 1)
            
            # Response actions based on distance and severity
            if distance < 1.0:  # Within 1km - immediate response
                await self._immediate_emergency_response(emergency_alert)
            elif distance < 10.0:  # Within 10km - monitoring response
                await self._monitoring_emergency_response(emergency_alert)
            elif distance < 50.0:  # Within 50km - awareness response
                await self._awareness_emergency_response(emergency_alert)
            
            # Update response status
            emergency_alert["responding_devices"].append({
                "device_id": self.collaboration_manager.device_id,
                "response_type": "monitoring",
                "distance_km": distance,
                "responded_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Emergency response execution failed: {e}")
    
    async def _immediate_emergency_response(self, emergency_alert: Dict):
        """Execute immediate emergency response for nearby emergencies"""
        print("⚠️  IMMEDIATE EMERGENCY RESPONSE ACTIVATED")
        
        # Maximum monitoring frequency
        if hasattr(self.collaboration_manager.core, 'monitoring'):
            self.collaboration_manager.core.monitoring.interval_seconds = 10
        
        # Activate all available sensors
        await self._activate_all_sensors()
        
        # Continuous data sharing
        await self._start_continuous_data_sharing()
        
        # Emergency beacon mode
        await self._activate_emergency_beacon()
    
    async def _monitoring_emergency_response(self, emergency_alert: Dict):
        """Execute monitoring response for regional emergencies"""
        print("📊 MONITORING EMERGENCY RESPONSE ACTIVATED")
        
        # Increased monitoring frequency
        if hasattr(self.collaboration_manager.core, 'monitoring'):
            self.collaboration_manager.core.monitoring.interval_seconds = 60
        
        # Enhanced data sharing
        await self._start_enhanced_data_sharing()
    
    async def _awareness_emergency_response(self, emergency_alert: Dict):
        """Execute awareness response for distant emergencies"""
        print("ℹ️  EMERGENCY AWARENESS MODE ACTIVATED")
        
        # Standard monitoring with emergency context
        if hasattr(self.collaboration_manager.core, 'monitoring'):
            self.collaboration_manager.core.monitoring.emergency_context = emergency_alert
    
    async def _re_propagate_alert(self, alert_message: Dict, propagation_path: List[str]):
        """Re-propagate emergency alert to extend reach"""
        try:
            # Add self to propagation path
            new_path = propagation_path + [self.collaboration_manager.device_id]
            alert_message["propagation_path"] = new_path
            
            # Send to peers not in propagation path
            for device_id, device in self.collaboration_manager.peer_devices.items():
                if device_id not in new_path and device.trust_score >= 0.8:
                    try:
                        await self.collaboration_manager._send_to_peer(device_id, alert_message)
                    except Exception as e:
                        print(f"Failed to re-propagate to {device_id}: {e}")
            
        except Exception as e:
            print(f"Alert re-propagation failed: {e}")

    # Placeholder methods for emergency response implementation
    async def _increase_monitoring_frequency(self, severity: str):
        """Increase monitoring frequency based on emergency severity"""
        frequency_map = {
            "INFO": 300,      # 5 minutes
            "WARNING": 180,   # 3 minutes
            "ALERT": 60,      # 1 minute
            "EMERGENCY": 30,  # 30 seconds
            "CRITICAL": 10    # 10 seconds
        }
        
        new_interval = frequency_map.get(severity, 300)
        print(f"Setting monitoring interval to {new_interval} seconds for {severity} alert")

    async def _activate_emergency_sensors(self, emergency_type: str):
        """Activate specific sensors based on emergency type"""
        print(f"Activating emergency sensors for {emergency_type}")

    async def _log_emergency_for_authorities(self, emergency_alert: Dict):
        """Log emergency for authorities and emergency services"""
        print(f"Logging emergency for authorities: {emergency_alert['type']}")

    async def _send_emergency_notifications(self, emergency_alert: Dict):
        """Send emergency notifications to stakeholders"""
        print(f"Sending emergency notifications: {emergency_alert['message']}")

    async def _activate_all_sensors(self):
        """Activate all available sensors for emergency monitoring"""
        print("Activating all sensors for emergency monitoring")

    async def _start_continuous_data_sharing(self):
        """Start continuous data sharing for emergency coordination"""
        print("Starting continuous data sharing")

    async def _activate_emergency_beacon(self):
        """Activate emergency beacon for location identification"""
        print("Activating emergency beacon")

    async def _start_enhanced_data_sharing(self):
        """Start enhanced data sharing for monitoring response"""
        print("Starting enhanced data sharing")
```

## Part 5: Integration and Deployment

### Integration with Furcate Nano Core

```python
# Integration example in furcate_nano/core.py

class FurcateNanoCore:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Initialize collaborative systems
        self.collaboration_manager = None
        self.data_fusion = None
        self.collaborative_ml = None
        self.emergency_system = None
        
        if config.collaboration.enabled:
            self._setup_collaboration()
    
    def _setup_collaboration(self):
        """Setup collaborative networking and systems"""
        try:
            # Determine device role based on configuration
            role_mapping = {
                "coordinator": CollaborationRole.COORDINATOR,
                "specialist": CollaborationRole.SPECIALIST,
                "relay": CollaborationRole.RELAY,
                "worker": CollaborationRole.WORKER,
                "emergency": CollaborationRole.EMERGENCY
            }
            
            device_role = role_mapping.get(
                self.config.collaboration.role, 
                CollaborationRole.WORKER
            )
            
            # Initialize collaboration manager
            self.collaboration_manager = CollaborativeNetworkManager(
                self, device_role
            )
            
            # Initialize collaborative systems
            self.data_fusion = DistributedDataFusion(self.collaboration_manager)
            self.collaborative_ml = CollaborativeMLSystem(self.collaboration_manager)
            self.emergency_system = CollaborativeEmergencySystem(self.collaboration_manager)
            
            print("✅ Collaborative systems initialized")
            
        except Exception as e:
            print(f"❌ Collaboration setup failed: {e}")
    
    async def start_monitoring(self):
        """Start monitoring with collaborative features"""
        # ... existing monitoring setup ...
        
        # Initialize collaboration if enabled
        if self.collaboration_manager:
            await self.collaboration_manager.initialize_collaboration()
        
        # Start collaborative monitoring loop
        if self.config.collaboration.enabled:
            asyncio.create_task(self._collaborative_monitoring_loop())
    
    async def _collaborative_monitoring_loop(self):
        """Enhanced monitoring loop with collaborative features"""
        while self.monitoring_active:
            try:
                # Standard sensor reading
                sensor_data = await self.hardware.read_all_sensors()
                
                # ML processing
                ml_results = await self.edge_ml.process_sensor_data(sensor_data)
                
                # Enhanced ML with collaborative insights
                if self.collaborative_ml:
                    ensemble_results = await self.collaborative_ml.ensemble_prediction(sensor_data)
                    ml_results.update(ensemble_results)
                
                # Emergency detection
                if self.emergency_system:
                    emergencies = await self.emergency_system.detect_emergency_condition(
                        sensor_data, ml_results
                    )
                
                # Data fusion with peer devices
                if self.data_fusion and self.config.collaboration.data_fusion_enabled:
                    # Create consensus for critical environmental parameters
                    temperature_consensus = await self.data_fusion.create_environmental_consensus(
                        "temperature", timeout_seconds=15
                    )
                    air_quality_consensus = await self.data_fusion.create_environmental_consensus(
                        "air_quality", timeout_seconds=15
                    )
                    
                    # Use consensus data if available and reliable
                    if temperature_consensus and temperature_consensus.get("fusion_metadata", {}).get("confidence", 0) > 0.8:
                        sensor_data["temperature_consensus"] = temperature_consensus
                    
                    if air_quality_consensus and air_quality_consensus.get("fusion_metadata", {}).get("confidence", 0) > 0.8:
                        sensor_data["air_quality_consensus"] = air_quality_consensus
                
                # Share insights with collaborative network
                if self.collaborative_ml and ml_results.get("confidence", 0) > 0.7:
                    await self.collaborative_ml.share_model_insights(ml_results)
                
                # Store enhanced data
                await self.storage.store_sensor_data(sensor_data, ml_results)
                
                # Network sharing
                await self._share_collaborative_data(sensor_data, ml_results)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring.interval_seconds)
                
            except Exception as e:
                print(f"Collaborative monitoring error: {e}")
                await asyncio.sleep(30)  # Error recovery delay
    
    async def _share_collaborative_data(self, sensor_data: Dict, ml_results: Dict):
        """Share data with collaborative networks"""
        try:
            # Prepare collaborative data package
            collaborative_data = {
                "device_id": self.device_id,
                "timestamp": datetime.now().isoformat(),
                "sensor_data": sensor_data,
                "ml_results": ml_results,
                "location": self.config.device.location,
                "data_quality": self._assess_data_quality(sensor_data),
                "sharing_permissions": self.config.collaboration.sharing_permissions
            }
            
            # Share through available networks
            if hasattr(self, 'tenzro_client'):
                await self.tenzro_client.send_sensor_data(collaborative_data)
            
            if hasattr(self, 'furcate_client'):
                await self.furcate_client.share_environmental_data(
                    sensor_data, ml_results
                )
            
        except Exception as e:
            print(f"Collaborative data sharing failed: {e}")
    
    def _assess_data_quality(self, sensor_data: Dict) -> float:
        """Assess overall quality of sensor data"""
        quality_scores = []
        
        for sensor_name, sensor_value in sensor_data.items():
            if isinstance(sensor_value, dict) and 'quality' in sensor_value:
                quality_scores.append(sensor_value['quality'])
            else:
                quality_scores.append(0.8)  # Default quality
        
        return np.mean(quality_scores) if quality_scores else 0.5
```

### Configuration for Collaborative Features

```yaml
# Enhanced configuration with collaboration settings
collaboration:
  enabled: true
  role: "worker"  # coordinator, specialist, relay, worker, emergency
  
  # Network configuration
  max_peers: 20
  trust_threshold: 0.7
  consensus_threshold: 0.8
  discovery_interval: 60
  
  # Data sharing preferences
  data_fusion_enabled: true
  model_sharing_enabled: true
  emergency_response_enabled: true
  
  sharing_permissions:
    sensor_data: true
    ml_insights: true
    emergency_alerts: true
    location_data: true
    device_status: true
  
  # Emergency response
  emergency_propagation_hops: 5
  emergency_response_distance_km: 50