# ============================================================================
# furcate_nano/mesh.py
"""Bio-inspired mesh networking for Furcate Nano devices."""

import asyncio
import logging
import json
import time
import random
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of mesh network messages."""
    HEARTBEAT = "heartbeat"
    ENVIRONMENTAL_DATA = "environmental_data"
    ALERT = "alert"
    DISCOVERY = "discovery"
    MESH_COORDINATION = "mesh_coordination"
    ASSET_NOTIFICATION = "asset_notification"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class MeshMessage:
    """Standardized mesh network message."""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    source_device: str
    target_device: Optional[str]
    timestamp: str
    payload: Dict[str, Any]
    ttl: int = 3
    hop_count: int = 0
    path: List[str] = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = [self.source_device]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission."""
        return asdict(self)

@dataclass
class MeshPeer:
    """Information about a mesh network peer."""
    device_id: str
    last_seen: datetime
    signal_strength: float
    hop_distance: int
    capabilities: Set[str]
    environmental_zone: str

class MeshNetworkManager:
    """Bio-inspired mesh network manager using mycelial network principles."""
    
    def __init__(self, config: Dict[str, Any], device_id: str):
        """Initialize mesh network manager.
        
        Args:
            config: Mesh network configuration
            device_id: This device's identifier
        """
        self.config = config
        self.device_id = device_id
        self.simulation_mode = config.get("simulation", True)
        
        # Network state
        self.peers: Dict[str, MeshPeer] = {}
        self.message_history: Dict[str, float] = {}  # message_id -> timestamp
        self.routing_table: Dict[str, str] = {}  # target -> next_hop
        
        # Bio-inspired parameters
        self.max_connections = config.get("max_connections", 8)
        self.signal_decay_rate = config.get("signal_decay_rate", 0.1)
        self.trust_threshold = config.get("trust_threshold", 0.5)
        self.environmental_zone = config.get("environmental_zone", "default")
        
        # Communication protocols
        self.protocols = []
        self._init_protocols()
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_relayed": 0,
            "peer_connections": 0
        }
        
        logger.info(f"Mesh network manager initialized for {device_id}")
    
    def _init_protocols(self):
        """Initialize communication protocols (LoRa, Bluetooth, etc.)."""
        # This would initialize actual radio protocols
        # For now, simulate the protocols
        if self.simulation_mode:
            self.protocols = ["lora_sim", "bluetooth_sim"]
            logger.info("🕸️ Mesh protocols in simulation mode")
        else:
            # Real protocol initialization would go here
            logger.info("🕸️ Initializing mesh protocols...")
    
    async def initialize(self) -> bool:
        """Initialize mesh networking."""
        try:
            # Start discovery process
            asyncio.create_task(self._discovery_loop())
            
            # Start maintenance tasks
            asyncio.create_task(self._maintenance_loop())
            
            # Start message processing
            asyncio.create_task(self._message_processing_loop())
            
            logger.info("✅ Mesh network initialized")
            return True
            
        except Exception as e:
            logger.error(f"Mesh network initialization failed: {e}")
            return False
    
    async def _discovery_loop(self):
        """Periodic discovery of mesh peers."""
        while True:
            try:
                # Send discovery beacon
                discovery_message = MeshMessage(
                    message_id=f"discovery_{int(time.time())}_{self.device_id}",
                    message_type=MessageType.DISCOVERY,
                    priority=MessagePriority.LOW,
                    source_device=self.device_id,
                    target_device=None,  # Broadcast
                    timestamp=datetime.now().isoformat(),
                    payload={
                        "capabilities": ["environmental_monitoring", "edge_ml"],
                        "environmental_zone": self.environmental_zone,
                        "device_type": "furcate_nano"
                    }
                )
                
                await self._broadcast_message(discovery_message)
                
                # Simulate discovery responses in simulation mode
                if self.simulation_mode:
                    await self._simulate_peer_discovery()
                
                # Wait before next discovery
                await asyncio.sleep(self.config.get("discovery_interval", 60))
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(30)
    
    async def _simulate_peer_discovery(self):
        """Simulate peer discovery for development."""
        simulated_peers = [
            f"nano-{i:03d}" for i in range(1, 6) if f"nano-{i:03d}" != self.device_id
        ]
        
        for peer_id in random.sample(simulated_peers, min(3, len(simulated_peers))):
            if peer_id not in self.peers:
                self.peers[peer_id] = MeshPeer(
                    device_id=peer_id,
                    last_seen=datetime.now(),
                    signal_strength=random.uniform(-60, -30),  # dBm
                    hop_distance=1,
                    capabilities={"environmental_monitoring", "edge_ml"},
                    environmental_zone=self.environmental_zone
                )
                logger.info(f"🔗 Discovered peer: {peer_id}")
    
    async def _maintenance_loop(self):
        """Maintain mesh network health."""
        while True:
            try:
                current_time = datetime.now()
                
                # Remove stale peers
                stale_peers = []
                for peer_id, peer in self.peers.items():
                    if (current_time - peer.last_seen).seconds > 300:  # 5 minutes
                        stale_peers.append(peer_id)
                
                for peer_id in stale_peers:
                    del self.peers[peer_id]
                    logger.info(f"🔌 Removed stale peer: {peer_id}")
                
                # Update routing table
                self._update_routing_table()
                
                # Clean old message history
                cutoff_time = time.time() - 3600  # 1 hour
                old_messages = [mid for mid, timestamp in self.message_history.items() 
                               if timestamp < cutoff_time]
                for mid in old_messages:
                    del self.message_history[mid]
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(60)
    
    async def _message_processing_loop(self):
        """Process incoming mesh messages."""
        while True:
            try:
                # In real implementation, this would listen to radio protocols
                # For simulation, we'll just wait
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(5)
    
    def _update_routing_table(self):
        """Update routing table using bio-inspired algorithms."""
        # Simple routing: direct connection or through best peer
        self.routing_table.clear()
        
        for peer_id in self.peers.keys():
            # Direct connection
            self.routing_table[peer_id] = peer_id
        
        # For multi-hop routing, we'd implement more sophisticated algorithms
        # based on mycelial network optimization principles
    
    async def broadcast_environmental_update(self, data: Dict[str, Any]) -> bool:
        """Broadcast environmental data update to mesh network.
        
        Args:
            data: Environmental data to broadcast
            
        Returns:
            Success status
        """
        try:
            message = MeshMessage(
                message_id=f"env_{int(time.time())}_{self.device_id}",
                message_type=MessageType.ENVIRONMENTAL_DATA,
                priority=MessagePriority.NORMAL,
                source_device=self.device_id,
                target_device=None,  # Broadcast
                timestamp=datetime.now().isoformat(),
                payload=data
            )
            
            return await self._broadcast_message(message)
            
        except Exception as e:
            logger.error(f"Environmental broadcast failed: {e}")
            return False
    
    async def broadcast_alert(self, alert: Dict[str, Any]) -> bool:
        """Broadcast environmental alert with high priority.
        
        Args:
            alert: Alert data to broadcast
            
        Returns:
            Success status
        """
        try:
            priority = MessagePriority.CRITICAL if alert.get("severity") == "critical" else MessagePriority.HIGH
            
            message = MeshMessage(
                message_id=f"alert_{int(time.time())}_{self.device_id}",
                message_type=MessageType.ALERT,
                priority=priority,
                source_device=self.device_id,
                target_device=None,  # Broadcast
                timestamp=datetime.now().isoformat(),
                payload=alert,
                ttl=5  # Higher TTL for alerts
            )
            
            return await self._broadcast_message(message)
            
        except Exception as e:
            logger.error(f"Alert broadcast failed: {e}")
            return False
    
    async def _broadcast_message(self, message: MeshMessage) -> bool:
        """Broadcast message to mesh network."""
        try:
            # Record message
            self.message_history[message.message_id] = time.time()
            
            # In real implementation, send via radio protocols
            if self.simulation_mode:
                logger.debug(f"📡 Broadcasting {message.message_type.value} message")
            
            self.stats["messages_sent"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Message broadcast failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get mesh network status."""
        return {
            "device_id": self.device_id,
            "peer_count": len(self.peers),
            "environmental_zone": self.environmental_zone,
            "routing_table_size": len(self.routing_table),
            "stats": self.stats.copy(),
            "simulation_mode": self.simulation_mode
        }
    
    def get_peer_info(self) -> List[Dict[str, Any]]:
        """Get information about connected peers."""
        return [
            {
                "device_id": peer.device_id,
                "last_seen": peer.last_seen.isoformat(),
                "signal_strength": peer.signal_strength,
                "hop_distance": peer.hop_distance,
                "environmental_zone": peer.environmental_zone
            }
            for peer in self.peers.values()
        ]
    
    async def shutdown(self):
        """Shutdown mesh network."""
        # Send goodbye message
        goodbye_message = MeshMessage(
            message_id=f"goodbye_{int(time.time())}_{self.device_id}",
            message_type=MessageType.MESH_COORDINATION,
            priority=MessagePriority.NORMAL,
            source_device=self.device_id,
            target_device=None,
            timestamp=datetime.now().isoformat(),
            payload={"action": "shutdown"}
        )
        
        await self._broadcast_message(goodbye_message)
        
        # Clean up
        self.peers.clear()
        self.routing_table.clear()
        
        logger.info("Mesh network manager shutdown complete")