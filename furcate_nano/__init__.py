# ============================================================================
# furcate_nano/__init__.py
"""
Furcate Nano - Open Source Environmental Edge Computing Framework

Transform Raspberry Pi 5 devices into intelligent environmental monitoring nodes
with bio-inspired mesh networking and edge AI capabilities.
"""

__version__ = "1.0.0"
__author__ = "Furcate Team"
__email__ = "opensource@furcate.org"
__license__ = "MIT"
__homepage__ = "https://github.com/furcate-team/furcate-nano"

# Core framework imports
from .core import FurcateNanoCore
from .hardware import HardwareManager, SensorReading, SensorType
from .edge_ml import EdgeMLEngine, ModelType
from .mesh import MeshNetworkManager, MessageType, MessagePriority
from .power import PowerManager, PowerMode
from .config import NanoConfig, load_config

# Version information
VERSION_INFO = tuple(int(v) for v in __version__.split("."))

# System requirements check
import sys
import platform

if sys.version_info < (3, 8):
    raise RuntimeError("Furcate Nano requires Python 3.8 or higher")

if platform.machine() not in ['aarch64', 'armv7l', 'x86_64']:
    print("Warning: Furcate Nano is optimized for ARM64 (Raspberry Pi 5)")

__all__ = [
    # Core
    "FurcateNanoCore",
    
    # Hardware management  
    "HardwareManager",
    "SensorReading", 
    "SensorType",
    
    # Edge ML
    "EdgeMLEngine",
    "ModelType",
    
    # Mesh networking
    "MeshNetworkManager", 
    "MessageType",
    "MessagePriority",
    
    # Power management
    "PowerManager",
    "PowerMode",
    
    # Configuration
    "NanoConfig",
    "load_config",
    
    # Metadata
    "__version__",
    "VERSION_INFO"
]
