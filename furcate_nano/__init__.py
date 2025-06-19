"""Furcate Nano - Open Source Environmental Edge Computing Framework"""
__version__ = "1.0.0"

print("🌿 Furcate Nano initialized with ML frameworks!")

# Test imports
try:
    import torch
    print(f"   📦 PyTorch {torch.__version__} ready")
except ImportError:
    print("   ❌ PyTorch not available")

try:
    import tensorflow as tf
    print(f"   📦 TensorFlow {tf.__version__} ready")
except ImportError:
    print("   ❌ TensorFlow not available")
