#!/usr/bin/env python3
"""Test ML frameworks."""

print("🧪 Testing ML Frameworks...")

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple GPU) acceleration")
    else:
        device = torch.device("cpu")
        print("ℹ️  Using CPU")
    
    # Simple test
    x = torch.rand(1000, 1000).to(device)
    y = torch.rand(1000, 1000).to(device)
    z = torch.mm(x, y)
    print(f"✅ PyTorch matrix multiplication successful on {device}")
    
except Exception as e:
    print(f"❌ PyTorch error: {e}")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
    
    # Simple test
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(f"✅ TensorFlow matrix multiplication successful")
    
except Exception as e:
    print(f"❌ TensorFlow error: {e}")

print("🎉 ML framework testing complete!")
