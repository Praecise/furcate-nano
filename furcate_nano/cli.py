#!/usr/bin/env python3
"""
Fixed Command Line Interface for Furcate Nano
"""
import click

@click.group()
def cli():
    """Furcate Nano - Environmental Edge Computing"""
    pass

@cli.command()
def test():
    """Test ML frameworks."""
    print("🧪 Testing ML Frameworks...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.backends.mps.is_available():
            print("✅ Using MPS (Apple GPU) acceleration")
        else:
            print("📱 Using CPU")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not available")
    
    print("🎉 ML framework testing complete!")

@cli.command()
def start():
    """Start monitoring (placeholder)."""
    print("🌿 Furcate Nano monitoring started!")
    print("📊 Dashboard: http://localhost:8000 (coming soon)")
    print("🧠 ML frameworks ready for environmental analysis")
    print("⚡ Press Ctrl+C to stop")
    
    try:
        import time
        while True:
            print("📈 Simulating environmental monitoring...")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

# This is the main entry point that setup.py looks for
def main():
    cli()

if __name__ == "__main__":
    main()