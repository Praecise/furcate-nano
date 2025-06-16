# ============================================================================
# scripts/install.sh
"""
#!/bin/bash
# Furcate Nano Installation Script for Raspberry Pi 5

set -e

echo "🌿 Furcate Nano Installation Script"
echo "======================================"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y git sqlite3

# Install hardware libraries
echo "🔧 Installing hardware libraries..."
sudo apt install -y python3-rpi.gpio
pip3 install adafruit-circuitpython-dht
pip3 install adafruit-circuitpython-bmp280
pip3 install adafruit-circuitpython-ads1x15

# Create virtual environment
echo "🏗️  Creating Python virtual environment..."
python3 -m venv /home/pi/furcate-nano-env
source /home/pi/furcate-nano-env/bin/activate

# Install Furcate Nano
echo "🌱 Installing Furcate Nano..."
pip install furcate-nano

# Create directories
echo "📁 Creating directories..."
sudo mkdir -p /data/furcate-nano
sudo mkdir -p /etc/furcate-nano
sudo chown pi:pi /data/furcate-nano

# Create default configuration
echo "⚙️  Creating default configuration..."
furcate-nano init --output /etc/furcate-nano/config.yaml

# Install systemd service
echo "🔄 Installing systemd service..."
sudo tee /etc/systemd/system/furcate-nano.service > /dev/null <<EOF
[Unit]
Description=Furcate Nano Environmental Monitoring
After=network.target

[Service]
Type=simple
User=pi
Environment=PATH=/home/pi/furcate-nano-env/bin
ExecStart=/home/pi/furcate-nano-env/bin/furcate-nano start --config /etc/furcate-nano/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable furcate-nano

echo "✅ Furcate Nano installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit configuration: sudo nano /etc/furcate-nano/config.yaml"
echo "2. Test hardware: furcate-nano test"
echo "3. Start service: sudo systemctl start furcate-nano"
echo "4. Check status: furcate-nano status"
echo ""
echo "Documentation: https://docs.furcate-nano.org"
echo "Community: https://discord.gg/furcate-nano"
"""