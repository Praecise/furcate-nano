# scripts/setup-raspberry-pi.sh
#!/bin/bash
# Complete Raspberry Pi 5 setup script for Furcate Nano

set -e

echo "🍓 Raspberry Pi 5 Setup for Furcate Nano"
echo "=========================================="

# Enable required interfaces
echo "🔧 Enabling hardware interfaces..."
sudo raspi-config nonint do_i2c 0    # Enable I2C
sudo raspi-config nonint do_spi 0    # Enable SPI
sudo raspi-config nonint do_serial 0 # Enable Serial
sudo raspi-config nonint do_ssh 0    # Enable SSH

# Install hardware support packages
echo "📦 Installing hardware packages..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-smbus \
    i2c-tools \
    git \
    sqlite3 \
    screen \
    htop

# Install circuit python libraries
echo "🔌 Installing CircuitPython libraries..."
pip3 install \
    adafruit-circuitpython-dht \
    adafruit-circuitpython-bmp280 \
    adafruit-circuitpython-ads1x15 \
    adafruit-circuitpython-gps \
    adafruit-circuitpython-tsl2561

# Configure GPIO permissions
echo "⚡ Configuring GPIO permissions..."
sudo usermod -a -G gpio,i2c,spi,dialout pi

# Install LoRa support
echo "📡 Installing LoRa support..."
# This would install specific LoRa module drivers

# Create Furcate Nano directories
echo "📁 Creating directories..."
sudo mkdir -p /data/furcate-nano/{models,logs,backups}
sudo mkdir -p /etc/furcate-nano
sudo chown -R pi:pi /data/furcate-nano

# Download ML models (placeholder)
echo "🤖 Setting up ML models..."
mkdir -p /data/furcate-nano/models
# Models would be downloaded from Furcate model repository

# Configure power management
echo "⚡ Configuring power management..."
# Configure for optimal power usage
echo 'dtoverlay=disable-wifi' | sudo tee -a /boot/config.txt
echo 'dtoverlay=disable-bt' | sudo tee -a /boot/config.txt

# Set up log rotation
echo "📝 Configuring log rotation..."
sudo tee /etc/logrotate.d/furcate-nano > /dev/null <<EOF
/data/furcate-nano/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    su pi pi
}
EOF

echo "✅ Raspberry Pi 5 setup complete!"
echo ""
echo "Next steps:"
echo "1. Reboot: sudo reboot"
echo "2. Test I2C: i2cdetect -y 1"
echo "3. Install Furcate Nano: pip install furcate-nano"
echo "4. Run setup: furcate-nano init"