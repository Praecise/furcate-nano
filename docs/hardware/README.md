# Furcate Nano Hardware Guide

Comprehensive hardware documentation for Furcate Nano environmental monitoring devices based on 2025 specifications and latest technology.

## Table of Contents

- [Overview](#overview)
- [Supported Platforms](#supported-platforms)
- [GPIO and Pin Configuration](#gpio-and-pin-configuration)
- [Environmental Sensors](#environmental-sensors)
- [Communication Interfaces](#communication-interfaces)
- [Power Management](#power-management)
- [Assembly and Integration](#assembly-and-integration)
- [Calibration and Maintenance](#calibration-and-maintenance)
- [Troubleshooting](#troubleshooting)
- [Performance Specifications](#performance-specifications)

## Overview

Furcate Nano leverages modern embedded computing platforms optimized for environmental monitoring applications. The system is designed around the principle of modularity, reliability, and educational accessibility while maintaining professional-grade performance.

### 2025 Hardware Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Furcate Nano Device (2025)                  │
├─────────────────────────────────────────────────────────────┤
│  Compute Module                                            │
│  ├─── Raspberry Pi 5 (BCM2712 + RP1) - Preferred          │
│  ├─── NVIDIA Jetson Orin Nano - AI/ML Applications        │
│  ├─── Raspberry Pi 4B - Legacy Support                    │
│  └─── Orange Pi 5 - Experimental                          │
├─────────────────────────────────────────────────────────────┤
│  RP1 I/O Controller (Raspberry Pi 5)                      │
│  ├─── 28 GPIO pins with 3.3V logic, 5V tolerant          │
│  ├─── Hardware interfaces: I2C, SPI, UART, PWM            │
│  ├─── Enhanced timing precision                           │
│  └─── Backward compatibility with Pi 4                    │
├─────────────────────────────────────────────────────────────┤
│  Modern Environmental Sensors                              │
│  ├─── BME680/BME688 (4-in-1: Temp/Humidity/Pressure/VOC) │
│  ├─── SDS011/PMS5003 (Particulate Matter PM2.5/PM10)     │
│  ├─── SHT30/SHT40 (High-precision Temp/Humidity)         │
│  └─── Additional sensors as needed                        │
├─────────────────────────────────────────────────────────────┤
│  Connectivity (2025)                                       │
│  ├─── WiFi 6 (802.11ax) - Pi 5                           │
│  ├─── Bluetooth 5.0/5.2                                   │
│  ├─── LoRa 433/868/915 MHz (Optional)                     │
│  ├─── Ethernet Gigabit                                    │
│  └─── USB 3.0 ports                                       │
├─────────────────────────────────────────────────────────────┤
│  Power Management                                          │
│  ├─── 27W USB-C Power Supply (Pi 5)                       │
│  ├─── Li-ion Battery Support                              │
│  ├─── Solar Panel Integration                             │
│  └─── Smart Power Management                              │
└─────────────────────────────────────────────────────────────┘
```

## Supported Platforms

### Primary Platform: Raspberry Pi 5 (2025 Recommended)

**Hardware Specifications:**
- **CPU**: Broadcom BCM2712, Quad-core ARM Cortex-A76 @ 2.4GHz
- **GPU**: VideoCore VII with OpenGL ES 3.1, Vulkan 1.2
- **RAM**: 4GB, 8GB LPDDR4X-4267 SDRAM
- **I/O Controller**: RP1 (designed in-house by Raspberry Pi)
- **GPIO**: 40-pin header with 28 GPIO pins
- **Power**: 5V via USB-C (27W official power supply recommended)
- **Operating Temperature**: 0°C to +70°C (extended range with cooling)

**Key Advantages for 2025:**
- Up to 3x faster than Raspberry Pi 4
- RP1 chip provides enhanced I/O capabilities
- Better power efficiency
- Improved thermal management
- Future-proof architecture
- Excellent community and educational support

**RP1 I/O Controller Features:**
- 28 multi-functional GPIO pins
- 5V tolerance when powered
- Hardware support for: 5×UART, 6×SPI, 4×I2C, 2×I2S
- 24-bit DPI output
- 4-channel PWM output
- Enhanced timing precision
- Built-in debouncing capabilities

### Secondary Platforms

#### NVIDIA Jetson Orin Nano (AI/ML Focus)

**Specifications:**
- **CPU**: 6-core ARM Cortex-A78AE @ 1.5GHz
- **GPU**: 1024-core NVIDIA Ampere architecture GPU
- **AI Performance**: 40 TOPS (INT8)
- **RAM**: 8GB 128-bit LPDDR5
- **Power**: 7W-15W configurable

**Use Cases:**
- Advanced ML processing
- Computer vision applications
- Real-time AI inference
- Research environments

#### Raspberry Pi 4B (Legacy Support)

**Specifications:**
- **CPU**: Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5GHz
- **RAM**: 4GB or 8GB LPDDR4
- **GPIO**: 40-pin header (legacy GPIO access)
- **Power**: 5V via USB-C (15W)

**Note**: Legacy GPIO libraries (RPi.GPIO) work on Pi 4 but **NOT** on Pi 5.

## GPIO and Pin Configuration

### Raspberry Pi 5 GPIO Pinout (2025 Standard)

The Raspberry Pi 5 uses the RP1 southbridge chip for GPIO control, requiring modern GPIO libraries.

```
Physical Pin Layout (40-pin header):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│3.3V │ 01  │ 02  │ 5V  │ 05  │ 06  │ GND │ 08  │ GND │ 10  │
│ SDA │ 03  │ 04  │ 5V  │ 07  │ GPIO│ 09  │GPIO │ 11  │GPIO │
│ SCL │ 05  │ 06  │ GND │ 09  │  4  │ 11  │ 14  │ 13  │ 15  │
│GPIO │ 07  │ 08  │GPIO │ 11  │GPIO │ 13  │ 14  │ 15  │GPIO │
│  4  │     │ 14  │ 17  │     │ 18  │ 27  │ GND │ 22  │ 23  │
│ GND │ 09  │ 10  │GPIO │ 13  │GPIO │ 15  │GPIO │ 17  │GPIO │
│     │     │ 15  │     │     │ 18  │ 22  │ 23  │     │ 24  │
│GPIO │ 11  │ 12  │GPIO │ 15  │GPIO │ 17  │ 18  │ 19  │ 20  │
│ 17  │     │ 18  │ 27  │ 22  │ 23  │3.3V │GPIO │GPIO │ GND │
│GPIO │ 13  │ 14  │GPIO │ 17  │GPIO │ 19  │ 24  │ 10  │     │
│ 27  │     │ GND │ 22  │3.3V │ 24  │GPIO │ 20  │ 21  │ 22  │
│GPIO │ 15  │ 16  │GPIO │ 19  │GPIO │ 10  │ GND │GPIO │GPIO │
│ 22  │     │ 23  │ 23  │GPIO │ 24  │ 21  │     │  9  │ 25  │
│3.3V │ 17  │ 18  │GPIO │ 21  │GPIO │GPIO │ 23  │GPIO │ 26  │
│     │     │ 24  │     │  9  │ 25  │ 11  │GPIO │  7  │GPIO │
│GPIO │ 19  │ 20  │GPIO │ 23  │ 26  │GPIO │ 25  │ 27  │ 28  │
│ 10  │     │ GND │ 25  │GPIO │GPIO │ 11  │ GND │GPIO │GPIO │
│GPIO │ 21  │ 22  │GPIO │ 25  │ 28  │ 23  │     │  0  │  1  │
│  9  │     │ 25  │     │ GND │GPIO │GPIO │ 27  │GPIO │ 30  │
│GPIO │ 23  │ 24  │GPIO │ 27  │  7  │ 11  │ 28  │  5  │ GND │
│ 11  │     │  8  │     │GPIO │ 28  │GPIO │GPIO │ 29  │     │
│GND  │ 25  │ 26  │GPIO │ 29  │  1  │ 23  │  1  │GPIO │ 31  │
│     │     │  7  │ 0   │GPIO │ 30  │GPIO │ 30  │  6  │ 32  │
│GPIO │ 27  │ 28  │GPIO │ 31  │ GND │ 11  │GPIO │ 33  │GPIO │
│  0  │     │  1  │  5  │GPIO │     │ 23  │  1  │GPIO │ 12  │
│GPIO │ 29  │ 30  │GPIO │ 33  │ 34  │GPIO │ 32  │ 35  │ 36  │
│  5  │     │ GND │  6  │GPIO │ GND │  8  │GPIO │GPIO │GPIO │
│GPIO │ 31  │ 32  │GPIO │ 35  │ 36  │ 25  │ 12  │ 19  │ 16  │
│  6  │     │ 12  │ 13  │GPIO │GPIO │ GND │ 34  │ 37  │ 38  │
│GPIO │ 33  │ 34  │GPIO │ 37  │ 38  │     │ GND │GPIO │GPIO │
│ 13  │     │ GND │ 19  │GPIO │GPIO │ 39  │     │ 26  │ 20  │
│GPIO │ 35  │ 36  │GPIO │ 39  │ 40  │GND  │ 40  │     │GPIO │
│ 19  │     │ 16  │ 26  │     │GPIO │     │GPIO │     │ 21  │
│GPIO │ 37  │ 38  │GPIO │     │ 21  │     │     │     │     │
│ 26  │     │ 20  │     │     │     │     │     │     │     │
│GND  │ 39  │ 40  │GPIO │     │     │     │     │     │     │
│     │     │ 21  │     │     │     │     │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### Modern GPIO Libraries for Raspberry Pi 5 (2025)

**Important**: Traditional RPi.GPIO library does **NOT** work with Raspberry Pi 5 due to the RP1 chip architecture.

#### Recommended Libraries (2025):

1. **gpiod** (Primary recommendation)
   - Modern Linux GPIO interface
   - Works with RP1 chip
   - Professional-grade reliability
   - Thread-safe operation

2. **GPIO Zero** (Educational friendly)
   - Updated for Pi 5 compatibility
   - Simple Python interface
   - Good for beginners
   - Built on lgpio backend

3. **lgpio** (Alternative)
   - Lightweight GPIO library
   - Pi 5 compatible
   - Lower-level access

#### GPIO Zero Example (2025 Compatible):
```python
from gpiozero import LED, Button
from time import sleep

# Works on both Pi 4 and Pi 5
led = LED(18)
button = Button(2)

while True:
    if button.is_pressed:
        led.on()
    else:
        led.off()
    sleep(0.1)
```

#### gpiod Example for Pi 5:
```python
import gpiod
import time

# For Raspberry Pi 5 - GPIO access via gpiochip4
chip = gpiod.Chip('gpiochip4')  # RP1 GPIO controller
led_line = chip.get_line(18)

# Configure as output
led_line.request(consumer="furcate_nano", type=gpiod.LINE_REQ_DIR_OUT)

try:
    while True:
        led_line.set_value(1)  # LED on
        time.sleep(1)
        led_line.set_value(0)  # LED off
        time.sleep(1)
finally:
    led_line.release()
```

### Furcate Nano GPIO Assignments

```yaml
# Standard GPIO pin assignments for Furcate Nano
gpio_pins:
  # Environmental sensors
  dht22_data: 4              # Temperature/humidity sensor
  sds011_tx: 14              # Particulate matter sensor TX
  sds011_rx: 15              # Particulate matter sensor RX
  
  # I2C devices (BME680, etc.)
  sda: 2                     # I2C data line
  scl: 3                     # I2C clock line
  
  # Status indicators
  status_led: 18             # Main status LED
  error_led: 19              # Error indicator LED
  activity_led: 20           # Activity indicator
  
  # Power management
  power_enable: 21           # Sensor power control
  battery_monitor: 26        # Battery level monitoring
  
  # Optional components
  buzzer: 13                 # Audio alerts
  user_button: 16            # User input button
  reset_button: 12           # System reset
  
  # LoRa module (if used)
  lora_reset: 22             # LoRa module reset
  lora_dio0: 24              # LoRa DIO0 interrupt
  lora_cs: 8                 # LoRa chip select (SPI)
```

## Environmental Sensors

### Primary Recommendation: BME680/BME688 (2025)

The BME680 is the preferred 4-in-1 environmental sensor for modern Furcate Nano deployments.

**Specifications:**
- **Temperature**: -40°C to +85°C (±1.0°C accuracy)
- **Humidity**: 0-100% RH (±3% accuracy)
- **Pressure**: 300-1100 hPa (±1 hPa accuracy)
- **Gas Sensor**: VOC detection in ppb range
- **Interface**: I2C (primary) or SPI
- **Power**: 1.7V-3.6V DC
- **Current**: <0.1mA average, 12mA during gas measurement
- **Package**: 3.0 × 3.0 × 0.95 mm³

**Key Advantages:**
- All-in-one solution reduces complexity
- Factory calibrated
- BSEC software for air quality index calculation
- Low power consumption
- Long-term stability (10+ years)
- Excellent for IoT applications

**Wiring (I2C):**
```
BME680 Module → Raspberry Pi 5
VCC     → 3.3V (Pin 1)
GND     → GND (Pin 6)
SDA     → GPIO 2 (Pin 3)
SCL     → GPIO 3 (Pin 5)
```

**Configuration:**
```yaml
hardware:
  sensors:
    environmental:
      type: "BME680"
      i2c_address: 0x77        # Alternative: 0x76
      enabled: true
      gas_heater: true         # Enable VOC measurement
      features: 
        - "temperature"
        - "humidity" 
        - "pressure"
        - "gas"
      calibration:
        temperature_offset: 0.0
        humidity_offset: 0.0
        pressure_offset: 0.0
      measurement_settings:
        temperature_oversample: 8
        humidity_oversample: 2
        pressure_oversample: 4
        gas_heater_temp: 320    # °C
        gas_heater_duration: 150 # ms
```

### Particulate Matter Sensors

#### SDS011 (Proven Choice)

**Specifications:**
- **PM2.5 Range**: 0-999.9 μg/m³
- **PM10 Range**: 0-999.9 μg/m³
- **Accuracy**: ±15% and ±10 μg/m³
- **Interface**: UART (3.3V TTL)
- **Power**: 5V DC, 70mA active, 4mA sleep
- **Lifespan**: 8000 hours continuous operation

**Advantages:**
- Proven reliability in citizen science projects
- Good accuracy for the price
- Sleep mode for power saving
- Well-documented protocol

**Wiring:**
```
SDS011 → Raspberry Pi 5
Pin 1 (TX) → GPIO 15 (Pin 10) RX
Pin 2 (RX) → GPIO 14 (Pin 8) TX  
Pin 3 (GND) → GND (Pin 6)
Pin 4 (5V) → 5V (Pin 2)
Pin 5 (NC) → Not connected
```

#### PMS5003 (Advanced Alternative)

**Specifications:**
- **Range**: PM1.0, PM2.5, PM10: 0-500 μg/m³
- **Particle Count**: 0.3-10 μm size bins
- **Interface**: UART (3.3V TTL)
- **Power**: 5V DC, 100mA typical

**Advantages:**
- PM1.0 measurement capability
- Particle count data
- More compact design
- Digital interface

### Temperature/Humidity Sensors (Alternatives)

#### SHT30/SHT40 (High Precision)

**SHT30 Specifications:**
- **Temperature**: -40°C to +125°C (±0.2°C accuracy)
- **Humidity**: 0-100% RH (±2% accuracy)
- **Interface**: I2C
- **Response Time**: <4 seconds (humidity)

**Advantages:**
- Superior accuracy to DHT sensors
- Fast response time
- I2C interface allows multiple sensors
- Industrial-grade reliability

**Wiring:**
```
SHT30 → Raspberry Pi 5
VCC → 3.3V (Pin 1)
GND → GND (Pin 6)  
SDA → GPIO 2 (Pin 3)
SCL → GPIO 3 (Pin 5)
```

#### DHT22 (Educational/Budget)

**Specifications:**
- **Temperature**: -40°C to +80°C (±0.5°C accuracy)
- **Humidity**: 0-100% RH (±2-5% accuracy)
- **Interface**: Single-wire digital
- **Power**: 3.3V-6V DC

**Note**: While still functional, DHT22 is considered legacy. BME680 or SHT30 recommended for new deployments.

### Additional Sensors

#### Light Sensors

**TSL2591** (Recommended for 2025):
- **Range**: 188 μlx to 88,000 lx
- **Interface**: I2C
- **Features**: IR compensation, programmable gain

#### Soil Moisture

**Capacitive Sensors** (Recommended):
- Corrosion resistant
- Analog output via ADC
- Long lifespan

## Communication Interfaces

### I2C Configuration

Multiple sensors can share the I2C bus:

```python
# Modern gpiod-based I2C access for Pi 5
import board
import busio

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Scan for devices
import adafruit_bus_device.i2c_device as i2c_device

def scan_i2c():
    devices = []
    for addr in range(0x08, 0x78):
        try:
            device = i2c_device.I2CDevice(i2c, addr)
            with device:
                devices.append(hex(addr))
        except:
            pass
    return devices

print("I2C devices found:", scan_i2c())
```

### UART Configuration

For particulate matter sensors:

```python
# Modern UART access for Pi 5
import serial
import time

# Configure UART for SDS011
ser = serial.Serial(
    port='/dev/ttyS0',
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

def read_sds011():
    """Read PM2.5 and PM10 from SDS011"""
    data = ser.read(10)
    if len(data) == 10 and data[0] == 0xAA and data[1] == 0xC0:
        pm25 = (data[3] * 256 + data[2]) / 10.0
        pm10 = (data[5] * 256 + data[4]) / 10.0
        return pm25, pm10
    return None, None
```

## Power Management

### Raspberry Pi 5 Power Requirements

**Official 27W USB-C Power Supply** (Recommended):
- Input: 100-240V AC
- Output: 5V DC, 5A
- Connector: USB-C
- Features: Overcurrent protection, low ripple

**Power Consumption:**
- Idle: ~3W
- Normal operation: 5-8W
- Peak load: 12-15W
- With active cooling: +2W

### Battery Integration

**Recommended Setup:**
```yaml
power:
  battery:
    type: "18650_pack"
    configuration: "3S2P"  # 3 cells in series, 2 in parallel
    capacity_mah: 6000     # Total capacity
    voltage_nominal: 11.1  # 3.7V × 3 cells
    
  charging:
    method: "solar_mppt"
    panel_watts: 30
    charge_controller: "MPPT_20A"
    
  management:
    low_voltage_cutoff: 9.0    # Volts
    undervoltage_warning: 9.6  # Volts
    power_save_threshold: 20   # Percent
```

### Power Optimization

**Software Power Management:**
```python
# Power-aware sensor reading
import asyncio
from furcate_nano import PowerManager

class PowerAwareSensorManager:
    def __init__(self, power_manager):
        self.power = power_manager
        
    async def adaptive_monitoring(self):
        """Adjust monitoring frequency based on power level"""
        battery_level = await self.power.get_battery_level()
        
        if battery_level > 50:
            interval = 60  # Normal: every minute
        elif battery_level > 20:
            interval = 300  # Low power: every 5 minutes
        else:
            interval = 900  # Critical: every 15 minutes
            
        return interval
```

## Assembly and Integration

### Standard Furcate Nano Assembly

**Component Layout:**
1. **Base**: Raspberry Pi 5 with heat sink/fan
2. **Sensor Board**: Custom PCB or breadboard with sensors
3. **Power Module**: Battery pack and charging circuit
4. **Enclosure**: Weather-resistant case
5. **Antenna**: LoRa antenna (if applicable)

**Assembly Steps:**

1. **Prepare Raspberry Pi 5:**
```bash
# Install latest Raspberry Pi OS (Bookworm or later)
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv
sudo apt install -y python3-gpiod python3-serial
sudo apt install -y i2c-tools

# Enable interfaces
sudo raspi-config
# Enable I2C, SPI, UART
```

2. **Install Libraries:**
```bash
# Create virtual environment
python3 -m venv furcate-env
source furcate-env/bin/activate

# Install Furcate Nano
pip install furcate-nano

# Install sensor libraries
pip install adafruit-circuitpython-bme680
pip install pyserial
```

3. **Wire Sensors:**
   - Follow pinout diagrams
   - Use proper gauge wire (22-24 AWG recommended)
   - Add pull-up resistors where needed (I2C: 4.7kΩ)
   - Secure all connections

4. **Test Hardware:**
```bash
# Test I2C devices
i2cdetect -y 1

# Test GPIO
pinout  # Show GPIO layout

# Test sensors
python3 -c "
from furcate_nano import HardwareManager
import asyncio

async def test():
    hw = HardwareManager({'simulation': False})
    await hw.initialize()
    readings = await hw.read_all_sensors()
    print('Sensor readings:', readings)

asyncio.run(test())
"
```

## Calibration and Maintenance

### Sensor Calibration (2025 Best Practices)

**BME680 Calibration:**
```python
# BME680 comes factory calibrated, but can be fine-tuned
calibration_config = {
    "temperature_offset": -1.5,  # Compensate for self-heating
    "humidity_offset": 2.0,      # Adjust based on known reference
    "pressure_offset": 0.0,      # Usually not needed
    "gas_baseline": 150000       # Established during burn-in
}
```

**Particulate Matter Calibration:**
- Compare with reference monitor
- Apply correction factors
- Account for humidity effects

### Maintenance Schedule

**Weekly:**
- Visual inspection of connections
- Check data quality
- Monitor power levels

**Monthly:**
- Clean sensors (following manufacturer guidelines)
- Check calibration drift
- Update software

**Quarterly:**
- Deep clean fan and cooling
- Replace consumable components
- Comprehensive calibration check

## Troubleshooting

### Common Issues (2025)

#### 1. GPIO Library Compatibility

**Problem**: "RPi.GPIO doesn't work on Pi 5"
**Solution**: 
```bash
# Remove old GPIO library
pip uninstall RPi.GPIO

# Install modern alternatives
pip install gpiod
pip install gpiozero

# Use GPIO Zero for simple applications
# Use gpiod for advanced applications
```

#### 2. I2C Device Not Detected

**Problem**: BME680 not found at 0x77
**Solution**:
```bash
# Check I2C bus
i2cdetect -y 1

# Try alternative address
# BME680 can be 0x76 or 0x77 depending on SDO pin

# Check connections
# Verify 3.3V power and ground
# Ensure SDA/SCL are connected correctly
```

#### 3. UART Permission Issues

**Problem**: Cannot access /dev/ttyS0
**Solution**:
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Enable UART in config
echo "enable_uart=1" | sudo tee -a /boot/config.txt

# Disable console on UART
sudo systemctl disable hciuart

# Reboot
sudo reboot
```

#### 4. Power Issues

**Problem**: Random shutdowns or instability
**Solution**:
```bash
# Check power supply
vcgencmd get_throttled

# Result 0x0 = good
# Non-zero indicates power problems

# Use official 27W power supply
# Check for voltage drop in cables
```

### Diagnostic Tools

```python
# Hardware diagnostic script
import asyncio
from furcate_nano import HardwareManager

async def comprehensive_test():
    """Run complete hardware diagnostics"""
    hw = HardwareManager({'simulation': False})
    
    # Initialize hardware
    init_result = await hw.initialize()
    print(f"Hardware initialization: {'PASS' if init_result else 'FAIL'}")
    
    # Test all sensors
    diagnostics = await hw.run_diagnostics()
    
    print("\n=== SENSOR DIAGNOSTICS ===")
    for sensor, result in diagnostics['sensor_tests'].items():
        status = "PASS" if result['status'] == 'pass' else "FAIL"
        print(f"{sensor}: {status}")
        
    print(f"\nOverall Status: {diagnostics['overall_status'].upper()}")
    print(f"Success Rate: {diagnostics['summary']['sensor_success_rate']:.1%}")

# Run diagnostics
asyncio.run(comprehensive_test())
```

## Performance Specifications

### Expected Performance (2025)

**Data Collection:**
- Reading interval: 30-300 seconds (configurable)
- Data accuracy: 95%+ (properly calibrated sensors)
- Uptime: 99%+ (with proper power management)

**Power Consumption:**
- Raspberry Pi 5: 3-8W average
- Sensors: 0.5-2W total
- Total system: 4-12W depending on configuration

**Network Performance:**
- WiFi: 802.11ac/ax (WiFi 6 on Pi 5)
- LoRa: Up to 10km range (ideal conditions)
- Data throughput: Sufficient for environmental monitoring

**Environmental Operating Range:**
- Temperature: -10°C to +60°C (with enclosure)
- Humidity: 10-90% RH (non-condensing)
- Protection: IP65 (with proper enclosure)

### Scalability

**Single Device:**
- Up to 10 sensors per device
- 1-minute minimum sampling interval
- Local storage: 30+ days

**Network Deployment:**
- Up to 100 devices per LoRa gateway
- Mesh networking: 8 direct peers
- Cloud synchronization: Real-time capability

This hardware guide provides comprehensive coverage of modern Furcate Nano deployment using 2025 technology standards, with particular emphasis on Raspberry Pi 5 compatibility and current best practices.