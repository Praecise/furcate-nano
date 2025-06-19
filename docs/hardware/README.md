# Furcate Nano Hardware Guide 2025

Comprehensive dual-platform hardware documentation for Furcate Nano environmental monitoring devices optimized for both Raspberry Pi 5 (TensorFlow Lite) and NVIDIA Jetson Nano (PyTorch) deployments.

## Table of Contents

- [Platform Overview](#platform-overview)
- [Raspberry Pi 5 Platform](#raspberry-pi-5-platform)
- [NVIDIA Jetson Nano Platform](#nvidia-jetson-nano-platform)
- [Platform Comparison](#platform-comparison)
- [GPIO and Pin Configuration](#gpio-and-pin-configuration)
- [Environmental Sensors](#environmental-sensors)
- [Communication Interfaces](#communication-interfaces)
- [Power Management](#power-management)
- [Assembly and Integration](#assembly-and-integration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Platform Overview

Furcate Nano 2025 supports dual AI compute platforms, each optimized for different deployment scenarios:

- **Raspberry Pi 5**: TensorFlow Lite optimization, educational environments, cost-effective deployments
- **NVIDIA Jetson Nano**: PyTorch optimization, research applications, GPU-accelerated inference

### 2025 Architecture Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                Furcate Nano Dual Platform (2025)            │
├─────────────────────────────────────────────────────────────┤
│  Platform A: Raspberry Pi 5      │  Platform B: Jetson Nano │
│  ├─ BCM2712 Quad-core A76 @2.4GHz│  ├─ Tegra X1 A57 @1.43GHz│
│  ├─ VideoCore VII GPU             │  ├─ 128-core Maxwell GPU │
│  ├─ 4-8GB LPDDR4X-4267           │  ├─ 4GB LPDDR4-1600      │
│  ├─ TensorFlow Lite Optimized    │  ├─ PyTorch + TensorRT   │
│  ├─ Educational Focus            │  ├─ Research Focus       │
│  └─ $60-80 USD                   │  └─ $149-199 USD (EOL)   │
├─────────────────────────────────────────────────────────────┤
│  Common Features:                                           │
│  ├─ Environmental Sensors (BME688, SDS011, etc.)          │
│  ├─ Mesh Networking (LoRa, WiFi, Bluetooth)               │
│  ├─ Edge ML Processing                                     │
│  ├─ Solar + Battery Power Management                       │
│  └─ Educational & Research Applications                     │
└─────────────────────────────────────────────────────────────┘
```

## Raspberry Pi 5 Platform

**Recommended for: Education, Cost-Effective Deployments, TensorFlow Lite Models**

### Hardware Specifications (2025)

- **CPU**: Broadcom BCM2712, Quad-core ARM Cortex-A76 @ 2.4GHz
- **GPU**: VideoCore VII with OpenGL ES 3.1, Vulkan 1.2, H.265 4Kp60 decode
- **RAM**: 4GB or 8GB LPDDR4X-4267 SDRAM (unified memory)
- **I/O Controller**: RP1 chip (custom-designed by Raspberry Pi)
- **GPIO**: 40-pin header with 28 GPIO pins (5V tolerant when powered)
- **Power**: 5V via USB-C (27W official power supply recommended)
- **Storage**: MicroSD (UHS-I), M.2 NVMe SSD support via HAT
- **Connectivity**: WiFi 6 (802.11ax), Bluetooth 5.0/5.2, Gigabit Ethernet
- **Dimensions**: 85 × 56 × 17mm
- **Operating Temperature**: 0°C to +70°C (with proper cooling)

### Key Advantages for Environmental Monitoring

**Performance Improvements (vs Pi 4):**
- **3x faster CPU** performance with Cortex-A76 cores
- **Enhanced I/O** with RP1 controller for reliable sensor communication
- **Better power efficiency** for solar-powered deployments
- **Improved thermal management** for extended outdoor operation
- **Native TensorFlow Lite** optimization for edge AI

**Educational Benefits:**
- **Cost-effective** for classroom deployments (20-30 units)
- **Extensive community support** and educational resources
- **Backward compatibility** with existing Pi accessories
- **Simple setup** with Raspberry Pi OS Bookworm
- **GPIO libraries** fully compatible with educational curricula

### Pi 5 Specific Features

**RP1 I/O Controller Enhanced Capabilities:**
- 28 multi-functional GPIO pins with improved timing precision
- Hardware debouncing for reliable sensor readings
- Enhanced PWM for precise control applications
- 5x UART, 6x SPI, 4x I2C interfaces
- Built-in voltage translation (3.3V/5V compatibility)

**TensorFlow Lite Optimization:**
- **4x faster inference** compared to Pi 4
- Support for **TensorFlow Lite 2.16.1** with ARM64 optimizations
- Hardware-accelerated neural network operators
- Quantized model support (INT8, INT16)
- **Sub-100ms inference** for environmental classification models

## NVIDIA Jetson Nano Platform

**Recommended for: Research, Advanced AI, GPU-Accelerated Applications**

### Hardware Specifications (2025)

**Note**: *Jetson Nano Developer Kit is EOL as of December 2023, but modules available until January 2027. Consider Jetson Orin Nano for new projects.*

- **CPU**: NVIDIA Tegra X1 SoC, Quad-core ARM Cortex-A57 @ 1.43GHz
- **GPU**: 128-core NVIDIA Maxwell GPU @ 921MHz
- **AI Performance**: 472 GFLOPS (FP16), 21 TOPS (INT8) with optimization
- **RAM**: 4GB 64-bit LPDDR4-1600 (shared between CPU and GPU)
- **Storage**: MicroSD (up to 256GB), optional eMMC module
- **Power**: 5V⎓3A (15W) via barrel jack or USB-C
- **I/O**: 40-pin GPIO header (compatible with Pi), 4x USB 3.0, HDMI 2.0
- **Connectivity**: Gigabit Ethernet, M.2 Key E (WiFi modules)
- **Dimensions**: 100 × 80 × 29mm (with heatsink)
- **Operating Temperature**: -25°C to +80°C

### Key Advantages for Environmental Monitoring

**GPU Acceleration:**
- **Real-time processing** of multiple sensor streams
- **Computer vision** integration for visual environmental monitoring
- **PyTorch native support** with GPU acceleration
- **TensorRT optimization** for maximum inference performance
- **CUDA cores** enable parallel processing of environmental data

**Research Capabilities:**
- **Custom model training** on-device for specialized environmental applications
- **Multi-modal fusion** of sensor data, imagery, and temporal sequences
- **Advanced ML algorithms** including transformer models for environmental prediction
- **NVDLA (Deep Learning Accelerator)** support in newer revisions
- **Professional development** with same SDK as enterprise Jetson platforms

### Jetson Nano Specific Features

**CUDA and TensorRT Integration:**
- Native **PyTorch 2.0+** support with CUDA acceleration
- **TensorRT inference engine** for optimized deployment
- **torch2trt converter** for seamless PyTorch to TensorRT conversion
- **JetPack SDK 4.6.x** with comprehensive ML libraries
- **Docker support** for containerized ML environments

**Advanced AI Capabilities:**
- **Real-time object detection** at 30+ FPS
- **Semantic segmentation** for environmental analysis
- **Time-series forecasting** with LSTM/Transformer models
- **Multi-camera processing** for comprehensive environmental monitoring
- **Edge-cloud hybrid processing** with model synchronization

### Migration Path: Jetson Orin Nano (2025)

For new deployments, consider the **Jetson Orin Nano Super**:
- **67 TOPS AI performance** (80x faster than original Nano)
- **8GB LPDDR5 memory** with higher bandwidth
- **6-core Arm Cortex-A78AE CPU** @ 1.5GHz
- **1024-core Ampere GPU** with Tensor cores
- **TensorRT-LLM support** for large language models
- **Price**: ~$249-399 (vs $149 for original Nano)

## Platform Comparison

### Performance Metrics (2025 Benchmarks)

| Metric | Raspberry Pi 5 (8GB) | Jetson Nano | Jetson Orin Nano |
|--------|----------------------|-------------|-------------------|
| **CPU Performance** | ARM A76 @2.4GHz | ARM A57 @1.43GHz | ARM A78AE @1.5GHz |
| **AI Inference (TOPS)** | ~2.4 TOPS (estimated) | ~0.5 TOPS | 67 TOPS |
| **TensorFlow Lite (FPS)** | 45-60 FPS | 25-35 FPS | 120+ FPS |
| **PyTorch (FP32)** | 15-20 FPS | 30-40 FPS | 200+ FPS |
| **Memory Bandwidth** | 17.1 GB/s | 25.6 GB/s | 102 GB/s |
| **Power Consumption** | 3-8W | 5-10W | 7-15W |
| **Cost (2025)** | $60-80 | $149+ (limited) | $249-399 |
| **Educational Suitability** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| **Research Capability** | ★★★☆☆ | ★★★★☆ | ★★★★★ |

### Use Case Recommendations

**Choose Raspberry Pi 5 for:**
- **Educational environments** (K-12, undergraduate courses)
- **Cost-sensitive deployments** (>10 devices)
- **TensorFlow Lite models** and standard ML applications
- **Community projects** and citizen science initiatives
- **Simple environmental monitoring** with basic AI features
- **Long-term availability** and ecosystem support

**Choose NVIDIA Jetson Nano for:**
- **Research applications** requiring custom AI models
- **Advanced computer vision** and multi-modal processing
- **PyTorch development** and GPU-accelerated computing
- **Real-time processing** of complex environmental data
- **Professional prototyping** before enterprise deployment
- **Existing inventory** (since EOL announced)

**Choose Jetson Orin Nano for:**
- **Next-generation AI applications** requiring high performance
- **Large language models** for environmental analysis
- **Real-time video processing** with multiple cameras
- **Edge-cloud hybrid** architectures
- **Commercial deployment** preparation
- **Future-proof** development platform

## GPIO and Pin Configuration

### Unified GPIO Approach (40-pin Header)

Both platforms use the same physical 40-pin GPIO header layout, enabling **unified sensor connectivity**:

```
Physical Pin Layout (40-pin header - Both Platforms):
     3.3V  [ 1] [ 2]  5V     
 GPIO 2    [ 3] [ 4]  5V     
 GPIO 3    [ 5] [ 6]  GND    
 GPIO 4    [ 7] [ 8]  GPIO 14
     GND   [ 9] [10]  GPIO 15
 GPIO 17   [11] [12]  GPIO 18
 GPIO 27   [13] [14]  GND    
 GPIO 22   [15] [16]  GPIO 23
     3.3V  [17] [18]  GPIO 24
 GPIO 10   [19] [20]  GND    
 GPIO 9    [21] [22]  GPIO 25
 GPIO 11   [23] [24]  GPIO 8 
     GND   [25] [26]  GPIO 7 
 GPIO 0    [27] [28]  GPIO 1 
 GPIO 5    [29] [30]  GND    
 GPIO 6    [31] [32]  GPIO 12
 GPIO 13   [33] [34]  GND    
 GPIO 19   [35] [36]  GPIO 16
 GPIO 26   [37] [38]  GPIO 20
     GND   [39] [40]  GPIO 21
```

### Platform-Specific GPIO Libraries

#### Raspberry Pi 5 GPIO Libraries (2025)

**Primary: gpiod (Recommended)**
```python
import gpiod
import time

# For Raspberry Pi 5 - GPIO via gpiochip4 (RP1 controller)
chip = gpiod.Chip('gpiochip4')
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

**Alternative: GPIO Zero (Educational)**
```python
from gpiozero import LED, Button, MCP3008
from time import sleep

# Works on both Pi 4 and Pi 5 with updated libraries
led = LED(18)
button = Button(2)
adc = MCP3008(channel=0)  # For analog sensors

while True:
    if button.is_pressed:
        led.on()
        print(f"Analog value: {adc.value}")
    else:
        led.off()
    sleep(0.1)
```

#### NVIDIA Jetson Nano GPIO Libraries

**Primary: Jetson.GPIO**
```python
import Jetson.GPIO as GPIO
import time

# Setup
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(18, GPIO.OUT)  # LED on pin 18
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button

try:
    while True:
        if GPIO.input(16) == GPIO.LOW:  # Button pressed
            GPIO.output(18, GPIO.HIGH)  # LED on
        else:
            GPIO.output(18, GPIO.LOW)   # LED off
        time.sleep(0.1)
finally:
    GPIO.cleanup()
```

**Advanced: PyTorch Integration for AI GPIO**
```python
import Jetson.GPIO as GPIO
import torch
import torchvision.transforms as transforms
from PIL import Image

# GPIO setup for sensor control
GPIO.setmode(GPIO.BOARD)
GPIO.setup(18, GPIO.OUT)  # Sensor power control

# Load environmental classification model
model = torch.jit.load('environmental_classifier.pt')
model.eval()

def process_environmental_data(sensor_data):
    # Convert sensor data to tensor
    input_tensor = torch.tensor(sensor_data).float().unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)
    
    return prediction.item()

# Control sensors based on AI predictions
def adaptive_sensor_control(prediction):
    if prediction == 0:  # Good air quality
        GPIO.output(18, GPIO.LOW)   # Normal sampling
    else:  # Poor air quality
        GPIO.output(18, GPIO.HIGH)  # Increased sampling
```

### Furcate Nano Unified GPIO Assignments

```yaml
# Platform-agnostic GPIO configuration
gpio_pins:
  # Environmental sensors (same for both platforms)
  temperature_humidity_data: 4      # DHT22, BME688 I2C
  air_quality_uart_tx: 14          # SDS011, PMS5003 TX
  air_quality_uart_rx: 15          # SDS011, PMS5003 RX
  
  # I2C bus (same pins, different controllers)
  i2c_sda: 2                       # Pi5: RP1, Jetson: Tegra X1
  i2c_scl: 3                       # Pi5: RP1, Jetson: Tegra X1
  
  # Status indicators
  status_led_green: 18             # System operational
  status_led_red: 19               # Error/warning indicator
  activity_led_blue: 20            # ML processing indicator
  
  # Power management
  sensor_power_enable: 21          # Master sensor power control
  solar_panel_enable: 22           # Solar charging control
  battery_monitor_adc: 26          # Battery voltage monitoring
  
  # Optional peripherals
  buzzer: 13                       # Audio alerts
  user_button: 16                  # User interaction
  reset_button: 12                 # System reset
  
  # LoRa module (SPI)
  lora_chip_select: 8              # SPI CS0
  lora_reset: 22                   # LoRa module reset
  lora_dio0: 24                    # LoRa interrupt pin
  
  # Platform-specific optimizations
  raspberry_pi_5:
    cooling_fan_pwm: 12            # PWM fan control (RP1)
    performance_led: 25            # Performance monitoring
  
  jetson_nano:
    gpu_activity_led: 25           # GPU utilization indicator
    tensorrt_status: 23            # TensorRT optimization status
```

## Environmental Sensors

### Multi-Platform Sensor Support

All sensors support both platforms with platform-optimized drivers:

#### BME688 - Primary Environmental Sensor (2025 Recommended)

**Universal sensor for comprehensive environmental monitoring**

**Specifications:**
- **Temperature**: -40°C to +85°C (±0.5°C accuracy)
- **Humidity**: 0-100% RH (±3% accuracy)  
- **Pressure**: 300-1100 hPa (±0.6 hPa accuracy)
- **Gas Sensor**: VOC detection with AI-driven analysis
- **Interface**: I2C (0x76, 0x77) or SPI
- **Power**: 1.7-3.6V, <2mA active, <0.15µA sleep
- **AI Features**: BSEC 2.0 library with air quality index calculation

**Platform-Specific Optimizations:**

*Raspberry Pi 5:*
```python
# Optimized for Pi 5 with RP1 I2C controller
import board
import busio
import adafruit_bme680

# Initialize I2C with Pi 5 optimizations
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
bme688 = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x77)

# Configure for environmental monitoring
bme688.sea_level_pressure = 1013.25
bme688.temperature_oversample = 8
bme688.humidity_oversample = 2
bme688.pressure_oversample = 4
bme688.gas_heater_temperature = 320
bme688.gas_heater_duration = 150

def read_environmental_data():
    return {
        'temperature': round(bme688.temperature, 2),
        'humidity': round(bme688.relative_humidity, 2), 
        'pressure': round(bme688.pressure, 2),
        'gas_resistance': bme688.gas,
        'altitude': round(bme688.altitude, 2)
    }
```

*NVIDIA Jetson Nano:*
```python
# Optimized for Jetson with CUDA-accelerated processing
import board
import busio
import adafruit_bme680
import torch
import numpy as np

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
bme688 = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x77)

# Load PyTorch model for gas analysis
gas_classifier = torch.jit.load('bme688_gas_classifier.pt')
gas_classifier.eval()

def read_environmental_data_ai():
    # Read sensor data
    raw_data = {
        'temperature': bme688.temperature,
        'humidity': bme688.relative_humidity,
        'pressure': bme688.pressure,
        'gas_resistance': bme688.gas
    }
    
    # AI-enhanced gas analysis
    gas_tensor = torch.tensor([
        raw_data['gas_resistance'],
        raw_data['temperature'], 
        raw_data['humidity']
    ]).float().unsqueeze(0)
    
    with torch.no_grad():
        gas_classification = gas_classifier(gas_tensor)
        air_quality_index = torch.softmax(gas_classification, dim=1)
    
    raw_data['ai_air_quality'] = air_quality_index.max().item()
    raw_data['air_quality_class'] = torch.argmax(air_quality_index).item()
    
    return raw_data
```

#### SDS011 - Particulate Matter Sensor

**Professional-grade PM2.5/PM10 monitoring**

**Specifications:**
- **Range**: PM2.5 & PM10: 0-999.9 μg/m³
- **Accuracy**: ±15% and ±10 μg/m³
- **Interface**: UART (3.3V TTL, 9600 baud)
- **Power**: 5V, 70mA active, 4mA sleep mode
- **Lifespan**: 8000+ hours continuous operation
- **Response Time**: 1 second data update rate

**Platform Implementation:**

*Unified UART Interface:*
```python
import serial
import struct
import time

class SDS011:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.serial = serial.Serial(port, baudrate, timeout=2)
        self.serial.flushInput()
    
    def read_measurement(self):
        """Read PM2.5 and PM10 values"""
        for _ in range(10):  # Try up to 10 times
            data = self.serial.read(10)
            if len(data) == 10 and data[0] == 0xAA and data[1] == 0xC0:
                # Parse measurement data
                pm25 = struct.unpack('<H', data[2:4])[0] / 10.0
                pm10 = struct.unpack('<H', data[4:6])[0] / 10.0
                
                return {
                    'pm2_5': round(pm25, 1),
                    'pm10': round(pm10, 1),
                    'timestamp': time.time(),
                    'quality': 'good' if pm25 < 12 else 'moderate' if pm25 < 35 else 'poor'
                }
        return None
    
    def set_sleep_mode(self, sleep=True):
        """Control sensor sleep mode for power saving"""
        cmd = b'\xAA\xB4\x06\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xAB' if sleep else \
              b'\xAA\xB4\x06\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xAB'
        self.serial.write(cmd)
        self.serial.flushInput()

# Platform-specific optimizations
class SDS011Enhanced:
    def __init__(self, port='/dev/ttyUSB0', platform='auto'):
        self.sds = SDS011(port)
        self.platform = self._detect_platform() if platform == 'auto' else platform
        
        # Load platform-specific AI models
        if self.platform == 'jetson':
            import torch
            self.pm_predictor = torch.jit.load('pm_forecasting_model.pt')
        elif self.platform == 'raspberry_pi_5':
            import tflite_runtime.interpreter as tflite
            self.pm_predictor = tflite.Interpreter('pm_forecasting_model.tflite')
            self.pm_predictor.allocate_tensors()
    
    def _detect_platform(self):
        """Auto-detect hardware platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return 'raspberry_pi_5'
                elif 'jetson-nano' in model.lower():
                    return 'jetson'
        except:
            pass
        return 'unknown'
    
    def read_with_prediction(self, history_data=None):
        """Read measurement with AI-enhanced prediction"""
        measurement = self.sds.read_measurement()
        if measurement and self.platform in ['jetson', 'raspberry_pi_5']:
            # Add AI prediction for next hour
            if history_data and len(history_data) >= 12:  # Need 1 hour of data
                prediction = self._predict_next_hour(history_data)
                measurement['predicted_pm25_1h'] = prediction
        
        return measurement
```

#### Additional Sensors

**SHT40 - High-Precision Temperature/Humidity**
```python
# Universal implementation for both platforms
import board
import busio
import adafruit_sht4x

i2c = busio.I2C(board.SCL, board.SDA)
sht40 = adafruit_sht4x.SHT4x(i2c)

# Configure precision mode
sht40.mode = adafruit_sht4x.Mode.NOHEAT_HIGHPRECISION

def read_precision_climate():
    temperature, humidity = sht40.measurements
    return {
        'temperature': round(temperature, 3),  # ±0.2°C accuracy
        'humidity': round(humidity, 2),        # ±1.8% accuracy
        'comfort_index': calculate_comfort_index(temperature, humidity)
    }
```

**DS18B20 - Waterproof Temperature Sensor**
```python
# Platform-agnostic implementation
import os
import glob
import time

class DS18B20:
    def __init__(self):
        # Enable 1-wire interface
        os.system('modprobe w1-gpio')
        os.system('modprobe w1-therm')
        
        # Find sensor devices
        base_dir = '/sys/bus/w1/devices/'
        device_folders = glob.glob(base_dir + '28*')
        
        if device_folders:
            self.device_file = device_folders[0] + '/w1_slave'
        else:
            raise RuntimeError("No DS18B20 sensors found")
    
    def read_temperature(self):
        """Read temperature with error handling"""
        try:
            with open(self.device_file, 'r') as f:
                lines = f.readlines()
            
            # Check for valid reading
            if lines[0].strip()[-3:] == 'YES':
                equals_pos = lines[1].find('t=')
                if equals_pos != -1:
                    temp_string = lines[1][equals_pos+2:]
                    temperature = float(temp_string) / 1000.0
                    return {
                        'temperature': round(temperature, 2),
                        'sensor_type': 'DS18B20',
                        'timestamp': time.time()
                    }
        except Exception as e:
            print(f"DS18B20 read error: {e}")
        
        return None
```

## Communication Interfaces

### I2C Configuration (Platform-Optimized)

#### Raspberry Pi 5 I2C Setup
```python
# Optimized for RP1 I2C controller
import board
import busio
from adafruit_bus_device.i2c_device import I2CDevice

# Pi 5 supports higher I2C frequencies
i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)  # 1MHz

# Scan for devices with enhanced error handling
def scan_i2c_devices():
    devices_found = []
    for addr in range(0x08, 0x78):
        try:
            device = I2CDevice(i2c, addr)
            with device:
                devices_found.append(f"0x{addr:02X}")
        except ValueError:
            continue
    return devices_found

# Multi-sensor I2C manager
class I2CManager:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.devices = {}
        self.scan_interval = 300  # Rescan every 5 minutes
        
    def register_device(self, name, address, driver_class):
        """Register an I2C device with automatic retry"""
        try:
            device = driver_class(self.i2c, address=address)
            self.devices[name] = {
                'device': device,
                'address': address,
                'last_read': 0,
                'error_count': 0
            }
            print(f"Registered {name} at 0x{address:02X}")
        except Exception as e:
            print(f"Failed to register {name}: {e}")
    
    def read_all_sensors(self):
        """Read from all registered sensors with error handling"""
        results = {}
        for name, device_info in self.devices.items():
            try:
                # Device-specific reading logic here
                results[name] = self._read_device(device_info)
                device_info['error_count'] = 0
            except Exception as e:
                device_info['error_count'] += 1
                print(f"Error reading {name}: {e}")
                if device_info['error_count'] > 5:
                    print(f"Too many errors for {name}, removing from active devices")
        
        return results
```

#### Jetson Nano I2C Setup
```python
# Optimized for Tegra X1 I2C controller
import smbus2
import time

class JetsonI2CManager:
    def __init__(self, bus_number=1):
        self.bus = smbus2.SMBus(bus_number)
        self.devices = {}
    
    def scan_devices(self):
        """Scan I2C bus for connected devices"""
        devices = []
        for addr in range(0x08, 0x78):
            try:
                self.bus.read_byte(addr)
                devices.append(f"0x{addr:02X}")
            except OSError:
                continue
        return devices
    
    def read_sensor_data(self, address, register, num_bytes=2):
        """Read sensor data with automatic retry"""
        for attempt in range(3):
            try:
                data = self.bus.read_i2c_block_data(address, register, num_bytes)
                return data
            except OSError as e:
                if attempt == 2:
                    raise e
                time.sleep(0.1)
```

### UART Configuration for Air Quality Sensors

```python
# Universal UART configuration for both platforms
import serial
import threading
import queue
import time

class UniversalUART:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, platform='auto'):
        self.platform = self._detect_platform() if platform == 'auto' else platform
        self.port = self._get_uart_port(port)
        
        # Platform-specific optimizations
        uart_config = {
            'raspberry_pi_5': {
                'timeout': 1.0,
                'write_timeout': 1.0,
                'inter_byte_timeout': 0.1
            },
            'jetson': {
                'timeout': 2.0,
                'write_timeout': 2.0,
                'inter_byte_timeout': 0.2
            }
        }
        
        config = uart_config.get(self.platform, uart_config['raspberry_pi_5'])
        
        self.serial = serial.Serial(
            self.port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            **config
        )
        
        # Threading for non-blocking reads
        self.data_queue = queue.Queue()
        self.read_thread = threading.Thread(target=self._continuous_read, daemon=True)
        self.running = True
        self.read_thread.start()
    
    def _detect_platform(self):
        """Auto-detect hardware platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return 'raspberry_pi_5'
                elif 'jetson-nano' in model.lower():
                    return 'jetson'
        except:
            pass
        return 'raspberry_pi_5'  # Default
    
    def _get_uart_port(self, default_port):
        """Get appropriate UART port for platform"""
        if self.platform == 'raspberry_pi_5':
            # Pi 5 may use different UART ports
            possible_ports = ['/dev/ttyAMA0', '/dev/ttyUSB0', '/dev/ttyACM0']
        else:
            # Jetson typically uses ttyUSB or ttyTHS
            possible_ports = ['/dev/ttyUSB0', '/dev/ttyTHS1', '/dev/ttyACM0']
        
        # Try to find working port
        for port in possible_ports:
            try:
                test_serial = serial.Serial(port, 9600, timeout=0.1)
                test_serial.close()
                return port
            except (serial.SerialException, FileNotFoundError):
                continue
        
        return default_port  # Fallback to provided port
    
    def _continuous_read(self):
        """Continuous reading in background thread"""
        while self.running:
            try:
                if self.serial.in_waiting > 0:
                    data = self.serial.readline()
                    if data:
                        self.data_queue.put(data)
            except Exception as e:
                print(f"UART read error: {e}")
                time.sleep(0.1)
    
    def get_latest_data(self, timeout=5.0):
        """Get latest data with timeout"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        """Clean shutdown"""
        self.running = False
        self.read_thread.join(timeout=1.0)
        if self.serial.is_open:
            self.serial.close()
```

## Power Management

### Unified Power Management System

```python
import psutil
import time
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class PowerConfig:
    """Platform-agnostic power configuration"""
    battery_capacity_mah: int = 10000
    solar_panel_watts: int = 20
    low_power_threshold: float = 0.3
    critical_power_threshold: float = 0.15
    max_power_consumption_w: float = 15.0
    power_save_mode: bool = False

class UniversalPowerManager:
    """Unified power management for both platforms"""
    
    def __init__(self, config: PowerConfig, platform='auto'):
        self.config = config
        self.platform = self._detect_platform() if platform == 'auto' else platform
        self.power_states = {
            'normal': {'cpu_freq': 'max', 'monitoring_interval': 60},
            'power_save': {'cpu_freq': 'conservative', 'monitoring_interval': 300},
            'critical': {'cpu_freq': 'powersave', 'monitoring_interval': 900}
        }
        self.current_state = 'normal'
        
        # Platform-specific initialization
        if self.platform == 'raspberry_pi_5':
            self._init_pi5_power_management()
        elif self.platform == 'jetson':
            self._init_jetson_power_management()
    
    def _detect_platform(self):
        """Auto-detect hardware platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return 'raspberry_pi_5'
                elif 'jetson-nano' in model.lower():
                    return 'jetson'
        except:
            pass
        return 'unknown'
    
    def _init_pi5_power_management(self):
        """Initialize Raspberry Pi 5 specific power features"""
        # Enable advanced power management features
        try:
            # Configure USB power management
            import subprocess
            subprocess.run(['sudo', 'sh', '-c', 
                          'echo auto > /sys/bus/usb/devices/*/power/control'], 
                          check=False)
            
            # Configure HDMI power management (turn off for headless)
            subprocess.run(['sudo', 'tvservice', '-o'], check=False)
            
            print("Pi 5 power management initialized")
        except Exception as e:
            print(f"Pi 5 power management warning: {e}")
    
    def _init_jetson_power_management(self):
        """Initialize Jetson Nano specific power features"""
        try:
            import subprocess
            
            # Set power mode to 5W (MAXN for 10W)
            subprocess.run(['sudo', 'nvpmodel', '-m', '1'], check=False)
            
            # Enable jetson_clocks for performance mode
            if not self.config.power_save_mode:
                subprocess.run(['sudo', 'jetson_clocks'], check=False)
            
            print("Jetson power management initialized")
        except Exception as e:
            print(f"Jetson power management warning: {e}")
    
    def get_power_status(self) -> Dict:
        """Get comprehensive power status"""
        status = {
            'platform': self.platform,
            'current_state': self.current_state,
            'timestamp': time.time()
        }
        
        if self.platform == 'raspberry_pi_5':
            status.update(self._get_pi5_power_status())
        elif self.platform == 'jetson':
            status.update(self._get_jetson_power_status())
        else:
            status.update(self._get_generic_power_status())
        
        return status
    
    def _get_pi5_power_status(self) -> Dict:
        """Get Raspberry Pi 5 power status"""
        try:
            # Read power supply status
            with open('/sys/class/power_supply/rpi-poe-power/online', 'r') as f:
                poe_power = f.read().strip() == '1'
        except:
            poe_power = False
        
        try:
            # CPU temperature and frequency
            cpu_temp = psutil.sensors_temperatures().get('cpu_thermal', [{}])[0].get('current', 0)
            
            # Estimate power consumption based on CPU usage and temperature
            cpu_usage = psutil.cpu_percent(interval=1)
            estimated_power = 3.0 + (cpu_usage / 100.0 * 5.0) + max(0, (cpu_temp - 50) * 0.1)
            
        except:
            cpu_temp = 0
            estimated_power = 5.0
        
        return {
            'poe_powered': poe_power,
            'cpu_temperature': cpu_temp,
            'estimated_power_w': round(estimated_power, 1),
            'thermal_throttling': cpu_temp > 80
        }
    
    def _get_jetson_power_status(self) -> Dict:
        """Get Jetson Nano power status"""
        try:
            import subprocess
            
            # Get power mode
            result = subprocess.run(['cat', '/etc/nvpmodel.conf'], 
                                  capture_output=True, text=True)
            power_mode = 'MAXN' if '0' in result.stdout else '5W'
            
            # GPU usage
            try:
                gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw', 
                                           '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                gpu_util, gpu_temp, gpu_power = gpu_result.stdout.strip().split(', ')
            except:
                gpu_util, gpu_temp, gpu_power = '0', '0', '0'
            
        except:
            power_mode = 'Unknown'
            gpu_util, gpu_temp, gpu_power = '0', '0', '0'
        
        return {
            'power_mode': power_mode,
            'gpu_utilization': int(gpu_util),
            'gpu_temperature': int(gpu_temp),
            'gpu_power_w': float(gpu_power),
            'estimated_total_power_w': float(gpu_power) + 3.0  # Add CPU baseline
        }
    
    def _get_generic_power_status(self) -> Dict:
        """Get generic power status for unknown platforms"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'estimated_power_w': 5.0
        }
    
    def set_power_mode(self, mode: str):
        """Set system power mode"""
        if mode not in self.power_states:
            raise ValueError(f"Invalid power mode: {mode}")
        
        self.current_state = mode
        mode_config = self.power_states[mode]
        
        try:
            if self.platform == 'raspberry_pi_5':
                self._set_pi5_power_mode(mode_config)
            elif self.platform == 'jetson':
                self._set_jetson_power_mode(mode_config)
            
            print(f"Power mode set to: {mode}")
        except Exception as e:
            print(f"Failed to set power mode: {e}")
    
    def _set_pi5_power_mode(self, config: Dict):
        """Set Raspberry Pi 5 power mode"""
        import subprocess
        
        if config['cpu_freq'] == 'powersave':
            # Reduce CPU frequency
            subprocess.run(['sudo', 'cpufreq-set', '-g', 'powersave'], check=False)
        elif config['cpu_freq'] == 'conservative':
            subprocess.run(['sudo', 'cpufreq-set', '-g', 'conservative'], check=False)
        else:
            subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'], check=False)
    
    def _set_jetson_power_mode(self, config: Dict):
        """Set Jetson Nano power mode"""
        import subprocess
        
        if config['cpu_freq'] == 'powersave':
            # 5W mode
            subprocess.run(['sudo', 'nvpmodel', '-m', '1'], check=False)
        else:
            # 10W MAXN mode
            subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=False)
            if config['cpu_freq'] == 'max':
                subprocess.run(['sudo', 'jetson_clocks'], check=False)
    
    def adaptive_power_management(self, battery_level: Optional[float] = None):
        """Automatically adjust power mode based on conditions"""
        if battery_level is None:
            # Estimate battery level (placeholder - would integrate with actual battery monitoring)
            battery_level = 0.7
        
        # Determine appropriate power mode
        if battery_level < self.config.critical_power_threshold:
            target_mode = 'critical'
        elif battery_level < self.config.low_power_threshold:
            target_mode = 'power_save'
        else:
            target_mode = 'normal'
        
        if target_mode != self.current_state:
            self.set_power_mode(target_mode)
            return True
        
        return False

# Platform-specific solar charging integration
class SolarChargingController:
    """Solar charging controller for both platforms"""
    
    def __init__(self, platform='auto'):
        self.platform = platform
        self.charging_enabled = False
        
        # Initialize ADC for solar panel monitoring
        if self.platform == 'raspberry_pi_5':
            self._init_pi5_adc()
        elif self.platform == 'jetson':
            self._init_jetson_adc()
    
    def _init_pi5_adc(self):
        """Initialize ADC for Pi 5 (via MCP3008)"""
        try:
            import board
            import busio
            import digitalio
            import adafruit_mcp3xxx.mcp3008 as MCP
            from adafruit_mcp3xxx.analog_in import AnalogIn
            
            # SPI setup for MCP3008
            spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
            cs = digitalio.DigitalInOut(board.D8)
            mcp = MCP.MCP3008(spi, cs)
            
            # ADC channels
            self.solar_voltage_adc = AnalogIn(mcp, MCP.P0)
            self.battery_voltage_adc = AnalogIn(mcp, MCP.P1)
            
        except Exception as e:
            print(f"Pi 5 ADC initialization failed: {e}")
            self.solar_voltage_adc = None
            self.battery_voltage_adc = None
    
    def _init_jetson_adc(self):
        """Initialize ADC for Jetson (built-in)"""
        try:
            # Jetson Nano has built-in ADC
            self.adc_base_path = '/sys/bus/iio/devices/iio:device0'
        except Exception as e:
            print(f"Jetson ADC initialization failed: {e}")
            self.adc_base_path = None
    
    def read_solar_status(self) -> Dict:
        """Read solar panel and battery status"""
        if self.platform == 'raspberry_pi_5' and self.solar_voltage_adc:
            solar_voltage = self.solar_voltage_adc.voltage * 2  # Voltage divider
            battery_voltage = self.battery_voltage_adc.voltage * 3  # Voltage divider
        elif self.platform == 'jetson' and self.adc_base_path:
            # Read from Jetson ADC
            try:
                with open(f'{self.adc_base_path}/in_voltage0_raw', 'r') as f:
                    solar_raw = int(f.read().strip())
                with open(f'{self.adc_base_path}/in_voltage1_raw', 'r') as f:
                    battery_raw = int(f.read().strip())
                
                # Convert to voltage (12-bit ADC, 1.8V reference)
                solar_voltage = (solar_raw / 4095.0) * 1.8 * 4  # Scale factor
                battery_voltage = (battery_raw / 4095.0) * 1.8 * 5  # Scale factor
            except:
                solar_voltage = 0.0
                battery_voltage = 12.0  # Assume good battery
        else:
            # Fallback values
            solar_voltage = 5.0
            battery_voltage = 12.0
        
        # Calculate charging status
        charging = solar_voltage > (battery_voltage + 0.5)
        battery_percentage = max(0, min(100, (battery_voltage - 10.5) / 2.1 * 100))
        
        return {
            'solar_voltage': round(solar_voltage, 2),
            'battery_voltage': round(battery_voltage, 2),
            'battery_percentage': round(battery_percentage, 1),
            'charging': charging,
            'solar_power_available': solar_voltage > 5.0
        }
```

## Performance Optimization

### Platform-Specific Optimizations

#### Raspberry Pi 5 Optimizations

```python
import subprocess
import os

class Pi5Optimizer:
    """Raspberry Pi 5 specific optimizations"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def apply_system_optimizations(self):
        """Apply system-level optimizations for Pi 5"""
        optimizations = [
            self._optimize_memory,
            self._optimize_gpu_split,
            self._optimize_i2c_baudrate,
            self._optimize_uart,
            self._disable_unnecessary_services
        ]
        
        for optimization in optimizations:
            try:
                optimization()
            except Exception as e:
                print(f"Optimization failed: {e}")
    
    def _optimize_memory(self):
        """Optimize memory allocation"""
        # Increase GPU memory split for video processing
        config_lines = [
            'gpu_mem=128',
            'arm_64bit=1',
            'max_framebuffers=2',
            'disable_overscan=1'
        ]
        
        self._append_to_config('/boot/config.txt', config_lines)
        self.optimizations_applied.append('memory_optimization')
    
    def _optimize_gpu_split(self):
        """Optimize GPU memory split for ML workloads"""
        subprocess.run(['sudo', 'raspi-config', 'nonint', 'do_memory_split', '128'], 
                      check=False)
        self.optimizations_applied.append('gpu_split')
    
    def _optimize_i2c_baudrate(self):
        """Increase I2C speed for faster sensor communication"""
        config_lines = [
            'dtparam=i2c_arm=on',
            'dtparam=i2c_arm_baudrate=400000'  # 400kHz I2C
        ]
        self._append_to_config('/boot/config.txt', config_lines)
        self.optimizations_applied.append('i2c_optimization')
    
    def _optimize_uart(self):
        """Enable and optimize UART for sensors"""
        config_lines = [
            'enable_uart=1',
            'dtoverlay=disable-bt'  # Disable Bluetooth to free up UART
        ]
        self._append_to_config('/boot/config.txt', config_lines)
        self.optimizations_applied.append('uart_optimization')
    
    def _disable_unnecessary_services(self):
        """Disable unnecessary services to save resources"""
        services_to_disable = [
            'bluetooth',
            'hciuart',
            'cups',
            'cups-browsed'
        ]
        
        for service in services_to_disable:
            try:
                subprocess.run(['sudo', 'systemctl', 'disable', service], 
                              check=False, capture_output=True)
            except:
                pass
        
        self.optimizations_applied.append('service_optimization')
    
    def _append_to_config(self, config_file, lines):
        """Safely append lines to config file"""
        try:
            with open(config_file, 'r') as f:
                existing_content = f.read()
            
            lines_to_add = []
            for line in lines:
                if line not in existing_content:
                    lines_to_add.append(line)
            
            if lines_to_add:
                with open(config_file, 'a') as f:
                    f.write('\n# Furcate Nano optimizations\n')
                    for line in lines_to_add:
                        f.write(f'{line}\n')
                        
        except Exception as e:
            print(f"Config file update failed: {e}")
```

#### NVIDIA Jetson Nano Optimizations

```python
import subprocess
import torch

class JetsonOptimizer:
    """NVIDIA Jetson Nano specific optimizations"""
    
    def __init__(self):
        self.optimizations_applied = []
        self.jetson_clocks_enabled = False
    
    def apply_system_optimizations(self):
        """Apply system-level optimizations for Jetson"""
        optimizations = [
            self._set_power_mode,
            self._enable_jetson_clocks,
            self._optimize_cuda_settings,
            self._configure_swap,
            self._optimize_filesystem
        ]
        
        for optimization in optimizations:
            try:
                optimization()
            except Exception as e:
                print(f"Jetson optimization failed: {e}")
    
    def _set_power_mode(self):
        """Set optimal power mode"""
        # Set to MAXN mode (10W) for maximum performance
        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=False)
        self.optimizations_applied.append('power_mode_maxn')
    
    def _enable_jetson_clocks(self):
        """Enable jetson_clocks for maximum performance"""
        subprocess.run(['sudo', 'jetson_clocks'], check=False)
        self.jetson_clocks_enabled = True
        self.optimizations_applied.append('jetson_clocks')
    
    def _optimize_cuda_settings(self):
        """Optimize CUDA settings for ML workloads"""
        # Set CUDA cache directory
        os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache'
        
        # Optimize PyTorch for Jetson
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set memory growth to avoid OOM
            torch.cuda.empty_cache()
            
        self.optimizations_applied.append('cuda_optimization')
    
    def _configure_swap(self):
        """Configure swap for memory-intensive operations"""
        try:
            # Create 4GB swap file
            swap_commands = [
                ['sudo', 'fallocate', '-l', '4G', '/swapfile'],
                ['sudo', 'chmod', '600', '/swapfile'],
                ['sudo', 'mkswap', '/swapfile'],
                ['sudo', 'swapon', '/swapfile']
            ]
            
            for cmd in swap_commands:
                subprocess.run(cmd, check=False)
            
            # Add to fstab for persistence
            with open('/etc/fstab', 'a') as f:
                f.write('/swapfile none swap sw 0 0\n')
                
            self.optimizations_applied.append('swap_configuration')
        except Exception as e:
            print(f"Swap configuration failed: {e}")
    
    def _optimize_filesystem(self):
        """Optimize filesystem for better I/O performance"""
        try:
            # Increase write cache
            subprocess.run(['sudo', 'sysctl', 'vm.dirty_ratio=20'], check=False)
            subprocess.run(['sudo', 'sysctl', 'vm.dirty_background_ratio=10'], check=False)
            
            self.optimizations_applied.append('filesystem_optimization')
        except Exception as e:
            print(f"Filesystem optimization failed: {e}")
    
    def optimize_for_ml_inference(self):
        """Specific optimizations for ML inference"""
        if torch.cuda.is_available():
            # Enable TensorRT optimization
            try:
                import torch_tensorrt
                
                # Set optimal TensorRT settings
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                
                self.optimizations_applied.append('tensorrt_optimization')
            except ImportError:
                print("TensorRT not available, using standard PyTorch")
            
            # CUDA optimization
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
    def monitor_performance(self):
        """Monitor Jetson performance metrics"""
        try:
            # GPU utilization
            gpu_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'], 
                capture_output=True, text=True
            )
            
            if gpu_result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = gpu_result.stdout.strip().split(', ')
                
                return {
                    'gpu_utilization': int(gpu_util),
                    'memory_used_mb': int(mem_used),
                    'memory_total_mb': int(mem_total),
                    'temperature': int(temp),
                    'jetson_clocks_enabled': self.jetson_clocks_enabled,
                    'optimizations_applied': self.optimizations_applied
                }
        except Exception as e:
            print(f"Performance monitoring failed: {e}")
        
        return {}
```

### Unified Performance Monitoring

```python
import psutil
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    """Platform-agnostic performance metrics"""
    cpu_usage: float
    memory_usage: float
    temperature: float
    disk_usage: float
    network_io: Dict
    timestamp: float
    platform_specific: Dict

class UnifiedPerformanceMonitor:
    """Cross-platform performance monitoring"""
    
    def __init__(self, platform='auto'):
        self.platform = self._detect_platform() if platform == 'auto' else platform
        self.history = []
        self.max_history = 1000
        
        # Initialize platform-specific monitoring
        if self.platform == 'raspberry_pi_5':
            self.pi_optimizer = Pi5Optimizer()
        elif self.platform == 'jetson':
            self.jetson_optimizer = JetsonOptimizer()
    
    def _detect_platform(self):
        """Auto-detect hardware platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return 'raspberry_pi_5'
                elif 'jetson-nano' in model.lower():
                    return 'jetson'
        except:
            pass
        return 'unknown'
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        # Common metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        network_io = psutil.net_io_counters()._asdict()
        
        # Temperature (platform-specific)
        temperature = self._get_cpu_temperature()
        
        # Platform-specific metrics
        platform_specific = {}
        if self.platform == 'raspberry_pi_5':
            platform_specific = self._get_pi5_metrics()
        elif self.platform == 'jetson':
            platform_specific = self._get_jetson_metrics()
        
        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_info.percent,
            temperature=temperature,
            disk_usage=disk_info.used / disk_info.total * 100,
            network_io=network_io,
            timestamp=time.time(),
            platform_specific=platform_specific
        )
        
        # Store in history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return metrics
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature for both platforms"""
        try:
            if self.platform == 'raspberry_pi_5':
                # Pi 5 temperature
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
            elif self.platform == 'jetson':
                # Jetson temperature zones
                with open('/sys/class/thermal/thermal_zone1/temp', 'r') as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
        except:
            pass
        
        # Fallback: try psutil
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                if entries:
                    return entries[0].current
        except:
            pass
        
        return 0.0
    
    def _get_pi5_metrics(self) -> Dict:
        """Get Raspberry Pi 5 specific metrics"""
        metrics = {}
        
        try:
            # GPU memory usage
            result = subprocess.run(['vcgencmd', 'get_mem', 'gpu'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_mem = result.stdout.strip().split('=')[1]
                metrics['gpu_memory'] = gpu_mem
        except:
            pass
        
        try:
            # Voltage and frequency
            voltage_result = subprocess.run(['vcgencmd', 'measure_volts'], 
                                          capture_output=True, text=True)
            freq_result = subprocess.run(['vcgencmd', 'measure_clock', 'arm'], 
                                       capture_output=True, text=True)
            
            if voltage_result.returncode == 0:
                voltage = voltage_result.stdout.strip().split('=')[1]
                metrics['core_voltage'] = voltage
            
            if freq_result.returncode == 0:
                frequency = freq_result.stdout.strip().split('=')[1]
                metrics['cpu_frequency'] = frequency
                
        except:
            pass
        
        return metrics
    
    def _get_jetson_metrics(self) -> Dict:
        """Get Jetson Nano specific metrics"""
        metrics = {}
        
        try:
            # GPU metrics via nvidia-smi
            gpu_result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if gpu_result.returncode == 0:
                gpu_data = gpu_result.stdout.strip().split(', ')
                metrics.update({
                    'gpu_utilization': int(gpu_data[0]),
                    'gpu_memory_used': int(gpu_data[1]),
                    'gpu_memory_total': int(gpu_data[2]),
                    'gpu_temperature': int(gpu_data[3]),
                    'gpu_power_draw': float(gpu_data[4])
                })
        except:
            pass
        
        try:
            # Jetson-specific power mode
            power_result = subprocess.run(['cat', '/etc/nvpmodel.conf'], 
                                        capture_output=True, text=True)
            if 'MODE_10W' in power_result.stdout:
                metrics['power_mode'] = '10W_MAXN'
            else:
                metrics['power_mode'] = '5W'
        except:
            metrics['power_mode'] = 'unknown'
        
        return metrics
    
    def get_performance_summary(self, hours=1) -> Dict:
        """Get performance summary over specified time period"""
        if not self.history:
            return {}
        
        # Filter history by time period
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            recent_metrics = self.history[-10:]  # Last 10 measurements
        
        # Calculate statistics
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        temperatures = [m.temperature for m in recent_metrics]
        
        summary = {
            'time_period_hours': hours,
            'measurements': len(recent_metrics),
            'cpu_usage': {
                'average': sum(cpu_usage) / len(cpu_usage),
                'max': max(cpu_usage),
                'min': min(cpu_usage)
            },
            'memory_usage': {
                'average': sum(memory_usage) / len(memory_usage),
                'max': max(memory_usage),
                'min': min(memory_usage)
            },
            'temperature': {
                'average': sum(temperatures) / len(temperatures),
                'max': max(temperatures),
                'min': min(temperatures)
            },
            'platform': self.platform
        }
        
        # Add platform-specific summary
        if self.platform == 'jetson' and recent_metrics:
            gpu_utils = [m.platform_specific.get('gpu_utilization', 0) for m in recent_metrics]
            if gpu_utils:
                summary['gpu_utilization'] = {
                    'average': sum(gpu_utils) / len(gpu_utils),
                    'max': max(gpu_utils),
                    'min': min(gpu_utils)
                }
        
        return summary
    
    def apply_optimizations(self):
        """Apply platform-specific optimizations"""
        try:
            if self.platform == 'raspberry_pi_5':
                self.pi_optimizer.apply_system_optimizations()
                return self.pi_optimizer.optimizations_applied
            elif self.platform == 'jetson':
                self.jetson_optimizer.apply_system_optimizations()
                self.jetson_optimizer.optimize_for_ml_inference()
                return self.jetson_optimizer.optimizations_applied
        except Exception as e:
            print(f"Optimization failed: {e}")
        
        return []
```

## Assembly and Integration

### Unified Assembly Guide

```python
import yaml
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class AssemblyConfiguration:
    """Configuration for Furcate Nano assembly"""
    platform: str
    sensors: List[str]
    power_source: str
    connectivity: List[str]
    enclosure_type: str
    deployment_environment: str

class FurcateNanoAssembly:
    """Unified assembly guide for both platforms"""
    
    def __init__(self, config: AssemblyConfiguration):
        self.config = config
        self.assembly_steps = []
        self.validation_tests = []
    
    def generate_assembly_guide(self) -> Dict:
        """Generate platform-specific assembly guide"""
        if self.config.platform == 'raspberry_pi_5':
            return self._generate_pi5_assembly()
        elif self.config.platform == 'jetson_nano':
            return self._generate_jetson_assembly()
        else:
            raise ValueError(f"Unsupported platform: {self.config.platform}")
    
    def _generate_pi5_assembly(self) -> Dict:
        """Generate Raspberry Pi 5 assembly guide"""
        steps = [
            {
                "step": 1,
                "title": "Prepare Raspberry Pi 5 Base",
                "description": "Install heatsink and prepare the Pi 5 board",
                "details": [
                    "Install official Pi 5 active cooler or third-party heatsink",
                    "Ensure thermal paste is properly applied",
                    "Mount Pi 5 in case with GPIO access",
                    "Connect 27W USB-C power supply"
                ],
                "warnings": [
                    "Handle board by edges only",
                    "Ensure proper ESD protection",
                    "Do not force components"
                ]
            },
            {
                "step": 2,
                "title": "Install Environmental Sensors",
                "description": "Connect sensors using GPIO and I2C",
                "details": self._get_sensor_connections('raspberry_pi_5'),
                "wiring_diagram": "pi5_sensor_wiring.png"
            },
            {
                "step": 3,
                "title": "Setup Communication Modules", 
                "description": "Install LoRa, WiFi, and other communication hardware",
                "details": [
                    "Install LoRa HAT if using long-range communication",
                    "Verify WiFi 6 antenna connection",
                    "Test Bluetooth 5.0 functionality",
                    "Configure Ethernet if used"
                ]
            },
            {
                "step": 4,
                "title": "Power System Integration",
                "description": "Setup solar charging and battery backup",
                "details": self._get_power_setup('raspberry_pi_5')
            },
            {
                "step": 5,
                "title": "Software Installation",
                "description": "Install and configure Furcate Nano software",
                "details": [
                    "Flash Raspberry Pi OS Bookworm (64-bit)",
                    "Install Furcate Nano package",
                    "Configure TensorFlow Lite optimizations",
                    "Setup GPIO libraries (gpiod, GPIO Zero)",
                    "Test all sensor connections"
                ]
            }
        ]
        
        return {
            "platform": "Raspberry Pi 5",
            "assembly_time": "2-3 hours",
            "difficulty": "Intermediate",
            "tools_required": [
                "Screwdriver set", "Anti-static wrist strap", 
                "Multimeter", "Wire strippers", "Soldering iron (optional)"
            ],
            "steps": steps,
            "validation": self._get_validation_tests('raspberry_pi_5')
        }
    
    def _generate_jetson_assembly(self) -> Dict:
        """Generate Jetson Nano assembly guide"""
        steps = [
            {
                "step": 1,
                "title": "Prepare Jetson Nano Base",
                "description": "Setup Jetson Nano with cooling and power",
                "details": [
                    "Install large heatsink with fan (recommended)",
                    "Mount Jetson in case with GPIO access",
                    "Connect 5V 4A barrel jack power supply",
                    "Insert microSD card (64GB+ recommended)"
                ],
                "warnings": [
                    "Jetson runs hot - cooling is essential",
                    "Use quality power supply for stability",
                    "Handle SoM carefully"
                ]
            },
            {
                "step": 2,
                "title": "Install Environmental Sensors",
                "description": "Connect sensors using GPIO and I2C",
                "details": self._get_sensor_connections('jetson_nano'),
                "wiring_diagram": "jetson_sensor_wiring.png"
            },
            {
                "step": 3,
                "title": "Setup Communication Modules",
                "description": "Install communication hardware",
                "details": [
                    "Install M.2 WiFi module if needed",
                    "Connect LoRa module via GPIO/SPI",
                    "Verify Ethernet connectivity",
                    "Test Bluetooth functionality"
                ]
            },
            {
                "step": 4,
                "title": "Power System Integration", 
                "description": "Setup advanced power management",
                "details": self._get_power_setup('jetson_nano')
            },
            {
                "step": 5,
                "title": "Software Installation",
                "description": "Install JetPack and Furcate Nano",
                "details": [
                    "Flash JetPack 4.6.x SD card image",
                    "Install Furcate Nano with PyTorch support",
                    "Configure CUDA and TensorRT",
                    "Setup Jetson.GPIO library",
                    "Install PyTorch and torchvision",
                    "Test GPU acceleration"
                ]
            }
        ]
        
        return {
            "platform": "NVIDIA Jetson Nano",
            "assembly_time": "3-4 hours",
            "difficulty": "Advanced",
            "tools_required": [
                "Screwdriver set", "Anti-static wrist strap",
                "Multimeter", "Wire strippers", "Thermal paste",
                "Soldering iron (for advanced connections)"
            ],
            "steps": steps,
            "validation": self._get_validation_tests('jetson_nano')
        }
    
    def _get_sensor_connections(self, platform: str) -> List[str]:
        """Get sensor connection details for platform"""
        base_connections = [
            "BME688: Connect VCC to 3.3V (Pin 1), GND to GND (Pin 6), SDA to GPIO 2 (Pin 3), SCL to GPIO 3 (Pin 5)",
            "SDS011: Connect 5V to Pin 2, GND to Pin 6, TX to GPIO 15 (Pin 10), RX to GPIO 14 (Pin 8)",
            "Status LEDs: Green LED to GPIO 18 (Pin 12), Red LED to GPIO 19 (Pin 35)"
        ]
        
        if platform == 'raspberry_pi_5':
            base_connections.extend([
                "Use RP1 I2C controller for improved timing",
                "Enable I2C with 400kHz baudrate in config.txt",
                "Connect 4.7kΩ pull-up resistors on I2C lines if needed"
            ])
        elif platform == 'jetson_nano':
            base_connections.extend([
                "Use Tegra X1 I2C controller",
                "Verify 3.3V logic compatibility",
                "Consider level shifters for 5V sensors"
            ])
        
        return base_connections
    
    def _get_power_setup(self, platform: str) -> List[str]:
        """Get power setup instructions for platform"""
        if platform == 'raspberry_pi_5':
            return [
                "Use 27W USB-C power supply for reliable operation",
                "Install PoE+ HAT if using Power over Ethernet", 
                "Connect solar charge controller to 5V input",
                "Add 18650 battery pack with protection circuit",
                "Install power monitoring via MCP3008 ADC",
                "Configure undervoltage detection"
            ]
        elif platform == 'jetson_nano':
            return [
                "Use 5V 4A barrel jack power supply",
                "Connect solar MPPT charge controller",
                "Install 12V battery pack with voltage regulation",
                "Add power monitoring via built-in ADC",
                "Configure nvpmodel for power optimization",
                "Setup automatic power mode switching"
            ]
        
        return []
    
    def _get_validation_tests(self, platform: str) -> List[Dict]:
        """Get validation tests for platform"""
        base_tests = [
            {
                "test": "Hardware Detection",
                "command": "sudo i2cdetect -y 1",
                "expected": "Devices detected at 0x77 (BME688)",
                "troubleshooting": "Check wiring and pull-up resistors"
            },
            {
                "test": "Sensor Reading",
                "command": "python3 -c \"from furcate_nano import HardwareManager; import asyncio; print(asyncio.run(HardwareManager().read_all_sensors()))\"",
                "expected": "Temperature, humidity, pressure readings",
                "troubleshooting": "Verify sensor power and I2C communication"
            },
            {
                "test": "GPIO Control",
                "command": "python3 -c \"import gpiozero; led = gpiozero.LED(18); led.on(); import time; time.sleep(1); led.off()\"",
                "expected": "LED blinks once",
                "troubleshooting": "Check GPIO library installation and permissions"
            }
        ]
        
        if platform == 'raspberry_pi_5':
            base_tests.extend([
                {
                    "test": "TensorFlow Lite",
                    "command": "python3 -c \"import tflite_runtime.interpreter as tflite; print('TFLite OK')\"",
                    "expected": "TFLite OK",
                    "troubleshooting": "Install tflite-runtime package"
                },
                {
                    "test": "Pi 5 Specific Features",
                    "command": "cat /proc/device-tree/model",
                    "expected": "Raspberry Pi 5",
                    "troubleshooting": "Verify Pi 5 board and OS compatibility"
                }
            ])
        elif platform == 'jetson_nano':
            base_tests.extend([
                {
                    "test": "CUDA Availability",
                    "command": "python3 -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"",
                    "expected": "CUDA available: True",
                    "troubleshooting": "Install PyTorch for Jetson, verify CUDA installation"
                },
                {
                    "test": "GPU Monitoring",
                    "command": "nvidia-smi",
                    "expected": "GPU information displayed",
                    "troubleshooting": "Check NVIDIA drivers and JetPack installation"
                },
                {
                    "test": "TensorRT",
                    "command": "python3 -c \"import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')\"",
                    "expected": "TensorRT version displayed",
                    "troubleshooting": "Verify TensorRT installation in JetPack"
                }
            ])
        
        return base_tests

class DeploymentValidator:
    """Validate Furcate Nano deployment"""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.test_results = {}
    
    async def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive deployment validation"""
        tests = [
            self._test_hardware_detection,
            self._test_sensor_readings,
            self._test_ml_inference,
            self._test_network_connectivity,
            self._test_power_management,
            self._test_storage_systems
        ]
        
        results = {
            'platform': self.platform,
            'test_timestamp': time.time(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        for test in tests:
            test_name = test.__name__.replace('_test_', '')
            try:
                test_result = await test()
                results['test_details'][test_name] = test_result
                if test_result.get('passed', False):
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1
            except Exception as e:
                results['test_details'][test_name] = {
                    'passed': False,
                    'error': str(e),
                    'details': 'Test execution failed'
                }
                results['tests_failed'] += 1
        
        results['overall_success'] = results['tests_failed'] == 0
        results['success_rate'] = results['tests_passed'] / (results['tests_passed'] + results['tests_failed'])
        
        return results
    
    async def _test_hardware_detection(self) -> Dict:
        """Test hardware component detection"""
        try:
            from furcate_nano import HardwareManager
            
            hw_manager = HardwareManager({'simulation': False})
            await hw_manager.initialize()
            
            detected_sensors = await hw_manager.detect_sensors()
            
            return {
                'passed': len(detected_sensors) > 0,
                'details': f"Detected {len(detected_sensors)} sensors",
                'sensor_list': list(detected_sensors.keys())
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Hardware detection failed'
            }
    
    async def _test_sensor_readings(self) -> Dict:
        """Test sensor data reading"""
        try:
            from furcate_nano import HardwareManager
            
            hw_manager = HardwareManager({'simulation': False})
            await hw_manager.initialize()
            
            readings = await hw_manager.read_all_sensors()
            
            # Validate readings are reasonable
            valid_readings = 0
            total_readings = 0
            
            for sensor_name, data in readings.items():
                if isinstance(data, dict) and 'value' in data:
                    total_readings += 1
                    value = data['value']
                    
                    # Basic sanity checks
                    if sensor_name == 'temperature' and -50 <= value <= 80:
                        valid_readings += 1
                    elif sensor_name == 'humidity' and 0 <= value <= 100:
                        valid_readings += 1
                    elif sensor_name == 'pressure' and 800 <= value <= 1200:
                        valid_readings += 1
                    else:
                        valid_readings += 1  # Accept other sensors
            
            success_rate = valid_readings / total_readings if total_readings > 0 else 0
            
            return {
                'passed': success_rate > 0.5,
                'details': f"Valid readings: {valid_readings}/{total_readings}",
                'success_rate': success_rate,
                'readings': readings
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Sensor reading test failed'
            }
    
    async def _test_ml_inference(self) -> Dict:
        """Test ML inference capability"""
        try:
            if self.platform == 'raspberry_pi_5':
                # Test TensorFlow Lite
                import tflite_runtime.interpreter as tflite
                import numpy as np
                
                # Create dummy model test
                test_data = np.random.random((1, 10)).astype(np.float32)
                
                return {
                    'passed': True,
                    'details': 'TensorFlow Lite available',
                    'framework': 'TensorFlow Lite'
                }
                
            elif self.platform == 'jetson_nano':
                # Test PyTorch + CUDA
                import torch
                
                # Test CUDA availability
                cuda_available = torch.cuda.is_available()
                
                if cuda_available:
                    # Test basic tensor operations on GPU
                    test_tensor = torch.randn(100, 100).cuda()
                    result = torch.matmul(test_tensor, test_tensor)
                    
                    return {
                        'passed': True,
                        'details': 'PyTorch with CUDA working',
                        'framework': 'PyTorch + CUDA',
                        'gpu_name': torch.cuda.get_device_name(0)
                    }
                else:
                    return {
                        'passed': False,
                        'details': 'CUDA not available',
                        'framework': 'PyTorch (CPU only)'
                    }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'ML inference test failed'
            }
    
    async def _test_network_connectivity(self) -> Dict:
        """Test network connectivity"""
        try:
            import subprocess
            import socket
            
            # Test internet connectivity
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, text=True, timeout=10)
            internet_ok = result.returncode == 0
            
            # Test local network
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Test WiFi if available
            wifi_info = {}
            try:
                iwconfig_result = subprocess.run(['iwconfig'], 
                                               capture_output=True, text=True)
                if 'ESSID' in iwconfig_result.stdout:
                    wifi_info['wifi_available'] = True
            except:
                wifi_info['wifi_available'] = False
            
            return {
                'passed': internet_ok,
                'details': f"Internet: {'OK' if internet_ok else 'Failed'}",
                'local_ip': local_ip,
                'hostname': hostname,
                'wifi_info': wifi_info
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Network connectivity test failed'
            }
    
    async def _test_power_management(self) -> Dict:
        """Test power management systems"""
        try:
            from furcate_nano.power import UniversalPowerManager, PowerConfig
            
            power_config = PowerConfig()
            power_manager = UniversalPowerManager(power_config, self.platform)
            
            power_status = power_manager.get_power_status()
            
            return {
                'passed': True,
                'details': 'Power management functional',
                'power_status': power_status
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Power management test failed'
            }
    
    async def _test_storage_systems(self) -> Dict:
        """Test storage and database systems"""
        try:
            import os
            import sqlite3
            import tempfile
            
            # Test database creation
            with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as tmp_db:
                conn = sqlite3.connect(tmp_db.name)
                cursor = conn.cursor()
                
                # Create test table
                cursor.execute('''
                    CREATE TABLE test_data (
                        id INTEGER PRIMARY KEY,
                        timestamp REAL,
                        value REAL
                    )
                ''')
                
                # Insert test data
                cursor.execute('INSERT INTO test_data (timestamp, value) VALUES (?, ?)',
                             (time.time(), 23.5))
                
                conn.commit()
                
                # Read back test data
                cursor.execute('SELECT * FROM test_data')
                result = cursor.fetchone()
                
                conn.close()
                
                return {
                    'passed': result is not None,
                    'details': 'Database operations successful',
                    'test_result': result
                }
                
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Storage system test failed'
            }

# Usage example for assembly validation
async def validate_deployment(platform: str):
    """Run complete deployment validation"""
    validator = DeploymentValidator(platform)
    results = await validator.run_comprehensive_tests()
    
    print(f"\n=== Furcate Nano Deployment Validation ===")
    print(f"Platform: {results['platform']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Overall Success: {results['overall_success']}")
    
    print(f"\n=== Test Details ===")
    for test_name, test_result in results['test_details'].items():
        status = "✅ PASS" if test_result.get('passed', False) else "❌ FAIL"
        print(f"{status} {test_name}: {test_result.get('details', 'No details')}")
        
        if not test_result.get('passed', False) and 'error' in test_result:
            print(f"    Error: {test_result['error']}")
    
    return results
```

## Troubleshooting

### Platform-Specific Troubleshooting Guide

```python
import subprocess
import os
import time
from typing import Dict, List, Tuple

class TroubleshootingGuide:
    """Comprehensive troubleshooting for both platforms"""
    
    def __init__(self, platform: str = 'auto'):
        self.platform = self._detect_platform() if platform == 'auto' else platform
        self.common_issues = self._load_common_issues()
        
    def _detect_platform(self) -> str:
        """Auto-detect hardware platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return 'raspberry_pi_5'
                elif 'jetson-nano' in model.lower():
                    return 'jetson_nano'
        except:
            pass
        return 'unknown'
    
    def _load_common_issues(self) -> Dict:
        """Load common issues and solutions"""
        return {
            'raspberry_pi_5': {
                'gpio_library_incompatibility': {
                    'symptoms': ['RPi.GPIO ImportError', 'GPIO library not working'],
                    'cause': 'RPi.GPIO not compatible with Pi 5 RP1 chip',
                    'solution': [
                        'Uninstall RPi.GPIO: pip uninstall RPi.GPIO',
                        'Install gpiod: pip install gpiod',
                        'Or use GPIO Zero: pip install gpiozero',
                        'Update code to use gpiod or GPIO Zero'
                    ]
                },
                'insufficient_power': {
                    'symptoms': ['Random crashes', 'USB devices disconnect', 'Lightning bolt icon'],
                    'cause': 'Insufficient power supply for Pi 5',
                    'solution': [
                        'Use official 27W USB-C power supply',
                        'Check USB-C cable quality',
                        'Reduce connected USB devices',
                        'Check vcgencmd get_throttled output'
                    ]
                },
                'i2c_communication_issues': {
                    'symptoms': ['I2C devices not detected', 'Sensor read timeouts'],
                    'cause': 'RP1 I2C controller differences or configuration issues',
                    'solution': [
                        'Enable I2C: sudo raspi-config nonint do_i2c 0',
                        'Set I2C baudrate: add dtparam=i2c_arm_baudrate=400000 to /boot/config.txt',
                        'Check connections and pull-up resistors',
                        'Verify device addresses with i2cdetect -y 1'
                    ]
                }
            },
            'jetson_nano': {
                'cuda_not_available': {
                    'symptoms': ['torch.cuda.is_available() returns False', 'GPU not detected'],
                    'cause': 'CUDA drivers not properly installed or configured',
                    'solution': [
                        'Verify JetPack installation',
                        'Check nvidia-smi output',
                        'Reinstall CUDA toolkit',
                        'Verify PyTorch Jetson wheel installation'
                    ]
                },
                'thermal_throttling': {
                    'symptoms': ['Performance degradation', 'High temperatures', 'Thermal warnings'],
                    'cause': 'Insufficient cooling for Jetson Nano',
                    'solution': [
                        'Install larger heatsink with fan',
                        'Improve case ventilation',
                        'Set lower power mode: sudo nvpmodel -m 1',
                        'Monitor with tegrastats'
                    ]
                },
                'memory_issues': {
                    'symptoms': ['Out of memory errors', 'Model loading failures'],
                    'cause': 'Limited 4GB RAM on Jetson Nano',
                    'solution': [
                        'Enable swap: sudo systemctl enable nvzramconfig',
                        'Optimize models for memory usage',
                        'Use model quantization',
                        'Clear GPU cache: torch.cuda.empty_cache()'
                    ]
                }
            },
            'common': {
                'sensor_not_detected': {
                    'symptoms': ['Sensor not appearing in i2cdetect', 'No sensor readings'],
                    'cause': 'Wiring issues, power problems, or sensor failure',
                    'solution': [
                        'Check wiring connections',
                        'Verify power supply (3.3V or 5V as required)',
                        'Test with multimeter',
                        'Try different I2C address',
                        'Check sensor datasheet'
                    ]
                },
                'network_connectivity': {
                    'symptoms': ['No internet connection', 'WiFi not connecting'],
                    'cause': 'Network configuration or hardware issues',
                    'solution': [
                        'Check WiFi credentials',
                        'Restart network manager: sudo systemctl restart NetworkManager',
                        'Check antenna connections',
                        'Verify network configuration'
                    ]
                }
            }
        }
    
    def diagnose_issue(self, symptoms: List[str]) -> Dict:
        """Diagnose issue based on symptoms"""
        potential_issues = []
        
        # Check platform-specific issues
        platform_issues = self.common_issues.get(self.platform, {})
        for issue_name, issue_data in platform_issues.items():
            symptom_match = any(symptom.lower() in ' '.join(issue_data['symptoms']).lower() 
                              for symptom in symptoms)
            if symptom_match:
                potential_issues.append({
                    'issue': issue_name,
                    'data': issue_data,
                    'platform_specific': True
                })
        
        # Check common issues
        common_issues = self.common_issues.get('common', {})
        for issue_name, issue_data in common_issues.items():
            symptom_match = any(symptom.lower() in ' '.join(issue_data['symptoms']).lower() 
                              for symptom in symptoms)
            if symptom_match:
                potential_issues.append({
                    'issue': issue_name,
                    'data': issue_data,
                    'platform_specific': False
                })
        
        return {
            'platform': self.platform,
            'symptoms_provided': symptoms,
            'potential_issues': potential_issues,
            'diagnosis_confidence': len(potential_issues) > 0
        }
    
    def run_system_diagnostics(self) -> Dict:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            'platform': self.platform,
            'timestamp': time.time(),
            'hardware_status': {},
            'software_status': {},
            'performance_status': {},
            'recommendations': []
        }
        
        # Hardware diagnostics
        diagnostics['hardware_status'] = self._diagnose_hardware()
        
        # Software diagnostics  
        diagnostics['software_status'] = self._diagnose_software()
        
        # Performance diagnostics
        diagnostics['performance_status'] = self._diagnose_performance()
        
        # Generate recommendations
        diagnostics['recommendations'] = self._generate_recommendations(diagnostics)
        
        return diagnostics
    
    def _diagnose_hardware(self) -> Dict:
        """Diagnose hardware status"""
        hardware_status = {}
        
        # GPIO availability
        try:
            if self.platform == 'raspberry_pi_5':
                import gpiod
                chip = gpiod.Chip('gpiochip4')
                hardware_status['gpio'] = {'status': 'OK', 'controller': 'RP1'}
                chip.close()
            elif self.platform == 'jetson_nano':
                import Jetson.GPIO as GPIO
                GPIO.setmode(GPIO.BOARD)
                hardware_status['gpio'] = {'status': 'OK', 'controller': 'Tegra X1'}
                GPIO.cleanup()
        except Exception as e:
            hardware_status['gpio'] = {'status': 'ERROR', 'error': str(e)}
        
        # I2C bus status
        try:
            result = subprocess.run(['i2cdetect', '-y', '1'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                detected_devices = []
                for line in result.stdout.split('\n')[1:]:
                    for addr in line.split()[1:]:
                        if addr != '--' and len(addr) == 2:
                            detected_devices.append(f"0x{addr}")
                
                hardware_status['i2c'] = {
                    'status': 'OK',
                    'devices_detected': detected_devices,
                    'device_count': len(detected_devices)
                }
            else:
                hardware_status['i2c'] = {'status': 'ERROR', 'error': 'i2cdetect failed'}
        except Exception as e:
            hardware_status['i2c'] = {'status': 'ERROR', 'error': str(e)}
        
        # Temperature monitoring
        try:
            temp = self._get_cpu_temperature()
            hardware_status['temperature'] = {
                'status': 'OK' if temp < 70 else 'WARNING' if temp < 80 else 'CRITICAL',
                'temperature_c': temp,
                'thermal_status': 'Normal' if temp < 70 else 'Elevated' if temp < 80 else 'Critical'
            }
        except Exception as e:
            hardware_status['temperature'] = {'status': 'ERROR', 'error': str(e)}
        
        # Platform-specific diagnostics
        if self.platform == 'raspberry_pi_5':
            hardware_status.update(self._diagnose_pi5_hardware())
        elif self.platform == 'jetson_nano':
            hardware_status.update(self._diagnose_jetson_hardware())
        
        return hardware_status
    
    def _diagnose_pi5_hardware(self) -> Dict:
        """Raspberry Pi 5 specific hardware diagnostics"""
        pi5_status = {}
        
        try:
            # Check power supply
            result = subprocess.run(['vcgencmd', 'get_throttled'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                throttled_value = result.stdout.strip().split('=')[1]
                pi5_status['power_supply'] = {
                    'status': 'OK' if throttled_value == '0x0' else 'WARNING',
                    'throttled_value': throttled_value,
                    'undervoltage_detected': throttled_value != '0x0'
                }
        except:
            pi5_status['power_supply'] = {'status': 'UNKNOWN'}
        
        try:
            # Check GPU memory
            result = subprocess.run(['vcgencmd', 'get_mem', 'gpu'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_mem = result.stdout.strip().split('=')[1]
                pi5_status['gpu_memory'] = {
                    'status': 'OK',
                    'allocated_mb': gpu_mem
                }
        except:
            pi5_status['gpu_memory'] = {'status': 'UNKNOWN'}
        
        return pi5_status
    
    def _diagnose_jetson_hardware(self) -> Dict:
        """Jetson Nano specific hardware diagnostics"""
        jetson_status = {}
        
        try:
            # Check CUDA/GPU status
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                jetson_status['gpu'] = {
                    'status': 'OK',
                    'name': gpu_data[0],
                    'temperature': int(gpu_data[1]),
                    'utilization': int(gpu_data[2]),
                    'memory_used_mb': int(gpu_data[3]),
                    'memory_total_mb': int(gpu_data[4])
                }
            else:
                jetson_status['gpu'] = {'status': 'ERROR', 'error': 'nvidia-smi failed'}
        except Exception as e:
            jetson_status['gpu'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            # Check power mode
            with open('/etc/nvpmodel.conf', 'r') as f:
                config_content = f.read()
                if 'MODE_10W' in config_content:
                    power_mode = '10W_MAXN'
                else:
                    power_mode = '5W'
            
            jetson_status['power_mode'] = {
                'status': 'OK',
                'current_mode': power_mode
            }
        except Exception as e:
            jetson_status['power_mode'] = {'status': 'ERROR', 'error': str(e)}
        
        return jetson_status
    
    def _diagnose_software(self) -> Dict:
        """Diagnose software status"""
        software_status = {}
        
        # Python and package versions
        try:
            import sys
            software_status['python'] = {
                'status': 'OK',
                'version': sys.version,
                'platform': sys.platform
            }
        except Exception as e:
            software_status['python'] = {'status': 'ERROR', 'error': str(e)}
        
        # Platform-specific ML libraries
        if self.platform == 'raspberry_pi_5':
            try:
                import tflite_runtime.interpreter as tflite
                software_status['tflite'] = {'status': 'OK', 'available': True}
            except ImportError:
                software_status['tflite'] = {'status': 'ERROR', 'error': 'TensorFlow Lite not available'}
        
        elif self.platform == 'jetson_nano':
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                software_status['pytorch'] = {
                    'status': 'OK' if cuda_available else 'WARNING',
                    'version': torch.__version__,
                    'cuda_available': cuda_available
                }
                
                if cuda_available:
                    software_status['pytorch']['gpu_name'] = torch.cuda.get_device_name(0)
            except ImportError:
                software_status['pytorch'] = {'status': 'ERROR', 'error': 'PyTorch not available'}
        
        # Furcate Nano package
        try:
            import furcate_nano
            software_status['furcate_nano'] = {
                'status': 'OK',
                'version': getattr(furcate_nano, '__version__', 'unknown')
            }
        except ImportError:
            software_status['furcate_nano'] = {'status': 'ERROR', 'error': 'Furcate Nano not installed'}
        
        return software_status
    
    def _diagnose_performance(self) -> Dict:
        """Diagnose system performance"""
        performance_status = {}
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            performance_status['cpu'] = {
                'status': 'OK' if cpu_percent < 80 else 'WARNING',
                'usage_percent': cpu_percent
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            performance_status['memory'] = {
                'status': 'OK' if memory.percent < 80 else 'WARNING',
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            performance_status['disk'] = {
                'status': 'OK' if disk.percent < 80 else 'WARNING',
                'usage_percent': disk.percent,
                'free_gb': disk.free / (1024**3)
            }
            
        except Exception as e:
            performance_status['system_monitoring'] = {'status': 'ERROR', 'error': str(e)}
        
        return performance_status
    
    def _generate_recommendations(self, diagnostics: Dict) -> List[str]:
        """Generate recommendations based on diagnostics"""
        recommendations = []
        
        # Temperature recommendations
        temp_status = diagnostics['hardware_status'].get('temperature', {})
        if temp_status.get('status') == 'WARNING':
            recommendations.append("Install better cooling - CPU temperature is elevated")
        elif temp_status.get('status') == 'CRITICAL':
            recommendations.append("URGENT: CPU temperature critical - improve cooling immediately")
        
        # Power supply recommendations
        if self.platform == 'raspberry_pi_5':
            power_status = diagnostics['hardware_status'].get('power_supply', {})
            if power_status.get('undervoltage_detected'):
                recommendations.append("Use official 27W USB-C power supply - undervoltage detected")
        
        # Memory recommendations
        mem_status = diagnostics['performance_status'].get('memory', {})
        if mem_status.get('status') == 'WARNING':
            if self.platform == 'jetson_nano':
                recommendations.append("Enable swap file to handle memory pressure")
            recommendations.append("Consider memory optimization or model quantization")
        
        # Software recommendations
        if self.platform == 'raspberry_pi_5':
            tflite_status = diagnostics['software_status'].get('tflite', {})
            if tflite_status.get('status') == 'ERROR':
                recommendations.append("Install TensorFlow Lite: pip install tflite-runtime")
        
        elif self.platform == 'jetson_nano':
            pytorch_status = diagnostics['software_status'].get('pytorch', {})
            if pytorch_status.get('status') == 'ERROR':
                recommendations.append("Install PyTorch for Jetson from NVIDIA wheels")
            elif not pytorch_status.get('cuda_available', False):
                recommendations.append("CUDA not available - check JetPack installation")
        
        return recommendations
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        try:
            if self.platform == 'raspberry_pi_5':
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    return int(f.read().strip()) / 1000.0
            elif self.platform == 'jetson_nano':
                with open('/sys/class/thermal/thermal_zone1/temp', 'r') as f:
                    return int(f.read().strip()) / 1000.0
        except:
            pass
        return 0.0

# Usage examples
def quick_troubleshoot():
    """Quick troubleshooting session"""
    guide = TroubleshootingGuide()
    
    print(f"=== Furcate Nano Troubleshooting ===")
    print(f"Platform: {guide.platform}")
    
    # Run diagnostics
    diagnostics = guide.run_system_diagnostics()
    
    print(f"\n=== System Diagnostics ===")
    print(f"Hardware Status: {len([k for k, v in diagnostics['hardware_status'].items() if v.get('status') == 'OK'])} OK")
    print(f"Software Status: {len([k for k, v in diagnostics['software_status'].items() if v.get('status') == 'OK'])} OK")
    
    if diagnostics['recommendations']:
        print(f"\n=== Recommendations ===")
        for i, rec in enumerate(diagnostics['recommendations'], 1):
            print(f"{i}. {rec}")
    
    return diagnostics

if __name__ == "__main__":
    # Run quick troubleshooting
    quick_troubleshoot()