# ============================================================================
# furcate_nano/power.py
"""Power management for solar-powered Furcate Nano devices."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

try:
    import RPi.GPIO as GPIO
    POWER_GPIO_AVAILABLE = True
except ImportError:
    POWER_GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class PowerMode(Enum):
    """Power management modes."""
    NORMAL = "normal"          # Full operation
    BALANCED = "balanced"      # Moderate power saving
    LOW_POWER = "low_power"    # Aggressive power saving
    EMERGENCY = "emergency"    # Minimal operation only
    SLEEP = "sleep"           # Deep sleep mode

@dataclass
class PowerStatus:
    """Current power system status."""
    battery_voltage: float
    battery_level: float  # 0.0 - 1.0
    solar_voltage: float
    solar_current: float
    charging: bool
    estimated_runtime_hours: float
    power_mode: PowerMode
    temperature: float

class PowerManager:
    """Manages solar power and battery systems for autonomous operation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize power manager.
        
        Args:
            config: Power management configuration
        """
        self.config = config
        self.simulation_mode = not POWER_GPIO_AVAILABLE or config.get("simulation", True)
        
        # Battery configuration
        self.battery_config = config.get("battery", {
            "capacity_mah": 10000,
            "voltage_min": 3.0,
            "voltage_max": 4.2,
            "voltage_nominal": 3.7
        })
        
        # Solar configuration
        self.solar_config = config.get("solar", {
            "panel_watts": 20,
            "voltage_max": 6.0,
            "mppt_enabled": True
        })
        
        # Power management thresholds
        self.thresholds = config.get("thresholds", {
            "low_battery": 0.2,      # 20%
            "critical_battery": 0.1,  # 10%
            "emergency_battery": 0.05 # 5%
        })
        
        # Current state
        self.current_mode = PowerMode.NORMAL
        self.power_history = []
        self.last_measurement = None
        
        # ADC channels for power monitoring
        self.adc_channels = config.get("adc_channels", {
            "battery_voltage": 0,
            "solar_voltage": 1,
            "solar_current": 2
        })
        
        # Power control pins
        self.control_pins = config.get("control_pins", {
            "cpu_freq_control": 23,
            "wifi_enable": 24,
            "sensor_power": 25
        })
        
        logger.info(f"Power manager initialized (simulation: {self.simulation_mode})")
    
    async def initialize(self) -> bool:
        """Initialize power monitoring and control systems."""
        try:
            if self.simulation_mode:
                logger.info("⚡ Power management in simulation mode")
                return True
            
            # Initialize GPIO for power control
            for pin_name, pin_number in self.control_pins.items():
                GPIO.setup(pin_number, GPIO.OUT)
                GPIO.output(pin_number, GPIO.HIGH)  # Start in normal mode
            
            # Start power monitoring
            asyncio.create_task(self._power_monitoring_loop())
            
            logger.info("✅ Power management initialized")
            return True
            
        except Exception as e:
            logger.error(f"Power management initialization failed: {e}")
            return False
    
    async def _power_monitoring_loop(self):
        """Continuous power monitoring and management."""
        while True:
            try:
                # Read power status
                status = await self.get_status()
                self.last_measurement = status
                
                # Store power history
                self.power_history.append({
                    "timestamp": datetime.now(),
                    "battery_level": status.battery_level,
                    "solar_power": status.solar_voltage * status.solar_current,
                    "mode": status.power_mode.value
                })
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.power_history = [h for h in self.power_history if h["timestamp"] > cutoff_time]
                
                # Determine optimal power mode
                optimal_mode = self._calculate_optimal_power_mode(status)
                
                # Switch modes if needed
                if optimal_mode != self.current_mode:
                    await self.set_mode(optimal_mode)
                
                # Wait before next measurement
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Power monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def get_status(self) -> PowerStatus:
        """Get current power system status."""
        if self.simulation_mode:
            return self._simulate_power_status()
        
        try:
            # Read battery voltage (assuming ADC available)
            battery_voltage = await self._read_adc_voltage(self.adc_channels["battery_voltage"])
            
            # Calculate battery level from voltage
            v_min = self.battery_config["voltage_min"]
            v_max = self.battery_config["voltage_max"]
            battery_level = max(0.0, min(1.0, (battery_voltage - v_min) / (v_max - v_min)))
            
            # Read solar parameters
            solar_voltage = await self._read_adc_voltage(self.adc_channels["solar_voltage"])
            solar_current = await self._read_adc_current(self.adc_channels["solar_current"])
            
            # Determine charging status
            charging = solar_voltage > battery_voltage + 0.5  # Simple charging detection
            
            # Estimate runtime
            estimated_runtime = self._estimate_runtime(battery_level)
            
            return PowerStatus(
                battery_voltage=battery_voltage,
                battery_level=battery_level,
                solar_voltage=solar_voltage,
                solar_current=solar_current,
                charging=charging,
                estimated_runtime_hours=estimated_runtime,
                power_mode=self.current_mode,
                temperature=25.0  # Would read from temperature sensor
            )
            
        except Exception as e:
            logger.error(f"Power status read failed: {e}")
            return self._simulate_power_status()
    
    def _simulate_power_status(self) -> PowerStatus:
        """Simulate power status for development."""
        import random
        
        # Simulate time-of-day solar charging
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Daylight hours
            solar_voltage = 4.0 + random.random() * 2.0
            solar_current = 0.5 + random.random() * 1.5
            charging = True
        else:  # Night
            solar_voltage = 0.1 + random.random() * 0.2
            solar_current = 0.0
            charging = False
        
        # Simulate battery discharge/charge
        base_battery_level = 0.6
        if charging:
            base_battery_level += 0.2
        else:
            base_battery_level -= 0.1
        
        battery_level = max(0.1, min(1.0, base_battery_level + random.gauss(0, 0.1)))
        battery_voltage = 3.0 + battery_level * 1.2
        
        return PowerStatus(
            battery_voltage=battery_voltage,
            battery_level=battery_level,
            solar_voltage=solar_voltage,
            solar_current=solar_current,
            charging=charging,
            estimated_runtime_hours=battery_level * 24,  # Simple estimation
            power_mode=self.current_mode,
            temperature=20.0 + random.random() * 10
        )
    
    async def _read_adc_voltage(self, channel: int) -> float:
        """Read voltage from ADC channel."""
        # This would interface with actual ADC
        # For now, return simulated value
        import random
        return 3.0 + random.random() * 2.0
    
    async def _read_adc_current(self, channel: int) -> float:
        """Read current from ADC channel."""
        # This would interface with current sensor
        import random
        return random.random() * 2.0
    
    def _calculate_optimal_power_mode(self, status: PowerStatus) -> PowerMode:
        """Calculate optimal power mode based on current status."""
        battery_level = status.battery_level
        
        # Critical battery levels
        if battery_level <= self.thresholds["emergency_battery"]:
            return PowerMode.EMERGENCY
        elif battery_level <= self.thresholds["critical_battery"]:
            return PowerMode.LOW_POWER
        elif battery_level <= self.thresholds["low_battery"]:
            return PowerMode.BALANCED
        
        # Consider charging status and time trends
        if status.charging and battery_level > 0.8:
            return PowerMode.NORMAL
        elif not status.charging and battery_level < 0.5:
            return PowerMode.BALANCED
        
        return PowerMode.NORMAL
    
    def _estimate_runtime(self, battery_level: float) -> float:
        """Estimate remaining runtime in hours."""
        capacity_remaining = battery_level * self.battery_config["capacity_mah"]
        
        # Estimate current consumption based on mode
        consumption_ma = {
            PowerMode.NORMAL: 200,
            PowerMode.BALANCED: 120,
            PowerMode.LOW_POWER: 60,
            PowerMode.EMERGENCY: 20,
            PowerMode.SLEEP: 5
        }
        
        current_consumption = consumption_ma.get(self.current_mode, 200)
        
        if current_consumption <= 0:
            return 999.0  # Effectively unlimited
        
        return capacity_remaining / current_consumption
    
    async def set_mode(self, mode: PowerMode) -> bool:
        """Set power management mode.
        
        Args:
            mode: Target power mode
            
        Returns:
            Success status
        """
        try:
            if mode == self.current_mode:
                return True
            
            logger.info(f"⚡ Switching from {self.current_mode.value} to {mode.value}")
            
            if not self.simulation_mode:
                # Apply hardware power settings
                await self._apply_power_mode(mode)
            
            self.current_mode = mode
            return True
            
        except Exception as e:
            logger.error(f"Power mode switch failed: {e}")
            return False
    
    async def _apply_power_mode(self, mode: PowerMode):
        """Apply hardware settings for power mode."""
        if mode == PowerMode.LOW_POWER:
            # Reduce CPU frequency
            await self._set_cpu_frequency("600MHz")
            # Disable WiFi
            GPIO.output(self.control_pins["wifi_enable"], GPIO.LOW)
            # Power down non-essential sensors
            GPIO.output(self.control_pins["sensor_power"], GPIO.LOW)
            
        elif mode == PowerMode.EMERGENCY:
            # Minimum CPU frequency
            await self._set_cpu_frequency("300MHz")
            # Disable all non-essential systems
            GPIO.output(self.control_pins["wifi_enable"], GPIO.LOW)
            GPIO.output(self.control_pins["sensor_power"], GPIO.LOW)
            
        elif mode == PowerMode.NORMAL:
            # Full performance
            await self._set_cpu_frequency("1800MHz")
            # Enable all systems
            GPIO.output(self.control_pins["wifi_enable"], GPIO.HIGH)
            GPIO.output(self.control_pins["sensor_power"], GPIO.HIGH)
    
    async def _set_cpu_frequency(self, frequency: str):
        """Set CPU frequency for power management."""
        try:
            # This would use system calls to change CPU governor
            # subprocess.run(['sudo', 'cpufreq-set', '-f', frequency])
            logger.debug(f"CPU frequency set to {frequency}")
        except Exception as e:
            logger.warning(f"CPU frequency change failed: {e}")
    
    def get_power_history(self, hours: int = 24) -> list:
        """Get power history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.power_history if h["timestamp"] > cutoff_time]
    
    async def shutdown(self):
        """Shutdown power management."""
        if not self.simulation_mode and POWER_GPIO_AVAILABLE:
            # Set all pins to safe states
            for pin in self.control_pins.values():
                GPIO.output(pin, GPIO.LOW)
        
        logger.info("Power manager shutdown complete")