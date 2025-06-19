# Setting Up a Classroom Network

Learn how to deploy multiple Furcate Nano devices in an educational environment for collaborative environmental monitoring.

## Overview

This tutorial guides you through setting up a network of Furcate Nano devices across a classroom or campus environment. You'll learn how to:

- Configure multiple devices for educational collaboration
- Set up secure local P2P networks
- Create shared monitoring dashboards
- Implement educational data collection workflows
- Manage device fleets for educational institutions

## Prerequisites

### Hardware Requirements
- 3-8 Furcate Nano devices (minimum 3 for mesh networking)
- Educational WiFi network with WPA3 security
- Central display device (smart board, tablet, or computer)
- Optional: LoRa modules for extended range connectivity

### Network Requirements
- Dedicated IoT VLAN for educational devices
- MQTT broker access (can be cloud or local)
- Internet connectivity for cloud integration
- Network segmentation for security

### Educational Permissions
- IT administrator approval for device deployment
- Student data privacy compliance (FERPA, COPPA)
- Institutional research board approval if collecting student data

## Step 1: Network Planning and Security

### Network Architecture Design

```
┌─────────────────────────────────────────────────┐
│                School Network                    │
├─────────────────┬───────────────────────────────┤
│   Main VLAN     │        IoT VLAN              │
│   (Staff/Admin) │     (Furcate Devices)        │
├─────────────────┼───────────────────────────────┤
│                 │  Device 1 ← → Device 2       │
│   Gateway       │      ↕         ↕             │
│   Firewall      │  Device 3 ← → Device 4       │
│   DHCP Server   │      ↕         ↕             │
│                 │  MQTT Broker  Dashboard       │
└─────────────────┴───────────────────────────────┘
```

### Network Segmentation Configuration

**Create Dedicated IoT VLAN:**
```bash
# Network administrator configuration (example for Cisco)
vlan 100
 name FURCATE_EDUCATIONAL_IOT
 state active

interface vlan100
 ip address 192.168.100.1 255.255.255.0
 no shutdown

# DHCP pool for educational IoT devices
ip dhcp pool EDUCATIONAL_IOT
 network 192.168.100.0 255.255.255.0
 default-router 192.168.100.1
 dns-server 8.8.8.8 1.1.1.1
 lease 0 8 0
```

**Firewall Rules for Educational Safety:**
```bash
# Allow internal communication between Furcate devices
access-list 100 permit tcp 192.168.100.0 0.0.0.255 192.168.100.0 0.0.0.255 eq 8883
access-list 100 permit tcp 192.168.100.0 0.0.0.255 192.168.100.0 0.0.0.255 eq 8000
access-list 100 permit udp 192.168.100.0 0.0.0.255 239.255.255.250 0.0.0.0 eq 5683

# Block access to main school network
access-list 100 deny ip 192.168.100.0 0.0.0.255 192.168.1.0 0.0.0.255

# Allow internet access for educational platforms
access-list 100 permit tcp 192.168.100.0 0.0.0.255 any eq 443
access-list 100 permit tcp 192.168.100.0 0.0.0.255 any eq 80
```

## Step 2: Device Configuration for Educational Use

### Educational Configuration Template

Create `classroom-config.yaml`:

```yaml
device:
  id: "classroom-{DEVICE_NUMBER}"
  name: "Environmental Station {DEVICE_NUMBER}"
  location:
    latitude: 40.7128  # School coordinates
    longitude: -74.0060
    altitude: 10.0
    building: "Science Building"
    room: "Room 205"
  purpose: "education"
  institution: "Lincoln High School"
  course_code: "ENV-SCI-101"
  privacy_mode: "educational"  # Enhanced privacy for students

hardware:
  platform: "auto_detect"
  simulation: false
  sensors:
    temperature_humidity:
      type: "dht22"
      pin: 4
      enabled: true
      educational_context: "Weather monitoring"
    air_quality:
      type: "mq135"
      adc_channel: 0
      enabled: true
      educational_context: "Indoor air quality"
    soil_moisture:
      type: "moisture"
      adc_channel: 1
      enabled: false  # Optional for outdoor units

ml:
  simulation: false
  educational_features:
    model_explanation: true
    confidence_display: true
    learning_mode: true

mesh:
  simulation: false
  max_connections: 12
  educational_zone: "classroom_205"
  discovery_interval: 30
  collaboration_enabled: true

monitoring:
  interval_seconds: 300  # 5 minutes for educational use
  educational_alerts: true
  alert_thresholds:
    temperature_humidity:
      temperature: [15, 30]  # Classroom comfort range
      humidity: [30, 70]
    air_quality:
      aqi: [0, 150]  # Educational alert at moderate levels

protocol:
  version: "1.0"
  compression_enabled: true
  educational_mode: true

# Educational-specific configurations
educational:
  data_retention_days: 90  # Semester data retention
  student_privacy: true
  anonymize_data: true
  sharing_permissions:
    internal_research: true
    external_research: false
    student_access: true
    parent_access: true
  lesson_integration:
    subjects: ["Environmental Science", "Physics", "Chemistry"]
    grade_levels: [9, 10, 11, 12]
    curriculum_standards: ["NGSS", "AP Environmental Science"]

# Cloud integrations for education
integrations:
  classroom_dashboard:
    enabled: true
    url: "https://dashboard.lincolnhigh.edu/environmental"
  google_classroom:
    enabled: true
    course_id: "12345"
  canvas_lms:
    enabled: false
  educational_apis:
    weather_underground: true
    epa_airnow: true
```

### Device Setup Script

Create `setup_classroom_device.sh`:

```bash
#!/bin/bash

# Educational device setup script
echo "Setting up Furcate Nano for educational use..."

# Check if running as educational user
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root for educational safety"
   exit 1
fi

# Set device number from argument
DEVICE_NUM=${1:-1}
echo "Configuring device $DEVICE_NUM"

# Create educational directory structure
mkdir -p ~/furcate-classroom/device-$DEVICE_NUM
cd ~/furcate-classroom/device-$DEVICE_NUM

# Download educational configuration
curl -L https://raw.githubusercontent.com/praecise/furcate-nano/main/configs/classroom-config.yaml \
  -o classroom-config.yaml

# Replace device number placeholder
sed -i "s/{DEVICE_NUMBER}/$DEVICE_NUM/g" classroom-config.yaml

# Install educational dependencies
pip install -r requirements-educational.txt

# Set up educational logging
mkdir -p logs data
echo "Educational device $DEVICE_NUM setup complete"

# Create systemd service for educational use
sudo tee /etc/systemd/system/furcate-classroom-$DEVICE_NUM.service > /dev/null <<EOF
[Unit]
Description=Furcate Nano Educational Device $DEVICE_NUM
After=network.target

[Service]
Type=simple
User=student
WorkingDirectory=/home/student/furcate-classroom/device-$DEVICE_NUM
ExecStart=/usr/local/bin/furcate-nano start --config classroom-config.yaml
Restart=always
RestartSec=10
Environment=PYTHONPATH=/home/student/furcate-classroom
StandardOutput=append:/home/student/furcate-classroom/device-$DEVICE_NUM/logs/service.log
StandardError=append:/home/student/furcate-classroom/device-$DEVICE_NUM/logs/error.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable furcate-classroom-$DEVICE_NUM
```

## Step 3: Local MQTT Broker Setup

### Install Mosquitto MQTT Broker

```bash
# Install on Ubuntu/Debian
sudo apt update
sudo apt install mosquitto mosquitto-clients

# Create educational MQTT configuration
sudo tee /etc/mosquitto/conf.d/educational.conf > /dev/null <<EOF
# Educational MQTT Configuration
listener 1883 0.0.0.0
listener 8883 0.0.0.0
certfile /etc/mosquitto/certs/server.crt
keyfile /etc/mosquitto/certs/server.key
cafile /etc/mosquitto/certs/ca.crt

# Authentication for educational use
allow_anonymous false
password_file /etc/mosquitto/passwd

# Access control for classroom
acl_file /etc/mosquitto/acl

# Logging for educational monitoring
log_dest file /var/log/mosquitto/mosquitto.log
log_type all
log_timestamp true
EOF

# Create educational users
sudo mosquitto_passwd -c /etc/mosquitto/passwd classroom_admin
sudo mosquitto_passwd /etc/mosquitto/passwd student_readonly
sudo mosquitto_passwd /etc/mosquitto/passwd teacher_access

# Create access control list
sudo tee /etc/mosquitto/acl > /dev/null <<EOF
# Admin access
user classroom_admin
topic readwrite #

# Teacher access
user teacher_access
topic readwrite classroom/+/sensors/+
topic readwrite classroom/+/alerts/+
topic read classroom/+/status

# Student read-only access
user student_readonly
topic read classroom/+/sensors/+
topic read classroom/+/dashboard/+
EOF

sudo systemctl restart mosquitto
sudo systemctl enable mosquitto
```

### MQTT Security Configuration

```bash
# Generate TLS certificates for educational security
sudo mkdir -p /etc/mosquitto/certs
cd /etc/mosquitto/certs

# Create CA certificate
sudo openssl req -new -x509 -days 365 -extensions v3_ca \
  -keyout ca.key -out ca.crt -subj "/CN=Educational-CA"

# Create server certificate
sudo openssl genrsa -out server.key 2048
sudo openssl req -new -key server.key -out server.csr \
  -subj "/CN=classroom-mqtt.school.edu"
sudo openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 365

# Set permissions
sudo chown mosquitto:mosquitto /etc/mosquitto/certs/*
sudo chmod 600 /etc/mosquitto/certs/*.key
```

## Step 4: Collaborative Mesh Network

### Device Network Discovery

Each device automatically discovers others using multiple protocols:

```python
# Educational mesh discovery configuration
classroom_mesh_config = {
    "discovery_methods": [
        "mdns_bonjour",      # Local network discovery
        "bluetooth_scan",     # Short-range peer discovery
        "udp_broadcast",     # Classroom-wide broadcast
        "wifi_direct"        # Direct device connections
    ],
    "educational_features": {
        "student_friendly_names": True,
        "collaboration_groups": True,
        "shared_experiments": True,
        "peer_learning": True
    }
}
```

### Mesh Collaboration Features

```yaml
# Educational mesh features in device config
mesh:
  educational_collaboration:
    experiment_sharing: true
    data_comparison: true
    peer_validation: true
    group_projects: true
    competitive_monitoring: false  # Disable competition for collaboration
  
  classroom_groups:
    - group_id: "team_alpha"
      devices: ["classroom-1", "classroom-2", "classroom-3"]
      project: "Indoor Air Quality Study"
    - group_id: "team_beta"
      devices: ["classroom-4", "classroom-5", "classroom-6"]
      project: "Temperature Variation Analysis"
```

## Step 5: Educational Dashboard Setup

### Real-Time Classroom Dashboard

Create `classroom_dashboard.py`:

```python
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import json
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'educational_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class ClassroomDashboard:
    def __init__(self):
        self.devices = {}
        self.mqtt_client = mqtt.Client()
        self.setup_mqtt()
    
    def setup_mqtt(self):
        """Setup MQTT client for educational data"""
        self.mqtt_client.username_pw_set("teacher_access", "teacher_password")
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.loop_start()
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Subscribe to classroom topics"""
        topics = [
            "classroom/+/sensors/+",
            "classroom/+/alerts/+",
            "classroom/+/status",
            "classroom/+/collaboration/+"
        ]
        for topic in topics:
            client.subscribe(topic)
    
    def on_mqtt_message(self, client, userdata, msg):
        """Process educational data messages"""
        try:
            topic_parts = msg.topic.split('/')
            device_id = topic_parts[1]
            data_type = topic_parts[2]
            
            payload = json.loads(msg.payload.decode())
            
            # Update device data
            if device_id not in self.devices:
                self.devices[device_id] = {
                    'last_update': datetime.now(),
                    'sensors': {},
                    'alerts': [],
                    'status': 'active'
                }
            
            self.devices[device_id][data_type] = payload
            self.devices[device_id]['last_update'] = datetime.now()
            
            # Broadcast to web dashboard
            socketio.emit('device_update', {
                'device_id': device_id,
                'data_type': data_type,
                'data': payload,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")

dashboard = ClassroomDashboard()

@app.route('/')
def classroom_dashboard():
    """Main classroom dashboard page"""
    return render_template('classroom_dashboard.html')

@app.route('/api/devices')
def get_devices():
    """Get all classroom devices"""
    return jsonify({
        'devices': dashboard.devices,
        'device_count': len(dashboard.devices),
        'last_update': datetime.now().isoformat()
    })

@app.route('/api/classroom/summary')
def classroom_summary():
    """Get classroom environmental summary"""
    if not dashboard.devices:
        return jsonify({'error': 'No devices available'})
    
    # Calculate classroom averages
    temps = []
    humidity = []
    aqi = []
    
    for device in dashboard.devices.values():
        if 'sensors' in device:
            sensor_data = device['sensors']
            if 'temperature' in sensor_data:
                temps.append(sensor_data['temperature'])
            if 'humidity' in sensor_data:
                humidity.append(sensor_data['humidity'])
            if 'aqi' in sensor_data:
                aqi.append(sensor_data['aqi'])
    
    summary = {
        'classroom_average': {
            'temperature': sum(temps) / len(temps) if temps else 0,
            'humidity': sum(humidity) / len(humidity) if humidity else 0,
            'air_quality': sum(aqi) / len(aqi) if aqi else 0
        },
        'device_count': len(dashboard.devices),
        'active_devices': len([d for d in dashboard.devices.values() 
                              if (datetime.now() - d['last_update']).seconds < 600]),
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(summary)

@socketio.on('connect')
def handle_connect():
    """Handle student/teacher connection"""
    emit('connected', {'status': 'Connected to classroom dashboard'})
    # Send current device data
    emit('initial_data', dashboard.devices)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

### Dashboard HTML Template

Create `templates/classroom_dashboard.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classroom Environmental Dashboard</title>
    <script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .device-card {
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .device-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .status-active { border-left: 4px solid #10B981; }
        .status-warning { border-left: 4px solid #F59E0B; }
        .status-error { border-left: 4px solid #EF4444; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">
                Classroom Environmental Monitoring
            </h1>
            <p class="text-gray-600">
                Real-time environmental data from classroom monitoring stations
            </p>
        </div>

        <!-- Classroom Summary -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Average Temperature</h3>
                <p class="text-3xl font-bold text-blue-600" id="avg-temperature">--°C</p>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Average Humidity</h3>
                <p class="text-3xl font-bold text-green-600" id="avg-humidity">--%</p>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Air Quality</h3>
                <p class="text-3xl font-bold text-purple-600" id="avg-aqi">-- AQI</p>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Active Devices</h3>
                <p class="text-3xl font-bold text-orange-600" id="active-devices">--</p>
            </div>
        </div>

        <!-- Device Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8" id="device-grid">
            <!-- Devices will be populated here -->
        </div>

        <!-- Real-time Chart -->
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-xl font-semibold text-gray-800 mb-4">
                Real-time Environmental Trends
            </h3>
            <canvas id="environmental-chart" width="400" height="200"></canvas>
        </div>
    </div>

    <script>
        // Socket.IO connection
        const socket = io();
        
        // Chart setup
        const ctx = document.getElementById('environmental-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                }, {
                    label: 'Humidity (%)',
                    data: [],
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    }
                }
            }
        });

        // Device data storage
        let devices = {};

        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to classroom dashboard');
        });

        socket.on('initial_data', function(data) {
            devices = data;
            updateDashboard();
        });

        socket.on('device_update', function(data) {
            const deviceId = data.device_id;
            if (!devices[deviceId]) {
                devices[deviceId] = {};
            }
            devices[deviceId][data.data_type] = data.data;
            devices[deviceId]['last_update'] = data.timestamp;
            
            updateDashboard();
            updateChart(data);
        });

        function updateDashboard() {
            updateSummary();
            updateDeviceGrid();
        }

        function updateSummary() {
            fetch('/api/classroom/summary')
                .then(response => response.json())
                .then(data => {
                    if (data.classroom_average) {
                        document.getElementById('avg-temperature').textContent = 
                            data.classroom_average.temperature.toFixed(1) + '°C';
                        document.getElementById('avg-humidity').textContent = 
                            data.classroom_average.humidity.toFixed(1) + '%';
                        document.getElementById('avg-aqi').textContent = 
                            data.classroom_average.air_quality.toFixed(0) + ' AQI';
                    }
                    document.getElementById('active-devices').textContent = 
                        data.active_devices + '/' + data.device_count;
                });
        }

        function updateDeviceGrid() {
            const grid = document.getElementById('device-grid');
            grid.innerHTML = '';

            Object.keys(devices).forEach(deviceId => {
                const device = devices[deviceId];
                const deviceCard = createDeviceCard(deviceId, device);
                grid.appendChild(deviceCard);
            });
        }

        function createDeviceCard(deviceId, device) {
            const card = document.createElement('div');
            card.className = 'device-card bg-white rounded-lg shadow p-6 status-active';
            
            const lastUpdate = new Date(device.last_update);
            const isRecent = (new Date() - lastUpdate) < 600000; // 10 minutes
            
            card.innerHTML = `
                <div class="flex justify-between items-start mb-4">
                    <h3 class="text-lg font-semibold text-gray-800">${deviceId}</h3>
                    <span class="px-2 py-1 text-xs rounded-full ${isRecent ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                        ${isRecent ? 'Active' : 'Offline'}
                    </span>
                </div>
                
                <div class="space-y-3">
                    ${device.sensors ? `
                        <div class="flex justify-between">
                            <span class="text-gray-600">Temperature:</span>
                            <span class="font-medium">${device.sensors.temperature || '--'}°C</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Humidity:</span>
                            <span class="font-medium">${device.sensors.humidity || '--'}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Air Quality:</span>
                            <span class="font-medium">${device.sensors.aqi || '--'} AQI</span>
                        </div>
                    ` : '<p class="text-gray-500">No sensor data</p>'}
                </div>
                
                <div class="mt-4 text-xs text-gray-500">
                    Last update: ${lastUpdate.toLocaleTimeString()}
                </div>
            `;
            
            return card;
        }

        function updateChart(data) {
            if (data.data_type === 'sensors') {
                const now = new Date().toLocaleTimeString();
                
                // Add new data point
                chart.data.labels.push(now);
                
                if (data.data.temperature) {
                    chart.data.datasets[0].data.push(data.data.temperature);
                }
                if (data.data.humidity) {
                    chart.data.datasets[1].data.push(data.data.humidity);
                }
                
                // Keep only last 20 data points
                if (chart.data.labels.length > 20) {
                    chart.data.labels.shift();
                    chart.data.datasets.forEach(dataset => {
                        dataset.data.shift();
                    });
                }
                
                chart.update('none');
            }
        }

        // Initialize dashboard
        updateSummary();
        setInterval(updateSummary, 30000); // Update every 30 seconds
    </script>
</body>
</html>
```

## Step 6: Data Collection and Analysis

### Educational Data Pipeline

```python
# Educational data collection for analysis
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

class ClassroomDataAnalyzer:
    def __init__(self, db_path="classroom_data.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup educational database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                temperature REAL,
                humidity REAL,
                air_quality INTEGER,
                location TEXT,
                experiment_id TEXT,
                student_group TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                start_date DATETIME,
                end_date DATETIME,
                description TEXT,
                participating_devices TEXT,
                student_groups TEXT,
                hypothesis TEXT,
                results TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_sensor_data(self, device_id, sensor_data, experiment_id=None, student_group=None):
        """Log sensor data for educational analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_readings 
            (device_id, timestamp, temperature, humidity, air_quality, experiment_id, student_group)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id,
            datetime.now(),
            sensor_data.get('temperature'),
            sensor_data.get('humidity'),
            sensor_data.get('aqi'),
            experiment_id,
            student_group
        ))
        
        conn.commit()
        conn.close()
    
    def generate_classroom_report(self, days=7):
        """Generate educational data report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get data from last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = pd.read_sql_query('''
            SELECT * FROM sensor_readings 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        ''', conn, params=[start_date, end_date])
        
        conn.close()
        
        if df.empty:
            return {"error": "No data available for the specified period"}
        
        # Calculate educational metrics
        report = {
            "summary": {
                "data_points": len(df),
                "devices": df['device_id'].nunique(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            },
            "environmental_averages": {
                "temperature": {
                    "mean": df['temperature'].mean(),
                    "min": df['temperature'].min(),
                    "max": df['temperature'].max(),
                    "std": df['temperature'].std()
                },
                "humidity": {
                    "mean": df['humidity'].mean(),
                    "min": df['humidity'].min(),
                    "max": df['humidity'].max(),
                    "std": df['humidity'].std()
                },
                "air_quality": {
                    "mean": df['air_quality'].mean(),
                    "min": df['air_quality'].min(),
                    "max": df['air_quality'].max(),
                    "std": df['air_quality'].std()
                }
            },
            "device_performance": {},
            "educational_insights": []
        }
        
        # Per-device analysis
        for device_id in df['device_id'].unique():
            device_data = df[df['device_id'] == device_id]
            report["device_performance"][device_id] = {
                "data_points": len(device_data),
                "uptime_percentage": self._calculate_uptime(device_data, days),
                "avg_temperature": device_data['temperature'].mean(),
                "avg_humidity": device_data['humidity'].mean()
            }
        
        # Educational insights
        report["educational_insights"] = self._generate_educational_insights(df)
        
        return report
    
    def _calculate_uptime(self, device_data, days):
        """Calculate device uptime percentage"""
        expected_readings = days * 24 * 12  # Assuming 5-minute intervals
        actual_readings = len(device_data)
        return min(100, (actual_readings / expected_readings) * 100)
    
    def _generate_educational_insights(self, df):
        """Generate educational insights from data"""
        insights = []
        
        # Temperature insights
        temp_range = df['temperature'].max() - df['temperature'].min()
        if temp_range > 10:
            insights.append({
                "category": "Temperature Variation",
                "insight": f"Significant temperature variation observed ({temp_range:.1f}°C range)",
                "educational_value": "Great for studying thermal dynamics and comfort zones"
            })
        
        # Air quality insights
        avg_aqi = df['air_quality'].mean()
        if avg_aqi > 100:
            insights.append({
                "category": "Air Quality",
                "insight": f"Average AQI of {avg_aqi:.0f} indicates moderate air quality",
                "educational_value": "Opportunity to study indoor air quality factors"
            })
        
        # Correlation insights
        correlation = df['temperature'].corr(df['humidity'])
        if abs(correlation) > 0.7:
            insights.append({
                "category": "Environmental Correlation",
                "insight": f"Strong correlation ({correlation:.2f}) between temperature and humidity",
                "educational_value": "Demonstrates atmospheric science principles"
            })
        
        return insights

# Usage example
analyzer = ClassroomDataAnalyzer()
```

## Step 7: Student Projects and Experiments

### Project Templates

Create structured experiments for students:

```yaml
# Environmental Science Project Templates
projects:
  indoor_air_quality_study:
    title: "Indoor Air Quality Investigation"
    duration: "2 weeks"
    grade_levels: [9, 10, 11]
    learning_objectives:
      - "Understand factors affecting indoor air quality"
      - "Learn data collection and analysis skills"
      - "Identify sources of air pollution"
    hypothesis_template: |
      "We hypothesize that [factor] will [effect] the indoor air quality 
      because [scientific reasoning]"
    data_collection:
      variables: ["air_quality", "temperature", "humidity", "time_of_day"]
      duration_days: 14
      measurement_interval: 300  # 5 minutes
    analysis_questions:
      - "What time of day has the best air quality?"
      - "How does temperature affect air quality?"
      - "What activities in the classroom impact air quality?"
    deliverables:
      - "Data visualization chart"
      - "Written hypothesis and conclusion"
      - "Presentation to class"
  
  microclimate_mapping:
    title: "Classroom Microclimate Mapping"
    duration: "1 week"
    grade_levels: [7, 8, 9]
    learning_objectives:
      - "Understand microclimate concepts"
      - "Practice spatial data analysis"
      - "Learn about environmental gradients"
    setup:
      device_placement: "different_locations"
      variables: ["temperature", "humidity", "light_levels"]
    analysis_activities:
      - "Create temperature maps"
      - "Identify hot and cold spots"
      - "Explain microclimate variations"
```

### Student Data Access Portal

```python
# Student-friendly data access
from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/student-portal')
def student_portal():
    """Student data exploration portal"""
    return render_template('student_portal.html')

@app.route('/api/student/data')
def get_student_data():
    """Get anonymized data for student analysis"""
    days = request.args.get('days', 7, type=int)
    experiment_id = request.args.get('experiment_id', None)
    
    conn = sqlite3.connect('classroom_data.db')
    
    query = '''
        SELECT 
            device_id,
            timestamp,
            temperature,
            humidity,
            air_quality,
            experiment_id
        FROM sensor_readings 
        WHERE timestamp >= datetime('now', '-{} days')
    '''.format(days)
    
    if experiment_id:
        query += f" AND experiment_id = '{experiment_id}'"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Anonymize device IDs for student privacy
    device_mapping = {device: f"Station {i+1}" 
                     for i, device in enumerate(df['device_id'].unique())}
    df['device_id'] = df['device_id'].map(device_mapping)
    
    return jsonify({
        'data': df.to_dict('records'),
        'summary': {
            'total_readings': len(df),
            'stations': len(device_mapping),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
    })

@app.route('/api/student/visualization')
def create_visualization():
    """Create educational visualizations"""
    chart_type = request.args.get('type', 'temperature_trend')
    days = request.args.get('days', 7, type=int)
    
    conn = sqlite3.connect('classroom_data.db')
    df = pd.read_sql_query('''
        SELECT * FROM sensor_readings 
        WHERE timestamp >= datetime('now', '-{} days')
    '''.format(days), conn)
    conn.close()
    
    if chart_type == 'temperature_trend':
        fig = px.line(df, x='timestamp', y='temperature', 
                     color='device_id',
                     title='Temperature Trends Over Time',
                     labels={'temperature': 'Temperature (°C)',
                            'timestamp': 'Time'})
    
    elif chart_type == 'air_quality_histogram':
        fig = px.histogram(df, x='air_quality',
                          title='Air Quality Distribution',
                          labels={'air_quality': 'Air Quality Index',
                                 'count': 'Frequency'})
    
    elif chart_type == 'correlation_matrix':
        corr_data = df[['temperature', 'humidity', 'air_quality']].corr()
        fig = px.imshow(corr_data,
                       title='Environmental Variable Correlations',
                       color_continuous_scale='RdBu_r')
    
    else:
        return jsonify({'error': 'Unknown chart type'})
    
    return fig.to_json()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
```

## Step 8: Network Management and Troubleshooting

### Device Fleet Management

```bash
#!/bin/bash
# Educational device fleet management script

CLASSROOM_DEVICES=(
    "classroom-1"
    "classroom-2" 
    "classroom-3"
    "classroom-4"
    "classroom-5"
    "classroom-6"
)

function check_device_status() {
    echo "Checking educational device status..."
    
    for device in "${CLASSROOM_DEVICES[@]}"; do
        echo -n "Device $device: "
        
        # Check if service is running
        if systemctl is-active --quiet furcate-classroom-$device; then
            echo -n "RUNNING "
        else
            echo -n "STOPPED "
        fi
        
        # Check network connectivity
        if ping -c1 -W1 $device.local >/dev/null 2>&1; then
            echo "NETWORK_OK"
        else
            echo "NETWORK_ERROR"
        fi
    done
}

function restart_classroom_devices() {
    echo "Restarting all classroom devices..."
    
    for device in "${CLASSROOM_DEVICES[@]}"; do
        echo "Restarting $device..."
        sudo systemctl restart furcate-classroom-$device
        sleep 2
    done
    
    echo "All devices restarted"
}

function update_classroom_config() {
    config_file=$1
    
    if [[ ! -f "$config_file" ]]; then
        echo "Configuration file not found: $config_file"
        exit 1
    fi
    
    echo "Updating configuration for all classroom devices..."
    
    for i in "${!CLASSROOM_DEVICES[@]}"; do
        device=${CLASSROOM_DEVICES[$i]}
        device_num=$((i + 1))
        
        # Copy and customize config for each device
        cp "$config_file" "/home/student/furcate-classroom/device-$device_num/classroom-config.yaml"
        
        # Replace device-specific placeholders
        sed -i "s/{DEVICE_NUMBER}/$device_num/g" \
            "/home/student/furcate-classroom/device-$device_num/classroom-config.yaml"
        
        echo "Updated config for $device"
    done
    
    echo "Configuration update complete"
}

function generate_classroom_report() {
    echo "Generating classroom network report..."
    
    report_file="classroom_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "=== Furcate Nano Classroom Network Report ==="
        echo "Generated: $(date)"
        echo "========================================"
        echo
        
        echo "Device Status:"
        check_device_status
        echo
        
        echo "Network Configuration:"
        echo "MQTT Broker: $(systemctl is-active mosquitto)"
        echo "Dashboard: $(systemctl is-active classroom-dashboard)"
        echo
        
        echo "Recent Device Activity:"
        journalctl --since "1 hour ago" -u "furcate-classroom-*" --no-pager | tail -20
        
    } > "$report_file"
    
    echo "Report saved to: $report_file"
}

# Main script menu
case "$1" in
    "status")
        check_device_status
        ;;
    "restart")
        restart_classroom_devices
        ;;
    "update-config")
        update_classroom_config "$2"
        ;;
    "report")
        generate_classroom_report
        ;;
    *)
        echo "Usage: $0 {status|restart|update-config <file>|report}"
        echo
        echo "Commands:"
        echo "  status        - Check all device status"
        echo "  restart       - Restart all classroom devices"
        echo "  update-config - Update configuration for all devices"
        echo "  report        - Generate classroom network report"
        exit 1
        ;;
esac
```

### Network Troubleshooting Guide

```markdown
## Common Classroom Network Issues

### Device Not Connecting to WiFi
1. Check WiFi credentials in device configuration
2. Verify device is on correct VLAN
3. Check firewall rules for IoT VLAN
4. Restart network services: `sudo systemctl restart NetworkManager`

### MQTT Connection Issues
1. Verify MQTT broker is running: `sudo systemctl status mosquitto`
2. Check MQTT logs: `sudo tail -f /var/log/mosquitto/mosquitto.log`
3. Test connection: `mosquitto_pub -h localhost -t test -m "hello"`
4. Verify authentication credentials

### Dashboard Not Updating
1. Check dashboard service: `sudo systemctl status classroom-dashboard`
2. Verify WebSocket connections in browser developer tools
3. Check MQTT topic subscriptions
4. Restart dashboard: `sudo systemctl restart classroom-dashboard`

### Device Mesh Network Issues
1. Check device discovery logs
2. Verify Bluetooth/WiFi Direct permissions
3. Ensure devices are in same educational zone
4. Check mesh network configuration
```

## Step 9: Educational Integration

### Curriculum Standards Alignment

```yaml
# NGSS Standards Alignment
curriculum_alignment:
  middle_school:
    MS-ESS3-3:
      title: "Human impact on environment"
      furcate_activities:
        - "Indoor air quality monitoring"
        - "Energy efficiency measurement"
        - "Environmental factor correlation"
    MS-ETS1-1:
      title: "Engineering design process"
      furcate_activities:
        - "Sensor network design"
        - "Data collection optimization"
        - "Problem-solving with environmental data"
  
  high_school:
    HS-ESS3-3:
      title: "Human impact on environment"
      furcate_activities:
        - "Long-term environmental monitoring"
        - "Urban heat island studies"
        - "Climate data analysis"
    HS-ETS1-4:
      title: "Computer simulation to model impacts"
      furcate_activities:
        - "Environmental data modeling"
        - "Predictive analytics"
        - "IoT system design"

# AP Environmental Science Integration
ap_environmental_science:
  units:
    unit_7_atmospheric_pollution:
      activities:
        - "Indoor vs outdoor air quality comparison"
        - "Particulate matter monitoring"
        - "Pollution source identification"
    unit_9_global_change:
      activities:
        - "Microclimate documentation"
        - "Temperature trend analysis"
        - "Climate variability studies"
```

### Assessment Rubrics

```yaml
# Student Assessment Rubrics for Furcate Projects
assessment_rubrics:
  data_collection_skills:
    exemplary:
      description: "Systematically collects accurate data over extended periods"
      criteria:
        - "No missing data points"
        - "Proper calibration documented"
        - "Multiple variables tracked"
      points: 4
    
    proficient:
      description: "Collects mostly accurate data with minor gaps"
      criteria:
        - "Less than 10% missing data"
        - "Basic calibration performed"
        - "Primary variables tracked"
      points: 3
    
    developing:
      description: "Collects some data with significant gaps"
      criteria:
        - "10-25% missing data"
        - "Inconsistent methodology"
        - "Limited variable tracking"
      points: 2
    
    beginning:
      description: "Limited or inaccurate data collection"
      criteria:
        - "More than 25% missing data"
        - "No calibration documented"
        - "Single variable focus"
      points: 1
  
  scientific_analysis:
    exemplary:
      description: "Sophisticated analysis with multiple statistical methods"
      criteria:
        - "Advanced statistical analysis"
        - "Correlation and trend identification"
        - "Predictive modeling attempted"
      points: 4
    
    proficient:
      description: "Good analysis using appropriate statistical methods"
      criteria:
        - "Basic statistical analysis"
        - "Clear trend identification"
        - "Appropriate graph types"
      points: 3
```

## Step 10: Maintenance and Updates

### Automated Maintenance

```python
#!/usr/bin/env python3
"""
Educational Network Maintenance Script
Automated maintenance for classroom Furcate Nano deployments
"""

import subprocess
import sqlite3
import smtplib
from email.mime.text import MimeText
from datetime import datetime, timedelta
import logging

class ClassroomMaintenance:
    def __init__(self):
        self.devices = [f"classroom-{i}" for i in range(1, 7)]
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            filename='/var/log/classroom_maintenance.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def check_device_health(self):
        """Check health of all classroom devices"""
        health_report = {
            'healthy': [],
            'unhealthy': [],
            'offline': []
        }
        
        for device in self.devices:
            try:
                # Check service status
                result = subprocess.run(
                    ['systemctl', 'is-active', f'furcate-classroom-{device}'],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    # Check last data timestamp
                    last_data = self.get_last_data_time(device)
                    if last_data and (datetime.now() - last_data).seconds < 600:
                        health_report['healthy'].append(device)
                    else:
                        health_report['unhealthy'].append(device)
                else:
                    health_report['offline'].append(device)
                    
            except Exception as e:
                logging.error(f"Error checking {device}: {e}")
                health_report['offline'].append(device)
        
        return health_report
    
    def get_last_data_time(self, device):
        """Get last data timestamp for device"""
        try:
            conn = sqlite3.connect('/home/student/classroom_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT MAX(timestamp) FROM sensor_readings 
                WHERE device_id = ?
            ''', (device,))
            
            result = cursor.fetchone()[0]
            conn.close()
            
            if result:
                return datetime.fromisoformat(result)
            return None
            
        except Exception as e:
            logging.error(f"Database error for {device}: {e}")
            return None
    
    def restart_unhealthy_devices(self, unhealthy_devices):
        """Restart devices that are unhealthy"""
        restarted = []
        
        for device in unhealthy_devices:
            try:
                subprocess.run(
                    ['sudo', 'systemctl', 'restart', f'furcate-classroom-{device}'],
                    check=True
                )
                restarted.append(device)
                logging.info(f"Restarted {device}")
                
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to restart {device}: {e}")
        
        return restarted
    
    def cleanup_old_data(self, days=90):
        """Clean up old educational data"""
        try:
            conn = sqlite3.connect('/home/student/classroom_data.db')
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                DELETE FROM sensor_readings 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            logging.info(f"Cleaned up {deleted_rows} old data records")
            return deleted_rows
            
        except Exception as e:
            logging.error(f"Data cleanup error: {e}")
            return 0
    
    def send_maintenance_report(self, health_report, restarted_devices):
        """Send maintenance report to administrators"""
        report_text = f"""
        Classroom Network Maintenance Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Device Health Status:
        - Healthy devices: {len(health_report['healthy'])} ({', '.join(health_report['healthy'])})
        - Unhealthy devices: {len(health_report['unhealthy'])} ({', '.join(health_report['unhealthy'])})
        - Offline devices: {len(health_report['offline'])} ({', '.join(health_report['offline'])})
        
        Actions Taken:
        - Restarted devices: {', '.join(restarted_devices) if restarted_devices else 'None'}
        
        Recommendations:
        {self.generate_recommendations(health_report)}
        """
        
        # Email configuration (configure as needed)
        try:
            msg = MimeText(report_text)
            msg['Subject'] = 'Classroom Network Maintenance Report'
            msg['From'] = 'furcate-system@school.edu'
            msg['To'] = 'it-admin@school.edu'
            
            # Send email (configure SMTP settings)
            # smtp_server = smtplib.SMTP('smtp.school.edu', 587)
            # smtp_server.send_message(msg)
            # smtp_server.quit()
            
            logging.info("Maintenance report generated")
            
        except Exception as e:
            logging.error(f"Failed to send maintenance report: {e}")
    
    def generate_recommendations(self, health_report):
        """Generate maintenance recommendations"""
        recommendations = []
        
        if len(health_report['offline']) > 2:
            recommendations.append("- Multiple devices offline: Check network connectivity")
        
        if len(health_report['unhealthy']) > 1:
            recommendations.append("- Multiple unhealthy devices: Consider system update")
        
        if not recommendations:
            recommendations.append("- System operating normally")
        
        return '\n'.join(recommendations)
    
    def run_maintenance(self):
        """Run complete maintenance routine"""
        logging.info("Starting classroom maintenance routine")
        
        # Check device health
        health_report = self.check_device_health()
        
        # Restart unhealthy devices
        restarted = self.restart_unhealthy_devices(health_report['unhealthy'])
        
        # Clean up old data
        self.cleanup_old_data()
        
        # Send report
        self.send_maintenance_report(health_report, restarted)
        
        logging.info("Maintenance routine completed")

if __name__ == "__main__":
    maintenance = ClassroomMaintenance()
    maintenance.run_maintenance()
```

### System Update Script

```bash
#!/bin/bash
# Educational system update script

echo "Starting Furcate Nano educational system update..."

# Backup current configurations
echo "Backing up configurations..."
mkdir -p /home/student/furcate-backups/$(date +%Y%m%d_%H%M%S)
cp -r /home/student/furcate-classroom/* /home/student/furcate-backups/$(date +%Y%m%d_%H%M%S)/

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Update Furcate Nano
echo "Updating Furcate Nano..."
pip install --upgrade furcate-nano

# Update educational dependencies
echo "Updating educational dependencies..."
pip install --upgrade -r /home/student/furcate-classroom/requirements-educational.txt

# Restart all classroom services
echo "Restarting classroom services..."
for i in {1..6}; do
    sudo systemctl restart furcate-classroom-$i
done

sudo systemctl restart mosquitto
sudo systemctl restart classroom-dashboard

echo "Educational system update complete!"
```

## Conclusion

You've successfully set up a comprehensive classroom network of Furcate Nano devices! This network provides:

- **Collaborative Environmental Monitoring**: Multiple devices working together
- **Educational Data Collection**: Real-time data for student projects
- **Secure Network Architecture**: Properly segmented and secured
- **Interactive Dashboards**: Real-time visualization for learning
- **Automated Maintenance**: Self-maintaining educational infrastructure

### Next Steps

1. **Expand the Network**: Add more devices to cover additional areas
2. **Custom Projects**: Develop subject-specific monitoring projects
3. **Advanced Analytics**: Implement machine learning for data analysis
4. **Student Presentations**: Have students present their findings
5. **Cross-Curricular Integration**: Connect with math, physics, and computer science

### Educational Benefits

- **STEM Skills Development**: Hands-on experience with IoT technology
- **Data Literacy**: Real-world data collection and analysis
- **Environmental Awareness**: Understanding of environmental factors
- **Collaboration**: Working together on shared monitoring goals
- **Technology Integration**: Modern technology in education

### Support Resources

- **Technical Documentation**: Complete API and configuration references
- **Troubleshooting Guide**: Common issues and solutions
- **Educational Resources**: Lesson plans and project templates
- **Community Support**: Connect with other educational users

Your classroom network is now ready for educational environmental monitoring and STEM learning activities!