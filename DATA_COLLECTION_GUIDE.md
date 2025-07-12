# ASL Sensor Data Collection Guide

## Overview
This guide covers the improved sensor data collection system for ASL recognition using flex sensors and MPU6050 IMU connected to an ESP32.

## Files Overview

### Core Files
- `esp32_asl/esp32_asl.ino` - Improved ESP32 firmware with calibration and stable readings
- `asl_data_collector.py` - Advanced data collection with real-time visualization
- `asl_train.py` - Simple data collection script for basic use
- `data_collection_config.py` - Configuration settings

## Hardware Setup

### Required Components
- ESP32 development board
- 5x Flex sensors (2.2" or 4.5")
- MPU6050 IMU sensor
- Connecting wires
- Glove for mounting sensors

### Wiring Diagram
```
ESP32 Pin 32 → Flex Sensor 1 (Thumb)
ESP32 Pin 33 → Flex Sensor 2 (Index)
ESP32 Pin 34 → Flex Sensor 3 (Middle)
ESP32 Pin 35 → Flex Sensor 4 (Ring)
ESP32 Pin 36 → Flex Sensor 5 (Pinky)
ESP32 Pin 21 → MPU6050 SDA
ESP32 Pin 22 → MPU6050 SCL
ESP32 3.3V → MPU6050 VCC
ESP32 GND → MPU6050 GND
```

## Software Setup

### 1. Install Required Python Packages
```bash
pip install pyserial numpy matplotlib pandas
```

### 2. Upload ESP32 Code
1. Open Arduino IDE
2. Install ESP32 board support
3. Install required libraries:
   - Adafruit MPU6050
   - Adafruit Unified Sensor
4. Upload `esp32_asl.ino` to your ESP32

### 3. Configure Settings
Edit `data_collection_config.py` to match your setup:
- Change `SERIAL_PORT` to your ESP32 port
- Adjust flex sensor thresholds if needed
- Modify supported gestures list

## Data Collection Process

### Method 1: Advanced Data Collection (Recommended)

#### Step 1: Start the Advanced Collector
```bash
python asl_data_collector.py
```

#### Step 2: Calibrate Sensors
1. Choose option 1: "Calibrate sensors"
2. Keep your hand flat and still for 3 seconds
3. Wait for calibration to complete

#### Step 3: Check Sensor Status
1. Choose option 2: "Show sensor status"
2. Verify all sensors are working properly
3. Check flex sensor values are within expected range (0-4095)

#### Step 4: Collect Data
1. Choose option 3: "Start data collection"
2. Real-time visualization will appear
3. For each gesture:
   - Enter 'y' to collect data
   - Specify number of samples (default 50)
   - Make the gesture and hold steady
   - Wait for collection to complete
   - Enter 'n' to skip or 'q' to quit

#### Step 5: Save Data
1. Choose option 5: "Save data"
2. Enter filename or press Enter for auto-generated name
3. Data is saved as CSV with timestamp

### Method 2: Simple Data Collection

#### Step 1: Start Simple Collector
```bash
python asl_train.py
```

#### Step 2: Collect Data
1. Enter gesture label (e.g., 'A', 'B', 'HELLO')
2. Make the gesture with the glove
3. Press Enter to record
4. Repeat for multiple samples
5. Enter 'exit' to quit

## Best Practices

### Sensor Calibration
- Always calibrate before data collection
- Keep hand flat and still during calibration
- Recalibrate if sensors seem inaccurate

### Data Collection
- Collect 50-100 samples per gesture for good ML training
- Maintain consistent hand position and orientation
- Take breaks between gestures to avoid fatigue
- Collect data in different lighting conditions if needed

### Data Quality
- Check real-time plots for sensor stability
- Avoid collecting data when readings are "UNSTABLE"
- Ensure flex sensor values are within expected ranges
- Validate data before training ML models

### File Management
- Use descriptive filenames with timestamps
- Keep backups of important datasets
- Organize data by collection session

## Troubleshooting

### Connection Issues
- Check COM port in Device Manager (Windows) or `/dev/tty*` (Linux/Mac)
- Verify USB cable is working
- Try different USB ports
- Restart Arduino IDE if needed

### Sensor Issues
- Check wiring connections
- Verify sensor power supply (3.3V)
- Test individual sensors with multimeter
- Adjust thresholds in configuration file

### Data Quality Issues
- Recalibrate sensors
- Check for loose connections
- Ensure stable hand position
- Reduce sample rate if needed

### Performance Issues
- Close unnecessary applications
- Reduce plot update frequency
- Use simpler visualization if needed
- Check available system memory

## Data Format

### CSV Structure
```csv
label,flex1,flex2,flex3,flex4,flex5,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,timestamp
A,1850,2200,2100,2000,1950,0.5,-0.2,9.8,0.1,0.0,0.0,1640995200.123
```

### Data Fields
- `label`: Gesture identifier (A-Z, 0-9, words)
- `flex1-5`: Flex sensor values (0-4095)
- `accel_x,y,z`: Accelerometer readings (m/s²)
- `gyro_x,y,z`: Gyroscope readings (deg/s)
- `timestamp`: Unix timestamp

## Next Steps

After collecting data:
1. Use `data_preprocessing.py` to prepare data for ML
2. Train models with `asl_ml_model.py` or `asl_deep_learning.py`
3. Test real-time prediction with `pc_asl_tts.py`

## Advanced Features

### Custom Gestures
Add new gestures to `SUPPORTED_GESTURES` in config file:
```python
SUPPORTED_GESTURES = ['A', 'B', 'C', 'CUSTOM_GESTURE']
```

### Sensor Configuration
Modify pin assignments in ESP32 code:
```cpp
#define FLEX1_PIN 32  // Change to your preferred pin
```

### Data Validation
Enable/disable validation in config:
```python
ENABLE_DATA_VALIDATION = True
ENABLE_STABILITY_CHECK = True
```

## Support

For issues or questions:
1. Check this guide first
2. Review error messages carefully
3. Verify hardware connections
4. Test with simple examples
5. Check GitHub issues for similar problems 