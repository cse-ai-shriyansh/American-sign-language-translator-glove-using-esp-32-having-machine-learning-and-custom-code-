#!/usr/bin/env python3
"""
ASL Recognition System - Step-by-Step Information Feeding Guide
This script demonstrates how to feed information into each stage of the system.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
from datetime import datetime

class ASLInformationFeeder:
    def __init__(self):
        self.system_status = {
            'hardware': False,
            'firmware': False,
            'data_collection': False,
            'preprocessing': False,
            'ml_training': False,
            'prediction': False
        }
        self.config = self.load_config()
        
    def load_config(self):
        """Load system configuration"""
        config = {
            'serial_port': 'COM3',
            'baud_rate': 115200,
            'timeout': 2,
            'samples_per_gesture': 50,
            'supported_gestures': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'flex_thresholds': {'min': 1800, 'max': 2200},
            'accel_range': {'min': -20, 'max': 20},
            'gyro_range': {'min': -500, 'max': 500}
        }
        return config
    
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"🔧 {title}")
        print("="*60)
    
    def print_step(self, step_num, description):
        """Print formatted step"""
        print(f"\n📋 Step {step_num}: {description}")
        print("-" * 40)
    
    def validate_input(self, data, data_type):
        """Validate input data based on type"""
        if data_type == 'flex_sensors':
            return all(self.config['flex_thresholds']['min'] <= val <= self.config['flex_thresholds']['max'] 
                      for val in data[:5])
        elif data_type == 'accel':
            return all(self.config['accel_range']['min'] <= val <= self.config['accel_range']['max'] 
                      for val in data[5:8])
        elif data_type == 'gyro':
            return all(self.config['gyro_range']['min'] <= val <= self.config['gyro_range']['max'] 
                      for val in data[8:11])
        return True
    
    def stage_1_hardware_information_feeding(self):
        """Stage 1: Hardware Information Feeding"""
        self.print_header("HARDWARE INFORMATION FEEDING")
        
        print("🔌 Hardware Components Required:")
        components = [
            "ESP32 Development Board",
            "5x Flex Sensors (2.2\" or 4.5\")",
            "MPU6050 IMU Sensor",
            "Connecting Wires",
            "Data Glove for Mounting"
        ]
        
        for i, component in enumerate(components, 1):
            print(f"  {i}. {component}")
        
        self.print_step(1, "Physical Assembly")
        print("Wiring Configuration:")
        wiring = {
            "Flex Sensor 1 (Thumb)": "ESP32 Pin 32",
            "Flex Sensor 2 (Index)": "ESP32 Pin 33", 
            "Flex Sensor 3 (Middle)": "ESP32 Pin 34",
            "Flex Sensor 4 (Ring)": "ESP32 Pin 35",
            "Flex Sensor 5 (Pinky)": "ESP32 Pin 36",
            "MPU6050 SDA": "ESP32 Pin 21",
            "MPU6050 SCL": "ESP32 Pin 22",
            "MPU6050 VCC": "ESP32 3.3V",
            "MPU6050 GND": "ESP32 GND"
        }
        
        for sensor, pin in wiring.items():
            print(f"  {sensor} → {pin}")
        
        self.print_step(2, "Power Supply Verification")
        print("✅ Connect USB cable to ESP32")
        print("✅ Verify 3.3V power supply to sensors")
        print("✅ Check for proper grounding")
        
        self.print_step(3, "Sensor Testing")
        print("Expected Sensor Ranges:")
        print(f"  Flex Sensors: {self.config['flex_thresholds']['min']}-{self.config['flex_thresholds']['max']}")
        print(f"  Accelerometer: {self.config['accel_range']['min']}-{self.config['accel_range']['max']} m/s²")
        print(f"  Gyroscope: {self.config['gyro_range']['min']}-{self.config['gyro_range']['max']}°/s")
        
        self.system_status['hardware'] = True
        print("\n✅ Hardware information feeding completed!")
    
    def stage_2_firmware_information_feeding(self):
        """Stage 2: Firmware Information Feeding"""
        self.print_header("FIRMWARE INFORMATION FEEDING")
        
        self.print_step(1, "Arduino IDE Setup")
        print("Required Configuration:")
        print("  Board: ESP32 Dev Module")
        print("  Upload Speed: 115200")
        print("  Required Libraries:")
        print("    - Adafruit_MPU6050")
        print("    - Adafruit_Unified_Sensor")
        
        self.print_step(2, "Firmware Parameters")
        firmware_config = {
            "Pin Definitions": {
                "FLEX1_PIN": 32,
                "FLEX2_PIN": 33, 
                "FLEX3_PIN": 34,
                "FLEX4_PIN": 35,
                "FLEX5_PIN": 36
            },
            "Calibration Settings": {
                "SAMPLES_PER_READING": 10,
                "STABLE_READING_THRESHOLD": 100,
                "CALIBRATION_SAMPLES": 30
            },
            "MPU6050 Settings": {
                "ACCELEROMETER_RANGE": "MPU6050_RANGE_8_G",
                "GYRO_RANGE": "MPU6050_RANGE_500_DEG",
                "FILTER_BANDWIDTH": "MPU6050_BAND_21_HZ"
            }
        }
        
        for category, settings in firmware_config.items():
            print(f"\n{category}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        self.print_step(3, "Upload Process")
        print("1. Open Arduino IDE")
        print("2. Select ESP32 board")
        print("3. Install required libraries")
        print("4. Upload esp32_asl.ino")
        print("5. Verify upload success")
        
        self.system_status['firmware'] = True
        print("\n✅ Firmware information feeding completed!")
    
    def stage_3_data_collection_information_feeding(self):
        """Stage 3: Data Collection Information Feeding"""
        self.print_header("DATA COLLECTION INFORMATION FEEDING")
        
        self.print_step(1, "Configuration Setup")
        print("Data Collection Parameters:")
        for key, value in self.config.items():
            if key != 'flex_thresholds' and key != 'accel_range' and key != 'gyro_range':
                print(f"  {key}: {value}")
        
        self.print_step(2, "Gesture Definitions")
        gestures = {
            'A': 'Thumb out, fingers closed',
            'B': 'All fingers straight',
            'C': 'Fingers curved like C',
            'D': 'Index finger pointing up',
            'E': 'All fingers bent',
            'F': 'Thumb and index touching',
            'G': 'Index finger pointing left',
            'H': 'Index and middle fingers straight',
            'I': 'Pinky finger up',
            'J': 'Index finger drawing J'
        }
        
        for gesture, description in gestures.items():
            print(f"  {gesture}: {description}")
        
        self.print_step(3, "Data Collection Methods")
        print("Method 1: Advanced Data Collection")
        print("  Command: python asl_data_collector.py")
        print("  Features: Real-time visualization, calibration, validation")
        
        print("\nMethod 2: Simple Data Collection")
        print("  Command: python asl_train.py")
        print("  Features: Manual input, basic validation")
        
        self.print_step(4, "Data Validation Rules")
        validation_rules = [
            "Flex sensor values: 0-4095",
            "Accelerometer: ±20 m/s²",
            "Gyroscope: ±500°/s",
            "Stability check: variance < threshold",
            "Minimum samples per gesture: 50"
        ]
        
        for rule in validation_rules:
            print(f"  ✅ {rule}")
        
        self.system_status['data_collection'] = True
        print("\n✅ Data collection information feeding completed!")
    
    def stage_4_preprocessing_information_feeding(self):
        """Stage 4: Data Preprocessing Information Feeding"""
        self.print_header("DATA PREPROCESSING INFORMATION FEEDING")
        
        self.print_step(1, "Data Sources")
        data_sources = [
            "asl_training_data.csv (Real sensor data)",
            "synthetic_asl_data.csv (Generated data)",
            "External datasets (Public repositories)",
            "Preprocessed data (Previous runs)"
        ]
        
        for source in data_sources:
            print(f"  📁 {source}")
        
        self.print_step(2, "Preprocessing Parameters")
        preprocessing_config = {
            "Scaling": "StandardScaler",
            "Test Size": "20%",
            "Random State": 42,
            "Feature Columns": [
                "flex1", "flex2", "flex3", "flex4", "flex5",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z"
            ]
        }
        
        for category, value in preprocessing_config.items():
            print(f"  {category}: {value}")
        
        self.print_step(3, "Processing Steps")
        steps = [
            "Load and combine all data sources",
            "Handle missing values and outliers",
            "Scale features to standard range",
            "Encode categorical labels",
            "Split into training and test sets",
            "Generate data visualizations",
            "Save preprocessed data"
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        
        self.print_step(4, "Output Files")
        output_files = [
            "X_train.npy (Training features)",
            "X_test.npy (Test features)",
            "y_train.npy (Training labels)",
            "y_test.npy (Test labels)",
            "feature_names.csv (Feature names)",
            "label_mapping.csv (Label encoding)"
        ]
        
        for file in output_files:
            print(f"  💾 {file}")
        
        self.system_status['preprocessing'] = True
        print("\n✅ Data preprocessing information feeding completed!")
    
    def stage_5_ml_training_information_feeding(self):
        """Stage 5: Machine Learning Information Feeding"""
        self.print_header("MACHINE LEARNING INFORMATION FEEDING")
        
        self.print_step(1, "Model Configuration")
        models = {
            "Random Forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "SVM": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"
            },
            "Gradient Boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3
            },
            "Neural Network": {
                "hidden_layer_sizes": "(100, 50)",
                "activation": "relu",
                "solver": "adam"
            }
        }
        
        for model_name, params in models.items():
            print(f"\n{model_name}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        self.print_step(2, "Training Parameters")
        training_config = {
            "Cross-validation folds": 5,
            "Scoring metric": "accuracy",
            "Parallel processing": "enabled",
            "Verbose output": "enabled"
        }
        
        for param, value in training_config.items():
            print(f"  {param}: {value}")
        
        self.print_step(3, "Training Process")
        training_steps = [
            "Load preprocessed training data",
            "Initialize multiple models",
            "Perform cross-validation",
            "Train models on full dataset",
            "Evaluate performance metrics",
            "Select best performing model",
            "Save trained models and scalers"
        ]
        
        for i, step in enumerate(training_steps, 1):
            print(f"  {i}. {step}")
        
        self.print_step(4, "Evaluation Metrics")
        metrics = [
            "Accuracy score",
            "Precision and recall",
            "F1-score",
            "Confusion matrix",
            "Cross-validation score"
        ]
        
        for metric in metrics:
            print(f"  📊 {metric}")
        
        self.system_status['ml_training'] = True
        print("\n✅ Machine learning information feeding completed!")
    
    def stage_6_prediction_information_feeding(self):
        """Stage 6: Real-time Prediction Information Feeding"""
        self.print_header("REAL-TIME PREDICTION INFORMATION FEEDING")
        
        self.print_step(1, "Model Loading")
        model_files = [
            "best_model.pkl (Trained model)",
            "scaler.pkl (Feature scaler)",
            "label_encoder.pkl (Label encoder)"
        ]
        
        for file in model_files:
            print(f"  📁 {file}")
        
        self.print_step(2, "Prediction Configuration")
        prediction_config = {
            "Confidence threshold": 0.7,
            "Prediction interval": "0.5 seconds",
            "Smoothing window": 5,
            "Text-to-speech": "enabled"
        }
        
        for param, value in prediction_config.items():
            print(f"  {param}: {value}")
        
        self.print_step(3, "Real-time Data Flow")
        data_flow = [
            "Read sensor data from ESP32",
            "Parse and validate data",
            "Extract features",
            "Scale features using saved scaler",
            "Make prediction using trained model",
            "Check confidence threshold",
            "Convert prediction to gesture label",
            "Output text-to-speech"
        ]
        
        for i, step in enumerate(data_flow, 1):
            print(f"  {i}. {step}")
        
        self.print_step(4, "Output Generation")
        output_types = [
            "Console text output",
            "Text-to-speech audio",
            "Real-time visualization",
            "Prediction confidence display"
        ]
        
        for output in output_types:
            print(f"  🔊 {output}")
        
        self.system_status['prediction'] = True
        print("\n✅ Real-time prediction information feeding completed!")
    
    def run_complete_workflow(self):
        """Run complete information feeding workflow"""
        print("🚀 ASL Recognition System - Complete Information Feeding Workflow")
        print("="*80)
        
        # Run all stages
        self.stage_1_hardware_information_feeding()
        self.stage_2_firmware_information_feeding()
        self.stage_3_data_collection_information_feeding()
        self.stage_4_preprocessing_information_feeding()
        self.stage_5_ml_training_information_feeding()
        self.stage_6_prediction_information_feeding()
        
        # Final summary
        self.print_header("WORKFLOW SUMMARY")
        print("System Status:")
        for stage, status in self.system_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {stage.replace('_', ' ').title()}")
        
        print("\n🎯 Next Steps:")
        print("1. Follow the hardware setup guide")
        print("2. Upload firmware to ESP32")
        print("3. Collect training data")
        print("4. Preprocess data")
        print("5. Train machine learning models")
        print("6. Test real-time prediction")
        
        print("\n📚 Additional Resources:")
        print("- DATA_COLLECTION_GUIDE.md")
        print("- INFORMATION_FEEDING_GUIDE.md")
        print("- asl_system_flowchart.png")
        
        print("\n✅ Complete information feeding workflow finished!")

def main():
    """Main function to run the information feeding workflow"""
    feeder = ASLInformationFeeder()
    feeder.run_complete_workflow()

if __name__ == "__main__":
    main() 