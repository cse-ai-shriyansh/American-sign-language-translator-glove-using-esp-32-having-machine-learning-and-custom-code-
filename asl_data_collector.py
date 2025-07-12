import serial
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import os
from datetime import datetime

class ASLDataCollector:
    def __init__(self, port='COM3', baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.data_queue = queue.Queue()
        self.is_collecting = False
        self.current_data = None
        
        # Data storage
        self.collected_data = []
        self.labels = []
        
        # Visualization
        self.fig, self.ax = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('ASL Sensor Data Collection', fontsize=16)
        
    def connect(self):
        """Connect to ESP32"""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=2)
            print(f"Connected to ESP32 on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to ESP32: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        if self.ser:
            self.ser.close()
            print("Disconnected from ESP32")
    
    def send_command(self, command):
        """Send command to ESP32"""
        if self.ser:
            self.ser.write((command + '\n').encode())
            time.sleep(0.1)
    
    def calibrate_sensors(self):
        """Calibrate sensors on ESP32"""
        print("Calibrating sensors...")
        print("Keep your hand flat and still for 3 seconds")
        self.send_command("CALIBRATE")
        time.sleep(4)  # Wait for calibration to complete
        print("Calibration complete!")
    
    def get_sensor_status(self):
        """Get current sensor status"""
        self.send_command("STATUS")
        time.sleep(0.5)
        while self.ser.in_waiting:
            print(self.ser.readline().decode('utf-8').strip())
    
    def parse_sensor_data(self, data_string):
        """Parse sensor data string into dictionary"""
        try:
            if data_string == "UNSTABLE":
                return None
            
            values = [float(x.strip()) for x in data_string.split(',')]
            if len(values) != 11:
                return None
            
            return {
                'flex1': values[0],  # Thumb
                'flex2': values[1],  # Index
                'flex3': values[2],  # Middle
                'flex4': values[3],  # Ring
                'flex5': values[4],  # Pinky
                'accel_x': values[5],
                'accel_y': values[6],
                'accel_z': values[7],
                'gyro_x': values[8],
                'gyro_y': values[9],
                'gyro_z': values[10],
                'timestamp': time.time()
            }
        except:
            return None
    
    def data_collection_thread(self):
        """Background thread for collecting sensor data"""
        while self.is_collecting:
            if self.ser and self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        data = self.parse_sensor_data(line)
                        if data:
                            self.data_queue.put(data)
                            self.current_data = data
                except:
                    pass
            time.sleep(0.01)
    
    def start_data_collection(self):
        """Start background data collection"""
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self.data_collection_thread)
        self.collection_thread.daemon = True
        self.collection_thread.start()
    
    def stop_data_collection(self):
        """Stop background data collection"""
        self.is_collecting = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join()
    
    def collect_samples(self, label, num_samples=50, sample_interval=0.2):
        """Collect multiple samples for a specific gesture"""
        print(f"\nCollecting {num_samples} samples for gesture '{label}'")
        print("Make the gesture and hold it steady...")
        
        samples = []
        stable_count = 0
        
        for i in range(num_samples):
            # Wait for stable data
            while True:
                if not self.data_queue.empty():
                    data = self.data_queue.get()
                    if data:
                        samples.append(data)
                        stable_count += 1
                        break
                time.sleep(0.01)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Collected {i + 1}/{num_samples} samples")
            
            time.sleep(sample_interval)
        
        # Add samples to collected data
        for sample in samples:
            self.collected_data.append(sample)
            self.labels.append(label)
        
        print(f"Successfully collected {len(samples)} samples for '{label}'")
        return samples
    
    def save_data(self, filename=None):
        """Save collected data to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"asl_training_data_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['label', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 
                         'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, data in enumerate(self.collected_data):
                row = data.copy()
                row['label'] = self.labels[i]
                writer.writerow(row)
        
        print(f"Data saved to {filename}")
        return filename
    
    def show_data_summary(self):
        """Show summary of collected data"""
        if not self.collected_data:
            print("No data collected yet.")
            return
        
        print("\n=== Data Collection Summary ===")
        print(f"Total samples: {len(self.collected_data)}")
        
        # Count samples per label
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Samples per gesture:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        # Show data ranges
        if self.collected_data:
            flex_data = np.array([[d['flex1'], d['flex2'], d['flex3'], d['flex4'], d['flex5']] 
                                 for d in self.collected_data])
            print(f"\nFlex sensor ranges:")
            print(f"  Min: {np.min(flex_data, axis=0)}")
            print(f"  Max: {np.max(flex_data, axis=0)}")
            print(f"  Mean: {np.mean(flex_data, axis=0)}")
    
    def real_time_plot(self, frame):
        """Update real-time plot"""
        if self.current_data:
            # Clear previous plots
            self.ax[0].clear()
            self.ax[1].clear()
            
            # Plot flex sensors
            flex_values = [self.current_data['flex1'], self.current_data['flex2'], 
                          self.current_data['flex3'], self.current_data['flex4'], 
                          self.current_data['flex5']]
            fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            
            self.ax[0].bar(fingers, flex_values, color=['red', 'blue', 'green', 'orange', 'purple'])
            self.ax[0].set_title('Flex Sensor Values')
            self.ax[0].set_ylabel('Value')
            self.ax[0].set_ylim(0, 4095)
            
            # Plot IMU data
            accel_values = [self.current_data['accel_x'], self.current_data['accel_y'], self.current_data['accel_z']]
            gyro_values = [self.current_data['gyro_x'], self.current_data['gyro_y'], self.current_data['gyro_z']]
            
            x = np.arange(3)
            width = 0.35
            
            self.ax[1].bar(x - width/2, accel_values, width, label='Accelerometer', alpha=0.8)
            self.ax[1].bar(x + width/2, gyro_values, width, label='Gyroscope', alpha=0.8)
            self.ax[1].set_title('IMU Data')
            self.ax[1].set_ylabel('Value')
            self.ax[1].set_xticks(x)
            self.ax[1].set_xticklabels(['X', 'Y', 'Z'])
            self.ax[1].legend()
            
            plt.tight_layout()

def main():
    """Main data collection interface"""
    print("🤖 ASL Sensor Data Collection System")
    print("="*50)
    
    # Initialize collector
    collector = ASLDataCollector()
    
    # Connect to ESP32
    if not collector.connect():
        print("Please check your ESP32 connection and try again.")
        return
    
    try:
        # Show menu
        while True:
            print("\nOptions:")
            print("1. Calibrate sensors")
            print("2. Show sensor status")
            print("3. Start data collection")
            print("4. Show data summary")
            print("5. Save data")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                collector.calibrate_sensors()
                
            elif choice == '2':
                collector.get_sensor_status()
                
            elif choice == '3':
                # Start data collection
                collector.start_data_collection()
                
                # Start real-time plotting
                ani = FuncAnimation(collector.fig, collector.real_time_plot, interval=100)
                plt.show(block=False)
                
                # Collect data for different gestures
                gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                
                for gesture in gestures:
                    response = input(f"\nCollect data for gesture '{gesture}'? (y/n/skip): ").strip().lower()
                    
                    if response == 'y':
                        samples = int(input("Number of samples (default 50): ") or "50")
                        collector.collect_samples(gesture, samples)
                    elif response == 'skip':
                        continue
                    else:
                        break
                
                # Stop data collection
                collector.stop_data_collection()
                plt.close()
                
            elif choice == '4':
                collector.show_data_summary()
                
            elif choice == '5':
                if collector.collected_data:
                    filename = input("Enter filename (or press Enter for auto-generated): ").strip()
                    if not filename:
                        filename = None
                    collector.save_data(filename)
                else:
                    print("No data to save.")
                    
            elif choice == '6':
                break
                
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nData collection interrupted.")
    
    finally:
        # Save any remaining data
        if collector.collected_data:
            save = input("\nSave collected data before exiting? (y/n): ").strip().lower()
            if save == 'y':
                collector.save_data()
        
        collector.disconnect()
        print("Data collection session ended.")

if __name__ == "__main__":
    main() 