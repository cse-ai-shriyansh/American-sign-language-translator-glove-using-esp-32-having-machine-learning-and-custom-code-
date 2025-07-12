#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Pin definitions
#define FLEX1_PIN 32  // Thumb
#define FLEX2_PIN 33  // Index
#define FLEX3_PIN 34  // Middle
#define FLEX4_PIN 35  // Ring
#define FLEX5_PIN 4  // Pinky (Adjusted for 35-pin ESP32 board - please verify this pin is suitable)

// Calibration values (adjust these based on your sensors)
#define FLEX_STRAIGHT_THRESHOLD 1800
#define FLEX_BENT_THRESHOLD 2200
#define SAMPLES_PER_READING 10
#define STABLE_READING_THRESHOLD 100

Adafruit_MPU6050 mpu;

// Calibration offsets
int flexOffsets[5] = {0, 0, 0, 0, 0};
float accelOffsets[3] = {0, 0, 0};
float gyroOffsets[3] = {0, 0, 0};

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Initialize flex sensor pins
  pinMode(FLEX1_PIN, INPUT);
  pinMode(FLEX2_PIN, INPUT);
  pinMode(FLEX3_PIN, INPUT);
  pinMode(FLEX4_PIN, INPUT);
  pinMode(FLEX5_PIN, INPUT);

  Serial.println("ESP32 ASL Sensor System Ready");
  Serial.println("Commands:");
  Serial.println("CALIBRATE - Start calibration");
  Serial.println("DATA - Start data collection");
  Serial.println("STATUS - Show sensor status");
}

void calibrateSensors() {
  Serial.println("Starting calibration...");
  Serial.println("Keep your hand flat and still for 3 seconds");
  
  // Calibrate flex sensors
  int flexSum[5] = {0, 0, 0, 0, 0};
  float accelSum[3] = {0, 0, 0};
  float gyroSum[3] = {0, 0, 0};
  
  for (int i = 0; i < 30; i++) {  // 30 samples over 3 seconds
    flexSum[0] += analogRead(FLEX1_PIN);
    flexSum[1] += analogRead(FLEX2_PIN);
    flexSum[2] += analogRead(FLEX3_PIN);
    flexSum[3] += analogRead(FLEX4_PIN);
    flexSum[4] += analogRead(FLEX5_PIN);
    
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    accelSum[0] += a.acceleration.x;
    accelSum[1] += a.acceleration.y;
    accelSum[2] += a.acceleration.z;
    
    gyroSum[0] += g.gyro.x;
    gyroSum[1] += g.gyro.y;
    gyroSum[2] += g.gyro.z;
    
    delay(100);
  }
  
  // Calculate offsets
  for (int i = 0; i < 5; i++) {
    flexOffsets[i] = flexSum[i] / 30;
  }
  
  for (int i = 0; i < 3; i++) {
    accelOffsets[i] = accelSum[i] / 30;
    gyroOffsets[i] = gyroSum[i] / 30;
  }
  
  Serial.println("Calibration complete!");
  Serial.println("Flex offsets: " + String(flexOffsets[0]) + "," + String(flexOffsets[1]) + "," + 
                 String(flexOffsets[2]) + "," + String(flexOffsets[3]) + "," + String(flexOffsets[4]));
}

int getStableFlexReading(int pin, int offset) {
  int readings[SAMPLES_PER_READING];
  int sum = 0;
  
  // Take multiple readings
  for (int i = 0; i < SAMPLES_PER_READING; i++) {
    readings[i] = analogRead(pin) - offset;
    sum += readings[i];
    delay(5);
  }
  
  // Check if readings are stable (low variance)
  int avg = sum / SAMPLES_PER_READING;
  int variance = 0;
  for (int i = 0; i < SAMPLES_PER_READING; i++) {
    variance += (readings[i] - avg) * (readings[i] - avg);
  }
  variance /= SAMPLES_PER_READING;
  
  // If readings are stable, return average; otherwise return -1
  if (variance < STABLE_READING_THRESHOLD) {
    return avg;
  } else {
    return -1;  // Unstable reading
  }
}

void getStableIMUReadings(float* accel, float* gyro) {
  float accelSum[3] = {0, 0, 0};
  float gyroSum[3] = {0, 0, 0};
  
  for (int i = 0; i < SAMPLES_PER_READING; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    accelSum[0] += a.acceleration.x - accelOffsets[0];
    accelSum[1] += a.acceleration.y - accelOffsets[1];
    accelSum[2] += a.acceleration.z - accelOffsets[2];
    
    gyroSum[0] += g.gyro.x - gyroOffsets[0];
    gyroSum[1] += g.gyro.y - gyroOffsets[1];
    gyroSum[2] += g.gyro.z - gyroOffsets[2];
    
    delay(5);
  }
  
  for (int i = 0; i < 3; i++) {
    accel[i] = accelSum[i] / SAMPLES_PER_READING;
    gyro[i] = gyroSum[i] / SAMPLES_PER_READING;
  }
}

void sendSensorData() {
  // Get stable flex sensor readings
  int flex1 = getStableFlexReading(FLEX1_PIN, flexOffsets[0]);
  int flex2 = getStableFlexReading(FLEX2_PIN, flexOffsets[1]);
  int flex3 = getStableFlexReading(FLEX3_PIN, flexOffsets[2]);
  int flex4 = getStableFlexReading(FLEX4_PIN, flexOffsets[3]);
  int flex5 = getStableFlexReading(FLEX5_PIN, flexOffsets[4]);
  
  // Get stable IMU readings
  float accel[3], gyro[3];
  getStableIMUReadings(accel, gyro);
  
  // Check if all readings are stable
  if (flex1 != -1 && flex2 != -1 && flex3 != -1 && flex4 != -1 && flex5 != -1) {
    // Format: flex1,flex2,flex3,flex4,flex5,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
    String sensor_data = String(flex1) + "," + String(flex2) + "," + String(flex3) + "," + 
                        String(flex4) + "," + String(flex5) + "," + 
                        String(accel[0], 3) + "," + String(accel[1], 3) + "," + String(accel[2], 3) + "," +
                        String(gyro[0], 3) + "," + String(gyro[1], 3) + "," + String(gyro[2], 3);
    
    Serial.println(sensor_data);
  } else {
    Serial.println("UNSTABLE");  // Indicate unstable readings
  }
}

void showSensorStatus() {
  Serial.println("=== Sensor Status ===");
  Serial.println("Flex Sensors (raw values):");
  Serial.println("Thumb: " + String(analogRead(FLEX1_PIN)));
  Serial.println("Index: " + String(analogRead(FLEX2_PIN)));
  Serial.println("Middle: " + String(analogRead(FLEX3_PIN)));
  Serial.println("Ring: " + String(analogRead(FLEX4_PIN)));
  Serial.println("Pinky: " + String(analogRead(FLEX5_PIN)));
  
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  Serial.println("Accelerometer: " + String(a.acceleration.x, 3) + "," + 
                 String(a.acceleration.y, 3) + "," + String(a.acceleration.z, 3));
  Serial.println("Gyroscope: " + String(g.gyro.x, 3) + "," + 
                 String(g.gyro.y, 3) + "," + String(g.gyro.z, 3));
  Serial.println("===================");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "CALIBRATE") {
      calibrateSensors();
    } else if (command == "STATUS") {
      showSensorStatus();
    } else if (command == "DATA") {
      Serial.println("Starting data collection mode...");
      // Continue to data collection loop
    }
  }
  
  // Default: send sensor data continuously
  sendSensorData();
  delay(100);  // 10Hz sampling rate
} 