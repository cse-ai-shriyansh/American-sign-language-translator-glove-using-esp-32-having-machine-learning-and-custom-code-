# ASL Data Collection Configuration
# Modify these settings according to your hardware setup and requirements

# Serial Communication Settings
SERIAL_PORT = 'COM3'  # Change to your ESP32 port (e.g., 'COM3', '/dev/ttyUSB0', '/dev/ttyACM0')
BAUD_RATE = 115200
TIMEOUT = 2

# Data Collection Settings
SAMPLES_PER_GESTURE = 50  # Number of samples to collect per gesture
SAMPLE_INTERVAL = 0.2     # Time between samples in seconds
STABLE_READING_COUNT = 5  # Number of readings to average for stability

# Sensor Calibration Settings
CALIBRATION_SAMPLES = 30  # Number of samples for calibration
CALIBRATION_DURATION = 3  # Duration of calibration in seconds

# Flex Sensor Thresholds (adjust based on your sensors)
FLEX_STRAIGHT_THRESHOLD = 1800  # Value when finger is straight
FLEX_BENT_THRESHOLD = 2200      # Value when finger is bent
STABLE_READING_THRESHOLD = 100  # Maximum variance for stable readings

# Supported Gestures
SUPPORTED_GESTURES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'HELLO', 'THANK_YOU', 'PLEASE', 'YES', 'NO', 'GOOD', 'BAD'
]

# Data File Settings
DEFAULT_OUTPUT_FILE = 'asl_training_data.csv'
AUTO_SAVE_INTERVAL = 100  # Save data every N samples

# Visualization Settings
REAL_TIME_PLOT = True
PLOT_UPDATE_INTERVAL = 100  # milliseconds
PLOT_WINDOW_SIZE = (12, 8)

# ESP32 Pin Configuration
ESP32_PINS = {
    'FLEX1_PIN': 32,  # Thumb
    'FLEX2_PIN': 33,  # Index
    'FLEX3_PIN': 34,  # Middle
    'FLEX4_PIN': 35,  # Ring
    'FLEX5_PIN': 36,  # Pinky
    'MPU_SDA': 21,    # MPU6050 SDA
    'MPU_SCL': 22     # MPU6050 SCL
}

# MPU6050 Settings
MPU6050_CONFIG = {
    'ACCELEROMETER_RANGE': 'MPU6050_RANGE_8_G',
    'GYRO_RANGE': 'MPU6050_RANGE_500_DEG',
    'FILTER_BANDWIDTH': 'MPU6050_BAND_21_HZ'
}

# Data Validation Settings
MIN_FLEX_VALUE = 0
MAX_FLEX_VALUE = 4095
MIN_ACCEL_VALUE = -20
MAX_ACCEL_VALUE = 20
MIN_GYRO_VALUE = -500
MAX_GYRO_VALUE = 500

# Quality Control Settings
ENABLE_DATA_VALIDATION = True
ENABLE_STABILITY_CHECK = True
MIN_STABLE_SAMPLES = 3
MAX_RETRY_ATTEMPTS = 10

# Logging Settings
ENABLE_LOGGING = True
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'asl_data_collection.log'

# Advanced Settings
USE_MULTITHREADING = True
BUFFER_SIZE = 1000
DATA_COMPRESSION = False
BACKUP_INTERVAL = 50  # Create backup every N samples 