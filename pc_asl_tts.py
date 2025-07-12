import serial
import pyttsx3

SERIAL_PORT = 'COM3'  # Change to your ESP32 port
BAUD_RATE = 115200

engine = pyttsx3.init()
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

print("Listening for ASL translations...")

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            print("Detected:", line)
            engine.say(line)
            engine.runAndWait()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close() 