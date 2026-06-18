import serial
import time

# --- Configuration ---
SERIAL_PORT = 'COM3'  # <--- CHANGE THIS to match your ESP32 COM port
BAUD_RATE = 115200
OUTPUT_FILE = 'dataset.csv'

def start_logging():
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print(f"Writing data to {OUTPUT_FILE}... (Press Ctrl+C to stop)")
        
        # Open the CSV file in write mode
        with open(OUTPUT_FILE, 'w') as file:
            while True:
                # Check if there is data waiting in the serial buffer
                if ser.in_waiting > 0:
                    # Read the line, decode it, and strip extra spaces/newlines
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Print to the console so you can monitor it
                    print(line)
                    
                    # Write to the CSV file and force it to save to the disk immediately
                    file.write(line + '\n')
                    file.flush() 
                    
    except KeyboardInterrupt:
        print("\nLogging stopped by user. File saved successfully.")
    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Check if the COM port is correct and not open in the Arduino IDE.")
    finally:
        # Safely close the serial port when exiting
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    start_logging()