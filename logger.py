import serial
import time
import os

# --- Configuration ---
SERIAL_PORT = 'COM8'  # <--- CHANGE THIS to match your ESP32's COM port (e.g., COM8)
BAUD_RATE = 115200
OUTPUT_FILE = 'dataset.csv'

def start_logging():
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print(f"Logging clean sensor data to {OUTPUT_FILE}... (Press Ctrl+C to stop)")

        # Check if file exists and has content
        file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0

        # Open the CSV file in append mode
        with open(OUTPUT_FILE, 'a') as file:
            # Write the header only if file is new or empty
            if not file_exists:
                file.write("Timestamp_ms,K_Type_Temp_C,PT100_Temp_C,Error_C\n")
            file.flush()
            
            while True:
                # Check if there is data waiting in the serial buffer
                if ser.in_waiting > 0:
                    # Read the line, decode it, and strip extra spaces/newlines
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Ignore boot/diagnostic or error messages (which start with '[')
                    if not line or line.startswith('['):
                        if line:
                            print(f"  [ESP32 System Msg] {line}")
                        continue
                        
                    # Ignore the header printed by the ESP32
                    if "Timestamp_ms" in line:
                        continue
                    
                    # Parse the comma-separated data values
                    parts = line.split(',')
                    if len(parts) >= 8:
                        try:
                            timestamp = parts[0].strip()
                            live_k_temp = float(parts[1].strip())
                            live_pt100_temp = float(parts[3].strip())
                            
                            # Read sensor validity flags (1 = OK, 0 = fault/fallback)
                            k_sensor_ok = parts[6].strip() == "1"
                            pt100_sensor_ok = parts[7].strip() == "1"
                            
                            if not k_sensor_ok or not pt100_sensor_ok:
                                # Warn the user but do not write faulty sensor data to the training set
                                warnings = []
                                if not k_sensor_ok:
                                    warnings.append("K-Type Fault")
                                if not pt100_sensor_ok:
                                    warnings.append("PT100 Fault")
                                print(f"  [WARNING] Skipping sample - Sensor Error: {', '.join(warnings)}")
                                continue
                            
                            # Calculate the true dynamic error between cheap sensor and reference
                            error = live_k_temp - live_pt100_temp
                            
                            # Format line matching the training CSV
                            csv_row = f"{timestamp},{live_k_temp:.2f},{live_pt100_temp:.2f},{error:.2f}"
                            
                            # Print to the console so you can monitor real sensor readings
                            print(f"Time: {int(timestamp)/1000:6.1f}s | K-Type: {live_k_temp:5.2f}°C | PT100: {live_pt100_temp:5.2f}°C | Error: {error:+5.2f}°C")
                            
                            # Write to the CSV file and flush to disk
                            file.write(csv_row + '\n')
                            file.flush()
                            
                        except ValueError:
                            # Skip lines that are corrupted or fail to parse as numbers
                            pass
                            
    except KeyboardInterrupt:
        print("\nLogging stopped by user. File saved successfully.")
    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Check if the COM port is correct and not open in the Arduino IDE or GUI.")
    finally:
        # Safely close the serial port when exiting
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    start_logging()