import tkinter as tk
from tkinter import ttk
import serial
import threading
import time
import random

# --- Configuration ---
SERIAL_PORT = 'COM3'  # Change this to your ESP32's COM port later
BAUD_RATE = 115200

class SensorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor System Manager")
        self.root.geometry("500x350")
        self.root.configure(bg="#1e1e1e") # Dark mode background
        self.root.resizable(False, False)

        # Style Configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#d4d4d4", font=("Consolas", 12))
        style.configure("Header.TLabel", foreground="#569cd6", font=("Consolas", 14, "bold"))
        style.configure("Value.TLabel", foreground="#4ec9b0", font=("Consolas", 12, "bold"))

        # String Variables to hold our dynamic text
        self.status_var = tk.StringVar(value="[+] Hardware Status: DISCONNECTED (Simulation Mode)")
        self.pt100_var = tk.StringVar(value="--.-- °C")
        self.ktype_var = tk.StringVar(value="--.-- °C")
        self.corrected_var = tk.StringVar(value="--.-- °C")
        self.error_var = tk.StringVar(value="--.-- °C")

        self.build_ui()
        
        # Start the data-reading thread
        self.running = True
        self.data_thread = threading.Thread(target=self.read_data, daemon=True)
        self.data_thread.start()

    def build_ui(self):
        # Header
        ttk.Label(self.root, text="==================================================", font=("Consolas", 12)).pack(pady=(10, 0))
        ttk.Label(self.root, text="  SENSOR SYSTEM MANAGER", style="Header.TLabel").pack(anchor="w", padx=10)
        ttk.Label(self.root, text="==================================================", font=("Consolas", 12)).pack()

        # Status
        ttk.Label(self.root, textvariable=self.status_var, foreground="#ce9178").pack(anchor="w", padx=15, pady=(5, 10))
        
        # Device Tree Frame
        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(fill="both", expand=True, padx=15)

        # 1. Physical Sensors Section
        ttk.Label(tree_frame, text="[-] Physical Sensors").grid(row=0, column=0, sticky="w", pady=(5, 0))
        
        ttk.Label(tree_frame, text="  |-- RTD PT100 (Reference)   : ").grid(row=1, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.pt100_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")

        ttk.Label(tree_frame, text="  |-- K-Type Thermocouple     : ").grid(row=2, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.ktype_var, style="Value.TLabel").grid(row=2, column=1, sticky="w")

        # 2. Virtual Devices Section
        ttk.Label(tree_frame, text="\n[-] Virtual Devices").grid(row=3, column=0, sticky="w")
        
        ttk.Label(tree_frame, text="  |-- LSTM Correction Engine  : ").grid(row=4, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.corrected_var, style="Value.TLabel", foreground="#c586c0").grid(row=4, column=1, sticky="w")

        ttk.Label(tree_frame, text="  |-- Current Dynamic Error   : ").grid(row=5, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.error_var, style="Value.TLabel").grid(row=5, column=1, sticky="w")

        ttk.Label(self.root, text="==================================================", font=("Consolas", 12)).pack(side="bottom", pady=10)

    # --- NEW: Thread-Safe UI Updater ---
    def update_ui_safe(self, status, pt100, ktype, corrected, error):
        if status: self.status_var.set(status)
        if pt100: self.pt100_var.set(pt100)
        if ktype: self.ktype_var.set(ktype)
        if corrected: self.corrected_var.set(corrected)
        if error: self.error_var.set(error)

    def read_data(self):
        """Runs in the background, reading COM port or generating test data"""
        try:
            # Try to connect to ESP32
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            self.root.after(0, self.update_ui_safe, f"[+] Hardware Status: CONNECTED ({SERIAL_PORT})", None, None, None, None)
            last_data_time = time.time()

            while self.running:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            # Safely send new data to the main UI thread
                            k_val = f"{float(parts[1]):.2f} °C"
                            pt_val = f"{float(parts[2]):.2f} °C"
                            corr_val = f"{float(parts[3]):.2f} °C"
                            err_val = f"{(float(parts[3]) - float(parts[2])):+.2f} °C"
                            
                            self.root.after(0, self.update_ui_safe, None, pt_val, k_val, corr_val, err_val)
                            last_data_time = time.time()
                        except ValueError:
                            pass

                # If no valid serial data arrives for several seconds, switch to simulation mode.
                if time.time() - last_data_time > 5:
                    raise RuntimeError("No serial data received, switching to simulation mode")

                time.sleep(0.1)
        except (serial.SerialException, OSError, RuntimeError):
            # If no ESP32 is found or the port stays silent, run Simulation Mode
            self.root.after(0, self.update_ui_safe, "[+] Hardware Status: DISCONNECTED (Simulation Mode)", None, None, None, None)
            base_temp = 25.0
            
            while self.running:
                # Generate fake numbers slightly drifting
                base_temp += random.uniform(-0.5, 0.5)
                k_temp = base_temp + random.uniform(-1.0, 1.0) # Noisy
                lstm_temp = base_temp + random.uniform(-0.1, 0.1) # Accurate
                
                pt_val = f"{base_temp:.2f} °C"
                k_val = f"{k_temp:.2f} °C"
                corr_val = f"{lstm_temp:.2f} °C"
                err_val = f"{lstm_temp - base_temp:+.2f} °C"
                
                # Safely send simulated data to the main UI thread
                self.root.after(0, self.update_ui_safe, None, pt_val, k_val, corr_val, err_val)
                time.sleep(0.5) # Update at 2Hz

    def on_close(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()