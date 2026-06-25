import tkinter as tk
from tkinter import ttk
import serial
import threading
import time
import random

# --- Configuration ---
SERIAL_PORT = 'COM8'  # Change this to your ESP32's COM port later
BAUD_RATE = 115200


class SensorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MAX6675 / MAX31865 Sensor System Manager")
        self.root.geometry("620x450")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#d4d4d4", font=("Consolas", 12))
        style.configure("Header.TLabel", foreground="#569cd6", font=("Consolas", 14, "bold"))
        style.configure("Value.TLabel", foreground="#4ec9b0", font=("Consolas", 12, "bold"))

        self.status_var = tk.StringVar(value="[+] Hardware Status: DISCONNECTED (Simulation Mode)")
        self.ktype_var = tk.StringVar(value="--.-- C")
        self.live_pt100_var = tk.StringVar(value="--.-- C")
        self.synthetic_ktype_var = tk.StringVar(value="--.-- C")
        self.synthetic_pt100_var = tk.StringVar(value="--.-- C")
        self.corrected_var = tk.StringVar(value="--.-- C")
        self.error_var = tk.StringVar(value="--.-- C")
        self.ktype_status_var = tk.StringVar(value="--")
        self.pt100_status_var = tk.StringVar(value="--")

        self.build_ui()

        self.running = True
        self.data_thread = threading.Thread(target=self.read_data, daemon=True)
        self.data_thread.start()

    def build_ui(self):
        ttk.Label(self.root, text="======================================================", font=("Consolas", 12)).pack(pady=(10, 0))
        ttk.Label(self.root, text="  MAX6675 / MAX31865 SENSOR MANAGER", style="Header.TLabel").pack(anchor="w", padx=10)
        ttk.Label(self.root, text="======================================================", font=("Consolas", 12)).pack()

        ttk.Label(self.root, textvariable=self.status_var, foreground="#ce9178").pack(anchor="w", padx=15, pady=(5, 10))

        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(fill="both", expand=True, padx=15)

        ttk.Label(tree_frame, text="[-] Live Sensor Inputs").grid(row=0, column=0, sticky="w", pady=(5, 0))
        ttk.Label(tree_frame, text="  |-- K-Type Thermocouple     : ").grid(row=1, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.ktype_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(tree_frame, text="  |-- PT100 RTD               : ").grid(row=2, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.live_pt100_var, style="Value.TLabel").grid(row=2, column=1, sticky="w")
        ttk.Label(tree_frame, text="  |-- MAX6675 Sensor Status   : ").grid(row=3, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.ktype_status_var, style="Value.TLabel").grid(row=3, column=1, sticky="w")
        ttk.Label(tree_frame, text="  |-- MAX31865 Sensor Status  : ").grid(row=4, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.pt100_status_var, style="Value.TLabel").grid(row=4, column=1, sticky="w")

        ttk.Label(tree_frame, text="\n[-] Simulated / Model Values").grid(row=5, column=0, sticky="w")
        ttk.Label(tree_frame, text="  |-- Synthetic PT100 Ref     : ").grid(row=6, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.synthetic_pt100_var, style="Value.TLabel").grid(row=6, column=1, sticky="w")
        ttk.Label(tree_frame, text="  |-- Synthetic K-Type Input  : ").grid(row=7, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.synthetic_ktype_var, style="Value.TLabel").grid(row=7, column=1, sticky="w")
        ttk.Label(tree_frame, text="  |-- Correction Engine Output: ").grid(row=8, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.corrected_var, style="Value.TLabel", foreground="#c586c0").grid(row=8, column=1, sticky="w")
        ttk.Label(tree_frame, text="  |-- Dynamic Error           : ").grid(row=9, column=0, sticky="w")
        ttk.Label(tree_frame, textvariable=self.error_var, style="Value.TLabel").grid(row=9, column=1, sticky="w")

        ttk.Label(self.root, text="======================================================", font=("Consolas", 12)).pack(side="bottom", pady=10)

    def update_ui_safe(self, status, live_pt100, synthetic_pt100, ktype, synthetic_ktype,
                       corrected, error, ktype_status, pt100_status):
        if status:
            self.status_var.set(status)
        if live_pt100:
            self.live_pt100_var.set(live_pt100)
        if synthetic_pt100:
            self.synthetic_pt100_var.set(synthetic_pt100)
        if ktype:
            self.ktype_var.set(ktype)
        if synthetic_ktype:
            self.synthetic_ktype_var.set(synthetic_ktype)
        if corrected:
            self.corrected_var.set(corrected)
        if error:
            self.error_var.set(error)
        if ktype_status is not None:
            self.ktype_status_var.set(ktype_status)
        if pt100_status is not None:
            self.pt100_status_var.set(pt100_status)

    def read_data(self):
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            self.root.after(
                0, self.update_ui_safe,
                f"[+] Hardware Status: CONNECTED ({SERIAL_PORT})",
                None, None, None, None, None, None, None, None
            )
            last_data_time = time.time()

            while self.running:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    parts = line.split(',')
                    if len(parts) == 8:
                        try:
                            live_k_val = f"{float(parts[1]):.2f} C"
                            synthetic_k_val = f"{float(parts[2]):.2f} C"
                            live_pt_val = f"{float(parts[3]):.2f} C"
                            synthetic_pt_val = f"{float(parts[4]):.2f} C"
                            corr_val = f"{float(parts[5]):.2f} C"
                            err_val = f"{(float(parts[5]) - float(parts[4])):+.2f} C"
                            ktype_status = "OK" if parts[6].strip() == "1" else "FALLBACK"
                            pt100_status = "OK" if parts[7].strip() == "1" else "FALLBACK"

                            self.root.after(
                                0, self.update_ui_safe,
                                None, live_pt_val, synthetic_pt_val, live_k_val,
                                synthetic_k_val, corr_val, err_val,
                                ktype_status, pt100_status
                            )
                            last_data_time = time.time()
                        except ValueError:
                            pass

                if time.time() - last_data_time > 5:
                    raise RuntimeError("No serial data received, switching to simulation mode")

                time.sleep(0.1)
        except (serial.SerialException, OSError, RuntimeError):
            self.root.after(
                0, self.update_ui_safe,
                "[+] Hardware Status: DISCONNECTED (Simulation Mode)",
                None, None, None, None, None, None, "N/A", "N/A"
            )
            base_temp = 25.0

            while self.running:
                base_temp += random.uniform(-0.5, 0.5)
                live_pt100_temp = base_temp + random.uniform(-0.3, 0.3)
                k_temp = base_temp + random.uniform(-1.0, 1.0)
                synthetic_k_temp = base_temp + random.uniform(-0.8, 0.8)
                lstm_temp = base_temp + random.uniform(-0.1, 0.1)

                live_pt_val = f"{live_pt100_temp:.2f} C"
                synthetic_pt_val = f"{base_temp:.2f} C"
                k_val = f"{k_temp:.2f} C"
                synthetic_k_val = f"{synthetic_k_temp:.2f} C"
                corr_val = f"{lstm_temp:.2f} C"
                err_val = f"{lstm_temp - base_temp:+.2f} C"

                self.root.after(
                    0, self.update_ui_safe,
                    None, live_pt_val, synthetic_pt_val, k_val, synthetic_k_val,
                    corr_val, err_val, "SIM", "SIM"
                )
                time.sleep(0.5)

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SensorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
