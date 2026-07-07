"""
gui_multimodel_v2.py
----------------------
Tkinter + embedded matplotlib GUI for ESP32_multimodel_v2's serial CSV output.
Plots the PT100 reference against all 5 corrected outputs (Dense_v2,
TCN_Hadamard_v2, RandomForest, Kalman, Hybrid_Physics_v2) and shows a live
readout table of each model's current value and inference time.

Same disconnect -> simulation-mode fallback UX convention as v1's gui_max6675.py.
"""

import tkinter as tk
from tkinter import ttk
import serial
import threading
import time
import random
import math
import queue
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Configuration ---
SERIAL_PORT = 'COM8'  # change to match your ESP32's COM port
BAUD_RATE = 115200
DISCONNECT_TIMEOUT_SEC = 5

MAXLEN = 200

# Exact column order emitted by ESP32_multimodel_v2.ino's serial CSV header
CSV_COLUMNS = [
    'Timestamp_ms', 'Live_K_Temp_C', 'Live_PT100_Temp_C', 'K_Sensor_OK', 'PT100_Sensor_OK',
    'Dense_v2_C', 'Dense_v2_us', 'TCN_Hadamard_C', 'TCN_Hadamard_us',
    'RF_C', 'RF_us', 'Kalman_C', 'Kalman_us', 'Hybrid_C', 'Hybrid_us',
]

# (csv_column, display_label, line_color)
SERIES = [
    ('Dense_v2_C', 'Dense_v2', '#569cd6'),
    ('TCN_Hadamard_C', 'TCN_Hadamard', '#c586c0'),
    ('RF_C', 'RandomForest', '#dcdcaa'),
    ('Kalman_C', 'Kalman', '#4ec9b0'),
    ('Hybrid_C', 'Hybrid', '#ce9178'),
]


class MultiModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model Sensor Correction Comparison (v2)")
        self.root.geometry("1000x780")
        self.root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#d4d4d4", font=("Consolas", 11))
        style.configure("Header.TLabel", foreground="#569cd6", font=("Consolas", 14, "bold"))
        style.configure("Value.TLabel", foreground="#4ec9b0", font=("Consolas", 11, "bold"))

        self.status_var = tk.StringVar(value="[+] Hardware Status: DISCONNECTED (Simulation Mode)")

        self.sample_counter = 0
        self.t_buf = deque(maxlen=MAXLEN)
        self.pt100_buf = deque(maxlen=MAXLEN)
        self.series_bufs = {key: deque(maxlen=MAXLEN) for key, _, _ in SERIES}
        self.val_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
        self.us_vars = {key: tk.StringVar(value="-- us") for key, _, _ in SERIES}

        self.row_queue = queue.Queue()
        self.running = True

        self.build_ui()

        self.data_thread = threading.Thread(target=self.read_data, daemon=True)
        self.data_thread.start()

        self.root.after(200, self.update_plot)

    def build_ui(self):
        ttk.Label(self.root, text="MULTI-MODEL SENSOR CORRECTION COMPARISON (v2)", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
        ttk.Label(self.root, textvariable=self.status_var, foreground="#ce9178").pack(anchor="w", padx=15, pady=(5, 10))

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Temperature (C)')
        self.ax.set_title('PT100 reference vs corrected outputs')

        self.pt100_line, = self.ax.plot([], [], color='white', linewidth=2.5, label='Live_PT100 (reference)')
        self.model_lines = {}
        for key, label, color in SERIES:
            line, = self.ax.plot([], [], color=color, linewidth=1.2, label=label)
            self.model_lines[key] = line
        self.ax.legend(loc='upper right', fontsize=8)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=15)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        table_frame = ttk.Frame(self.root)
        table_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(table_frame, text="Model", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(table_frame, text="Corrected Temp", style="Header.TLabel").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(table_frame, text="Inference Time", style="Header.TLabel").grid(row=0, column=2, sticky="w", padx=5)

        for i, (key, label, color) in enumerate(SERIES):
            ttk.Label(table_frame, text=label).grid(row=i + 1, column=0, sticky="w", padx=5)
            ttk.Label(table_frame, textvariable=self.val_vars[key], style="Value.TLabel").grid(row=i + 1, column=1, sticky="w", padx=5)
            ttk.Label(table_frame, textvariable=self.us_vars[key], style="Value.TLabel").grid(row=i + 1, column=2, sticky="w", padx=5)

    def read_data(self):
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            self.row_queue.put(('status', f"[+] Hardware Status: CONNECTED ({SERIAL_PORT})"))
            last_data_time = time.time()

            while self.running:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line or line.startswith('[') or 'Timestamp_ms' in line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= len(CSV_COLUMNS):
                        row = {col: parts[i] for i, col in enumerate(CSV_COLUMNS)}
                        self.row_queue.put(('data', row))
                        last_data_time = time.time()

                if time.time() - last_data_time > DISCONNECT_TIMEOUT_SEC:
                    raise RuntimeError("No serial data received, switching to simulation mode")

                time.sleep(0.05)
        except (serial.SerialException, OSError, RuntimeError):
            self.row_queue.put(('status', "[+] Hardware Status: DISCONNECTED (Simulation Mode)"))
            t = 0.0
            while self.running:
                t += 0.5
                base_temp = 25.0 + 10.0 * (0.5 + 0.5 * math.sin(t / 20.0))
                pt100 = base_temp + random.uniform(-0.3, 0.3)
                row = {
                    'Timestamp_ms': str(int(t * 1000)),
                    'Live_K_Temp_C': f"{pt100 + random.uniform(-1.0, 1.0):.2f}",
                    'Live_PT100_Temp_C': f"{pt100:.2f}",
                    'K_Sensor_OK': '0',
                    'PT100_Sensor_OK': '0',
                    'Dense_v2_C': f"{pt100 + random.uniform(-0.6, 0.6):.2f}",
                    'Dense_v2_us': str(random.randint(200, 500)),
                    'TCN_Hadamard_C': f"{pt100 + random.uniform(-0.8, 0.8):.2f}",
                    'TCN_Hadamard_us': str(random.randint(800, 1500)),
                    'RF_C': f"{pt100 + random.uniform(-0.7, 0.7):.2f}",
                    'RF_us': str(random.randint(50, 150)),
                    'Kalman_C': f"{pt100 + random.uniform(-2.3, 2.3):.2f}",
                    'Kalman_us': str(random.randint(5, 20)),
                    'Hybrid_C': f"{pt100 + random.uniform(-0.4, 0.4):.2f}",
                    'Hybrid_us': str(random.randint(200, 500)),
                }
                self.row_queue.put(('data', row))
                time.sleep(0.5)

    def update_plot(self):
        redraw = False
        while not self.row_queue.empty():
            kind, payload = self.row_queue.get_nowait()
            if kind == 'status':
                self.status_var.set(payload)
            elif kind == 'data':
                row = payload
                try:
                    pt100 = float(row['Live_PT100_Temp_C'])
                except (ValueError, KeyError):
                    continue

                self.sample_counter += 1
                self.t_buf.append(self.sample_counter)
                self.pt100_buf.append(pt100)

                for key, _, _ in SERIES:
                    try:
                        val = float(row[key])
                    except (ValueError, KeyError):
                        val = float('nan')
                    self.series_bufs[key].append(val)
                    self.val_vars[key].set(f"{val:.2f} C")
                    us_key = key.replace('_C', '_us')
                    self.us_vars[key].set(f"{row.get(us_key, '--')} us")
                redraw = True

        if redraw:
            self.pt100_line.set_data(self.t_buf, self.pt100_buf)
            for key, _, _ in SERIES:
                self.model_lines[key].set_data(self.t_buf, self.series_bufs[key])
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()

        self.root.after(200, self.update_plot)

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiModelGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
