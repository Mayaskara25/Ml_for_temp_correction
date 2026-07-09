"""
gui_multimodel_v2.py
----------------------
Tkinter + embedded matplotlib GUI for ESP32_multimodel_v2's serial CSV output.

Two pages, reachable via a nav bar:
  - "Live": PT100 reference, sensor status, main plot, and a 5-column readout
    table (value / error vs PT100 / rolling RMSE / inference time per model).
  - "Details": residual (error) plot, best-model-by-rolling-RMSE label,
    cumulative session stats table, adjustable rolling-window size, and a
    pause/resume control for both plots.

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
ROLLING_WINDOW_DEFAULT = 50  # samples; ~25s at 500ms/sample. Adjustable on the Details page.

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
        self.root.geometry("1000x860")
        self.root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#d4d4d4", font=("Consolas", 11))
        style.configure("Header.TLabel", foreground="#569cd6", font=("Consolas", 14, "bold"))
        style.configure("Value.TLabel", foreground="#4ec9b0", font=("Consolas", 11, "bold"))
        style.configure("Nav.TButton", relief="raised")
        style.configure("NavActive.TButton", relief="sunken")

        self.status_var = tk.StringVar(value="[+] Hardware Status: DISCONNECTED (Simulation Mode)")
        self.pt100_display_var = tk.StringVar(value="Live PT100 Reference: --.-- C")
        self.k_status_var = tk.StringVar(value="K-Type: --")
        self.pt100_status_var = tk.StringVar(value="PT100: --")
        self.best_model_var = tk.StringVar(value="Best model (lowest rolling RMSE): --")

        self.sample_counter = 0
        self.t_buf = deque(maxlen=MAXLEN)
        self.pt100_buf = deque(maxlen=MAXLEN)
        self.series_bufs = {key: deque(maxlen=MAXLEN) for key, _, _ in SERIES}
        self.error_buf = {key: deque(maxlen=MAXLEN) for key, _, _ in SERIES}
        self.val_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
        self.us_vars = {key: tk.StringVar(value="-- us") for key, _, _ in SERIES}
        self.error_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
        self.rmse_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}

        self.rolling_window_size = ROLLING_WINDOW_DEFAULT
        self.rolling_sq_err_bufs = {key: deque(maxlen=self.rolling_window_size) for key, _, _ in SERIES}

        self.cumulative_stats = {
            key: {'sum_abs_err': 0.0, 'sum_sq_err': 0.0, 'count': 0, 'max_abs_err': 0.0}
            for key, _, _ in SERIES
        }
        self.session_mae_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
        self.session_rmse_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
        self.session_max_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}

        self.current_page = 'live'
        self.paused = False

        self.row_queue = queue.Queue()
        self.running = True

        self.build_ui()

        self.data_thread = threading.Thread(target=self.read_data, daemon=True)
        self.data_thread.start()

        self.root.after(200, self.update_plot)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_ui(self):
        self.build_nav_bar()

        self.pages_container = ttk.Frame(self.root)
        self.pages_container.pack(fill="both", expand=True)
        self.pages_container.grid_rowconfigure(0, weight=1)
        self.pages_container.grid_columnconfigure(0, weight=1)

        self.page1_frame = ttk.Frame(self.pages_container)
        self.page2_frame = ttk.Frame(self.pages_container)
        self.page1_frame.grid(row=0, column=0, sticky="nsew")
        self.page2_frame.grid(row=0, column=0, sticky="nsew")

        self.build_page1()
        self.build_page2()

        self.show_page('live')

    def build_nav_bar(self):
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill="x", padx=10, pady=(10, 0))

        self.nav_live_button = ttk.Button(nav_frame, text="Live", style="Nav.TButton", command=lambda: self.show_page('live'))
        self.nav_live_button.pack(side="left", padx=(0, 5))
        self.nav_details_button = ttk.Button(nav_frame, text="Details", style="Nav.TButton", command=lambda: self.show_page('details'))
        self.nav_details_button.pack(side="left")

    def show_page(self, page):
        self.current_page = page
        if page == 'live':
            self.page1_frame.tkraise()
        else:
            self.page2_frame.tkraise()
        self.nav_live_button.configure(style=("NavActive.TButton" if page == 'live' else "Nav.TButton"))
        self.nav_details_button.configure(style=("NavActive.TButton" if page == 'details' else "Nav.TButton"))

    def build_page1(self):
        ttk.Label(self.page1_frame, text="MULTI-MODEL SENSOR CORRECTION COMPARISON (v2)", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
        ttk.Label(self.page1_frame, textvariable=self.status_var, foreground="#ce9178").pack(anchor="w", padx=15, pady=(5, 10))

        ttk.Label(self.page1_frame, textvariable=self.pt100_display_var, style="Header.TLabel").pack(anchor="w", padx=15, pady=(0, 5))

        sensor_status_frame = ttk.Frame(self.page1_frame)
        sensor_status_frame.pack(anchor="w", padx=15, pady=(0, 10))
        self.k_status_label = ttk.Label(sensor_status_frame, textvariable=self.k_status_var)
        self.k_status_label.pack(side="left", padx=(0, 20))
        self.pt100_status_label = ttk.Label(sensor_status_frame, textvariable=self.pt100_status_var)
        self.pt100_status_label.pack(side="left")

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

        canvas_frame = ttk.Frame(self.page1_frame)
        canvas_frame.pack(fill="both", expand=True, padx=15)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        table_frame = ttk.Frame(self.page1_frame)
        table_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(table_frame, text="Model", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(table_frame, text="Corrected Temp", style="Header.TLabel").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(table_frame, text="Error w.r.t PT100", style="Header.TLabel").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Label(table_frame, text="Rolling RMSE", style="Header.TLabel").grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(table_frame, text="Inference Time", style="Header.TLabel").grid(row=0, column=4, sticky="w", padx=5)

        for i, (key, label, color) in enumerate(SERIES):
            ttk.Label(table_frame, text=label).grid(row=i + 1, column=0, sticky="w", padx=5)
            ttk.Label(table_frame, textvariable=self.val_vars[key], style="Value.TLabel").grid(row=i + 1, column=1, sticky="w", padx=5)
            ttk.Label(table_frame, textvariable=self.error_vars[key], style="Value.TLabel").grid(row=i + 1, column=2, sticky="w", padx=5)
            ttk.Label(table_frame, textvariable=self.rmse_vars[key], style="Value.TLabel").grid(row=i + 1, column=3, sticky="w", padx=5)
            ttk.Label(table_frame, textvariable=self.us_vars[key], style="Value.TLabel").grid(row=i + 1, column=4, sticky="w", padx=5)

    def build_page2(self):
        self.fig2, self.ax2 = plt.subplots(figsize=(9, 4))
        self.fig2.patch.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        self.ax2.set_xlabel('Sample')
        self.ax2.set_ylabel('Error (C)')
        self.ax2.set_title('Per-model error (prediction - PT100)')
        self.ax2.axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)

        self.error_lines = {}
        for key, label, color in SERIES:
            line, = self.ax2.plot([], [], color=color, linewidth=1.0, label=label)
            self.error_lines[key] = line
        self.ax2.legend(loc='upper right', fontsize=8)

        canvas2_frame = ttk.Frame(self.page2_frame)
        canvas2_frame.pack(fill="both", expand=True, padx=15, pady=(15, 5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=canvas2_frame)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

        ttk.Label(self.page2_frame, textvariable=self.best_model_var, style="Header.TLabel").pack(anchor="w", padx=15, pady=5)

        stats_header_frame = ttk.Frame(self.page2_frame)
        stats_header_frame.pack(fill="x", padx=15, pady=(10, 0))
        ttk.Label(stats_header_frame, text="Cumulative Session Stats", style="Header.TLabel").pack(side="left")
        ttk.Button(stats_header_frame, text="Reset session stats", command=self.reset_session_stats).pack(side="left", padx=10)

        stats_table_frame = ttk.Frame(self.page2_frame)
        stats_table_frame.pack(fill="x", padx=15, pady=5)

        ttk.Label(stats_table_frame, text="Model", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(stats_table_frame, text="Session MAE", style="Header.TLabel").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(stats_table_frame, text="Session RMSE", style="Header.TLabel").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Label(stats_table_frame, text="Max Abs Error", style="Header.TLabel").grid(row=0, column=3, sticky="w", padx=5)

        for i, (key, label, color) in enumerate(SERIES):
            ttk.Label(stats_table_frame, text=label).grid(row=i + 1, column=0, sticky="w", padx=5)
            ttk.Label(stats_table_frame, textvariable=self.session_mae_vars[key], style="Value.TLabel").grid(row=i + 1, column=1, sticky="w", padx=5)
            ttk.Label(stats_table_frame, textvariable=self.session_rmse_vars[key], style="Value.TLabel").grid(row=i + 1, column=2, sticky="w", padx=5)
            ttk.Label(stats_table_frame, textvariable=self.session_max_vars[key], style="Value.TLabel").grid(row=i + 1, column=3, sticky="w", padx=5)

        controls_frame = ttk.Frame(self.page2_frame)
        controls_frame.pack(anchor="w", padx=15, pady=10)

        ttk.Label(controls_frame, text="Rolling RMSE window (samples): ").pack(side="left")
        self.window_size_var = tk.IntVar(value=ROLLING_WINDOW_DEFAULT)
        window_spinbox = ttk.Spinbox(controls_frame, from_=5, to=500, increment=5,
                                      textvariable=self.window_size_var, width=6,
                                      command=self.on_window_size_change)
        window_spinbox.pack(side="left", padx=5)

        self.pause_button = ttk.Button(controls_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=(20, 0))

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def on_window_size_change(self):
        new_size = self.window_size_var.get()
        self.rolling_window_size = new_size
        # Recreate each model's rolling buffer with the new maxlen, preserving
        # as much recent history as fits (deque maxlen cannot be changed in
        # place). This intentionally resets the "warming up" state if the new
        # window is larger than the data currently held.
        for key, _, _ in SERIES:
            old_buf = self.rolling_sq_err_bufs[key]
            self.rolling_sq_err_bufs[key] = deque(old_buf, maxlen=new_size)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")

    def reset_session_stats(self):
        for key, _, _ in SERIES:
            self.cumulative_stats[key] = {'sum_abs_err': 0.0, 'sum_sq_err': 0.0, 'count': 0, 'max_abs_err': 0.0}
            self.session_mae_vars[key].set("--.-- C")
            self.session_rmse_vars[key].set("--.-- C")
            self.session_max_vars[key].set("--.-- C")

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Redraw / update loop
    # ------------------------------------------------------------------
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

                self.pt100_display_var.set(f"Live PT100 Reference: {pt100:.2f} C")

                k_ok = row.get('K_Sensor_OK', '0') == '1'
                pt_ok = row.get('PT100_Sensor_OK', '0') == '1'
                self.k_status_var.set(f"K-Type: {'OK' if k_ok else 'FAULT'}")
                self.k_status_label.configure(foreground="#4ec9b0" if k_ok else "#f14c4c")
                self.pt100_status_var.set(f"PT100: {'OK' if pt_ok else 'FAULT'}")
                self.pt100_status_label.configure(foreground="#4ec9b0" if pt_ok else "#f14c4c")

                for key, _, _ in SERIES:
                    try:
                        val = float(row[key])
                    except (ValueError, KeyError):
                        val = float('nan')
                    self.series_bufs[key].append(val)
                    self.val_vars[key].set(f"{val:.2f} C")
                    us_key = key.replace('_C', '_us')
                    self.us_vars[key].set(f"{row.get(us_key, '--')} us")

                    if not math.isnan(val):
                        error = val - pt100
                        self.error_vars[key].set(f"{error:+.2f} C")
                        self.error_buf[key].append(error)

                        self.rolling_sq_err_bufs[key].append(error ** 2)
                        if len(self.rolling_sq_err_bufs[key]) >= 5:  # minimum sample count before showing a number
                            rolling_rmse = math.sqrt(sum(self.rolling_sq_err_bufs[key]) / len(self.rolling_sq_err_bufs[key]))
                            self.rmse_vars[key].set(f"{rolling_rmse:.2f} C")
                        else:
                            self.rmse_vars[key].set("warming up...")

                        stats = self.cumulative_stats[key]
                        stats['sum_abs_err'] += abs(error)
                        stats['sum_sq_err'] += error ** 2
                        stats['count'] += 1
                        stats['max_abs_err'] = max(stats['max_abs_err'], abs(error))

                        mae = stats['sum_abs_err'] / stats['count']
                        rmse = math.sqrt(stats['sum_sq_err'] / stats['count'])
                        self.session_mae_vars[key].set(f"{mae:.2f} C")
                        self.session_rmse_vars[key].set(f"{rmse:.2f} C")
                        self.session_max_vars[key].set(f"{stats['max_abs_err']:.2f} C")
                    else:
                        self.error_vars[key].set("-- C")
                        self.rmse_vars[key].set("-- C")
                        self.error_buf[key].append(float('nan'))
                redraw = True

        if redraw:
            best_key = None
            best_rmse = None
            for key, _, _ in SERIES:
                if len(self.rolling_sq_err_bufs[key]) >= 5:
                    rmse = math.sqrt(sum(self.rolling_sq_err_bufs[key]) / len(self.rolling_sq_err_bufs[key]))
                    if best_rmse is None or rmse < best_rmse:
                        best_rmse = rmse
                        best_key = key
            if best_key is not None:
                best_label = next(lbl for k, lbl, _ in SERIES if k == best_key)
                self.best_model_var.set(f"Best model (lowest rolling RMSE): {best_label} ({best_rmse:.2f} C)")

            if not self.paused:
                self.pt100_line.set_data(self.t_buf, self.pt100_buf)
                for key, _, _ in SERIES:
                    self.model_lines[key].set_data(self.t_buf, self.series_bufs[key])
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw_idle()

                if self.current_page == 'details':
                    for key, _, _ in SERIES:
                        self.error_lines[key].set_data(self.t_buf, self.error_buf[key])
                    self.ax2.relim()
                    self.ax2.autoscale_view()
                    self.canvas2.draw_idle()

        self.root.after(200, self.update_plot)

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiModelGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
