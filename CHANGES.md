# CHANGES.md — gui_multimodel_v2.py: Two-Page Layout + Live Error Stats

## Scope

This changes **only** `gui_multimodel_v2.py`. Do not touch any other file
(training script, firmware, headers, `dataset.csv`, or any v1 file). This is
a UI restructuring of an existing v2 file, not a new deliverable — edit it
in place.

## Goal

Split the GUI into two pages, reachable via a minimal nav bar at the top:

- **Page 1 ("Live")** — the primary/default view. Adds 3 things to what
  already exists: the live PT100 reference reading, sensor fault status, and
  two new per-model table columns (`Error w.r.t PT100`, `Rolling RMSE`).
- **Page 2 ("Details")** — everything else suggested previously (residual
  plot, best-model highlight, cumulative session stats, adjustable rolling
  window size, pause button). Reached by clicking a nav button; not shown
  by default.

Keep both pages visually minimal — no extra decoration, no new colors beyond
what the file already uses (`#1e1e1e` background, existing accent colors).

---

## 1. Navigation bar (new, top of window, above everything else)

- A single-row `ttk.Frame` at the very top of `self.root`, above the existing
  header label.
- Two `ttk.Button` widgets: `"Live"` and `"Details"`.
- Clicking `"Live"` shows `self.page1_frame` and hides `self.page2_frame`.
  Clicking `"Details"` does the reverse. Implement with
  `frame.tkraise()` on two frames stacked in the same grid cell (both
  `page1_frame` and `page2_frame` placed via `.grid(row=0, column=0, sticky="nsew")`
  inside a shared `self.pages_container` frame), OR with `.pack_forget()` /
  `.pack()` — either approach is fine as long as switching is instant and
  doesn't recreate widgets (don't destroy/rebuild frames on every click,
  since that would lose the plot's matplotlib figure state).
- Default page on launch: Page 1 ("Live").
- Style the currently-active nav button distinctly (e.g. `relief="sunken"`
  vs `relief="raised"`) so it's clear which page is showing. No other visual
  flourish needed.

---

## 2. Page 1 ("Live") — the 3 additions

Everything that currently exists on the single page (header, status line,
main plot, readout table) moves into `self.page1_frame`, unchanged in
position relative to each other, EXCEPT for the 3 additions below inserted
in the order given.

### 2a. Live PT100 reference reading (new — currently missing entirely)

Insert directly below the existing `self.status_var` label (the
"Hardware Status: ..." line) and above the plot:

```python
self.pt100_display_var = tk.StringVar(value="Live PT100 Reference: --.-- C")
ttk.Label(self.page1_frame, textvariable=self.pt100_display_var, style="Header.TLabel").pack(anchor="w", padx=15, pady=(0, 5))
```

Update `self.pt100_display_var` every time a new data row arrives (in
`update_plot`'s data-handling branch), using the same `pt100` value already
being appended to `self.pt100_buf`:

```python
self.pt100_display_var.set(f"Live PT100 Reference: {pt100:.2f} C")
```

### 2b. Sensor status indicators (new — currently missing)

Directly below the PT100 reference label, add a row of two status labels,
one for each sensor, reading `K_Sensor_OK` and `PT100_Sensor_OK` from the
incoming CSV row (both are already columns in `CSV_COLUMNS`, just not
displayed anywhere currently):

```python
sensor_status_frame = ttk.Frame(self.page1_frame)
sensor_status_frame.pack(anchor="w", padx=15, pady=(0, 10))

self.k_status_var = tk.StringVar(value="K-Type: --")
self.pt100_status_var = tk.StringVar(value="PT100: --")

self.k_status_label = ttk.Label(sensor_status_frame, textvariable=self.k_status_var)
self.k_status_label.pack(side="left", padx=(0, 20))
self.pt100_status_label = ttk.Label(sensor_status_frame, textvariable=self.pt100_status_var)
self.pt100_status_label.pack(side="left")
```

On every data row, update both. Text and color both change based on the
`*_Sensor_OK` flag (`'1'` -> OK, anything else -> FAULT):

```python
k_ok = row.get('K_Sensor_OK', '0') == '1'
pt_ok = row.get('PT100_Sensor_OK', '0') == '1'

self.k_status_var.set(f"K-Type: {'OK' if k_ok else 'FAULT'}")
self.k_status_label.configure(foreground="#4ec9b0" if k_ok else "#f14c4c")

self.pt100_status_var.set(f"PT100: {'OK' if pt_ok else 'FAULT'}")
self.pt100_status_label.configure(foreground="#4ec9b0" if pt_ok else "#f14c4c")
```

(`#4ec9b0` matches the existing teal "Value.TLabel" color already used
elsewhere in the file; `#f14c4c` is a new red, used only for the FAULT state.)

### 2c. Two new readout table columns: "Error w.r.t PT100" and "Rolling RMSE"

The existing readout table has columns `Model | Corrected Temp | Inference Time`.
Change it to 5 columns, in this exact order:

```
Model | Corrected Temp | Error w.r.t PT100 | Rolling RMSE | Inference Time
```

Insert the two new columns between the existing "Corrected Temp" and
"Inference Time" columns (do not append them at the end — the order above
is what the user asked for).

**New state needed** (add to `__init__`, alongside the existing
`self.series_bufs` / `self.val_vars` / `self.us_vars`):

```python
ROLLING_WINDOW_DEFAULT = 50  # samples; ~25s at 500ms/sample. Adjustable on Page 2.
self.rolling_window_size = ROLLING_WINDOW_DEFAULT

self.error_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
self.rmse_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}

# Rolling buffers of squared error, one per model, maxlen = current rolling window.
self.rolling_sq_err_bufs = {key: deque(maxlen=self.rolling_window_size) for key, _, _ in SERIES}
```

**Update logic**, in the same per-row loop in `update_plot` where
`self.val_vars[key]` is currently set (right after `val = float(row[key])`
is computed), add:

```python
if not math.isnan(val):
    error = val - pt100
    self.error_vars[key].set(f"{error:+.2f} C")

    self.rolling_sq_err_bufs[key].append(error ** 2)
    if len(self.rolling_sq_err_bufs[key]) >= 5:  # require a minimum sample count before showing a number
        rolling_rmse = math.sqrt(sum(self.rolling_sq_err_bufs[key]) / len(self.rolling_sq_err_bufs[key]))
        self.rmse_vars[key].set(f"{rolling_rmse:.2f} C")
    else:
        self.rmse_vars[key].set("warming up...")
else:
    self.error_vars[key].set("-- C")
    self.rmse_vars[key].set("-- C")
```

(The `>= 5` minimum-sample guard avoids showing a misleadingly precise RMSE
from only 1-2 points right after the GUI starts or after a window-size change.)

**Table widget changes** — update the header row and the per-model row
loop to include the 2 new columns in the correct position:

```python
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
```

---

## 3. Page 2 ("Details") — everything else

All of the following live in `self.page2_frame`, stacked vertically, in
this order. Keep it minimal — plain labels/widgets, no extra framing beyond
what's needed for layout.

### 3a. Residual (error) subplot

A second matplotlib figure (separate from Page 1's main plot — do not try
to reuse or share the same `Figure`/`Axes` objects across pages), embedded
the same way Page 1's plot is (`FigureCanvasTkAgg`), showing each model's
error (`pred - PT100`) over the same sample index, with a horizontal line
at zero for reference.

```python
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
```

Maintain a parallel `self.error_buf = {key: deque(maxlen=MAXLEN) for key, _, _ in SERIES}`
(full-precision error values, not squared — separate from the rolling RMSE
buffer used on Page 1, which only stores squared error and uses a
different, adjustable maxlen). Append to it in the same place `error` is
computed in section 2c above. In `update_plot`, after updating Page 1's
lines, also update:

```python
for key, _, _ in SERIES:
    self.error_lines[key].set_data(self.t_buf, self.error_buf[key])
self.ax2.relim()
self.ax2.autoscale_view()
self.canvas2.draw_idle()
```

Only redraw Page 2's canvas if Page 2 is currently the visible page (check
a `self.current_page` variable, set by the nav button handlers) — no need
to waste CPU redrawing a hidden matplotlib figure every 200ms.

### 3b. Best-model highlight

A single label above or below the residual plot:

```python
self.best_model_var = tk.StringVar(value="Best model (lowest rolling RMSE): --")
ttk.Label(self.page2_frame, textvariable=self.best_model_var, style="Header.TLabel").pack(anchor="w", padx=15, pady=5)
```

Computed each update cycle: among models whose rolling RMSE buffer has
reached the minimum sample count (the same `>= 5` threshold from 2c),
find the one with lowest `sqrt(mean(rolling_sq_err_bufs[key]))` and set
the label to its display name. If none qualify yet, leave the default text.

### 3c. Cumulative session stats table

A second small table, same widget pattern as Page 1's readout table, with
columns `Model | Session MAE | Session RMSE | Max Abs Error`. These are
**not** windowed — they accumulate from the moment the GUI started (or was
last reset) to now.

**New state** (in `__init__`):

```python
self.cumulative_stats = {
    key: {'sum_abs_err': 0.0, 'sum_sq_err': 0.0, 'count': 0, 'max_abs_err': 0.0}
    for key, _, _ in SERIES
}
self.session_mae_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
self.session_rmse_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
self.session_max_vars = {key: tk.StringVar(value="--.-- C") for key, _, _ in SERIES}
```

Update alongside the rolling-error computation in section 2c (same `error`
value, just also fed into the cumulative accumulators instead of only the
windowed deque):

```python
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
```

Add a small "Reset session stats" `ttk.Button` next to this table's title
that zeroes out `self.cumulative_stats` for all models back to the
`{'sum_abs_err': 0.0, ...}` initial state.

### 3d. Adjustable rolling window size

A small control (label + `ttk.Spinbox` or `+`/`-` buttons) to change
`self.rolling_window_size` (used by Page 1's rolling RMSE, section 2c):

```python
window_frame = ttk.Frame(self.page2_frame)
window_frame.pack(anchor="w", padx=15, pady=10)
ttk.Label(window_frame, text="Rolling RMSE window (samples): ").pack(side="left")
self.window_size_var = tk.IntVar(value=ROLLING_WINDOW_DEFAULT)
window_spinbox = ttk.Spinbox(window_frame, from_=5, to=500, increment=5,
                              textvariable=self.window_size_var, width=6,
                              command=self.on_window_size_change)
window_spinbox.pack(side="left", padx=5)
```

```python
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
```

### 3e. Pause/freeze plot button

A `ttk.Button` labeled `"Pause"` / `"Resume"` (toggles its own label) that
stops Page 1's main plot and Page 2's residual plot from redrawing, while
data continues to be read and accumulated in the background (so rolling
RMSE, session stats, and buffers keep updating — only the plot rendering
freezes, so the user can screenshot the current view without it scrolling
away).

```python
self.paused = False

def toggle_pause(self):
    self.paused = not self.paused
    self.pause_button.configure(text="Resume" if self.paused else "Pause")
```

In `update_plot`, wrap only the plotting/`draw_idle()` calls (both Page 1's
and Page 2's) in `if not self.paused:` — keep all the data-append and
StringVar-update logic (Page 1's table values, Page 2's cumulative stats)
running regardless of pause state, since those are cheap and the user
likely still wants the numeric readouts moving even while the plot is frozen.

Place the pause button on Page 2 near the other controls (e.g. next to the
rolling-window spinbox), not on Page 1 — keep Page 1 strictly to the 3
items specified in section 2.

---

## Definition of done

- [ ] Nav bar with "Live"/"Details" buttons switches pages instantly, no widget recreation
- [ ] Page 1 default on launch
- [ ] Page 1 shows live PT100 reference reading, updating every row
- [ ] Page 1 shows K-Type and PT100 sensor status, color-coded OK (teal) / FAULT (red)
- [ ] Page 1's readout table has exactly 5 columns in this order: Model, Corrected Temp, Error w.r.t PT100, Rolling RMSE, Inference Time
- [ ] Rolling RMSE shows "warming up..." until at least 5 samples are in its window, not a misleading early number
- [ ] Page 2 has: residual/error subplot with zero-line, best-model-by-rolling-RMSE label, cumulative session stats table with a reset button, adjustable rolling window spinbox, pause/resume button
- [ ] Page 2's plot only redraws while Page 2 is the visible page (no wasted redraw work while on Page 1)
- [ ] Pausing freezes only the plot rendering, not the underlying data/stat updates
- [ ] Simulation-mode fallback (disconnect after 5s) still populates all of the above identically to live serial data — the new UI elements must work in simulation mode too, since that's the primary way this will be visually verified before hardware is connected
- [ ] No changes made to any file other than `gui_multimodel_v2.py`
