idealabEL ESP32 Temperature Correction Project
==============================================

Current purpose
---------------

This project runs a small Dense TensorFlow Lite Micro model on an ESP32 and
shows live/simulated temperature values in a Python Tkinter GUI.

The current test setup is designed for hardware bring-up:

1. Read a real K-type thermocouple through a MAX6675 module.
2. Read a real 3-wire PT100 RTD through an Adafruit MAX31865 module.
3. Keep the ML model running on synthetic K-type input for now.
4. Keep a synthetic PT100/reference value for model comparison.
5. Display both real sensor readings and synthetic/model values in the GUI.

The training files and `model_data.h` are not changed by this hardware test
flow. Retraining can be done later after real data collection is ready.


Important files
---------------

Arduino sketch:

`C:\ArduinoWorkspace\ESP32_dense_temp\ESP32_dense_temp.ino`

Model header used by the Arduino sketch:

`C:\ArduinoWorkspace\ESP32_dense_temp\model_data.h`

Python GUI:

`C:\projects\idealabEL\gui_max6675.py`

Training script, not required for the current hardware test:

`C:\projects\idealabEL\train_dense_esp32.py`

Note: this script now trains on `dataset.csv` (real collected sensor data)
instead of `synthetic_training_data.csv`, and writes its output to
`model_data_real.h` rather than `model_data.h` directly -- so re-running it no
longer touches the file the firmware actually includes. Copy
`model_data_real.h` over `model_data.h` manually once you're ready to switch
the firmware to the real-data-trained model.


Hardware connections
--------------------

Use 3.3 V logic/power for the sensor breakout boards. Do not wire the PT100 or
K-type probe directly to the ESP32.

MAX6675 K-type thermocouple module:

`MAX6675 VCC` -> `ESP32 3V3`
`MAX6675 GND` -> `ESP32 GND`
`MAX6675 SCK` -> `ESP32 GPIO18`
`MAX6675 SO`  -> `ESP32 GPIO19`
`MAX6675 CS`  -> `ESP32 GPIO14`

Adafruit MAX31865 PT100 RTD module:

`MAX31865 3V3` -> `ESP32 3V3`
`MAX31865 GND` -> `ESP32 GND`
`MAX31865 CLK` -> `ESP32 GPIO18`
`MAX31865 SDO` -> `ESP32 GPIO19`
`MAX31865 SDI` -> `ESP32 GPIO23`
`MAX31865 CS`  -> `ESP32 GPIO27`
`MAX31865 RDY` -> not connected

The MAX6675 and MAX31865 share `CLK` and `SDO/MISO`, but they must use
different chip-select pins. The MAX6675 uses `GPIO14`; the MAX31865 uses
`GPIO27`.


3-wire PT100 wiring on Adafruit MAX31865
----------------------------------------

The Adafruit MAX31865 board must be configured for 3-wire RTD mode. In this
project the board was prepared using the Adafruit instructions:

1. Solder the `2/3` pad.
2. Cut the thin connection between pads `2` and `4`.
3. Solder pads `4` and `3`.

After that modification, connect the 3-wire PT100 to the terminal block:

`Two same-color PT100 wires` -> `F+` and `RTD+`
`Single remaining PT100 wire` -> `RTD-`
`F-` -> leave empty


What ESP32_dense_temp.ino does
-----------------------------

The sketch initializes:

1. `MAX6675` for the real K-type thermocouple.
2. `Adafruit_MAX31865` in `MAX31865_3WIRE` mode for the real PT100.
3. `EloquentTinyML` for running the Dense model from `model_data.h`.

Every 500 ms, the sketch:

1. Generates synthetic PT100/reference data.
2. Generates synthetic K-type data for the model input.
3. Reads the real K-type temperature from MAX6675.
4. Reads the real PT100 temperature from MAX31865.
5. Falls back to synthetic-like PT100 data if the MAX31865 reports a fault or
   an invalid reading.
6. Runs the ML model using the synthetic K-type rolling input buffer.
7. Prints one CSV line over Serial at `115200` baud.

The model currently does not use the real K-type or real PT100 readings as
input. Those real readings are displayed in the GUI so the hardware can be
tested before retraining or changing the model input flow.


Serial CSV format
-----------------

The ESP32 sends this header:

`Timestamp_ms,Live_K_Temp_C,Synthetic_K_Temp_C,Live_PT100_Temp_C,Synthetic_PT100_Temp_C,Corrected_Temp_C,K_Sensor_OK,PT100_Sensor_OK`

Each data row has 8 comma-separated fields:

1. `Timestamp_ms` - ESP32 `millis()` timestamp.
2. `Live_K_Temp_C` - real MAX6675 K-type reading.
3. `Synthetic_K_Temp_C` - synthetic K-type value fed into the ML model.
4. `Live_PT100_Temp_C` - real MAX31865 PT100 reading, or fallback value.
5. `Synthetic_PT100_Temp_C` - synthetic reference value.
6. `Corrected_Temp_C` - ML model output after unscaling.
7. `K_Sensor_OK` - `1` if MAX6675 read is valid, otherwise `0`.
8. `PT100_Sensor_OK` - `1` if MAX31865 read is valid, otherwise `0`.


What gui_max6675.py does
------------------------

`gui_max6675.py` opens the configured serial port and displays the ESP32 CSV
stream in two sections.

Live Sensor Inputs:

`K-Type Thermocouple` - real MAX6675 reading.
`PT100 RTD` - real MAX31865 reading, or fallback value if the RTD read failed.
`MAX6675 Sensor Status` - `OK`, `FALLBACK`, `SIM`, or `N/A`.
`MAX31865 Sensor Status` - `OK`, `FALLBACK`, `SIM`, or `N/A`.

Correction Engine:

`Correction Engine Output` - corrected temperature predicted by the model.
`Dynamic Error` - `Corrected_Temp_C - Live_PT100_Temp_C` (compared against the
real PT100 reading now, not a synthetic reference -- the GUI no longer
displays the synthetic values the ESP32 still sends internally).

If the ESP32 is not connected, the COM port is wrong, or valid serial rows do
not arrive for more than 5 seconds, the GUI switches to local simulation mode.


Arduino dependencies
--------------------

The Arduino sketch expects these libraries to be available:

1. ESP32 board package by Espressif.
2. EloquentTinyML `2.4.4`.
3. MAX6675 library `0.3.4`.
4. Adafruit MAX31865 library.
5. Adafruit BusIO.

The current workspace has these libraries copied under:

`C:\ArduinoWorkspace\libraries`

The sketch has been compiled for:

`ESP32 Dev Module`

FQBN used for command-line compile:

`esp32:esp32:esp32`


How to upload the ESP32 sketch
------------------------------

1. Open Arduino IDE.
2. Open:
   `C:\ArduinoWorkspace\ESP32_dense_temp\ESP32_dense_temp.ino`
3. Select board:
   `ESP32 Dev Module`
4. Select the ESP32 COM port.
5. Upload the sketch.

If upload fails with `Wrong boot mode detected` or Arduino IDE remains at
`Connecting...`, manually enter download mode:

1. Start upload.
2. Hold the ESP32 `BOOT` button when Arduino shows `Connecting...`.
3. Release `BOOT` after upload starts writing.
4. If needed, hold `BOOT`, tap `EN/RST`, then try upload again.


How to run the GUI
------------------

Set the serial port in `gui_max6675.py`:

`SERIAL_PORT = 'COM8'`

Change `COM8` if Windows assigns a different port to the ESP32.

Run the GUI from PowerShell:

`cd C:\projects\idealabEL`
`python gui_max6675.py`

The GUI requires Python packages:

`pyserial`

If needed, install it in the project environment:

`pip install pyserial`


Current behavior and limitations
--------------------------------

The real MAX6675 and MAX31865 readings are currently for display/testing.

The ESP32 firmware (`ESP32_dense_temp.ino`) itself is unchanged: the ML model
still uses synthetic K-type history as its input, and the sketch still sends
both live and synthetic values over serial. This keeps the old model test
flow stable while hardware sensors are being verified. Only the GUI display
and the training data source have moved to using real data (see the notes on
`train_dense_esp32.py` and `gui_max6675.py` above) -- the firmware's model
input flow has not been changed yet.

The PT100 fallback behavior only means the code detected a bad/faulted RTD
reading and substituted a synthetic-like value so the GUI continues updating.
If `MAX31865 Sensor Status` shows `FALLBACK`, check wiring, jumper/solder
configuration, and PT100 terminal placement.

`model_data.h` and `train_dense_esp32.py` should only be changed when you are
ready to retrain or change the model behavior.


Version 2 -- Multi-Model Comparison System
===========================================

A separate, entirely additive system that trains and compares 5 different
K-type-to-PT100 correction approaches on the real `dataset.csv`, instead of
the single synthetic-trained Dense model used above. Every v2 file ends in
`_v2` (or lives in a `_v2`-suffixed folder) and none of the v1 files
(`ESP32_dense_temp.ino`, `train_dense_esp32.py`, `gui_max6675.py`, `logger.py`,
`model_data.h`, `dataset.csv`, `synthetic_training_data.csv`) were modified to
build it. Model inputs in this path come only from the real live K-type
reading -- there is no synthetic data anywhere in v2.

The 5 approaches:

1. `Dense_v2` - 20-input Dense MLP (10-sample K-type window + its derivative).
2. `TCN_Hadamard_v2` - dilated causal Conv1D branch + a Hadamard-transform
   branch with a trainable soft-threshold.
3. `Hybrid_Physics_v2` - a fitted first-order thermal lag correction plus a
   small Dense residual net.
4. `RandomForest` - scikit-learn RF exported to C via `emlearn`.
5. `Kalman` - a hand-written constant-velocity Kalman filter.

Files:

`features_v2.py` - shared data loading / feature engineering, imported by
the two scripts below.

`train_multimodel_v2.py` - trains all 5 approaches and exports:
  - `model_data_dense_v2_float32.h`, `_ptq_int8.h`, `_qat_int8.h`
  - `model_data_tcn_hadamard_v2_float32.h`, `_ptq_int8.h` (no `_qat_int8.h` --
    see Known limitations below)
  - `model_data_hybrid_v2_float32.h`, `_ptq_int8.h`, `_qat_int8.h`
  - `rf_model_v2.h`
  - `model_comparison_v2.csv`

`kalman_filter_v2.h` - hand-written, not generated by the training script.

`model_comparison_report_v2.py` - reads `model_comparison_v2.csv` and writes
`model_comparison_v2.md` (table) and `model_comparison_v2.png` (chart).

`ESP32_multimodel_v2/ESP32_multimodel_v2.ino` - new firmware sketch, in its
own folder because Arduino requires a sketch's folder name to match its
filename. Runs all 5 approaches every cycle and prints timing for each.

`gui_multimodel_v2.py` - Tkinter + embedded matplotlib GUI: live-plots the
PT100 reference against all 5 corrected outputs, plus a readout table of each
model's current value and inference time. Same disconnect-to-simulation-mode
fallback convention as `gui_max6675.py`.

How to run:

`python train_multimodel_v2.py --csv dataset.csv`
`python model_comparison_report_v2.py`

Extra Python dependencies beyond v1 (`pyserial`):

`pip install scipy scikit-learn matplotlib emlearn tensorflow-model-optimization tabulate`

A note on reproducibility: `train_multimodel_v2.py` fixes a random seed
(`SEED = 42`, both `numpy` and `tensorflow`) before building any of the 3
neural nets. Earlier unseeded runs showed the cross-model and
cross-precision rankings below changing substantially between runs (e.g.
`TCN_Hadamard_v2` flipped from the worst neural approach to the best) purely
from random weight initialization -- not a real difference between the
approaches. The numbers below are from a seeded run and should reproduce
if you re-run the script unchanged.

Results (from `model_comparison_v2.md`, current `dataset.csv`, seed 42):

`Dense_v2` (float32) is the best overall at 0.39 C RMSE, with `TCN_Hadamard_v2`
(float32) close behind at 0.43 C and smaller besides, and `Hybrid_Physics_v2`
(qat_int8) also close at 0.43 C in just 2.2KB. `RandomForest` is the most
accurate of the non-neural approaches (0.92 C RMSE) but far the largest
artifact at ~99KB even after shrinking the forest (see Known limitations).
`Kalman` is the weakest (2.35 C RMSE) since it only smooths the raw K-type
signal and never sees PT100, so it mostly reproduces the average K-vs-PT100
sensor offset rather than correcting for it.

Notably, for both `Dense_v2` and `TCN_Hadamard_v2` the float32 variant beat
their own int8 (PTQ/QAT) variants this run, on both accuracy and (for TCN)
size. Quantization is not a free win here -- for models this small, whichever
random initialization the float32 model landed on strongly affects how well
it tolerates quantization afterward, so re-running with a different seed
could favor a different precision. Re-check `model_comparison_v2.csv` before
assuming a specific variant is best if you retrain.

`TCN_Hadamard_v2`'s int8 variant (12160 B) is larger than its own float32
variant (11096 B) -- consistent across every run regardless of seed, since
it's a structural property of the model (the two-branch merge adds
quantize/dequantize boundary ops that cost more than the int8 weight
compression saves for a model this small), not noise. Worth reporting as a
finding, not something to "fix".

A finding on `Hybrid_Physics_v2`'s physics term: `TAU_HAT` is tiny
(~6.6e-06), meaning the fitted first-order lag correction contributes almost
nothing -- the residual Dense net is doing nearly all of the correction, not
a real physics+residual blend. `train_multimodel_v2.py` tested fitting `tau`
against both the raw single-step K-type derivative and a smoothed (3-sample
causal moving average) version; both converge to the same negligible `tau`
and the same validation error, which in turn matches the error from using
raw, completely uncorrected K-type against PT100. This is a decisive (not
just suggestive) result: the K-type-vs-PT100 gap in this dataset is
dominated by a roughly constant sensor offset, not a rate-dependent thermal
lag, and the physics formula `k + tau*rate` has no bias/intercept term to
capture a constant offset, regardless of how the rate is estimated. State
this honestly if citing `Hybrid_Physics_v2` as a "physics-informed" result --
in practice here it is a residual-net result with a negligible physics term.

Firmware notes:

The chosen precision variant per model (cited from `model_comparison_v2.csv`,
seed 42 -- re-verify against a fresh run before trusting these if you retrain):
  - `Dense_v2`: `float32` -- best RMSE of its 3 variants this run; both int8
    variants were meaningfully worse despite being smaller.
  - `TCN_Hadamard_v2`: `float32` -- best RMSE AND smaller size than its only
    other variant (`ptq_int8`; no QAT variant exists for this model at all).
  - `Hybrid_Physics_v2`: `float32` always -- its physics/derivative term needs
    float precision regardless of what the comparison table shows.

All three neural nets are float32 in the current firmware -- there is no
int8 quantize/dequantize logic anywhere in `ESP32_multimodel_v2.ino`. The
Hybrid physics term also uses the 3-sample smoothed derivative (see the
finding above), not the raw single-step diff, to match what
`train_multimodel_v2.py` actually validated.

The 5 header files (`model_data_dense_v2_float32.h`,
`model_data_tcn_hadamard_v2_float32.h`, `model_data_hybrid_v2_float32.h`,
`rf_model_v2.h`, `kalman_filter_v2.h`) are copied into
`ESP32_multimodel_v2/` alongside the `.ino`, since Arduino only resolves
quoted `#include`s relative to the sketch's own folder. If
`train_multimodel_v2.py` is re-run and produces updated headers, re-copy the
updated files into that folder before recompiling -- and re-check the
firmware's `#include` lines and NOTE comment still cite the actually-best
variant, since a different seed or retrain could favor int8 again.

Known limitations:

- The sketch has not been compile-tested -- no Arduino/ESP32 toolchain was
  available in the environment that wrote it. Verify it compiles before
  uploading, and set the Arduino IDE partition scheme to one with a larger
  app partition (e.g. "Huge APP") given `rf_model_v2.h`'s size plus two
  TFLite models plus `AllOpsResolver` (which registers every TFLM kernel).
- `TCN_Hadamard_v2` has no QAT (`_qat_int8.h`) variant: `Conv1D` layers are
  not supported by `tensorflow-model-optimization`'s default quantize
  registry, independent of any TF/Keras version issue.
- `rf_model_v2.h` is ~99KB after reducing the forest to `n_estimators=8,
  max_depth=6` (down from 15/8, which produced an ~822KB header). This
  trades some RF accuracy (0.68 C RMSE at 15/8 vs 0.92 C at 8/6, in an
  earlier unseeded comparison) for a much smaller artifact. If flash space
  allows, the forest size can be tuned back up; if 99KB is still too large,
  `emlearn`'s `method='loadable'` produces a much more compact array-based
  representation instead of unrolled if/else code, but forces
  `dtype='int16_t'` (reintroducing integer truncation of raw °C
  features/thresholds -- would need a fixed-point scale-factor workaround to
  preserve precision) and requires vendoring `emlearn`'s `eml_trees.h`/
  `eml_common.h` runtime into the sketch folder. Not done here since the
  simpler forest-shrinking approach was sufficient and lower-risk.
- `AllOpsResolver` is used for all 3 interpreters (registers every TFLM
  kernel, larger flash footprint) rather than a curated
  `MicroMutableOpResolver` -- deliberate for this first bring-up so no
  unexpected op in any of the 3 models' graphs gets silently rejected. Once
  the sketch is confirmed compiling and running correctly, consider
  switching to a minimal resolver with just the ops these float32 models
  actually use (`FULLY_CONNECTED`, `RELU`, `CONV_2D`, `ADD`, `MEAN`,
  `CONCATENATION`, `PAD`) to reduce flash usage.


Troubleshooting checklist
-------------------------

If the GUI shows `DISCONNECTED (Simulation Mode)`:

1. Confirm the ESP32 sketch is uploaded and running.
2. Confirm `SERIAL_PORT` in `gui_max6675.py` matches Device Manager.
3. Close Arduino Serial Monitor before running the Python GUI.
4. Confirm ESP32 baud rate is `115200`.

If K-type status shows `FALLBACK`:

1. Check MAX6675 power and ground.
2. Check `SO -> GPIO19`, `SCK -> GPIO18`, and `CS -> GPIO14`.
3. Check that the thermocouple probe is firmly connected to the MAX6675 module.

If PT100 status shows `FALLBACK`:

1. Check MAX31865 power and ground.
2. Check `SDO -> GPIO19`, `SDI -> GPIO23`, `CLK -> GPIO18`, and `CS -> GPIO27`.
3. Confirm the Adafruit board is modified for 3-wire PT100 mode.
4. Confirm the two same-color RTD wires are on `F+` and `RTD+`.
5. Confirm the single RTD wire is on `RTD-`.
6. Leave `F-` empty for this 3-wire setup.
