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

Simulated / Model Values:

`Synthetic PT100 Ref` - synthetic PT100/reference signal.
`Synthetic K-Type Input` - synthetic K-type value used by the ML model.
`Correction Engine Output` - corrected temperature predicted by the model.
`Dynamic Error` - `Corrected_Temp_C - Synthetic_PT100_Temp_C`.

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

The ML model still uses synthetic K-type history as its input. This keeps the
old model test flow stable while hardware sensors are being verified.

The PT100 fallback behavior only means the code detected a bad/faulted RTD
reading and substituted a synthetic-like value so the GUI continues updating.
If `MAX31865 Sensor Status` shows `FALLBACK`, check wiring, jumper/solder
configuration, and PT100 terminal placement.

`model_data.h` and `train_dense_esp32.py` should only be changed when you are
ready to retrain or change the model behavior.


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
