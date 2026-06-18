idealabEL ESP32 Dense Model Fix Notes
=====================================

What was changed
----------------

1. The Python training script `train_dense_esp32.py` was kept as the source for generating `model_data.h`.
2. The generated `model_data.h` was copied into the Arduino sketch folder:
   `C:\ArduinoWorkspace\ESP32_dense_engine\temp_correction_esp32`
3. The Arduino sketch `temp_correction_esp32.ino` was updated to match EloquentTinyML v2.4.4.
4. The installed Arduino EloquentTinyML library was repaired so the compiler could find the correct headers.

Arduino code fixes
------------------

The sketch originally used the wrong include and class names for the installed library version.

Old:
`#include <eloquent_tinyml.h>`
`Eloquent::TinyML::TFLite<...>`

New:
`#include <EloquentTinyML.h>`
`#include <eloquent_tinyml/tensorflow.h>`
`Eloquent::TinyML::TensorFlow::TensorFlow<...>`

The inference call was also aligned with v2.4.4:

Old:
`ml.predict(input_buffer, output_buffer)`

New:
`float corrected_scaled = ml.predict(input_buffer);`

Startup error reporting was improved with:
`ml.getErrorMessage()`

Library fix
----------

The Arduino library folder originally pointed to a broken/incomplete EloquentTinyML install.
It was replaced with the cached EloquentTinyML 2.4.4 package contents so Arduino could compile the sketch.

The library now contains the required files:

`EloquentTinyML.h`
`eloquent_tinyml/tensorflow.h`

Upload fix
----------

The ESP32 initially failed to upload with:
`Wrong boot mode detected (0x13)`

The fix was to manually enter download mode:

1. Start upload.
2. Hold the `BOOT` button when Arduino shows `Connecting...`
3. Keep holding until upload starts writing.
4. Release `BOOT`

If needed, `BOOT` + tap `EN/RST` also works.

Verification
------------

After the fixes, the sketch compiled successfully for:
`ESP32 Dev Module`

The serial output was then working and producing CSV-style rows like:
`timestamp,k_temp,pt100_temp,corrected_temp`

Notes
-----

The current sketch is still using synthetic temperature values for testing.
To use real hardware, replace the fake `pt100_temp` and `k_temp` generation in `loop()` with actual sensor reads.
