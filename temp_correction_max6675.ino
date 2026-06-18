/*
  temp_correction_max6675.ino
  ---------------------------
  ESP32 + MAX6675 K-type thermocouple reader with the existing dense ML model.
  The sketch keeps the synthetic PT100/reference signal and synthetic model
  input for testing, but also reads the live thermocouple value from the MAX6675
  and prints it for display only.

  Wiring used by the user:
    SO  -> GPIO19
    SCK -> GPIO18
    CS  -> GPIO14
    GND -> GND
    VCC -> 3V3
*/

#include <Arduino.h>
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#include <max6675.h>
#include "model_data.h"

#define NUMBER_OF_INPUTS   10
#define NUMBER_OF_OUTPUTS  1
#define TENSOR_ARENA_SIZE  4096

// MAX6675 pins
static const int MAX6675_SCK = 18;
static const int MAX6675_CS  = 14;
static const int MAX6675_SO  = 19;

MAX6675 thermocouple(MAX6675_SCK, MAX6675_CS, MAX6675_SO);

Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

static const float T_MIN = 20.0f;
static const float T_MAX = 80.0f;

inline float scale_temp(float t) {
    return (t - T_MIN) / (T_MAX - T_MIN);
}

inline float unscale_temp(float s) {
    return s * (T_MAX - T_MIN) + T_MIN;
}

float input_buffer[NUMBER_OF_INPUTS];

static unsigned long previousMillis = 0;
static const unsigned long INTERVAL_MS = 500;
static float fake_time = 0.0f;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    Serial.println("[BOOT] Initialising MAX6675 and TFLite model...");

    if (!ml.begin(model_data)) {
        Serial.println("[ERROR] ml.begin() failed!");
        Serial.print("        ");
        Serial.println(ml.getErrorMessage());
        while (true) { delay(1000); }
    }

    Serial.println("[OK]   Model loaded successfully.");
    Serial.println("Timestamp_ms,K_Temp_C,PT100_Temp_C,Corrected_Temp_C,Sensor_OK");

    float neutral = scale_temp(25.0f);
    for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
        input_buffer[i] = neutral;
    }
}

void loop() {
    unsigned long now = millis();
    if (now - previousMillis < INTERVAL_MS) return;
    previousMillis = now;

    fake_time += 0.5f;

    // Keep a synthetic reference stream so the model still behaves like the old setup.
    float pt100_temp = 40.0f + 10.0f * sinf(fake_time / 10.0f);

    // Read the actual thermocouple for display only.
    float k_temp = thermocouple.readCelsius();
    bool sensor_ok = true;

    // MAX6675 commonly returns NAN when the probe is open/disconnected.
    if (isnan(k_temp) || k_temp < -100.0f || k_temp > 1024.0f) {
        sensor_ok = false;
        k_temp = 25.0f; // fallback for display if the probe is disconnected
    }

    // The model input remains fully synthetic.
    float synthetic_k_for_model = 40.0f + 10.0f * sinf((fake_time - 2.0f) / 10.0f) + 0.6f;

    for (int i = 0; i < NUMBER_OF_INPUTS - 1; i++) {
        input_buffer[i] = input_buffer[i + 1];
    }
    input_buffer[NUMBER_OF_INPUTS - 1] = scale_temp(synthetic_k_for_model);

    float corrected_scaled = ml.predict(input_buffer);
    if (!ml.isOk()) {
        Serial.println("[ERROR] ml.predict() failed - skipping sample.");
        Serial.print("        ");
        Serial.println(ml.getErrorMessage());
        return;
    }

    float corrected_temp = unscale_temp(corrected_scaled);

    Serial.print(now);
    Serial.print(",");
    Serial.print(k_temp, 2);
    Serial.print(",");
    Serial.print(pt100_temp, 2);
    Serial.print(",");
    Serial.print(corrected_temp, 2);
    Serial.print(",");
    Serial.println(sensor_ok ? "1" : "0");
}
