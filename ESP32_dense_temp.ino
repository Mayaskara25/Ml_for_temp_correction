/*
  ESP32_dense_temp.ino
  --------------------
  Runs the existing dense temperature correction model using synthetic input,
  while also reading a real K-type thermocouple through a MAX6675 module and
  a real PT100 RTD through a MAX31865 module for display in gui_max6675.py.

  MAX6675 wiring:
    SO  -> GPIO19
    SCK -> GPIO18
    CS  -> GPIO14
    GND -> GND
    VCC -> 3V3

  MAX31865 wiring:
    SDO -> GPIO19
    SDI -> GPIO23
    CLK -> GPIO18
    CS  -> GPIO27
    GND -> GND
    3V3 -> 3V3

  Serial CSV format expected by gui_max6675.py:
    The GUI reads the first 8 fields. Extra MAX31865 diagnostic fields are
    appended after that for serial-monitor debugging.
*/

#include <Arduino.h>
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#include <MAX6675.h>
#include <Adafruit_MAX31865.h>
#include "model_data.h"

#define NUMBER_OF_INPUTS   10
#define NUMBER_OF_OUTPUTS  1
#define TENSOR_ARENA_SIZE  4096

static const int MAX6675_SCK = 18;
static const int MAX6675_CS  = 14;
static const int MAX6675_SO  = 19;

static const int MAX31865_CLK  = 18;
static const int MAX31865_CS   = 27;
static const int MAX31865_SDO  = 19;
static const int MAX31865_SDI  = 23;

static const float RTD_NOMINAL = 100.0f;
static const float RTD_REF_RESISTOR = 430.0f;

MAX6675 thermocouple(MAX6675_CS, MAX6675_SO, MAX6675_SCK);
Adafruit_MAX31865 rtd(MAX31865_CS, MAX31865_SDI, MAX31865_SDO, MAX31865_CLK);

Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

static const float T_MIN = 20.0f;
static const float T_MAX = 80.0f;

float input_buffer[NUMBER_OF_INPUTS];

static unsigned long previousMillis = 0;
static const unsigned long INTERVAL_MS = 500;
static float fake_time = 0.0f;

inline float scale_temp(float t) {
    return (t - T_MIN) / (T_MAX - T_MIN);
}

inline float unscale_temp(float s) {
    return s * (T_MAX - T_MIN) + T_MIN;
}

inline float rtd_resistance_from_raw(uint16_t raw) {
    return ((float)raw * RTD_REF_RESISTOR) / 32768.0f;
}
void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    Serial.println("[BOOT] Initialising sensors and TFLite model...");
    thermocouple.begin();
    rtd.begin(MAX31865_3WIRE);

    if (!ml.begin(model_data)) {
        Serial.println("[ERROR] ml.begin() failed!");
        Serial.print("        ");
        Serial.println(ml.getErrorMessage());
        while (true) { delay(1000); }
    }

    Serial.println("[OK]   Model loaded successfully.");
    Serial.println("Timestamp_ms,Live_K_Temp_C,Synthetic_K_Temp_C,Live_PT100_Temp_C,Synthetic_PT100_Temp_C,Corrected_Temp_C,K_Sensor_OK,PT100_Sensor_OK,RTD3_Raw,RTD3_Resistance_Ohm,RTD3_Temp_C,RTD3_Fault,RTD2_Raw,RTD2_Resistance_Ohm,RTD2_Temp_C,RTD2_Fault,RTD3_minus_RTD2_C");

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

    float synthetic_pt100 = 40.0f + 10.0f * sinf(fake_time / 10.0f);
    float synthetic_k_for_model = 40.0f + 10.0f * sinf((fake_time - 2.0f) / 10.0f) + 0.6f;

    uint8_t sensor_status = thermocouple.read();
    float live_k_temp = thermocouple.getCelsius();
    bool sensor_ok = (sensor_status == STATUS_OK);

    if (isnan(live_k_temp) || live_k_temp < -100.0f || live_k_temp > 1024.0f) {
        sensor_ok = false;
        live_k_temp = 25.0f;
    }

    rtd.setWires(MAX31865_3WIRE);
    uint16_t rtd3_raw = rtd.readRTD();
    float rtd3_resistance = rtd_resistance_from_raw(rtd3_raw);
    float rtd3_temp = rtd.calculateTemperature(rtd3_raw, RTD_NOMINAL, RTD_REF_RESISTOR);
    uint8_t rtd3_fault = rtd.readFault();

    rtd.setWires(MAX31865_2WIRE);
    uint16_t rtd2_raw = rtd.readRTD();
    float rtd2_resistance = rtd_resistance_from_raw(rtd2_raw);
    float rtd2_temp = rtd.calculateTemperature(rtd2_raw, RTD_NOMINAL, RTD_REF_RESISTOR);
    uint8_t rtd2_fault = rtd.readFault();

    rtd.setWires(MAX31865_3WIRE);

    float live_pt100_temp = rtd3_temp;
    bool rtd_ok = (rtd3_fault == 0);

    if (isnan(live_pt100_temp) || live_pt100_temp < -100.0f || live_pt100_temp > 850.0f) {
        rtd_ok = false;
    }

    if (!rtd_ok) {
        float fallback_noise = (float)random(-50, 51) / 100.0f;
        live_pt100_temp = synthetic_pt100 + fallback_noise;
        rtd.clearFault();
    }

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
    Serial.print(live_k_temp, 2);
    Serial.print(",");
    Serial.print(synthetic_k_for_model, 2);
    Serial.print(",");
    Serial.print(live_pt100_temp, 2);
    Serial.print(",");
    Serial.print(synthetic_pt100, 2);
    Serial.print(",");
    Serial.print(corrected_temp, 2);
    Serial.print(",");
    Serial.print(sensor_ok ? "1" : "0");
    Serial.print(",");
    Serial.print(rtd_ok ? "1" : "0");
    Serial.print(",");
    Serial.print(rtd3_raw);
    Serial.print(",");
    Serial.print(rtd3_resistance, 3);
    Serial.print(",");
    Serial.print(rtd3_temp, 3);
    Serial.print(",");
    Serial.print(rtd3_fault);
    Serial.print(",");
    Serial.print(rtd2_raw);
    Serial.print(",");
    Serial.print(rtd2_resistance, 3);
    Serial.print(",");
    Serial.print(rtd2_temp, 3);
    Serial.print(",");
    Serial.print(rtd2_fault);
    Serial.print(",");
    Serial.println(rtd3_temp - rtd2_temp, 3);
}


