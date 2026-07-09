/*
  ESP32_multimodel_v2.ino
  ------------------------
  Runs and compares 5 correction approaches against real MAX6675 (K-type) and
  MAX31865 (PT100) sensor readings -- no synthetic data anywhere in this
  sketch. Model inputs are built exclusively from the live K-type reading's
  rolling history.

  Approaches run every cycle:
    1. Dense_v2       -- float32 (variant chosen -- see NOTE below)
    2. TCN_Hadamard_v2 -- float32 (variant chosen -- see NOTE below)
    3. RandomForest    -- emlearn-generated C function (float features)
    4. Kalman          -- hand-written constant-velocity Kalman filter
    5. Hybrid_Physics_v2 -- fitted first-order lag + float32 residual net

  NOTE on variant selection (per model_comparison_v2.csv from a SEEDED run of
  train_multimodel_v2.py -- see that script's SEED constant; without a fixed
  random seed, Dense_v2/TCN_Hadamard_v2/Hybrid_Physics_v2's cross-model and
  cross-precision rankings were observed to change substantially between
  training runs, since the training script never previously seeded TF/Keras's
  random init):
    Dense_v2:        float32 chosen -- best RMSE (0.3957 C) of its 3 variants;
                      ptq_int8 (0.4279 C) and qat_int8 (0.4426 C) were both
                      worse this run, despite being smaller.
    TCN_Hadamard_v2: float32 chosen -- best RMSE (0.3684 C) AND smaller size
                      (11144 B) than its only other variant, ptq_int8
                      (0.5813 C, 12208 B -- worse on both axes this run). QAT
                      was not available for this model at all:
                      tensorflow-model-optimization's default quantize
                      registry does not support Conv1D layers.
    Hybrid_Physics_v2: float32 always used here (per plan) -- the physics lag
                      term's derivative math needs float precision, regardless
                      of what the comparison table shows for its own PTQ/QAT
                      variants (which were produced for comparison-table
                      completeness only, per train_multimodel_v2.py Step 2.7).
                      qat_int8 (0.4187 C) edged out float32 (0.4267 C) this
                      run, but the float32 default here is a precision
                      requirement, not an accuracy-driven choice.

  All three neural nets are float32 in this build -- no int8 quantize/
  dequantize logic anywhere in this sketch. Re-run train_multimodel_v2.py's
  comparison and update this NOTE (and the #include lines below) if a future
  seeded run favors a different variant for Dense_v2 or TCN_Hadamard_v2.

  NOTE on Hybrid_Physics_v2's physics term: train_multimodel_v2.py tested
  fitting TAU_HAT against both the raw single-step K-type derivative and a
  3-sample causal moving average of it; the smoothed version won (by a
  razor-thin margin -- TAU_HAT itself is ~6.6e-06, so this makes negligible
  practical difference either way). The firmware below therefore feeds the
  physics term a 3-sample causal moving average of the derivative, computed
  from the same dk_history buffer used by Dense_v2/TCN_Hadamard_v2 (whose own
  inputs are unaffected -- they still use the raw single-step derivative).
  See train_multimodel_v2.py's printed DIAGNOSTIC line: the physics-only
  baseline (either variant) is statistically indistinguishable from using the
  raw K-type reading with zero correction at all, meaning the K-vs-PT100 gap
  in this dataset is dominated by a roughly-constant sensor offset rather
  than a rate-dependent thermal lag -- this simple tau-only formula has no
  bias/intercept term, so it cannot capture a constant offset regardless of
  how the derivative is estimated. Hybrid_Physics_v2's correction is, in
  practice, coming almost entirely from the residual Dense net, not the
  physics term. State this honestly in any write-up of these results.

  IMPORTANT -- header file dependency:
    model_data_dense_v2_float32.h, model_data_tcn_hadamard_v2_float32.h,
    model_data_hybrid_v2_float32.h, rf_model_v2.h, and kalman_filter_v2.h are
    generated/maintained at the REPO ROOT (C:\projects\idealabEl\), not in this
    folder -- Arduino requires headers to live alongside the .ino for a quoted
    #include to resolve, so copies of these 5 files live in this sketch folder
    too. If train_multimodel_v2.py is re-run and produces updated headers,
    re-copy the updated files into this folder before recompiling.

  MAX6675 wiring (identical to ESP32_dense_temp.ino):
    SO  -> GPIO19
    SCK -> GPIO18
    CS  -> GPIO14
    GND -> GND
    VCC -> 3V3

  MAX31865 wiring (identical to ESP32_dense_temp.ino, 3-wire mode):
    SDO -> GPIO19
    SDI -> GPIO23
    CLK -> GPIO18
    CS  -> GPIO27
    GND -> GND
    3V3 -> 3V3

  Serial CSV format (115200 baud), one header line then one data line per cycle:
    Timestamp_ms,Live_K_Temp_C,Live_PT100_Temp_C,K_Sensor_OK,PT100_Sensor_OK,
    Dense_v2_C,Dense_v2_us,TCN_Hadamard_C,TCN_Hadamard_us,RF_C,RF_us,
    Kalman_C,Kalman_us,Hybrid_C,Hybrid_us

  COMPILATION NOTE: this sketch could not be compiled in the environment that
  wrote it (no Arduino/ESP32 toolchain available there). The TFLite Micro API
  used below (tflite::GetModel, tflite::AllOpsResolver, tflite::MicroInterpreter,
  TfLiteTensor->data.f, ->params.scale/.zero_point) matches the API that
  has been stable across recent versions of the "TensorFlow Lite for
  Microcontrollers" Arduino library, but has not been verified against the
  specific library version installed in this project's Arduino workspace.
  If tflite::AllOpsResolver is unavailable in the installed library version,
  switch to tflite::MicroMutableOpResolver<N> and manually register (verified
  by loading the actual generated .tflite bytes and inspecting the interpreter's
  op list, not guessed): FULLY_CONNECTED, RELU, CONV_2D, ADD, SUB, NEG, MEAN,
  CONCATENATION, PAD, RESHAPE, EXPAND_DIMS, SPACE_TO_BATCH_ND, BATCH_TO_SPACE_ND
  (the last two come from TCN_Hadamard_v2's dilated Conv1D layer, which TFLite
  lowers to a space-to-batch/conv/batch-to-space sequence).

  KNOWN BUG, FIXED: HadamardTransform's soft-threshold originally used
  tf.sign()/tf.abs(), which lowers to a SIGN builtin op. This ESP32's TFLite
  Micro build's AllOpsResolver does not have SIGN registered, so
  AllocateTensors() failed for TCN_Hadamard_v2 with "Didn't find op for
  builtin opcode 'SIGN'", cascading into a crash once the sketch tried to run
  inference against an unallocated interpreter. Replacing tf.sign() with an
  equivalent tf.where()-based sign did NOT fix this -- the TFLite converter
  can canonicalize a "where(x>=0, 1, -1)" pattern back into a SIGN op during
  graph optimization, independent of what the Python source calls directly.
  The fix (see train_multimodel_v2.py's HadamardTransform.call()) expresses
  the same soft-threshold using only RELU and subtraction --
  sign(x)*relu(|x|-theta) == relu(x-theta) - relu(-x-theta) -- which cannot
  produce a SIGN op under any graph optimization. Verified by loading the
  regenerated .tflite bytes and confirming SIGN is absent from the op list,
  not just by re-reading the Python source.
*/

#include <Arduino.h>
#include <math.h>
#include <MAX6675.h>
#include <Adafruit_MAX31865.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data_dense_v2_float32.h"
#include "model_data_tcn_hadamard_v2_float32.h"
#include "model_data_hybrid_v2_float32.h"
#include "rf_model_v2.h"
#include "kalman_filter_v2.h"

// ---------------------------------------------------------------------------
// Sensor wiring (identical to ESP32_dense_temp.ino)
// ---------------------------------------------------------------------------
static const int MAX6675_SCK = 18;
static const int MAX6675_CS  = 14;
static const int MAX6675_SO  = 19;

static const int MAX31865_CLK = 18;
static const int MAX31865_CS  = 27;
static const int MAX31865_SDO = 19;
static const int MAX31865_SDI = 23;

static const float RTD_NOMINAL = 100.0f;
static const float RTD_REF_RESISTOR = 430.0f;

MAX6675 thermocouple(MAX6675_CS, MAX6675_SO, MAX6675_SCK);
Adafruit_MAX31865 rtd(MAX31865_CS, MAX31865_SDI, MAX31865_SDO, MAX31865_CLK);

// ---------------------------------------------------------------------------
// Constants transcribed VERBATIM from train_multimodel_v2.py's printed output
// ---------------------------------------------------------------------------
static const float T_MIN = 20.0f;
static const float T_MAX = 80.0f;
static const float D_MIN = -10.5f;
static const float D_MAX = 12.5f;
static const float TAU_HAT = 6.643747950818272e-06f;
static const float RESIDUAL_SCALE = 20.530044555664062f;
static const float DT_SECONDS = 0.5f;
static const unsigned long INTERVAL_MS = (unsigned long)(DT_SECONDS * 1000.0f);

#define LOOKBACK 10
#define HADAMARD_WINDOW 8
#define PHYSICS_SMOOTH_WINDOW 3

static unsigned long previousMillis = 0;

inline float scale_temp(float t) {
    return (t - T_MIN) / (T_MAX - T_MIN);
}

inline float unscale_temp(float s) {
    return s * (T_MAX - T_MIN) + T_MIN;
}

inline float scale_derivative(float d) {
    if (D_MAX - D_MIN < 1e-9f) return 0.0f;
    return (d - D_MIN) / (D_MAX - D_MIN);
}

// ---------------------------------------------------------------------------
// Rolling buffers -- raw K-type history and its derivative history.
// dk_history[i] is recomputed as (k_history[i] - k_history[i-1]) / DT_SECONDS
// each cycle via the retained raw_prev scalar, which is mathematically
// equivalent to recomputing all consecutive diffs of k_history from scratch
// every cycle, but avoids the redundant O(LOOKBACK) recomputation.
// ---------------------------------------------------------------------------
static float k_history[LOOKBACK];
static float dk_history[LOOKBACK];
static float raw_prev = 25.0f;

// ---------------------------------------------------------------------------
// TFLite Micro setup -- three independent interpreters, three independent
// tensor arenas (not shared), one shared AllOpsResolver (stateless registry).
// ---------------------------------------------------------------------------
tflite::AllOpsResolver resolver;

constexpr int kDenseArenaSize = 4096;
alignas(16) static uint8_t dense_arena[kDenseArenaSize];
tflite::MicroInterpreter* dense_interpreter = nullptr;
TfLiteTensor* dense_input = nullptr;
TfLiteTensor* dense_output = nullptr;

constexpr int kTcnArenaSize = 16384;
alignas(16) static uint8_t tcn_arena[kTcnArenaSize];
tflite::MicroInterpreter* tcn_interpreter = nullptr;
TfLiteTensor* tcn_seq_input = nullptr;
TfLiteTensor* tcn_had_input = nullptr;
TfLiteTensor* tcn_output = nullptr;

constexpr int kHybridArenaSize = 4096;
alignas(16) static uint8_t hybrid_arena[kHybridArenaSize];
tflite::MicroInterpreter* hybrid_interpreter = nullptr;
TfLiteTensor* hybrid_input = nullptr;
TfLiteTensor* hybrid_output = nullptr;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    Serial.println("[BOOT] Initialising sensors and TFLite models...");
    thermocouple.begin();
    rtd.begin(MAX31865_3WIRE);

    // --- Dense_v2 (float32) ---
    const tflite::Model* dense_model = tflite::GetModel(model_data_dense_v2_float32);
    if (dense_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ERROR] Dense_v2 model schema version mismatch!");
    }
    static tflite::MicroInterpreter dense_static_interpreter(dense_model, resolver, dense_arena, kDenseArenaSize);
    dense_interpreter = &dense_static_interpreter;
    if (dense_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ERROR] Dense_v2 AllocateTensors() failed -- increase kDenseArenaSize.");
    }
    dense_input = dense_interpreter->input(0);
    dense_output = dense_interpreter->output(0);

    // --- TCN_Hadamard_v2 (float32, two inputs) ---
    const tflite::Model* tcn_model = tflite::GetModel(model_data_tcn_hadamard_v2_float32);
    if (tcn_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ERROR] TCN_Hadamard_v2 model schema version mismatch!");
    }
    static tflite::MicroInterpreter tcn_static_interpreter(tcn_model, resolver, tcn_arena, kTcnArenaSize);
    tcn_interpreter = &tcn_static_interpreter;
    if (tcn_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ERROR] TCN_Hadamard_v2 AllocateTensors() failed -- increase kTcnArenaSize.");
    }

    TfLiteTensor* tcn_in0 = tcn_interpreter->input(0);
    TfLiteTensor* tcn_in1 = tcn_interpreter->input(1);
    Serial.print("[INFO] TCN_Hadamard_v2 input(0) dims: ");
    for (int i = 0; i < tcn_in0->dims->size; i++) { Serial.print(tcn_in0->dims->data[i]); Serial.print(" "); }
    Serial.println();
    Serial.print("[INFO] TCN_Hadamard_v2 input(1) dims: ");
    for (int i = 0; i < tcn_in1->dims->size; i++) { Serial.print(tcn_in1->dims->data[i]); Serial.print(" "); }
    Serial.println();

    // Identify the 3D sequence tensor vs the 2D Hadamard tensor by rank --
    // do not assume a fixed input order (mirrors train_multimodel_v2.py's
    // run_tflite_multi_input() Python-side approach, validated during training).
    if (tcn_in0->dims->size == 3) {
        tcn_seq_input = tcn_in0;
        tcn_had_input = tcn_in1;
    } else {
        tcn_seq_input = tcn_in1;
        tcn_had_input = tcn_in0;
    }
    tcn_output = tcn_interpreter->output(0);

    // --- Hybrid_Physics_v2 residual net (float32) ---
    const tflite::Model* hybrid_model = tflite::GetModel(model_data_hybrid_v2_float32);
    if (hybrid_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ERROR] Hybrid_Physics_v2 model schema version mismatch!");
    }
    static tflite::MicroInterpreter hybrid_static_interpreter(hybrid_model, resolver, hybrid_arena, kHybridArenaSize);
    hybrid_interpreter = &hybrid_static_interpreter;
    if (hybrid_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ERROR] Hybrid_Physics_v2 AllocateTensors() failed -- increase kHybridArenaSize.");
    }
    hybrid_input = hybrid_interpreter->input(0);
    hybrid_output = hybrid_interpreter->output(0);

    Serial.println("[OK]   All models loaded.");
    Serial.println("Timestamp_ms,Live_K_Temp_C,Live_PT100_Temp_C,K_Sensor_OK,PT100_Sensor_OK,Dense_v2_C,Dense_v2_us,TCN_Hadamard_C,TCN_Hadamard_us,RF_C,RF_us,Kalman_C,Kalman_us,Hybrid_C,Hybrid_us");

    // Cold-start buffers with a neutral value, same convention as ESP32_dense_temp.ino
    for (int i = 0; i < LOOKBACK; i++) {
        k_history[i] = 25.0f;
        dk_history[i] = 0.0f;
    }
    raw_prev = 25.0f;
}

void loop() {
    unsigned long now = millis();
    if (now - previousMillis < INTERVAL_MS) return;
    previousMillis = now;

    // --- 7a. Read live sensors -- same fault-detection as ESP32_dense_temp.ino.
    // No synthetic waveform substitution: on fault, clamp to a neutral value
    // and flag *_OK=0 so downstream consumers (GUI/logger) know to discard it.
    uint8_t sensor_status = thermocouple.read();
    float live_k_temp = thermocouple.getCelsius();
    bool sensor_ok = (sensor_status == STATUS_OK);
    if (isnan(live_k_temp) || live_k_temp < -100.0f || live_k_temp > 1024.0f) {
        sensor_ok = false;
        live_k_temp = 25.0f;
    }

    uint16_t rtd_raw = rtd.readRTD();
    float live_pt100_temp = rtd.calculateTemperature(rtd_raw, RTD_NOMINAL, RTD_REF_RESISTOR);
    uint8_t rtd_fault = rtd.readFault();
    bool rtd_ok = (rtd_fault == 0);
    if (isnan(live_pt100_temp) || live_pt100_temp < -100.0f || live_pt100_temp > 850.0f) {
        rtd_ok = false;
    }
    if (!rtd_ok) {
        live_pt100_temp = 25.0f;
        rtd.clearFault();
    }

    // --- 7b. Update rolling buffers ---
    float dk_new = (live_k_temp - raw_prev) / DT_SECONDS;
    for (int i = 0; i < LOOKBACK - 1; i++) {
        k_history[i] = k_history[i + 1];
        dk_history[i] = dk_history[i + 1];
    }
    k_history[LOOKBACK - 1] = live_k_temp;
    dk_history[LOOKBACK - 1] = dk_new;
    raw_prev = live_k_temp;

    float k_scaled_history[LOOKBACK];
    float dk_scaled_history[LOOKBACK];
    for (int i = 0; i < LOOKBACK; i++) {
        k_scaled_history[i] = scale_temp(k_history[i]);
        dk_scaled_history[i] = scale_derivative(dk_history[i]);
    }

    // Flattened [k_0..k_9, dk_0..dk_9] window, matches features_v2.build_windows()
    float x_flat[LOOKBACK * 2];
    for (int i = 0; i < LOOKBACK; i++) {
        x_flat[i] = k_scaled_history[i];
        x_flat[LOOKBACK + i] = dk_scaled_history[i];
    }

    // Most recent 8 of the 10 K-scaled readings, matches Step 2.3's Hadamard branch input
    float x_had[HADAMARD_WINDOW];
    for (int i = 0; i < HADAMARD_WINDOW; i++) {
        x_had[i] = k_scaled_history[LOOKBACK - HADAMARD_WINDOW + i];
    }

    // --- 7c. Dense_v2 inference (float32) -- time Invoke() only ---
    for (int i = 0; i < LOOKBACK * 2; i++) {
        dense_input->data.f[i] = x_flat[i];
    }
    unsigned long t0 = micros();
    dense_interpreter->Invoke();
    unsigned long dense_us = micros() - t0;
    float dense_pred_c = unscale_temp(dense_output->data.f[0]);

    // --- 7d. TCN_Hadamard_v2 inference (float32, two inputs) -- time Invoke() only ---
    for (int t = 0; t < LOOKBACK; t++) {
        tcn_seq_input->data.f[t * 2 + 0] = k_scaled_history[t];
        tcn_seq_input->data.f[t * 2 + 1] = dk_scaled_history[t];
    }
    for (int i = 0; i < HADAMARD_WINDOW; i++) {
        tcn_had_input->data.f[i] = x_had[i];
    }
    t0 = micros();
    tcn_interpreter->Invoke();
    unsigned long tcn_us = micros() - t0;
    float tcn_pred_c = unscale_temp(tcn_output->data.f[0]);

    // --- 7e. Random Forest inference (emlearn, raw °C features, no scaling) ---
    float rf_features[LOOKBACK * 2];
    for (int i = 0; i < LOOKBACK; i++) {
        rf_features[i] = k_history[i];
        rf_features[LOOKBACK + i] = dk_history[i];
    }
    t0 = micros();
    float rf_pred_c = rf_model_predict(rf_features, LOOKBACK * 2);
    unsigned long rf_us = micros() - t0;

    // --- 7f. Kalman filter ---
    t0 = micros();
    float kalman_pred_c = kalman_v2_step(live_k_temp, DT_SECONDS);
    unsigned long kalman_us = micros() - t0;

    // --- 7g. Hybrid: physics term (float math, not timed) + residual net (TFLite timed) ---
    // Uses a 3-sample causal moving average of the derivative, NOT the raw
    // single-step diff -- train_multimodel_v2.py's smoothed-derivative tau
    // fit won (see the file header NOTE). Dense_v2/TCN_Hadamard_v2 above are
    // unaffected -- they still use the raw per-sample derivative.
    float physics_rate = 0.0f;
    for (int i = 0; i < PHYSICS_SMOOTH_WINDOW; i++) {
        physics_rate += dk_history[LOOKBACK - 1 - i];
    }
    physics_rate /= (float)PHYSICS_SMOOTH_WINDOW;
    float physics_correction = live_k_temp + TAU_HAT * physics_rate;
    for (int i = 0; i < LOOKBACK * 2; i++) {
        hybrid_input->data.f[i] = x_flat[i];
    }
    t0 = micros();
    hybrid_interpreter->Invoke();
    unsigned long hybrid_us = micros() - t0;
    float residual_scaled = hybrid_output->data.f[0];
    float hybrid_pred_c = physics_correction + residual_scaled * RESIDUAL_SCALE;

    // --- 8. Serial CSV output ---
    Serial.print(now); Serial.print(",");
    Serial.print(live_k_temp, 2); Serial.print(",");
    Serial.print(live_pt100_temp, 2); Serial.print(",");
    Serial.print(sensor_ok ? "1" : "0"); Serial.print(",");
    Serial.print(rtd_ok ? "1" : "0"); Serial.print(",");
    Serial.print(dense_pred_c, 2); Serial.print(",");
    Serial.print(dense_us); Serial.print(",");
    Serial.print(tcn_pred_c, 2); Serial.print(",");
    Serial.print(tcn_us); Serial.print(",");
    Serial.print(rf_pred_c, 2); Serial.print(",");
    Serial.print(rf_us); Serial.print(",");
    Serial.print(kalman_pred_c, 2); Serial.print(",");
    Serial.print(kalman_us); Serial.print(",");
    Serial.print(hybrid_pred_c, 2); Serial.print(",");
    Serial.println(hybrid_us);
}
