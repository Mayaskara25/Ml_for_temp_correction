#ifndef KALMAN_FILTER_V2_H
#define KALMAN_FILTER_V2_H

// Tunable noise parameters -- starting values, may need empirical tuning
#define KALMAN_Q_TEMP   0.001f   // process noise: temperature
#define KALMAN_Q_VEL    0.0001f  // process noise: rate of change
#define KALMAN_R_MEAS   0.25f    // measurement noise variance (~0.5 C std dev, K-type sensor)

typedef struct {
    float x;        // estimated temperature
    float v;        // estimated rate of change (deg C / s)
    float p[2][2];  // state covariance matrix
    bool initialized;
} KalmanState;

static KalmanState kalman_v2_state = { .initialized = false };

// Call once per firmware loop cycle. dt is the elapsed time in seconds since
// the previous call. measurement is the raw live K-type reading in Celsius.
// Returns the Kalman-filtered temperature estimate in Celsius.
static inline float kalman_v2_step(float measurement, float dt) {
    KalmanState *s = &kalman_v2_state;

    if (!s->initialized) {
        s->x = measurement;
        s->v = 0.0f;
        s->p[0][0] = 1.0f; s->p[0][1] = 0.0f;
        s->p[1][0] = 0.0f; s->p[1][1] = 1.0f;
        s->initialized = true;
        return s->x;
    }

    // --- Predict step ---
    float x_pred = s->x + s->v * dt;
    float v_pred = s->v;

    float p00 = s->p[0][0] + dt * (s->p[1][0] + s->p[0][1]) + dt * dt * s->p[1][1] + KALMAN_Q_TEMP;
    float p01 = s->p[0][1] + dt * s->p[1][1];
    float p10 = s->p[1][0] + dt * s->p[1][1];
    float p11 = s->p[1][1] + KALMAN_Q_VEL;

    // --- Update step (measurement = temperature only) ---
    float y = measurement - x_pred;              // innovation
    float S = p00 + KALMAN_R_MEAS;                // innovation covariance
    float k0 = p00 / S;                           // Kalman gain, temp
    float k1 = p10 / S;                           // Kalman gain, velocity

    s->x = x_pred + k0 * y;
    s->v = v_pred + k1 * y;

    s->p[0][0] = (1 - k0) * p00;
    s->p[0][1] = (1 - k0) * p01;
    s->p[1][0] = p10 - k1 * p00;
    s->p[1][1] = p11 - k1 * p01;

    return s->x;
}

#endif // KALMAN_FILTER_V2_H
