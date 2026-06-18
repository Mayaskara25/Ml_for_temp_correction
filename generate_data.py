import numpy as np
import pandas as pd

# --- Configuration ---
SAMPLING_RATE_HZ = 2
DURATION_MINUTES = 180  # 3 hours of testing
TOTAL_SAMPLES = DURATION_MINUTES * 60 * SAMPLING_RATE_HZ
DT = 1.0 / SAMPLING_RATE_HZ  # Time step in seconds (0.5s)

# Sensor Simulation Parameters
THERMAL_LAG_FACTOR = 0.08  # Lower means more lag (K-type reacts slower to changes)
STATIC_BIAS = 0.6          # The K-type constantly reads 0.6°C higher than reality
NOISE_STD_DEV = 0.15       # Random electrical noise fluctuations in °C

def generate_true_profile():
    """Generates a dynamic 3-hour temperature profile for the water bath."""
    true_temp = np.zeros(TOTAL_SAMPLES)
    current_temp = 25.0  # Start at room temperature (25°C)
    
    for i in range(TOTAL_SAMPLES):
        time_min = (i * DT) / 60.0
        
        # 0 to 15 mins: Stable room temp
        if time_min < 15:
            current_temp = 25.0
        # 15 to 45 mins: Gradual heating up to 75°C
        elif time_min < 45:
            current_temp += 0.0277 * DT  # Linear ramp up
        # 45 to 65 mins: Hold stable at high temperature
        elif time_min < 65:
            current_temp = 75.0
        # 65 to 115 mins: Messy thermostat oscillations (cycles between 68°C and 78°C)
        elif time_min < 115:
            # Combining two sine waves to make it look like irregular heating cycles
            current_temp = 73.0 + 4.0 * np.sin(2 * np.pi * time_min / 10.0) + 1.0 * np.cos(2 * np.pi * time_min / 3.0)
        # 115 to 155 mins: Cooling down to 40°C
        elif time_min < 155:
            current_temp -= 0.0145 * DT  # Linear ramp down
        # 155 to 180 mins: Random walk fluctuations at lower temp
        else:
            current_temp += np.random.normal(0, 0.02)
            current_temp = np.clip(current_temp, 39.0, 41.0)
            
        true_temp[i] = current_temp
        
    return true_temp

def main():
    print("Generating synthetic water bath profile...")
    
    # 1. Generate the Ground Truth (What the high-accuracy PT100 would see)
    pt100_temp = generate_true_profile()
    
    # 2. Simulate the Cheap K-Type Thermocouple with Lag, Bias, and Noise
    k_type_temp = np.zeros(TOTAL_SAMPLES)
    
    # Initialize the first reading
    k_type_temp[0] = pt100_temp[0] + STATIC_BIAS
    
    for i in range(1, TOTAL_SAMPLES):
        # Apply exponential smoothing to model thermal lag physics
        # T_lag(t) = alpha * T_true(t) + (1 - alpha) * T_lag(t-1)
        lagged_value = (THERMAL_LAG_FACTOR * pt100_temp[i]) + ((1.0 - THERMAL_LAG_FACTOR) * k_type_temp[i-1])
        
        # Add random high-frequency electrical noise
        noise = np.random.normal(0, NOISE_STD_DEV)
        
        # Final reading includes lag, static calibration bias, and noise
        k_type_temp[i] = lagged_value + noise

    # Add the static bias fully to the beginning baseline
    k_type_temp[:int(15*60*SAMPLING_RATE_HZ)] += STATIC_BIAS

    # 3. Calculate the Dynamic Error (Cheap Sensor - Ground Truth)
    error = k_type_temp - pt100_temp
    
    # 4. Generate Timestamps in Milliseconds (matching ESP32 millis() environment)
    timestamps_ms = np.arange(0, TOTAL_SAMPLES * (DT * 1000), DT * 1000, dtype=int)
    
    # 5. Package into a DataFrame matching our exact CSV structure
    df = pd.DataFrame({
        'Timestamp_ms': timestamps_ms,
        'K_Type_Temp_C': np.round(k_type_temp, 2),
        'PT100_Temp_C': np.round(pt100_temp, 2),
        'Error_C': np.round(error, 2)
    })
    
    # Save to file
    output_filename = 'synthetic_training_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"Success! Saved {len(df)} rows of data to '{output_filename}'.")

if __name__ == '__main__':
    main()