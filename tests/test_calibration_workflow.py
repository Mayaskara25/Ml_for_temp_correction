import unittest

import numpy as np
import pandas as pd

from calibration_io import dynamic_error, parse_serial_line, sensor_status_text
from train_dense_esp32 import build_windows, chronological_split, regression_metrics, scale, unscale_temp


class CalibrationIoTests(unittest.TestCase):
    def test_parse_readme_8_field_row(self):
        row = parse_serial_line("1000,31.0,30.5,30.8,30.7,30.9,1,1")

        self.assertEqual(row.schema, "readme_8_field")
        self.assertEqual(row.timestamp_ms, 1000)
        self.assertEqual(row.live_k_temp_c, 31.0)
        self.assertEqual(row.synthetic_k_temp_c, 30.5)
        self.assertEqual(row.live_pt100_temp_c, 30.8)
        self.assertTrue(row.k_sensor_ok)
        self.assertAlmostEqual(dynamic_error(row), 0.1)

    def test_parse_legacy_5_field_row(self):
        row = parse_serial_line("1000,31.0,30.8,30.9,0")

        self.assertEqual(row.schema, "legacy_max6675_5_field")
        self.assertFalse(row.k_sensor_ok)
        self.assertIsNone(row.live_pt100_temp_c)
        self.assertEqual(sensor_status_text(row.pt100_sensor_ok), "N/A")
        self.assertAlmostEqual(dynamic_error(row), 0.1)

    def test_parse_legacy_4_field_row(self):
        row = parse_serial_line("1000,31.0,30.8,30.9")

        self.assertEqual(row.schema, "legacy_synthetic_4_field")
        self.assertTrue(row.k_sensor_ok)
        self.assertTrue(row.pt100_sensor_ok)
        self.assertAlmostEqual(dynamic_error(row), 0.1)

    def test_parse_ignores_headers_and_debug_lines(self):
        self.assertIsNone(parse_serial_line("Timestamp_ms,K_Temp_C,PT100_Temp_C,Corrected_Temp_C"))
        self.assertIsNone(parse_serial_line("[BOOT] Model loaded"))


class TrainingLogicTests(unittest.TestCase):
    def test_scale_round_trip(self):
        values = np.array([20.0, 50.0, 80.0], dtype=np.float32)
        np.testing.assert_allclose(unscale_temp(scale(values)), values)

    def test_same_time_window_alignment(self):
        df = pd.DataFrame(
            {
                "K_Type_Temp_C": [20, 21, 22, 23, 24],
                "PT100_Temp_C": [30, 31, 32, 33, 34],
            }
        )
        X, y, target_c, newest_k_c = build_windows(df, lookback=3, target_mode="same-time")

        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(float(target_c[0]), 32.0)
        self.assertEqual(float(newest_k_c[0]), 22.0)
        self.assertAlmostEqual(float(unscale_temp(y[0][0])), 32.0)

    def test_forecast_window_alignment(self):
        df = pd.DataFrame(
            {
                "K_Type_Temp_C": [20, 21, 22, 23, 24],
                "PT100_Temp_C": [30, 31, 32, 33, 34],
            }
        )
        X, y, target_c, newest_k_c = build_windows(df, lookback=3, target_mode="forecast")

        self.assertEqual(X.shape, (2, 3))
        self.assertEqual(float(target_c[0]), 33.0)
        self.assertEqual(float(newest_k_c[0]), 22.0)
        self.assertAlmostEqual(float(unscale_temp(y[0][0])), 33.0)

    def test_chronological_split_and_metrics(self):
        X = np.arange(100, dtype=np.float32).reshape(20, 5)
        y = np.arange(20, dtype=np.float32).reshape(20, 1)
        target = np.arange(20, dtype=np.float32)
        k = target + 1

        split = chronological_split(X, y, target, k)
        self.assertEqual(len(split.X_train), 14)
        self.assertEqual(len(split.X_val), 3)
        self.assertEqual(len(split.X_test), 3)

        metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 4.0])
        self.assertAlmostEqual(metrics["mae_c"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse_c"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(metrics["bias_c"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
