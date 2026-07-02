import argparse
import csv
from pathlib import Path

import serial

from calibration_io import CANONICAL_FIELDS, parse_serial_line


SERIAL_PORT = "COM3"
BAUD_RATE = 115200
OUTPUT_FILE = "dataset.csv"


def start_logging(serial_port: str, baud_rate: int, output_file: str, source: str) -> None:
    output_path = Path(output_file)
    rows_written = 0
    ignored_rows = 0
    ser = None

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Connected to {serial_port} at {baud_rate} baud.")
        print(f"Writing canonical data to {output_path}... (Press Ctrl+C to stop)")

        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CANONICAL_FIELDS)

            while True:
                if ser.in_waiting <= 0:
                    continue

                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                row = parse_serial_line(raw_line, default_source=source)
                if row is None:
                    ignored_rows += 1
                    print(f"ignored: {raw_line}")
                    continue

                writer.writerow(row.to_csv_fields())
                file.flush()
                rows_written += 1
                print(",".join(row.to_csv_fields()))

    except KeyboardInterrupt:
        print(
            f"\nLogging stopped. Wrote {rows_written} rows to {output_path}; "
            f"ignored {ignored_rows} non-data rows."
        )
    except serial.SerialException as exc:
        print(f"\nSerial Error: {exc}")
        print("Check if the COM port is correct and not open in the Arduino IDE.")
    finally:
        if ser is not None and ser.is_open:
            ser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log ESP32 temperature calibration rows to a canonical CSV."
    )
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument(
        "--source",
        default="hardware",
        help="Source label written to each row. Use synthetic only for simulated data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_logging(args.port, args.baud, args.output, args.source)
