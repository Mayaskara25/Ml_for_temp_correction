from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


CANONICAL_FIELDS = [
    "Timestamp_ms",
    "Live_K_Temp_C",
    "Synthetic_K_Temp_C",
    "Live_PT100_Temp_C",
    "Synthetic_PT100_Temp_C",
    "Corrected_Temp_C",
    "K_Sensor_OK",
    "PT100_Sensor_OK",
    "Source",
    "Schema",
]


@dataclass(frozen=True)
class SensorRow:
    timestamp_ms: int
    live_k_temp_c: Optional[float]
    synthetic_k_temp_c: Optional[float]
    live_pt100_temp_c: Optional[float]
    synthetic_pt100_temp_c: Optional[float]
    corrected_temp_c: Optional[float]
    k_sensor_ok: Optional[bool]
    pt100_sensor_ok: Optional[bool]
    source: str
    schema: str

    def to_csv_fields(self) -> list[str]:
        return [
            str(self.timestamp_ms),
            _format_optional_float(self.live_k_temp_c),
            _format_optional_float(self.synthetic_k_temp_c),
            _format_optional_float(self.live_pt100_temp_c),
            _format_optional_float(self.synthetic_pt100_temp_c),
            _format_optional_float(self.corrected_temp_c),
            _format_optional_bool(self.k_sensor_ok),
            _format_optional_bool(self.pt100_sensor_ok),
            self.source,
            self.schema,
        ]


def parse_serial_line(line: str, default_source: str = "hardware") -> Optional[SensorRow]:
    stripped = line.strip()
    if not stripped or stripped.startswith("["):
        return None

    parts = [part.strip() for part in stripped.split(",")]
    if not parts or not _looks_numeric(parts[0]):
        return None

    try:
        if len(parts) == 8:
            return SensorRow(
                timestamp_ms=int(float(parts[0])),
                live_k_temp_c=float(parts[1]),
                synthetic_k_temp_c=float(parts[2]),
                live_pt100_temp_c=float(parts[3]),
                synthetic_pt100_temp_c=float(parts[4]),
                corrected_temp_c=float(parts[5]),
                k_sensor_ok=_parse_bool(parts[6]),
                pt100_sensor_ok=_parse_bool(parts[7]),
                source=default_source,
                schema="readme_8_field",
            )

        if len(parts) == 5:
            return SensorRow(
                timestamp_ms=int(float(parts[0])),
                live_k_temp_c=float(parts[1]),
                synthetic_k_temp_c=None,
                live_pt100_temp_c=None,
                synthetic_pt100_temp_c=float(parts[2]),
                corrected_temp_c=float(parts[3]),
                k_sensor_ok=_parse_bool(parts[4]),
                pt100_sensor_ok=None,
                source=default_source,
                schema="legacy_max6675_5_field",
            )

        if len(parts) == 4:
            k_temp = float(parts[1])
            pt100_temp = float(parts[2])
            return SensorRow(
                timestamp_ms=int(float(parts[0])),
                live_k_temp_c=k_temp,
                synthetic_k_temp_c=k_temp,
                live_pt100_temp_c=pt100_temp,
                synthetic_pt100_temp_c=pt100_temp,
                corrected_temp_c=float(parts[3]),
                k_sensor_ok=True,
                pt100_sensor_ok=True,
                source=default_source,
                schema="legacy_synthetic_4_field",
            )
    except ValueError:
        return None

    return None


def dynamic_error(row: SensorRow) -> Optional[float]:
    reference = row.live_pt100_temp_c
    if reference is None or row.pt100_sensor_ok is False:
        reference = row.synthetic_pt100_temp_c
    if reference is None or row.corrected_temp_c is None:
        return None
    return row.corrected_temp_c - reference


def sensor_status_text(status: Optional[bool], fallback: str = "N/A") -> str:
    if status is True:
        return "OK"
    if status is False:
        return "FALLBACK"
    return fallback


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_bool(value: str) -> bool:
    return value.strip() in {"1", "true", "True", "OK", "ok"}


def _format_optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_optional_bool(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "1" if value else "0"
