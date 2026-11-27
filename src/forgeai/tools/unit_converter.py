"""Deterministic engineering, maritime, mass, distance, pressure, and temperature unit converter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Canonical conversion factors to SI base per category
_CATEGORIES: dict[str, dict[str, float]] = {
    "mass": {
        "kg": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "g": 0.001,
        "gram": 0.001,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
        "mt": 1000.0,
        "tonne": 1000.0,
        "tonnes": 1000.0,
    },
    "distance": {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "mi": 1609.344,
        "mile": 1609.344,
        "miles": 1609.344,
        "nm": 1852.0,
        "nmi": 1852.0,
        "nautical_mile": 1852.0,
        "ft": 0.3048,
        "feet": 0.3048,
        "mm": 0.001,
    },
    "pressure": {
        "bar": 1.0,
        "psi": 0.06894757,
        "kpa": 0.01,
        "mpa": 10.0,
    },
    "speed": {
        "knots": 1.0,
        "knot": 1.0,
        "kt": 1.0,
        "kmh": 0.539957,
        "km/h": 0.539957,
        "mph": 0.868976,
        "mps": 1.943844,
        "m/s": 1.943844,
    },
    "volume": {
        "l": 1.0,
        "liter": 1.0,
        "liters": 1.0,
        "gal": 3.785411784,
        "gallon": 3.785411784,
        "gallons": 3.785411784,
        "m3": 1000.0,
    },
}


def _convert_temp(val: float, src: str, dst: str) -> float:
    s = src.upper().replace("°", "")
    d = dst.upper().replace("°", "")
    c_val = val if s == "C" else ((val - 32.0) * 5.0 / 9.0 if s == "F" else val - 273.15)
    if d == "C":
        return c_val
    if d == "F":
        return c_val * 9.0 / 5.0 + 32.0
    if d == "K":
        return c_val + 273.15
    raise ValueError(f"Unsupported temperature unit '{dst}'")


def run_unit_converter(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Convert engineering, weight, pressure, distance, speed, or temperature values."""
    try:
        value = float(args.get("value"))
    except (TypeError, ValueError):
        return json.dumps({"error": "Missing or non-numeric 'value' argument."})

    from_unit = str(args.get("from_unit") or "").strip().lower()
    to_unit = str(args.get("to_unit") or "").strip().lower()
    if not from_unit or not to_unit:
        return json.dumps({"error": "Provide both 'from_unit' and 'to_unit'."})

    if from_unit.replace("°", "") in {"c", "f", "k"} and to_unit.replace("°", "") in {"c", "f", "k"}:
        out = round(_convert_temp(value, from_unit, to_unit), 4)
        return json.dumps({"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": out})

    for cat, table in _CATEGORIES.items():
        if from_unit in table and to_unit in table:
            base_val = value * table[from_unit]
            converted = base_val / table[to_unit]
            return json.dumps(
                {
                    "category": cat,
                    "value": value,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "result": round(converted, 4),
                }
            )

    return json.dumps({"error": f"Incompatible or unsupported conversion: '{from_unit}' -> '{to_unit}'."})
