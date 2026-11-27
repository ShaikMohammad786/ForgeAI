"""Keyless public HTTP API tool using Open-Meteo (geocoding + forecast)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


def run_open_meteo_weather(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Fetch current & forecast weather for a city or port via Open-Meteo."""
    location = str(args.get("location") or args.get("city") or "").strip()
    if not location:
        return json.dumps({"error": "Missing 'location' argument."})

    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=8.0,
        )
        geo.raise_for_status()
        geo_data = geo.json()
        results = geo_data.get("results") or []
        if not results:
            return json.dumps({"error": f"Location '{location}' not found on Open-Meteo."})

        place = results[0]
        lat = place["latitude"]
        lon = place["longitude"]

        wx = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=8.0,
        )
        wx.raise_for_status()
        wx_data = wx.json()
        return json.dumps(
            {
                "resolved_location": f"{place.get('name')}, {place.get('country', '')}".strip(", "),
                "latitude": lat,
                "longitude": lon,
                "current": wx_data.get("current", {}),
                "daily_3day": wx_data.get("daily", {}),
            }
        )
    except Exception as exc:
        return json.dumps({"location": location, "error": f"Open-Meteo request failed: {exc}"})
