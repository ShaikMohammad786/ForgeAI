"""Keyless public REST Countries API tool for ISO codes, currencies, capitals, and timezones."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import httpx


def run_country_info(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Query RestCountries public keyless API for country metadata."""
    country = str(args.get("country") or args.get("name") or "").strip()
    if not country:
        return json.dumps({"error": "Missing 'country' argument."})

    encoded = urllib.parse.quote(country)
    url = (
        f"https://restcountries.com/v3.1/name/{encoded}"
        "?fields=name,cca2,cca3,capital,region,subregion,currencies,timezones,idd"
    )
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                return json.dumps({"error": f"Country '{country}' not found."})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        return json.dumps({"error": f"RestCountries API lookup failed: {exc}"})

    if not isinstance(payload, list) or not payload:
        return json.dumps({"error": f"No country data returned for '{country}'."})

    top = payload[0]
    name_obj = top.get("name") or {}
    currencies = top.get("currencies") or {}
    idd = top.get("idd") or {}
    calling_root = idd.get("root") or ""
    suffixes = idd.get("suffixes") or []
    calling_code = f"{calling_root}{suffixes[0]}" if calling_root and suffixes else calling_root

    return json.dumps(
        {
            "common_name": name_obj.get("common"),
            "official_name": name_obj.get("official"),
            "iso_alpha2": top.get("cca2"),
            "iso_alpha3": top.get("cca3"),
            "capital": (top.get("capital") or [None])[0],
            "region": top.get("region"),
            "subregion": top.get("subregion"),
            "currencies": {
                code: meta.get("name") for code, meta in currencies.items() if isinstance(meta, dict)
            },
            "calling_code": calling_code,
            "timezones": (top.get("timezones") or [])[:4],
        }
    )
