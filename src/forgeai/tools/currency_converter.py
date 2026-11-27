"""Keyless live FX currency conversion via Frankfurter (ECB reference rates)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


def run_currency_converter(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Convert amounts between ISO currencies using live keyless Frankfurter ECB rates."""
    try:
        amount = float(args.get("amount", 1.0))
    except (TypeError, ValueError):
        return json.dumps({"error": "Invalid numeric 'amount'."})

    src = str(args.get("from_currency") or args.get("from") or "USD").strip().upper()
    dst = str(args.get("to_currency") or args.get("to") or "EUR").strip().upper()
    if len(src) != 3 or len(dst) != 3:
        return json.dumps({"error": "Currencies must be 3-letter ISO codes (e.g. USD, EUR, SGD, GBP)."})

    if src == dst:
        return json.dumps({"amount": amount, "from": src, "to": dst, "rate": 1.0, "converted": amount})

    try:
        resp = httpx.get(
            "https://api.frankfurter.app/latest",
            params={"amount": amount, "from": src, "to": dst},
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates") or {}
        if dst not in rates:
            return json.dumps({"error": f"Currency '{dst}' not supported by Frankfurter API."})
        converted = float(rates[dst])
        rate = round(converted / amount, 6) if amount else 0.0
        return json.dumps(
            {
                "amount": amount,
                "from": src,
                "to": dst,
                "date": data.get("date"),
                "rate": rate,
                "converted": round(converted, 4),
            }
        )
    except Exception as exc:
        return json.dumps({"error": f"FX lookup failed: {exc}"})
