"""Current date and date-arithmetic tool."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def run_current_date(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Return today's date, day of week, and optional date offset/difference calculations."""
    today_local = datetime.now().date()
    now_utc = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "local_date": today_local.isoformat(),
        "utc_datetime": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "day_of_week": today_local.strftime("%A"),
        "iso_week": today_local.isocalendar().week,
    }

    if "add_days" in args and args["add_days"] is not None:
        base_str = str(args.get("base_date") or today_local.isoformat())
        base = date.fromisoformat(base_str[:10])
        offset = int(args["add_days"])
        target = base + timedelta(days=offset)
        payload["target_date"] = target.isoformat()
        payload["target_day_of_week"] = target.strftime("%A")

    if args.get("start_date") and args.get("end_date"):
        d1 = date.fromisoformat(str(args["start_date"])[:10])
        d2 = date.fromisoformat(str(args["end_date"])[:10])
        payload["days_between"] = (d2 - d1).days

    return json.dumps(payload)
