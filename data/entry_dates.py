"""
Persistent entry-date ledger for real holdings.

Brokers don't always expose a position's open date, and the execution /
laddering analysis needs an anchor for the cost-basis VWAP. This records, per
(platform, ticker), the earliest known entry date:

  • the broker-provided open date when the API supplies one (authoritative), or
  • the first date we observed the position (first-seen fallback).

Stored in cache/ so it is restored across ephemeral CI runs by the existing
actions/cache step. Broker dates are the primary source and backfill the
ledger whenever they appear, so an evicted cache self-heals on the next run.
"""
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LEDGER_PATH = Path("cache/entry_dates.json")


def _key(platform: str, ticker: str) -> str:
    return f"{platform or '?'}|{ticker}"


def normalize_date(raw) -> Optional[str]:
    """Coerce a broker date field to an ISO 'YYYY-MM-DD' string, or None."""
    if raw is None:
        return None
    if isinstance(raw, (date, datetime)):
        return raw.date().isoformat() if isinstance(raw, datetime) else raw.isoformat()
    s = str(raw).strip()
    if not s:
        return None
    # IBKR compact form: 20240115 or 20240115;143000
    m = re.match(r"^(\d{4})(\d{2})(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    # ISO / ISO-datetime (T212 initialFillDate, eToro OpenDateTime)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None


def load() -> dict:
    if not LEDGER_PATH.exists():
        return {}
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"entry_dates ledger corrupt: {exc} — starting fresh")
        return {}


def save(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(LEDGER_PATH)


def enrich(holdings: list[dict]) -> None:
    """
    Fill each holding's `entry_date` (ISO string) and persist the ledger.

    A broker-supplied date is authoritative: it is normalized, written onto the
    holding, and backfilled into the ledger. Otherwise the stored first-seen
    date is used; if none exists, today is recorded as first-seen.
    """
    if not holdings:
        return
    ledger = load()
    today = date.today().isoformat()
    changed = False

    for h in holdings:
        key = _key(h.get("platform", ""), h.get("ticker", ""))
        broker_date = normalize_date(h.get("entry_date"))

        if broker_date:
            if ledger.get(key) != broker_date:
                ledger[key] = broker_date
                changed = True
            h["entry_date"] = broker_date
        elif key in ledger:
            h["entry_date"] = ledger[key]
        else:
            ledger[key] = today
            h["entry_date"] = today
            changed = True

    if changed:
        try:
            save(ledger)
            logger.info(f"entry_dates ledger updated ({len(ledger)} positions tracked)")
        except OSError as exc:
            logger.warning(f"entry_dates ledger save failed: {exc}")
