"""
Alpaca Markets paper trading client.

Paper base URL: https://paper-api.alpaca.markets
Data base URL:  https://data.alpaca.markets

Free tier: unlimited paper trading, 200 req/min.
Required env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
Optional env var:  PAPER_DRY_RUN=true  (log orders, skip execution)
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

_PAPER_BASE = "https://paper-api.alpaca.markets"
_DATA_BASE  = "https://data.alpaca.markets"
_TIMEOUT    = 10


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
        "accept":              "application/json",
        "content-type":        "application/json",
    }


def is_configured() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def is_market_open() -> bool:
    """Return True if the US equity market is currently open per Alpaca clock."""
    try:
        r = requests.get(f"{_PAPER_BASE}/v2/clock", headers=_headers(), timeout=_TIMEOUT)
        r.raise_for_status()
        return bool(r.json().get("is_open", False))
    except Exception as e:
        logger.warning(f"Alpaca clock check failed ({e}) — assuming open to avoid blocking")
        return True


# ── Account ───────────────────────────────────────────────────────────────────

def get_account() -> dict:
    r = requests.get(f"{_PAPER_BASE}/v2/account", headers=_headers(), timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions() -> dict[str, dict]:
    """Return {symbol: position_dict} for all open paper positions."""
    r = requests.get(f"{_PAPER_BASE}/v2/positions", headers=_headers(), timeout=_TIMEOUT)
    r.raise_for_status()
    return {p["symbol"]: p for p in r.json()}


def close_position(symbol: str) -> dict | None:
    """Close entire position. Returns None if ticker not held."""
    r = requests.delete(
        f"{_PAPER_BASE}/v2/positions/{symbol}",
        headers=_headers(), timeout=_TIMEOUT,
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


# ── Orders ────────────────────────────────────────────────────────────────────

def get_open_orders() -> list[dict]:
    r = requests.get(
        f"{_PAPER_BASE}/v2/orders",
        headers=_headers(), params={"status": "open"}, timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def cancel_all_orders() -> None:
    requests.delete(f"{_PAPER_BASE}/v2/orders", headers=_headers(), timeout=_TIMEOUT)


def place_order(
    symbol:     str,
    qty:        float,
    side:       str,           # "buy" | "sell"
    order_type: str = "market",
    tif:        str = "day",
) -> dict:
    """Place a market order supporting fractional shares. Raises on HTTP error."""
    qty_str = f"{qty:.6g}"   # "10" for whole shares, "2.5" for fractional
    body = {
        "symbol":        symbol,
        "qty":           qty_str,
        "side":          side,
        "type":          order_type,
        "time_in_force": tif,
    }
    r = requests.post(
        f"{_PAPER_BASE}/v2/orders",
        json=body, headers=_headers(), timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ── Market data (Alpaca free tier) ────────────────────────────────────────────

def get_latest_price(symbol: str) -> float | None:
    """Latest trade price from Alpaca data API. Returns None on failure."""
    try:
        r = requests.get(
            f"{_DATA_BASE}/v2/stocks/{symbol}/trades/latest",
            headers=_headers(), timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        return float(r.json().get("trade", {}).get("p") or 0) or None
    except Exception:
        return None
