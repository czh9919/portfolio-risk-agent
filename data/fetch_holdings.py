"""
M1 — Holdings consolidation from IBKR, Trading 212, eToro.

Each holding dict:
  ticker, platform, quantity, cost_basis_eur, market_value_eur,
  unrealised_pnl_eur, currency, weight (filled later)
"""
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_RETRY = 3
_TIMEOUT = 15

# Brokers sometimes keep stale tickers after corporate events (spin-offs, delistings,
# relistings).  Map old → current ticker so price fetching and risk calculations work.
_TICKER_REMAP: dict[str, str] = {
    "YNDX": "NBIS",   # Yandex → Nebius Group (relisted as NBIS on NASDAQ, Oct 2024)
}


def _get(url: str, headers: dict = None, retries: int = _RETRY) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers or {}, timeout=_TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"GET {url} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


def _parse_bond_symbol(symbol: str) -> dict:
    """Extract coupon rate and maturity from IBKR bond ticker strings.

    Handles formats like:
      'T 3 3/4 06/30/27'   → coupon=3.75,  maturity=06/30/27
      'T 4.5 12/31/28'     → coupon=4.5,   maturity=12/31/28
    """
    result = {}
    # Mixed fraction: "3 3/4 MM/DD/YY"
    m = re.search(r'(\d+)\s+(\d+)/(\d+)\s+(\d{2}/\d{2}/\d{2,4})', symbol)
    if m:
        result["coupon"]   = str(int(m.group(1)) + int(m.group(2)) / int(m.group(3)))
        result["maturity"] = m.group(4)
        return result
    # Decimal coupon: "4.5 MM/DD/YY"
    m2 = re.search(r'(\d+\.?\d*)\s+(\d{2}/\d{2}/\d{2,4})', symbol)
    if m2:
        result["coupon"]   = m2.group(1)
        result["maturity"] = m2.group(2)
    return result


# ── IBKR Flex Query ───────────────────────────────────────────────────────────

_IBKR_SEND = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
_IBKR_GET  = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement"

def fetch_ibkr(fx_rates: dict) -> list[dict]:
    token    = os.environ.get("IBKR_FLEX_TOKEN")
    query_id = os.environ.get("IBKR_QUERY_ID")
    if not token or not query_id:
        logger.warning("IBKR_FLEX_TOKEN / IBKR_QUERY_ID not set — skipping IBKR")
        return []

    r = _get(f"{_IBKR_SEND}?t={token}&q={query_id}&v=3")
    if r is None:
        return []

    try:
        root = ET.fromstring(r.text)
        ref  = root.findtext(".//ReferenceCode")
        if not ref:
            logger.error("IBKR: no ReferenceCode in response")
            return []
    except ET.ParseError as e:
        logger.error(f"IBKR: XML parse error on SendRequest: {e}")
        return []

    time.sleep(3)
    r2 = _get(f"{_IBKR_GET}?t={token}&q={ref}&v=3")
    if r2 is None:
        return []

    holdings = []
    try:
        root2 = ET.fromstring(r2.text)
        for pos in root2.findall(".//OpenPosition"):
            asset_cat = pos.get("assetCategory", "")
            symbol    = pos.get("symbol", "")
            # If assetCategory absent, infer bond from ticker format (spaces/slashes = bond notation)
            is_bond   = (asset_cat == "BOND") or (
                            not asset_cat and (" " in symbol or "/" in symbol))

            # Skip options, futures, warrants, etc. — but keep BOND
            if asset_cat and asset_cat not in ("STK", "ETF", "FUND", "BOND"):
                logger.info(f"IBKR: skipping {symbol} (assetCategory={asset_cat})")
                continue
            # For equities only: skip malformed tickers (spaces/slashes indicate non-equity)
            if not is_bond and (" " in symbol or "/" in symbol):
                logger.info(f"IBKR: skipping {symbol} (non-equity ticker format)")
                continue

            currency   = pos.get("currency", "USD")
            rate       = fx_rates.get(f"{currency}EUR", 1.0 if currency == "EUR" else 0.92)
            position   = float(pos.get("position", 0))
            cost       = float(pos.get("costBasisMoney", 0))

            # Prefer positionValue (direct market value) if the Flex Query includes it
            pos_value_str = pos.get("positionValue", "")
            if pos_value_str:
                mkt_val = float(pos_value_str)
            else:
                mark_price = float(pos.get("markPrice", 0))
                # Bonds: markPrice is clean price as % of par; position is face value
                mkt_val = (mark_price / 100) * position if is_bond else mark_price * position

            # Parse coupon & maturity from symbol string when XML fields are absent
            parsed = _parse_bond_symbol(symbol) if is_bond else {}
            maturity = pos.get("maturity") or parsed.get("maturity", "")
            coupon   = pos.get("coupon")   or parsed.get("coupon", "")

            holdings.append({
                "ticker":            symbol,
                "description":       pos.get("description", symbol),
                "asset_class":       "bond" if is_bond else "equity",
                "maturity":          maturity,
                "coupon":            coupon,
                "platform":          "IBKR",
                # openDateTime is present only when the Flex Query includes the
                # "Open Date/Time" field; the ledger backfills it otherwise.
                "entry_date":        pos.get("openDateTime") or pos.get("holdingPeriodDateTime", ""),
                "quantity":          position,
                "cost_basis_eur":    cost * rate,
                "market_value_eur":  mkt_val * rate,
                "unrealised_pnl_eur":(mkt_val - cost) * rate,
                "currency":          currency,
            })
    except Exception as e:
        logger.error(f"IBKR: parse positions failed: {e}")

    logger.info(f"IBKR: {len(holdings)} positions loaded")
    return holdings


# ── Trading 212 ───────────────────────────────────────────────────────────────

_T212_BASE = "https://live.trading212.com/api/v0"

_T212_TYPE_SUFFIXES = {"EQ", "ETF", "BOND", "CFD", "REIT"}

def _t212_ticker(raw: str) -> str:
    """Convert T212 ticker (e.g. AAPL_US_EQ) to a clean yfinance ticker (AAPL)."""
    parts = raw.rsplit("_", 2)
    if parts[-1].upper() in _T212_TYPE_SUFFIXES:
        parts = parts[:-1]
    if len(parts) > 1 and re.match(r'^[A-Z]{2,3}$', parts[-1]):
        parts = parts[:-1]
    return "_".join(parts)

def fetch_t212(fx_rates: dict) -> list[dict]:
    import base64
    api_key    = os.environ.get("T212_API_KEY")
    api_secret = os.environ.get("T212_API_SECRET")
    if not api_key or not api_secret:
        logger.warning("T212_API_KEY / T212_API_SECRET not set — skipping Trading 212")
        return []

    credentials = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
    r = _get(f"{_T212_BASE}/equity/portfolio", headers={"Authorization": f"Basic {credentials}"})
    if r is None:
        return []

    holdings = []
    try:
        for pos in r.json():
            mv    = float(pos.get("currentPrice", 0)) * float(pos.get("quantity", 0))
            cost  = float(pos.get("averagePrice", 0)) * float(pos.get("quantity", 0))
            holdings.append({
                "ticker":            _t212_ticker(pos.get("ticker", "")),
                "description":       _t212_ticker(pos.get("ticker", "")),
                "asset_class":       "equity",
                "platform":          "T212",
                "entry_date":        pos.get("initialFillDate", ""),
                "quantity":          float(pos.get("quantity", 0)),
                "cost_basis_eur":    cost,
                "market_value_eur":  mv,
                "unrealised_pnl_eur":float(pos.get("ppl", 0)),
                "currency":          "EUR",
            })
    except Exception as e:
        logger.error(f"T212: parse failed: {e}")

    logger.info(f"T212: {len(holdings)} positions loaded")
    return holdings


# ── eToro ─────────────────────────────────────────────────────────────────────

_ETORO_BASE    = "https://www.etoro.com/sapi/trade-data-real/live/public/portfolios"
_ETORO_CSV     = "config/etoro_holdings.csv"

# eToro instrument names → yfinance-compatible tickers
# Smart Portfolios and commodities don't have exchange tickers of their own.
_ETORO_TICKER_MAP: dict[str, str] = {
    "MAG-7": "MAGS",   # eToro MAG-7 Smart Portfolio → Roundhill Magnificent Seven ETF
    "MAG7":  "MAGS",
    "GOLD":  "GLD",    # eToro Gold position → SPDR Gold Shares ETF
    "Gold":  "GLD",
}


def _load_etoro_csv(fx_rates: dict = None) -> list[dict]:
    """
    Load eToro holdings from config/etoro_holdings.csv.
    CSV stores quantity + cost_basis_eur; market value is computed live
    from yfinance so it stays current without manual updates.
    """
    import csv
    import yfinance as yf
    path = _ETORO_CSV
    fx = fx_rates or {}

    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        logger.info(f"eToro: {path} not found — no manual holdings")
        return []
    except Exception as e:
        logger.error(f"eToro: CSV parse failed: {e}")
        return []

    if not rows:
        return []

    # Batch-fetch latest prices + live FX rates in one request
    tickers = [r["ticker"].strip() for r in rows if r.get("ticker", "").strip()]
    fetch_symbols = tickers + ["EURUSD=X"]   # EUR/USD for USD→EUR conversion
    try:
        raw = yf.download(fetch_symbols, period="2d", auto_adjust=True, progress=False)
        if hasattr(raw.columns, "levels"):
            close_df = raw["Close"]
        else:
            close_df = raw[["Close"]].rename(columns={"Close": fetch_symbols[0]})
        latest = {t: float(close_df[t].dropna().iloc[-1])
                  for t in fetch_symbols if t in close_df.columns and not close_df[t].dropna().empty}

        # Derive live FX rates from the batch (override caller's values if fresher)
        if "EURUSD=X" in latest:
            fx = dict(fx)   # don't mutate caller's dict
            fx["USDEUR"] = 1.0 / latest["EURUSD=X"]
            logger.debug(f"eToro: live USD/EUR = {fx['USDEUR']:.4f}")
    except Exception as e:
        logger.warning(f"eToro: live price fetch failed ({e}) — market values will be zero")
        latest = {}

    holdings = []
    for row in rows:
        ticker = row.get("ticker", "").strip()
        if not ticker:
            continue
        currency     = row.get("currency", "USD").strip()
        rate         = fx.get(f"{currency}EUR", 0.92 if currency == "USD" else 1.0)
        qty          = float(row.get("quantity", 0))
        cost_eur     = float(row.get("cost_basis_eur", 0))
        price_native = latest.get(ticker)

        if price_native is not None:
            mv_eur = price_native * rate * qty
        else:
            logger.warning(f"eToro CSV: no live price for {ticker} — market value unavailable")
            mv_eur = 0.0

        holdings.append({
            "ticker":            ticker,
            "description":       row.get("description", ticker).strip(),
            "asset_class":       row.get("asset_class", "equity").strip(),
            "platform":          "eToro",
            "entry_date":        (row.get("entry_date") or "").strip(),
            "quantity":          qty,
            "cost_basis_eur":    cost_eur,
            "market_value_eur":  mv_eur,
            "unrealised_pnl_eur":mv_eur - cost_eur,
            "currency":          currency,
        })

    logger.info(f"eToro: {len(holdings)} positions loaded from {path} (live prices)")
    return holdings


def fetch_etoro(fx_rates: dict) -> list[dict]:
    key = os.environ.get("ETORO_API_KEY")
    if not key:
        logger.info("ETORO_API_KEY not set — loading eToro from CSV fallback")
        return _load_etoro_csv(fx_rates)

    r = _get(_ETORO_BASE, headers={"AccountType": "Real", "Authorization": f"Bearer {key}"})
    if r is None:
        logger.warning("eToro API unreachable — falling back to CSV")
        return _load_etoro_csv(fx_rates)

    holdings = []
    try:
        for pos in r.json().get("PublicPortfolio", {}).get("Positions", []):
            currency = "USD"
            rate     = fx_rates.get("USDEUR", 0.92)
            mv       = float(pos.get("NetValue", 0)) * rate
            cost     = float(pos.get("OpenRate", 0)) * float(pos.get("Units", 0)) * rate
            raw_sym  = pos.get("Instrument", {}).get("SymbolFull", "")
            ticker   = _ETORO_TICKER_MAP.get(raw_sym, raw_sym)
            holdings.append({
                "ticker":            ticker,
                "description":       raw_sym,
                "asset_class":       "equity",
                "platform":          "eToro",
                "entry_date":        pos.get("OpenDateTime", ""),
                "quantity":          float(pos.get("Units", 0)),
                "cost_basis_eur":    cost,
                "market_value_eur":  mv,
                "unrealised_pnl_eur":mv - cost,
                "currency":          currency,
            })
    except Exception as e:
        logger.error(f"eToro API parse failed: {e} — falling back to CSV")
        return _load_etoro_csv(fx_rates)

    if not holdings:
        logger.warning("eToro API returned empty portfolio — falling back to CSV")
        return _load_etoro_csv(fx_rates)

    logger.info(f"eToro: {len(holdings)} positions loaded from API")
    return holdings


# ── Merge ─────────────────────────────────────────────────────────────────────

def fetch_all_holdings(fx_rates: dict) -> list[dict]:
    """Fetch from all three platforms, merge, compute weights."""
    holdings = fetch_ibkr(fx_rates) + fetch_t212(fx_rates) + fetch_etoro(fx_rates)

    # Apply ticker remapping for corporate events (delistings, relistings, renames)
    for h in holdings:
        old = h.get("ticker", "")
        new = _TICKER_REMAP.get(old)
        if new:
            logger.info(f"Remapping {old} → {new} ({h.get('platform', '?')})")
            h["ticker"] = new
            # Only update description when it echoes the old ticker (broker-provided name)
            if h.get("description") == old:
                h["description"] = new

    # Record / backfill entry dates (broker open date wins; else first-seen)
    try:
        from data import entry_dates
        entry_dates.enrich(holdings)
    except Exception as e:
        logger.warning(f"entry-date enrichment skipped: {e}")

    total_nav = sum(h["market_value_eur"] for h in holdings)
    for h in holdings:
        h["weight"] = h["market_value_eur"] / total_nav if total_nav else 0.0

    logger.info(f"Total holdings: {len(holdings)}, NAV: €{total_nav:,.2f}")
    return holdings
