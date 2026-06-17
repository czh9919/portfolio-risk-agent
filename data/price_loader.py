"""
M2 — Historical price data via yfinance (batch primary) + Massive REST (fallback).

Fetch strategy:
  1. yfinance batch download — all tickers in a single HTTP request (fast, free)
  2. Massive REST — per-ticker fallback for anything yfinance can't serve,
     paced at ≤5 req/min to respect the Stocks Basic plan limit

Availability flags (PRD §6.1):
  ok          ≥252 days → full window
  reduced     21-251 days → use actual, flag [{N}d window]
  excluded    <21 days → skip from VaR/vol calcs, include in NAV/P&L
  unavailable fetch failed
  not_found   ticker not on Yahoo Finance
"""
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

STALE_PRICE_DAYS = 4   # last close must be ≤4 calendar days old

logger = logging.getLogger(__name__)

FX_TICKERS = {
    "USDEUR": "EURUSD=X",   # invert
    "GBPEUR": "EURGBP=X",   # invert
    "AUDEUR": "EURAUD=X",   # invert
}

# 5 req/min plan → 12 s minimum gap; add 1 s margin
_MASSIVE_MIN_INTERVAL = 13.0


@dataclass
class PriceData:
    ticker:  str
    closes:  Optional[pd.Series]    # daily closing prices
    returns: Optional[pd.Series]    # daily pct returns
    days:    int  = 0
    status:  str  = "ok"            # ok | reduced | excluded | unavailable | not_found
    flag:    str  = ""              # shown in report footer


# ── Public entry point ────────────────────────────────────────────────────────

def load_prices(tickers: list[str], days: int = 252) -> dict[str, PriceData]:
    """
    Batch-fetch all tickers via yfinance; fall back to Massive REST for
    any ticker yfinance cannot serve (paced at 5 req/min).
    """
    if not tickers:
        return {}

    end   = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    # ── Step 1: yfinance batch (one HTTP request for all tickers) ─────────────
    batch: dict[str, Optional[pd.Series]] = _fetch_yfinance_batch(tickers, start, end)

    # ── Step 2: Massive fallback for tickers that yfinance couldn't serve ─────
    has_massive = bool(os.environ.get("MASSIVE_API_KEY"))
    failed      = [t for t in tickers if batch.get(t) is None]
    if has_massive and failed:
        logger.info(f"Massive fallback for {len(failed)} tickers: {failed}")
        for i, ticker in enumerate(failed):
            if i > 0:
                time.sleep(_MASSIVE_MIN_INTERVAL)
            batch[ticker] = _try_massive(ticker, days)

    # ── Step 3: classify each result ──────────────────────────────────────────
    return {ticker: _classify(ticker, batch.get(ticker), days) for ticker in tickers}


# ── yfinance helpers ──────────────────────────────────────────────────────────

def _fetch_yfinance_batch(
    tickers: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, Optional[pd.Series]]:
    """
    Download all tickers in a single yf.download() call.
    Returns {ticker: pd.Series of closes} or {ticker: None} on failure.
    """
    result: dict[str, Optional[pd.Series]] = {t: None for t in tickers}
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            logger.warning("yfinance batch: empty response")
            return result

        # Multi-ticker → MultiIndex columns; single ticker → flat columns
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"]
        else:
            close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})

        for ticker in tickers:
            if ticker not in close_df.columns:
                continue
            series = close_df[ticker].dropna()
            if series.empty:
                continue
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            result[ticker] = series
            logger.info(f"{ticker}: {len(series)} closes via yfinance batch")

    except Exception as exc:
        logger.error(f"yfinance batch failed: {exc} — will retry per-ticker via Massive")

    return result


# ── Massive REST fallback ─────────────────────────────────────────────────────

def _try_massive(ticker: str, days: int) -> Optional[pd.Series]:
    try:
        from data.massive import fetch_daily_closes
        return fetch_daily_closes(ticker, days)
    except Exception as exc:
        logger.debug(f"Massive {ticker}: {exc}")
        return None


# ── Classification ────────────────────────────────────────────────────────────

def _classify(ticker: str, closes: Optional[pd.Series], target_days: int) -> PriceData:
    if closes is None:
        return PriceData(ticker=ticker, closes=None, returns=None,
                         status="unavailable", flag="[Price data unavailable]")

    n = len(closes)

    if n > 0:
        last_date = pd.Timestamp(closes.index[-1]).date()
        days_old  = (date.today() - last_date).days
        if days_old > STALE_PRICE_DAYS:
            logger.warning(f"{ticker}: last close {last_date} is {days_old}d old")

    returns = closes.pct_change().dropna()

    if n < 21:
        logger.warning(f"{ticker}: only {n} days — excluded from VaR")
        return PriceData(ticker=ticker, closes=closes, returns=returns, days=n,
                         status="excluded", flag="[Excluded from VaR]")
    elif n < target_days:
        return PriceData(ticker=ticker, closes=closes.tail(n), returns=returns.tail(n),
                         days=n, status="reduced", flag=f"[{n}d window]")
    else:
        closes_  = closes.tail(target_days)
        returns_ = closes_.pct_change().dropna()
        return PriceData(ticker=ticker, closes=closes_, returns=returns_,
                         days=len(closes_), status="ok")


# ── OHLCV (full bars for execution analysis) ───────────────────────────────────

def load_ohlcv(tickers: list[str], days: int = 252) -> dict[str, pd.DataFrame]:
    """
    Batch-fetch full OHLCV bars (Open/High/Low/Close/Volume) via yfinance.

    Separate from load_prices(): the execution-analysis layer needs High/Low
    (for ATR) and Volume (for VWAP, HVN, ADV), which the close-only PriceData
    path discards. Returns {ticker: DataFrame} for every ticker that resolved;
    tickers yfinance can't serve are simply absent from the dict.
    """
    if not tickers:
        return {}

    end   = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    cols  = ["Open", "High", "Low", "Close", "Volume"]
    out: dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True, group_by="ticker",
        )
    except Exception as exc:
        logger.error(f"load_ohlcv: yfinance batch failed: {exc}")
        return out

    if raw is None or raw.empty:
        logger.warning("load_ohlcv: empty response")
        return out

    multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            if multi:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker]
            else:
                df = raw  # single ticker → flat columns
            df = df[[c for c in cols if c in df.columns]].dropna(how="all")
            if df.empty or "Close" not in df.columns:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            out[ticker] = df.tail(days)
        except Exception as exc:
            logger.debug(f"load_ohlcv {ticker}: {exc}")

    logger.info(f"load_ohlcv: {len(out)}/{len(tickers)} tickers with OHLCV bars")
    return out


# ── FX rates ──────────────────────────────────────────────────────────────────

def load_fx_rates() -> dict[str, float]:
    """Return {USDEUR: float, GBPEUR: float, ...} — all rates to EUR."""
    defaults = {"USDEUR": 0.92, "GBPEUR": 1.17, "AUDEUR": 0.59}
    rates = dict(defaults)
    for pair, yticker in FX_TICKERS.items():
        try:
            df = yf.Ticker(yticker).history(period="5d", auto_adjust=True)
            if df.empty:
                logger.warning(f"FX {pair}: empty response — using stale default {defaults[pair]:.4f}")
                continue
            last_date = pd.Timestamp(df.index[-1]).date()
            days_old  = (date.today() - last_date).days
            if days_old > STALE_PRICE_DAYS:
                logger.warning(
                    f"FX {pair}: rate from {last_date} ({days_old}d ago) — "
                    f"possible weekend/holiday, using it but flag as stale"
                )
            raw = float(df["Close"].iloc[-1])
            rates[pair] = 1.0 / raw   # EUR/XXX → XXX/EUR
        except Exception as e:
            logger.warning(f"FX {pair}: fetch failed, using default {defaults[pair]:.4f} — {e}")
    return rates
