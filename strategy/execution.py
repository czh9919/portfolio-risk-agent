"""
Execution / exit-laddering analysis.

For each stock (held or on the watchlist) this produces a trade-execution plan
rather than a buy/sell *decision* — the decision comes from the FF5 factor model.
The plan answers "if I am going to scale into or out of this name, at what prices,
in what size, and at what pace?".

Components (per ticker):
  • Dual Anchored VWAP — a fair-value reference anchored at (a) the last earnings
    date and (b) for held names, the date price last crossed average cost (an
    entry proxy, since brokers here don't expose a fill date). Current price vs
    anchor → REDUCE/OBSERVE (exits) or ACCUMULATE/WAIT (entries).
  • ATR(14) — volatility unit used to space ladder rungs.
  • HVN (high-volume nodes) — volume-profile peaks; liquidity-thick prices that
    make natural limit levels.
  • Ladder — 4–6 rungs spaced rung_spacing × ATR, snapped to the nearest HVN,
    placed above price for exits / below for entries (Almgren-Chriss style split
    of one fill into several to trade timing risk against market impact).
  • ADV participation cap — max_shares/day = participation_pct × ADV; the hard
    constraint. days_to_exit = position / max_shares_per_day.
  • Options Max Pain + peak-gamma strike — nearest monthly expiry, as a timing
    reference (dealer hedging flow tends to gravitate toward these near OPEX).

Everything is best-effort and wrapped so a single bad ticker never breaks the
pipeline; missing inputs yield None fields, not exceptions.
"""
import contextlib
import logging
import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_yfinance():
    """Silence yfinance's internal ERROR spam for best-effort lookups.

    yfinance logs at ERROR for expected misses — ETFs with no earnings dates,
    bond CUSIP-style tickers with no option chain — even though we catch the
    empty result. Raise its loggers to CRITICAL for the duration of the call.
    """
    names = ("yfinance", "yfinance.utils", "yfinance.data", "yfinance.ticker")
    saved = {n: logging.getLogger(n).level for n in names}
    for n in names:
        logging.getLogger(n).setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)

# Defaults — overridden by config/settings.yaml::execution
DEFAULTS = {
    "atr_period":       14,
    "n_rungs":          5,      # 4–6 rungs
    "rung_spacing_atr": 0.5,    # 0.5 × ATR between rungs
    "hvn_bins":         24,     # price buckets for the volume profile
    "hvn_lookback":     120,    # trading days of profile history
    "hvn_top":          4,      # number of HVN peaks to keep
    "hvn_snap_atr":     0.5,    # snap a rung to an HVN within this × ATR
    "adv_window":       20,     # trading days for average daily volume
    "participation_pct": 0.08,  # ≤8% of ADV per day (5–10% band)
    "max_position_pct": 0.15,   # watchlist sizing: target ≤15% of NAV per name
    "options_enabled":  True,
    "risk_free_rate":   0.035,
}


# ── Price/volume primitives ─────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average True Range (Wilder), last value. Needs High/Low/Close."""
    if df is None or len(df) < period + 1:
        return None
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing ≈ EMA with alpha = 1/period
    val = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    return float(val) if val == val else None


def anchored_vwap(df: pd.DataFrame, anchor: Optional[date]) -> Optional[float]:
    """Volume-weighted average typical price from `anchor` to the last bar."""
    if df is None or anchor is None or "Volume" not in df.columns:
        return None
    seg = df[df.index >= pd.Timestamp(anchor)]
    if seg.empty or seg["Volume"].sum() <= 0:
        return None
    typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
    vwap = float((typical * seg["Volume"]).sum() / seg["Volume"].sum())
    return vwap if vwap == vwap else None


def volume_profile(df: pd.DataFrame, bins: int, lookback: int
                   ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Binned volume profile over the last `lookback` bars.

    Returns (edges, centres, mass) where mass[i] is the total volume traded with
    a close inside bucket i. Returns (None, None, None) when there isn't enough
    clean data to build a profile.
    """
    if df is None or "Volume" not in df.columns:
        return None, None, None
    seg = df.tail(lookback)
    closes, vols = seg["Close"].to_numpy(), seg["Volume"].to_numpy()
    mask = np.isfinite(closes) & np.isfinite(vols) & (vols > 0)
    closes, vols = closes[mask], vols[mask]
    if closes.size < bins:
        return None, None, None
    lo, hi = closes.min(), closes.max()
    if hi <= lo:
        return None, None, None
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(closes, edges) - 1, 0, bins - 1)
    binned = np.zeros(bins)
    np.add.at(binned, idx, vols)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return edges, centres, binned


def volume_profile_hvn(df: pd.DataFrame, bins: int, lookback: int,
                       top: int) -> list[float]:
    """High-volume-node price levels (bin centres), highest volume first."""
    edges, centres, binned = volume_profile(df, bins, lookback)
    if centres is None:
        return []
    order = np.argsort(binned)[::-1]
    return [float(centres[i]) for i in order[:top] if binned[i] > 0]


def _mass_at_price(price: float, edges: Optional[np.ndarray],
                   mass: Optional[np.ndarray]) -> float:
    """Volume mass of the profile bucket containing `price` (0.0 if out of range)."""
    if edges is None or mass is None or price is None:
        return 0.0
    i = int(np.clip(np.digitize([price], edges)[0] - 1, 0, len(mass) - 1))
    return float(mass[i])


def adv(df: pd.DataFrame, window: int) -> tuple[Optional[float], Optional[float]]:
    """(average daily share volume, average daily dollar volume) over `window`."""
    if df is None or "Volume" not in df.columns:
        return None, None
    seg = df.tail(window)
    vol = seg["Volume"].dropna()
    if vol.empty:
        return None, None
    adv_shares = float(vol.mean())
    adv_dollar = float((seg["Close"] * seg["Volume"]).dropna().tail(window).mean())
    return adv_shares, adv_dollar


def _cost_cross_date(df: pd.DataFrame, avg_cost: float) -> Optional[date]:
    """Most recent date the close crossed `avg_cost` — an entry-proxy anchor."""
    if df is None or avg_cost is None or avg_cost <= 0:
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    above = closes >= avg_cost
    crossings = above.ne(above.shift(1))
    crossings.iloc[0] = False   # first bar has no prior bar → not a crossing
    cross_idx = crossings[crossings].index
    if len(cross_idx) == 0:
        return None
    return pd.Timestamp(cross_idx[-1]).date()


# ── Ladder ───────────────────────────────────────────────────────────────────

def build_ladder(price: float, atr_val: float, hvn: list[float], side: str,
                 n_rungs: int, spacing_atr: float, snap_atr: float,
                 *,
                 edges: Optional[np.ndarray] = None,
                 mass: Optional[np.ndarray] = None,
                 ref_shares: Optional[float] = None,
                 max_shares_day: Optional[float] = None) -> list[dict]:
    """
    Generate n_rungs limit levels spaced spacing_atr × ATR from the current
    price (above for "exit"/sell, below for "entry"/buy), each snapped to the
    nearest HVN when one sits within snap_atr × ATR.

    When a volume profile (edges, mass) is supplied each rung is weighted by the
    liquidity at its price bucket, normalised across the ladder; otherwise weights
    are equal. With ref_shares the weight is turned into a share count, and with
    max_shares_day each rung is flagged over_cap when it exceeds the ADV cap.
    """
    if price is None or atr_val is None or atr_val <= 0 or n_rungs <= 0:
        return []
    step = spacing_atr * atr_val
    direction = 1.0 if side == "exit" else -1.0
    snap_tol = snap_atr * atr_val
    rungs: list[dict] = []
    for i in range(1, n_rungs + 1):
        base = price + direction * step * i
        if base <= 0:
            continue
        snapped = None
        if hvn:
            nearest = min(hvn, key=lambda h: abs(h - base))
            if abs(nearest - base) <= snap_tol:
                snapped = nearest
        level = snapped if snapped is not None else base
        rungs.append({
            "rung":    i,
            "price":   round(float(level), 2),
            "at_hvn":  snapped is not None,
            "dist_pct": round((level / price - 1.0) * 100.0, 2),
            "_mass":   _mass_at_price(level, edges, mass),
        })
    if not rungs:
        return []

    # Liquidity-weighted sizing, normalised across the ladder (equal if no mass)
    total_mass = sum(r["_mass"] for r in rungs)
    n = len(rungs)
    for r in rungs:
        w = (r["_mass"] / total_mass) if total_mass > 0 else (1.0 / n)
        r["weight"] = round(w, 4)
        if ref_shares and ref_shares > 0:
            sh = round(w * float(ref_shares))
            r["shares"] = int(sh)
            r["over_cap"] = bool(max_shares_day and sh > max_shares_day)
        else:
            r["shares"] = None
            r["over_cap"] = False
        del r["_mass"]
    return rungs


# ── Options: Max Pain + peak gamma ─────────────────────────────────────────────

def _bs_gamma(spot: float, strike: float, t_years: float, iv: float,
              r: float) -> float:
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return pdf / (spot * iv * math.sqrt(t_years))


def options_levels(ticker: str, spot: float, r: float) -> Optional[dict]:
    """
    Max Pain (OI-implied pin) and the peak-gamma strike (dealer gamma magnet)
    for the nearest monthly-ish expiry. Returns None when no chain is available.
    """
    try:
        import yfinance as yf
        with _quiet_yfinance():
            tk = yf.Ticker(ticker)
            expiries = list(tk.options or [])
            if not expiries:
                return None
            today = date.today()
            # Nearest expiry at least ~10 calendar days out (skip same-week noise)
            future = [e for e in expiries if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 10]
            expiry = future[0] if future else expiries[-1]
            chain = tk.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        if calls is None or puts is None or calls.empty or puts.empty:
            return None

        t_years = max((datetime.strptime(expiry, "%Y-%m-%d").date() - today).days, 1) / 365.0
        strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        call_oi = dict(zip(calls["strike"], calls["openInterest"].fillna(0)))
        put_oi  = dict(zip(puts["strike"],  puts["openInterest"].fillna(0)))

        # Max pain: strike minimising total intrinsic cash paid to option holders
        best_k, best_cash = None, float("inf")
        for k in strikes:
            cash = 0.0
            for ks, oi in call_oi.items():
                if ks < k:
                    cash += (k - ks) * oi
            for ks, oi in put_oi.items():
                if ks > k:
                    cash += (ks - k) * oi
            if cash < best_cash:
                best_cash, best_k = cash, k

        # Peak-gamma strike: largest gamma × OI concentration (dealer magnet)
        best_gk, best_gv = None, -1.0
        for df in (calls, puts):
            oi_col = df["openInterest"].fillna(0)
            iv_col = df["impliedVolatility"].fillna(0)
            for ks, oi, iv in zip(df["strike"], oi_col, iv_col):
                if oi <= 0 or iv <= 0:
                    continue
                gv = _bs_gamma(spot, float(ks), t_years, float(iv), r) * float(oi)
                if gv > best_gv:
                    best_gv, best_gk = gv, float(ks)

        return {
            "expiry":      expiry,
            "max_pain":    round(float(best_k), 2) if best_k is not None else None,
            "peak_gamma":  round(float(best_gk), 2) if best_gk is not None else None,
        }
    except Exception as exc:
        logger.debug(f"options_levels {ticker}: {exc}")
        return None


# ── Earnings anchor ─────────────────────────────────────────────────────────────

def last_earnings_date(ticker: str) -> Optional[date]:
    """Most recent *past* earnings date (anchor for the earnings VWAP)."""
    try:
        import yfinance as yf
        with _quiet_yfinance():
            tk = yf.Ticker(ticker)
            df = tk.get_earnings_dates(limit=16)
        if df is None or df.empty:
            return None
        idx = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None \
            else pd.to_datetime(df.index)
        past = [d.date() for d in idx if d.date() <= date.today()]
        return max(past) if past else None
    except Exception as exc:
        logger.debug(f"last_earnings_date {ticker}: {exc}")
        return None


# ── Top-level per-ticker analysis ───────────────────────────────────────────────

def analyze_execution(
    ticker: str,
    df: pd.DataFrame,
    *,
    side: str = "exit",
    position_shares: Optional[float] = None,
    avg_cost_native: Optional[float] = None,
    entry_date: Optional[date] = None,
    ref_shares: Optional[float] = None,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
    """
    Build the full execution plan for one ticker.

    Every stock gets BOTH a descending entry ladder (建仓, below price) and an
    ascending exit ladder (减仓, above price); `side` only marks which one is the
    primary intent ("exit" for held names, "entry" for watchlist BUY candidates).

    `ref_shares` sizes the ladders: for held names it defaults to position_shares
    (the exit ladder scales out the position); for watchlist names the caller
    passes a target share count (max_position_pct × NAV ÷ price). Returns a dict
    consumed by the report and (for the cap) the paper engine, or None if price
    data is too thin to say anything.
    """
    c = {**DEFAULTS, **(cfg or {})}
    if df is None or df.empty or "Close" not in df.columns:
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    price = float(closes.iloc[-1])

    atr_val = atr(df, c["atr_period"])
    edges, _centres, mass = volume_profile(df, c["hvn_bins"], c["hvn_lookback"])
    hvn = volume_profile_hvn(df, c["hvn_bins"], c["hvn_lookback"], c["hvn_top"])
    adv_shares, adv_dollar = adv(df, c["adv_window"])

    earn_anchor = last_earnings_date(ticker)
    vwap_earn = anchored_vwap(df, earn_anchor)
    # Prefer the recorded entry date; fall back to the cost-cross proxy.
    if entry_date is not None:
        cost_anchor, cost_anchor_src = entry_date, "entry_date"
    elif avg_cost_native:
        cost_anchor, cost_anchor_src = _cost_cross_date(df, avg_cost_native), "cost_cross"
    else:
        cost_anchor, cost_anchor_src = None, None
    vwap_cost = anchored_vwap(df, cost_anchor)

    # Verdict relative to the primary (earnings) anchor, cost anchor as backup
    ref_vwap = vwap_earn if vwap_earn is not None else vwap_cost
    verdict, prem_pct = _verdict(side, price, ref_vwap)

    # Participation cap
    pct = float(c["participation_pct"])
    max_shares_day = adv_shares * pct if adv_shares else None

    # Sizing reference: held → real position; watchlist → caller-supplied target
    up_ref = ref_shares if ref_shares is not None else position_shares
    down_ref = ref_shares if ref_shares is not None else position_shares

    # Both ladders, always. Up = 减仓 (sell, above), Down = 建仓 (buy, below).
    ladder_up = build_ladder(
        price, atr_val, hvn, "exit", c["n_rungs"],
        c["rung_spacing_atr"], c["hvn_snap_atr"],
        edges=edges, mass=mass, ref_shares=up_ref, max_shares_day=max_shares_day)
    ladder_down = build_ladder(
        price, atr_val, hvn, "entry", c["n_rungs"],
        c["rung_spacing_atr"], c["hvn_snap_atr"],
        edges=edges, mass=mass, ref_shares=down_ref, max_shares_day=max_shares_day)
    primary = "up" if side == "exit" else "down"

    days_to_exit = None
    if position_shares and max_shares_day and max_shares_day > 0:
        days_to_exit = round(float(position_shares) / max_shares_day, 1)

    opts = options_levels(ticker, price, c["risk_free_rate"]) if c["options_enabled"] else None

    return {
        "ticker":          ticker,
        "side":            side,
        "price":           round(price, 2),
        "atr":             round(atr_val, 2) if atr_val else None,
        "atr_pct":         round(atr_val / price * 100, 2) if atr_val else None,
        "vwap_earnings":   round(vwap_earn, 2) if vwap_earn else None,
        "earnings_anchor": earn_anchor.isoformat() if earn_anchor else None,
        "vwap_cost":       round(vwap_cost, 2) if vwap_cost else None,
        "cost_anchor":     cost_anchor.isoformat() if cost_anchor else None,
        "cost_anchor_src": cost_anchor_src,
        "avg_cost":        round(avg_cost_native, 2) if avg_cost_native else None,
        "verdict":         verdict,
        "premium_pct":     prem_pct,
        "hvn":             [round(h, 2) for h in hvn],
        "ladder_up":       ladder_up,
        "ladder_down":     ladder_down,
        "primary":         primary,
        "ref_shares":      int(up_ref) if up_ref else None,
        "adv_shares":      int(adv_shares) if adv_shares else None,
        "adv_dollar":      adv_dollar,
        "participation_pct": pct,
        "max_shares_day":  int(max_shares_day) if max_shares_day else None,
        "position_shares": position_shares,
        "days_to_exit":    days_to_exit,
        "options":         opts,
    }


def _verdict(side: str, price: float, ref_vwap: Optional[float]) -> tuple[str, Optional[float]]:
    if ref_vwap is None or ref_vwap <= 0:
        return ("OBSERVE" if side == "exit" else "WAIT"), None
    prem = round((price / ref_vwap - 1.0) * 100.0, 2)
    if side == "exit":
        return ("REDUCE" if price > ref_vwap else "OBSERVE"), prem
    return ("ACCUMULATE" if price < ref_vwap else "WAIT"), prem


def participation_cap_shares(df: pd.DataFrame, window: int, pct: float) -> Optional[float]:
    """Max shares tradeable in one day under the ADV participation cap.

    Standalone helper so the paper engine can apply the same cap to BUY sizing
    without recomputing the whole execution plan.
    """
    adv_shares, _ = adv(df, window)
    if not adv_shares or adv_shares <= 0:
        return None
    return adv_shares * float(pct)
