"""
Unit tests for the execution / laddering analysis.

All synthetic — no network. Network-dependent pieces (options Max Pain / peak
gamma, earnings-date lookup) are exercised only via analyze_execution() with
options_enabled=False, so these tests never hit yfinance.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.execution import (
    atr,
    anchored_vwap,
    volume_profile_hvn,
    adv,
    build_ladder,
    participation_cap_shares,
    analyze_execution,
    _verdict,
    _cost_cross_date,
    _bs_gamma,
)


def _ohlcv(n=160, start=100.0, step=0.25, vol=1_000_000, seed=0):
    """Deterministic rising-price OHLCV frame on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = start + step * np.arange(n)
    high = close + 1.0
    low = close - 1.0
    open_ = close - step
    volume = np.full(n, vol, dtype=float) + rng.integers(0, 1000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ── ATR ─────────────────────────────────────────────────────────────────────

def test_atr_constant_range():
    df = _ohlcv()
    # High-Low is a constant 2.0; with a steady +0.25 drift TR stays near 2.0
    a = atr(df, period=14)
    assert a is not None
    assert 1.8 < a < 2.6


def test_atr_too_short_returns_none():
    assert atr(_ohlcv(n=5), period=14) is None


# ── Anchored VWAP ─────────────────────────────────────────────────────────────

def test_anchored_vwap_within_price_range():
    df = _ohlcv()
    anchor = df.index[100].date()
    v = anchored_vwap(df, anchor)
    seg = df[df.index >= pd.Timestamp(anchor)]
    assert v is not None
    assert seg["Close"].min() - 2 <= v <= seg["Close"].max() + 2


def test_anchored_vwap_none_anchor():
    assert anchored_vwap(_ohlcv(), None) is None


# ── Volume profile / HVN ────────────────────────────────────────────────────

def test_hvn_levels_sorted_and_in_range():
    df = _ohlcv(n=160)
    hvn = volume_profile_hvn(df, bins=24, lookback=120, top=4)
    assert len(hvn) <= 4
    seg = df.tail(120)
    for h in hvn:
        assert seg["Close"].min() <= h <= seg["Close"].max()


# ── ADV ───────────────────────────────────────────────────────────────────────

def test_adv_shares_and_dollar():
    df = _ohlcv(vol=500_000)
    shares, dollar = adv(df, window=20)
    assert shares is not None and 500_000 <= shares < 501_000
    assert dollar is not None and dollar > shares  # price > 1


def test_participation_cap_shares():
    df = _ohlcv(vol=1_000_000)
    cap = participation_cap_shares(df, window=20, pct=0.08)
    assert cap is not None
    assert 79_000 < cap < 81_000  # ~8% of ~1M


# ── Ladder ────────────────────────────────────────────────────────────────────

def test_ladder_exit_rungs_above_price():
    rungs = build_ladder(price=100.0, atr_val=2.0, hvn=[], side="exit",
                         n_rungs=5, spacing_atr=0.5, snap_atr=0.5)
    assert len(rungs) == 5
    prices = [r["price"] for r in rungs]
    assert all(p > 100.0 for p in prices)
    # spacing = 0.5 * 2.0 = 1.0 per rung
    assert prices == sorted(prices)
    assert abs(prices[0] - 101.0) < 1e-6


def test_ladder_entry_rungs_below_price():
    rungs = build_ladder(price=100.0, atr_val=2.0, hvn=[], side="entry",
                         n_rungs=4, spacing_atr=0.5, snap_atr=0.5)
    assert len(rungs) == 4
    assert all(r["price"] < 100.0 for r in rungs)


def test_ladder_snaps_to_hvn():
    # An HVN at 101.2 sits within 0.5*ATR(=1.0) of the first rung (101.0) → snap
    rungs = build_ladder(price=100.0, atr_val=2.0, hvn=[101.2], side="exit",
                         n_rungs=2, spacing_atr=0.5, snap_atr=0.5)
    assert rungs[0]["at_hvn"] is True
    assert abs(rungs[0]["price"] - 101.2) < 1e-6


def test_ladder_no_atr_returns_empty():
    assert build_ladder(100.0, None, [], "exit", 5, 0.5, 0.5) == []


# ── Verdict ────────────────────────────────────────────────────────────────────

def test_verdict_exit_above_anchor_reduces():
    v, prem = _verdict("exit", price=110.0, ref_vwap=100.0)
    assert v == "REDUCE"
    assert prem == pytest.approx(10.0)


def test_verdict_exit_below_anchor_observes():
    v, _ = _verdict("exit", price=90.0, ref_vwap=100.0)
    assert v == "OBSERVE"


def test_verdict_entry_below_anchor_accumulates():
    v, _ = _verdict("entry", price=90.0, ref_vwap=100.0)
    assert v == "ACCUMULATE"


def test_verdict_entry_above_anchor_waits():
    v, _ = _verdict("entry", price=110.0, ref_vwap=100.0)
    assert v == "WAIT"


def test_verdict_no_anchor():
    assert _verdict("exit", 100.0, None) == ("OBSERVE", None)
    assert _verdict("entry", 100.0, None) == ("WAIT", None)


# ── Cost-cross anchor ───────────────────────────────────────────────────────

def test_cost_cross_date_finds_last_crossing():
    df = _ohlcv(n=40, start=100.0, step=0.25)  # 100.00 → 109.75, monotonic
    # avg cost 105 is crossed exactly once on the way up
    d = _cost_cross_date(df, 105.0)
    assert d is not None
    assert df.loc[pd.Timestamp(d), "Close"] >= 105.0


def test_cost_cross_date_no_crossing():
    df = _ohlcv(n=40, start=100.0, step=0.25)
    assert _cost_cross_date(df, 500.0) is None  # never reached


# ── Black-Scholes gamma sanity ────────────────────────────────────────────────

def test_bs_gamma_peaks_atm():
    g_atm = _bs_gamma(100, 100, 0.1, 0.3, 0.035)
    g_otm = _bs_gamma(100, 130, 0.1, 0.3, 0.035)
    assert g_atm > g_otm > 0


# ── End-to-end (no network) ────────────────────────────────────────────────────

def test_analyze_execution_exit_no_options():
    df = _ohlcv(n=160, start=100.0, step=0.25, vol=1_000_000)
    cfg = {"options_enabled": False, "participation_pct": 0.08, "adv_window": 20}
    plan = analyze_execution("FAKE", df, side="exit",
                             position_shares=400_000, avg_cost_native=None, cfg=cfg)
    assert plan is not None
    assert plan["ticker"] == "FAKE"
    assert plan["side"] == "exit"
    assert plan["options"] is None
    assert plan["max_shares_day"] is not None
    # 400k shares / (~8% of 1M ≈ 80k per day) ≈ 5 days
    assert plan["days_to_exit"] == pytest.approx(5.0, abs=0.2)


def test_analyze_execution_builds_both_ladders():
    df = _ohlcv(n=160, start=100.0, step=0.25, vol=1_000_000)
    cfg = {"options_enabled": False}
    plan = analyze_execution("FAKE", df, side="exit",
                             position_shares=400_000, cfg=cfg)
    assert plan["primary"] == "up"
    up, down = plan["ladder_up"], plan["ladder_down"]
    assert up and down
    price = plan["price"]
    assert all(r["price"] > price for r in up)     # sell ladder above
    assert all(r["price"] < price for r in down)   # buy ladder below
    # Holdings size both ladders off position_shares; weights normalise to ~1
    assert sum(r["weight"] for r in up) == pytest.approx(1.0, abs=1e-3)
    assert all(r["shares"] is not None for r in up)


def test_analyze_execution_entry_primary_down():
    df = _ohlcv(n=160, start=100.0, step=0.25, vol=1_000_000)
    plan = analyze_execution("FAKE", df, side="entry",
                             ref_shares=1_000, cfg={"options_enabled": False})
    assert plan["primary"] == "down"
    assert plan["ref_shares"] == 1_000
    assert sum(r["shares"] for r in plan["ladder_down"]) == pytest.approx(1_000, abs=2)


def test_ladder_over_cap_flag():
    # ref_shares far above the per-day cap → every rung flagged over_cap
    rungs = build_ladder(price=100.0, atr_val=2.0, hvn=[], side="exit",
                         n_rungs=4, spacing_atr=0.5, snap_atr=0.5,
                         ref_shares=1_000_000, max_shares_day=10_000)
    assert all(r["over_cap"] for r in rungs)
    assert all(r["shares"] > 10_000 for r in rungs)


def test_analyze_execution_thin_data_returns_none():
    df = pd.DataFrame({"Close": []})
    assert analyze_execution("FAKE", df, side="exit", cfg={"options_enabled": False}) is None


def test_analyze_execution_uses_entry_date_anchor():
    import datetime as dt
    df = _ohlcv(n=160, start=100.0, step=0.25)
    cfg = {"options_enabled": False}
    plan = analyze_execution("FAKE", df, side="exit",
                             entry_date=df.index[100].date(), cfg=cfg)
    assert plan["cost_anchor_src"] == "entry_date"
    assert plan["cost_anchor"] == df.index[100].date().isoformat()
    assert plan["vwap_cost"] is not None


def test_analyze_execution_cost_cross_fallback():
    df = _ohlcv(n=160, start=100.0, step=0.25)
    plan = analyze_execution("FAKE", df, side="exit",
                             avg_cost_native=110.0, cfg={"options_enabled": False})
    assert plan["cost_anchor_src"] == "cost_cross"


# ── Entry-date ledger ───────────────────────────────────────────────────────

def test_normalize_date_formats():
    from data.entry_dates import normalize_date
    assert normalize_date("20240115") == "2024-01-15"
    assert normalize_date("20240115;143000") == "2024-01-15"
    assert normalize_date("2024-01-15T09:30:00.000+00:00") == "2024-01-15"
    assert normalize_date("2024-01-15") == "2024-01-15"
    assert normalize_date("") is None
    assert normalize_date(None) is None
    assert normalize_date("garbage") is None


def test_ledger_enrich_first_seen_and_broker_wins(tmp_path, monkeypatch):
    import datetime as dt
    from data import entry_dates
    monkeypatch.setattr(entry_dates, "LEDGER_PATH", tmp_path / "entry_dates.json")

    # No broker date → first-seen recorded as today
    h1 = [{"platform": "eToro", "ticker": "AAA", "entry_date": ""}]
    entry_dates.enrich(h1)
    assert h1[0]["entry_date"] == dt.date.today().isoformat()

    # A later run with a broker date overrides and backfills the ledger
    h2 = [{"platform": "eToro", "ticker": "AAA", "entry_date": "20230301"}]
    entry_dates.enrich(h2)
    assert h2[0]["entry_date"] == "2023-03-01"
    assert entry_dates.load()["eToro|AAA"] == "2023-03-01"
