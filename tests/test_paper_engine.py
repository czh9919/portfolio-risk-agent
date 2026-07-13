"""
Unit tests for the financially-consequential paper-trade logic:
risk gates, IR-proportional sizing, SELL-signal exits, CGT-aware rebalancing,
and the performance calculation. All synthetic — no network, no Alpaca.

dry_run=True everywhere so no order is ever placed.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.paper_engine import (
    _risk_gates,
    _compute_performance,
    _near_earnings,
    _rebalance_orders,
    generate_orders,
)


class FakePD:
    """Minimal stand-in for data.price_loader.PriceData."""
    def __init__(self, price, returns=None, status="ok"):
        self.closes  = pd.Series([price], dtype=float)
        self.returns = returns
        self.status  = status


def _permissive_cfg(**over):
    cfg = {
        "max_positions":      10,
        "var_limit_pct":      0.05,
        "max_portfolio_beta": 1.5,
        "max_drawdown_halt":  0.10,
        "starting_equity":    100_000,
        "vix_halt_threshold": 25,
        "stop_loss_pct":      0.0,
        "trailing_stop_pct":  0.0,
        "rebalance_threshold": 0.0,
        "earnings_avoid_days": 0,   # 0 → _near_earnings short-circuits, no network
        "cgt_rate":           0.33,
    }
    cfg.update(over)
    return cfg


def _ok_risk():
    return {"n_positions": 0, "var_1d_95": 0.02,
            "portfolio_beta": 1.0, "hhi": 0.0, "weights": {}}


# ── Risk gates ────────────────────────────────────────────────────────────────

def test_gate_allows_when_all_clear():
    allow, reason = _risk_gates(_ok_risk(), _permissive_cfg(), 100_000, vix=15)
    assert allow is True
    assert reason == ""


def test_gate_blocks_on_max_positions():
    risk = _ok_risk(); risk["n_positions"] = 10
    allow, reason = _risk_gates(risk, _permissive_cfg(), 100_000)
    assert allow is False and "max positions" in reason


def test_gate_blocks_on_var():
    risk = _ok_risk(); risk["var_1d_95"] = 0.08
    allow, reason = _risk_gates(risk, _permissive_cfg(), 100_000)
    assert allow is False and "VaR" in reason


def test_gate_blocks_on_beta():
    risk = _ok_risk(); risk["portfolio_beta"] = 1.8
    allow, reason = _risk_gates(risk, _permissive_cfg(), 100_000)
    assert allow is False and "beta" in reason


def test_gate_blocks_on_drawdown():
    # equity 88k vs starting 100k = 12% drawdown > 10% halt
    allow, reason = _risk_gates(_ok_risk(), _permissive_cfg(), 88_000)
    assert allow is False and "drawdown" in reason


def test_gate_blocks_on_vix():
    allow, reason = _risk_gates(_ok_risk(), _permissive_cfg(), 100_000, vix=30)
    assert allow is False and "VIX" in reason


def test_gate_vix_ignored_when_none():
    allow, _ = _risk_gates(_ok_risk(), _permissive_cfg(), 100_000, vix=None)
    assert allow is True


# ── IR-proportional new-buy sizing ────────────────────────────────────────────

def test_new_buys_respect_cap_and_budget():
    results = [
        {"ticker": "AAA", "robust_signal": "BUY", "signal": "BUY", "ir": 3.0, "t_alpha": 3.0},
        {"ticker": "BBB", "robust_signal": "BUY", "signal": "BUY", "ir": 1.0, "t_alpha": 2.0},
    ]
    price_data = {"AAA": FakePD(100.0), "BBB": FakePD(50.0)}
    orders = generate_orders(
        results, positions={}, equity=100_000, price_data=price_data,
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_permissive_cfg(), buying_power=100_000,
        vix=15, market_open=True,
    )
    buys = [o for o in orders if o["side"] == "buy"]
    assert len(buys) == 2
    # Higher IR (AAA) must get >= weight of lower IR (BBB)
    wmap = {o["ticker"]: o["weight"] for o in buys}
    assert wmap["AAA"] >= wmap["BBB"]
    # Each allocation capped at max_pos_pct
    assert all(o["weight"] <= 0.15 + 1e-9 for o in buys)
    # Total deployed value cannot exceed equity
    assert sum(o["value_usd"] for o in buys) <= 100_000 + 1e-6
    # Fractional quantities allowed
    assert all(o["qty"] > 0 for o in buys)


def test_no_buys_when_gate_blocks():
    results = [{"ticker": "AAA", "robust_signal": "BUY", "signal": "BUY", "ir": 2.0, "t_alpha": 3.0}]
    risk = _ok_risk(); risk["n_positions"] = 10   # trips max-positions gate
    orders = generate_orders(
        results, positions={}, equity=100_000, price_data={"AAA": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=risk,
        paper_cfg=_permissive_cfg(), buying_power=100_000, market_open=True,
    )
    assert [o for o in orders if o["side"] == "buy"] == []


def test_no_buys_when_market_closed():
    results = [{"ticker": "AAA", "robust_signal": "BUY", "signal": "BUY", "ir": 2.0, "t_alpha": 3.0}]
    orders = generate_orders(
        results, positions={}, equity=100_000, price_data={"AAA": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_permissive_cfg(), buying_power=100_000, market_open=False,
    )
    assert [o for o in orders if o["side"] == "buy"] == []


# ── SELL-signal exit ──────────────────────────────────────────────────────────

def test_sell_signal_closes_held_position():
    results = [{"ticker": "AAA", "robust_signal": "SELL", "signal": "SELL", "ir": -1.0, "t_alpha": -3.0}]
    positions = {"AAA": {"qty": "10", "market_value": "1000", "unrealized_plpc": "0.05"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000, price_data={"AAA": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_permissive_cfg(), buying_power=100_000, market_open=True,
    )
    sells = [o for o in orders if o["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["ticker"] == "AAA"


# ── CGT-aware rebalancing ─────────────────────────────────────────────────────

def _rebalance_single(plpc, rebalance_threshold=0.05):
    """One held BUY, overweight at 30% vs 15% cap → 15% drift. plpc sets the gain."""
    results = [{"ticker": "AAA", "robust_signal": "BUY", "ir": 1.0}]
    positions = {"AAA": {"qty": "100", "market_value": "30000",
                         "unrealized_plpc": str(plpc)}}
    price_data = {"AAA": FakePD(100.0)}
    orders, _ = _rebalance_orders(
        results, positions, equity=100_000, price_data=price_data,
        risk_weights={"AAA": 0.30}, max_pos_pct=0.15,
        rebalance_threshold=rebalance_threshold, cgt_rate=0.33,
        gates_allow_buys=True, remaining_bp=100_000, dry_run=True,
        exiting=set(),
    )
    return orders


def test_rebalance_trims_loss_position():
    # No gain → no CGT friction → 15% drift > 5% threshold → trim fires.
    orders = _rebalance_single(plpc=0.0)
    trims = [o for o in orders if o["side"] == "sell"]
    assert len(trims) == 1


def test_rebalance_no_topup_when_sleeve_in_balance():
    # One held BUY at 10% with the rest of the book in non-BUY names. Targets
    # are scaled to the sleeve's current total weight, so the name is already
    # at target → no order. (Regression: targets normalised over the sleeve
    # alone made every BUY name look underweight → perpetual top-ups.)
    results = [{"ticker": "AAA", "robust_signal": "BUY", "ir": 1.0}]
    positions = {
        "AAA": {"qty": "100", "market_value": "10000", "unrealized_plpc": "0.0"},
        "ZZZ": {"qty": "100", "market_value": "90000", "unrealized_plpc": "0.0"},
    }
    orders, _ = _rebalance_orders(
        results, positions, equity=100_000,
        price_data={"AAA": FakePD(100.0)},
        risk_weights={"AAA": 0.10, "ZZZ": 0.90}, max_pos_pct=0.15,
        rebalance_threshold=0.05, cgt_rate=0.33,
        gates_allow_buys=True, remaining_bp=100_000, dry_run=True,
        exiting=set(),
    )
    assert orders == []


def test_cgt_friction_suppresses_trim_on_big_winner():
    # +50% gain: gain_frac = 0.5/1.5 = 0.333; friction = 0.333*0.33 = 0.11;
    # effective threshold = 0.05 + 0.11 = 0.16 > 0.15 drift → NO trim.
    orders = _rebalance_single(plpc=0.50)
    assert [o for o in orders if o["side"] == "sell"] == []


# ── Core parking (idle cash → SPY) ────────────────────────────────────────────

def _core_cfg(**over):
    return _permissive_cfg(
        core_parking_ticker="SPY", core_cash_buffer_pct=0.02, **over
    )


def _core_mix_cfg(**over):
    return _permissive_cfg(
        core_parking_tickers={"SPY": 0.60, "QQQ": 0.40},
        core_cash_buffer_pct=0.02, **over,
    )


def test_core_sweep_buys_spy_with_idle_cash():
    orders = generate_orders(
        [], positions={}, equity=100_000, price_data={"SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_cfg(), buying_power=100_000, market_open=True,
        cash=100_000,
    )
    assert len(orders) == 1
    o = orders[0]
    assert o["ticker"] == "SPY" and o["side"] == "buy"
    # 100k cash − 2% buffer = 98k swept
    assert o["value_usd"] == pytest.approx(98_000, rel=1e-3)


def test_core_sweep_blocked_by_vix_gate():
    # VIX halt gate must block sweep regardless of market state
    orders = generate_orders(
        [], positions={}, equity=100_000, price_data={"SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_cfg(), buying_power=100_000, cash=100_000,
        vix=30, market_open=True,
    )
    assert orders == []


def test_core_sweep_fires_when_market_closed_and_gates_pass():
    # After-hours cron: market closed but risk gates OK → sweep still queues
    # (Alpaca fills day-tif orders on the next open)
    orders = generate_orders(
        [], positions={}, equity=100_000, price_data={"SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_cfg(), buying_power=100_000, cash=100_000,
        market_open=False,
    )
    assert len(orders) == 1
    assert orders[0]["ticker"] == "SPY" and orders[0]["side"] == "buy"


def test_core_sweep_splits_across_spy_and_qqq_by_weight():
    orders = generate_orders(
        [], positions={}, equity=100_000,
        price_data={"SPY": FakePD(500.0), "QQQ": FakePD(400.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_mix_cfg(), buying_power=100_000, market_open=True,
        cash=100_000,
    )
    # 98k swept: 58.8k SPY + 39.2k QQQ
    by_ticker = {o["ticker"]: o for o in orders if o["side"] == "buy"}
    assert set(by_ticker) == {"SPY", "QQQ"}
    assert by_ticker["SPY"]["value_usd"] == pytest.approx(58_800, rel=1e-3)
    assert by_ticker["QQQ"]["value_usd"] == pytest.approx(39_200, rel=1e-3)


def test_core_funding_pulls_proportionally_from_mix():
    # Cash 1k, SPY 60k + QQQ 40k parked, one BUY signal wants 15k → the mix
    # must fund shortfall (16k) 60/40 across the two names.
    results = [{"ticker": "AAA", "robust_signal": "BUY", "signal": "BUY",
                "ir": 2.0, "t_alpha": 3.0}]
    positions = {
        "SPY": {"qty": "120", "market_value": "60000", "unrealized_plpc": "0.05"},
        "QQQ": {"qty": "100", "market_value": "40000", "unrealized_plpc": "0.05"},
    }
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"AAA": FakePD(100.0), "SPY": FakePD(500.0),
                    "QQQ": FakePD(400.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_mix_cfg(), buying_power=2_000, market_open=True,
        cash=1_000,
    )
    core_sells = {o["ticker"]: o for o in orders
                  if o["side"] == "sell" and o["ticker"] in {"SPY", "QQQ"}}
    assert set(core_sells) == {"SPY", "QQQ"}
    # 16k shortfall × 60/40 mix
    assert core_sells["SPY"]["value_usd"] == pytest.approx(9_600, rel=1e-3)
    assert core_sells["QQQ"]["value_usd"] == pytest.approx(6_400, rel=1e-3)
    assert any(o["ticker"] == "AAA" and o["side"] == "buy" for o in orders)


def test_core_funding_sell_covers_new_buy():
    # Cash 1k, core SPY 97k. One BUY signal wants 15% (15k) → core must fund it.
    results = [{"ticker": "AAA", "robust_signal": "BUY", "signal": "BUY",
                "ir": 2.0, "t_alpha": 3.0}]
    positions = {"SPY": {"qty": "194", "market_value": "97000",
                         "unrealized_plpc": "0.05"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"AAA": FakePD(100.0), "SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_cfg(), buying_power=2_000, market_open=True,
        cash=1_000,
    )
    core_sells = [o for o in orders if o["ticker"] == "SPY" and o["side"] == "sell"]
    buys       = [o for o in orders if o["ticker"] == "AAA" and o["side"] == "buy"]
    assert len(core_sells) == 1 and "CORE FUNDING" in core_sells[0]["reason"]
    assert len(buys) == 1
    # Funding must cover the buy: 15k target + 2k buffer − 1k cash = 16k
    assert core_sells[0]["value_usd"] == pytest.approx(16_000, rel=1e-3)
    assert buys[0]["value_usd"] == pytest.approx(15_000, rel=1e-3)


def test_core_exempt_from_sell_signal_and_stop_loss():
    # SPY flags SELL and is 20% underwater — core must NOT be closed.
    results = [{"ticker": "SPY", "robust_signal": "SELL", "signal": "SELL",
                "ir": -1.0, "t_alpha": -2.0}]
    positions = {"SPY": {"qty": "194", "market_value": "97000",
                         "unrealized_plpc": "-0.20"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_core_cfg(stop_loss_pct=0.15, trailing_stop_pct=0.0),
        buying_power=3_000, market_open=True, cash=3_000,
    )
    assert [o for o in orders if o["side"] == "sell"] == []


def test_core_does_not_consume_max_positions_slot():
    # 10 positions incl. core; gate limit 10 → core exemption keeps buys open.
    results = [{"ticker": "AAA", "robust_signal": "BUY", "signal": "BUY",
                "ir": 2.0, "t_alpha": 3.0}]
    risk = _ok_risk(); risk["n_positions"] = 10
    positions = {"SPY": {"qty": "10", "market_value": "5000", "unrealized_plpc": "0.0"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"AAA": FakePD(100.0), "SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=risk,
        paper_cfg=_core_cfg(), buying_power=100_000, market_open=True,
        cash=95_000,
    )
    assert any(o["ticker"] == "AAA" and o["side"] == "buy" for o in orders)


# ── Short overlay ─────────────────────────────────────────────────────────────

def _short_cfg(**over):
    return _permissive_cfg(
        short_pos_pct=0.04, short_max_total_pct=0.12, **over
    )


def test_short_opens_on_robust_sell():
    results = [{"ticker": "AAA", "robust_signal": "SELL", "signal": "SELL",
                "ir": -2.0, "t_alpha": -3.0}]
    orders = generate_orders(
        results, positions={}, equity=100_000, price_data={"AAA": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_short_cfg(), buying_power=100_000, market_open=True,
        cash=100_000,
    )
    shorts = [o for o in orders if o["side"] == "short"]
    assert len(shorts) == 1
    o = shorts[0]
    assert o["ticker"] == "AAA"
    assert o["qty"] == 40                       # 4% × 100k / $100, whole shares
    assert float(o["qty"]) == int(o["qty"])


def test_short_capped_by_gross_exposure():
    # Existing short worth 12% of equity → cap reached, no new short.
    results = [{"ticker": "BBB", "robust_signal": "SELL", "signal": "SELL",
                "ir": -2.0, "t_alpha": -3.0}]
    positions = {"AAA": {"qty": "-120", "market_value": "-12000",
                         "unrealized_plpc": "0.0"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"AAA": FakePD(100.0), "BBB": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_short_cfg(), buying_power=100_000, market_open=True,
        cash=100_000,
    )
    assert [o for o in orders if o["side"] == "short"] == []


def test_short_disabled_by_default():
    results = [{"ticker": "AAA", "robust_signal": "SELL", "signal": "SELL",
                "ir": -2.0, "t_alpha": -3.0}]
    orders = generate_orders(
        results, positions={}, equity=100_000, price_data={"AAA": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_permissive_cfg(), buying_power=100_000, market_open=True,
        cash=100_000,
    )
    assert [o for o in orders if o["side"] == "short"] == []


def test_cover_when_signal_fades():
    results = [{"ticker": "AAA", "robust_signal": "HOLD", "signal": "HOLD",
                "ir": 0.1, "t_alpha": 0.2}]
    positions = {"AAA": {"qty": "-40", "market_value": "-4000",
                         "unrealized_plpc": "0.03"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"AAA": FakePD(100.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_short_cfg(), buying_power=100_000, market_open=True,
        cash=100_000,
    )
    covers = [o for o in orders if o["side"] == "cover"]
    assert len(covers) == 1 and covers[0]["ticker"] == "AAA"
    assert covers[0]["qty"] == pytest.approx(40)


def test_short_held_while_signal_persists_and_stop_loss_covers():
    # Signal still SELL → no cover. But a short 20% underwater trips stop-loss.
    results = [{"ticker": "AAA", "robust_signal": "SELL", "signal": "SELL",
                "ir": -2.0, "t_alpha": -3.0}]
    positions = {"AAA": {"qty": "-40", "market_value": "-4800",
                         "unrealized_plpc": "-0.20"}}
    orders = generate_orders(
        results, positions=positions, equity=100_000,
        price_data={"AAA": FakePD(120.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_short_cfg(stop_loss_pct=0.15), buying_power=100_000,
        market_open=True, cash=100_000,
    )
    covers = [o for o in orders if o["side"] == "cover"]
    assert len(covers) == 1
    assert "STOP-LOSS" in covers[0]["reason"]
    assert covers[0]["qty"] == 40               # abs, whole
    # No long "sell" close of the short, and no re-short of a held name
    assert [o for o in orders if o["side"] in ("sell", "short")] == []


def test_short_proceeds_do_not_inflate_core_sweep():
    # Cash 10k (net of shorts), one new short opens — sweep must still be
    # 10k − 2% buffer, not 10k + short proceeds.
    results = [{"ticker": "AAA", "robust_signal": "SELL", "signal": "SELL",
                "ir": -2.0, "t_alpha": -3.0}]
    orders = generate_orders(
        results, positions={}, equity=100_000,
        price_data={"AAA": FakePD(100.0), "SPY": FakePD(500.0)},
        max_pos_pct=0.15, dry_run=True, risk=_ok_risk(),
        paper_cfg=_short_cfg(core_parking_ticker="SPY", core_cash_buffer_pct=0.02),
        buying_power=100_000, market_open=True, cash=10_000,
    )
    sweeps = [o for o in orders if o["side"] == "buy" and o["ticker"] == "SPY"]
    assert len(sweeps) == 1
    assert sweeps[0]["value_usd"] == pytest.approx(8_000, rel=1e-3)


# ── Performance metrics ───────────────────────────────────────────────────────

def test_performance_empty_history():
    perf = _compute_performance([])
    assert perf["total_return"] == 0.0
    assert perf["sharpe"] is None
    assert perf["n_days"] == 0


def test_performance_total_return_and_drawdown():
    hist = [
        {"date": "2026-01-01", "equity": 100_000, "cash": 0},
        {"date": "2026-01-02", "equity": 110_000, "cash": 0},
        {"date": "2026-01-03", "equity": 99_000,  "cash": 0},
    ]
    perf = _compute_performance(hist)
    assert perf["total_return"] == pytest.approx(-0.01, rel=1e-6)
    # Peak 110k → trough 99k → max drawdown = 10%
    assert perf["max_drawdown"] == pytest.approx(0.10, rel=1e-6)
    assert perf["n_days"] == 3


# ── Earnings gate short-circuit ───────────────────────────────────────────────

def test_near_earnings_disabled_returns_false():
    # avoid_days <= 0 must short-circuit before any network call.
    assert _near_earnings("AAPL", 0) is False
    assert _near_earnings("AAPL", -1) is False
