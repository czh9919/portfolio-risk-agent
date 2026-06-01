"""
Unit tests for the core risk math in risk/risk_engine.py.

These functions drive the alert triggers and the paper-trade VaR gate, so they
are the highest-consequence pure functions in the repo. They are tested in
isolation with synthetic data — no network, no yfinance.
"""
import numpy as np
import pandas as pd
import pytest

from risk.risk_engine import (
    cornish_fisher_z,
    cf_var,
    ewma_var,
    ewma_variance_series,
    evt_var_gpd,
    fixed_var_cvar,
    max_drawdown,
    hhi,
    sharpe_ratio,
    Z95,
)


# ── Cornish-Fisher quantile ───────────────────────────────────────────────────

def test_cf_z_reduces_to_normal_when_no_skew_kurt():
    # With zero skew and zero excess kurtosis the CF quantile == the normal one.
    z = -1.645
    assert cornish_fisher_z(z, 0.0, 0.0) == pytest.approx(z)


def test_cf_z_negative_skew_fattens_lower_tail():
    # Left-skewed returns must push the lower-tail quantile MORE negative,
    # i.e. a larger VaR — otherwise the gate under-sizes a left-skewed book.
    z = -1.645
    assert cornish_fisher_z(z, -1.0, 0.0) < z


def test_cf_z_excess_kurtosis_fattens_deep_tail():
    # The CF kurtosis term is quantile-dependent: at the 99% quantile
    # (z = -2.326) fat tails push the quantile MORE negative (larger VaR).
    # (Near 95% the cubic term is positive and the effect is the opposite,
    #  which is the correct, well-known CF property.)
    z = -2.326
    assert cornish_fisher_z(z, 0.0, 3.0) < z


# ── Cornish-Fisher VaR ────────────────────────────────────────────────────────

def test_cf_var_nan_for_short_series():
    s = pd.Series(np.zeros(10))
    assert np.isnan(cf_var(s, 0.02, 0.95))


def test_cf_var_close_to_normal_for_gaussian():
    rng = np.random.default_rng(0)
    r   = pd.Series(rng.normal(0, 0.01, 2000))
    sigma = float(r.std())
    cf = cf_var(r, sigma, 0.95)
    # For near-Gaussian data CF VaR ≈ 1.645 * sigma (within 15%).
    assert cf == pytest.approx(Z95 * sigma, rel=0.15)


def test_cf_var_larger_for_left_skewed():
    rng = np.random.default_rng(1)
    sym = pd.Series(rng.normal(0, 0.01, 2000))
    # Inject a left (negative) skew by appending large negative shocks.
    skewed = pd.concat([sym, pd.Series(rng.normal(-0.05, 0.01, 200))], ignore_index=True)
    sigma_s = float(skewed.std())
    sigma_y = float(sym.std())
    assert cf_var(skewed, sigma_s, 0.95) > cf_var(sym, sigma_y, 0.95)


# ── EWMA VaR ──────────────────────────────────────────────────────────────────

def test_ewma_var_nan_for_short_series():
    s = pd.Series(np.zeros(5))
    assert np.isnan(ewma_var(s, init_window=21))


def test_ewma_var_positive_and_scales_with_vol():
    rng = np.random.default_rng(2)
    lo  = pd.Series(rng.normal(0, 0.005, 500))
    hi  = pd.Series(rng.normal(0, 0.020, 500))
    v_lo = ewma_var(lo)
    v_hi = ewma_var(hi)
    assert v_lo > 0 and v_hi > 0
    assert v_hi > v_lo


def test_ewma_variance_series_aligns_index():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 100))
    vs = ewma_variance_series(r)
    assert len(vs) == len(r)
    assert (vs.values >= 0).all()


# ── Historical VaR / CVaR ─────────────────────────────────────────────────────

def test_cvar_at_least_var():
    rng = np.random.default_rng(4)
    r   = pd.Series(rng.normal(0, 0.01, 1000))
    var, cvar = fixed_var_cvar(r, 0.95)
    # CVaR is the mean loss beyond VaR, so it must be >= VaR.
    assert cvar >= var > 0


def test_fixed_var_cvar_empty():
    var, cvar = fixed_var_cvar(pd.Series(dtype=float))
    assert np.isnan(var) and np.isnan(cvar)


# ── EVT / GPD ─────────────────────────────────────────────────────────────────

def test_evt_var_nan_for_short_series():
    s = pd.Series(np.zeros(30))
    assert np.isnan(evt_var_gpd(s))


def test_evt_var_positive_for_fat_tailed():
    rng = np.random.default_rng(5)
    # Student-t style fat tails
    r = pd.Series(rng.standard_t(4, 1000) * 0.01)
    out = evt_var_gpd(r, confidence=0.99)
    assert not np.isnan(out)
    assert out > 0


# ── Drawdown / HHI / Sharpe ───────────────────────────────────────────────────

def test_max_drawdown_zero_for_monotonic_up():
    r = pd.Series([0.01] * 50)          # never declines
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known_value():
    # +10% then -50% → cum goes 1.10 then 0.55; peak 1.10 → dd = -0.5
    r = pd.Series([0.10, -0.50])
    assert max_drawdown(r) == pytest.approx(-0.5, rel=1e-6)


def test_hhi_equal_weights():
    holdings = [{"weight": 0.25} for _ in range(4)]
    # 4 equal weights → HHI = 4 * 0.25^2 = 0.25
    assert hhi(holdings) == pytest.approx(0.25)


def test_hhi_single_position_is_one():
    assert hhi([{"weight": 1.0}]) == pytest.approx(1.0)


def test_sharpe_nan_for_short_series():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 5)))
