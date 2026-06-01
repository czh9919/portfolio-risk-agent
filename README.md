# Portfolio Risk Agent
An automated portfolio risk, factor analysis, and paper trading system. A single unified pipeline fetches live broker holdings, runs Fama-French 5-factor regressions on every watchlist stock, computes a full risk dashboard, delivers a bilingual (English + Chinese) HTML email report after each US market close, and optionally executes paper trades on Alpaca Markets.

All live trading decisions remain manual. The system provides clean quantitative input and a paper trading sandbox for strategy validation.

---

## Known Limitations

### Backtest Survivorship Bias

The walk-forward backtest (`RUN_MODE=backtest`) fetches the **current** S&P 500 constituent list from Wikipedia. Companies removed through bankruptcy, forced acquisition, or index ejection are **never included** in the historical universe. This overstates out-of-sample Sharpe ratios and understates drawdowns. Treat figures as **upper-bound estimates**.

Eliminating the bias requires a point-in-time constituent database (CRSP, Compustat, EODHD). Every backtest result is flagged `[!] SURVIVORSHIP BIAS`.

### VaR Model Risk: Normal Distribution Assumption

Plain EWMA VaR (`var_95_ewma`) underestimates tail losses by 15–30% in stress regimes because equity returns are negatively skewed and fat-tailed. Three estimates are always computed side-by-side:

| Metric | Method | Role |
|---|---|---|
| `var_95_ewma` / `var_99_ewma` | Normal EWMA | Reference — underestimates in stress |
| `var_95_cf` / `var_99_cf` | **Cornish-Fisher** — adjusts for empirical skewness and excess kurtosis | **Primary alert trigger and report display** |
| `var_99_evt` | **EVT / Generalized Pareto** (peaks-over-threshold) | Most reliable for extreme quantiles; falls back to historical if < 20 exceedances |

For 21-day horizon tail risk the Monte Carlo DCC-GARCH + Hawkes simulation provides the most comprehensive fat-tail coverage.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│               UNIFIED PIPELINE  (main.py)                │
│                                                          │
│  Holdings   IBKR Flex Query XML                         │
│  fetched    T212 REST API                               │
│  from:      eToro (mock/CSV)                            │
│             ↓ fallback to local JSON cache              │
│      │                                                   │
│      ▼                                                   │
│  Price Loader   yfinance equity prices + FX rates       │
│                 EUR base: EURUSD=X, EURGBP=X, etc.      │
│                 Staleness checks; auto_adjust=True       │
│      │                                                   │
│      ▼                                                   │
│  FF5 Factor Model   (RUN_MODE=full)                     │
│    ┌────────────────────────────────────────────────┐    │
│    │ pandas-datareader → Ken French FF5 daily data  │    │
│    │ OLS per watchlist equity (252-day window,      │    │
│    │   fallback to available, min 63 days)          │    │
│    │ Outputs per stock:                             │    │
│    │   α (ann.), t(α), IR, R², β_MKT/SMB/HML/      │    │
│    │   RMW/CMA, signal (t>1.5→BUY / t<−1.5→SELL)  │    │
│    │ Robust signal: PSR + deflated IR + multi-      │    │
│    │   window consistency + OOS hit rate            │    │
│    │ Collinearity: VIF, condition number, corr      │    │
│    │ IC analysis: cross-sectional Spearman IC,      │    │
│    │   IC-IR, p-value (8 rolling eval periods)      │    │
│    │ Signal weights: PSR×IR×consistency → capped    │    │
│    │   20% model portfolio + factor exposure check  │    │
│    │ Portfolio aggregate: market-value-weighted β  │    │
│    │ Performance attribution: last 21 trading days │    │
│    │   RF + Σβ·factor + residual α                 │    │
│    └────────────────────────────────────────────────┘    │
│      │                                                   │
│      ▼                                                   │
│  Risk Engine    EWMA VaR (λ=0.94)                       │
│                 Cornish-Fisher VaR (primary alert)      │
│                 EVT/GPD VaR (extreme quantiles)         │
│                 CVaR / Expected Shortfall               │
│                 Sharpe ratio, Beta vs SPY               │
│                 Max drawdown, HHI concentration         │
│      │                                                   │
│      ▼                                                   │
│  Monte Carlo    DCC-GARCH + Hawkes self-exciting jumps  │
│                 50 000 paths × 21 days                  │
│                 Irish CGT 33%, trading costs,           │
│                 IBKR RegT margin monitoring             │
│      │                                                   │
│      ▼                                                   │
│  Stress Tests   2020 COVID crash, 2022 rate shock,      │
│                 2008 GFC, custom scenario                │
│      │                                                   │
│      ▼                                                   │
│  Optimizer      Markowitz efficient frontier            │
│                 FF5-implied μ (same window, no look-    │
│                 ahead); 3-tier rebalancing suggestions  │
│                 with α-based "why" decision trace       │
│                 Marginal impact for bond/gold           │
│      │                                                   │
│      ▼                                                   │
│  Alert System   Threshold checks → 24h-deduped emails   │
│      │                                                   │
│      ▼                                                   │
│  Notify Layer   Bilingual HTML email + embedded         │
│                 risk-return chart PNG (CID inline)      │
└──────────────────────────────────────────────────────────┘
```

---

## Run Modes

Set via the `RUN_MODE` environment variable:

| Mode | GitHub Actions time (UTC) | Behavior |
|---|---|---|
| `full` | 21:30 (16:30 ET, after US close) | Full pipeline: factor analysis for all watchlist equities + portfolio risk report |
| `portfolio` | Manual / local | Portfolio risk report only; skips watchlist factor ranking |
| `alert_check` | 14:00 (09:00 ET, pre-market) | Threshold check only; fires alert emails when a metric is RED; no report |
| `backtest` | Manual | Walk-forward validation — FF5 μ vs historical mean μ in Markowitz |
| `paper_trade` | 13:45 (09:45 ET, 15 min after open) + 22:00 Mon–Fri | Market-calendar gated; screen 50-ticker universe, prune/promote watchlist, run risk/VIX gates, execute fractional Alpaca paper orders, record NAV + performance |

---

## Modules

### `data/`

| File | Role |
|---|---|
| `fetcher.py` | Downloads OHLCV (`auto_adjust=True`) and current price via yfinance |
| `cleaner.py` | Validates price data — detects gaps > 10 trading days, price spikes > 50%, and **stale last-row** (last close > 4 calendar days old); validates news freshness |
| `storage.py` | SQLite wrapper — stores price history, data warnings, and backtest evaluations |
| `price_loader.py` | Loads equity price history into `PriceData` objects (`ok` / `reduced` / `excluded` / `unavailable`); fetches FX rates with staleness warning; EUR base via inverted `EURUSD=X` / `EURGBP=X` / `EURAUD=X` |
| `fetch_holdings.py` | Aggregates live positions from IBKR (Flex Query XML), T212 (REST API), and eToro; converts all values to EUR |
| `cache.py` | **Atomic** JSON snapshot writes (`.tmp` → rename); corrupt-file recovery; staleness warning when fallback snapshot is > 2 days old; 7-day rolling retention |
| `spy_universe.py` | Scrapes current S&P 500 tickers from Wikipedia; bulk yfinance download for walk-forward universe; 20h price cache |

### `brokers/`

| File | Role |
|---|---|
| `alpaca.py` | Alpaca Markets paper trading REST client: `get_account`, `get_positions`, `place_order`, `close_position`, `get_latest_price`, `get_open_orders`, `cancel_all_orders`, and `is_market_open` (`/v2/clock`). Paper base URL; requires `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`. |

### `strategy/`

| File | Role |
|---|---|
| `indicators.py` | SMA 20/50/200, EMA 12/26, MACD + signal, RSI (14), Bollinger Bands (20, 2σ), ATR (14), 10-day volume SMA |
| `signals.py` | Rule-based scoring — assigns bullish/bearish/neutral labels; returns composite score |
| `factor_model.py` | **FF5 regression engine**: `run_factor_regression` — OLS of excess returns on 5 Fama-French factors; 252-day window, min 63 days; outputs α, t(α), IR, R², 5 betas, BUY/SELL/HOLD signal. `portfolio_factor_exposure` — market-value-weighted aggregate betas. `compute_attribution` — decomposes last 21-day portfolio return into factor contributions + residual α. `enrich_suggestions` — attaches α-based "why" reason to each rebalancing suggestion. `compute_robust_signal` — multi-window PSR + deflated IR + OOS hit rate for statistically robust BUY/SELL signals. |
| `factor_ortho.py` | **Factor collinearity diagnostics**: `check_factor_collinearity` — pairwise correlations, VIF per factor, condition number of the FF5 correlation matrix, and high-correlation pair flagging (|r|>0.4). Warns when VIF>5, condition number>10, or any high-corr pair is found. |
| `ic_analysis.py` | **Cross-sectional IC analysis**: `compute_universe_ic` — rolling Spearman rank-IC between predicted alpha ranks (estimated over a 126-day training window) and realised forward returns, aggregated over 8 evaluation periods. Reports IC mean, IC-IR (annualised), t-stat, and p-value. `compute_alpha_decay` — hit rate of the alpha signal direction across 5/21/63/126-day horizons for a single ticker. |
| `weight_allocator.py` | **Risk-constrained weight allocation**: `compute_signal_weights` — converts robust BUY signals (PSR≥65%, IR_deflated>0) into portfolio weights using score = PSR × deflated IR × consistency boost; caps each position at 20%; iteratively redistributes excess to uncapped positions; flags portfolio-level FF5 factor exposures exceeding 1.5×. |
| `paper_engine.py` | **Paper trading pipeline**: market-calendar gate (`is_market_open`), universe expansion (permanent watchlist + date-seeded random S&P 500 fill to 50 tickers), FF5 screening, watchlist pruning (Rule A: SELL signal; Rule B: HOLD > 30 days), watchlist promotion, portfolio risk snapshot (Cornish-Fisher VaR, beta vs SPY, HHI), pre-trade risk gates (incl. VIX halt), trailing-stop/stop-loss exits, earnings-date avoidance, IR-proportional **fractional-share** order sizing, rebalance trim/top-up with CGT friction, order idempotency (`cancel_all_orders`), durable NAV history + trade log + performance metrics, Alpaca order execution. |

### `backtest/`

| File | Role |
|---|---|
| `historical.py` | SMA-20/50 crossover backtest via `vectorbt` (optional); returns annual return, Sharpe, max drawdown, win rate |
| `walk_forward.py` | Rolling walk-forward validation: FF5-implied μ vs historical mean μ in Markowitz across 3-year train / 1-quarter test windows. FF5 factor premiums **restricted to the training window** (no look-ahead). Assets must be present from the start of each training window to prevent covariance truncation. Reports `[!] SURVIVORSHIP BIAS` on every result. |

### `risk/`

| File | Role |
|---|---|
| `risk_engine.py` | 1-day risk metrics: EWMA VaR (λ=0.94), **Cornish-Fisher VaR** (primary alert trigger), **EVT/GPD VaR** (99%, peaks-over-threshold), Historical VaR, CVaR, Sharpe (rf=3.5% ECB), Beta vs SPY, max drawdown, HHI, sector concentration; RED/AMBER/GREEN RAG |
| `mc_engine.py` | **DCC-GARCH + Hawkes** path simulator. Per-asset GARCH(1,1) volatilities; time-varying correlation via DCC recursion; self-exciting Hawkes jump process calibrated from tail frequency. Vectorised: 50 000 paths × 21 days in ~1 s. Fallback: CC-GARCH → GBM. |
| `mc_portfolio.py` | Buy-and-hold simulation over `mc_engine` paths. Applies: Irish CGT 33% on gains > €1 270 exemption; trading costs (commission + half-spread + √ market-impact); IBKR RegT 25% margin breach flag. Returns VaR/CVaR/drawdown/Sharpe/Sortino/Calmar. |
| `stress.py` | Historical scenario stress tests: COVID-19 (Mar 2020, −34%), 2022 rate shock (−25%), 2008 GFC (−57%); EUR loss and % drawdown per scenario |
| `optimizer.py` | Markowitz optimisation with **Fama-French 5-factor implied μ** (factor premiums restricted to training window). `compute_frontier` — MC cloud + SLSQP max-Sharpe; `rebalancing_suggestions` — 3-tier vs optimal; `marginal_impact` — ΔSharpe/ΔVol for bond/gold candidates. |

### `notify/`

| File | Role |
|---|---|
| `report_gen.py` | Renders bilingual portfolio risk HTML report with all sections listed below |
| `chart.py` | Generates the efficient frontier PNG using headless matplotlib (Agg backend, CJK font fallback); returns raw bytes for CID inline embedding |
| `mailer.py` | Sends HTML email via SMTP (primary) or SendGrid (fallback); supports CID-embedded inline images; writes backup to `data/email_backup/` on failure |

### Root files

| File | Role |
|---|---|
| `main.py` | Unified entry point — reads `RUN_MODE`, runs portfolio + factor analysis pipeline |
| `scheduler.py` | APScheduler local runner: full pipeline at 21:30 UTC, alert check every 4h UTC (Mon–Fri) |
| `alert.py` | Fires bilingual alert emails for RED metrics; 24-hour deduplication via `cache/alert_dedup.json` |

---

## FF5 Factor Model

### Per-Stock Regression

`run_factor_regression()` solves via OLS:

```
Rᵢ − RF = α + β_MKT·(Mkt-RF) + β_SMB·SMB + β_HML·HML + β_RMW·RMW + β_CMA·CMA + ε
```

| Output | Description |
|---|---|
| `alpha_ann` | Annualised intercept α = α_daily × 252 |
| `t_alpha` | OLS t-statistic of α; t > 1.5 → BUY, t < −1.5 → SELL |
| `ir` | Information Ratio = (α_daily / σ_residuals) × √252 |
| `r_squared` | Model fit quality |
| `beta_mkt/smb/hml/rmw/cma` | Factor loadings |
| `signal` | BUY / SELL / HOLD — driven by statistical significance of α |

**Window:** last 252 aligned trading days (falls back to available if < 252 but ≥ 63; skips ticker entirely if < 63).

The stock factor table is **sorted by IR** — stocks with the highest risk-adjusted excess return appear first.

### Portfolio Factor Exposure

`portfolio_factor_exposure()` computes market-value-weighted averages of each beta and α across equity holdings. Shown in the report as a single aggregate row: useful for detecting unintended style tilts (e.g., unintended growth/value bet, concentration in small-caps).

### Performance Attribution

`compute_attribution()` decomposes the realized equity portfolio return over the **last 21 trading days** into:

```
R_portfolio = RF + β_MKT·(Mkt-RF) + β_SMB·SMB + β_HML·HML + β_RMW·RMW + β_CMA·CMA + α_residual
```

| Row | Meaning |
|---|---|
| Each factor row | β × realized factor return over the period |
| Factor total | Sum of all factor contributions + RF |
| **Alpha (stock picking)** | Actual return − factor total — tells you whether your holdings outperformed their factor model |

Large positive α: stock selection added value beyond what the factor tilts explain.  
Large negative α: you're being compensated only for beta exposure, not for selection.

### Robust Signal (PSR + Deflated IR + OOS)

`compute_robust_signal()` applies three additional filters on top of the naive t(α) signal to reduce false discoveries:

| Filter | What it measures | Threshold |
|---|---|---|
| **PSR** (Probabilistic Sharpe Ratio) | P(true IR > 0) corrected for non-normality (skewness, kurtosis) | PSR > 85% → BUY |
| **Deflated IR** | IR adjusted for multiple-testing across M strategies (Bailey–Lopez de Prado 2014) | IR_deflated > 0 |
| **Multi-window consistency** | α positive across 63/126/252-day windows simultaneously | ≥ 2/3 windows positive |
| **OOS hit rate** | Rolling 126-day train / 21-day test — fraction of periods where sign(α) = sign(realized excess return) | > 55% |

A BUY `robust_signal` requires all four filters to pass. The naive t(α) signal is shown in brackets in the report if it differs.

### Factor Collinearity Diagnostics

`check_factor_collinearity()` runs every cycle in `RUN_MODE=full` to detect problematic linear dependencies among the FF5 factors (typically HML↔CMA and Mkt-RF↔RMW):

| Diagnostic | Warning threshold |
|---|---|
| VIF per factor | > 5 concerning, > 10 severe |
| Correlation matrix condition number | > 10 |
| Pairwise |r| > 0.4 pairs | Any detected |

Collinearity does not invalidate betas but inflates their standard errors — signals should be weighted less heavily when VIFs are elevated.

### Cross-Sectional IC Analysis

`compute_universe_ic()` measures whether the alpha signal has genuine cross-sectional predictive power, not just in-sample fit:

1. At each of 8 evaluation points, estimate alpha for every watchlist stock using the prior 126 trading days of data
2. Rank stocks by estimated alpha
3. Observe actual 21-day forward excess returns and rank those
4. IC = Spearman rank correlation between predicted and realised ranks

| Output | Interpretation |
|---|---|
| `ic_mean` | Average signal quality; IC > 0.05 is meaningful for daily equity data |
| `ic_ir` | IC divided by IC volatility, annualised — analogous to Sharpe of the signal |
| `pvalue` | Two-tailed p-value (normal approximation); < 0.05 → statistically significant |

IC < 0 (signal predicts wrong direction) or p > 0.10 (noise) are warnings that the FF5 alpha ranking should be used with caution.

### Signal-Weighted Allocation

`compute_signal_weights()` translates robust BUY signals into a model portfolio:

```
Score = PSR × IR_deflated × (1 + 0.2 × alpha_consistency / n_windows)
Weight = Score / ΣScores   (before capping)
```

- Each position capped at **20%**; excess redistributed to uncapped positions iteratively
- Portfolio-level FF5 factor exposures are computed as Σ(weight × beta_factor)
- Exposures exceeding **1.5×** trigger a warning in the report

The signal portfolio is shown in the email report with per-position PSR, IR, and consistency, plus the portfolio-level factor exposure summary.

### Decision Trace

`enrich_suggestions()` attaches a concise reason to every rebalancing suggestion, cross-referencing the FF5 regression:

- `α=+8.2% t=+2.3 → BUY signal; underweight` — optimizer is increasing weight of a statistically significant outperformer
- `α=−3.8% t=−1.8 → SELL signal; overweight` — optimizer is trimming an underperformer
- `α=+1.4% t=+0.9; trim to reduce concentration` — neutral alpha, purely a concentration reduction

This lets you evaluate each manual trade decision in terms of the underlying factor signal, not just the weight delta.

### Factor Set: Why Five Factors, Not Six (Momentum Evaluation)

The Carhart momentum factor (UMD) was formally evaluated as a sixth factor and **deliberately excluded**. The decision rests on three escalating tests, run offline against a 16-year, 116-stock S&P 500 sample (reproducible via `experiment_momentum.py` and `experiment_momentum_walkforward.py`):

| Test | Question | Result | Verdict |
|---|---|---|---|
| Coefficient significance | Is momentum *priced* in our universe? | mean \|t(Mom)\| = 3.3; significant (\|t\|>2) for **59%** of stocks | Momentum is real |
| Incremental explanatory power | Does adding Mom improve the regression fit? | ΔR² ≈ **0.3%** | Negligible |
| Walk-forward economics | Does FF6-implied μ beat FF5-implied μ on out-of-sample portfolio P&L? | OOS Sharpe **1.514 vs 1.514** (45 windows), identical drawdown/turnover | No improvement |

**Conclusion — significant yet redundant.** Momentum is genuinely priced, but it adds nothing the model doesn't already capture: `ff_implied_mu` estimates each stock's α over the same window, and α already absorbs momentum-driven outperformance. Adding Mom merely reshuffles return from the α term into a β·premium term, leaving total expected return — and therefore the optimiser's output — unchanged. Five factors are retained; momentum would add a download dependency and estimation noise for no economic gain. (It remains worth revisiting only as factor-attribution hygiene, never for return forecasting.)

---

## Portfolio Risk Metrics

### Risk Dashboard

| Metric | Threshold (RED) | Method |
|---|---|---|
| VaR 95% (1-day) | > 5% | **Cornish-Fisher** (EWMA σ adjusted for skewness + kurtosis) |
| CVaR/VaR ratio | > 1.80× | Expected Shortfall / Historical simulation |
| Max Drawdown | > 20% | Rolling peak-to-trough |
| Largest position | > 30% weight | — |
| Single-day loss | > 3% | Previous close vs today |
| HHI concentration | > 0.25 | Herfindahl–Hirschman Index |
| Beta vs SPY | > 1.50 | 252-day rolling regression |
| Sharpe ratio | < 0.50 | rf = 3.5% (ECB) |

RAG thresholds are fully configurable in `config/thresholds.yaml`.

### VaR Methodology

| Key | Formula | Notes |
|---|---|---|
| `var_95_ewma` | `−μ + 1.645 · σ_EWMA` | Normal; underestimates fat tails — reference only |
| `var_95_cf` | `−μ − z_cf · σ_EWMA` where `z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36` | Cornish-Fisher; **primary alert trigger** |
| `var_99_evt` | GPD fit to worst-10% losses via `scipy.stats.genpareto` | EVT peaks-over-threshold; most reliable at 99%+ |

### Monte Carlo (21-day horizon)

`run_mc_portfolio()` runs 50 000 DCC-GARCH + Hawkes paths:

| Output | Description |
|---|---|
| VaR 95% / 99% | Empirical percentile of terminal P&L distribution |
| CVaR 95% | Mean loss in worst 5% of paths |
| Max drawdown (mean / p95) | Peak-to-trough along each path |
| P(loss > 10%) / P(loss > 20%) | Loss-probability ladder |
| Margin call probability | IBKR RegT breach on any day |
| Sharpe / Sortino / Calmar | Median across paths |
| Trading costs | Commission + half-spread + √ market-impact |
| CGT (median) | Irish CGT 33% on realised gains > €1 270 |

Configured in `config/settings.yaml` under `monte_carlo:`.

---

## Markowitz Optimisation

### Efficient Frontier

`compute_frontier()` samples 1 200 random Dirichlet portfolios, then solves for the max-Sharpe portfolio via SLSQP. Expected returns come from FF5-implied μ using the same in-sample window (no look-ahead). The chart is a matplotlib PNG embedded inline (CID) in the email.

### Rebalancing Suggestions (3-tier)

| Tier | Condition | Label |
|---|---|---|
| 1 — Act | \|Δweight\| ≥ 5%, no Irish CGT obstacle | Red — immediate action |
| 2 — Watch | 2–5% delta, or reducing a winner (CGT 33% applies) | Amber — monitor |
| 3 — Hold | \|Δweight\| < 2% | Grey — trade cost exceeds benefit |

Each suggestion includes an α-based **decision trace** explaining the underlying factor signal.

### Diversification Candidates (Bonds & Gold)

`marginal_impact()` evaluates each bond/gold entry from the watchlist: ΔSharpe, ΔVol, correlation to current equity portfolio, suggested weight, and FX rate note.

Current candidates in `config/watchlist.csv`: TLT, IEF, AGG (Treasuries), GLD.

---

## Paper Trading (`RUN_MODE=paper_trade`)

Runs at **13:45 UTC** (09:45 ET, 15 minutes after the US open) as part of `daily_run.yml`, and again at 22:00 UTC via `.github/workflows/paper_trade.yml`. The market-calendar gate (below) means only sessions when Alpaca reports the market open will actually place orders — the 22:00 run no-ops order entry on a normal day and just refreshes screening / NAV. Isolated from the portfolio risk pipeline beyond the shared watchlist and FF5 price cache.

### Market-calendar gate

Before anything else the engine calls Alpaca `/v2/clock` (`is_market_open`). When the market is closed, **all new buys and rebalances are suppressed** (exits still evaluate), the email carries a yellow "market closed" banner, and no orders are submitted. This prevents queuing orders against a closed book on holidays or off-hours dispatches.

### Pipeline

1. **Universe expansion** — current watchlist equities (permanent) + date-seeded random S&P 500 fill up to `universe_size` (default 50) total tickers. Same seed per day → reproducible within a day.
2. **FF5 screening** — runs `run_factor_regression` + `compute_robust_signal` on every ticker; sorts by IR descending.
3. **Watchlist pruning** — removes auto-promoted tickers only (identified by `FF5 signal=` prefix in the `notes` field):
   - **Rule A**: current signal is SELL → remove immediately
   - **Rule B**: current signal is not BUY AND entry is older than 30 days (date parsed from `[YYYY-MM-DD]` in notes) → remove
   - Manual entries are **never touched**
4. **Watchlist promotion** — S&P 500 candidates whose **robust signal** (PSR + deflated IR + multi-window consistency + OOS, *not* the naive t(α) signal) is BUY or SELL are appended to `watchlist.csv` (idempotent; skips existing tickers). Promoted entries have notes formatted as `FF5 signal=BUY t=X IR=Y [YYYY-MM-DD]` — this is what distinguishes them from manual entries.
5. **Portfolio risk snapshot** — computed from current Alpaca positions:
   - EWMA covariance (λ=0.94) → 1-day VaR_95 via **Cornish-Fisher** (skew/kurtosis-adjusted, matching the production alert trigger; falls back to the normal quantile when < 21 days align)
   - CAPM beta vs SPY
   - HHI (position concentration)
6. **Exit orders (bypass all risk gates)** — evaluated before any new entry:
   - **Signal exit**: a held position whose *raw or robust* signal is SELL is closed (asymmetric on purpose — exit fast)
   - **Stop-loss**: a held position whose Alpaca unrealised P&L falls below `−stop_loss_pct` (default −15%) is closed
   - **Trailing-stop**: each position's all-time-high price is tracked in a durable watermark file; the position is closed when price falls more than `trailing_stop_pct` (default 15%) below that high
7. **Pre-trade risk gates** — checked before opening any new position (exit/close orders bypass all gates):

   | Gate | Default threshold |
   |---|---|
   | Market open (Alpaca `/v2/clock`) | must be open |
   | Max open positions | 10 |
   | Portfolio 1-day VaR_95 (Cornish-Fisher) | > 5% |
   | Portfolio beta vs SPY | > 1.5× |
   | Equity drawdown from `starting_equity` | > 10% |
   | VIX close | > 25 |

8. **Order entry & sizing** —
   - A new BUY requires a **robust BUY** signal — the naive t(α) signal alone is not sufficient, which guards against data-snooped false positives from the daily random S&P 500 draw
   - **Earnings avoidance**: a candidate is skipped if its next earnings date is within `earnings_avoid_days` (default 3) calendar days; earnings dates come from `yf.Ticker().calendar` and are cached per-day under `cache/`
   - IR-proportional allocation: each new BUY targets `equity × (IR_i / ΣIR)`, capped at `max_position_pct` (15%); higher-conviction (higher IR) names get a larger slice
   - **Fractional shares**: order quantity is `target_value / price` rounded to 6 decimals (min 0.001 share), so small accounts and high-priced names still get filled
   - Total entries are bounded by available **buying power** — orders that cannot be funded are skipped rather than truncated; unfilled capacity stays as cash
9. **Rebalance trim / top-up** — held auto-promoted BUY positions whose weight has drifted more than `rebalance_threshold` (5%) from target are trimmed or topped up. Trimming a **winner** is dampened by **CGT friction**: the effective trim threshold is inflated by `gain_frac × cgt_rate` (Irish 33%), so unrealised gains are not churned for marginal rebalancing benefit.
10. **Order idempotency** — when live (not dry-run) and the market is open, `cancel_all_orders` clears any stale/open orders before submitting, so an overlapping or double-scheduled run can't double-fill.

### NAV tracking & performance

After each run the engine appends a `{date, equity, cash}` snapshot to a durable NAV history (same-day snapshots overwrite). From that history it computes **total return, annualised Sharpe & Sortino (×√252), and max drawdown**, which are surfaced in a Performance section of the paper-trade email.

### Durable state

NAV history, the trade log, and trailing-stop watermarks live under **`data/paper_state/`** (committed by CI), **not** `cache/` — the GitHub Actions cache is ephemeral and evicted, so anything needed across runs for performance metrics must be committed. Files: `paper_nav_history.json`, `paper_trade_log.json`, `paper_highs.json`.

### Watchlist auto-management

The promoted/pruned watchlist **and** the `data/paper_state/` files are committed back to the repo by GitHub Actions after each run (`[skip ci]`; `git pull --rebase` before push to avoid non-fast-forward races). The full daily pipeline (`RUN_MODE=full`) picks up newly promoted tickers on its next run.

### Configuration

All thresholds live in `config/settings.yaml` under `paper_trade:` — no code change needed to tune them.

---

## Walk-Forward Backtest

`RUN_MODE=backtest` runs a rolling walk-forward validation:

| Parameter | Default | Description |
|---|---|---|
| Training window | 756 days (~3 yr) | Used to fit betas, covariance, and FF5 premiums |
| Test window | 63 days (~1 quarter) | Out-of-sample Sharpe, max drawdown, turnover |
| Min asset history | 252 days | Asset excluded from a window if data starts mid-window |

**Bias controls:** FF5 factor premiums restricted to training slice; assets must have data from the first 5 days of the training window; test gaps forward-filled then zero-filled.

**Output flags (always shown):**
- `[!] SURVIVORSHIP BIAS` — universe is current S&P 500 only
- `[OK] No look-ahead` — FF5 premiums restricted to training window

---

## Email Report Contents

### Portfolio Risk Report (`RUN_MODE=full` or `portfolio`)

- RAG status badge (RED / AMBER / GREEN) with active alert list
- Snapshot cards: NAV (€), P&L, VaR 95% (Cornish-Fisher), Sharpe, bond exposure %
- Week-on-week delta indicators (▲/▼ pp)
- Stress test table: EUR loss + % drawdown for COVID / rate shock / GFC
- **Portfolio Factor Exposures** — aggregate β_MKT, β_SMB, β_HML, β_RMW, β_CMA and α for current holdings
- **Performance Attribution** — last 21-day return decomposed into factor contributions + residual α (stock-picking contribution highlighted)
- **Factor Model Quality** — cross-sectional IC mean/IC-IR/p-value (signal validation) + FF5 collinearity diagnostics (VIF, condition number, high-corr pairs)
- **Stock Factor Rankings** — watchlist equities sorted by IR with α, t(α), β_MKT, R², robust_signal, PSR, OOS hit rate, and 63-day attribution driver
- **Signal-Weighted Allocation** — model portfolio from robust BUY signals with score-proportional weights, portfolio FF5 factor exposures, and exposure warnings
- Embedded efficient frontier chart (MC cloud + current + max-Sharpe)
- Rebalancing suggestions with **decision trace** (α-based "why" under each ticker)
- Diversification candidates (bonds + gold, sorted by ΔSharpe, with FX rates)
- Monte Carlo table: 21-day VaR/CVaR/drawdown/loss-probabilities, Sharpe/Sortino, costs
- Correlation breakdown: normal σ vs crisis σ, diversification decay ratio
- Top-10 equity positions
- Bond holdings (coupon, maturity, weight, P&L %)
- Data quality footer

### Alert Emails (`RUN_MODE=alert_check`)

Fires a separate bilingual alert email for each RED metric not already sent in the last 24 hours (`cache/alert_dedup.json`).

---

## Data Quality Checks

| Check | Threshold | Where |
|---|---|---|
| Price staleness | Last close > 4 calendar days old → WARNING | `cleaner.py`, `price_loader.py` |
| Data gaps | > 10 missing trading days in window → WARNING | `cleaner.py` |
| Price spikes | Single-day move > 50% → WARNING | `cleaner.py` |
| FX staleness | FX rate > 4 days old → WARNING (still used) | `price_loader.py` |
| Cache staleness | Snapshot fallback > 2 days old → WARNING | `cache.py` |
| Corrupt cache | Invalid JSON silently skipped; next file tried | `cache.py` |
| Split/dividend adjustment | `auto_adjust=True` explicit on all yfinance calls | `price_loader.py`, `fetcher.py`, `spy_universe.py` |

---

## Watchlist

`config/watchlist.csv` — edit and commit; changes take effect on next run.

```csv
ticker,weight,notes,asset_class,currency
AAPL,1.0,Apple Inc,equity,USD
MSFT,1.0,Microsoft Corp,equity,USD
NVDA,1.0,NVIDIA Corp,equity,USD
TLT,1.0,iShares 20+ Year Treasury Bond ETF,bond,USD
GLD,1.0,SPDR Gold Shares,gold,USD
```

- `asset_class=equity` rows are included in the FF5 factor ranking
- `asset_class=bond` and `gold` rows are evaluated as diversification candidates
- `notes` becomes the display name in the candidates table
- Rows with `notes` starting with `FF5 signal=` are auto-promoted by the paper trading engine and may be auto-pruned; all other rows are treated as manual and never auto-removed

---

## Configuration

### `config/settings.yaml`

```yaml
limits:
  history_days: 730

factor_model:
  window_days: 252        # OLS regression window; falls back if fewer days available
  min_window_days: 63     # skip ticker if aligned days < this

monte_carlo:
  n_paths:  50000
  horizon:  21
  model:    hawkes   # hawkes | garch | gbm

paper_trade:
  universe_size: 50        # permanent watchlist + random S&P 500 fill
  max_position_pct: 0.15   # hard cap per position (fraction of equity)
  max_positions: 10        # halt new buys above this count
  var_limit_pct: 0.05      # halt new buys if portfolio Cornish-Fisher VaR_95 > 5%
  max_portfolio_beta: 1.5  # halt new buys if portfolio beta > 1.5
  max_drawdown_halt: 0.10  # halt new buys if equity down > 10% from starting_equity
  starting_equity: 100000  # Alpaca paper account starting equity (USD)
  stop_loss_pct: 0.15      # close any position whose unrealised loss exceeds 15%
  trailing_stop_pct: 0.15  # close if price falls >15% from the position's all-time high
  rebalance_threshold: 0.05 # trim/top-up held BUY positions when weight drifts > 5%
  cgt_rate: 0.33           # Irish CGT rate; inflates effective trim threshold on winners
  vix_halt_threshold: 25   # halt new buys when VIX closes above this level
  earnings_avoid_days: 3   # skip opening a position if earnings within this many days
```

### `config/thresholds.yaml`

Per-metric alert thresholds, severity levels, and bilingual labels. All configurable without code changes.

### `config/volatility.yaml`

EWMA lambda (0.94), GARCH parameters, lookback windows, ECB risk-free rate (3.5%).

---

## Environment Variables

### Required

| Variable | Purpose |
|---|---|
| `SMTP_HOST` | Outbound mail server (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | Usually `587` |
| `SMTP_USER` | Sender email address |
| `SMTP_PASS` | Gmail app password or SMTP password |
| `REPORT_TO_EMAIL` | Portfolio report recipient |
| `RECIPIENT_EMAIL` | Additional recipient |

### Recommended

| Variable | Purpose |
|---|---|
| `SENDGRID_API_KEY` | SendGrid fallback if SMTP fails |
| `ALPHA_VANTAGE_KEY` | Optional price data enrichment |

### Broker APIs (at least one needed for live portfolio)

| Variable | Platform |
|---|---|
| `IBKR_FLEX_TOKEN` + `IBKR_QUERY_ID` | Interactive Brokers Flex Query |
| `T212_API_KEY` + `T212_API_SECRET` | Trading 212 REST API |
| `ETORO_API_KEY` | eToro (mock/CSV if absent) |

If all broker APIs fail the pipeline falls back to the most recent local snapshot in `cache/`.

### Paper Trading (Alpaca — free)

| Variable | Purpose |
|---|---|
| `ALPACA_API_KEY` | Alpaca paper trading key ID |
| `ALPACA_SECRET_KEY` | Alpaca paper trading secret key |
| `PAPER_DRY_RUN` | `true` → screen only, no orders placed (default `false`) |

Paper trading is skipped gracefully if these are not set (screening still runs).

---

## Quick Start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in SMTP_*, REPORT_TO_EMAIL at minimum

# 3. Full pipeline after US market close
RUN_MODE=full python main.py

# 4. Portfolio risk only (skip watchlist factor ranking)
RUN_MODE=portfolio python main.py

# 5. Alert check only (no report)
RUN_MODE=alert_check python main.py

# 6. Walk-forward backtest
RUN_MODE=backtest python main.py

# 7. Paper trade (dry-run — screen only, no Alpaca orders)
PAPER_DRY_RUN=true RUN_MODE=paper_trade python main.py

# 8. Paper trade (real — places Alpaca paper orders)
RUN_MODE=paper_trade python main.py

# 9. Local scheduler (full pipeline 21:30 UTC, alerts every 4h)
python scheduler.py
```

---

## Testing

```bash
pytest -q tests/
```

| File | Coverage |
|---|---|
| `tests/test_risk_math.py` | Core risk math in `risk/risk_engine.py` — Cornish-Fisher quantile & VaR (incl. quantile-dependent kurtosis behaviour), EWMA VaR/variance series, EVT/GPD VaR, historical VaR/CVaR, max drawdown, HHI, Sharpe. Synthetic data only — no network. |
| `tests/test_paper_engine.py` | Paper-trade order generation — every risk gate (positions/VaR/beta/drawdown/VIX), IR-proportional sizing (cap + budget), no-buys-when-blocked / market-closed, SELL exit, CGT-friction trim suppression, performance metrics, earnings short-circuit. Runs with `dry_run=True`. |

The same `pytest -q tests/` step runs in CI on every push/PR before the import smoke test.

---

## GitHub Actions

### Workflows

| File | Trigger | What it does |
|---|---|---|
| `daily_run.yml` | Cron × 3/day + manual dispatch | Resolves `RUN_MODE` from UTC hour (alert_check / paper_trade / full); runs `main.py`; sends bilingual report; commits `watchlist.csv` + `data/paper_state/` on paper_trade runs (`contents: write`) |
| `paper_trade.yml` | Cron 22:00 UTC Mon–Fri + manual dispatch | Paper trading pipeline; commits updated `watchlist.csv` + `data/paper_state/` back to repo (`git pull --rebase` before push) |
| `ci.yml` | Push/PR to main or master | Syntax-checks all Python files; runs **pytest** (`tests/`); import smoke test including factor model, attribution, decision trace, and report section assertions |

### Cron Schedule

| UTC time | Dublin (IST) | New York (ET) | Mode |
|---|---|---|---|
| 13:45 | 14:45 | 09:45 | `paper_trade` (15 min after open) |
| 14:00 | 15:00 | 10:00 | `alert_check` |
| 21:30 | 22:30 | 17:30 | `full` |
| 22:00 | 23:00 | 18:00 | `paper_trade` (refresh; market-gated) |

### Setup

1. Go to **Settings → Secrets and variables → Actions**
2. Add each variable from the environment table as a repository secret
3. Push to `master` — CI runs automatically; daily analysis starts from the next scheduled cron

**Manual trigger:** Actions → Daily Stock & Portfolio Analysis → Run workflow → choose a run mode.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data | `yfinance`, `pandas`, `numpy`, `pandas-datareader` (FF5 factors via Ken French library) |
| Factor model | `numpy` (OLS), FF5 regression, performance attribution |
| Indicators | `ta` (0.11+) |
| Risk models | `scipy` (EVT/GPD, SLSQP), `arch` (GARCH) |
| Monte Carlo | `numpy` vectorised DCC-GARCH + Hawkes (50 000 paths in ~1 s) |
| Optimisation | `scipy.optimize.minimize` (SLSQP — Markowitz max-Sharpe) |
| Charts | `matplotlib` (headless Agg, CJK font fallback — CID-embedded PNG) |
| Backtest | `vectorbt` (optional, SMA crossover), walk-forward (built-in) |
| Email | `smtplib` (stdlib), `sendgrid` |
| Paper trading | Alpaca paper REST API (`requests`) — fractional orders, market clock |
| Testing | `pytest` (risk math + paper-engine unit tests, run in CI) |
| Scheduling | `APScheduler` / GitHub Actions |
| Config | `PyYAML`, `python-dotenv` |
