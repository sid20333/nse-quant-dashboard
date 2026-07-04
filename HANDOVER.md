# Handover: NSE Quant Engine — Real-Data Backtest

## What this project is

A modular Python quant engine for Indian equities (NSE, scoped to
mid-cap/large-cap only), built incrementally through a design conversation
with Claude (chat). All strategy logic, valuation math, and risk
constraints below were deliberately reasoned through, not defaults — see
each module's docstring for the theory and known limitations before
changing anything.

**The one thing this handover exists to fix:** every backtest run so far
used `SyntheticDataProvider` (a random-walk fake data generator) because
Claude-in-chat has no network access to real financial data sources
(Yahoo Finance and Stooq both block server-side/robots-disallowed
fetching; NSE requires session cookies). **You (Claude Code, running
locally with real network access) do not have this restriction.** The
immediate task is to swap in real data and rerun.

## Current state — what's built and tested

All files are in `quant_engine/`. Everything below has been run
end-to-end on synthetic data with no errors — the code is mechanically
sound, but **zero conclusions about real profitability exist yet.**

| File | Status | Purpose |
|---|---|---|
| `data_provider.py` | Working (synthetic + Breeze wrapper drafted, untested) | Data source abstraction |
| `valuation.py` | Tested | DCF, Graham's formula, EPV → blended Intrinsic Value |
| `technical.py` | Tested | Multi-timeframe Bollinger, Keltner, triple-squeeze detector |
| `zones.py` | Tested | Support/resistance clustering, order block detection |
| `vwap.py` | Tested | Anchored VWAP from swing low / earnings date |
| `knowledge_base.py` | Tested (stub data source) | Red-flag filter (promoter pledge, earnings decline, audit flags) |
| `catalysts.py` | Tested (stub data source) | FII/DII trend, promoter buying, bulk/block deal tagging (informational, non-gating) |
| `moving_average_screener.py` | Tested | SMA(20/50/200) crossovers + RSI(14) → single `bullish_score` |
| `engine.py` | Tested | Combines valuation/KB/trend/zone/volatility gates into `evaluate_stock()` |
| `backtest.py` | Tested | General walk-forward backtester with look-ahead controls |
| `backtests/run_ma_rsi_backtest_v1.py` | Tested (synthetic only) | Equal-weight sizing, rank-based exits |
| `backtests/run_ma_rsi_backtest_v2.py` | Tested (synthetic only) | + vol-weighted sizing + correlation cap |
| `backtests/run_ma_rsi_backtest_v3.py` | Tested (synthetic only) | + trailing-stop exits (let winners run), 60-stock universe, 6-window testing |
| `backtests/run_ma_rsi_backtest_v4.py` | Tested (synthetic only) | + two-stage stop (flat %) + market regime filter |
| `backtests/run_ma_rsi_backtest_v5.py` | Tested (synthetic only), CURRENT BEST | + ATR-scaled two-stage stop (fixes v4's flat-% whipsaw problem) |

## IMPORTANT: two unresolved flags from the synthetic multi-window testing

These are not solved yet and need real-data investigation, not just a
"looks good" read of v5's headline numbers:

### 1. The market regime filter has never been validated — it literally cannot be on this data

Across all 3 versions that included it (v4, v5), `regime_blocked_weeks`
was **0 out of 13 every single window.** This is because
`SyntheticDataProvider` builds each stock from an INDEPENDENT random
walk — averaging 60 independent walks produces a smooth, always-rising
index with no correlated crash behavior. A regime filter is testing
"does the whole market fall together sometimes" — real NSE data has
this property (that's the whole point of an index), synthetic data by
construction does not. **Do not treat the regime filter as validated
just because v5's other numbers improved.** It has had zero opportunity
to do anything in every test so far. Once real index data
(Nifty Midcap150 or Nifty100, matching your universe) is wired in,
check `regime_blocked_weeks` specifically — if it's still 0 across
real historical windows that included known drawdowns (e.g. any
2025-2026 correction), something is wrong with the regime logic itself,
not the data.

### 2. v5's trade frequency is very high — investigate before trusting the return number

v5 (ATR-scaled stops) produced the best return/Sharpe numbers so far
(mean +4.53% per 3-month window, Sharpe 3.86), but generated
**78-92 trades per 13-week window** across a max-15-position portfolio —
roughly 6 trades/week. This could mean:
  (a) the ATR-based stop is working as intended, cutting bad entries
      fast and compounding small wins repeatedly, or
  (b) `ATR_INITIAL_MULT = 1.5` is still tight enough to cause whipsaw
      overtrading, and the improved return is partly an artifact of
      this specific synthetic volatility profile, not a real edge.
This cannot be distinguished on fake data. When real data is in: check
whether trade frequency stays this high, whether `COST_PCT`/
`SLIPPAGE_PCT` (currently rough estimates, not measured) meaningfully
eat into returns at this turnover, and whether widening
`ATR_INITIAL_MULT` (e.g. to 2.0-2.5) changes the picture. Report actual
turnover to the user before presenting any headline return number from
this version.

## THE IMMEDIATE TASK

Get `run_ma_rsi_backtest_v2.py` running on **real NSE data**, under these
exact constraints (given directly by the user, do not change without
asking):

- **Capital:** Rs 5,00,000
- **Position count:** minimum 5, maximum 15 held simultaneously
- **Long only** — no shorting
- **Test window:** 3 months (pick a real, recent 3-month window that
  includes some genuine volatility — not a cherry-picked calm quarter —
  and confirm the choice with the user before running if ambiguous)
- **Universe:** NSE mid-cap and large-cap only (user explicitly excluded
  small-cap/BSE-wide after discussing liquidity problems). Use NIFTY 100
  (large-cap) + NIFTY Midcap 150 constituents, or a reasonable subset if
  full coverage is impractical — confirm list source with the user.

### Step 1 — Build a real data provider

Two options, in order of ease:

**Option A (recommended, no auth headache): `yfinance`**
```bash
pip install yfinance
```
```python
import yfinance as yf
df = yf.download("RELIANCE.NS", start="2025-01-01", end="2026-06-30")
```
Write a `YFinanceDataProvider(DataProvider)` in `data_provider.py`
matching the existing interface (`get_daily_ohlcv(symbol, from_date,
to_date) -> DataFrame[date, open, high, low, close, volume]`). Yahoo
blocks *server-side/robots-respecting* fetches (which is why Claude-in-chat
couldn't do this), but `yfinance` running from a real local Python
process with a normal user-agent works fine as of this writing — verify
this is still true when you run it.

**Option B: Breeze Connect** (the user's existing ICICI Direct API,
already used for their portfolio tracker). `BreezeDataProvider` is
already drafted in `data_provider.py` but **untested against a live
session** — the user will need to supply `api_key`, `api_secret`, and
generate a fresh daily `session_token` via ICICI Direct login. Prefer
this only if yfinance turns out to be unreliable, since it requires
daily manual session regeneration.

### Step 2 — Get a real mid-cap/large-cap symbol list

Fetch or source the current NIFTY 100 + NIFTY Midcap 150 constituent
lists (NSE publishes index factsheets; also check if `nsepython`/`nse`
packages expose an index-constituents method). Cross-check count and a
few sample tickers with the user before running a multi-hour backtest
on a possibly-wrong universe.

### Step 3 — Warmup period

The 200-day SMA needs ~200+ trading days of history *before* the
3-month test window starts. Pull at least 12-14 months of history total
per symbol (matches what `run_ma_rsi_backtest_v2.py` already does with
`HIST_START`).

### Step 4 — Run `backtests/run_ma_rsi_backtest_v5.py` with real data

This is the current best version (ATR-scaled two-stage stop + market
regime filter + correlation-capped, vol-weighted sizing + 60-stock
universe + 6-window testing harness). Swap the `SyntheticDataProvider`
block for the new real provider, keep everything else (position sizing,
correlation cap, MIN/MAX_POSITIONS, capital, ATR stop multiples) unchanged
on the first run, and run across several real 3-month windows (the script
already loops over `TEST_WINDOWS` — replace these with real recent
quarters, ideally including at least one known correction/drawdown period
so the regime filter has a chance to actually do something). Report back:
- Total return, Sharpe, max drawdown, win rate — same metrics already
  printed by the script
- **How many weeks (if any) hit the forced-inclusion path** (fewer than
  5 genuinely bullish setups) — this is a real, expected possibility on
  real data that never triggered on synthetic data, and the user should
  know if/how often it happened
- **Whether the correlation cap actually did anything this time** — on
  synthetic data it skipped 0 candidates because fake stocks are
  uncorrelated by construction; real NSE mid/large-caps should show real
  correlation, so this is the first real test of whether that feature
  earns its keep

### Step 5 — Be honest about the result

Whatever number comes out, the user has been walked through, at length,
why backtest results (even on real data) can be optimistic due to:
survivorship bias (today's index constituents didn't include anything
delisted/failed during the window), transaction cost assumptions
(`COST_PCT`, `SLIPPAGE_PCT` in the script are estimates), and a single
3-month window being a small, possibly-lucky/unlucky sample. Report the
number, then repeat these caveats — don't let a good-looking number go
unqualified, and don't let a bad one go unexplained either.

## Design principles established during the build (preserve these)

- **No look-ahead bias**, ever: signals must only see data up to the
  decision date; entries fill at the *next* bar's open, not the
  signal bar's close.
- **Gates over scores** for the qualitative layers (valuation, KB) — a
  stock either clears a bar or it doesn't; only the catalyst tag is
  informational/non-gating, by design, because it's too soft to trust as
  a hard filter.
- **Every module's docstring explains the theory AND the known failure
  mode.** The user cares about understanding *why*, not just getting a
  working script — keep that pattern if you extend anything.
- **Don't fabricate confidence.** Every synthetic-data result in this
  project has been explicitly caveated as "proves the code runs, proves
  nothing about profitability." Keep doing this until real-data results
  exist, and even then, keep caveating appropriately (small sample size,
  survivorship bias, etc).

## Open threads / things the user may ask about next

- Quality screen module (ROE/ROCE/debt-trend) discussed at length but
  **not yet built** — sits conceptually between a liquidity filter and
  the valuation gate in `engine.py`.
- `catalysts.py` and `knowledge_base.py` both currently use
  `StaticKnowledgeBase`/`StaticCatalystProvider` (manually populated,
  CSV-backed) — live NSE/BSE automation was discussed but explicitly
  deferred until the core strategy is validated on real data.
- `nse` (PyPI) / `nsepython` were identified as the free unofficial
  Python wrappers for NSE announcements, bulk/block deals, and
  shareholding pattern data if/when the KB and catalyst layers get
  automated — both rate-limit to ~3 req/sec and are known to block
  cloud/server IPs, so run them from a local machine, not a hosted
  service.
