# Quant Engine — Weekly Swing/Value Recommendation System

A modular Python engine combining fundamental valuation, a qualitative
red-flag filter, trend confirmation, structural price zones, and
volatility-compression timing into a single weekly stock scan for the
Indian market — built to be studied, not just run.

## Why it's built this way

The original spec asked for five things to work together: Intrinsic Value
vs Market Value, a Knowledge Base of red flags, Anchored VWAP, structured
demand zones, and a triple-timeframe Bollinger squeeze. Rather than one
monolithic script, each concept is its own module with its own docstring
explaining **the theory, the formula, and where it can mislead you**. Read
each module's top-of-file docstring — that's the "study" part.

## Pipeline order (engine.py)

```
1. Valuation gate       -> is it cheap? (DCF / Graham / EPV vs Market Value)
2. Knowledge Base gate  -> is it cheap for a BAD reason? (red flags disqualify)
3. Trend gate           -> has smart money started buying? (Anchored VWAP hold)
4. Zone gate            -> is price at a structurally significant level?
5. Volatility gate      -> is a big move imminent? (triple BB squeeze)
```

Gates are cheapest-first: valuation and KB checks are near-instant, so bad
stocks are discarded before the more expensive pattern-detection scans run
across their full price history.

## Files

| File | Purpose |
|---|---|
| `data_provider.py` | Breeze API wrapper (for live data) + synthetic generator (for testing) |
| `valuation.py` | DCF, Graham's formula, EPV -> blended Intrinsic Value + Margin of Safety |
| `technical.py` | Multi-timeframe Bollinger Bands, Keltner Channels, triple-squeeze detector |
| `zones.py` | Support/resistance clustering, Drop-Base-Rally order block detection |
| `vwap.py` | Swing-low / earnings-date anchor detection, Anchored VWAP |
| `knowledge_base.py` | Red-flag filter interface (promoter pledging, earnings decline, audit flags) |
| `catalysts.py` | FII/DII shareholding trend, promoter buying, bulk/block deal tagging (informational, non-gating) |
| `engine.py` | Combines all five gates into `evaluate_stock()` / `run_weekly_scan()`, attaches catalyst tag |
| `backtest.py` | Walk-forward backtester with look-ahead-bias controls |

## Honest limitations (read this before trusting any output)

1. **DCF is extremely sensitive to inputs.** A 1-2% change in discount rate
   can swing Intrinsic Value by 15-20%+. Never trust a single DCF number —
   always look at `agreement_spread` between DCF/Graham/EPV, and ideally
   run DCF across a range of discount-rate/growth assumptions.

2. **Order blocks and demand zones are retail heuristics, not rigorously
   validated concepts.** They come from "Smart Money Concepts" trading
   content, not institutional academic literature. `zones.py` implements
   them as a transparent, tunable state machine specifically so you can
   backtest which threshold values (if any) actually produce an edge on
   your universe, rather than trusting the defaults.

3. **The Knowledge Base here is a stub interface, not a live scraper.**
   I don't have network access to NSE/BSE from this environment. Populate
   `StaticKnowledgeBase` from a CSV you export from screener.in/Tickertape
   to start, then automate later with `nsepython` or BSE's announcement
   feed (notes are in `knowledge_base.py`).

4. **More simultaneous filters = fewer signals, and a higher chance the
   in-sample backtest looks better than live results will.** Five
   independent-ish gates all aligning is rare almost by construction.
   Don't be surprised if your investable universe (Nifty 500) produces
   only a handful of fully-qualified signals per month. Treat a "clean"
   backtest on a small trade count with real suspicion.

5. **Backtest gaps:** no realistic per-stock slippage modeling for
   illiquid names, no STT/brokerage/tax modeling (add via
   `cost_pct_per_trade`), and — most importantly — **survivorship bias**
   if you backtest against today's Nifty 500 constituents rather than a
   point-in-time list. This can meaningfully inflate results.

6. **A squeeze tells you volatility is compressed, not which direction the
   move will go.** Direction has to come from the valuation + trend layers.
   Never treat `passed_volatility_trigger=True` alone as a buy signal.

## Getting real data flowing

Your existing Breeze Connect setup (from the portfolio tracker) plugs
straight in:

```python
from breeze_connect import BreezeConnect
from quant_engine.data_provider import BreezeDataProvider

breeze = BreezeConnect(api_key="...")
breeze.generate_session(api_secret="...", session_token="...")  # daily re-login required
provider = BreezeDataProvider(breeze)
df = provider.get_daily_ohlcv("RELIANCE", "2020-01-01", "2026-06-30")
```

## Suggested study order

1. Run `test_synthetic.py` and read the output alongside each module's
   docstring — see what each layer actually computes on fake-but-plausible
   data.
2. Swap in 2-3 real Nifty stocks via Breeze, and manually sanity check the
   valuation numbers against screener.in's own DCF/intrinsic value figures.
3. Populate a small `StaticKnowledgeBase` CSV for ~20 stocks you know well.
4. Run `run_weekly_scan()` against that small universe weekly for a month
   and track how often it actually fires, before trusting a backtest.
5. Only then run `walk_forward_backtest()` over multi-year history, and be
   skeptical of any result built on fewer than ~30-50 trades.
