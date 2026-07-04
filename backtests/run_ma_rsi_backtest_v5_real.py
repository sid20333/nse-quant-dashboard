"""
run_ma_rsi_backtest_v5_real.py — v5 strategy (ATR-scaled two-stage stop +
market-regime filter + correlation-capped, vol-weighted sizing) run on REAL
NSE data via YFinanceDataProvider, per HANDOVER.md.

This is a line-for-line port of run_ma_rsi_backtest_v5.py. The ONLY things
changed from the synthetic version, deliberately, are:
  1. Data source: SyntheticDataProvider -> YFinanceDataProvider (real NSE
     daily OHLCV, split/div-adjusted, disk-cached).
  2. Universe: 60 independent random walks -> real NIFTY 100 + Midcap 150
     subset (see nse_universe.py).
  3. Regime index: synthetic mean-of-walks stand-in -> real ^CNX100
     (NIFTY 100) close series. THIS is the whole point — a real index has
     correlated drawdowns the synthetic data could not produce, so the
     regime filter finally has something to detect.
  4. Added instrumentation the handover asked for: correlation-cap skip
     count, per-window regime-blocked weeks against real drawdowns, and
     forced-inclusion (fewer than MIN_POSITIONS genuine setups) events.

Strategy parameters (capital, position count, cost/slippage, ATR stop
multiples, correlation cap) are UNCHANGED from v5 on this first real run, as
instructed. Do not tune them until the honest baseline is on the table.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")

import pandas as pd
import numpy as np

from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.moving_average_screener import compute_ma_state
from quant_engine.technical import average_true_range
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

HIST_START = "2023-01-01"          # ~2 yrs warmup before first test window (200-SMA needs it)
DATA_END = "2026-06-30"
CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
MIN_HISTORY_ROWS = 500             # drop anything without enough clean history

CAPITAL = 500_000.0
MIN_POSITIONS = 5
MAX_POSITIONS = 15
COST_PCT = 0.001                   # ESTIMATE (brokerage+taxes), not measured
SLIPPAGE_PCT = 0.0015              # ESTIMATE
ATR_INITIAL_MULT = 1.5
ATR_PROFIT_LOCK_MULT = 1.0
ATR_TRAILING_MULT = 2.0
CORR_LOOKBACK_DAYS = 60
MAX_PAIRWISE_CORRELATION = 0.70
REGIME_SMA_WINDOW = 200

# Six real, consecutive recent quarters (covers 1.5 yrs of actual NSE tape,
# including whatever drawdowns genuinely occurred — not cherry-picked).
TEST_WINDOWS = [
    ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-06-30"),
]

provider = YFinanceDataProvider(cache_dir=CACHE_DIR)

print(f"Fetching real NSE universe ({HIST_START} to {DATA_END}) via yfinance...")
symbol_dfs = {}
skipped = []
for sym in UNIVERSE:
    try:
        df = provider.get_daily_ohlcv(sym, HIST_START, DATA_END)
    except Exception as e:
        skipped.append((sym, str(e)[:50]))
        continue
    if len(df) < MIN_HISTORY_ROWS:
        skipped.append((sym, f"only {len(df)} rows"))
        continue
    symbol_dfs[sym] = df.reset_index(drop=True)

print(f"  usable symbols: {len(symbol_dfs)}   skipped: {len(skipped)}")
if skipped:
    print("  skipped:", ", ".join(f"{s}({r})" for s, r in skipped))

print(f"Fetching real regime index {REGIME_INDEX} (NIFTY 100)...")
index_df = provider.get_daily_ohlcv(REGIME_INDEX, HIST_START, DATA_END)
index_series = index_df.set_index("date")["close"]
index_sma200 = index_series.rolling(REGIME_SMA_WINDOW).mean()

atr_cache = {
    sym: average_true_range(df["high"], df["low"], df["close"], 14)
    for sym, df in symbol_dfs.items()
}
print("Done.\n")


def regime_bullish(as_of_date) -> bool:
    idx_slice = index_series[index_series.index <= as_of_date]
    sma_slice = index_sma200[index_sma200.index <= as_of_date]
    if idx_slice.empty or sma_slice.empty or pd.isna(sma_slice.iloc[-1]):
        return True
    return idx_slice.iloc[-1] > sma_slice.iloc[-1]


def atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    a = average_true_range(df["high"], df["low"], df["close"], window)
    price = df["close"].iloc[-1]
    return float(a.iloc[-1] / price) if price > 0 else np.nan


def get_atr_at(symbol: str, as_of_date) -> float:
    series = atr_cache[symbol]
    df = symbol_dfs[symbol]
    idx = df.index[df["date"] <= as_of_date]
    if len(idx) == 0:
        return np.nan
    return series.iloc[idx[-1]]


def returns_series(df: pd.DataFrame, as_of, lookback: int = CORR_LOOKBACK_DAYS) -> pd.Series:
    sliced = df[df["date"] <= as_of].tail(lookback)
    return sliced.set_index("date")["close"].pct_change().dropna()


def select_with_correlation_cap(ranked_symbols, symbol_dfs, decision_date, max_n, max_corr,
                                already_held, skip_counter):
    selected = list(already_held)
    cache = {}

    def get_returns(sym):
        if sym not in cache:
            cache[sym] = returns_series(symbol_dfs[sym], decision_date)
        return cache[sym]

    for sym in ranked_symbols:
        if len(selected) >= max_n:
            break
        if sym in already_held:
            continue
        candidate_returns = get_returns(sym)
        if candidate_returns.empty:
            continue
        too_correlated = False
        for held in selected:
            held_returns = get_returns(held)
            joined = pd.concat([candidate_returns, held_returns], axis=1).dropna()
            if len(joined) < 10:
                continue
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            if pd.notna(corr) and corr > max_corr:
                too_correlated = True
                break
        if too_correlated:
            skip_counter[0] += 1
            continue
        selected.append(sym)
    return [s for s in selected if s not in already_held]


def run_backtest_window(start_date: str, end_date: str) -> dict:
    decision_dates = pd.date_range(start_date, end_date, freq="W-SAT")

    positions = {}
    cash = CAPITAL
    equity_curve, equity_dates, position_count_history = [], [], []
    trade_log = []
    forced_inclusion_events = 0
    regime_blocked_weeks = 0
    corr_skips = [0]
    total_weeks = len(decision_dates)

    for idx_pos, decision_date in enumerate(decision_dates):
        prior_decision = decision_dates[idx_pos - 1] if idx_pos > 0 else pd.to_datetime(start_date) - pd.Timedelta(days=1)

        for symbol in list(positions.keys()):
            df = symbol_dfs[symbol]
            daily_bars = df[(df["date"] > prior_decision) & (df["date"] <= decision_date)]
            pos = positions[symbol]
            for _, bar in daily_bars.iterrows():
                pos["peak_price"] = max(pos["peak_price"], bar["close"])
                entry_atr = pos["entry_atr"]

                profit_lock_level = pos["entry_price"] + ATR_PROFIT_LOCK_MULT * entry_atr
                if pos["peak_price"] >= profit_lock_level:
                    stop_level = pos["peak_price"] - ATR_TRAILING_MULT * entry_atr
                    reason = "trailing_stop"
                else:
                    stop_level = pos["entry_price"] - ATR_INITIAL_MULT * entry_atr
                    reason = "initial_stop"

                if bar["close"] <= stop_level:
                    exit_price = stop_level * (1 - SLIPPAGE_PCT)
                    proceeds = pos["shares"] * exit_price * (1 - COST_PCT)
                    cash += proceeds
                    ret = (exit_price * (1 - COST_PCT) - pos["entry_price"]) / pos["entry_price"]
                    trade_log.append({"symbol": symbol, "entry_date": pos["entry_date"], "exit_date": bar["date"],
                                       "entry_price": pos["entry_price"], "exit_price": exit_price,
                                       "return_pct": ret, "exit_reason": reason})
                    del positions[symbol]
                    break

        scores = {}
        for symbol, df in symbol_dfs.items():
            sliced = df[df["date"] <= decision_date]
            if len(sliced) < 210:
                continue
            state = compute_ma_state(sliced)
            scores[symbol] = (state.bullish_score, state.death_cross_50_200)

        for symbol in list(positions.keys()):
            if symbol in scores and scores[symbol][1]:
                df = symbol_dfs[symbol]
                exit_bars = df[df["date"] > decision_date]
                if exit_bars.empty:
                    continue
                exit_price = exit_bars.iloc[0]["open"] * (1 - SLIPPAGE_PCT)
                pos = positions.pop(symbol)
                proceeds = pos["shares"] * exit_price * (1 - COST_PCT)
                cash += proceeds
                ret = (exit_price * (1 - COST_PCT) - pos["entry_price"]) / pos["entry_price"]
                trade_log.append({"symbol": symbol, "entry_date": pos["entry_date"], "exit_date": exit_bars.iloc[0]["date"],
                                   "entry_price": pos["entry_price"], "exit_price": exit_price,
                                   "return_pct": ret, "exit_reason": "death_cross"})

        market_ok = regime_bullish(decision_date)
        if not market_ok:
            regime_blocked_weeks += 1

        ranked_all = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)]
        positive_ranked = [s for s in ranked_all if scores[s][0] > 0]

        currently_held = list(positions.keys())
        open_slots = MAX_POSITIONS - len(currently_held)

        new_entries = []
        if market_ok and open_slots > 0:
            candidates = positive_ranked if len(positive_ranked) > 0 else ranked_all
            new_entries = select_with_correlation_cap(candidates, symbol_dfs, decision_date,
                                                      MAX_POSITIONS, MAX_PAIRWISE_CORRELATION,
                                                      currently_held, corr_skips)

        if len(currently_held) + len(new_entries) < MIN_POSITIONS:
            forced_inclusion_events += 1
            for sym in ranked_all:
                if sym not in currently_held and sym not in new_entries:
                    new_entries.append(sym)
                if len(currently_held) + len(new_entries) >= MIN_POSITIONS:
                    break

        if new_entries:
            held_value = sum(
                p["shares"] * symbol_dfs[s][symbol_dfs[s]["date"] <= decision_date]["close"].iloc[-1]
                for s, p in positions.items()
            )
            total_deployable = cash + held_value
            target_total = len(currently_held) + len(new_entries)

            vol_weights = {}
            for sym in new_entries:
                sliced = symbol_dfs[sym][symbol_dfs[sym]["date"] <= decision_date]
                a = atr_pct(sliced)
                vol_weights[sym] = 1.0 / a if a and a > 0 else 0.0
            weight_sum = sum(vol_weights.values()) or 1.0
            per_slot_base = total_deployable / target_total if target_total > 0 else 0

            for symbol in new_entries:
                df = symbol_dfs[symbol]
                entry_bars = df[df["date"] > decision_date]
                if entry_bars.empty:
                    continue
                entry_bar = entry_bars.iloc[0]
                entry_price = entry_bar["open"] * (1 + SLIPPAGE_PCT)
                allocation = min(per_slot_base * (vol_weights[symbol] / (weight_sum / len(new_entries))) if weight_sum > 0 else per_slot_base, cash)
                allocation = min(allocation, cash)
                if allocation <= 0:
                    continue
                shares = allocation / entry_price
                cash -= shares * entry_price * (1 + COST_PCT)
                entry_atr = get_atr_at(symbol, decision_date)
                if pd.isna(entry_atr) or entry_atr <= 0:
                    entry_atr = entry_price * 0.02
                positions[symbol] = {"shares": shares, "entry_price": entry_price, "entry_date": entry_bar["date"],
                                      "peak_price": entry_price, "entry_atr": entry_atr}

        mtm = cash
        for symbol, pos in positions.items():
            df = symbol_dfs[symbol]
            recent = df[df["date"] <= decision_date]
            if not recent.empty:
                mtm += pos["shares"] * recent["close"].iloc[-1]
        equity_curve.append(mtm)
        equity_dates.append(decision_date)
        position_count_history.append(len(positions))

    final_date = pd.to_datetime(end_date)
    for symbol, pos in list(positions.items()):
        df = symbol_dfs[symbol]
        final_bars = df[df["date"] <= final_date]
        if final_bars.empty:
            continue
        exit_price = final_bars.iloc[-1]["close"]
        proceeds = pos["shares"] * exit_price * (1 - COST_PCT)
        cash += proceeds
        ret = (exit_price * (1 - COST_PCT) - pos["entry_price"]) / pos["entry_price"]
        trade_log.append({"symbol": symbol, "entry_date": pos["entry_date"], "exit_date": final_date,
                           "entry_price": pos["entry_price"], "exit_price": exit_price,
                           "return_pct": ret, "exit_reason": "window_end"})

    equity = pd.Series(equity_curve, index=equity_dates)
    final_value = cash
    total_return = (final_value - CAPITAL) / CAPITAL
    weekly_returns = equity.pct_change().dropna()
    sharpe = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52) if weekly_returns.std() > 0 else 0
    running_max = equity.cummax()
    max_dd = ((equity - running_max) / running_max).min() if len(equity) > 0 else 0

    trades_df = pd.DataFrame(trade_log)
    win_rate = (trades_df["return_pct"] > 0).mean() if not trades_df.empty else 0

    return {
        "start": start_date, "end": end_date, "total_return": total_return, "sharpe": sharpe,
        "max_dd": max_dd, "num_trades": len(trades_df), "win_rate": win_rate,
        "forced_inclusion_events": forced_inclusion_events, "regime_blocked_weeks": regime_blocked_weeks,
        "total_weeks": total_weeks, "corr_skips": corr_skips[0],
        "initial_stop_exits": (trades_df["exit_reason"] == "initial_stop").sum() if not trades_df.empty else 0,
        "trailing_stop_exits": (trades_df["exit_reason"] == "trailing_stop").sum() if not trades_df.empty else 0,
        "death_cross_exits": (trades_df["exit_reason"] == "death_cross").sum() if not trades_df.empty else 0,
        "avg_win": trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].mean() if not trades_df.empty else 0,
        "avg_loss": trades_df.loc[trades_df["return_pct"] <= 0, "return_pct"].mean() if not trades_df.empty else 0,
    }


print("=" * 108)
print("MULTI-WINDOW BACKTEST v5 on REAL NSE DATA: ATR-scaled two-stage stop + NIFTY100 regime filter")
print(f"universe={len(symbol_dfs)} names | capital=Rs {CAPITAL:,.0f} | positions {MIN_POSITIONS}-{MAX_POSITIONS} | long-only")
print("=" * 108)

results = []
for start, end in TEST_WINDOWS:
    r = run_backtest_window(start, end)
    results.append(r)
    print(f"{start}->{end}: ret={r['total_return']:>7.2%}  sharpe={r['sharpe']:>5.2f}  "
          f"maxDD={r['max_dd']:>7.2%}  trades={r['num_trades']:>3d}  win={r['win_rate']:>5.1%}  "
          f"init={r['initial_stop_exits']:>2d} trail={r['trailing_stop_exits']:>2d} dcross={r['death_cross_exits']:>2d}  "
          f"regime_blk={r['regime_blocked_weeks']}/{r['total_weeks']}  forced={r['forced_inclusion_events']}  corr_skip={r['corr_skips']}")

returns = [r["total_return"] for r in results]
sharpes = [r["sharpe"] for r in results]
dds = [r["max_dd"] for r in results]

print()
print("=" * 108)
print("SUMMARY — REAL DATA, v5, first run, NO parameter tuning")
print("=" * 108)
print(f"Return -> mean: {np.mean(returns):>7.2%}   median: {np.median(returns):>7.2%}   "
      f"min: {min(returns):>7.2%}   max: {max(returns):>7.2%}")
print(f"Sharpe -> mean: {np.mean(sharpes):>7.2f}   median: {np.median(sharpes):>7.2f}   "
      f"min: {min(sharpes):>7.2f}   max: {max(sharpes):>7.2f}")
print(f"MaxDD  -> mean: {np.mean(dds):>7.2%}   worst: {min(dds):>7.2%}")
print(f"negative-return windows: {sum(1 for r in returns if r < 0)} / {len(returns)}")
print(f"total trades across all windows: {sum(r['num_trades'] for r in results)}   "
      f"avg trades/window: {np.mean([r['num_trades'] for r in results]):.1f}")
print(f"weeks regime-blocked (total): {sum(r['regime_blocked_weeks'] for r in results)} / {sum(r['total_weeks'] for r in results)}")
print(f"forced-inclusion events (total): {sum(r['forced_inclusion_events'] for r in results)}")
print(f"correlation-cap skips (total): {sum(r['corr_skips'] for r in results)}")
