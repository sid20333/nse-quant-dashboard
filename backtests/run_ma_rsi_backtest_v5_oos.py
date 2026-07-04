"""
run_ma_rsi_backtest_v5_oos.py — v5 strategy, REAL data, extended OUT-OF-SAMPLE
run across 2015-2024 (40 quarterly windows) WITH the realistic Indian delivery
cost model (costs.py) instead of the flat 0.1% guess.

Purpose (handover items #3 + #4): the first real run (2025-2026, 6 windows)
showed the strategy's whole edge is CRASH PROTECTION — it beat NIFTY100
buy-and-hold by +16.7% compounded, driven almost entirely by two down-quarters.
Two problems with trusting that:
  #3 costs were a guess (0.1%); real STT alone is 0.2% round-trip.
  #4 six quarters, two of which drive everything, is a tiny sample.

This script re-tests over 10 prior years that contain OTHER real corrections
(2015-16 China/Fed taper, 2018 midcap crash, 2020 COVID, 2022 selloff). If the
"protects in crashes, lags in rallies" signature repeats out-of-sample and
survives real costs, the edge is structural, not two lucky quarters.

REMAINING BIAS (cannot fix here): universe = TODAY's constituents, so it is
still survivorship-biased. Partially mitigated in early windows because recent
IPOs (Polycab 2019, Zomato 2021, etc.) simply have no pre-listing data and are
auto-skipped — the tradeable universe correctly shrinks going back in time.
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
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

HIST_START = "2013-01-01"          # warmup for the first 2015 window's 200-SMA
DATA_END = "2024-12-31"
CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
MIN_HISTORY_ROWS = 250

CAPITAL = 500_000.0
MIN_POSITIONS = 5
MAX_POSITIONS = 15
SLIPPAGE_PCT = 0.0015              # execution assumption (not a statutory charge)
ATR_INITIAL_MULT = 1.5
ATR_PROFIT_LOCK_MULT = 1.0
ATR_TRAILING_MULT = 2.0
CORR_LOOKBACK_DAYS = 60
MAX_PAIRWISE_CORRELATION = 0.70
REGIME_SMA_WINDOW = 200

COSTS = IndianEquityCosts()        # discount-broker delivery defaults
BUY_FRAC = COSTS.buy_cost(1.0)     # buy costs are linear in turnover

# 40 quarterly windows, 2015Q1 .. 2024Q4.
TEST_WINDOWS = []
for yr in range(2015, 2025):
    TEST_WINDOWS += [
        (f"{yr}-01-01", f"{yr}-03-31"),
        (f"{yr}-04-01", f"{yr}-06-30"),
        (f"{yr}-07-01", f"{yr}-09-30"),
        (f"{yr}-10-01", f"{yr}-12-31"),
    ]

provider = YFinanceDataProvider(cache_dir=CACHE_DIR)

print(f"Fetching real NSE universe ({HIST_START} to {DATA_END})...")
symbol_dfs = {}
for sym in UNIVERSE:
    try:
        df = provider.get_daily_ohlcv(sym, HIST_START, DATA_END)
    except Exception:
        continue
    if len(df) >= MIN_HISTORY_ROWS:
        symbol_dfs[sym] = df.reset_index(drop=True)
print(f"  usable symbols (full 2013-2024 pool): {len(symbol_dfs)}")

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


def select_with_correlation_cap(ranked_symbols, decision_date, max_n, max_corr, already_held, skip_counter):
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
            joined = pd.concat([candidate_returns, get_returns(held)], axis=1).dropna()
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


def run_backtest_window(start_date, end_date) -> dict:
    decision_dates = pd.date_range(start_date, end_date, freq="W-SAT")
    positions = {}
    cash = CAPITAL
    equity_curve, equity_dates = [], []
    trade_log = []
    forced_inclusion_events = 0
    regime_blocked_weeks = 0
    corr_skips = [0]

    def sell(pos, shares, ref_price):
        exit_price = ref_price * (1 - SLIPPAGE_PCT)
        turnover = shares * exit_price
        net = turnover - COSTS.sell_cost(turnover)
        ret = (net / shares) / pos["entry_price"] - 1
        return exit_price, net, ret

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
                    exit_price, net, ret = sell(pos, pos["shares"], stop_level)
                    cash += net
                    trade_log.append({"symbol": symbol, "return_pct": ret, "exit_reason": reason})
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
                pos = positions.pop(symbol)
                exit_price, net, ret = sell(pos, pos["shares"], exit_bars.iloc[0]["open"])
                cash += net
                trade_log.append({"symbol": symbol, "return_pct": ret, "exit_reason": "death_cross"})

        market_ok = regime_bullish(decision_date)
        if not market_ok:
            regime_blocked_weeks += 1

        ranked_all = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)]
        positive_ranked = [s for s in ranked_all if scores[s][0] > 0]
        currently_held = list(positions.keys())
        open_slots = MAX_POSITIONS - len(currently_held)

        new_entries = []
        if market_ok and open_slots > 0:
            candidates = positive_ranked if positive_ranked else ranked_all
            new_entries = select_with_correlation_cap(candidates, decision_date, MAX_POSITIONS,
                                                      MAX_PAIRWISE_CORRELATION, currently_held, corr_skips)

        if len(currently_held) + len(new_entries) < MIN_POSITIONS and ranked_all:
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
                allocation = per_slot_base * (vol_weights[symbol] / (weight_sum / len(new_entries))) if weight_sum > 0 else per_slot_base
                # cap so turnover + buy cost never exceeds available cash
                allocation = min(allocation, cash / (1 + BUY_FRAC))
                if allocation <= 0:
                    continue
                shares = allocation / entry_price
                turnover = shares * entry_price
                cash -= turnover + COSTS.buy_cost(turnover)
                entry_atr = get_atr_at(symbol, decision_date)
                if pd.isna(entry_atr) or entry_atr <= 0:
                    entry_atr = entry_price * 0.02
                positions[symbol] = {"shares": shares, "entry_price": entry_price,
                                      "peak_price": entry_price, "entry_atr": entry_atr}

        mtm = cash
        for symbol, pos in positions.items():
            recent = symbol_dfs[symbol][symbol_dfs[symbol]["date"] <= decision_date]
            if not recent.empty:
                mtm += pos["shares"] * recent["close"].iloc[-1]
        equity_curve.append(mtm)
        equity_dates.append(decision_date)

    final_date = pd.to_datetime(end_date)
    for symbol, pos in list(positions.items()):
        final_bars = symbol_dfs[symbol][symbol_dfs[symbol]["date"] <= final_date]
        if final_bars.empty:
            continue
        exit_price, net, ret = sell(pos, pos["shares"], final_bars.iloc[-1]["close"] / (1 - SLIPPAGE_PCT))
        cash += net
        trade_log.append({"symbol": symbol, "return_pct": ret, "exit_reason": "window_end"})

    equity = pd.Series(equity_curve, index=equity_dates)
    total_return = (cash - CAPITAL) / CAPITAL
    weekly_returns = equity.pct_change().dropna()
    sharpe = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52) if weekly_returns.std() > 0 else 0
    running_max = equity.cummax()
    max_dd = ((equity - running_max) / running_max).min() if len(equity) > 0 else 0
    trades_df = pd.DataFrame(trade_log)
    win_rate = (trades_df["return_pct"] > 0).mean() if not trades_df.empty else 0

    # NIFTY100 buy-and-hold over the same window
    seg = index_series[(index_series.index >= pd.to_datetime(start_date)) & (index_series.index <= final_date)]
    bh_return = seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else 0.0

    return {
        "start": start_date, "end": end_date, "total_return": total_return, "sharpe": sharpe,
        "max_dd": max_dd, "num_trades": len(trades_df), "win_rate": win_rate,
        "forced_inclusion_events": forced_inclusion_events, "regime_blocked_weeks": regime_blocked_weeks,
        "corr_skips": corr_skips[0], "bh_return": bh_return, "excess": total_return - bh_return,
        "universe_active": sum(1 for df in symbol_dfs.values()
                               if len(df[df["date"] <= final_date]) >= 210),
    }


print("=" * 118)
print("OUT-OF-SAMPLE 2015-2024 | v5 strategy | REAL data | REAL Indian delivery costs")
print(f"round-trip statutory cost on a Rs {CAPITAL/MAX_POSITIONS:,.0f} trade = "
      f"{COSTS.round_trip_pct(CAPITAL/MAX_POSITIONS):.3%} (+ {2*SLIPPAGE_PCT:.2%} slippage)")
print("=" * 118)
print(f"{'window':<24}{'strat':>8}{'NIFTY100':>10}{'excess':>9}{'sharpe':>8}{'maxDD':>8}"
      f"{'trades':>8}{'win':>7}{'regblk':>8}{'univ':>6}")

results = []
for start, end in TEST_WINDOWS:
    r = run_backtest_window(start, end)
    results.append(r)
    flag = "  <-- CRASH" if r["bh_return"] < -0.05 else ""
    print(f"{start+' -> '+end[5:]:<24}{r['total_return']:>8.2%}{r['bh_return']:>10.2%}"
          f"{r['excess']:>9.2%}{r['sharpe']:>8.2f}{r['max_dd']:>8.2%}{r['num_trades']:>8d}"
          f"{r['win_rate']:>7.1%}{r['regime_blocked_weeks']:>6d}/13{r['universe_active']:>6d}{flag}")

strat = np.array([r["total_return"] for r in results])
bh = np.array([r["bh_return"] for r in results])
excess = np.array([r["excess"] for r in results])
crash_mask = bh < -0.05
up_mask = bh > 0.05

print("\n" + "=" * 118)
print("OUT-OF-SAMPLE SUMMARY (40 quarters, 2015-2024)")
print("=" * 118)
print(f"Strategy   mean/qtr: {strat.mean():>7.2%}   compounded: {np.prod(1+strat)-1:>8.2%}   "
      f"negative qtrs: {(strat<0).sum()}/40")
print(f"NIFTY100   mean/qtr: {bh.mean():>7.2%}   compounded: {np.prod(1+bh)-1:>8.2%}   "
      f"negative qtrs: {(bh<0).sum()}/40")
print(f"Excess     mean/qtr: {excess.mean():>7.2%}   qtrs strategy beat B&H: {(excess>0).sum()}/40")
print()
print(f"CRASH quarters (B&H < -5%): n={crash_mask.sum()}   "
      f"strat mean {strat[crash_mask].mean():>7.2%}  vs B&H {bh[crash_mask].mean():>7.2%}  "
      f"-> excess {excess[crash_mask].mean():>7.2%}")
print(f"RALLY quarters (B&H > +5%): n={up_mask.sum()}   "
      f"strat mean {strat[up_mask].mean():>7.2%}  vs B&H {bh[up_mask].mean():>7.2%}  "
      f"-> excess {excess[up_mask].mean():>7.2%}")
print(f"Sharpe mean: {np.mean([r['sharpe'] for r in results]):.2f}   "
      f"worst window maxDD: {min(r['max_dd'] for r in results):.2%}   "
      f"total trades: {sum(r['num_trades'] for r in results)}")
