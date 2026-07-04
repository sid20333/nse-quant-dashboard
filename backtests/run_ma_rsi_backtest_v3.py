"""
run_ma_rsi_backtest_v3.py — Three structural changes from v2:

1. LET WINNERS RUN (trailing stop instead of rank-based exit)
   v1/v2 exited a position whenever it fell out of the top-15 rank at the
   next weekly rebalance - even a position still up 15% would get cut
   just because something else ranked higher that week. v3 instead:
     - Tracks each position's peak CLOSE price since entry, checked DAILY
       (not just weekly), so a genuine trend isn't held hostage to a
       weekly snapshot.
     - Exits only on: (a) trailing stop breach - price falls
       TRAILING_STOP_PCT from its peak since entry, or (b) a confirmed
       death cross (50-SMA < 200-SMA) at a weekly check, i.e. the
       long-term regime itself has flipped.
     - A new candidate ranking higher than an existing holding does NOT
       force an exit - new entries only fill genuinely open slots
       (position count < MAX_POSITIONS).

2. WIDER UNIVERSE
   25 -> 60 synthetic names, more candidates to rank from at each
   decision point, matching NIFTY100+Midcap150-scale coverage.

3. MULTI-WINDOW TESTING
   The same strategy is run across SIX separate, non-overlapping 3-month
   windows spanning the synthetic history, and results are reported as a
   DISTRIBUTION (mean/std/min/max), not a single number. A strategy is
   not "the 6% one quarter got" - it's the distribution of what it does
   across many different market conditions. Still synthetic data - this
   proves the harness works and shows you HOW to read multi-window
   results; it does not validate real profitability.
"""
import sys
sys.path.insert(0, "/home/claude")

import pandas as pd
import numpy as np

from quant_engine.data_provider import SyntheticDataProvider
from quant_engine.moving_average_screener import compute_ma_state
from quant_engine.technical import average_true_range

UNIVERSE_SIZE = 60
HIST_START = "2023-01-01"
DATA_END = "2026-06-30"

CAPITAL = 500_000.0
MIN_POSITIONS = 5
MAX_POSITIONS = 15
COST_PCT = 0.001
SLIPPAGE_PCT = 0.0015
TRAILING_STOP_PCT = 0.10          # exit if price falls 10% from its peak since entry
CORR_LOOKBACK_DAYS = 60
MAX_PAIRWISE_CORRELATION = 0.70

TEST_WINDOWS = [
    ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-06-30"),
]

print(f"Building universe of {UNIVERSE_SIZE} synthetic stocks ({HIST_START} to {DATA_END})...")
symbol_dfs = {}
for i in range(UNIVERSE_SIZE):
    provider = SyntheticDataProvider(seed=200 + i)
    df = provider.get_daily_ohlcv(f"STOCK{i:03d}", HIST_START, DATA_END)
    symbol_dfs[f"STOCK{i:03d}"] = df
print("Done.\n")


def atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    a = average_true_range(df["high"], df["low"], df["close"], window)
    price = df["close"].iloc[-1]
    return float(a.iloc[-1] / price) if price > 0 else np.nan


def returns_series(df: pd.DataFrame, as_of, lookback: int = CORR_LOOKBACK_DAYS) -> pd.Series:
    sliced = df[df["date"] <= as_of].tail(lookback)
    return sliced.set_index("date")["close"].pct_change().dropna()


def select_with_correlation_cap(ranked_symbols, symbol_dfs, decision_date, max_n, max_corr, already_held):
    """Fills UP TO max_n total (already_held counts toward the cap) without
    forcing out existing holdings - only picks NEW names for open slots."""
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
        if not too_correlated:
            selected.append(sym)
    return [s for s in selected if s not in already_held]  # return only the NEW additions


def run_backtest_window(start_date: str, end_date: str) -> dict:
    decision_dates = pd.date_range(start_date, end_date, freq="W-SAT")
    all_dates = pd.date_range(start_date, end_date, freq="D")

    positions = {}   # symbol -> {shares, entry_price, entry_date, peak_price}
    cash = CAPITAL
    equity_curve, equity_dates, position_count_history = [], [], []
    trade_log = []
    forced_inclusion_events = 0

    for decision_date in decision_dates:
        # --- DAILY trailing-stop and death-cross exit check since last decision date ---
        prior_decision = decision_dates[decision_dates.get_loc(decision_date) - 1] if decision_dates.get_loc(decision_date) > 0 else pd.to_datetime(start_date) - pd.Timedelta(days=1)
        for symbol in list(positions.keys()):
            df = symbol_dfs[symbol]
            daily_bars = df[(df["date"] > prior_decision) & (df["date"] <= decision_date)]
            pos = positions[symbol]
            exited = False
            for _, bar in daily_bars.iterrows():
                pos["peak_price"] = max(pos["peak_price"], bar["close"])
                trailing_stop_level = pos["peak_price"] * (1 - TRAILING_STOP_PCT)
                if bar["close"] <= trailing_stop_level:
                    exit_price = trailing_stop_level * (1 - SLIPPAGE_PCT)
                    proceeds = pos["shares"] * exit_price * (1 - COST_PCT)
                    cash += proceeds
                    ret = (exit_price * (1 - COST_PCT) - pos["entry_price"]) / pos["entry_price"]
                    trade_log.append({"symbol": symbol, "entry_date": pos["entry_date"], "exit_date": bar["date"],
                                       "entry_price": pos["entry_price"], "exit_price": exit_price,
                                       "return_pct": ret, "exit_reason": "trailing_stop"})
                    del positions[symbol]
                    exited = True
                    break
            if exited:
                continue

        # --- Weekly: death-cross regime exit + scoring/ranking ---
        scores = {}
        for symbol, df in symbol_dfs.items():
            sliced = df[df["date"] <= decision_date]
            if len(sliced) < 210:
                continue
            state = compute_ma_state(sliced)
            scores[symbol] = (state.bullish_score, state.death_cross_50_200)

        for symbol in list(positions.keys()):
            if symbol in scores and scores[symbol][1]:  # confirmed death cross on 50/200
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

        ranked_all = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)]
        positive_ranked = [s for s in ranked_all if scores[s][0] > 0]

        currently_held = list(positions.keys())
        open_slots = MAX_POSITIONS - len(currently_held)

        if open_slots > 0:
            candidates = positive_ranked if len(positive_ranked) > 0 else ranked_all
            new_entries = select_with_correlation_cap(candidates, symbol_dfs, decision_date, MAX_POSITIONS, MAX_PAIRWISE_CORRELATION, currently_held)
        else:
            new_entries = []

        # Enforce MIN_POSITIONS floor if we're short (forced inclusion)
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
                positions[symbol] = {"shares": shares, "entry_price": entry_price, "entry_date": entry_bar["date"], "peak_price": entry_price}

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
        "forced_inclusion_events": forced_inclusion_events,
        "min_positions_held": min(position_count_history) if position_count_history else 0,
        "max_positions_held": max(position_count_history) if position_count_history else 0,
        "trailing_stop_exits": (trades_df["exit_reason"] == "trailing_stop").sum() if not trades_df.empty else 0,
        "death_cross_exits": (trades_df["exit_reason"] == "death_cross").sum() if not trades_df.empty else 0,
    }


print("=" * 90)
print("MULTI-WINDOW BACKTEST: letting winners run (trailing stop) + 60-stock universe (SYNTHETIC DATA)")
print("=" * 90)

results = []
for start, end in TEST_WINDOWS:
    r = run_backtest_window(start, end)
    results.append(r)
    print(f"{start} to {end}: return={r['total_return']:>7.2%}  sharpe={r['sharpe']:>5.2f}  "
          f"max_dd={r['max_dd']:>7.2%}  trades={r['num_trades']:>3d}  win_rate={r['win_rate']:>5.1%}  "
          f"trail_exits={r['trailing_stop_exits']:>2d}  death_cross_exits={r['death_cross_exits']:>2d}  "
          f"forced_incl_weeks={r['forced_inclusion_events']}")

returns = [r["total_return"] for r in results]
sharpes = [r["sharpe"] for r in results]

print()
print("=" * 90)
print("DISTRIBUTION ACROSS ALL WINDOWS (this is the honest summary, not any single window)")
print("=" * 90)
print(f"Return  -> mean: {np.mean(returns):.2%}  std: {np.std(returns):.2%}  min: {min(returns):.2%}  max: {max(returns):.2%}")
print(f"Sharpe  -> mean: {np.mean(sharpes):.2f}  std: {np.std(sharpes):.2f}  min: {min(sharpes):.2f}  max: {max(sharpes):.2f}")
print(f"Windows with a negative return: {sum(1 for r in returns if r < 0)} / {len(returns)}")
