"""
run_ma_rsi_backtest.py — Long-only MA+RSI strategy, backtested with:
  - Capital: Rs 5,00,000
  - Position count: min 5, max 15 held simultaneously
  - Long only, no shorting
  - 3-month test window

IMPORTANT: uses SyntheticDataProvider because this sandbox has no network
access to real NSE historical data. This validates the STRATEGY LOGIC AND
POSITION-SIZING MECHANICS run correctly end-to-end. It says NOTHING about
whether this makes money on real markets. Swap in BreezeDataProvider with
real NSE midcap/largecap history before drawing any real conclusion.

RANKING / SELECTION LOGIC:
  At each weekly rebalance date, every stock in the universe gets a
  bullish_score (see moving_average_screener.py) computed using ONLY data
  up to that date (no look-ahead). Stocks are ranked by score descending.
  - If >=5 stocks have a positive score: hold the top min(15, count) of them.
  - If <5 stocks have a positive score: still hold the best 5 available
    (to satisfy the "always 5-15 positions" constraint), but this is a
    forced inclusion and is flagged in the output - it means the strategy
    is holding names it wouldn't otherwise choose, purely to satisfy the
    position-count rule. This is a real tension worth knowing about your
    own constraint, not a bug.
  Existing holdings are only exited if they fall out of the top 15 ranked
  names OR flip to an outright death cross (50<200) - this avoids
  needless weekly turnover/costs for a name still reasonably ranked.
"""
import sys
sys.path.insert(0, "/home/claude")

import pandas as pd
import numpy as np
from datetime import date

from quant_engine.data_provider import SyntheticDataProvider
from quant_engine.moving_average_screener import compute_ma_state

UNIVERSE_SIZE = 25
HIST_START = "2024-06-01"
BACKTEST_START = "2026-04-01"
BACKTEST_END = "2026-06-30"

symbol_dfs = {}
for i in range(UNIVERSE_SIZE):
    provider = SyntheticDataProvider(seed=100 + i)
    df = provider.get_daily_ohlcv(f"MIDCAP{i:02d}", HIST_START, BACKTEST_END)
    symbol_dfs[f"MIDCAP{i:02d}"] = df

print(f"Universe: {UNIVERSE_SIZE} synthetic stocks, {HIST_START} to {BACKTEST_END}")
print(f"Backtest window: {BACKTEST_START} to {BACKTEST_END} (3 months)\n")

CAPITAL = 500_000.0
MIN_POSITIONS = 5
MAX_POSITIONS = 15
COST_PCT = 0.001
SLIPPAGE_PCT = 0.0015

decision_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="W-SAT")

positions = {}
cash = CAPITAL
equity_curve = []
equity_dates = []
position_count_history = []
trade_log = []
forced_inclusion_events = 0

for decision_date in decision_dates:
    scores = {}
    for symbol, df in symbol_dfs.items():
        sliced = df[df["date"] <= decision_date]
        if len(sliced) < 210:
            continue
        state = compute_ma_state(sliced)
        scores[symbol] = state.bullish_score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    positive_ranked = [s for s in ranked if s[1] > 0]

    if len(positive_ranked) >= MIN_POSITIONS:
        target_symbols = [s for s, _ in positive_ranked[:MAX_POSITIONS]]
    else:
        target_symbols = [s for s, _ in ranked[:MIN_POSITIONS]]
        forced_inclusion_events += 1

    target_set = set(target_symbols)
    current_set = set(positions.keys())

    for symbol in list(current_set - target_set):
        df = symbol_dfs[symbol]
        exit_bars = df[df["date"] > decision_date]
        if exit_bars.empty:
            continue
        exit_price = exit_bars.iloc[0]["open"] * (1 - SLIPPAGE_PCT)
        pos = positions.pop(symbol)
        proceeds = pos["shares"] * exit_price * (1 - COST_PCT)
        cash += proceeds
        ret = (exit_price * (1 - COST_PCT) - pos["entry_price"]) / pos["entry_price"]
        trade_log.append({
            "symbol": symbol, "entry_date": pos["entry_date"], "exit_date": exit_bars.iloc[0]["date"],
            "entry_price": pos["entry_price"], "exit_price": exit_price, "return_pct": ret,
        })

    new_entries = target_set - current_set
    if new_entries:
        held_value = sum(
            p["shares"] * symbol_dfs[s][symbol_dfs[s]["date"] <= decision_date]["close"].iloc[-1]
            for s, p in positions.items()
        )
        per_position_value = (cash + held_value) / len(target_set)

        for symbol in new_entries:
            df = symbol_dfs[symbol]
            entry_bars = df[df["date"] > decision_date]
            if entry_bars.empty:
                continue
            entry_bar = entry_bars.iloc[0]
            entry_price = entry_bar["open"] * (1 + SLIPPAGE_PCT)
            allocation = min(per_position_value, cash)
            if allocation <= 0:
                continue
            shares = allocation / entry_price
            cash -= shares * entry_price * (1 + COST_PCT)
            positions[symbol] = {"shares": shares, "entry_price": entry_price, "entry_date": entry_bar["date"]}

    mtm = cash
    for symbol, pos in positions.items():
        df = symbol_dfs[symbol]
        recent = df[df["date"] <= decision_date]
        if not recent.empty:
            mtm += pos["shares"] * recent["close"].iloc[-1]
    equity_curve.append(mtm)
    equity_dates.append(decision_date)
    position_count_history.append(len(positions))

final_date = pd.to_datetime(BACKTEST_END)
for symbol, pos in list(positions.items()):
    df = symbol_dfs[symbol]
    final_bars = df[df["date"] <= final_date]
    if final_bars.empty:
        continue
    exit_price = final_bars.iloc[-1]["close"]
    proceeds = pos["shares"] * exit_price * (1 - COST_PCT)
    cash += proceeds
    ret = (exit_price * (1 - COST_PCT) - pos["entry_price"]) / pos["entry_price"]
    trade_log.append({
        "symbol": symbol, "entry_date": pos["entry_date"], "exit_date": final_date,
        "entry_price": pos["entry_price"], "exit_price": exit_price, "return_pct": ret,
    })
positions = {}

equity = pd.Series(equity_curve, index=equity_dates)
final_value = cash
total_return = (final_value - CAPITAL) / CAPITAL
weekly_returns = equity.pct_change().dropna()
sharpe = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52) if weekly_returns.std() > 0 else 0
running_max = equity.cummax()
max_dd = ((equity - running_max) / running_max).min()

trades_df = pd.DataFrame(trade_log)
win_rate = (trades_df["return_pct"] > 0).mean() if not trades_df.empty else 0
avg_win = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].mean() if not trades_df.empty else 0
avg_loss = trades_df.loc[trades_df["return_pct"] <= 0, "return_pct"].mean() if not trades_df.empty else 0

print("=" * 60)
print("RESULTS (SYNTHETIC DATA - validates mechanics only)")
print("=" * 60)
print(f"Starting capital:      Rs {CAPITAL:,.0f}")
print(f"Ending capital:        Rs {final_value:,.0f}")
print(f"Total return (3mo):    {total_return:.2%}")
print(f"Naive annualized x4:   {total_return * 4:.2%}")
print(f"Sharpe (weekly, ann.): {sharpe:.2f}")
print(f"Max drawdown:          {max_dd:.2%}")
print(f"Number of trades:      {len(trades_df)}")
print(f"Win rate:              {win_rate:.1%}")
print(f"Avg win / avg loss:    {avg_win:.2%} / {avg_loss:.2%}")
print(f"Weeks with forced inclusion (<5 genuinely bullish): {forced_inclusion_events} / {len(decision_dates)}")
print(f"Position count range held: {min(position_count_history)} to {max(position_count_history)}")
print()
if not trades_df.empty:
    print("Sample trades:")
    print(trades_df.head(10).to_string(index=False))
