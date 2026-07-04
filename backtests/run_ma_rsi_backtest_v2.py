"""
run_ma_rsi_backtest_v2.py — Same strategy and constraints as v1
(Rs 5,00,000 capital, 5-15 positions, long only, 3 months), plus two
structural improvements aimed at Sharpe, not signal-tuning:

1. VOLATILITY-WEIGHTED POSITION SIZING
   Instead of equal rupee allocation per position, each position is sized
   inversely to its recent ATR% (ATR normalized by price). A stock that
   swings twice as much day-to-day gets roughly half the capital of a calm
   one, so each position contributes closer to equal RISK rather than
   equal capital. This is the single highest-value change here because it
   directly reduces portfolio volatility (Sharpe's denominator) without
   touching the signal logic at all - nothing to overfit.

2. CORRELATION-CAPPED SELECTION
   When building the target portfolio from the ranked candidate list, a
   candidate is skipped if its 60-day return correlation with any ALREADY
   SELECTED holding exceeds `MAX_PAIRWISE_CORRELATION`. This prevents
   "15 positions" from secretly being "3 independent bets, each held 5
   times over" - real diversification lowers volatility more than nominal
   position count does. If correlation filtering leaves fewer than
   MIN_POSITIONS, the filter is relaxed (best remaining by rank, ignoring
   correlation) until the floor is met - the floor still takes priority,
   consistent with v1.

Still uses SyntheticDataProvider - same caveat as v1 applies: this proves
the mechanics work, not that the improvements will hold on real markets.
"""
import sys
sys.path.insert(0, "/home/claude")

import pandas as pd
import numpy as np

from quant_engine.data_provider import SyntheticDataProvider
from quant_engine.moving_average_screener import compute_ma_state
from quant_engine.technical import average_true_range

UNIVERSE_SIZE = 25
HIST_START = "2024-06-01"
BACKTEST_START = "2026-04-01"
BACKTEST_END = "2026-06-30"

CAPITAL = 500_000.0
MIN_POSITIONS = 5
MAX_POSITIONS = 15
COST_PCT = 0.001
SLIPPAGE_PCT = 0.0015
CORR_LOOKBACK_DAYS = 60
MAX_PAIRWISE_CORRELATION = 0.70

symbol_dfs = {}
for i in range(UNIVERSE_SIZE):
    provider = SyntheticDataProvider(seed=100 + i)
    df = provider.get_daily_ohlcv(f"MIDCAP{i:02d}", HIST_START, BACKTEST_END)
    symbol_dfs[f"MIDCAP{i:02d}"] = df

print(f"Universe: {UNIVERSE_SIZE} synthetic stocks | Backtest: {BACKTEST_START} to {BACKTEST_END}\n")


def atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    """Latest ATR as a % of price - used as the inverse-vol sizing weight."""
    a = average_true_range(df["high"], df["low"], df["close"], window)
    price = df["close"].iloc[-1]
    return float(a.iloc[-1] / price) if price > 0 else np.nan


def returns_series(df: pd.DataFrame, as_of, lookback: int = CORR_LOOKBACK_DAYS) -> pd.Series:
    sliced = df[df["date"] <= as_of].tail(lookback)
    return sliced.set_index("date")["close"].pct_change().dropna()


def select_with_correlation_cap(ranked_symbols, symbol_dfs, decision_date, max_n, max_corr):
    """Greedily walk the ranked list, skipping a candidate if it's too
    correlated with anything already selected. Returns selected list."""
    selected = []
    return_series_cache = {}

    def get_returns(sym):
        if sym not in return_series_cache:
            return_series_cache[sym] = returns_series(symbol_dfs[sym], decision_date)
        return return_series_cache[sym]

    for sym in ranked_symbols:
        if len(selected) >= max_n:
            break
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
    return selected


decision_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="W-SAT")

positions = {}
cash = CAPITAL
equity_curve, equity_dates, position_count_history = [], [], []
trade_log = []
forced_inclusion_events = 0
correlation_skips_total = 0

for decision_date in decision_dates:
    scores = {}
    for symbol, df in symbol_dfs.items():
        sliced = df[df["date"] <= decision_date]
        if len(sliced) < 210:
            continue
        state = compute_ma_state(sliced)
        scores[symbol] = state.bullish_score

    ranked_all = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    positive_ranked = [s for s in ranked_all if scores[s] > 0]

    if len(positive_ranked) >= MIN_POSITIONS:
        candidates = positive_ranked
    else:
        candidates = ranked_all
        forced_inclusion_events += 1

    target_symbols = select_with_correlation_cap(candidates, symbol_dfs, decision_date, MAX_POSITIONS, MAX_PAIRWISE_CORRELATION)
    correlation_skips_total += max(0, min(len(candidates), MAX_POSITIONS) - len(target_symbols))

    # Backfill to MIN_POSITIONS ignoring correlation cap if the filter left us short
    if len(target_symbols) < MIN_POSITIONS:
        for sym in ranked_all:
            if sym not in target_symbols:
                target_symbols.append(sym)
            if len(target_symbols) >= MIN_POSITIONS:
                break

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
        trade_log.append({"symbol": symbol, "entry_date": pos["entry_date"], "exit_date": exit_bars.iloc[0]["date"],
                           "entry_price": pos["entry_price"], "exit_price": exit_price, "return_pct": ret})

    new_entries = target_set - current_set
    if new_entries:
        held_value = sum(
            p["shares"] * symbol_dfs[s][symbol_dfs[s]["date"] <= decision_date]["close"].iloc[-1]
            for s, p in positions.items()
        )
        total_deployable = cash + held_value

        # Inverse-volatility weights across the FULL target set (existing + new),
        # so capital is reallocated toward lower-ATR% names each rebalance.
        vol_weights = {}
        for sym in target_set:
            sliced = symbol_dfs[sym][symbol_dfs[sym]["date"] <= decision_date]
            a = atr_pct(sliced)
            vol_weights[sym] = 1.0 / a if a and a > 0 else 0.0
        weight_sum = sum(vol_weights.values()) or 1.0

        for symbol in new_entries:
            df = symbol_dfs[symbol]
            entry_bars = df[df["date"] > decision_date]
            if entry_bars.empty:
                continue
            entry_bar = entry_bars.iloc[0]
            entry_price = entry_bar["open"] * (1 + SLIPPAGE_PCT)
            allocation = min(total_deployable * (vol_weights[symbol] / weight_sum), cash)
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
    trade_log.append({"symbol": symbol, "entry_date": pos["entry_date"], "exit_date": final_date,
                       "entry_price": pos["entry_price"], "exit_price": exit_price, "return_pct": ret})
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
print("RESULTS v2: vol-weighted sizing + correlation cap (SYNTHETIC DATA)")
print("=" * 60)
print(f"Starting capital:      Rs {CAPITAL:,.0f}")
print(f"Ending capital:        Rs {final_value:,.0f}")
print(f"Total return (3mo):    {total_return:.2%}")
print(f"Sharpe (weekly, ann.): {sharpe:.2f}")
print(f"Max drawdown:          {max_dd:.2%}")
print(f"Number of trades:      {len(trades_df)}")
print(f"Win rate:              {win_rate:.1%}")
print(f"Avg win / avg loss:    {avg_win:.2%} / {avg_loss:.2%}")
print(f"Weeks with forced inclusion: {forced_inclusion_events} / {len(decision_dates)}")
print(f"Candidate slots skipped for correlation (cumulative): {correlation_skips_total}")
print(f"Position count range held: {min(position_count_history)} to {max(position_count_history)}")
