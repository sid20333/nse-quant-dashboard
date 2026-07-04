"""
run_last_3_months.py — What did the engines do in the most recent ~3 months?

HARD CAVEAT: n=1 quarter. This is a single random draw, not validation. The
entire thread showed the edge is only visible across DOZENS of quarters; one
window's result (good or bad) is noise. Reported because it's genuinely
out-of-sample (momentum engine was built on 2015-2024), but read it as an
anecdote, not evidence.

Window: 2026-04-01 -> latest available. Entry at first open on/after start,
exit at last close. Real costs (one round trip); short leg carries F&O carry.
Compares: momentum top-20 (long-only), EW-hold universe, NIFTY100, and the
Long100/Short50 book.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL = 500_000.0
TOP_N = 20
COSTS = IndianEquityCosts()
RT_COST = COSTS.round_trip_pct(CAPITAL / TOP_N) + 2 * 0.0015
SHORT_CARRY_ANNUAL = 0.01
START = "2026-04-01"
END = "2026-07-04"
FETCH_START = "2024-06-01"            # enough lookback for 12-1 momentum + 200SMA

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes, opens = {}, {}
for s in UNIVERSE:
    try:
        df = prov.get_daily_ohlcv(s, FETCH_START, END)
        closes[s] = df.set_index("date")["close"]
        opens[s] = df.set_index("date")["open"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
open_ = pd.DataFrame(opens).sort_index()
idx = prov.get_daily_ohlcv(REGIME_INDEX, FETCH_START, END).set_index("date")["close"]

start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
as_of = close.index[close.index < start_ts][-1]
pos = close.index.get_loc(as_of)
last = close.index[close.index <= end_ts][-1]
print(f"Momentum ranked as of {as_of.date()} | window {start_ts.date()} -> {last.date()} "
      f"({(last-start_ts).days} days, real out-of-sample)\n")

def px(o): return close.iloc[pos - o]
rank = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
active = close.iloc[pos].dropna().index
rank = rank.reindex(active).dropna()
longs = list(rank.nlargest(TOP_N).index)
shorts = list(rank.nsmallest(TOP_N).index)

def win_ret(sym):
    o = open_[sym][open_[sym].index >= start_ts].dropna()
    c = close[sym][close[sym].index <= end_ts].dropna()
    if o.empty or c.empty:
        return np.nan
    return c.iloc[-1] / o.iloc[0] - 1

long_rets = pd.Series({s: win_ret(s) for s in longs}).dropna()
short_rets = pd.Series({s: win_ret(s) for s in shorts}).dropna()
ew_rets = pd.Series({s: win_ret(s) for s in active}).dropna()
idx_ret = idx[idx.index <= end_ts].iloc[-1] / idx[idx.index >= start_ts].iloc[0] - 1
carry = SHORT_CARRY_ANNUAL * (last - start_ts).days / 365

mom = long_rets.mean() - RT_COST
ew = ew_rets.mean()
ls = 1.0 * long_rets.mean() - 0.5 * short_rets.mean() - RT_COST - 0.5 * (RT_COST + carry)

print("=" * 72)
print(f"LAST ~3 MONTHS (n=1 quarter — ANECDOTE, NOT VALIDATION)")
print("=" * 72)
print(f"{'book':<34}{'return':>10}{'vs EW':>10}")
print(f"{'Momentum top-20 (long-only)':<34}{mom:>9.2%}{(mom-ew)*100:>+9.2f}p")
print(f"{'Long100/Short50':<34}{ls:>9.2%}{(ls-ew)*100:>+9.2f}p")
print(f"{'EW-hold universe (bar)':<34}{ew:>9.2%}{0.0:>+9.2f}p")
print(f"{'NIFTY100 index':<34}{idx_ret:>9.2%}{(idx_ret-ew)*100:>+9.2f}p")
print("=" * 72)
print(f"momentum basket: {(long_rets>0).sum()}/{len(long_rets)} names up | "
      f"universe: {(ew_rets>0).sum()}/{len(ew_rets)} up | "
      f"context: {'RALLY' if idx_ret>0.03 else 'FLAT/DOWN'} window (index {idx_ret:+.1%})")
print("top-5 momentum picks:", ", ".join(longs[:5]))
print("\nn=1. A single quarter is noise. Do not update your priors on the strategy from this.")
