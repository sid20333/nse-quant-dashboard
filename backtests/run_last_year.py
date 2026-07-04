"""
run_last_year.py — Engines over the trailing ~12 months (2025-07 -> 2026-07).

More informative than one quarter: this window CONTAINS A REAL CRASH (NIFTY100
fell ~15% in Q1 2026, then recovered), so the trailing year is net-down and
actually stresses the defensive / hedge claims out-of-sample. Still only ~4
quarters — small sample, read as a stress-anecdote, not proof.

Monthly marks, semiannual momentum rebalance, real costs at rebalance months,
short leg carries F&O carry. Reports total return, max drawdown, monthly Sharpe,
and momentum's edge over the fair benchmark (EW-hold same universe).
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
SHORT_CARRY_M = 0.01 / 12
WIN_START, END = "2025-07-01", "2026-07-04"
FETCH_START = "2024-01-01"

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, FETCH_START, END).set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
idx = prov.get_daily_ohlcv(REGIME_INDEX, FETCH_START, END).set_index("date")["close"]
mret = close.resample("ME").last().pct_change()
midx = idx.resample("ME").last().pct_change()

# semiannual rebalance dates within lookback
reb = list(pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values)
reb = [pd.Timestamp(d) for d in reb]

def mom_basket(d, which="top"):
    prior = [r for r in reb if r <= d]
    if not prior:
        return None
    pos = close.index.get_loc(close.index[close.index <= prior[-1]][-1])
    if pos < 252:
        return None
    def px(o): return close.iloc[pos - o]
    r = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
    r = r.reindex(close.iloc[pos].dropna().index).dropna()
    return list(r.nlargest(TOP_N).index) if which == "top" else list(r.nsmallest(TOP_N).index)

months = [m for m in mret.index if pd.Timestamp(WIN_START) <= m <= pd.Timestamp(END)]
mom_c, ls_c, ew_c, ix_c = [], [], [], []
prev_long, prev_short = set(), set()
for m in months:
    rebal_date = [r for r in reb if r < m]
    key = rebal_date[-1] if rebal_date else None
    longs = mom_basket(key, "top") or []
    shorts = mom_basket(key, "bot") or []
    row = mret.loc[m]
    lr = row[[s for s in longs if s in row.index]].dropna().mean()
    sr = row[[s for s in shorts if s in row.index]].dropna().mean()
    er = row[row.notna()].mean()
    # cost only when basket changes (rebalance) — approximate by set diff
    l_turn = 1 - len(set(longs) & prev_long) / max(len(longs), 1) if longs else 0
    s_turn = 1 - len(set(shorts) & prev_short) / max(len(shorts), 1) if shorts else 0
    mom_c.append(lr - l_turn * RT_COST)
    ls_c.append(lr - 0.5 * sr - l_turn * RT_COST - 0.5 * s_turn * RT_COST - 0.5 * SHORT_CARRY_M)
    ew_c.append(er)
    ix_c.append(midx.loc[m])
    prev_long, prev_short = set(longs), set(shorts)

def stats(rs):
    rs = pd.Series(rs).fillna(0)
    eq = (1 + rs).cumprod()
    tot = eq.iloc[-1] - 1
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    sharpe = rs.mean() / rs.std() * np.sqrt(12) if rs.std() > 0 else 0
    return tot, dd, sharpe

print(f"Trailing year: {months[0].strftime('%Y-%m')} -> {months[-1].strftime('%Y-%m')} "
      f"({len(months)} months, out-of-sample)\n")
print("=" * 74)
print("LAST YEAR (~4 quarters incl. a real -15% crash quarter — small sample)")
print("=" * 74)
print(f"{'book':<32}{'total':>9}{'maxDD':>9}{'Sharpe':>8}{'vs EW':>9}")
ew_tot = stats(ew_c)[0]
for lbl, rs in [("Momentum top-20 (long-only)", mom_c), ("Long100/Short50", ls_c),
                ("EW-hold universe (bar)", ew_c), ("NIFTY100 index", ix_c)]:
    t, dd, sh = stats(rs)
    print(f"{lbl:<32}{t:>8.2%}{dd:>9.2%}{sh:>8.2f}{(t-ew_tot)*100:>+8.2f}p")
print("=" * 74)
# crash-month spotlight
crash_m = [m for m, ix in zip(months, ix_c) if ix < -0.05]
if crash_m:
    ci = [i for i, ix in enumerate(ix_c) if ix < -0.05]
    print(f"Worst index month(s): " + ", ".join(f"{months[i].strftime('%Y-%m')} "
          f"[idx {ix_c[i]:+.1%} | mom {mom_c[i]:+.1%} | L/S {ls_c[i]:+.1%}]" for i in ci))
print("\n~4 quarters. Includes a genuine crash, so it stresses the hedge — but still")
print("far too short to confirm or deny the 2015-2024 result. Anecdote, not verdict.")
