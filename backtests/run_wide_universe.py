"""
run_wide_universe.py — Does widening the universe strengthen momentum, and is
the extra alpha real or a small-cap liquidity/survivorship mirage?

Two measurements:
  1. SIGNAL: momentum alpha (regressed on EW-hold of the SAME universe) for the
     narrow (104) vs wide (~200) universe. Does breadth raise the alpha t-stat?
  2. LIQUIDITY REALITY: re-run the wide-universe momentum NET return under
     escalating slippage (0.15% liquid -> 0.5% -> 1.0% small-cap). If the alpha
     evaporates as slippage rises, widening was a backtest illusion.

Both universes are today's-survivors (Yahoo), so the wide one is MORE
survivorship-biased — a caveat no regression here can remove.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from scipy import stats
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE as NARROW
from quant_engine.backtests.nse_universe_wide import WIDE

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL, TOP_N = 500_000.0, 20
COSTS = IndianEquityCosts()
RF_M = 0.06 / 12
prov = YFinanceDataProvider(cache_dir=CACHE_DIR)

def load(symbols):
    d = {}
    for s in symbols:
        try:
            d[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
        except Exception:
            pass
    return pd.DataFrame(d).sort_index()

print("Fetching wide universe (new names hit Yahoo)...")
wide_close = load(WIDE)
narrow_close = wide_close[[c for c in NARROW if c in wide_close.columns]]
print(f"  narrow usable: {narrow_close.shape[1]}   wide usable: {wide_close.shape[1]}\n")


def momentum_monthly(close, top_scale=TOP_N):
    mret = close.resample("ME").last().pct_change()
    reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
    reb = [d for d in reb if d >= pd.Timestamp("2014-06-01")]
    baskets = {}
    for d in reb:
        pos = close.index.get_loc(close.index[close.index <= d][-1])
        if pos < 252:
            continue
        def px(o): return close.iloc[pos - o]
        r = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
        baskets[d] = list(r.reindex(close.iloc[pos].dropna().index).dropna().nlargest(top_scale).index)
    mom_r, ew_r, months = [], [], []
    for m in mret.index:
        if m < pd.Timestamp("2015-07-01"):
            continue
        prior = [d for d in reb if d < m]
        if not prior or prior[-1] not in baskets:
            continue
        bk = baskets[prior[-1]]
        row = mret.loc[m]
        mm = row[[s for s in bk if s in row.index]].dropna()
        ee = row[row.notna()]
        if len(mm) < 5 or len(ee) < 20:
            continue
        mom_r.append(mm.mean()); ew_r.append(ee.mean()); months.append(m)
    return pd.DataFrame({"mom": mom_r, "ew": ew_r}, index=months)


def alpha_vs_ew(df, label):
    y = df["mom"] - RF_M; x = df["ew"] - RF_M
    r = stats.linregress(x, y)
    t = r.intercept / r.intercept_stderr if r.intercept_stderr > 0 else 0
    print(f"{label:<34} alpha={r.intercept*12:>+7.2%}/yr  t={t:>5.2f}  beta={r.slope:>4.2f}  "
          f"{'SIGNIFICANT' if abs(t)>2 else 'not sig.'}")

print("=" * 90)
print("1) SIGNAL — momentum alpha vs EW-hold (same universe), gross of slippage")
print("=" * 90)
alpha_vs_ew(momentum_monthly(narrow_close), f"NARROW ({narrow_close.shape[1]} names)")
alpha_vs_ew(momentum_monthly(wide_close), f"WIDE ({wide_close.shape[1]} names)")

# 2) liquidity reality — net momentum CAGR on WIDE universe as slippage rises
print("\n" + "=" * 90)
print("2) LIQUIDITY REALITY — wide-universe momentum NET return as slippage escalates")
print("=" * 90)
def net_momentum_cagr(close, slip):
    rt = COSTS.round_trip_pct(CAPITAL / TOP_N) + 2 * slip
    mret = close.resample("ME").last().pct_change()
    reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
    reb = [d for d in reb if d >= pd.Timestamp("2014-06-01")]
    baskets, prev, rets = {}, set(), []
    for d in reb:
        pos = close.index.get_loc(close.index[close.index <= d][-1])
        if pos < 252:
            continue
        def px(o): return close.iloc[pos - o]
        r = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
        baskets[d] = set(r.reindex(close.iloc[pos].dropna().index).dropna().nlargest(TOP_N).index)
    for m in mret.index:
        if m < pd.Timestamp("2015-07-01"):
            continue
        prior = [d for d in reb if d < m]
        if not prior or prior[-1] not in baskets:
            continue
        bk = baskets[prior[-1]]
        turn = 1 - len(bk & prev) / TOP_N
        row = mret.loc[m]
        mm = row[[s for s in bk if s in row.index]].dropna().mean()
        rets.append(mm - turn * rt / 6)   # rebalance is semiannual; amortize turn cost across ~6 months? no -> apply at rebal only
        prev = bk
    # apply turnover cost properly: only at rebalance months. Simpler: recompute at semiannual granularity
    return None

def net_momentum_stats(close, slip):
    rt = COSTS.round_trip_pct(CAPITAL / TOP_N) + 2 * slip
    mret = close.resample("ME").last().pct_change()
    reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
    reb = [d for d in reb if d >= pd.Timestamp("2014-06-01")]
    baskets, prev, rets, ew = {}, set(), [], []
    for d in reb:
        pos = close.index.get_loc(close.index[close.index <= d][-1])
        if pos < 252:
            continue
        def px(o): return close.iloc[pos - o]
        r = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
        baskets[d] = set(r.reindex(close.iloc[pos].dropna().index).dropna().nlargest(TOP_N).index)
    last_reb = None
    for m in mret.index:
        if m < pd.Timestamp("2015-07-01"):
            continue
        prior = [d for d in reb if d < m]
        if not prior or prior[-1] not in baskets:
            continue
        rebd = prior[-1]
        bk = baskets[rebd]
        row = mret.loc[m]
        mm = row[[s for s in bk if s in row.index]].dropna().mean()
        cost = 0.0
        if rebd != last_reb:                     # rebalance happened this period
            turn = 1 - len(bk & prev) / TOP_N
            cost = turn * rt
            prev = bk; last_reb = rebd
        rets.append(mm - cost)
        ew.append(row[row.notna()].mean())
    rets = pd.Series(rets); ew = pd.Series(ew)
    eq = (1 + rets).cumprod(); eeq = (1 + ew).cumprod()
    cagr = eq.iloc[-1] ** (12/len(rets)) - 1
    ewcagr = eeq.iloc[-1] ** (12/len(ew)) - 1
    return cagr, ewcagr

print(f"{'slippage/side':<16}{'wide mom CAGR':>15}{'wide EW CAGR':>14}{'mom vs EW':>12}")
for slip, lbl in [(0.0015, "0.15% (liquid)"), (0.005, "0.50% (mid/small)"), (0.010, "1.0% (small)")]:
    mc, ec = net_momentum_stats(wide_close, slip)
    print(f"{lbl:<16}{mc:>14.1%}{ec:>14.1%}{(mc-ec)*100:>+11.1f}p")
print("=" * 90)
print("If 'mom vs EW' stays positive as slippage rises, wide-universe momentum survives")
print("realistic costs. If it collapses, the extra breadth was an un-tradeable mirage.")
