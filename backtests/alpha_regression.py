"""
alpha_regression.py — Is the momentum edge actually ALPHA, statistically?

'Beats the benchmark' in a backtest is not proof of alpha — it can be market
beta, a size premium, or luck. The formal test: regress the strategy's monthly
excess returns on a benchmark's excess returns.
    r_strat - rf = alpha + beta * (r_bench - rf) + e
alpha (the intercept) is the return NOT explained by exposure to the benchmark.
It is real only if it is positive AND its t-stat clears ~2 (95% confidence).

Two regressions, because the benchmark choice is the whole game:
  vs NIFTY100  : alpha after market beta. But momentum here tilts to mid-caps,
                 so a positive alpha vs the large-cap index could just be the
                 size premium, not skill.
  vs EW-HOLD   : the honest one. Regress against equal-weight holding the SAME
                 universe. A positive, significant intercept here = the momentum
                 RANKING adds value beyond owning the identical names. Survivor-
                 ship and universe composition cancel; only selection is left.

Monthly returns 2015-2024, semiannual top-20 momentum basket (the validated
low-turnover config), gross (costs shave ~0.5-1%/yr, noted separately).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from scipy import stats
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
RF_ANNUAL = 0.06                     # ~Indian risk-free
rf_m = RF_ANNUAL / 12
TOP_N = 20

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
idx = prov.get_daily_ohlcv(REGIME_INDEX, "2013-01-01", "2024-12-31").set_index("date")["close"]

# monthly returns
mclose = close.resample("ME").last()
mret = mclose.pct_change()
midx = idx.resample("ME").last().pct_change()

# semiannual momentum baskets, applied to following months
rebal = list(pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values)
rebal = [pd.Timestamp(d) for d in rebal if pd.Timestamp(d) >= pd.Timestamp("2015-01-01")]

def mom_top(d):
    pos = close.index.get_loc(close.index[close.index <= d][-1])
    if pos < 252:
        return None
    def px(o): return close.iloc[pos - o]
    r = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
    return list(r.dropna().nlargest(TOP_N).index)

baskets = {d: mom_top(d) for d in rebal}

mom_r, ew_r, mkt_r, months = [], [], [], []
for m in mret.index:
    if m < pd.Timestamp("2015-07-01"):
        continue
    active = mclose.loc[m].dropna().index if m in mclose.index else []
    prior = [d for d in rebal if d < m]
    if not prior:
        continue
    bk = baskets.get(prior[-1])
    if not bk:
        continue
    row = mret.loc[m]
    mom = row[[s for s in bk if s in row.index]].dropna()
    ew = row[[s for s in active if s in row.index]].dropna()
    if len(mom) < 5 or len(ew) < 20 or pd.isna(midx.loc[m]):
        continue
    mom_r.append(mom.mean()); ew_r.append(ew.mean()); mkt_r.append(midx.loc[m]); months.append(m)

df = pd.DataFrame({"mom": mom_r, "ew": ew_r, "mkt": mkt_r}, index=months)

def regress(y, x, label):
    yy = y - rf_m; xx = x - rf_m
    r = stats.linregress(xx, yy)
    alpha_m, a_se = r.intercept, r.intercept_stderr
    a_t = alpha_m / a_se if a_se > 0 else 0
    print(f"{label:<28} alpha={alpha_m*12:>+7.2%}/yr  t={a_t:>5.2f}  "
          f"beta={r.slope:>5.2f}  R2={r.rvalue**2:>4.2f}  {'*** SIGNIFICANT' if abs(a_t)>2 else '(not significant)'}")
    return a_t

print("=" * 92)
print(f"ALPHA REGRESSION — momentum top-{TOP_N}, monthly, {len(df)} obs (2015-2024), gross of costs")
print("=" * 92)
print("MOMENTUM strategy regressed on:")
regress(df["mom"], df["mkt"], "  vs NIFTY100 (market)")
regress(df["mom"], df["ew"], "  vs EW-hold same universe")
print()
print("For reference — is EW-hold itself just the market (size premium check)?")
regress(df["ew"], df["mkt"], "  EW-hold vs NIFTY100")
print("=" * 92)
print("Verdict rule: real, bankable alpha needs a POSITIVE intercept with |t| > 2")
print("in the 'vs EW-hold' regression. Costs (~0.5-1%/yr) reduce the alpha shown here.")
