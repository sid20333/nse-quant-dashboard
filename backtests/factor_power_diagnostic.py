"""
factor_power_diagnostic.py — Before building an alpha engine: does ANY clean
price-based factor actually have cross-sectional selection power in this
universe? If a factor can't sort the 104 names into future winners vs losers,
there is no alpha to harvest and no point wrapping a machine around it.

Method (point-in-time, monthly rebalance, 2015-2024):
  At each month-end t, for every stock with enough history, compute candidate
  factors from data up to t only. Then measure:
    - IC  : cross-sectional Spearman rank corr(factor_t, next-month return).
            Mean IC and its t-stat say whether the ranking predicts, on average.
            A mean |IC| ~0.03-0.05 with t>2 is a genuinely useful equity factor.
    - Long-only top-15 basket: each month hold the 15 highest-factor names,
            equal weight, vs holding ALL names (the EW-hold bar we must beat).
            This is the money test for a LONG-ONLY engine.

Candidate factors (all price/volume, no look-ahead fundamentals):
  mom_12_1   12-month return skipping the last month  (classic momentum)
  mom_6_1    6-month skip-1 momentum
  st_rev     last-1-month return   (expect NEGATIVE IC = short-term reversal)
  low_vol    negative of 60d realized vol   (low-volatility anomaly)
  near_high  close / 252-day high  (52-week-high proximity anomaly)
  trend      close / 200-day SMA - 1  (what the current engine already leans on)
  mom_x_lowvol  mom_12_1 z-score + low_vol z-score  (a first composite)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.backtests.nse_universe import UNIVERSE

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
mat = pd.DataFrame(closes).sort_index()
rets_d = mat.pct_change()

# month-end trading dates 2015-2024
month_ends = mat.loc["2014-12-01":"2024-12-31"].resample("ME").last().index
rebal = [mat.index[mat.index <= me][-1] for me in month_ends if (mat.index <= me).any()]
rebal = sorted(set(rebal))
rebal = [d for d in rebal if d >= pd.Timestamp("2015-01-01")]

def factors_at(pos):
    p = mat.iloc[pos]
    def px(off): return mat.iloc[pos - off] if pos - off >= 0 else pd.Series(np.nan, index=mat.columns)
    f = {}
    f["mom_12_1"] = px(21) / px(252) - 1
    f["mom_6_1"]  = px(21) / px(126) - 1
    f["st_rev"]   = p / px(21) - 1
    f["low_vol"]  = -rets_d.iloc[pos-60:pos].std()
    f["near_high"] = p / mat.iloc[pos-252:pos+1].max()
    f["trend"]    = p / mat.iloc[pos-200:pos].mean() - 1
    return pd.DataFrame(f)

FACTORS = ["mom_12_1", "mom_6_1", "st_rev", "low_vol", "near_high", "trend", "mom_x_lowvol"]
ic_acc = {f: [] for f in FACTORS}
top_ret = {f: [] for f in FACTORS}
ew_ret = []

for i in range(len(rebal) - 1):
    d, d1 = rebal[i], rebal[i + 1]
    pos = mat.index.get_loc(d)
    if pos < 252:
        continue
    F = factors_at(pos)
    fwd = (mat.loc[d1] / mat.loc[d] - 1)
    valid = F.dropna(how="any").index.intersection(fwd.dropna().index)
    if len(valid) < 30:
        continue
    F = F.loc[valid]; fwd_v = fwd.loc[valid]
    # composite
    z = (F[["mom_12_1", "low_vol"]] - F[["mom_12_1", "low_vol"]].mean()) / F[["mom_12_1", "low_vol"]].std()
    F["mom_x_lowvol"] = z.mean(axis=1)
    ew_ret.append(fwd_v.mean())
    for f in FACTORS:
        s = F[f]
        ic, _ = spearmanr(s, fwd_v)
        ic_acc[f].append(ic)
        top = s.nlargest(15).index
        top_ret[f].append(fwd_v.loc[top].mean())

n = len(ew_ret)
def comp(x): return np.prod(1 + np.array(x)) - 1

print("=" * 96)
print(f"CROSS-SECTIONAL FACTOR POWER — monthly, 2015-2024, {n} rebalances, point-in-time")
print("Bar to beat: EW-hold ALL names compounded = %.0f%%  (mean %.2f%%/mo)" % (comp(ew_ret)*100, np.mean(ew_ret)*100))
print("=" * 96)
print(f"{'factor':<14}{'mean_IC':>9}{'IC_t':>7}{'IC>0 %':>8}   "
      f"{'top15 comp':>11}{'top15/mo':>10}{'vs EW/mo':>10}")
for f in FACTORS:
    ic = np.array(ic_acc[f]); tr = np.array(top_ret[f])
    t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0
    hit = (ic > 0).mean() * 100
    print(f"{f:<14}{ic.mean():>9.3f}{t:>7.1f}{hit:>7.0f}%   "
          f"{comp(tr):>10.0%}{tr.mean()*100:>9.2f}%{(tr.mean()-np.mean(ew_ret))*100:>+9.2f}%")
print("=" * 96)
print("mean_IC: predictive strength (|0.03+| with IC_t>2 = real). st_rev should be NEGATIVE.")
print("vs EW/mo: does a long-only top-15 by this factor BEAT holding everything? (the alpha test)")
