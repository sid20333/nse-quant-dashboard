"""
fair_benchmark_check.py — Is v5's +429% real skill, or just survivorship?

The strategy trades TODAY's 104 index constituents (survivorship-biased: these
are the names that survived/thrived). Comparing it to the real NIFTY100 index
is unfair — the index is not survivorship-biased, the universe is. So the +8pt
CAGR 'alpha' could be mostly "these specific stocks did great", not timing skill.

FAIR benchmark: equal-weight BUY-AND-HOLD of the SAME universe (point-in-time:
only names with >=210d history at window start), no trading, held the quarter.
Both strategy and this benchmark eat the same survivorship bias, so the DIFFERENCE
between them is the strategy's true contribution: does its regime-filter + stops
+ selection actually beat passively holding the same basket?

Also reports the crash/rally split, because that's where the strategy claims its
edge - we want to know if EW-hold of the universe ALSO protected in crashes
(in which case the protection is the universe, not the strategy).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
idx = prov.get_daily_ohlcv(REGIME_INDEX, "2013-01-01", "2024-12-31").set_index("date")["close"]
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
mat = pd.DataFrame(closes).sort_index()

WINDOWS = []
for yr in range(2015, 2025):
    WINDOWS += [(f"{yr}-01-01", f"{yr}-03-31"), (f"{yr}-04-01", f"{yr}-06-30"),
                (f"{yr}-07-01", f"{yr}-09-30"), (f"{yr}-10-01", f"{yr}-12-31")]

# strategy per-window returns from the OOS run (real data + real costs)
STRAT = [0.0559,0.0001,0.0586,-0.0149,0.0253,0.0787,-0.0017,-0.0398,0.1086,0.0719,
         0.0614,0.0371,-0.0439,0.0908,0.0344,-0.0150,0.0350,0.0368,-0.0195,0.0813,
         -0.0261,0.0905,0.1030,0.1883,0.0864,0.0550,0.0963,0.0329,0.0118,-0.0204,
         0.1673,0.0086,-0.0168,0.1015,0.0468,0.1337,0.0263,0.0255,0.0615,-0.0524]

ew, bh = [], []
for (start, end) in WINDOWS:
    as_of = idx[idx.index < pd.to_datetime(start)].index[-1]
    active = mat.columns[mat[mat.index <= as_of].notna().sum() >= 210]  # point-in-time
    seg = mat[(mat.index >= pd.to_datetime(start)) & (mat.index <= pd.to_datetime(end))][active]
    per_stock = (seg.iloc[-1] / seg.iloc[0] - 1).dropna()
    ew.append(per_stock.mean())
    iseg = idx[(idx.index >= pd.to_datetime(start)) & (idx.index <= pd.to_datetime(end))]
    bh.append(iseg.iloc[-1] / iseg.iloc[0] - 1)

strat, ew, bh = np.array(STRAT), np.array(ew), np.array(bh)
crash, rally = bh < -0.05, bh > 0.05

def comp(x): return np.prod(1 + x) - 1
def cagr(x): return (1 + comp(x)) ** (1 / 10) - 1

print("=" * 96)
print("FAIR BENCHMARK: strategy vs EQUAL-WEIGHT BUY&HOLD of the SAME universe vs NIFTY100 index")
print("2015-2024, 40 quarters. EW-hold carries the SAME survivorship bias as the strategy.")
print("=" * 96)
print(f"{'':<34}{'strategy':>12}{'EW-hold univ':>14}{'NIFTY100':>12}")
print(f"{'compounded (10yr)':<34}{comp(strat):>11.0%}{comp(ew):>14.0%}{comp(bh):>12.0%}")
print(f"{'CAGR':<34}{cagr(strat):>11.1%}{cagr(ew):>14.1%}{cagr(bh):>12.1%}")
print(f"{'mean / quarter':<34}{strat.mean():>11.2%}{ew.mean():>14.2%}{bh.mean():>12.2%}")
print(f"{'std / quarter (vol)':<34}{strat.std():>11.2%}{ew.std():>14.2%}{bh.std():>12.2%}")
print(f"{'quarterly Sharpe (ann.)':<34}"
      f"{strat.mean()/strat.std()*2:>11.2f}{ew.mean()/ew.std()*2:>14.2f}{bh.mean()/bh.std()*2:>12.2f}")
print(f"{'worst quarter':<34}{strat.min():>11.2%}{ew.min():>14.2%}{bh.min():>12.2%}")
print(f"{'negative quarters':<34}{(strat<0).sum():>9d}/40{(ew<0).sum():>12d}/40{(bh<0).sum():>10d}/40")
print()
print(f"{'CRASH quarters (n=%d) mean' % crash.sum():<34}"
      f"{strat[crash].mean():>11.2%}{ew[crash].mean():>14.2%}{bh[crash].mean():>12.2%}")
print(f"{'RALLY quarters (n=%d) mean' % rally.sum():<34}"
      f"{strat[rally].mean():>11.2%}{ew[rally].mean():>14.2%}{bh[rally].mean():>12.2%}")
print()
print("-" * 96)
print("The number that matters: strategy vs EW-HOLD (both survivorship-biased).")
print(f"  strategy mean/qtr {strat.mean():.2%}  -  EW-hold mean/qtr {ew.mean():.2%}  "
      f"=  TRUE excess {strat.mean()-ew.mean():+.2%} / qtr")
print(f"  quarters strategy beat EW-hold: {(strat>ew).sum()}/40")
print(f"  In CRASH quarters:  strategy {strat[crash].mean():+.2%}  vs  EW-hold {ew[crash].mean():+.2%}"
      f"  -> protection is {'REAL (strategy, not universe)' if strat[crash].mean()>ew[crash].mean()+0.03 else 'mostly the universe'}")
