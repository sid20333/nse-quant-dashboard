"""
run_signal_variants.py — Improve momentum at the SIGNAL level, not via overlays.

Every timing/vol OVERLAY has failed (regime, market vol-target, own-vol scaling)
because crashes here come from calm. So instead change WHAT we hold, not WHEN:
  raw        : rank by 12-1 & 6-1 raw return (current engine)
  riskadj    : rank by momentum / own volatility (tilts away from the high-beta
               names that crash hardest — targets the crash tail structurally)
  idio       : idiosyncratic momentum — rank by momentum of the stock's return
               NET of its market beta (Blitz et al: milder momentum crashes)
Judge on Sharpe, maxDD, worst month vs raw momentum and EW-hold.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL, TOP_N = 500_000.0, 20
COSTS = IndianEquityCosts()
RT_COST = COSTS.round_trip_pct(CAPITAL / TOP_N) + 2 * 0.0015

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
dret = close.pct_change()
idx = prov.get_daily_ohlcv(REGIME_INDEX, "2013-01-01", "2024-12-31").set_index("date")["close"]
idx_ret = idx.pct_change()

reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
reb = [d for d in reb if d >= pd.Timestamp("2014-06-01")]

def scores(pos, kind):
    def px(o): return close.iloc[pos - o]
    raw = (px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank()
    if kind == "raw":
        return raw
    if kind == "riskadj":
        vol = dret.iloc[pos-126:pos].std()
        radj = ((px(21)/px(252)-1) / vol).rank() + ((px(21)/px(126)-1) / vol).rank()
        return radj
    if kind == "idio":
        win = dret.iloc[pos-252:pos]
        iw = idx_ret.iloc[pos-252:pos]
        resid_mom = {}
        for s in win.columns:
            y = win[s].dropna()
            x = iw.reindex(y.index)
            xy = pd.concat([x, y], axis=1).dropna()
            if len(xy) < 120:
                continue
            beta = np.polyfit(xy.iloc[:, 0], xy.iloc[:, 1], 1)[0]
            resid = xy.iloc[:, 1] - beta * xy.iloc[:, 0]
            resid_mom[s] = (1 + resid.iloc[-231:-21]).prod() - 1  # ~12-1 on residuals
        return pd.Series(resid_mom).rank()

def book_daily(kind):
    baskets = {}
    for d in reb:
        pos = close.index.get_loc(close.index[close.index <= d][-1])
        if pos < 252:
            continue
        sc = scores(pos, kind).reindex(close.iloc[pos].dropna().index).dropna()
        baskets[d] = list(sc.nlargest(TOP_N).index)
    bd = pd.Series(index=close.index, dtype=float)
    prev = set()
    for d in close.index:
        prior = [r for r in reb if r < d]
        if not prior or prior[-1] not in baskets:
            continue
        bk = baskets[prior[-1]]
        r = dret.loc[d, [s for s in bk if s in dret.columns]].dropna()
        if len(r):
            bd.loc[d] = r.mean()
    return bd.dropna()[lambda s: s.index >= pd.Timestamp("2015-01-01")]

def stats(daily, label):
    eq = (1 + daily).cumprod()
    m = (1 + daily).resample("ME").prod() - 1
    return dict(label=label, cagr=eq.iloc[-1] ** (252/len(daily)) - 1,
                sharpe=daily.mean()/daily.std()*np.sqrt(252),
                dd=((eq-eq.cummax())/eq.cummax()).min(), worst=m.min())

ew = dret.mean(axis=1)
ew = ew[ew.index >= pd.Timestamp("2015-01-01")]
print("Computing signal variants (idio regression is slow)...\n")
results = [stats(ew, "EW-hold (bar)")]
series = {}
for kind, lbl in [("raw", "Momentum raw (current)"), ("riskadj", "Risk-adjusted momentum"),
                  ("idio", "Idiosyncratic momentum")]:
    bd = book_daily(kind); series[lbl] = bd
    results.append(stats(bd, lbl))

print("=" * 84)
print("SIGNAL-LEVEL MOMENTUM VARIANTS — 2015-2024, semiannual top-20, real costs")
print("=" * 84)
print(f"{'signal':<28}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}{'worstMo':>9}")
for r in results:
    print(f"{r['label']:<28}{r['cagr']:>8.1%}{r['sharpe']:>8.2f}{r['dd']:>9.1%}{r['worst']:>9.1%}")
print("=" * 84)
raw_m = (1 + series["Momentum raw (current)"]).resample("ME").prod() - 1
worst = raw_m.nsmallest(4).index
print("Worst 4 raw-momentum months — do the variants cushion the crash tail?")
for w in worst:
    line = f"  {w.strftime('%Y-%m')}: raw {raw_m.loc[w]:+.1%}"
    for lbl in ["Risk-adjusted momentum", "Idiosyncratic momentum"]:
        mv = (1 + series[lbl]).resample("ME").prod() - 1
        line += f"   {lbl.split()[0][:6]} {mv.get(w, float('nan')):+.1%}"
    print(line)
