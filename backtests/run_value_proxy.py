"""
run_value_proxy.py — Add a SECOND, momentum-diversifying sleeve using data we
actually have.

True fundamental value needs point-in-time financials — confirmed unavailable
(yfinance returns only a current snapshot = look-ahead). So use the clean
PRICE-BASED cousin of value: LONG-TERM REVERSAL (DeBondt-Thaler) — buy stocks
that badly UNDER-performed over ~5yr-to-1yr; they mean-revert. It is the classic
value proxy and, crucially, is negatively correlated with 12-1 momentum, so it
is the natural diversifier for momentum's crash tail.

Tests:
  1. Is long-term reversal (LTR) a real factor here? (vs EW-hold)
  2. Is it actually anti-correlated with momentum? (the whole point)
  3. Does a 50/50 MOMENTUM + LTR book beat momentum alone on Sharpe / crash tail?

Needs 5yr lookback, so test window is 2018-2024 (~6.5y, shorter sample — caveat).
Narrow (survivorship-safest) universe, semiannual, real costs.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from scipy import stats
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL, TOP_N = 500_000.0, 20
COSTS = IndianEquityCosts()
RT = COSTS.round_trip_pct(CAPITAL / TOP_N) + 2 * 0.0015
START = pd.Timestamp("2018-07-01")

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
dret = close.pct_change()
mret = close.resample("ME").last().pct_change()

reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
reb = [d for d in reb if d >= pd.Timestamp("2018-01-01")]

def picks(pos, kind):
    def px(o): return close.iloc[pos - o]
    if kind == "mom":
        sc = (px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank()
    else:  # long-term reversal: biggest 5yr->1yr LOSERS get highest score
        sc = (-(px(252)/px(1260) - 1)).rank()
    return list(sc.reindex(close.iloc[pos].dropna().index).dropna().nlargest(TOP_N).index)

baskets = {"mom": {}, "ltr": {}}
for d in reb:
    pos = close.index.get_loc(close.index[close.index <= d][-1])
    if pos < 1260:
        continue
    baskets["mom"][d] = picks(pos, "mom")
    baskets["ltr"][d] = picks(pos, "ltr")

def sleeve_monthly(kind):
    prev, last_reb, rets = set(), None, []
    for m in mret.index:
        if m < START:
            continue
        prior = [d for d in reb if d < m and d in baskets[kind]]
        if not prior:
            continue
        rebd = prior[-1]; bk = baskets[kind][rebd]
        row = mret.loc[m]
        r = row[[s for s in bk if s in row.index]].dropna().mean()
        cost = 0.0
        if rebd != last_reb:
            cost = (1 - len(set(bk) & prev)/TOP_N) * RT
            prev, last_reb = set(bk), rebd
        rets.append((m, r - cost))
    return pd.Series(dict(rets))

mom = sleeve_monthly("mom")
ltr = sleeve_monthly("ltr")
idx_m = mret.mean(axis=1).reindex(mom.index)   # EW-hold
combo = 0.5 * mom + 0.5 * ltr

def stat(s, label):
    s = s.dropna()
    eq = (1 + s).cumprod()
    return dict(label=label, cagr=eq.iloc[-1]**(12/len(s))-1, sharpe=s.mean()/s.std()*np.sqrt(12),
                dd=((eq-eq.cummax())/eq.cummax()).min(), worst=s.min())

print("=" * 82)
print(f"MOMENTUM + LONG-TERM-REVERSAL (value proxy) — 2018-2024, narrow universe, real costs")
print("=" * 82)
print(f"{'sleeve':<26}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}{'worstMo':>9}")
for s, lbl in [(idx_m, "EW-hold (bar)"), (mom, "Momentum only"),
               (ltr, "Long-term reversal only"), (combo, "50/50 Momentum + LTR")]:
    st = stat(s, lbl)
    print(f"{st['label']:<26}{st['cagr']:>8.1%}{st['sharpe']:>8.2f}{st['dd']:>9.1%}{st['worst']:>9.1%}")
print("=" * 82)
corr = mom.corr(ltr)
print(f"Correlation(Momentum, LTR) monthly returns = {corr:+.2f}   "
      f"({'GOOD — diversifies' if corr < 0.5 else 'too correlated to help'})")
# LTR alpha vs EW
r = stats.linregress((idx_m - 0.005).dropna(), (ltr - 0.005).reindex(idx_m.index).dropna())
# crash months: worst 4 momentum months, does combo cushion?
worst = mom.nsmallest(4).index
print("\nWorst 4 momentum months — does the 50/50 combo cushion the crash tail?")
for w in worst:
    print(f"  {w.strftime('%Y-%m')}:  mom {mom[w]:+.1%}   LTR {ltr.get(w, float('nan')):+.1%}   combo {combo[w]:+.1%}")
print("\nCaveat: 2018-2024 only (~6.5y, needs 5yr lookback) = shorter sample than the momentum work.")
