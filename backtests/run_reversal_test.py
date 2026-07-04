"""
run_reversal_test.py — Is there a SECOND price-based alpha besides momentum?

The factor scan found short-term reversal has a real (negative) IC: last
month's losers tend to bounce. But reversal is a fast signal — it needs monthly
(or faster) rebalancing, and turnover is the thing that has killed every edge in
this project. So: does a reversal strategy actually NET positive after real
costs, or is it a cost trap? This closes the loop on 'is momentum the only
price factor that pays here'.

Long bottom-20 by trailing 21-day return (recent losers), equal-weight, monthly
rebalance, real costs. Compared to monthly momentum (fair turnover) and EW-hold.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE

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
mclose = close.resample("ME").last()
mret = mclose.pct_change()

# monthly rebalance dates
rebd = [close.index[close.index <= me][-1] for me in mclose.index]

def sig(pos, kind):
    def px(o): return close.iloc[pos - o]
    if kind == "reversal":       # bottom = recent losers -> want to BUY these -> rank ascending
        return (-(px(0) / px(21) - 1)).rank()          # high score = big loser
    if kind == "momentum":
        return ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())

def run(kind):
    prev, cash_curve = set(), []
    rets = []
    for i in range(len(rebd) - 1):
        d, d1 = rebd[i], rebd[i + 1]
        pos = close.index.get_loc(d)
        if pos < 252:
            continue
        sc = sig(pos, kind).reindex(close.iloc[pos].dropna().index).dropna()
        picks = set(sc.nlargest(TOP_N).index)
        row = mret.loc[mclose.index[mclose.index > d][0]] if (mclose.index > d).any() else None
        if row is None:
            continue
        r = row[[s for s in picks if s in row.index]].dropna().mean()
        turn = 1 - len(picks & prev) / TOP_N
        rets.append(r - turn * RT_COST)
        prev = picks
    rets = pd.Series(rets)
    eq = (1 + rets).cumprod()
    return dict(cagr=eq.iloc[-1] ** (12/len(rets)) - 1, sharpe=rets.mean()/rets.std()*np.sqrt(12),
                dd=((eq-eq.cummax())/eq.cummax()).min(),
                avg_turn=np.mean([1]*0) if False else None, n=len(rets))

# EW benchmark monthly
ewr = mret.mean(axis=1).dropna()
ewr = ewr[ewr.index >= pd.Timestamp("2015-02-01")]
eweq = (1 + ewr).cumprod()
ew = dict(cagr=eweq.iloc[-1] ** (12/len(ewr)) - 1, sharpe=ewr.mean()/ewr.std()*np.sqrt(12),
          dd=((eweq-eweq.cummax())/eweq.cummax()).min())

print("=" * 76)
print("SHORT-TERM REVERSAL vs MOMENTUM vs EW — monthly, top-20, real costs, 2015-2024")
print("=" * 76)
print(f"{'strategy':<30}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}")
print(f"{'EW-hold (bar)':<30}{ew['cagr']:>8.1%}{ew['sharpe']:>8.2f}{ew['dd']:>9.1%}")
for kind, lbl in [("reversal", "Short-term reversal (losers)"), ("momentum", "Momentum (monthly)")]:
    r = run(kind)
    print(f"{lbl:<30}{r['cagr']:>8.1%}{r['sharpe']:>8.2f}{r['dd']:>9.1%}")
print("=" * 76)
print("If reversal < EW after costs, it's a cost trap and momentum stands alone among")
print("price factors. If it beats EW and is uncorrelated w/ momentum, it's a real 2nd sleeve.")
