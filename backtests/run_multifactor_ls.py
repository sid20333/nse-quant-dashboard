"""
run_multifactor_ls.py — Does multi-factor MARKET-NEUTRAL beat single-factor
long-only momentum? Tests the core 'what works better' claim.

Long-only momentum + long-only reversal were +0.82 correlated (shared market
beta) -> no diversification. The claim: in LONG-SHORT (beta-stripped) form, the
same two factors decorrelate, so combining them raises Sharpe and cuts drawdown.
If true, market-neutral multi-factor is the real upgrade path.

Factors (dollar-neutral spreads, semiannual, real costs both legs):
  MOM_LS : long top-20 momentum  minus short bottom-20 momentum
  REV_LS : long 5yr-1yr losers   minus short 5yr-1yr winners  (value proxy)
  COMBO  : 50/50 of the two market-neutral factors
Reports each factor's Sharpe, the correlation between them, and whether COMBO
beats MOM_LS alone. 2018-2024 (needs 5yr lookback). NOTE: shorts assume F&O
(overnight illegal in cash) + carry — an optimistic upper bound.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL, N = 500_000.0, 20
COSTS = IndianEquityCosts()
RT = COSTS.round_trip_pct(CAPITAL / N) + 2 * 0.0015
CARRY_M = 0.01 / 12
START = pd.Timestamp("2018-07-01")

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
mret = close.resample("ME").last().pct_change()
reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
reb = [d for d in reb if d >= pd.Timestamp("2018-01-01")]

def rank_at(pos, kind):
    def px(o): return close.iloc[pos - o]
    if kind == "mom":
        sc = (px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank()
    else:  # reversal: high score = big 5y-1y loser
        sc = (-(px(252)/px(1260) - 1)).rank()
    return sc.reindex(close.iloc[pos].dropna().index).dropna()

legs = {"mom": {}, "rev": {}}
for d in reb:
    pos = close.index.get_loc(close.index[close.index <= d][-1])
    if pos < 1260:
        continue
    for k in ("mom", "rev"):
        sc = rank_at(pos, k)
        legs[k][d] = (list(sc.nlargest(N).index), list(sc.nsmallest(N).index))

def factor_monthly(kind):
    prevL, prevS, last, out = set(), set(), None, []
    for m in mret.index:
        if m < START:
            continue
        prior = [d for d in reb if d < m and d in legs[kind]]
        if not prior:
            continue
        rebd = prior[-1]; longs, shorts = legs[kind][rebd]
        row = mret.loc[m]
        lr = row[[s for s in longs if s in row.index]].dropna().mean()
        sr = row[[s for s in shorts if s in row.index]].dropna().mean()
        cost = 0.0
        if rebd != last:
            cost = ((1-len(set(longs)&prevL)/N) + (1-len(set(shorts)&prevS)/N)) * RT
            prevL, prevS, last = set(longs), set(shorts), rebd
        out.append((m, (lr - sr) - cost - CARRY_M))   # dollar-neutral spread, short carry
    return pd.Series(dict(out))

mom_ls = factor_monthly("mom")
rev_ls = factor_monthly("rev")
combo = 0.5 * mom_ls + 0.5 * rev_ls

def st(s, lbl):
    s = s.dropna(); eq = (1+s).cumprod()
    return (f"{lbl:<28}{s.mean()*12:>+8.1%}{s.mean()/s.std()*np.sqrt(12):>8.2f}"
            f"{((eq-eq.cummax())/eq.cummax()).min():>9.1%}{s.min():>9.1%}")

print("=" * 72)
print("MARKET-NEUTRAL MULTI-FACTOR (long-short) — 2018-2024, real costs + carry")
print("=" * 72)
print(f"{'factor':<28}{'ann.ret':>8}{'Sharpe':>8}{'maxDD':>9}{'worstMo':>9}")
print(st(mom_ls, "Momentum L/S"))
print(st(rev_ls, "Reversal/value L/S"))
print(st(combo, "50/50 COMBO (neutral)"))
print("=" * 72)
print(f"corr(Momentum_LS, Reversal_LS) = {mom_ls.corr(rev_ls):+.2f}   "
      f"(long-ONLY versions were +0.82)")
print("If corr is now low/negative AND combo Sharpe > momentum Sharpe, market-neutral")
print("multi-factor is the real upgrade — the diversification long-only cannot give.")
