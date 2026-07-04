"""
run_risk_managed_momentum.py — Fix momentum's crash weakness (seen LIVE in
Mar-2026: momentum -15% vs index -11.7%).

The overlays we already tried and rejected keyed off the MARKET (regime filter,
market vol-target) — both killed the alpha because momentum's vol spikes often
coincide with profitable trends, not crashes. The correct signal is momentum's
OWN volatility: momentum crashes are preceded by the momentum strategy's own
return stream turning volatile (Daniel & Moskowitz 2016, "Momentum Crashes").

Method: scale the book's exposure to target a constant strategy volatility,
using the momentum book's OWN trailing realized vol (lagged, no look-ahead),
reset monthly:
    exposure_t = min(cap, target_vol / trailing_ann_vol_of_book)
De-risk-only (cap=1.0) respects long-only/no-leverage. cap>1.0 is the fuller
version (needs futures leverage — flagged). Judge on Sharpe, drawdown, and the
crash months specifically — NOT raw return (de-risking lowers both return+risk).
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
VOL_LOOKBACK = 63          # ~3 months of trading days
FETCH_START, END = "2013-01-01", "2024-12-31"

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, FETCH_START, END).set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
dret = close.pct_change()
idx = prov.get_daily_ohlcv(REGIME_INDEX, FETCH_START, END).set_index("date")["close"]

reb = [pd.Timestamp(d) for d in pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values]
reb = [d for d in reb if d >= pd.Timestamp("2014-06-01")]

def basket(d):
    pos = close.index.get_loc(close.index[close.index <= d][-1])
    if pos < 252:
        return None
    def px(o): return close.iloc[pos - o]
    r = ((px(21)/px(252)-1).rank() + (px(21)/px(126)-1).rank())/2
    return list(r.reindex(close.iloc[pos].dropna().index).dropna().nlargest(TOP_N).index)

baskets = {d: basket(d) for d in reb}

# daily equal-weight book return of the held momentum basket
book_daily = pd.Series(index=close.index, dtype=float)
for i, d in enumerate(close.index):
    prior = [r for r in reb if r < d]
    if not prior:
        continue
    bk = baskets.get(prior[-1])
    if not bk:
        continue
    r = dret.loc[d, [s for s in bk if s in dret.columns]].dropna()
    if len(r):
        book_daily.loc[d] = r.mean()
book_daily = book_daily.dropna()
book_daily = book_daily[book_daily.index >= pd.Timestamp("2015-01-01")]

# EW-hold universe daily (benchmark)
ew_daily = dret.mean(axis=1).reindex(book_daily.index)

full_vol = book_daily.std() * np.sqrt(252)
print(f"Momentum book full-sample realized vol = {full_vol:.1%}/yr  (targets set around this)\n")

# trailing annualized vol, lagged one day
trail_vol = book_daily.rolling(VOL_LOOKBACK).std().shift(1) * np.sqrt(252)

def run_scaled(target, cap):
    # exposure set at each month start, held through the month
    exp = pd.Series(index=book_daily.index, dtype=float)
    month_id = book_daily.index.to_period("M")
    cur = 1.0
    prev_month = None
    for d in book_daily.index:
        m = d.to_period("M")
        if m != prev_month:
            tv = trail_vol.get(d, np.nan)
            cur = min(cap, target / tv) if (pd.notna(tv) and tv > 0) else 1.0
            prev_month = m
        exp.loc[d] = cur
    # monthly cost from exposure changes
    monthly_exp = exp.groupby(month_id).first()
    turn_cost_m = monthly_exp.diff().abs().fillna(0) * RT_COST
    scaled_daily = exp * book_daily
    # subtract monthly turnover cost on the first day of each month
    firsts = book_daily.groupby(month_id).head(1).index
    for d in firsts:
        scaled_daily.loc[d] -= turn_cost_m.get(d.to_period("M"), 0)
    return scaled_daily, exp

def stats(daily, label):
    m = (1 + daily).resample("ME").prod() - 1
    eq = (1 + daily).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(daily)) - 1
    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    worst_m = m.min()
    return dict(label=label, cagr=cagr, sharpe=sharpe, dd=dd, worst_m=worst_m,
                avg_exp=None)

rows = []
base_stats = stats(book_daily, "Momentum (unscaled)")
rows.append((base_stats, 1.0))
rows.append((stats(ew_daily, "EW-hold (bar)"), None))

configs = [(full_vol, 1.0, "RM de-risk only (cap 1.0)"),
           (full_vol, 1.5, "RM cap 1.5 (needs leverage)"),
           (0.15, 1.5, "RM target 15% cap 1.5")]
rm_series = {}
for target, cap, lbl in configs:
    sc, exp = run_scaled(target, cap)
    st = stats(sc, lbl); st["avg_exp"] = exp.mean()
    rm_series[lbl] = sc
    rows.append((st, cap))

print("=" * 96)
print(f"RISK-MANAGED MOMENTUM — scale by book's OWN vol (2015-2024, real costs, semiannual top-{TOP_N})")
print("=" * 96)
print(f"{'strategy':<32}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}{'worstMo':>9}{'avgExp':>8}")
for st, _ in rows:
    ae = f"{st['avg_exp']:.2f}" if st.get("avg_exp") else "  -"
    print(f"{st['label']:<32}{st['cagr']:>8.1%}{st['sharpe']:>8.2f}{st['dd']:>9.1%}"
          f"{st['worst_m']:>9.1%}{ae:>8}")
print("=" * 96)

# momentum-crash months: worst 5 unscaled book months, see if RM cushioned them
mb = (1 + book_daily).resample("ME").prod() - 1
worst = mb.nsmallest(5).index
best_rm = rm_series["RM cap 1.5 (needs leverage)"]
mrm = (1 + best_rm).resample("ME").prod() - 1
print("Worst 5 momentum months — did own-vol scaling cushion them?")
for w in worst:
    print(f"  {w.strftime('%Y-%m')}:  unscaled {mb.loc[w]:+.1%}   risk-managed {mrm.get(w, float('nan')):+.1%}")
print("\nJudge on Sharpe + maxDD + worstMonth, not CAGR. This targets the crash tail directly.")
