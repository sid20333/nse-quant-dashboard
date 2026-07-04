"""
run_momentum_longshort.py — Does ADDING SHORTS improve the momentum engine?

Thesis: the factor scan showed low-momentum names underperform (positive IC),
so shorting the losers could (a) add return and (b) cancel market beta ->
market-NEUTRAL, i.e. not eat crashes. That is the crash-resilience the regime
filter and vol-target overlay both failed to give without destroying the alpha.

INDIA REALITY (do not gloss over): overnight shorts are illegal in the cash
segment. The short leg must be single-stock FUTURES (F&O), available for only
~200 names, and carries monthly roll + financing costs a long cash book avoids.
Modeled here as SHORT_CARRY_ANNUAL (an ESTIMATE for roll/financing/borrow).
Also: not every universe name was F&O-eligible historically, so the short leg
is optimistic (assumes all shortable) — another reason to read this as an
upper bound on the short leg's benefit.

Clean period-return framework so long-only and long-short are compared on the
SAME footing (semiannual rebalance, 12&6-1 momentum, real costs approximated
by basket turnover). Reports CAGR, Sharpe, maxDD, and BETA to the index (does
the short leg actually neutralize market risk?).
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
# round-trip friction as a fraction of a name's notional (statutory + slippage)
RT_COST = COSTS.round_trip_pct(CAPITAL / TOP_N) + 2 * 0.0015
SHORT_CARRY_ANNUAL = 0.01     # ESTIMATE: futures roll + financing on short notional

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes = {}
for s in UNIVERSE:
    try:
        closes[s] = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31").set_index("date")["close"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
idx = prov.get_daily_ohlcv(REGIME_INDEX, "2013-01-01", "2024-12-31").set_index("date")["close"]

month_ends = close.loc["2014-12-01":"2024-12-31"].resample("ME").last().index
rebal = sorted(set(close.index[close.index <= me][-1] for me in month_ends if (close.index <= me).any()))
rebal = [d for d in rebal if d >= pd.Timestamp("2015-01-01")]
rebal = list(pd.Series(rebal, index=rebal).resample("2QE").last().dropna().values)  # semiannual


def mom_rank(pos):
    def px(off): return close.iloc[pos - off]
    return ((px(21) / px(252) - 1).rank() + (px(21) / px(126) - 1).rank()) / 2


def basket_ret(names, d0, d1):
    r = (close.loc[d1][names] / close.loc[d0][names] - 1).dropna()
    return r.mean() if len(r) else 0.0, set(r.index)


# build per-period leg returns + turnover
periods = []
prev_long, prev_short = set(), set()
for i in range(len(rebal) - 1):
    d0, d1 = pd.Timestamp(rebal[i]), pd.Timestamp(rebal[i + 1])
    pos = close.index.get_loc(close.index[close.index <= d0][-1])
    if pos < 252:
        continue
    active = close.iloc[pos].dropna().index
    r = mom_rank(pos).reindex(active).dropna()
    longs = set(r.nlargest(TOP_N).index)
    shorts = set(r.nsmallest(TOP_N).index)
    lret, lset = basket_ret(list(longs), d0, d1)
    sret, sset = basket_ret(list(shorts), d0, d1)
    ewret, _ = basket_ret(list(active), d0, d1)
    iret = idx.loc[d1] / idx.loc[d0] - 1
    days = (d1 - d0).days
    l_turn = 1 - len(longs & prev_long) / TOP_N
    s_turn = 1 - len(shorts & prev_short) / TOP_N
    periods.append(dict(lret=lret, sret=sret, ewret=ewret, iret=iret, days=days,
                        l_turn=l_turn, s_turn=s_turn))
    prev_long, prev_short = longs, shorts

P = pd.DataFrame(periods)


def strat(long_w, short_w):
    """period returns for a book with long_w long notional and short_w short notional
    (as fraction of capital). Costs on both legs by turnover; carry on short."""
    long_cost = long_w * P["l_turn"] * RT_COST
    short_cost = short_w * P["s_turn"] * RT_COST
    carry = short_w * SHORT_CARRY_ANNUAL * P["days"] / 365
    return long_w * P["lret"] - short_w * P["sret"] - long_cost - short_cost - carry


def stats(pr, label):
    eq = (1 + pr).cumprod()
    n_per_yr = 365 / P["days"].mean()
    cagr = eq.iloc[-1] ** (1 / (P["days"].sum() / 365)) - 1
    sharpe = pr.mean() / pr.std() * np.sqrt(n_per_yr) if pr.std() > 0 else 0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    beta = np.polyfit(P["iret"], pr, 1)[0]
    crash = P["iret"] < -0.05
    return dict(label=label, cagr=cagr, sharpe=sharpe, dd=dd, beta=beta,
                crashret=pr[crash].mean(), pos=(pr > 0).mean())


books = {
    "EW-hold (bar)":            strat(0, 0) * 0 + P["ewret"],   # passive basket
    "Long-only top20":          strat(1.0, 0.0),
    "Long-short 100/100 (neutral)": strat(1.0, 1.0),
    "Long-short 50/50 (half gross)": strat(0.5, 0.5),
    "Long 100 / Short 50":      strat(1.0, 0.5),
    "Long 130 / Short 30":      strat(1.3, 0.3),
    "Short-only losers":        -1.0 * P["sret"] - 1.0 * P["s_turn"] * RT_COST - 1.0 * SHORT_CARRY_ANNUAL * P["days"] / 365,
}

print("=" * 104)
print(f"LONG-SHORT MOMENTUM — semiannual, top/bottom {TOP_N}, 2015-2024, real costs "
      f"(RT {RT_COST:.2%}), short carry {SHORT_CARRY_ANNUAL:.0%}/yr")
print("=" * 104)
print(f"{'book':<32}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'beta':>7}{'crashRet':>10}{'pos%':>7}")
for lbl, pr in books.items():
    st = stats(pr, lbl)
    print(f"{lbl:<32}{st['cagr']:>8.1%}{st['sharpe']:>8.2f}{st['dd']:>8.1%}{st['beta']:>7.2f}"
          f"{st['crashret']:>10.2%}{st['pos']*100:>6.0f}%")
print("=" * 104)
print("beta ~0 = market-neutral (won't eat index crashes). crashRet = mean return when index < -5%.")
print("Read: does the short leg lift Sharpe / kill beta / protect in crashes, net of carry + costs?")
