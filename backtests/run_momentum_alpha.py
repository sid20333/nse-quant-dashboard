"""
run_momentum_alpha.py — First real ALPHA engine attempt.

Thesis (from factor_power_diagnostic.py): the ONLY clean price factor with
cross-sectional selection power in this universe is 12-1 momentum (monthly IC
+0.029; long-only top-15 compounded +791% vs EW-hold +512%). The current
engine's 50/200-SMA 'trend' signal barely selects, and inverse-vol sizing is a
raw-return drag. So: rank by momentum, hold the top names equal-weight,
rebalance monthly. Keep the 200-SMA regime filter ONLY as a separable risk
overlay (the one thing shown to protect in crashes), and measure with vs
without it so selection-alpha and risk-overlay are not conflated.

This is a CONTINUOUS backtest (one equity curve 2015-2024, positions carried
across months), which is a more honest track record than the earlier
quarterly-reset harness. Real Indian delivery costs + slippage on every trade.

The bar is NOT the index. It is EW-hold of the SAME universe, run through the
SAME engine and cost model (mode='ew_all'), so survivorship bias cancels and
what's left is genuine selection skill.

Modes:
  ew_all         benchmark: hold ALL active names equal-weight, monthly reweight
  mom            top-15 by 12-1 momentum, equal-weight, monthly (pure alpha)
  mom_regime     mom + go to cash when NIFTY100 < its 200-day SMA (alpha+overlay)
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
TOP_N = 15
COSTS = IndianEquityCosts()
BUY_FRAC = COSTS.buy_cost(1.0)
SLIP = 0.0015

prov = YFinanceDataProvider(cache_dir=CACHE_DIR)
closes, opens = {}, {}
for s in UNIVERSE:
    try:
        df = prov.get_daily_ohlcv(s, "2013-01-01", "2024-12-31")
        closes[s] = df.set_index("date")["close"]
        opens[s] = df.set_index("date")["open"]
    except Exception:
        pass
close = pd.DataFrame(closes).sort_index()
open_ = pd.DataFrame(opens).sort_index()
idx = prov.get_daily_ohlcv(REGIME_INDEX, "2013-01-01", "2024-12-31").set_index("date")["close"]
idx_sma = idx.rolling(200).mean()

month_ends = close.loc["2014-12-01":"2024-12-31"].resample("ME").last().index
rebal = sorted(set(close.index[close.index <= me][-1] for me in month_ends if (close.index <= me).any()))
rebal = [d for d in rebal if d >= pd.Timestamp("2015-01-01")]


def momentum_scores(pos):
    def px(off): return close.iloc[pos - off]
    m12 = px(21) / px(252) - 1
    m6 = px(21) / px(126) - 1
    return ((m12.rank() + m6.rank()) / 2)   # avg rank of 12-1 and 6-1


def regime_ok(d):
    i = idx[idx.index <= d]; s = idx_sma[idx_sma.index <= d]
    if i.empty or s.empty or pd.isna(s.iloc[-1]):
        return True
    return i.iloc[-1] > s.iloc[-1]


def next_open(sym, d):
    fut = open_[sym][open_[sym].index > d].dropna()
    return fut.iloc[0] if len(fut) else np.nan


def run(mode, freq="M", reweight=True):
    cash = CAPITAL
    shares = {}                      # sym -> share count
    equity_dates, equity = [], []
    turnover_val = 0.0
    cost_paid = 0.0

    if freq == "Q":
        rebal_set = set(pd.Series(rebal, index=rebal).resample("QE").last().dropna().values)
    else:
        rebal_set = set(rebal)

    for i in range(len(rebal) - 1):
        d = rebal[i]
        pos = close.index.get_loc(d)
        if pos < 252:
            equity_dates.append(d); equity.append(cash); continue

        if d not in rebal_set:       # not a rebalance date: just mark-to-market and carry
            mtm = cash + sum(shares[s] * close.iloc[pos][s] for s in shares if s in close.columns)
            equity_dates.append(d); equity.append(mtm); continue

        active = close.iloc[pos].dropna().index
        if mode == "ew_all":
            target = list(active)
        else:
            sc = momentum_scores(pos).dropna()
            target = list(sc.reindex(active).dropna().nlargest(TOP_N).index)

        go_cash = (mode in ("mom_regime",)) and (not regime_ok(d))
        if go_cash:
            target = []

        # mark current holdings, decide sells (exit anything not in target)
        for sym in list(shares.keys()):
            if sym not in target:
                px = next_open(sym, d)
                if np.isnan(px):
                    continue
                px *= (1 - SLIP)
                turn = shares[sym] * px
                c = COSTS.sell_cost(turn)
                cash += turn - c
                cost_paid += c; turnover_val += turn
                del shares[sym]

        # equal-weight target: compute portfolio value, desired per-name value
        holdings_val = sum(shares[s] * close.iloc[pos][s] for s in shares if s in close.columns)
        port_val = cash + holdings_val
        if target and not reweight:
            # low-turnover: only fund NEW entrants from freed cash; let holds drift
            new_names = [s for s in target if s not in shares]
            if new_names:
                per = cash / len(new_names)
                for sym in new_names:
                    px = next_open(sym, d)
                    if np.isnan(px):
                        continue
                    px *= (1 + SLIP)
                    spend = min(per, cash / (1 + BUY_FRAC))
                    if spend <= 0:
                        continue
                    add = spend / px
                    turn = add * px
                    c = COSTS.buy_cost(turn)
                    cash -= turn + c
                    cost_paid += c; turnover_val += turn
                    shares[sym] = shares.get(sym, 0) + add
        elif target:
            desired = port_val / len(target)
            # rebalance existing + add new
            for sym in target:
                cur_val = shares.get(sym, 0) * close.iloc[pos][sym]
                diff = desired - cur_val
                if diff > 0:  # buy up
                    px = next_open(sym, d)
                    if np.isnan(px):
                        continue
                    px *= (1 + SLIP)
                    spend = min(diff, cash / (1 + BUY_FRAC))
                    if spend <= 0:
                        continue
                    add = spend / px
                    turn = add * px
                    c = COSTS.buy_cost(turn)
                    cash -= turn + c
                    cost_paid += c; turnover_val += turn
                    shares[sym] = shares.get(sym, 0) + add
                elif diff < 0:  # trim
                    px = next_open(sym, d)
                    if np.isnan(px):
                        continue
                    px *= (1 - SLIP)
                    sell_sh = min(-diff / px, shares[sym])
                    turn = sell_sh * px
                    c = COSTS.sell_cost(turn)
                    cash += turn - c
                    cost_paid += c; turnover_val += turn
                    shares[sym] -= sell_sh

        # mark-to-market at this rebalance
        mtm = cash + sum(shares[s] * close.iloc[pos][s] for s in shares if s in close.columns)
        equity_dates.append(d); equity.append(mtm)

    eq = pd.Series(equity, index=equity_dates)
    total = eq.iloc[-1] / CAPITAL - 1
    monthly = eq.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    cagr = (eq.iloc[-1] / CAPITAL) ** (12 / len(monthly)) - 1 if len(monthly) else 0
    return {"mode": mode, "total": total, "cagr": cagr, "sharpe": sharpe, "maxDD": dd,
            "turnover_x": turnover_val / CAPITAL, "cost_paid": cost_paid, "eq": eq, "monthly": monthly}


res = {
    "ew_all": run("ew_all"),
    "mom": run("mom"),
    "mom_regime": run("mom_regime"),
    "mom_lowturn": run("mom", freq="Q", reweight=False),
    "mom_lowturn_reg": run("mom_regime", freq="Q", reweight=False),
}
ew = res["ew_all"]

print("=" * 104)
print("MOMENTUM ALPHA ENGINE — continuous 2015-2024, monthly rebalance, real Indian costs")
print("Bar to beat = EW-hold same universe (survivorship cancels; residual = selection skill)")
print("=" * 104)
print(f"{'mode':<14}{'final':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}{'turnover':>10}"
      f"{'costsRs':>10}{'vs EW CAGR':>12}")
for m in ["ew_all", "mom", "mom_regime", "mom_lowturn", "mom_lowturn_reg"]:
    r = res[m]
    print(f"{m:<14}{r['total']:>8.0%}{r['cagr']:>8.1%}{r['sharpe']:>8.2f}{r['maxDD']:>9.1%}"
          f"{r['turnover_x']:>9.1f}x{r['cost_paid']:>10,.0f}{(r['cagr']-ew['cagr'])*100:>+11.1f}pt")
print("=" * 104)

# crash/rally split vs EW using calendar quarters
def qtr_split(monthly):
    q = (1 + monthly).resample("QE").prod() - 1
    return q
ql = qtr_split(res["mom_lowturn"]["monthly"]); qe = qtr_split(ew["monthly"])
al = ql.index.intersection(qe.index)
ql, qe = ql[al], qe[al]
crash = qe < -0.05
exc = ql - qe
print(f"[mom_lowturn robustness]")
print(f"  CRASH quarters (EW<-5%, n={crash.sum()}): lowturn {ql[crash].mean():+.2%}  vs EW {qe[crash].mean():+.2%}")
print(f"  ALL quarters mean: lowturn {ql.mean():+.2%}  vs EW {qe.mean():+.2%}   quarters beat EW: {(ql>qe).sum()}/{len(ql)}")
print(f"  excess: mean {exc.mean():+.2%}/q  median {exc.median():+.2%}/q  worst {exc.min():+.2%}  best {exc.max():+.2%}")
print(f"  drop best 2 & worst 2 excess quarters, mean excess still: "
      f"{exc.sort_values().iloc[2:-2].mean():+.2%}/q  (checks it isn't a couple of outliers)")
