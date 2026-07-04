"""
run_momentum_hardening.py — Validate the low-turnover momentum alpha, then try
to keep it while cutting the -27% drawdown with a volatility-target overlay.

Two parts:
  PART 1 — SENSITIVITY. Is the +5pt/yr edge robust, or a knife-edge fit on my
  arbitrary 12-1 / top-15 / quarterly choices? Sweep momentum lookback, top_n,
  and rebalance frequency (all low-turnover: hold winners, replace dropouts).
  If MOST reasonable settings beat EW-hold, the alpha is structural. If only
  one cell works, it's overfit and we stop.

  PART 2 — VOL TARGET. Momentum is fully invested so it eats crashes. Instead
  of the binary regime filter (which killed momentum by missing recoveries),
  scale GROSS exposure smoothly to target a constant portfolio volatility:
      invested_fraction = min(1, target_vol / trailing_realized_vol)
  Long-only / no leverage, so this can only DE-risk (hold cash) when vol spikes
  - which is exactly when crashes cluster. Goal: shrink drawdown while keeping
  most of the selection alpha.

All runs: continuous 2015-2024, real Indian costs, benchmarked vs EW-hold of
the SAME universe (survivorship cancels).
"""
import sys, warnings
from dataclasses import dataclass, field
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")
import numpy as np, pandas as pd
from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL = 500_000.0
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
rets_d = close.pct_change()

month_ends = close.loc["2014-12-01":"2024-12-31"].resample("ME").last().index
rebal_all = sorted(set(close.index[close.index <= me][-1] for me in month_ends if (close.index <= me).any()))
rebal_all = [d for d in rebal_all if d >= pd.Timestamp("2015-01-01")]


@dataclass
class Cfg:
    name: str
    longs: tuple = (252, 126)   # momentum lookbacks (skip-adjusted), avg-ranked
    skip: int = 21
    top_n: int = 15
    freq: str = "Q"             # M | Q | 2Q
    reweight: bool = False
    vol_target: float = 0.0     # 0 = off; else annualized target (e.g. 0.15)
    vol_lookback: int = 60


def rebal_dates(freq):
    if freq == "M":
        return set(rebal_all)
    ser = pd.Series(rebal_all, index=rebal_all)
    rule = "QE" if freq == "Q" else "2QE"
    return set(ser.resample(rule).last().dropna().values)


def mom_scores(pos, cfg):
    def px(off): return close.iloc[pos - off]
    ranks = []
    for L in cfg.longs:
        ranks.append((px(cfg.skip) / px(L) - 1).rank())
    return sum(ranks) / len(ranks)


def next_open(sym, d):
    fut = open_[sym][open_[sym].index > d].dropna()
    return fut.iloc[0] if len(fut) else np.nan


def gross_exposure(pos, target_names, cfg):
    if cfg.vol_target <= 0 or not target_names:
        return 1.0
    sub = rets_d.iloc[pos - cfg.vol_lookback:pos][list(target_names)].dropna(axis=1, how="all")
    if sub.shape[1] == 0:
        return 1.0
    ew = sub.mean(axis=1)                       # equal-weight book daily return
    vol = ew.std() * np.sqrt(252)
    if not np.isfinite(vol) or vol <= 0:
        return 1.0
    return float(min(1.0, cfg.vol_target / vol))


def run(cfg: Cfg):
    cash = CAPITAL
    shares = {}
    edates, eq = [], []
    turnover_val = cost_paid = 0.0
    rset = rebal_dates(cfg.freq)

    for i in range(len(rebal_all) - 1):
        d = rebal_all[i]
        pos = close.index.get_loc(d)
        if pos < 252:
            edates.append(d); eq.append(cash); continue
        if d not in rset:
            mtm = cash + sum(shares[s] * close.iloc[pos][s] for s in shares if s in close.columns)
            edates.append(d); eq.append(mtm); continue

        active = close.iloc[pos].dropna().index
        sc = mom_scores(pos, cfg).reindex(active).dropna()
        target = list(sc.nlargest(cfg.top_n).index)
        gross = gross_exposure(pos, target, cfg)

        # sells: anything not in target
        for sym in list(shares.keys()):
            if sym not in target:
                px = next_open(sym, d)
                if np.isnan(px):
                    continue
                px *= (1 - SLIP); turn = shares[sym] * px; c = COSTS.sell_cost(turn)
                cash += turn - c; cost_paid += c; turnover_val += turn
                del shares[sym]

        holdings_val = sum(shares[s] * close.iloc[pos][s] for s in shares if s in close.columns)
        port_val = cash + holdings_val
        invest_val = port_val * gross

        if target and (cfg.reweight or cfg.vol_target > 0):
            desired = invest_val / len(target)
            for sym in target:
                cur = shares.get(sym, 0) * close.iloc[pos][sym]
                diff = desired - cur
                px = next_open(sym, d)
                if np.isnan(px):
                    continue
                if diff > 0:
                    spend = min(diff, cash / (1 + BUY_FRAC))
                    if spend <= 0:
                        continue
                    add = spend / (px * (1 + SLIP)); turn = add * px * (1 + SLIP); c = COSTS.buy_cost(turn)
                    cash -= turn + c; cost_paid += c; turnover_val += turn
                    shares[sym] = shares.get(sym, 0) + add
                elif diff < 0:
                    sell_sh = min(-diff / (px * (1 - SLIP)), shares[sym])
                    turn = sell_sh * px * (1 - SLIP); c = COSTS.sell_cost(turn)
                    cash += turn - c; cost_paid += c; turnover_val += turn
                    shares[sym] -= sell_sh
        elif target:  # low-turnover drift: only fund new entrants from cash
            new = [s for s in target if s not in shares]
            if new:
                per = cash / len(new)
                for sym in new:
                    px = next_open(sym, d)
                    if np.isnan(px):
                        continue
                    spend = min(per, cash / (1 + BUY_FRAC))
                    if spend <= 0:
                        continue
                    add = spend / (px * (1 + SLIP)); turn = add * px * (1 + SLIP); c = COSTS.buy_cost(turn)
                    cash -= turn + c; cost_paid += c; turnover_val += turn
                    shares[sym] = shares.get(sym, 0) + add

        mtm = cash + sum(shares[s] * close.iloc[pos][s] for s in shares if s in close.columns)
        edates.append(d); eq.append(mtm)

    s = pd.Series(eq, index=edates)
    m = s.pct_change().dropna()
    sharpe = m.mean() / m.std() * np.sqrt(12) if m.std() > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min()
    cagr = (s.iloc[-1] / CAPITAL) ** (12 / len(m)) - 1 if len(m) else 0
    return {"cagr": cagr, "sharpe": sharpe, "dd": dd, "turn": turnover_val / CAPITAL,
            "cost": cost_paid, "eq": s, "monthly": m}


EW = run(Cfg("ew", top_n=999))          # top_n huge -> holds ~all active names = EW-hold bar
ewc = EW["cagr"]

print("=" * 100)
print(f"BAR: EW-hold same universe  CAGR {ewc:.1%}  Sharpe {EW['sharpe']:.2f}  maxDD {EW['dd']:.1%}")
print("=" * 100)
print("PART 1 — SENSITIVITY (low-turnover momentum). Does the edge survive different params?")
print(f"{'lookback':>10}{'top_n':>7}{'freq':>6}{'CAGR':>8}{'vsEW':>8}{'Sharpe':>8}{'maxDD':>8}{'turn':>7}")
grid = []
for longs, lbl in [((252,), "12-1"), ((252, 126), "12&6-1"), ((189,), "9-1"), ((126,), "6-1"), ((252, 189, 126), "12/9/6")]:
    for top_n in [10, 15, 20, 30]:
        for freq in ["M", "Q", "2Q"]:
            r = run(Cfg("g", longs=longs, top_n=top_n, freq=freq, reweight=False))
            grid.append((lbl, top_n, freq, r))
            print(f"{lbl:>10}{top_n:>7}{freq:>6}{r['cagr']:>8.1%}{(r['cagr']-ewc)*100:>+7.1f}p"
                  f"{r['sharpe']:>8.2f}{r['dd']:>8.1%}{r['turn']:>6.0f}x")
beat = sum(1 for *_ , r in grid if r["cagr"] > ewc)
print(f"\n  configs beating EW-hold: {beat}/{len(grid)}   "
      f"median vsEW: {np.median([r['cagr']-ewc for *_, r in grid])*100:+.1f}pt")

print("\n" + "=" * 100)
print("PART 2 — VOL-TARGET overlay on 12&6-1 / top-15 / quarterly (smooth de-risk vs binary regime)")
print(f"{'config':<26}{'CAGR':>8}{'vsEW':>8}{'Sharpe':>8}{'maxDD':>8}{'turn':>7}{'costRs':>9}")
base = run(Cfg("base", longs=(252, 126), top_n=15, freq="Q", reweight=False))
print(f"{'base (no overlay)':<26}{base['cagr']:>8.1%}{(base['cagr']-ewc)*100:>+7.1f}p"
      f"{base['sharpe']:>8.2f}{base['dd']:>8.1%}{base['turn']:>6.0f}x{base['cost']:>9,.0f}")
for vt in [0.20, 0.15, 0.12]:
    r = run(Cfg("vt", longs=(252, 126), top_n=15, freq="Q", reweight=False, vol_target=vt))
    print(f"{'vol-target '+str(int(vt*100))+'%':<26}{r['cagr']:>8.1%}{(r['cagr']-ewc)*100:>+7.1f}p"
          f"{r['sharpe']:>8.2f}{r['dd']:>8.1%}{r['turn']:>6.0f}x{r['cost']:>9,.0f}")
