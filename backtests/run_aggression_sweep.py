"""
run_aggression_sweep.py — Dial the strategy from DEFENSIVE to AGGRESSIVE and
measure the real tradeoff on 2015-2024 (real data, real Indian costs).

We established: the edge is crash protection (+9.6% OOS excess in the 6 crash
quarters) bought by LAGGING rallies (-1.9%). "More aggressive" = give back some
crash protection to recover rally upside. The levers (user's hard constraints
- Rs 5L, 5-15 positions, long-only - are held FIXED; aggression comes only
from how we treat regime, stops, concentration, and sizing):

  regime_mode : 'block'  = sit out entirely when NIFTY100 < 200MA (current v5)
                'scaled' = stay invested in bear but at BEAR_SIZE_MULT size
                'off'    = ignore regime, always fully invested
  trail_mult  : ATR trailing-stop width. Higher = let winners run (less clipping)
  corr_cap    : max pairwise correlation for new entries. Higher = allow the
                book to concentrate into the strongest theme
  sizing      : 'invvol'   = 1/ATR%  (defensive, tilts to calm laggards)
                'equal'    = equal weight
                'momentum' = weight by trailing 60d return (chase strength)

Each config is judged on the SAME axes so the tradeoff is explicit:
compounded return, Sharpe, worst drawdown, and - crucially - the split between
CRASH-quarter and RALLY-quarter performance vs NIFTY100 buy-and-hold.
"""
import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")

import numpy as np
import pandas as pd

from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.moving_average_screener import compute_ma_state
from quant_engine.technical import average_true_range
from quant_engine.costs import IndianEquityCosts
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

HIST_START, DATA_END = "2013-01-01", "2024-12-31"
CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
CAPITAL = 500_000.0
MIN_POSITIONS, MAX_POSITIONS = 5, 15          # user hard constraint - fixed
SLIPPAGE_PCT = 0.0015
ATR_INITIAL_MULT = 1.5
ATR_PROFIT_LOCK_MULT = 1.0
CORR_LOOKBACK_DAYS = 60
REGIME_SMA_WINDOW = 200
COSTS = IndianEquityCosts()
BUY_FRAC = COSTS.buy_cost(1.0)


@dataclass
class Config:
    name: str
    regime_mode: str = "block"      # block | scaled | off
    bear_size_mult: float = 0.5     # only used when regime_mode == 'scaled'
    trail_mult: float = 2.0
    corr_cap: float = 0.70
    sizing: str = "invvol"          # invvol | equal | momentum


provider = YFinanceDataProvider(cache_dir=CACHE_DIR)
symbol_dfs = {}
for sym in UNIVERSE:
    try:
        df = provider.get_daily_ohlcv(sym, HIST_START, DATA_END)
    except Exception:
        continue
    if len(df) >= 250:
        symbol_dfs[sym] = df.reset_index(drop=True)
index_df = provider.get_daily_ohlcv(REGIME_INDEX, HIST_START, DATA_END)
index_series = index_df.set_index("date")["close"]
index_sma200 = index_series.rolling(REGIME_SMA_WINDOW).mean()
atr_cache = {s: average_true_range(d["high"], d["low"], d["close"], 14) for s, d in symbol_dfs.items()}

WINDOWS = []
for yr in range(2015, 2025):
    WINDOWS += [(f"{yr}-01-01", f"{yr}-03-31"), (f"{yr}-04-01", f"{yr}-06-30"),
                (f"{yr}-07-01", f"{yr}-09-30"), (f"{yr}-10-01", f"{yr}-12-31")]


def regime_bullish(as_of):
    i = index_series[index_series.index <= as_of]
    s = index_sma200[index_sma200.index <= as_of]
    if i.empty or s.empty or pd.isna(s.iloc[-1]):
        return True
    return i.iloc[-1] > s.iloc[-1]


def atr_pct(df, w=14):
    a = average_true_range(df["high"], df["low"], df["close"], w)
    p = df["close"].iloc[-1]
    return float(a.iloc[-1] / p) if p > 0 else np.nan


def momentum_score(df):
    c = df["close"]
    if len(c) < 61:
        return 0.0
    return max(c.iloc[-1] / c.iloc[-60] - 1.0, 0.0)


def get_atr_at(symbol, as_of):
    df = symbol_dfs[symbol]
    idx = df.index[df["date"] <= as_of]
    return atr_cache[symbol].iloc[idx[-1]] if len(idx) else np.nan


def returns_series(df, as_of, lookback=CORR_LOOKBACK_DAYS):
    s = df[df["date"] <= as_of].tail(lookback)
    return s.set_index("date")["close"].pct_change().dropna()


def select_with_corr(ranked, as_of, max_n, max_corr, held):
    selected = list(held)
    cache = {}

    def R(sym):
        if sym not in cache:
            cache[sym] = returns_series(symbol_dfs[sym], as_of)
        return cache[sym]

    for sym in ranked:
        if len(selected) >= max_n:
            break
        if sym in held:
            continue
        cr = R(sym)
        if cr.empty:
            continue
        bad = False
        for h in selected:
            j = pd.concat([cr, R(h)], axis=1).dropna()
            if len(j) < 10:
                continue
            c = j.iloc[:, 0].corr(j.iloc[:, 1])
            if pd.notna(c) and c > max_corr:
                bad = True
                break
        if not bad:
            selected.append(sym)
    return [s for s in selected if s not in held]


def run_window(cfg: Config, start, end):
    decision_dates = pd.date_range(start, end, freq="W-SAT")
    positions, cash = {}, CAPITAL
    eq, eq_dates, trades = [], [], []

    def sell(pos, shares, ref):
        px = ref * (1 - SLIPPAGE_PCT)
        turn = shares * px
        net = turn - COSTS.sell_cost(turn)
        return net, (net / shares) / pos["entry_price"] - 1

    for ip, dd in enumerate(decision_dates):
        prior = decision_dates[ip - 1] if ip > 0 else pd.to_datetime(start) - pd.Timedelta(days=1)

        for sym in list(positions.keys()):
            df = symbol_dfs[sym]
            bars = df[(df["date"] > prior) & (df["date"] <= dd)]
            pos = positions[sym]
            for _, bar in bars.iterrows():
                pos["peak_price"] = max(pos["peak_price"], bar["close"])
                ea = pos["entry_atr"]
                if pos["peak_price"] >= pos["entry_price"] + ATR_PROFIT_LOCK_MULT * ea:
                    stop = pos["peak_price"] - cfg.trail_mult * ea
                    reason = "trail"
                else:
                    stop = pos["entry_price"] - ATR_INITIAL_MULT * ea
                    reason = "init"
                if bar["close"] <= stop:
                    net, ret = sell(pos, pos["shares"], stop)
                    cash += net
                    trades.append({"return_pct": ret, "reason": reason})
                    del positions[sym]
                    break

        scores = {}
        for sym, df in symbol_dfs.items():
            sl = df[df["date"] <= dd]
            if len(sl) < 210:
                continue
            st = compute_ma_state(sl)
            scores[sym] = (st.bullish_score, st.death_cross_50_200)

        for sym in list(positions.keys()):
            if sym in scores and scores[sym][1]:
                df = symbol_dfs[sym]
                xb = df[df["date"] > dd]
                if xb.empty:
                    continue
                pos = positions.pop(sym)
                net, ret = sell(pos, pos["shares"], xb.iloc[0]["open"])
                cash += net
                trades.append({"return_pct": ret, "reason": "dcross"})

        bull = regime_bullish(dd)
        if cfg.regime_mode == "off":
            allow, size_mult = True, 1.0
        elif cfg.regime_mode == "scaled":
            allow, size_mult = True, (1.0 if bull else cfg.bear_size_mult)
        else:  # block
            allow, size_mult = bull, 1.0

        ranked = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)]
        positive = [s for s in ranked if scores[s][0] > 0]
        held = list(positions.keys())
        open_slots = MAX_POSITIONS - len(held)

        new = []
        if allow and open_slots > 0:
            cand = positive if positive else ranked
            new = select_with_corr(cand, dd, MAX_POSITIONS, cfg.corr_cap, held)

        if len(held) + len(new) < MIN_POSITIONS and ranked:
            for s in ranked:
                if s not in held and s not in new:
                    new.append(s)
                if len(held) + len(new) >= MIN_POSITIONS:
                    break
            if not allow and cfg.regime_mode == "scaled":  # scaled mode: forced bear entries sized down
                size_mult = min(size_mult, cfg.bear_size_mult)

        if new:
            held_val = sum(p["shares"] * symbol_dfs[s][symbol_dfs[s]["date"] <= dd]["close"].iloc[-1]
                           for s, p in positions.items())
            deployable = (cash + held_val) * size_mult
            target = len(held) + len(new)
            w = {}
            for s in new:
                sl = symbol_dfs[s][symbol_dfs[s]["date"] <= dd]
                if cfg.sizing == "invvol":
                    a = atr_pct(sl)
                    w[s] = 1.0 / a if a and a > 0 else 0.0
                elif cfg.sizing == "momentum":
                    w[s] = momentum_score(sl) + 0.01
                else:
                    w[s] = 1.0
            wsum = sum(w.values()) or 1.0
            per = deployable / target if target > 0 else 0
            for s in new:
                df = symbol_dfs[s]
                eb = df[df["date"] > dd]
                if eb.empty:
                    continue
                b = eb.iloc[0]
                px = b["open"] * (1 + SLIPPAGE_PCT)
                alloc = per * (w[s] / (wsum / len(new))) if wsum > 0 else per
                alloc = min(alloc, cash / (1 + BUY_FRAC))
                if alloc <= 0:
                    continue
                sh = alloc / px
                turn = sh * px
                cash -= turn + COSTS.buy_cost(turn)
                ea = get_atr_at(s, dd)
                if pd.isna(ea) or ea <= 0:
                    ea = px * 0.02
                positions[s] = {"shares": sh, "entry_price": px, "peak_price": px, "entry_atr": ea}

        mtm = cash
        for s, p in positions.items():
            rc = symbol_dfs[s][symbol_dfs[s]["date"] <= dd]
            if not rc.empty:
                mtm += p["shares"] * rc["close"].iloc[-1]
        eq.append(mtm)
        eq_dates.append(dd)

    fd = pd.to_datetime(end)
    for s, p in list(positions.items()):
        fb = symbol_dfs[s][symbol_dfs[s]["date"] <= fd]
        if fb.empty:
            continue
        net, ret = sell(p, p["shares"], fb.iloc[-1]["close"] / (1 - SLIPPAGE_PCT))
        cash += net
        trades.append({"return_pct": ret, "reason": "end"})

    equity = pd.Series(eq, index=eq_dates)
    tot = (cash - CAPITAL) / CAPITAL
    wr = equity.pct_change().dropna()
    sharpe = (wr.mean() / wr.std()) * np.sqrt(52) if wr.std() > 0 else 0
    mdd = ((equity - equity.cummax()) / equity.cummax()).min() if len(equity) else 0
    seg = index_series[(index_series.index >= pd.to_datetime(start)) & (index_series.index <= fd)]
    bh = seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else 0.0
    return {"ret": tot, "sharpe": sharpe, "mdd": mdd, "bh": bh, "ntr": len(trades)}


def evaluate(cfg: Config):
    rs = [run_window(cfg, s, e) for s, e in WINDOWS]
    ret = np.array([r["ret"] for r in rs])
    bh = np.array([r["bh"] for r in rs])
    crash = bh < -0.05
    up = bh > 0.05
    return {
        "compounded": np.prod(1 + ret) - 1,
        "mean_q": ret.mean(),
        "sharpe": np.mean([r["sharpe"] for r in rs]),
        "worst_dd": min(r["mdd"] for r in rs),
        "neg_q": int((ret < 0).sum()),
        "crash_excess": (ret[crash] - bh[crash]).mean(),
        "crash_ret": ret[crash].mean(),
        "rally_excess": (ret[up] - bh[up]).mean(),
        "rally_ret": ret[up].mean(),
        "trades": sum(r["ntr"] for r in rs),
    }


CONFIGS = [
    Config("v5 DEFENSIVE (current)", "block", trail_mult=2.0, corr_cap=0.70, sizing="invvol"),
    Config("A: let winners run",     "block", trail_mult=3.0, corr_cap=0.70, sizing="invvol"),
    Config("B: +concentrate",        "block", trail_mult=3.0, corr_cap=0.90, sizing="invvol"),
    Config("C: +momentum sizing",    "block", trail_mult=3.0, corr_cap=0.90, sizing="momentum"),
    Config("D: regime-SCALED (0.5x)","scaled", bear_size_mult=0.5, trail_mult=3.0, corr_cap=0.90, sizing="momentum"),
    Config("E: regime OFF (max agg)","off",   trail_mult=3.5, corr_cap=0.95, sizing="momentum"),
]

print("=" * 122)
print("AGGRESSION SWEEP — 2015-2024, real data + real costs. NIFTY100 B&H: compounded +161%, "
      "crash_ret -11.2%, rally_ret +10.4%")
print("=" * 122)
print(f"{'config':<28}{'compound':>10}{'mean/q':>8}{'sharpe':>8}{'worstDD':>9}{'neg_q':>7}"
      f"{'CRASHret':>10}{'crash_exc':>10}{'RALLYret':>10}{'rally_exc':>10}{'trades':>8}")
for cfg in CONFIGS:
    m = evaluate(cfg)
    print(f"{cfg.name:<28}{m['compounded']:>9.0%}{m['mean_q']:>8.2%}{m['sharpe']:>8.2f}"
          f"{m['worst_dd']:>9.1%}{m['neg_q']:>5d}/40{m['crash_ret']:>10.2%}{m['crash_excess']:>10.2%}"
          f"{m['rally_ret']:>10.2%}{m['rally_excess']:>10.2%}{m['trades']:>8d}")
print("=" * 122)
print("Read across a row: aggression should LIFT 'RALLYret' (upside capture) at the cost of")
print("a worse 'CRASHret' / 'worstDD'. The best dial setting depends on your crash tolerance.")
