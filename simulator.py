"""
simulator.py — Live forward paper-trading simulator for the momentum strategy.

Starts a Rs 5,00,000 paper book on first run by buying the top-15 idiosyncratic-
momentum picks equal-weight at *live* prices, then on every subsequent run marks
to market against live prices, applies ATR trailing stops (from the v5 research:
1.5x initial, graduate to 2.0x trailing once +1.0x ATR in profit), and redeploys
stopped-out cash into the next-ranked momentum name. State + a timestamped NAV /
trade log persist in dashboard/sim_state.json, so it can run intraday over a
month (call it repeatedly — background loop, cron, or the sim.yml workflow) and
build a real forward record.

Idempotent-safe: safe to call many times a day; each call = one mark-to-market
step. Real Indian costs + slippage on every fill.

Run:  python simulator.py            # one step (init on first call)
      python simulator.py --status   # print current state, no trading
"""
import os, json, sys, argparse, warnings, datetime as dt, importlib.util
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
def _imp(n, p):
    s = importlib.util.spec_from_file_location(n, os.path.join(_HERE, p))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
_uni = _imp("nse_universe", "backtests/nse_universe.py")
A = _imp("analytics", "dashboard/analytics.py")
COSTS_MOD = _imp("costs", "costs.py")
TECH = _imp("technical", "technical.py")
UNIVERSE, INDEX = _uni.UNIVERSE, _uni.REGIME_INDEX

CAPITAL, TOP_N = 500_000.0, 15
MIN_PRICE, MIN_ADV = 50.0, 5.0
ATR_INIT, ATR_LOCK, ATR_TRAIL = 1.5, 1.0, 2.0
SLIP = 0.0015
COSTS = COSTS_MOD.IndianEquityCosts()
STATE = os.path.join(_HERE, "docs", "sim_state.json")
HISTCACHE = os.path.join(_HERE, "dashboard", "sim_histcache.pkl")


def market_open():
    ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    return ist.weekday() < 5 and dt.time(9, 15) <= ist.time() <= dt.time(15, 30), ist


def load_history():
    """~2y OHLC for universe + index. Cached to pickle, refreshed daily."""
    if os.path.exists(HISTCACHE):
        m = dt.date.fromtimestamp(os.path.getmtime(HISTCACHE))
        if m == dt.date.today():
            return pd.read_pickle(HISTCACHE)
    data = yf.download([f"{s}.NS" for s in UNIVERSE] + [INDEX], period="2y",
                       auto_adjust=True, progress=False, group_by="ticker")
    data.to_pickle(HISTCACHE)
    return data


def live_prices():
    data = yf.download([f"{s}.NS" for s in UNIVERSE], period="1d", interval="1m",
                       auto_adjust=True, progress=False, group_by="ticker")
    px = {}
    for s in UNIVERSE:
        try:
            c = data[f"{s}.NS"]["Close"].dropna()
            if len(c):
                px[s] = float(c.iloc[-1])
        except Exception:
            pass
    return px


def rank_and_atr(data):
    close = pd.DataFrame({s: data[f"{s}.NS"]["Close"] for s in UNIVERSE
                          if f"{s}.NS" in data.columns.get_level_values(0)}).dropna(how="all")
    vol = pd.DataFrame({s: data[f"{s}.NS"]["Volume"] for s in close.columns}).reindex(close.index)
    idx = data[INDEX]["Close"].reindex(close.index).ffill()
    idio = A.idio_mom_scores(close, idx)
    ctx = A.context_metrics(close, vol)
    px_last = close.ffill().iloc[-1]
    elig = [s for s in idio.index if px_last.get(s, 0) >= MIN_PRICE
            and (ctx.loc[s, "adv_cr"] if s in ctx.index else 0) >= MIN_ADV]
    ranked = idio[elig].sort_values(ascending=False).index.tolist()
    atr = {}
    for s in ranked:
        d = data[f"{s}.NS"].dropna()
        a = TECH.average_true_range(d["High"], d["low" if "low" in d else "Low"], d["Close"], 14)
        atr[s] = float(a.iloc[-1])
    return ranked, atr, float(idx.iloc[-1])


def buy(cash, sym, price, atr, today):
    px = price * (1 + SLIP)
    turn = min(cash / (1 + COSTS.buy_cost(1.0)), cash) * 0.999
    shares = turn / px
    spent = shares * px + COSTS.buy_cost(shares * px)
    return {"shares": shares, "entry_price": px, "entry_atr": atr, "peak": px, "entry_date": today}, spent


def sell(pos, price):
    px = price * (1 - SLIP)
    turn = pos["shares"] * px
    return turn - COSTS.sell_cost(turn)


def step(status_only=False):
    data = load_history()
    ranked, atr, idx_level = rank_and_atr(data)
    px = live_prices()
    openq, ist = market_open()
    now = ist.strftime("%Y-%m-%d %H:%M IST")
    today = ist.date().isoformat()

    st = json.load(open(STATE)) if os.path.exists(STATE) else None

    if st is None:                       # ---- initialize the paper book ----
        cash = CAPITAL
        holdings = {}
        per = CAPITAL / TOP_N
        picks = ranked[:TOP_N]
        for s in picks:
            p = px.get(s)
            if not p:
                continue
            pos, spent = buy(per, s, p, atr.get(s, p * 0.02), today)
            holdings[s] = pos
            cash -= spent
        st = {"start": today, "idx_start": idx_level, "cash": cash, "holdings": holdings,
              "history": [], "trades": [{"date": today, "action": "BUY", "sym": s,
              "price": round(px.get(s, 0), 2)} for s in holdings]}
        print(f"[{now}] INITIALIZED paper book: bought {len(holdings)} names, cash left ₹{cash:,.0f}")

    elif not status_only:                # ---- advance one step ----
        held = st["holdings"]
        for s in list(held.keys()):
            p = px.get(s)
            if not p:
                continue
            pos = held[s]
            pos["peak"] = max(pos["peak"], p)
            ea = pos["entry_atr"]
            if pos["peak"] >= pos["entry_price"] + ATR_LOCK * ea:
                stop, why = pos["peak"] - ATR_TRAIL * ea, "trail_stop"
            else:
                stop, why = pos["entry_price"] - ATR_INIT * ea, "init_stop"
            if p <= stop:
                st["cash"] += sell(pos, p)
                ret = p / pos["entry_price"] - 1
                st["trades"].append({"date": today, "action": "SELL", "sym": s,
                                     "price": round(p, 2), "reason": why, "ret_pct": round(ret * 100, 2)})
                del held[s]
                print(f"[{now}] STOP {s} @ ₹{p:,.2f} ({why}, {ret:+.1%})")
        # redeploy freed cash into next-ranked names not held
        for s in ranked:
            if len(held) >= TOP_N:
                break
            if s in held:
                continue
            p = px.get(s)
            if not p or st["cash"] < CAPITAL / (TOP_N * 3):
                continue
            deploy = st["cash"] if len(held) == TOP_N - 1 else st["cash"] / (TOP_N - len(held))
            pos, spent = buy(deploy, s, p, atr.get(s, p * 0.02), today)
            held[s] = pos
            st["cash"] -= spent
            st["trades"].append({"date": today, "action": "BUY", "sym": s, "price": round(p, 2)})
            print(f"[{now}] REDEPLOY -> {s} @ ₹{p:,.2f}")

    # mark to market
    nav = st["cash"] + sum(p["shares"] * px.get(s, p["entry_price"]) for s, p in st["holdings"].items())
    idx_ret = (idx_level / st["idx_start"] - 1) * 100
    ret = (nav / CAPITAL - 1) * 100
    st["history"].append({"ts": now, "nav": round(nav, 2), "ret_pct": round(ret, 2),
                          "idx_ret_pct": round(idx_ret, 2), "positions": len(st["holdings"])})
    json.dump(st, open(STATE, "w"), indent=2)

    print(f"[{now}] {'OPEN' if openq else 'CLOSED'} | NAV ₹{nav:,.0f} ({ret:+.2f}%) | "
          f"NIFTY100 {idx_ret:+.2f}% | {len(st['holdings'])} positions | day {len(set(h['ts'][:10] for h in st['history']))}"
          f" of paper run (since {st['start']})")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    step(status_only=a.status)
