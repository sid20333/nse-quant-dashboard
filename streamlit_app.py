"""
streamlit_app.py — Live NSE momentum dashboard (Streamlit Community Cloud).

Elite build: live server-side prices (auto-refresh), IDIOSYNCRATIC momentum (the
best signal from our research), per-name context (volatility, 52w-high distance,
₹-crore liquidity), interactive Plotly charts, a backtested + live-forward track
record, and a "what changed" diff. Point-in-time/de-survivorship data is the one
thing not here — it needs a paid vendor (see backtests/ research + DEPLOY.md).

Layering keeps auto-refresh cheap: history + heavy analytics cached hourly;
only the latest price is refetched each tick. Fundamentals & the live track
record come from committed files (dashboard/*.json, docs/track_record.json).
"""
import os, json, importlib.util, datetime as dt, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

_HERE = os.path.dirname(os.path.abspath(__file__))
def _imp(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_uni = _imp("nse_universe", "backtests/nse_universe.py")
A = _imp("analytics", "dashboard/analytics.py")
UNIVERSE = _uni.UNIVERSE
INDEX = _uni.REGIME_INDEX

TOP_PICKS, MIN_PRICE, MIN_ADV, FUND_SANE, REFRESH_S = 15, 50.0, 5.0, 100.0, 15
TICKERS = [f"{s}.NS" for s in UNIVERSE]
st.set_page_config(page_title="NSE Momentum — Live", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Loading history & analytics…")
def load_core():
    data = yf.download(TICKERS + [INDEX], period="5y", auto_adjust=True, progress=False, group_by="ticker")
    close = pd.DataFrame({s: data[f"{s}.NS"]["Close"] for s in UNIVERSE
                          if f"{s}.NS" in data.columns.get_level_values(0)}).dropna(how="all")
    vol = pd.DataFrame({s: data[f"{s}.NS"]["Volume"] for s in close.columns}).reindex(close.index)
    idx = data[INDEX]["Close"].reindex(close.index).ffill()
    keep = [s for s in close.columns if close[s].dropna().shape[0] >= 220]
    close, vol = close[keep], vol[keep]

    idio = A.idio_mom_scores(close, idx)
    ctx = A.context_metrics(close, vol)
    base = {}
    for s in keep:
        c = close[s].dropna()
        if len(c) < 220:
            continue
        base[s] = {"prev_close": float(c.iloc[-1]),
                   "sma20": float(c.rolling(20).mean().iloc[-1]),
                   "sma50": float(c.rolling(50).mean().iloc[-1]),
                   "sma200": float(c.rolling(200).mean().iloc[-1]),
                   "idio": float(idio.get(s, np.nan)),
                   "vol_ann": ctx.loc[s, "vol_ann"] if s in ctx.index else None,
                   "dist_52w": ctx.loc[s, "dist_52w"] if s in ctx.index else None,
                   "adv_cr": ctx.loc[s, "adv_cr"] if s in ctx.index else None}
    curve = A.momentum_equity_curve(close, idx, top_n=TOP_PICKS, min_price=MIN_PRICE)
    return base, close, idx, curve, data


@st.cache_data(ttl=86400, show_spinner=False)
def load_fundamentals():
    out = {s: {"roe": None, "roce": None, "sector": None} for s in UNIVERSE}
    try:
        with open(os.path.join(_HERE, "dashboard", "fundamentals_cache.json")) as f:
            raw = json.load(f)
        for s in UNIVERSE:
            e = raw.get(f"{s}.NS")
            if e:
                out[s] = {"roe": e.get("roe"), "roce": e.get("roce"), "sector": e.get("sector")}
    except Exception:
        pass
    return out


@st.cache_data(ttl=10, show_spinner=False)
def live_prices():
    data = yf.download(TICKERS, period="2d", interval="1d", auto_adjust=True, progress=False, group_by="ticker")
    px = {}
    for s in UNIVERSE:
        try:
            c = data[f"{s}.NS"]["Close"].dropna()
            if len(c):
                px[s] = float(c.iloc[-1])
        except Exception:
            pass
    return px


def sig(p, s):
    if s is None or np.isnan(s):
        return "n/a"
    return "Buy" if p > s * 1.005 else "Sell" if p < s * 0.995 else "Hold"


def market_state():
    ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    return (ist.weekday() < 5 and dt.time(9, 15) <= ist.time() <= dt.time(15, 30)), ist


# ---- assemble live table ----------------------------------------------------
if st_autorefresh:
    st_autorefresh(interval=REFRESH_S * 1000, key="tick")

base, close_df, idx_series, curve, raw_data = load_core()
funds = load_fundamentals()
px = live_prices()

rows = []
for s, b in base.items():
    price = px.get(s, b["prev_close"])
    f = funds.get(s, {})
    rows.append({"symbol": s, "price": price, "chg1d": (price / b["prev_close"] - 1) * 100,
                 "s20": sig(price, b["sma20"]), "s50": sig(price, b["sma50"]), "s200": sig(price, b["sma200"]),
                 "idio": b["idio"], "vol_ann": b["vol_ann"], "dist_52w": b["dist_52w"], "adv_cr": b["adv_cr"],
                 "roe": f.get("roe"), "roce": f.get("roce"), "sector": f.get("sector")})
df = pd.DataFrame(rows)
df = df[df["idio"].notna()].copy()

# eligible = liquid enough (price + average daily traded value)
elig = df[(df["price"] >= MIN_PRICE) & (df["adv_cr"].fillna(0) >= MIN_ADV)].copy()
elig = elig.sort_values("idio", ascending=False).reset_index(drop=True)
elig["pos"] = np.arange(1, len(elig) + 1)
df = df.merge(elig[["symbol", "pos"]], on="symbol", how="left").sort_values("pos", na_position="last")
picks = list(elig.head(TOP_PICKS)["symbol"])
pickset = set(picks)

def overall(r):
    if r["s200"] == "Sell":
        return "Sell"
    a50, a200 = r["s50"] == "Buy", r["s200"] == "Buy"
    if r["symbol"] in pickset and a50 and a200:
        return "Strong Buy"
    if a50 and a200:
        return "Buy"
    if r["symbol"] in pickset and a200:
        return "Buy"
    return "Hold"
df["overall"] = df.apply(overall, axis=1)

# ---- header -----------------------------------------------------------------
openq, ist = market_state()
st.title("📈 NSE Momentum — Live")
st.caption(f"{'🟢 LIVE — market open' if openq else '🔴 market closed — last traded prices'} · "
           f"{ist.strftime('%Y-%m-%d %H:%M:%S IST')} · auto-refresh {REFRESH_S}s · idiosyncratic momentum · "
           f"prices via Yahoo (may lag ~15m)")
st.info("**Not investment advice.** 'Picks' = top idiosyncratic-momentum names (best signal from our research: "
        "~+5%/yr alpha vs equal-weight, but lumpy & crash-prone — it lost over the last year). Ranked among liquid "
        f"(≥₹{MIN_PRICE:.0f}, ≥₹{MIN_ADV:.0f}cr/day) names only.", icon="⚠️")

tab_sig, tab_perf, tab_chart, tab_diff, tab_sim = st.tabs(
    ["📊 Signals", "📈 Performance", "🔎 Charts", "🔔 What changed", "🧪 Simulator"])

# ---- TAB: signals -----------------------------------------------------------
with tab_sig:
    st.subheader(f"Top {TOP_PICKS} momentum picks")
    pk = elig.head(TOP_PICKS)
    for i in range(0, len(pk), 5):
        for col, (_, r) in zip(st.columns(5), pk.iloc[i:i + 5].iterrows()):
            col.metric(f"#{int(r['pos'])} {r['symbol']}", f"₹{r['price']:,.2f}", f"{r['chg1d']:+.2f}%")
    if not pk.empty:
        secs = pd.Series([funds.get(s, {}).get("sector") or "—" for s in pk["symbol"]]).value_counts()
        st.caption("Sector mix of picks: " + " · ".join(f"{k} ({v})" for k, v in secs.items()))

    def ffund(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else ("n/m" if abs(v) > FUND_SANE else f"{v:.1f}%")
    def fnum(v, suf=""):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}{suf}"
    show = pd.DataFrame({
        "#": df["pos"].apply(lambda v: "" if pd.isna(v) else str(int(v))),
        "Symbol": df["symbol"], "Sector": df["sector"].apply(lambda v: v or "—"),
        "Price": df["price"].apply(lambda v: f"₹{v:,.2f}"), "1D %": df["chg1d"].apply(lambda v: f"{v:+.2f}%"),
        "20 SMA": df["s20"], "50 SMA": df["s50"], "200 SMA": df["s200"], "Signal": df["overall"],
        "52w↓": df["dist_52w"].apply(lambda v: fnum(v, "%")), "Vol": df["vol_ann"].apply(lambda v: fnum(v, "%")),
        "ADV₹cr": df["adv_cr"].apply(lambda v: fnum(v)), "ROE": df["roe"].apply(ffund), "ROCE": df["roce"].apply(ffund),
    })
    C = {"Strong Buy": "#132e1a", "Buy": "#0f2417", "Hold": "#26210c", "Sell": "#2a1215"}
    T = {"Buy": "#3fb950", "Strong Buy": "#3fb950", "Hold": "#e3b341", "Sell": "#f85149"}
    def ssig(v): return f"background-color:{C.get(v,'')};color:{T.get(v,'#8b949e')}"
    def schg(v): return f"color:{'#3fb950' if str(v).startswith('+') else '#f85149'}"
    mname = "map" if hasattr(show.style, "map") else "applymap"
    styled = show.style
    styled = getattr(styled, mname)(ssig, subset=["20 SMA", "50 SMA", "200 SMA", "Signal"])
    styled = getattr(styled, mname)(schg, subset=["1D %"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=560)
    st.caption("Signal: Strong Buy (a top pick above its 50 & 200 SMA), Buy (above 50 & 200, or a top pick above its "
               "200 but below its 50), Sell (below 200), else Hold. 52w↓=below 52-week high, Vol=60d annualised, "
               "ADV=avg daily traded value. ROE=TTM; ROCE=EBIT÷(assets−curr.liab.), n/m when distorted.")

# ---- TAB: performance -------------------------------------------------------
with tab_perf:
    st.subheader("Backtested equity curve (5y) — Strategy vs Equal-weight vs Index")
    if not curve.empty:
        fig = go.Figure()
        for col, color in [("Strategy", "#3fb950"), ("EqualWeight", "#8b949e"), ("Index", "#58a6ff")]:
            if col in curve:
                fig.add_trace(go.Scatter(x=curve.index, y=curve[col], name=col, line=dict(color=color)))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="Growth of ₹1", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        s = curve["Strategy"].pct_change().dropna()
        cagr = curve["Strategy"].iloc[-1] ** (12 / len(s)) - 1
        dd = ((curve["Strategy"] - curve["Strategy"].cummax()) / curve["Strategy"].cummax()).min()
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy total (5y)", f"{(curve['Strategy'].iloc[-1]-1)*100:.0f}%")
        c2.metric("vs Index total", f"{(curve['Index'].iloc[-1]-1)*100:.0f}%")
        c3.metric("Worst drawdown", f"{dd*100:.0f}%")
    st.caption("Backtest is survivorship-biased (today's constituents) and gross of most costs — it OVERSTATES the "
               "edge. The honest number is ~+5%/yr alpha vs equal-weight; see backtests/ for the full, sober analysis.")

    st.subheader("Live-forward paper track record")
    tr_path = os.path.join(_HERE, "docs", "track_record.json")
    hist = []
    if os.path.exists(tr_path):
        try:
            hist = json.load(open(tr_path)).get("history", [])
        except Exception:
            pass
    if len(hist) >= 2:
        h = pd.DataFrame(hist); h["date"] = pd.to_datetime(h["date"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=h["date"], y=h["nav"], name="Strategy (paper)", line=dict(color="#3fb950")))
        if "idx" in h and h["idx"].notna().any():
            fig2.add_trace(go.Scatter(x=h["date"], y=h["idx"], name="Index", line=dict(color="#58a6ff")))
        fig2.update_layout(template="plotly_dark", height=320, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="NAV (start 100)", legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("The live track record accumulates once the daily GitHub Actions job starts logging picks "
                "(needs the cron enabled — see DEPLOY.md). It begins at 100 and grows out-of-sample from day one.")

# ---- TAB: charts ------------------------------------------------------------
with tab_chart:
    sym = st.selectbox("Symbol", df["symbol"].tolist())
    tk = f"{sym}.NS"
    try:
        o = raw_data[tk].dropna().tail(400)
        fig = go.Figure(go.Candlestick(x=o.index, open=o["Open"], high=o["High"], low=o["Low"], close=o["Close"],
                                       name=sym, increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
        for w, c in [(20, "#e3b341"), (50, "#58a6ff"), (200, "#f778ba")]:
            fig.add_trace(go.Scatter(x=o.index, y=o["Close"].rolling(w).mean(), name=f"SMA{w}", line=dict(color=c, width=1)))
        fig.update_layout(template="plotly_dark", height=480, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No chart data for {sym}.")

# ---- TAB: what changed ------------------------------------------------------
with tab_diff:
    st.subheader("Changes since the last committed snapshot")
    prev = None
    dpath = os.path.join(_HERE, "docs", "data.json")
    if os.path.exists(dpath):
        try:
            prev = json.load(open(dpath))
        except Exception:
            pass
    if prev:
        prev_picks = set(prev.get("picks", []))
        prev_sig = {r["symbol"]: r.get("overall") for r in prev.get("rows", [])}
        added = [s for s in picks if s not in prev_picks]
        dropped = [s for s in prev_picks if s not in pickset]
        cur_sig = dict(zip(df["symbol"], df["overall"]))
        upg = [s for s in cur_sig if prev_sig.get(s) and prev_sig[s] != cur_sig[s]]
        c1, c2 = st.columns(2)
        c1.markdown("**➕ New to picks**\n\n" + ("\n".join(f"- {s}" for s in added) or "_none_"))
        c2.markdown("**➖ Dropped from picks**\n\n" + ("\n".join(f"- {s}" for s in dropped) or "_none_"))
        st.markdown("**🔀 Signal changes**\n\n" + ("\n".join(
            f"- {s}: {prev_sig[s]} → {cur_sig[s]}" for s in upg) or "_none_"))
        st.caption(f"Compared against docs/data.json (snapshot: {prev.get('updated','?')}). "
                   "Refresh that via the daily job / `python dashboard/generate.py`.")
    else:
        st.info("No previous snapshot yet — run `python dashboard/generate.py` (or the daily job) to create one.")

# ---- TAB: simulator ---------------------------------------------------------
with tab_sim:
    st.subheader("Live paper-trading simulator")
    spath = os.path.join(_HERE, "docs", "sim_state.json")
    if os.path.exists(spath):
        try:
            sim = json.load(open(spath))
        except Exception:
            sim = None
    else:
        sim = None
    if sim:
        hist = pd.DataFrame(sim.get("history", []))
        if len(hist):
            last = hist.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Paper NAV", f"₹{last['nav']:,.0f}", f"{last['ret_pct']:+.2f}%")
            c2.metric("vs NIFTY100", f"{last['idx_ret_pct']:+.2f}%")
            c3.metric("Open positions", int(last["positions"]))
            hist["t"] = pd.to_datetime(hist["ts"].str[:16], errors="coerce")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["t"], y=hist["ret_pct"], name="Paper %", line=dict(color="#3fb950")))
            fig.add_trace(go.Scatter(x=hist["t"], y=hist["idx_ret_pct"], name="NIFTY100 %", line=dict(color="#58a6ff")))
            fig.update_layout(template="plotly_dark", height=340, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="% since start", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Started {sim.get('start')} with ₹5,00,000 · top-15 idiosyncratic momentum · ATR trailing stops · "
                   "redeploys stopped-out cash · advanced every ~30 min by GitHub Actions (simulate.yml).")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Holdings**")
            h = sim.get("holdings", {})
            if h:
                st.dataframe(pd.DataFrame([{"Symbol": s, "Shares": round(v["shares"], 1),
                                            "Entry ₹": round(v["entry_price"], 2)} for s, v in h.items()]),
                             hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Recent trades**")
            tr = sim.get("trades", [])[-15:][::-1]
            if tr:
                st.dataframe(pd.DataFrame(tr), hide_index=True, use_container_width=True)
    else:
        st.info("Simulator hasn't started yet — it initialises on the first `python simulator.py` run or Actions step.")
