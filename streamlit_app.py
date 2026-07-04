"""
streamlit_app.py — Live NSE momentum dashboard (Streamlit Community Cloud).

Why Streamlit instead of the static GitHub Pages version: live ticks need
server-side quote fetching. A static page is CORS-blocked from Yahoo; Streamlit
runs yfinance server-side, so it just works, and st_autorefresh re-pulls prices
every few seconds while the page is open.

Layering (so auto-refresh is cheap):
  - HISTORY + fundamentals are cached (SMAs, momentum ranks, ROE/ROCE change
    slowly) — refetched at most hourly / daily.
  - Only the LATEST PRICE is pulled fresh each refresh, then SMA signals, 1-day
    change and the overall signal are recomputed live against it.

Same logic + artifact fixes as dashboard/generate.py: momentum ranked only among
>= Rs50 names (no penny-stock noise), distorted ROE/ROCE shown as "n/m".

Run locally:  streamlit run streamlit_app.py
Deploy: push to GitHub, then share.streamlit.io -> New app -> this file.
"""
import os, importlib.util, datetime as dt, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ---- universe (load by file path; robust to repo layout) --------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_uni_path = os.path.join(_HERE, "backtests", "nse_universe.py")
_spec = importlib.util.spec_from_file_location("nse_universe", _uni_path)
_uni = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_uni)
UNIVERSE = _uni.UNIVERSE

TOP_PICKS, MIN_PRICE, FUND_SANE, REFRESH_S = 15, 50.0, 100.0, 15
TICKERS = [f"{s}.NS" for s in UNIVERSE]

st.set_page_config(page_title="NSE Momentum — Live", page_icon="📈", layout="wide")


# ---- slow layer: history -> SMAs + momentum (cached 1h) ---------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_history():
    data = yf.download(TICKERS, period="2y", auto_adjust=True, progress=False, group_by="ticker")
    base = {}
    for sym in UNIVERSE:
        tk = f"{sym}.NS"
        try:
            c = data[tk]["Close"].dropna()
        except Exception:
            continue
        if len(c) < 220:
            continue
        base[sym] = {
            "prev_close": float(c.iloc[-1]),
            "sma20": float(c.rolling(20).mean().iloc[-1]),
            "sma50": float(c.rolling(50).mean().iloc[-1]),
            "sma200": float(c.rolling(200).mean().iloc[-1]),
            "m12": float(c.iloc[-21] / c.iloc[-252] - 1) if len(c) >= 252 else np.nan,
            "m6": float(c.iloc[-21] / c.iloc[-126] - 1) if len(c) >= 126 else np.nan,
        }
    return base


@st.cache_data(ttl=86400, show_spinner=False)
def load_fundamentals():
    """Read the committed fundamentals cache (dashboard/fundamentals_cache.json,
    produced by dashboard/generate.py). Reading a file avoids 104 slow .info /
    .financials calls on Streamlit's shared IP, which Yahoo readily rate-limits.
    Refresh the file by running `python dashboard/generate.py` and committing."""
    import json
    path = os.path.join(_HERE, "dashboard", "fundamentals_cache.json")
    out = {sym: {"roe": None, "roce": None} for sym in UNIVERSE}
    try:
        with open(path) as f:
            raw = json.load(f)
        for sym in UNIVERSE:
            e = raw.get(f"{sym}.NS")
            if e:
                out[sym] = {"roe": e.get("roe"), "roce": e.get("roce")}
    except Exception:
        pass
    return out


# ---- live layer: latest price only (cached 10s so reruns share) -------------
@st.cache_data(ttl=10, show_spinner=False)
def live_prices():
    data = yf.download(TICKERS, period="2d", interval="1d", auto_adjust=True,
                       progress=False, group_by="ticker")
    px = {}
    for sym in UNIVERSE:
        try:
            c = data[f"{sym}.NS"]["Close"].dropna()
            if len(c):
                px[sym] = float(c.iloc[-1])
        except Exception:
            pass
    return px


def sig(price, sma):
    if sma is None or np.isnan(sma):
        return "n/a"
    return "Buy" if price > sma * 1.005 else "Sell" if price < sma * 0.995 else "Hold"


def market_state():
    ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    openq = (ist.weekday() < 5) and (dt.time(9, 15) <= ist.time() <= dt.time(15, 30))
    return openq, ist


# ---- build ------------------------------------------------------------------
if st_autorefresh:
    st_autorefresh(interval=REFRESH_S * 1000, key="tick")

base = load_history()
funds = load_fundamentals()
px = live_prices()

rows = []
for sym, b in base.items():
    price = px.get(sym, b["prev_close"])
    chg = (price / b["prev_close"] - 1) * 100
    f = funds.get(sym, {})
    rows.append({"symbol": sym, "price": price, "chg1d": chg,
                 "s20": sig(price, b["sma20"]), "s50": sig(price, b["sma50"]),
                 "s200": sig(price, b["sma200"]), "m12": b["m12"], "m6": b["m6"],
                 "roe": f.get("roe"), "roce": f.get("roce")})
df = pd.DataFrame(rows)
df = df[df["m12"].notna() & df["m6"].notna()].copy()

elig = df[df["price"] >= MIN_PRICE].copy()
elig["mom_rank"] = (elig["m12"].rank() + elig["m6"].rank()) / 2
elig = elig.sort_values("mom_rank", ascending=False).reset_index(drop=True)
elig["pos"] = np.arange(1, len(elig) + 1)
df = df.merge(elig[["symbol", "pos"]], on="symbol", how="left").sort_values("pos", na_position="last")
picks = set(elig.head(TOP_PICKS)["symbol"])

def overall(r):
    if r["symbol"] in picks and r["s200"] == "Buy":
        return "Strong Buy"
    if r["s200"] == "Buy" and r["s50"] == "Buy":
        return "Buy"
    if r["s200"] == "Sell":
        return "Sell"
    return "Hold"
df["overall"] = df.apply(overall, axis=1)

# ---- render -----------------------------------------------------------------
openq, ist = market_state()
badge = "🟢 LIVE — market open" if openq else "🔴 market closed — last traded prices"
st.title("📈 NSE Momentum — Live")
st.caption(f"{badge} · {ist.strftime('%Y-%m-%d %H:%M:%S IST')} · auto-refresh {REFRESH_S}s · "
           f"prices via Yahoo (may be delayed ~15m)")
st.info("**Not investment advice.** 'Picks' = top 12-month momentum names — the only signal that survived "
        "validation (~+5%/yr vs equal-weight, but lumpy & crash-prone). Momentum ranked among ≥₹50 names only.",
        icon="⚠️")

st.subheader(f"Top {TOP_PICKS} momentum picks")
picks_df = elig.head(TOP_PICKS)
ncol = 5
for i in range(0, len(picks_df), ncol):
    for col, (_, r) in zip(st.columns(ncol), picks_df.iloc[i:i + ncol].iterrows()):
        col.metric(f"#{int(r['pos'])} {r['symbol']}", f"₹{r['price']:,.2f}", f"{r['chg1d']:+.2f}%")

st.subheader("Full universe · signals & fundamentals")

def ffund(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return "n/m" if abs(v) > FUND_SANE else f"{v:.1f}%"

show = pd.DataFrame({
    "#": df["pos"].apply(lambda v: "" if pd.isna(v) else str(int(v))),
    "Symbol": df["symbol"], "Price": df["price"].apply(lambda v: f"₹{v:,.2f}"),
    "1D %": df["chg1d"].apply(lambda v: f"{v:+.2f}%"),
    "20 SMA": df["s20"], "50 SMA": df["s50"], "200 SMA": df["s200"],
    "Signal": df["overall"], "ROE": df["roe"].apply(ffund), "ROCE": df["roce"].apply(ffund),
})
COLORS = {"Strong Buy": "#132e1a", "Buy": "#0f2417", "Hold": "#26210c", "Sell": "#2a1215"}
TXT = {"Buy": "#3fb950", "Strong Buy": "#3fb950", "Hold": "#e3b341", "Sell": "#f85149"}
def style_sig(v):
    return f"background-color:{COLORS.get(v,'')};color:{TXT.get(v,'#8b949e')}"
def style_chg(v):
    return f"color:{'#3fb950' if str(v).startswith('+') else '#f85149'}"
# pandas >=2.1 renamed Styler.applymap -> Styler.map (applymap removed in 3.0)
_mname = "map" if hasattr(show.style, "map") else "applymap"
styled = show.style
styled = getattr(styled, _mname)(style_sig, subset=["20 SMA", "50 SMA", "200 SMA", "Signal"])
styled = getattr(styled, _mname)(style_chg, subset=["1D %"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=560)

st.caption("SMA tag: price >0.5% above=Buy, >0.5% below=Sell, else Hold. Overall: Strong Buy (a top pick above its "
           "200 SMA), Buy (above 50 & 200), Sell (below 200), else Hold. ROE=TTM; ROCE=EBIT÷(assets−current liab.), "
           "annualised last quarter (n/m when the balance sheet distorts it, e.g. banks/negative equity). "
           "Data may be delayed or wrong — verify before acting.")
