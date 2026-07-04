"""
generate.py — Build the static dashboard (docs/index.html + docs/data.json).

Data source: yfinance (EOD). For each name in the NSE large/mid-cap universe it
computes latest price, 1-day change, 20/50/200-day SMAs with a Buy/Hold/Sell tag
each, the validated 12-1 & 6-1 momentum score, an overall signal, and last-
quarter ROE / ROCE from the financial statements.

"Recommendations" = the top-ranked momentum names (the only edge that survived
validation — see the research in ../backtests). This is a monitoring dashboard,
NOT trading advice; momentum is lumpy, crash-prone, and ~+5%/yr alpha at best.

Run:  python dashboard/generate.py [--limit N]
Designed to be run by GitHub Actions on a schedule; commits docs/ for Pages.
"""
import sys, os, json, argparse, warnings, datetime as dt, importlib.util
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# Load the curated universe by file path (sibling backtests/ dir), so this works
# regardless of how the repo is named / packaged on GitHub.
_HERE = os.path.dirname(os.path.abspath(__file__))
_uni_path = os.path.join(os.path.dirname(_HERE), "backtests", "nse_universe.py")
_spec = importlib.util.spec_from_file_location("nse_universe", _uni_path)
_uni = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_uni)
UNIVERSE = _uni.UNIVERSE

TOP_PICKS = 15
MIN_PRICE = 50.0          # exclude sub-Rs50 penny/distressed names from momentum picks
                          # (their huge % swings are noise, e.g. IDEA at Rs14)
FUND_SANE = 100.0         # |ROE|/|ROCE| beyond this = balance-sheet artifact -> "n/m"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
FUND_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fundamentals_cache.json")


def sma_signal(price, sma):
    if sma is None or np.isnan(sma):
        return "n/a"
    if price > sma * 1.005:
        return "Buy"
    if price < sma * 0.995:
        return "Sell"
    return "Hold"


def load_fund_cache():
    if os.path.exists(FUND_CACHE):
        try:
            with open(FUND_CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def fundamentals(symbol, cache):
    """Last-quarter ROE (TTM from .info) + ROCE (EBIT / capital employed).
    Cached for 7 days so the daily price refresh doesn't re-hit slow endpoints."""
    key = f"{symbol}.NS"
    entry = cache.get(key)
    if entry and (dt.date.today().toordinal() - entry.get("asof_ord", 0)) < 7:
        return entry
    out = {"roe": None, "roce": None, "period": None, "asof_ord": dt.date.today().toordinal()}
    try:
        t = yf.Ticker(key)
        info = t.info
        roe = info.get("returnOnEquity")
        if roe is not None:
            out["roe"] = round(roe * 100, 1)
        bs = t.quarterly_balance_sheet
        is_ = t.quarterly_income_stmt
        if bs is not None and not bs.empty and is_ is not None and not is_.empty:
            bcol, icol = bs.columns[0], is_.columns[0]
            def g(df, col, *keys):
                for k in keys:
                    if k in df.index and pd.notna(df.loc[k, col]):
                        return float(df.loc[k, col])
                return None
            ta = g(bs, bcol, "Total Assets")
            cl = g(bs, bcol, "Current Liabilities", "Total Current Liabilities")
            ebit = g(is_, icol, "EBIT", "Operating Income", "OperatingIncome")
            if ta and cl and ebit:
                ce = ta - cl
                if ce > 0:
                    out["roce"] = round((ebit * 4) / ce * 100, 1)  # annualized last quarter
                    out["period"] = str(pd.Timestamp(icol).date())
    except Exception:
        pass
    cache[key] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fundamentals", action="store_true")
    args = ap.parse_args()

    universe = UNIVERSE[: args.limit] if args.limit else UNIVERSE
    tickers = [f"{s}.NS" for s in universe]

    print(f"Fetching prices for {len(tickers)} names...")
    data = yf.download(tickers, period="2y", auto_adjust=True, progress=False, group_by="ticker")

    cache = load_fund_cache()
    rows = []
    for sym in universe:
        tk = f"{sym}.NS"
        try:
            df = data[tk].dropna() if tk in data.columns.get_level_values(0) else None
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue
        close = df["Close"]
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        chg1d = (price / prev - 1) * 100
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        # validated momentum: avg rank of 12-1 and 6-1 (computed later cross-sectionally)
        m12 = float(close.iloc[-21] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan
        m6 = float(close.iloc[-21] / close.iloc[-126] - 1) if len(close) >= 126 else np.nan
        rows.append({"symbol": sym, "price": round(price, 2), "chg1d": round(chg1d, 2),
                     "sma20": round(sma20, 2), "sma50": round(sma50, 2), "sma200": round(sma200, 2),
                     "s20": sma_signal(price, sma20), "s50": sma_signal(price, sma50),
                     "s200": sma_signal(price, sma200), "m12": m12, "m6": m6})

    dfr = pd.DataFrame(rows)
    dfr = dfr[dfr["m12"].notna() & dfr["m6"].notna()].copy()
    # Rank momentum only among liquid (>= MIN_PRICE) names, so penny-stock noise
    # can't top the picks. Sub-floor names still appear in the table (no rank).
    elig = dfr[dfr["price"] >= MIN_PRICE].copy()
    elig["mom_rank"] = (elig["m12"].rank() + elig["m6"].rank()) / 2
    elig = elig.sort_values("mom_rank", ascending=False).reset_index(drop=True)
    elig["mom_pos"] = np.arange(1, len(elig) + 1)
    dfr = dfr.merge(elig[["symbol", "mom_pos"]], on="symbol", how="left")
    dfr = dfr.sort_values("mom_pos", na_position="last").reset_index(drop=True)
    picks = set(elig.head(TOP_PICKS)["symbol"])

    def overall(r):
        above = sum(x == "Buy" for x in (r["s20"], r["s50"], r["s200"]))
        if r["symbol"] in picks and r["s200"] == "Buy":
            return "Strong Buy"
        if r["s200"] == "Buy" and r["s50"] == "Buy":
            return "Buy"
        if r["s200"] == "Sell":
            return "Sell"
        return "Hold"
    dfr["overall"] = dfr.apply(overall, axis=1)

    if not args.no_fundamentals:
        print("Fetching fundamentals (ROE/ROCE)...")
        for i, sym in enumerate(dfr["symbol"]):
            f = fundamentals(sym, cache)
            dfr.loc[dfr["symbol"] == sym, "roe"] = f["roe"]
            dfr.loc[dfr["symbol"] == sym, "roce"] = f["roce"]
            dfr.loc[dfr["symbol"] == sym, "fund_period"] = f["period"]
        with open(FUND_CACHE, "w") as f:
            json.dump(cache, f)
    else:
        dfr["roe"] = None; dfr["roce"] = None; dfr["fund_period"] = None

    os.makedirs(OUT_DIR, exist_ok=True)
    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    records = dfr.replace({np.nan: None}).to_dict("records")
    with open(os.path.join(OUT_DIR, "data.json"), "w") as f:
        json.dump({"updated": updated, "rows": records,
                   "picks": list(dfr.head(TOP_PICKS)["symbol"])}, f, indent=2)

    render_html(dfr, updated, records)
    print(f"Wrote {OUT_DIR}/index.html and data.json  ({len(dfr)} names, {updated})")


def render_html(dfr, updated, records):
    picks = dfr.head(TOP_PICKS)
    def sig_span(v):
        cls = {"Buy": "buy", "Sell": "sell", "Hold": "hold", "Strong Buy": "sbuy"}.get(v, "na")
        return f'<span class="tag {cls}">{v}</span>'
    def ffund(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if abs(v) > FUND_SANE:
            return "<span class='mut'>n/m</span>"   # balance-sheet artifact (e.g. negative equity)
        return f"{v}%"
    def fpos(v):
        return str(int(v)) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"

    pick_cards = "".join(
        f'<div class="card"><div class="rank">#{int(r.mom_pos)}</div>'
        f'<div class="sym">{r.symbol}</div><div class="px">₹{r.price:,.2f}</div>'
        f'<div class="chg {"up" if r.chg1d>=0 else "down"}">{r.chg1d:+.2f}%</div>'
        f'{sig_span(r.overall)}</div>'
        for r in picks.itertuples())

    tbody = "".join(
        f"<tr><td class='mono'>{fpos(r['mom_pos'])}</td><td class='sym'>{r['symbol']}</td>"
        f"<td class='num'>₹{r['price']:,.2f}</td>"
        f"<td class='num {'up' if r['chg1d']>=0 else 'down'}'>{r['chg1d']:+.2f}%</td>"
        f"<td>{sig_span(r['s20'])}</td><td>{sig_span(r['s50'])}</td><td>{sig_span(r['s200'])}</td>"
        f"<td>{sig_span(r['overall'])}</td>"
        f"<td class='num'>{ffund(r.get('roe'))}</td>"
        f"<td class='num'>{ffund(r.get('roce'))}</td></tr>"
        for r in records)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NSE Momentum Dashboard</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#e6edf3;--mut:#8b949e}}
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}}
.wrap{{max-width:1150px;margin:0 auto;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mut);font-size:13px;margin-bottom:20px}}
.disc{{background:#1c1206;border:1px solid #5a3a0a;color:#e3b341;font-size:12px;padding:10px 14px;border-radius:8px;margin-bottom:22px}}
h2{{font-size:15px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin:26px 0 12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;position:relative}}
.card .rank{{position:absolute;top:8px;right:10px;color:var(--mut);font-size:11px}}
.card .sym{{font-weight:600;font-size:15px}}.card .px{{font-size:18px;margin:4px 0}}
.card .chg{{font-size:12px;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}}
th{{color:var(--mut);font-weight:500;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--bg)}}
td.num,td.mono{{text-align:right;font-variant-numeric:tabular-nums}}.mut{{color:var(--mut)}}.sym{{font-weight:600}}
.up{{color:#3fb950}}.down{{color:#f85149}}
.tag{{padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}}
.tag.sbuy{{background:#132e1a;color:#3fb950;border:1px solid #238636}}
.tag.buy{{background:#0f2417;color:#3fb950}}.tag.hold{{background:#26210c;color:#e3b341}}
.tag.sell{{background:#2a1215;color:#f85149}}.tag.na{{background:#21262d;color:var(--mut)}}
input{{background:var(--card);border:1px solid var(--bd);color:var(--tx);padding:7px 10px;border-radius:7px;width:220px;margin-bottom:10px}}
.foot{{color:var(--mut);font-size:11px;margin-top:24px;line-height:1.6}}
</style></head><body><div class="wrap">
<h1>NSE Momentum Dashboard</h1>
<div class="sub">Large/mid-cap universe · updated {updated} · EOD data via Yahoo Finance</div>
<div class="disc"><b>Not investment advice.</b> "Recommendations" = top-ranked 12-month momentum names, the only
signal that survived validation (~+5%/yr alpha vs equal-weight, but lumpy, crash-prone, and negative over the last year).
Do your own research.</div>
<h2>Top {TOP_PICKS} Momentum Picks</h2>
<div class="cards">{pick_cards}</div>
<h2>Full Universe · Signals &amp; Fundamentals</h2>
<input id="q" placeholder="Filter by symbol…" onkeyup="filt()">
<div style="overflow-x:auto"><table id="t"><thead><tr>
<th onclick="sortT(0,1)">#</th><th onclick="sortT(1,0)">Symbol</th><th onclick="sortT(2,1)">Price</th>
<th onclick="sortT(3,1)">1D%</th><th onclick="sortT(4,0)">20 SMA</th><th onclick="sortT(5,0)">50 SMA</th>
<th onclick="sortT(6,0)">200 SMA</th><th onclick="sortT(7,0)">Signal</th>
<th onclick="sortT(8,1)">ROE</th><th onclick="sortT(9,1)">ROCE</th></tr></thead>
<tbody>{tbody}</tbody></table></div>
<div class="foot">
SMA tag: price &gt;0.5% above = Buy, &gt;0.5% below = Sell, else Hold. Overall = Strong Buy (a top pick above its 200 SMA),
Buy (above 50 &amp; 200 SMA), Sell (below 200 SMA), else Hold. ROE = trailing-twelve-month; ROCE = EBIT ÷ (Total Assets −
Current Liabilities), annualized from the latest quarterly statement. Momentum rank = avg of 12-1 &amp; 6-1 month returns.<br>
Built from the research in /backtests. Data may be delayed or wrong — verify before acting.
</div></div>
<script>
function filt(){{var q=document.getElementById('q').value.toUpperCase();document.querySelectorAll('#t tbody tr').forEach(function(r){{r.style.display=r.cells[1].textContent.toUpperCase().indexOf(q)>-1?'':'none'}})}}
function sortT(c,num){{var t=document.getElementById('t'),b=t.tBodies[0],rs=[].slice.call(b.rows);
t._d=!t._d;rs.sort(function(x,y){{var a=x.cells[c].textContent.replace(/[₹,%+]/g,''),d=y.cells[c].textContent.replace(/[₹,%+]/g,'');
return num?(t._d?a-d:d-a):(t._d?a.localeCompare(d):d.localeCompare(a))}});rs.forEach(function(r){{b.appendChild(r)}})}}
</script></body></html>"""
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
