"""
generate.py — Build the static dashboard (docs/index.html + docs/data.json) and
ADVANCE the live paper track record (docs/track_record.json).

Run by GitHub Actions each weekday after NSE close: it recomputes signals from
EOD data, appends today's paper-portfolio NAV, and commits docs/ for GitHub
Pages. Shares logic with the live Streamlit app via dashboard/analytics.py:
idiosyncratic momentum, per-name context (volatility, 52w-high distance, ₹cr
liquidity), and the ≥₹50 / liquidity / "n/m" artifact fixes.

Run:  python dashboard/generate.py [--limit N] [--no-fundamentals]
"""
import sys, os, json, argparse, warnings, datetime as dt, importlib.util
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_uni = _imp("nse_universe", os.path.join(os.path.dirname(_HERE), "backtests", "nse_universe.py"))
A = _imp("analytics", os.path.join(_HERE, "analytics.py"))
TR = _imp("track_record", os.path.join(_HERE, "track_record.py"))
UNIVERSE, INDEX = _uni.UNIVERSE, _uni.REGIME_INDEX

TOP_PICKS, MIN_PRICE, MIN_ADV, FUND_SANE = 15, 50.0, 5.0, 100.0
OUT_DIR = os.path.join(os.path.dirname(_HERE), "docs")
FUND_CACHE = os.path.join(_HERE, "fundamentals_cache.json")


def sma_sig(p, s):
    if s is None or np.isnan(s):
        return "n/a"
    return "Buy" if p > s * 1.005 else "Sell" if p < s * 0.995 else "Hold"


def load_cache():
    if os.path.exists(FUND_CACHE):
        try:
            return json.load(open(FUND_CACHE))
        except Exception:
            return {}
    return {}


def fundamentals(symbol, cache):
    key = f"{symbol}.NS"
    e = cache.get(key)
    if e and (dt.date.today().toordinal() - e.get("asof_ord", 0)) < 7 and "sector" in e:
        return e
    out = {"roe": None, "roce": None, "sector": None, "asof_ord": dt.date.today().toordinal()}
    try:
        t = yf.Ticker(key); info = t.info
        out["sector"] = info.get("sector")
        roe = info.get("returnOnEquity")
        if roe is not None:
            out["roe"] = round(roe * 100, 1)
        bs, is_ = t.quarterly_balance_sheet, t.quarterly_income_stmt
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
            if ta and cl and ebit and (ta - cl) > 0:
                out["roce"] = round((ebit * 4) / (ta - cl) * 100, 1)
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

    prev_data = None                                   # for the "what changed" alert
    _dpath = os.path.join(OUT_DIR, "data.json")
    if os.path.exists(_dpath):
        try:
            prev_data = json.load(open(_dpath))
        except Exception:
            pass

    print(f"Fetching {len(tickers)} names + index…")
    data = yf.download(tickers + [INDEX], period="2y", auto_adjust=True, progress=False, group_by="ticker")
    close = pd.DataFrame({s: data[f"{s}.NS"]["Close"] for s in universe
                          if f"{s}.NS" in data.columns.get_level_values(0)}).dropna(how="all")
    vol = pd.DataFrame({s: data[f"{s}.NS"]["Volume"] for s in close.columns}).reindex(close.index)
    idx = data[INDEX]["Close"].reindex(close.index).ffill()
    keep = [s for s in close.columns if close[s].dropna().shape[0] >= 220]
    close, vol = close[keep], vol[keep]

    idio = A.idio_mom_scores(close, idx)
    ctx = A.context_metrics(close, vol)
    cache = load_cache()

    rows = []
    for s in keep:
        c = close[s].dropna()
        if len(c) < 220 or s not in idio.index:
            continue
        price, prev = float(c.iloc[-1]), float(c.iloc[-2])
        f = {} if args.no_fundamentals else fundamentals(s, cache)
        rows.append({"symbol": s, "price": round(price, 2), "chg1d": round((price / prev - 1) * 100, 2),
                     "sma20": round(float(c.rolling(20).mean().iloc[-1]), 2),
                     "sma50": round(float(c.rolling(50).mean().iloc[-1]), 2),
                     "sma200": round(float(c.rolling(200).mean().iloc[-1]), 2),
                     "idio": float(idio[s]),
                     "vol_ann": ctx.loc[s, "vol_ann"] if s in ctx.index else None,
                     "dist_52w": ctx.loc[s, "dist_52w"] if s in ctx.index else None,
                     "adv_cr": ctx.loc[s, "adv_cr"] if s in ctx.index else None,
                     "roe": f.get("roe"), "roce": f.get("roce"), "sector": f.get("sector")})
    if not args.no_fundamentals:
        json.dump(cache, open(FUND_CACHE, "w"))

    df = pd.DataFrame(rows)
    for col in ("sma20", "sma50", "sma200"):
        df["s" + col[3:]] = [sma_sig(p, s) for p, s in zip(df["price"], df[col])]
    elig = df[(df["price"] >= MIN_PRICE) & (df["adv_cr"].fillna(0) >= MIN_ADV)].copy()
    elig = elig.sort_values("idio", ascending=False).reset_index(drop=True)
    elig["mom_pos"] = np.arange(1, len(elig) + 1)
    df = df.merge(elig[["symbol", "mom_pos"]], on="symbol", how="left").sort_values("mom_pos", na_position="last")
    pickset = set(elig.head(TOP_PICKS)["symbol"])

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

    # advance the live paper track record
    picks = list(elig.head(TOP_PICKS)["symbol"])
    prices = dict(zip(df["symbol"], df["price"]))
    today = dt.date.today().isoformat()
    TR.advance(os.path.join(OUT_DIR, "track_record.json"), today, picks, prices, float(idx.iloc[-1]))

    os.makedirs(OUT_DIR, exist_ok=True)
    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    records = df.replace({np.nan: None}).to_dict("records")
    json.dump({"updated": updated, "rows": records, "picks": picks},
              open(os.path.join(OUT_DIR, "data.json"), "w"), indent=2)
    render_html(df, updated, records, picks)
    maybe_alert(prev_data, picks, df)
    print(f"Wrote docs/index.html, data.json, track_record.json  ({len(df)} names, {updated})")


def maybe_alert(prev, picks, df):
    """Optional: POST 'what changed' to a Slack/Discord webhook if the
    ALERT_WEBHOOK env var (a GitHub secret) is set. No-op otherwise."""
    hook = os.environ.get("ALERT_WEBHOOK")
    if not hook or not prev:
        return
    prev_picks = set(prev.get("picks", []))
    added = [s for s in picks if s not in prev_picks]
    dropped = [s for s in prev_picks if s not in set(picks)]
    prev_sig = {r["symbol"]: r.get("overall") for r in prev.get("rows", [])}
    cur_sig = dict(zip(df["symbol"], df["overall"]))
    changed = [f"{s}: {prev_sig[s]}→{cur_sig[s]}" for s in cur_sig
               if prev_sig.get(s) and prev_sig[s] != cur_sig[s]]
    if not (added or dropped or changed):
        return
    msg = "*NSE Momentum update*\n"
    if added:
        msg += "➕ new picks: " + ", ".join(added) + "\n"
    if dropped:
        msg += "➖ dropped: " + ", ".join(dropped) + "\n"
    if changed:
        msg += "🔀 " + "; ".join(changed[:12])
    try:
        import urllib.request
        payload = json.dumps({"text": msg, "content": msg}).encode()  # Slack "text" / Discord "content"
        req = urllib.request.Request(hook, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print("alert posted")
    except Exception as e:
        print("alert failed:", e)


def render_html(df, updated, records, picks):
    pk = df[df["symbol"].isin(picks)].sort_values("mom_pos").head(TOP_PICKS)
    def sig_span(v):
        cls = {"Buy": "buy", "Sell": "sell", "Hold": "hold", "Strong Buy": "sbuy"}.get(v, "na")
        return f'<span class="tag {cls}">{v}</span>'
    def fnum(v, suf=""):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}{suf}"
    def ffund(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return "<span class='mut'>n/m</span>" if abs(v) > FUND_SANE else f"{v:.1f}%"
    def fpos(v):
        return str(int(v)) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"

    pick_cards = "".join(
        f'<div class="card"><div class="rank">#{int(r.mom_pos)}</div><div class="sym">{r.symbol}</div>'
        f'<div class="px">₹{r.price:,.2f}</div><div class="chg {"up" if r.chg1d>=0 else "down"}">{r.chg1d:+.2f}%</div>'
        f'{sig_span(r.overall)}</div>' for r in pk.itertuples())
    tbody = "".join(
        f"<tr><td class='mono'>{fpos(r['mom_pos'])}</td><td class='sym'>{r['symbol']}</td>"
        f"<td>{r.get('sector') or '—'}</td><td class='num'>₹{r['price']:,.2f}</td>"
        f"<td class='num {'up' if r['chg1d']>=0 else 'down'}'>{r['chg1d']:+.2f}%</td>"
        f"<td>{sig_span(r['s20'])}</td><td>{sig_span(r['s50'])}</td><td>{sig_span(r['s200'])}</td>"
        f"<td>{sig_span(r['overall'])}</td><td class='num'>{fnum(r.get('dist_52w'),'%')}</td>"
        f"<td class='num'>{fnum(r.get('vol_ann'),'%')}</td><td class='num'>{fnum(r.get('adv_cr'))}</td>"
        f"<td class='num'>{ffund(r.get('roe'))}</td><td class='num'>{ffund(r.get('roce'))}</td></tr>"
        for r in records)

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>NSE Momentum Dashboard</title><style>
:root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#e6edf3;--mut:#8b949e}}
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}}
.wrap{{max-width:1250px;margin:0 auto;padding:24px}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.disc{{background:#1c1206;border:1px solid #5a3a0a;color:#e3b341;font-size:12px;padding:10px 14px;border-radius:8px;margin-bottom:22px}}
h2{{font-size:15px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin:26px 0 12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;position:relative}}
.card .rank{{position:absolute;top:8px;right:10px;color:var(--mut);font-size:11px}}.card .sym{{font-weight:600;font-size:15px}}
.card .px{{font-size:18px;margin:4px 0}}.card .chg{{font-size:12px;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}}
th{{color:var(--mut);font-weight:500;cursor:pointer;position:sticky;top:0;background:var(--bg)}}
td.num,td.mono{{text-align:right;font-variant-numeric:tabular-nums}}.mut{{color:var(--mut)}}.sym{{font-weight:600}}
.up{{color:#3fb950}}.down{{color:#f85149}}.tag{{padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}}
.tag.sbuy{{background:#132e1a;color:#3fb950;border:1px solid #238636}}.tag.buy{{background:#0f2417;color:#3fb950}}
.tag.hold{{background:#26210c;color:#e3b341}}.tag.sell{{background:#2a1215;color:#f85149}}.tag.na{{background:#21262d;color:var(--mut)}}
input{{background:var(--card);border:1px solid var(--bd);color:var(--tx);padding:7px 10px;border-radius:7px;width:220px;margin-bottom:10px}}
.foot{{color:var(--mut);font-size:11px;margin-top:24px;line-height:1.6}}</style></head><body><div class="wrap">
<h1>NSE Momentum Dashboard</h1><div class="sub">Large/mid-cap · updated {updated} · EOD via Yahoo · idiosyncratic momentum</div>
<div class="disc"><b>Not investment advice.</b> Picks = top idiosyncratic-momentum names (best signal from our research;
~+5%/yr alpha vs equal-weight, lumpy & crash-prone). Ranked among liquid (≥₹{MIN_PRICE:.0f}, ≥₹{MIN_ADV:.0f}cr/day) names.
For the live version + performance & charts, run the Streamlit app (see DEPLOY.md).</div>
<h2>Top {TOP_PICKS} Momentum Picks</h2><div class="cards">{pick_cards}</div>
<h2>Full Universe · Signals, Context &amp; Fundamentals</h2><input id="q" placeholder="Filter symbol…" onkeyup="filt()">
<div style="overflow-x:auto"><table id="t"><thead><tr>
<th onclick="s(0,1)">#</th><th onclick="s(1,0)">Symbol</th><th onclick="s(2,0)">Sector</th><th onclick="s(3,1)">Price</th>
<th onclick="s(4,1)">1D%</th><th onclick="s(5,0)">20</th><th onclick="s(6,0)">50</th><th onclick="s(7,0)">200</th>
<th onclick="s(8,0)">Signal</th><th onclick="s(9,1)">52w↓</th><th onclick="s(10,1)">Vol</th><th onclick="s(11,1)">ADV₹cr</th>
<th onclick="s(12,1)">ROE</th><th onclick="s(13,1)">ROCE</th></tr></thead><tbody>{tbody}</tbody></table></div>
<div class="foot">Signal: Strong Buy (a top pick above its 50 &amp; 200 SMA), Buy (above 50 &amp; 200, or a top pick above
200 but below 50), Sell (below 200), else Hold. 52w↓ = below 52-week high; Vol = 60d annualised; ADV = avg daily traded
value (₹cr); ROE = TTM; ROCE = EBIT÷(assets−curr.liab.), n/m when distorted (banks/negative equity). Data may be delayed
or wrong — verify before acting.</div></div>
<script>function filt(){{var q=document.getElementById('q').value.toUpperCase();document.querySelectorAll('#t tbody tr').forEach(function(r){{r.style.display=r.cells[1].textContent.toUpperCase().indexOf(q)>-1?'':'none'}})}}
function s(c,n){{var t=document.getElementById('t'),b=t.tBodies[0],rs=[].slice.call(b.rows);t._d=!t._d;rs.sort(function(x,y){{var a=x.cells[c].textContent.replace(/[₹,%+]/g,''),d=y.cells[c].textContent.replace(/[₹,%+]/g,'');return n?(t._d?a-d:d-a):(t._d?a.localeCompare(d):d.localeCompare(a))}});rs.forEach(function(r){{b.appendChild(r)}})}}</script></body></html>"""
    open(os.path.join(OUT_DIR, "index.html"), "w").write(html)


if __name__ == "__main__":
    main()
