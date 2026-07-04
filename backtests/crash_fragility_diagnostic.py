"""
crash_fragility_diagnostic.py — Can we see a crash coming?

Honest framing: crashes are triggered by exogenous shocks NOT in the price
data (COVID, a rate surprise, a default). So this does NOT try to predict a
crash's timing. It asks the answerable question instead: measured PURELY from
information available at the START of a quarter, do any classic FRAGILITY
signals separate the quarters that then crashed from the quarters that didn't?

Fragility != trigger. A dry forest doesn't tell you when lightning strikes,
but it tells you the fire will be bad if it does. These signals measure
"dry forest".

Signals (all point-in-time, computed on data up to the last trading day BEFORE
the quarter starts):
  1. VIX level            — India VIX close (fear/insurance price)
  2. VIX 1m change        — is fear already rising into the quarter?
  3. froth (idx vs 200MA) — % the index sits ABOVE its 200-day SMA (extension)
  4. realized-vol ratio   — 20d vol / 60d vol of the index (>1 = vol expanding)
  5. breadth              — % of universe trading above its OWN 200-day SMA
  6. avg pairwise corr    — mean 60d return correlation across the universe
                            (high = everything moving together = fragile)
  7. drawdown-from-high   — index vs its trailing 252d max (already off the top?)

Label: a quarter is a CRASH if NIFTY100 buy-and-hold that quarter < -5%.
CAVEAT up front: only 6 crash quarters in 2015-2024. Any 'signal' here is
illustrative, badly under-powered, and NOT a validated predictor. Read the
DIRECTION and consistency, not the precision.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/shrey/Downloads")

import numpy as np
import pandas as pd

from quant_engine.data_provider import YFinanceDataProvider
from quant_engine.backtests.nse_universe import UNIVERSE, REGIME_INDEX

CACHE_DIR = "/Users/shrey/Downloads/quant_engine/backtests/data_cache"
HIST_START, DATA_END = "2013-01-01", "2024-12-31"
provider = YFinanceDataProvider(cache_dir=CACHE_DIR)

idx = provider.get_daily_ohlcv(REGIME_INDEX, HIST_START, DATA_END).set_index("date")["close"]
vix = provider.get_daily_ohlcv("^INDIAVIX", HIST_START, DATA_END).set_index("date")["close"]

closes = {}
for sym in UNIVERSE:
    try:
        df = provider.get_daily_ohlcv(sym, HIST_START, DATA_END)
        closes[sym] = df.set_index("date")["close"]
    except Exception:
        pass
close_mat = pd.DataFrame(closes).sort_index()

sma200 = idx.rolling(200).mean()

WINDOWS = []
for yr in range(2015, 2025):
    WINDOWS += [(f"{yr}-01-01", f"{yr}-03-31"), (f"{yr}-04-01", f"{yr}-06-30"),
                (f"{yr}-07-01", f"{yr}-09-30"), (f"{yr}-10-01", f"{yr}-12-31")]


def signals_at(as_of):
    r = {}
    i = idx[idx.index <= as_of]
    s = sma200[sma200.index <= as_of]
    if len(i) < 260 or pd.isna(s.iloc[-1]):
        return None
    r["froth_%"] = (i.iloc[-1] / s.iloc[-1] - 1) * 100

    daily_ret = i.pct_change().dropna()
    rv20 = daily_ret.tail(20).std() * np.sqrt(252)
    rv60 = daily_ret.tail(60).std() * np.sqrt(252)
    r["rv20/rv60"] = rv20 / rv60 if rv60 > 0 else np.nan

    r["dd_from_high_%"] = (i.iloc[-1] / i.tail(252).max() - 1) * 100

    v = vix[vix.index <= as_of]
    r["vix"] = v.iloc[-1] if len(v) else np.nan
    r["vix_1m_chg_%"] = (v.iloc[-1] / v.iloc[-21] - 1) * 100 if len(v) > 21 else np.nan

    # breadth: % of universe above own 200MA
    sub = close_mat[close_mat.index <= as_of]
    live = sub.dropna(axis=1, thresh=200)
    if live.shape[1] >= 20:
        last = live.iloc[-1]
        ma = live.rolling(200).mean().iloc[-1]
        above = (last > ma).sum()
        r["breadth_%"] = above / ma.notna().sum() * 100
        # avg pairwise correlation over trailing 60d
        rets = live.tail(60).pct_change().dropna(how="all")
        rets = rets.dropna(axis=1)
        if rets.shape[1] >= 10 and len(rets) >= 20:
            c = rets.corr().values
            iu = np.triu_indices_from(c, k=1)
            r["avg_corr"] = np.nanmean(c[iu])
        else:
            r["avg_corr"] = np.nan
    else:
        r["breadth_%"] = np.nan
        r["avg_corr"] = np.nan
    return r


rows = []
for start, end in WINDOWS:
    as_of = idx[idx.index < pd.to_datetime(start)].index[-1]  # last bar before quarter
    sig = signals_at(as_of)
    if sig is None:
        continue
    seg = idx[(idx.index >= pd.to_datetime(start)) & (idx.index <= pd.to_datetime(end))]
    fwd = seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan
    sig["quarter"] = start[:7]
    sig["fwd_ret_%"] = fwd * 100
    sig["crash"] = fwd < -0.05
    rows.append(sig)

d = pd.DataFrame(rows)
cols = ["froth_%", "rv20/rv60", "dd_from_high_%", "vix", "vix_1m_chg_%", "breadth_%", "avg_corr"]

print("=" * 92)
print("QUARTER-START FRAGILITY SIGNALS vs NEXT-QUARTER OUTCOME (2015-2024, point-in-time)")
print("=" * 92)
hdr = f"{'quarter':<9}{'fwd%':>7}  " + "".join(f"{c:>13}" for c in cols)
print(hdr)
for _, row in d.iterrows():
    mark = " CRASH" if row["crash"] else ""
    line = f"{row['quarter']:<9}{row['fwd_ret_%']:>7.1f}  " + "".join(
        f"{row[c]:>13.2f}" if pd.notna(row[c]) else f"{'--':>13}" for c in cols)
    print(line + mark)

crash = d[d["crash"]]
calm = d[~d["crash"]]
print("\n" + "=" * 92)
print(f"SEPARATION: mean signal value  |  CRASH quarters (n={len(crash)})  vs  CALM (n={len(calm)})")
print("=" * 92)
print(f"{'signal':<16}{'crash_mean':>12}{'calm_mean':>12}{'gap':>10}   direction that flagged risk")
notes = {
    "froth_%": "lower/negative = already weak",
    "rv20/rv60": ">1 = vol expanding",
    "dd_from_high_%": "more negative = already off top",
    "vix": "higher = more fear",
    "vix_1m_chg_%": "positive = fear rising",
    "breadth_%": "lower = fewer stocks healthy",
    "avg_corr": "higher = everything moves together",
}
for c in cols:
    cm, km = crash[c].mean(), calm[c].mean()
    print(f"{c:<16}{cm:>12.2f}{km:>12.2f}{cm-km:>10.2f}   {notes[c]}")

print("\nRead: a signal is 'useful' only if crash_mean and calm_mean are far apart")
print("AND in the intuitive direction. With 6 crashes this is suggestive, not proof.")
