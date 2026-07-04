"""
analytics.py — Shared, network-free analytics for the dashboards. Pure functions
(take prefetched price frames, return numbers) so they're unit-testable and used
by both streamlit_app.py (live) and dashboard/generate.py (static).

Contents:
  idio_mom_scores   — idiosyncratic (beta-residual) 12-1 momentum. The signal our
                      research found best (Sharpe 1.43 vs 1.32 for raw momentum).
  raw_mom_scores    — fallback 12-1 & 6-1 rank momentum.
  context_metrics   — per-name volatility, distance from 52-week high, and average
                      daily traded value (ADV, a real liquidity measure).
  momentum_equity_curve — backtested equity curve of the strategy vs equal-weight
                      vs index, for the Performance tab. Semiannual, top-N, EW.
"""
import numpy as np
import pandas as pd


def raw_mom_scores(close: pd.DataFrame) -> pd.Series:
    if len(close) < 252:
        return pd.Series(dtype=float)
    m12 = close.iloc[-21] / close.iloc[-252] - 1
    m6 = close.iloc[-21] / close.iloc[-126] - 1
    return (m12.rank() + m6.rank()) / 2


def idio_mom_scores(close: pd.DataFrame, idx_close: pd.Series, lookback: int = 252,
                    skip: int = 21, min_obs: int = 180) -> pd.Series:
    """12-1 momentum of each stock's return NET of its market beta (residual
    momentum). Tilts away from high-beta names that crash hardest."""
    rets = close.pct_change()
    iret = idx_close.pct_change().reindex(rets.index)
    win = rets.iloc[-lookback:]
    iw = iret.iloc[-lookback:]
    out = {}
    for s in close.columns:
        y = win[s].dropna()
        if len(y) < min_obs:
            continue
        x = iw.reindex(y.index)
        xy = pd.concat([x, y], axis=1).dropna()
        if len(xy) < min_obs:
            continue
        beta = np.polyfit(xy.iloc[:, 0].values, xy.iloc[:, 1].values, 1)[0]
        resid = xy.iloc[:, 1] - beta * xy.iloc[:, 0]
        r = resid.iloc[:-skip] if skip else resid          # skip most-recent month
        out[s] = float((1 + r).prod() - 1)
    return pd.Series(out)


def context_metrics(close: pd.DataFrame, volume: pd.DataFrame,
                    vol_win: int = 60, adv_win: int = 20) -> pd.DataFrame:
    rows = {}
    rets = close.pct_change()
    for s in close.columns:
        c = close[s].dropna()
        if len(c) < vol_win:
            continue
        v = volume[s].reindex(c.index) if s in volume.columns else None
        vol_ann = rets[s].iloc[-vol_win:].std() * np.sqrt(252) * 100
        dist_52w = (c.iloc[-1] / c.iloc[-252:].max() - 1) * 100 if len(c) >= 252 else np.nan
        adv_cr = None
        if v is not None and len(v.dropna()) >= adv_win:
            adv_cr = float((c.iloc[-adv_win:] * v.iloc[-adv_win:]).mean() / 1e7)  # Rs crore/day
        rows[s] = {"vol_ann": round(float(vol_ann), 1),
                   "dist_52w": round(float(dist_52w), 1) if pd.notna(dist_52w) else None,
                   "adv_cr": round(adv_cr, 1) if adv_cr is not None else None}
    return pd.DataFrame(rows).T


def momentum_equity_curve(close: pd.DataFrame, idx_close: pd.Series, top_n: int = 15,
                          min_price: float = 50.0, use_idio: bool = True) -> pd.DataFrame:
    """Semiannual top-N momentum (equal weight) vs equal-weight-all vs index.
    Returns a DataFrame of cumulative growth of 1 (columns: Strategy, EqualWeight, Index)."""
    mret = close.resample("ME").last().pct_change()
    idx_m = idx_close.resample("ME").last().pct_change().reindex(mret.index)
    rebal = list(pd.Series(close.index, index=close.index).resample("2QE").last().dropna().values)
    rebal = [pd.Timestamp(d) for d in rebal]
    baskets = {}
    for d in rebal:
        sub = close[close.index <= d]
        if len(sub) < 252:
            continue
        sc = idio_mom_scores(sub, idx_close[idx_close.index <= d]) if use_idio else raw_mom_scores(sub)
        px = sub.iloc[-1]
        sc = sc[[s for s in sc.index if px.get(s, 0) >= min_price]]
        if len(sc):
            baskets[d] = list(sc.nlargest(top_n).index)
    strat, ew, idxr, dates = [], [], [], []
    for m in mret.index:
        prior = [d for d in rebal if d < m and d in baskets]
        if not prior:
            continue
        bk = baskets[prior[-1]]
        row = mret.loc[m]
        s = row[[x for x in bk if x in row.index]].dropna().mean()
        e = row[row.notna()].mean()
        if pd.isna(s) or pd.isna(e) or pd.isna(idx_m.loc[m]):
            continue
        strat.append(s); ew.append(e); idxr.append(idx_m.loc[m]); dates.append(m)
    if not dates:
        return pd.DataFrame()
    df = pd.DataFrame({"Strategy": strat, "EqualWeight": ew, "Index": idxr}, index=dates)
    return (1 + df).cumprod()
