"""
vwap.py — Anchored VWAP ("EWAP -> NW: Anchored VWAP from the New Wave origin").

THEORY
------
Regular VWAP resets every session and is mostly a intraday execution
benchmark. ANCHORED VWAP instead starts accumulating from one specific,
meaningful bar (an "anchor") and never resets — it becomes a running
volume-weighted average of every trade since that anchor point.

Why it's useful: if the anchor is a genuinely significant event (a major
swing low, an earnings gap, a market-wide capitulation low), the Anchored
VWAP line approximates "the average price paid by everyone who has bought
since that event." As long as price holds above that line, the average
participant since the anchor is in profit and structurally more likely to
hold/add rather than sell — a rough proxy for "smart money since the anchor
is still in control." This is a heuristic mental model, not a proven causal
mechanism — treat crossings as one input, not a standalone signal.

Choosing the anchor ("New Wave" origin) matters enormously and is the part
most engines get wrong by hardcoding it. Two defensible, systematic choices
are implemented here:
  1. last_major_swing_low(): the lowest low of a meaningful, mechanically
     detected swing (not just any local minimum — see swing_lookback logic).
  2. anchor_from_earnings_date(): anchor exactly on the trading day of the
     most recent quarterly results, since that's often when the fundamental
     picture genuinely changed and volume profile shifts.
"""

import pandas as pd
import numpy as np
from typing import Optional


def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    """
    df must have columns: high, low, close, volume, indexed 0..n-1 sequentially.
    anchor_idx: integer position to start accumulating from (inclusive).

    Uses typical price = (high+low+close)/3 per bar, standard VWAP convention.
    Returns a Series aligned to df.index; values before anchor_idx are NaN.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]

    vwap = pd.Series(index=df.index, dtype=float)
    cum_pv = 0.0
    cum_vol = 0.0
    for pos in range(anchor_idx, len(df)):
        cum_pv += pv.iloc[pos]
        cum_vol += df["volume"].iloc[pos]
        vwap.iloc[pos] = cum_pv / cum_vol if cum_vol > 0 else np.nan

    return vwap


def last_major_swing_low(
    df: pd.DataFrame,
    swing_lookback: int = 60,
    min_bounce_pct: float = 0.10,
) -> Optional[int]:
    """
    Finds the most recent bar whose low is the lowest point within a
    `swing_lookback`-bar window on EITHER side, AND which was followed by
    at least a `min_bounce_pct` rally off that low (to filter out minor
    noise lows vs. genuine structural swing points).

    Returns the integer position of that swing low, or None if not found.
    Scans backwards from the most recent data so it finds the LATEST
    qualifying swing, not the oldest.
    """
    n = len(df)
    low = df["low"].values
    close = df["close"].values

    for i in range(n - swing_lookback - 1, swing_lookback, -1):
        window_start = max(0, i - swing_lookback)
        window_end = min(n, i + swing_lookback)
        local_low = low[window_start:window_end].min()

        if low[i] == local_low:
            post_window_end = min(n, i + swing_lookback)
            max_close_after = close[i:post_window_end].max() if post_window_end > i else low[i]
            bounce_pct = (max_close_after - low[i]) / low[i]
            if bounce_pct >= min_bounce_pct:
                return i

    return None


def anchor_from_date(df: pd.DataFrame, date_col: str, anchor_date: str) -> Optional[int]:
    """
    df[date_col] must be datetime-like. Returns the integer position of the
    first bar on/after anchor_date (e.g. the date of the last quarterly
    results announcement, sourced from your knowledge_base layer).
    """
    matches = df.index[df[date_col] >= pd.to_datetime(anchor_date)]
    if len(matches) == 0:
        return None
    return df.index.get_loc(matches[0])


def price_above_vwap(df: pd.DataFrame, vwap: pd.Series, confirm_bars: int = 3) -> pd.Series:
    """
    True where close has held ABOVE the anchored VWAP for at least the last
    `confirm_bars` consecutive bars (a single touch above isn't enough —
    this requires the hold to be sustained, reducing whipsaw false signals).
    """
    above = df["close"] > vwap
    return above.rolling(confirm_bars).sum() == confirm_bars
