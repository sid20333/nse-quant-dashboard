"""
technical.py — Volatility structure layer: Bollinger Bands, Keltner Channels,
and the "Triple Squeeze" compression detector.

THEORY
------
Bollinger Bands (BB): middle = N-day SMA, bands = middle +/- k * rolling std dev.
They measure volatility RELATIVE to recent price action. When bands narrow,
realized volatility is low (a "squeeze"). Squeezes don't predict direction —
only that a large move (in either direction) is statistically overdue,
because volatility is mean-reverting (periods of calm are followed by
periods of turbulence, and vice versa).

Keltner Channels (KC): middle = N-day EMA, bands = middle +/- m * ATR.
ATR (Average True Range) is a smoother volatility proxy than std dev because
it accounts for gaps. The classic "squeeze" signal (John Carter's TTM Squeeze)
is specifically: BB moves INSIDE KC. That is the single-timeframe version of
what your engine calls a squeeze.

Multi-timeframe squeeze (20/50/100 day):
Overlaying three BB windows is a proxy for "is compression happening at the
micro, medium AND macro scale simultaneously?" This is NOT a standard,
academically-defined indicator — it's a heuristic. Treat "bandwidth
percentile" (how narrow is today's bandwidth vs. its own history) as the
real signal, and treat "all three agree" as a confidence multiplier, not
a distinct new indicator with its own edge.

IMPORTANT CAVEAT: none of this tells you WHICH direction the eventual move
will go. Squeeze = "high energy stored", not "buy" or "sell". Direction has
to come from the valuation/VWAP/zone layers elsewhere in this engine.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


def bollinger_bands(close: pd.Series, window: int, num_std: float = 2.0) -> pd.DataFrame:
    """Returns a DataFrame with columns: mid, upper, lower, bandwidth.
    bandwidth = (upper - lower) / mid  — a volatility measure normalized by price level,
    which lets you compare compression across different stocks/price ranges.
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth})


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def keltner_channel(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, atr_mult: float = 2.0
) -> pd.DataFrame:
    mid = close.ewm(span=window, adjust=False).mean()
    atr = average_true_range(high, low, close, window)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "atr": atr})


def bandwidth_percentile_rank(bandwidth: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Rank today's bandwidth against its own trailing history (0 = tightest
    it's been in `lookback` bars, 1 = widest). This is the honest version of
    "is this squeezed?" — using a fixed hard threshold on raw bandwidth is
    meaningless across different stocks; percentile rank is comparable.
    """
    def _pct_rank(window_vals):
        current = window_vals[-1]
        return (window_vals < current).sum() / (len(window_vals) - 1) if len(window_vals) > 1 else np.nan

    return bandwidth.rolling(lookback).apply(_pct_rank, raw=True)


@dataclass
class SqueezeState:
    is_squeezed: bool
    bb20_pct_rank: float
    bb50_pct_rank: float
    bb100_pct_rank: float
    ttm_squeeze_on: bool  # classic single-TF: 20-day BB inside 20-day KC


def detect_triple_squeeze(
    df: pd.DataFrame,
    percentile_threshold: float = 0.15,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    df must have columns: open, high, low, close, volume.

    Returns df augmented with bb_20/50/100 bandwidth percentile ranks and a
    boolean `triple_squeeze` column: True when all three timeframes are
    simultaneously in their tightest `percentile_threshold` of the last
    `lookback` bars, AND the classic TTM squeeze (BB inside KC on the 20-day)
    is also active as a confirming, direction-agnostic trigger.
    """
    out = df.copy()

    bb20 = bollinger_bands(out["close"], 20)
    bb50 = bollinger_bands(out["close"], 50)
    bb100 = bollinger_bands(out["close"], 100)
    kc20 = keltner_channel(out["high"], out["low"], out["close"], 20)

    out["bb20_bw"] = bb20["bandwidth"]
    out["bb50_bw"] = bb50["bandwidth"]
    out["bb100_bw"] = bb100["bandwidth"]
    out["bb20_lower"], out["bb20_upper"] = bb20["lower"], bb20["upper"]
    out["bb50_lower"], out["bb50_upper"] = bb50["lower"], bb50["upper"]
    out["bb100_lower"], out["bb100_upper"] = bb100["lower"], bb100["upper"]

    out["bb20_pct_rank"] = bandwidth_percentile_rank(out["bb20_bw"], lookback)
    out["bb50_pct_rank"] = bandwidth_percentile_rank(out["bb50_bw"], lookback)
    out["bb100_pct_rank"] = bandwidth_percentile_rank(out["bb100_bw"], lookback)

    out["ttm_squeeze_on"] = (bb20["upper"] < kc20["upper"]) & (bb20["lower"] > kc20["lower"])

    out["triple_squeeze"] = (
        (out["bb20_pct_rank"] <= percentile_threshold)
        & (out["bb50_pct_rank"] <= percentile_threshold)
        & (out["bb100_pct_rank"] <= percentile_threshold)
    )

    return out


def volume_confirmation(volume: pd.Series, mv_window: int = 20, mult_required: float = 1.5) -> pd.Series:
    """
    MV = 20-day SMA of volume ("Market Volume" baseline).
    Returns bool series: True where today's volume >= mult_required * MV.
    Used to confirm breakout candles have real institutional participation
    behind them, not just noise.
    """
    mv = volume.rolling(mv_window).mean()
    return volume >= (mult_required * mv)
