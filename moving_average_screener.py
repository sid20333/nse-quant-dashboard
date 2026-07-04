"""
moving_average_screener.py — SMA crossovers (20/50/200) paired with RSI,
producing a single ranked bullish score usable for long-only signal
generation. Built from the theory discussed: 50/200 cross = long-term
regime, 20/50 cross = medium-term confirmation, RSI = short-term momentum
overlay, with slope added so a fresh, accelerating cross scores higher than
a stale, flattening one.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def slope(series: pd.Series, lookback: int = 15) -> pd.Series:
    """Simple normalized slope: % change of the series over `lookback` bars,
    used as a proxy for whether an MA is still turning up/down or flattening."""
    return series.pct_change(lookback)


@dataclass
class MAState:
    sma20: float
    sma50: float
    sma200: float
    rsi14: float
    slope200: float
    slope50: float
    golden_cross_50_200: bool
    golden_cross_20_50: bool
    death_cross_50_200: bool
    death_cross_20_50: bool
    bullish_score: float


def compute_ma_state(df: pd.DataFrame) -> MAState:
    """df must have a 'close' column, at least 210 bars of history."""
    s20 = sma(df["close"], 20)
    s50 = sma(df["close"], 50)
    s200 = sma(df["close"], 200)
    r = rsi(df["close"], 14)
    slope200 = slope(s200, 15)
    slope50 = slope(s50, 10)

    latest_20, latest_50, latest_200 = s20.iloc[-1], s50.iloc[-1], s200.iloc[-1]
    latest_rsi = r.iloc[-1]
    latest_slope200 = slope200.iloc[-1]
    latest_slope50 = slope50.iloc[-1]

    golden_50_200 = latest_50 > latest_200
    golden_20_50 = latest_20 > latest_50

    score = 0.0
    if golden_50_200:
        score += 2.0
    if golden_20_50:
        score += 1.0
    if pd.notna(latest_slope200) and latest_slope200 > 0:
        score += 1.0
    if pd.notna(latest_slope50) and latest_slope50 > 0:
        score += 0.5
    if pd.notna(latest_rsi):
        if 40 <= latest_rsi <= 65:
            score += 1.0  # healthy momentum, room to run
        elif latest_rsi > 70:
            score -= 1.0  # overbought - discourage chasing
        elif latest_rsi < 30:
            score -= 0.5  # weak/falling momentum despite any bullish cross

    return MAState(
        sma20=latest_20, sma50=latest_50, sma200=latest_200, rsi14=latest_rsi,
        slope200=latest_slope200, slope50=latest_slope50,
        golden_cross_50_200=golden_50_200, golden_cross_20_50=golden_20_50,
        death_cross_50_200=not golden_50_200, death_cross_20_50=not golden_20_50,
        bullish_score=score,
    )
