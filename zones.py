"""
zones.py — Structured price zones: Support/Resistance clusters and
Order Blocks (Drop-Base-Rally demand zones).

THEORY
------
Support/Resistance (S/R) Clusters:
The idea is that price levels where the market has reversed multiple times
represent zones of concentrated buy/sell interest (stop clusters, resting
limit orders, psychological round numbers). We find this by:
  1. Detecting local pivot highs/lows (a bar whose high/low is more extreme
     than N bars on either side).
  2. Clustering pivots that are within a small % tolerance of each other,
     since real-world reversals rarely repeat at the EXACT same price.
  3. Keeping only clusters with >= min_touches — these are your "structured"
     zones as opposed to a single, possibly random, reversal.

HONEST CAVEAT: this is inherently a lagging, retrospective description of
where price already reversed. It is NOT predictive on its own — it only
becomes useful in combination with a forward-looking reason (valuation,
volume, trend) to expect the zone to hold again. Also, with enough
tolerance and history, you can "find" a cluster almost anywhere — be
skeptical of over-fit zone boundaries with very wide tolerance.

Order Blocks / "Drop-Base-Rally":
A heuristic pattern-detector, not a rigorously defined statistical concept
(this terminology comes from retail "Smart Money Concepts" trading content,
not institutional literature — treat it as a visual heuristic, and validate
it in your backtest before trusting it). The pattern:
  1. DROP: a clear directional down-move over `drop_window` bars.
  2. BASE: `base_window` bars of low-range consolidation immediately after
     (a "coiling" period — narrow high-low range relative to the drop).
  3. RALLY: a strong up-move breaking out of the base range, ideally on
     above-average volume (see technical.volume_confirmation).
The BASE range itself becomes the "demand zone" — the working hypothesis
is that a later pullback INTO that same price range represents renewed
opportunity to join the original institutional buyers, not a new,
unrelated support level.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SRZone:
    price_low: float
    price_high: float
    touches: int
    zone_type: str  # "support" or "resistance"


@dataclass
class OrderBlock:
    start_idx: int
    end_idx: int
    zone_low: float
    zone_high: float
    breakout_idx: int
    breakout_volume_ratio: float


def find_pivots(series: pd.Series, order: int = 3) -> Tuple[List[int], List[int]]:
    """
    Returns (pivot_high_indices, pivot_low_indices).
    A pivot high at i means series[i] is the max within [i-order, i+order].
    Simple, dependency-free implementation (avoids relying on scipy internals
    that vary by version).
    """
    highs, lows = [], []
    n = len(series)
    vals = series.values
    for i in range(order, n - order):
        window = vals[i - order : i + order + 1]
        if vals[i] == window.max() and (window == vals[i]).sum() == 1:
            highs.append(i)
        if vals[i] == window.min() and (window == vals[i]).sum() == 1:
            lows.append(i)
    return highs, lows


def cluster_levels(prices: List[float], tolerance_pct: float = 0.015) -> List[Tuple[float, float, int]]:
    """
    Greedy clustering of price levels within tolerance_pct of each other.
    Returns list of (low, high, touch_count) for each cluster.
    """
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters = []
    current_cluster = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        cluster_mean = np.mean(current_cluster)
        if (p - cluster_mean) / cluster_mean <= tolerance_pct:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)

    return [(min(c), max(c), len(c)) for c in clusters]


def find_support_resistance_zones(
    df: pd.DataFrame,
    pivot_order: int = 3,
    tolerance_pct: float = 0.015,
    min_touches: int = 3,
) -> List[SRZone]:
    """
    df must have columns: high, low.
    "Weekly chart has reversed at least 3 times in 6 months" -> pass in a
    weekly-resampled df and set min_touches=3.
    """
    pivot_high_idx, pivot_low_idx = find_pivots(df["high"], order=pivot_order)
    resistance_prices = df["high"].iloc[pivot_high_idx].tolist()

    pivot_high_idx2, pivot_low_idx2 = find_pivots(df["low"], order=pivot_order)
    support_prices = df["low"].iloc[pivot_low_idx2].tolist()

    zones = []
    for lo, hi, touches in cluster_levels(support_prices, tolerance_pct):
        if touches >= min_touches:
            zones.append(SRZone(price_low=lo, price_high=hi, touches=touches, zone_type="support"))
    for lo, hi, touches in cluster_levels(resistance_prices, tolerance_pct):
        if touches >= min_touches:
            zones.append(SRZone(price_low=lo, price_high=hi, touches=touches, zone_type="resistance"))

    return zones


def detect_order_blocks(
    df: pd.DataFrame,
    drop_window: int = 5,
    drop_threshold_pct: float = 0.06,
    base_window: int = 4,
    base_max_range_pct: float = 0.03,
    breakout_threshold_pct: float = 0.04,
    volume_mult_required: float = 1.5,
) -> List[OrderBlock]:
    """
    df must have columns: open, high, low, close, volume, indexed sequentially.

    Scans for: DROP (>= drop_threshold_pct decline over drop_window bars)
    -> BASE (base_window bars with high-low range <= base_max_range_pct of
    the base's average close) -> RALLY (breakout candle closing
    >= breakout_threshold_pct above the base range high, with volume
    >= volume_mult_required * 20-day average volume).

    This is intentionally a simple, transparent state machine — extend the
    thresholds/logic once you've backtested which parameter ranges actually
    produce an edge on your universe, rather than trusting these defaults.
    """
    from .technical import volume_confirmation

    vol_confirmed = volume_confirmation(df["volume"], mv_window=20, mult_required=volume_mult_required)

    blocks = []
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    i = drop_window
    while i < n - base_window - 1:
        # 1. DROP check: decline from close[i-drop_window] to close[i]
        drop_start_price = close[i - drop_window]
        drop_end_price = close[i]
        drop_pct = (drop_start_price - drop_end_price) / drop_start_price

        if drop_pct >= drop_threshold_pct:
            base_start = i
            base_end = i + base_window
            if base_end >= n:
                break

            base_high = high[base_start:base_end].max()
            base_low = low[base_start:base_end].min()
            base_avg_close = close[base_start:base_end].mean()
            base_range_pct = (base_high - base_low) / base_avg_close

            if base_range_pct <= base_max_range_pct:
                # 2. BASE confirmed. Check for RALLY breakout on next bar(s).
                breakout_idx = base_end
                if breakout_idx < n:
                    breakout_close = close[breakout_idx]
                    breakout_pct = (breakout_close - base_high) / base_high
                    if breakout_pct >= breakout_threshold_pct and vol_confirmed.iloc[breakout_idx]:
                        mv = df["volume"].rolling(20).mean().iloc[breakout_idx]
                        vol_ratio = df["volume"].iloc[breakout_idx] / mv if mv > 0 else np.nan
                        blocks.append(
                            OrderBlock(
                                start_idx=base_start,
                                end_idx=base_end,
                                zone_low=base_low,
                                zone_high=base_high,
                                breakout_idx=breakout_idx,
                                breakout_volume_ratio=vol_ratio,
                            )
                        )
                        i = breakout_idx + 1
                        continue
        i += 1

    return blocks


def price_in_zone(price: float, zone_low: float, zone_high: float, tolerance_pct: float = 0.01) -> bool:
    """True if price is inside [zone_low, zone_high], expanded by tolerance_pct on each side."""
    span = zone_high - zone_low
    buffer = span * tolerance_pct if span > 0 else zone_low * tolerance_pct
    return (zone_low - buffer) <= price <= (zone_high + buffer)
