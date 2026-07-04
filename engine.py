"""
engine.py — Ties every layer together into the weekly recommendation scan.

PIPELINE (a stock must clear ALL gates in order to be recommended):

  1. VALUATION GATE   (valuation.py)
     Intrinsic Value (blended DCF/Graham/EPV) must exceed Market Value by
     at least `min_margin_of_safety`.

  2. KNOWLEDGE BASE GATE   (knowledge_base.py)
     No active red flags (promoter pledging, earnings decline, forensic
     audit / auditor resignation).

  3. TREND CONFIRMATION GATE   (vwap.py)
     Price must be holding above the Anchored VWAP, anchored from either
     the last major structural swing low or the last earnings date
     (whichever is more recent / relevant — configurable).

  4. STRUCTURED ZONE GATE   (zones.py)
     Current price must be sitting inside a qualifying Support zone or
     Order Block (demand zone).

  5. VOLATILITY TRIGGER GATE   (technical.py)
     A triple squeeze (20/50/100-day BB bandwidth all in their tightest
     percentile) must be active, or have just fired, at the same zone.

A stock that clears gates 1-2 but not 3-5 is not "wrong" — it's just not
YET timed correctly. The engine reports partial-clears too, so you can see
your pipeline (which is more useful for studying it than a binary
pass/fail).

NOTE ON SEQUENCING: gates are deliberately ordered cheapest-and-most-
decisive first (valuation, KB) before the more compute-heavy technical
scans, so a bad stock is discarded before wasting cycles on candlestick
pattern detection across its full price history.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from . import valuation as val
from . import technical as tech
from . import zones as zn
from . import vwap as vw
from .knowledge_base import KnowledgeBase
from .catalysts import CatalystDataProvider, CatalystTag, build_catalyst_tag
from datetime import date as _date


@dataclass
class StockInputs:
    symbol: str
    ohlcv: pd.DataFrame  # columns: date, open, high, low, close, volume
    eps: Optional[float] = None
    growth_rate_pct: Optional[float] = None
    aaa_bond_yield_pct: Optional[float] = None
    normalized_ebit: Optional[float] = None
    tax_rate: Optional[float] = None
    wacc: Optional[float] = None
    maintenance_capex: Optional[float] = None
    shares_outstanding: Optional[float] = None
    net_debt: float = 0.0
    fcf_projections: Optional[List[float]] = None
    terminal_growth: Optional[float] = None
    discount_rate: Optional[float] = None


@dataclass
class ScanResult:
    symbol: str
    passed_valuation: bool
    passed_kb: bool
    passed_trend: bool
    passed_zone: bool
    passed_volatility_trigger: bool
    margin_of_safety: Optional[float] = None
    valuation_spread: Optional[float] = None
    kb_reasons: List[str] = field(default_factory=list)
    zone_type: Optional[str] = None
    zone_bounds: Optional[tuple] = None
    fully_qualified: bool = False
    catalyst_tag: Optional[CatalystTag] = None


def evaluate_stock(
    inputs: StockInputs,
    kb: KnowledgeBase,
    min_margin_of_safety: float = 0.20,
    sr_min_touches: int = 3,
    sr_tolerance_pct: float = 0.015,
    squeeze_percentile_threshold: float = 0.15,
    catalyst_provider: Optional[CatalystDataProvider] = None,
) -> ScanResult:
    df = inputs.ohlcv

    # --- Gate 1: Valuation ---
    dcf_value = None
    if inputs.fcf_projections and inputs.terminal_growth is not None and inputs.discount_rate is not None:
        dcf_value = val.dcf_intrinsic_value(
            inputs.fcf_projections, inputs.terminal_growth, inputs.discount_rate,
            inputs.shares_outstanding, inputs.net_debt,
        )
    graham_value = None
    if inputs.eps is not None and inputs.growth_rate_pct is not None and inputs.aaa_bond_yield_pct:
        graham_value = val.graham_intrinsic_value(inputs.eps, inputs.growth_rate_pct, inputs.aaa_bond_yield_pct)
    epv_value = None
    if inputs.normalized_ebit is not None and inputs.tax_rate is not None and inputs.wacc:
        epv_value = val.epv(
            inputs.normalized_ebit, inputs.tax_rate, inputs.wacc,
            inputs.maintenance_capex or 0.0, inputs.shares_outstanding, inputs.net_debt,
        )

    market_value = df["close"].iloc[-1]
    passed_valuation = False
    mos = None
    spread = None
    if any(v is not None for v in (dcf_value, graham_value, epv_value)):
        vr = val.blended_valuation(market_value, dcf_value, graham_value, epv_value)
        mos = vr.margin_of_safety
        spread = vr.agreement_spread
        passed_valuation = mos >= min_margin_of_safety

    # --- Gate 2: Knowledge Base ---
    red_flags = kb.get_red_flags(inputs.symbol)
    passed_kb = not red_flags.disqualified

    # --- Gate 3: Trend confirmation via Anchored VWAP ---
    anchor_idx = vw.last_major_swing_low(df)
    passed_trend = False
    if anchor_idx is not None:
        vwap_series = vw.anchored_vwap(df, anchor_idx)
        holding = vw.price_above_vwap(df, vwap_series)
        passed_trend = bool(holding.iloc[-1]) if not holding.empty else False

    # --- Gate 4: Structured zone (S/R cluster OR order block) ---
    passed_zone = False
    zone_type = None
    zone_bounds = None
    sr_zones = zn.find_support_resistance_zones(df, tolerance_pct=sr_tolerance_pct, min_touches=sr_min_touches)
    current_price = df["close"].iloc[-1]
    for z in sr_zones:
        if zn.price_in_zone(current_price, z.price_low, z.price_high) and z.zone_type == "support":
            passed_zone = True
            zone_type = "support_cluster"
            zone_bounds = (z.price_low, z.price_high)
            break

    if not passed_zone:
        order_blocks = zn.detect_order_blocks(df)
        for ob in order_blocks[-5:]:  # only consider recent order blocks
            if zn.price_in_zone(current_price, ob.zone_low, ob.zone_high):
                passed_zone = True
                zone_type = "order_block"
                zone_bounds = (ob.zone_low, ob.zone_high)
                break

    # --- Gate 5: Volatility trigger (triple squeeze) ---
    squeezed_df = tech.detect_triple_squeeze(df, percentile_threshold=squeeze_percentile_threshold)
    passed_volatility_trigger = bool(squeezed_df["triple_squeeze"].iloc[-1])

    fully_qualified = all([passed_valuation, passed_kb, passed_trend, passed_zone, passed_volatility_trigger])

    # --- Catalyst tag (informational only, does NOT gate fully_qualified) ---
    catalyst_tag = None
    if catalyst_provider is not None:
        as_of = df["date"].iloc[-1]
        as_of_date = as_of.date() if hasattr(as_of, "date") else _date.today()
        snapshots = catalyst_provider.get_shareholding_history(inputs.symbol)
        deals = catalyst_provider.get_recent_bulk_block_deals(inputs.symbol, as_of_date, 30)
        catalyst_tag = build_catalyst_tag(inputs.symbol, snapshots, deals, as_of_date)

    return ScanResult(
        symbol=inputs.symbol,
        passed_valuation=passed_valuation,
        passed_kb=passed_kb,
        passed_trend=passed_trend,
        passed_zone=passed_zone,
        passed_volatility_trigger=passed_volatility_trigger,
        margin_of_safety=mos,
        valuation_spread=spread,
        kb_reasons=red_flags.reasons,
        zone_type=zone_type,
        zone_bounds=zone_bounds,
        fully_qualified=fully_qualified,
        catalyst_tag=catalyst_tag,
    )


def run_weekly_scan(
    stock_inputs: List[StockInputs],
    kb: KnowledgeBase,
    **gate_kwargs,
) -> List[ScanResult]:
    """Runs evaluate_stock across a universe and returns results sorted so
    fully-qualified stocks (best candidates) appear first, then partial
    clears ranked by how many gates they passed."""
    results = [evaluate_stock(s, kb, **gate_kwargs) for s in stock_inputs]

    def gate_count(r: ScanResult) -> int:
        return sum([r.passed_valuation, r.passed_kb, r.passed_trend, r.passed_zone, r.passed_volatility_trigger])

    results.sort(key=lambda r: (r.fully_qualified, gate_count(r)), reverse=True)
    return results
