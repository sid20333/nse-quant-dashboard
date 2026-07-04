"""
valuation.py — Intrinsic Value (IV) vs Market Value (MV) layer.

THEORY
------
The core idea: a stock's "fair" per-share value can be estimated from its
underlying cash-generating ability, independent of what the market is
currently paying for it. If Intrinsic Value (IV) sits well above Market
Value (MV), you have a "margin of safety" — room for your valuation model
to be wrong and still make money.

Three methods are implemented, from most to least assumption-heavy:

1. Multi-stage DCF (Discounted Cash Flow)
   IV = sum of discounted projected Free Cash Flows + discounted Terminal Value
   This is the most "correct" in theory but the most sensitive to inputs.
   A 1% change in discount rate can move IV by 10-20%. Garbage in, garbage out.

2. Graham's Revised Formula (a sanity-check heuristic, not a real DCF)
   IV = EPS * (8.5 + 2g) * 4.4 / Y
   where g = expected 7-10yr growth rate (%), Y = current AAA corporate bond yield (%).
   Crude, but useful as a second opinion because it has almost no moving parts.

3. EPV (Earnings Power Value, Bruce Greenwald's method)
   EPV assumes NO future growth — it values the company purely on its
   current sustainable earning power. If EPV > current DCF-implied growth
   value, the market is already pricing in growth that may not materialize.
   This makes EPV a useful "floor" value / conservatism check.

WHY THIS MATTERS FOR YOUR ENGINE
---------------------------------
Never rely on IV from a single method as ground truth. The engine below
computes a *range* (DCF, Graham, EPV) and reports the spread. A stock where
all three broadly agree deserves more confidence than one where DCF says
"buy" but EPV says "overpriced".
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValuationResult:
    dcf_value: Optional[float]
    graham_value: Optional[float]
    epv_value: Optional[float]
    market_value: float
    blended_iv: float  # simple average of available methods
    margin_of_safety: float  # (blended_iv - mv) / mv
    agreement_spread: float  # (max - min) / blended_iv across available methods -> uncertainty measure


def dcf_intrinsic_value(
    fcf_projections: List[float],
    terminal_growth: float,
    discount_rate: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
) -> float:
    """
    Multi-stage DCF -> equity value per share.

    fcf_projections: list of projected Free Cash Flows for years 1..N (absolute currency, not per share)
    terminal_growth: perpetual growth rate applied after the projection window (keep < discount_rate!)
    discount_rate: WACC or required return (e.g. 0.12 for 12%)
    shares_outstanding: diluted share count
    net_debt: total debt - cash (subtracted to go from Enterprise Value to Equity Value)
    """
    if terminal_growth >= discount_rate:
        raise ValueError(
            "terminal_growth must be < discount_rate, otherwise the terminal "
            "value formula diverges (this is a very common DCF bug)."
        )

    pv_fcf = 0.0
    for year, fcf in enumerate(fcf_projections, start=1):
        pv_fcf += fcf / ((1 + discount_rate) ** year)

    n = len(fcf_projections)
    terminal_value = (fcf_projections[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** n)

    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - net_debt

    return equity_value / shares_outstanding


def graham_intrinsic_value(eps: float, growth_rate_pct: float, aaa_bond_yield_pct: float) -> float:
    """
    Benjamin Graham's revised formula (1974 update, normalized to a 4.4% baseline yield):
        IV = EPS * (8.5 + 2g) * 4.4 / Y

    eps: trailing twelve month diluted EPS
    growth_rate_pct: expected 7-10 year annual growth rate, as a plain number (e.g. 12 for 12%)
    aaa_bond_yield_pct: current AAA corporate bond yield, as a plain number (e.g. 7.5 for 7.5%)

    Caveat: this formula predates modern rate regimes and tends to overstate
    IV when growth assumptions are optimistic. Treat as a rough cross-check only.
    """
    return eps * (8.5 + 2 * growth_rate_pct) * 4.4 / aaa_bond_yield_pct


def epv(
    normalized_ebit: float,
    tax_rate: float,
    wacc: float,
    maintenance_capex: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
) -> float:
    """
    Earnings Power Value (Greenwald) per share — values the business at
    ZERO assumed growth, using only its current, normalized, sustainable
    after-tax earnings power minus reinvestment needed just to stand still.

    normalized_ebit: EBIT averaged/normalized across a full business cycle (not a single good/bad year)
    tax_rate: effective tax rate, plain fraction (e.g. 0.25)
    wacc: weighted average cost of capital, used as the no-growth discount/cap rate
    maintenance_capex: capex required merely to maintain current earning power (not growth capex)
    """
    nopat = normalized_ebit * (1 - tax_rate)
    distributable_earnings = nopat - maintenance_capex * (1 - tax_rate)
    enterprise_value = distributable_earnings / wacc
    equity_value = enterprise_value - net_debt
    return equity_value / shares_outstanding


def margin_of_safety(intrinsic_value: float, market_value: float) -> float:
    """(IV - MV) / MV. Returns e.g. 0.25 for a 25% margin of safety."""
    return (intrinsic_value - market_value) / market_value


def blended_valuation(
    market_value: float,
    dcf_value: Optional[float] = None,
    graham_value: Optional[float] = None,
    epv_value: Optional[float] = None,
) -> ValuationResult:
    """Combine whichever valuation methods are available and quantify disagreement between them."""
    values = [v for v in (dcf_value, graham_value, epv_value) if v is not None]
    if not values:
        raise ValueError("At least one valuation method must be provided.")

    blended = sum(values) / len(values)
    spread = (max(values) - min(values)) / blended if blended != 0 else float("inf")
    mos = margin_of_safety(blended, market_value)

    return ValuationResult(
        dcf_value=dcf_value,
        graham_value=graham_value,
        epv_value=epv_value,
        market_value=market_value,
        blended_iv=blended,
        margin_of_safety=mos,
        agreement_spread=spread,
    )
