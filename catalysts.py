"""
catalysts.py — The quantifiable half of the "why would the market re-rate
this now" layer: FII/DII shareholding trend and bulk/block deal activity.

THEORY
------
This module deliberately does NOT produce a pass/fail gate like valuation
or the knowledge base. It produces a TAG — context to display alongside a
fully-qualified stock, that you weigh yourself. Reducing "is there a
catalyst" to a hard threshold risks false precision on something that's
genuinely softer than valuation math.

1. FII/DII SHAREHOLDING TREND
   NSE/BSE require quarterly disclosure of shareholding pattern (promoter,
   FII/FPI, DII, public). A stock where FII or DII holding has risen for
   2-3 consecutive quarters suggests institutional money is already
   accumulating — often ahead of the price fully reflecting it, since large
   institutions build positions gradually to avoid moving the price against
   themselves.

   CAVEAT: this is necessarily a LAGGING quarterly snapshot (disclosed
   ~15-21 days after quarter-end per SEBI norms) — by the time you see 2
   quarters of rising FII holding, the accumulation has already been
   happening for 4-7 months. Treat it as confirmation of a trend already
   underway, not an early-warning signal.

2. PROMOTER BUYING (distinct from the pledging red flag elsewhere)
   Promoters increasing their own stake via open-market purchase is one of
   the highest-conviction signals available, since promoters have the best
   information about their own business and are risking their own capital.
   Distinguish this explicitly from a RISE in promoter % caused by a
   buyback (which mechanically raises promoter % without any promoter
   actually buying anything) — always check if a buyback was concurrent
   before crediting this as a genuine signal.

3. BULK / BLOCK DEALS
   NSE/BSE publish daily bulk deal (>0.5% of equity, on-exchange) and block
   deal (large single trades, negotiated window) reports for free. A
   cluster of buy-side bulk/block deals in a short window is a visible,
   timestamped, checkable event — though the counterparty and their reason
   for trading is not disclosed, so treat volume/direction as signal, not
   the buyer's identity or thesis.

WHAT THIS DOES NOT COVER
--------------------------
Earnings call sentiment, sector tailwind narratives, and order-book/capex
announcements are mentioned in the theory discussion but are NOT
implemented here — they either need judgment, an LLM-based reading pass
over transcripts/announcements, or a maintained policy calendar. Notes on
extending toward those are at the bottom of this file.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date
from abc import ABC, abstractmethod


@dataclass
class ShareholdingSnapshot:
    quarter_end: date
    promoter_pct: float
    promoter_pledge_pct: float
    fii_pct: float
    dii_pct: float
    public_pct: float
    buyback_active: bool = False  # True if a buyback was running this quarter


@dataclass
class BulkDeal:
    trade_date: date
    symbol: str
    client_name: str
    buy_or_sell: str  # "BUY" or "SELL"
    quantity: int
    price: float
    deal_type: str = "bulk"  # "bulk" or "block"


@dataclass
class CatalystTag:
    symbol: str
    fii_trend_quarters_rising: int
    dii_trend_quarters_rising: int
    fii_pct_change: float  # total change over the lookback window
    dii_pct_change: float
    promoter_buying_detected: bool
    promoter_pct_change: float
    recent_bulk_deals_count: int
    recent_bulk_deals_net_buy_qty: int
    recent_block_deals_count: int
    summary: str


def fii_dii_trend(
    snapshots: List[ShareholdingSnapshot],
    lookback_quarters: int = 3,
) -> dict:
    """
    snapshots must be sorted ascending by quarter_end, most recent last.
    Returns consecutive-rising-quarter counts and net % change over the
    lookback window for both FII and DII holding.
    """
    if len(snapshots) < 2:
        return {
            "fii_quarters_rising": 0, "dii_quarters_rising": 0,
            "fii_pct_change": 0.0, "dii_pct_change": 0.0,
        }

    window = snapshots[-lookback_quarters:] if len(snapshots) >= lookback_quarters else snapshots

    def _consecutive_rising(vals: List[float]) -> int:
        count = 0
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] > vals[i - 1]:
                count += 1
            else:
                break
        return count

    fii_vals = [s.fii_pct for s in window]
    dii_vals = [s.dii_pct for s in window]

    return {
        "fii_quarters_rising": _consecutive_rising(fii_vals),
        "dii_quarters_rising": _consecutive_rising(dii_vals),
        "fii_pct_change": fii_vals[-1] - fii_vals[0],
        "dii_pct_change": dii_vals[-1] - dii_vals[0],
    }


def promoter_buying_signal(snapshots: List[ShareholdingSnapshot]) -> dict:
    """
    Flags genuine promoter buying: promoter % rose AND pledge % did not
    also rise proportionally AND no buyback was running (a buyback
    mechanically inflates promoter % without any actual promoter purchase,
    since it reduces the denominator of total shares outstanding).
    """
    if len(snapshots) < 2:
        return {"promoter_buying_detected": False, "promoter_pct_change": 0.0}

    latest, prior = snapshots[-1], snapshots[-2]
    promoter_change = latest.promoter_pct - prior.promoter_pct

    genuine_buying = (
        promoter_change > 0.1  # meaningful increase, not rounding noise
        and not latest.buyback_active
        and not prior.buyback_active
        and latest.promoter_pledge_pct <= prior.promoter_pledge_pct  # not pledge-driven optics
    )

    return {"promoter_buying_detected": genuine_buying, "promoter_pct_change": promoter_change}


def bulk_block_deal_signal(
    deals: List[BulkDeal],
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
) -> dict:
    """
    Summarizes recent bulk/block deal activity for a symbol: counts by type
    and net buy quantity (positive = net buying, negative = net selling)
    for bulk deals specifically (block deals are reported separately since
    they're single large negotiated trades, less meaningful to net out).
    """
    relevant = [
        d for d in deals
        if d.symbol == symbol and (as_of_date - d.trade_date).days <= lookback_days
    ]

    bulk = [d for d in relevant if d.deal_type == "bulk"]
    block = [d for d in relevant if d.deal_type == "block"]

    net_buy_qty = sum(d.quantity if d.buy_or_sell == "BUY" else -d.quantity for d in bulk)

    return {
        "recent_bulk_deals_count": len(bulk),
        "recent_bulk_deals_net_buy_qty": net_buy_qty,
        "recent_block_deals_count": len(block),
    }


def build_catalyst_tag(
    symbol: str,
    snapshots: List[ShareholdingSnapshot],
    deals: List[BulkDeal],
    as_of_date: date,
    lookback_quarters: int = 3,
    lookback_days: int = 30,
) -> CatalystTag:
    """Combines all catalyst signals into a single tag for display alongside
    an engine.ScanResult. Does not gate/filter — purely informational."""
    fd = fii_dii_trend(snapshots, lookback_quarters)
    pb = promoter_buying_signal(snapshots)
    bd = bulk_block_deal_signal(deals, symbol, as_of_date, lookback_days)

    parts = []
    if fd["fii_quarters_rising"] >= 2:
        parts.append(f"FII holding rising {fd['fii_quarters_rising']}q (+{fd['fii_pct_change']:.1f}pp)")
    if fd["dii_quarters_rising"] >= 2:
        parts.append(f"DII holding rising {fd['dii_quarters_rising']}q (+{fd['dii_pct_change']:.1f}pp)")
    if pb["promoter_buying_detected"]:
        parts.append(f"promoter buying (+{pb['promoter_pct_change']:.2f}pp)")
    if bd["recent_bulk_deals_count"] > 0:
        direction = "net buy" if bd["recent_bulk_deals_net_buy_qty"] > 0 else "net sell"
        parts.append(f"{bd['recent_bulk_deals_count']} bulk deals, {direction}")
    if bd["recent_block_deals_count"] > 0:
        parts.append(f"{bd['recent_block_deals_count']} block deals")

    summary = "; ".join(parts) if parts else "No notable catalyst signals in window"

    return CatalystTag(
        symbol=symbol,
        fii_trend_quarters_rising=fd["fii_quarters_rising"],
        dii_trend_quarters_rising=fd["dii_quarters_rising"],
        fii_pct_change=fd["fii_pct_change"],
        dii_pct_change=fd["dii_pct_change"],
        promoter_buying_detected=pb["promoter_buying_detected"],
        promoter_pct_change=pb["promoter_pct_change"],
        recent_bulk_deals_count=bd["recent_bulk_deals_count"],
        recent_bulk_deals_net_buy_qty=bd["recent_bulk_deals_net_buy_qty"],
        recent_block_deals_count=bd["recent_block_deals_count"],
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Data source interface — mirrors knowledge_base.py's pattern.
# ---------------------------------------------------------------------------
class CatalystDataProvider(ABC):
    @abstractmethod
    def get_shareholding_history(self, symbol: str) -> List[ShareholdingSnapshot]:
        ...

    @abstractmethod
    def get_recent_bulk_block_deals(self, symbol: str, as_of_date: date, lookback_days: int) -> List[BulkDeal]:
        ...


class StaticCatalystProvider(CatalystDataProvider):
    """Manually populated, same philosophy as StaticKnowledgeBase — start
    here, automate later. Populate via upsert_shareholding / upsert_deal."""

    def __init__(self):
        self._shareholding: dict = {}
        self._deals: List[BulkDeal] = []

    def upsert_shareholding(self, symbol: str, snapshot: ShareholdingSnapshot):
        self._shareholding.setdefault(symbol.upper(), []).append(snapshot)
        self._shareholding[symbol.upper()].sort(key=lambda s: s.quarter_end)

    def add_deal(self, deal: BulkDeal):
        self._deals.append(deal)

    def get_shareholding_history(self, symbol: str) -> List[ShareholdingSnapshot]:
        return self._shareholding.get(symbol.upper(), [])

    def get_recent_bulk_block_deals(self, symbol: str, as_of_date: date, lookback_days: int = 30) -> List[BulkDeal]:
        return [
            d for d in self._deals
            if d.symbol.upper() == symbol.upper() and (as_of_date - d.trade_date).days <= lookback_days
        ]


# ---------------------------------------------------------------------------
# Notes on wiring a live feed later (not implemented here):
#
#   pip install nse
#   from nse import NSE
#   with NSE(download_folder="./") as nse_client:
#       deals = nse_client.blockDeals()   # and equivalent bulk deals method
#       # shareholding pattern is fetched per-symbol via NSE's corporate
#       # filings; check the `nse` package docs for the current method name,
#       # it's evolved across versions (v3.x at time of writing).
#
#   NSE rate-limits to ~3 req/sec and blocks known cloud IPs (AWS/GCP) — run
#   this from a local/residential connection, ideally after market hours,
#   and cache results rather than re-fetching on every scan.
# ---------------------------------------------------------------------------
