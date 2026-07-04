"""
knowledge_base.py — The qualitative "red flag" filter (KB).

THEORY
------
A stock can look statistically cheap (high margin of safety) purely because
the market has correctly priced in deteriorating fundamentals — a "value
trap." The KB's job is to catch reasons the market might be right to be
pessimistic, BEFORE the valuation/technical layers get a vote.

This module is an INTERFACE, not a working scraper — I don't have live
access to NSE/BSE filing feeds from this environment, and scraping them
reliably (rate limits, session cookies, anti-bot measures) is genuinely
fiddly. What's implemented:

  1. A clean, typed interface (`KnowledgeBase` ABC) so `engine.py` doesn't
     care whether red flags come from NSE/BSE filings, screener.in exports,
     a manually maintained CSV, or an LLM call over recent news.
  2. A `StaticKnowledgeBase` you can populate today from a CSV you maintain
     by hand or export from screener.in / Tickertape, so the rest of the
     engine is usable immediately.
  3. Notes on how to wire up a real feed later (nsepython for NSE
     announcements, BSE's own announcement API, or an LLM-based news
     sentiment pass using the Claude API with web search enabled — see the
     project's `anthropic_api_in_artifacts` pattern for that if you build
     a dashboard version of this).

RED FLAG DEFINITIONS (what "disqualify regardless of price" means here):
  - promoter_pledge_pct above a threshold (promoters using shares as loan
    collateral -> forced-selling risk if the stock falls)
    -> equity pledging data is disclosed quarterly in NSE/BSE shareholding
       pattern filings.
  - yoy_earnings_decline_pct beyond a threshold (deteriorating fundamentals
    the "cheap" price may already reflect)
  - forensic_audit_flag / auditor_resignation_flag (governance risk that a
    valuation model cannot price)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import csv


@dataclass
class RedFlagResult:
    disqualified: bool
    reasons: list


class KnowledgeBase(ABC):
    @abstractmethod
    def get_red_flags(self, symbol: str) -> RedFlagResult:
        ...

    @abstractmethod
    def get_last_earnings_date(self, symbol: str) -> Optional[str]:
        ...


class StaticKnowledgeBase(KnowledgeBase):
    """
    Populate from a CSV with columns:
        symbol, promoter_pledge_pct, yoy_earnings_decline_pct,
        forensic_audit_flag, auditor_resignation_flag, last_earnings_date

    You can build this CSV manually today (screener.in and Tickertape both
    show promoter pledging % and YoY earnings growth on the stock page), or
    later automate it by pulling the NSE "Shareholding Pattern" and
    "Corporate Announcements" endpoints via nsepython.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        pledge_threshold_pct: float = 20.0,
        earnings_decline_threshold_pct: float = 15.0,
    ):
        self.data: Dict[str, dict] = {}
        self.pledge_threshold_pct = pledge_threshold_pct
        self.earnings_decline_threshold_pct = earnings_decline_threshold_pct
        if csv_path:
            self.load_csv(csv_path)

    def load_csv(self, csv_path: str):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data[row["symbol"].strip().upper()] = row

    def upsert(self, symbol: str, **fields):
        self.data[symbol.upper()] = {"symbol": symbol.upper(), **fields}

    def get_red_flags(self, symbol: str) -> RedFlagResult:
        row = self.data.get(symbol.upper())
        if row is None:
            # No data = cannot clear it. Fail closed, not open — an unknown
            # stock should not silently pass the qualitative filter.
            return RedFlagResult(disqualified=True, reasons=["No KB data available for symbol"])

        reasons = []
        pledge = float(row.get("promoter_pledge_pct", 0) or 0)
        decline = float(row.get("yoy_earnings_decline_pct", 0) or 0)
        forensic = str(row.get("forensic_audit_flag", "")).strip().lower() in ("true", "1", "yes")
        auditor_resign = str(row.get("auditor_resignation_flag", "")).strip().lower() in ("true", "1", "yes")

        if pledge >= self.pledge_threshold_pct:
            reasons.append(f"Promoter pledge {pledge}% >= threshold {self.pledge_threshold_pct}%")
        if decline >= self.earnings_decline_threshold_pct:
            reasons.append(f"YoY earnings decline {decline}% >= threshold {self.earnings_decline_threshold_pct}%")
        if forensic:
            reasons.append("Forensic audit flag active")
        if auditor_resign:
            reasons.append("Auditor resignation flag active")

        return RedFlagResult(disqualified=len(reasons) > 0, reasons=reasons)

    def get_last_earnings_date(self, symbol: str) -> Optional[str]:
        row = self.data.get(symbol.upper())
        if row is None:
            return None
        return row.get("last_earnings_date")


# ---------------------------------------------------------------------------
# Notes for wiring up a live feed later (not implemented here, no network
# access to NSE/BSE from this sandbox):
#
#   pip install nsepython
#   from nsepython import nse_eq, nse_get_advances_declines
#   -> nse_eq(symbol) returns corporate info including recent announcements
#
#   BSE announcements: https://www.bseindia.com/corporates/ann.aspx has an
#   underlying JSON API (inspect network tab) you can poll per-symbol.
#
#   For "forensic audit" / negative-news style flags specifically, a
#   periodic LLM pass (Claude with web search enabled) reading recent
#   news headlines per symbol and classifying red/no-flag is more robust
#   than keyword-matching filings text, and is a good fit for a scheduled
#   job rather than a hot-path check.
# ---------------------------------------------------------------------------
