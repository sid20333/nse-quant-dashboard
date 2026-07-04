"""
costs.py — Realistic Indian equity DELIVERY (CNC) transaction cost model.

Replaces the single hand-waved COST_PCT=0.1% used in the backtests, which
under-counts real friction. On NSE cash-delivery the statutory charges are
ASYMMETRIC (buy vs sell differ) and STT alone is 0.1% PER SIDE — i.e. 0.2%
round-trip before anything else. At ~38 round-trips per quarter that is not
negligible, so it deserves an explicit, component-by-component model rather
than a guessed lump sum.

Components (fractions of trade turnover unless noted), delivery segment:
  - Brokerage        : broker-dependent. Discount brokers (Zerodha, etc.)
                       charge ZERO on delivery, so default = 0. Set
                       brokerage_pct to model a full-service broker.
  - STT              : 0.10% on BUY and 0.10% on SELL (delivery).
  - Exchange txn     : ~0.00297% (NSE) both sides.
  - SEBI turnover    : 0.0001% (Rs 10 / crore) both sides.
  - Stamp duty       : 0.015% on BUY only.
  - GST              : 18% on (brokerage + exchange + SEBI).
  - DP charge        : flat ~Rs 15.5 per scrip on SELL only (CDSL + broker).

SLIPPAGE is modelled separately, in the backtest, by nudging the fill price
(buys fill slightly above, sells slightly below the reference bar) — it is an
execution-quality assumption, not a statutory charge, so it does not belong
in this module.

Everything here is a documented estimate for mid/large-cap delivery as of
2026; tweak the parameters, don't bury new numbers in the backtest.
"""

from dataclasses import dataclass


@dataclass
class IndianEquityCosts:
    brokerage_pct: float = 0.0        # 0 = discount-broker delivery
    stt_pct: float = 0.001            # 0.10% each side
    exchange_pct: float = 0.0000297   # NSE ~0.00297%
    sebi_pct: float = 0.000001        # Rs 10 / crore
    stamp_pct_buy: float = 0.00015    # 0.015% buy only
    gst_rate: float = 0.18            # on brokerage + exchange + sebi
    dp_charge_flat: float = 15.5      # per scrip, sell only

    def _gst(self, turnover: float) -> float:
        taxable = (self.brokerage_pct + self.exchange_pct + self.sebi_pct) * turnover
        return self.gst_rate * taxable

    def buy_cost(self, turnover: float) -> float:
        """Total buy-side cost in rupees for a given trade turnover (shares*price)."""
        pct = (self.brokerage_pct + self.stt_pct + self.exchange_pct
               + self.sebi_pct + self.stamp_pct_buy)
        return pct * turnover + self._gst(turnover)

    def sell_cost(self, turnover: float) -> float:
        """Total sell-side cost in rupees for a given trade turnover (shares*price)."""
        pct = (self.brokerage_pct + self.stt_pct + self.exchange_pct + self.sebi_pct)
        return pct * turnover + self._gst(turnover) + self.dp_charge_flat

    def round_trip_pct(self, turnover: float = 1.0) -> float:
        """Approx round-trip cost as a fraction of turnover (excludes slippage).
        DP flat charge is amortised over `turnover`, so pass a realistic
        per-trade value (e.g. capital / positions) for a representative number."""
        return (self.buy_cost(turnover) + self.sell_cost(turnover)) / turnover
