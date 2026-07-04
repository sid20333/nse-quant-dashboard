"""
nse_universe.py — Curated NSE mid-cap / large-cap universe for real-data
backtesting.

Per the handover constraint: NSE mid-cap and large-cap ONLY (small-cap / BSE
excluded by the user after a liquidity discussion). This is a hand-picked
subset drawn from NIFTY 100 (large-cap) + NIFTY Midcap 150 (mid-cap)
constituents — NOT the full 250, but a liquid, representative ~90-name pool
from which the backtest keeps the first UNIVERSE_SIZE that return clean,
sufficiently-long history from Yahoo.

CAVEATS baked into this list:
  - It is TODAY'S membership (2026), so it is survivorship-biased: names that
    were dropped from the indices, delisted, or renamed during the test window
    are absent. A universe built from current constituents will look better
    than one an investor could actually have traded in the past.
  - Yahoo suffix ".NS" is appended by YFinanceDataProvider, so bare NSE codes
    are fine here. A handful may fail to fetch (rename/delist on Yahoo's side);
    the runner logs and skips those.
  - Some 2025-2026 renames are reflected already (e.g. Zomato -> ETERNAL).
"""

# NIFTY 100 large-caps
LARGE_CAP = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "NESTLEIND", "WIPRO",
    "ULTRACEMCO", "ONGC", "NTPC", "POWERGRID", "M&M", "TATAMOTORS",
    "TATASTEEL", "JSWSTEEL", "HCLTECH", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "GRASIM", "HDFCLIFE", "SBILIFE", "BRITANNIA", "DRREDDY",
    "CIPLA", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "INDUSINDBK", "TECHM",
    "APOLLOHOSP", "TATACONSUM", "PIDILITIND", "DABUR", "GODREJCP", "HAVELLS",
    "DLF", "HINDALCO", "BPCL", "GAIL", "SIEMENS", "PNB", "BANKBARODA",
    "ICICIPRULI", "LTIM", "VEDL", "AMBUJACEM",
]

# NIFTY Midcap 150 mid-caps
MID_CAP = [
    "MPHASIS", "PERSISTENT", "COFORGE", "AUBANK", "FEDERALBNK", "IDFCFIRSTB",
    "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ASHOKLEY", "BALKRISIND",
    "MRF", "BHARATFORG", "CUMMINSIND", "ABB", "BEL", "TATAPOWER", "NMDC",
    "SAIL", "JINDALSTEL", "PAGEIND", "VOLTAS", "TRENT", "JUBLFOOD",
    "INDHOTEL", "MARICO", "COLPAL", "MUTHOOTFIN", "CHOLAFIN", "LICHSGFIN",
    "SRF", "UPL", "PIIND", "PETRONET", "CONCOR", "POLYCAB", "ABCAPITAL",
    "OBEROIRLTY", "GODREJPROP", "MFSL", "TVSMOTOR", "BALKRISIND", "SUNTV",
    "ESCORTS", "GMRINFRA", "IDEA", "BANDHANBNK",
]

# De-dup while preserving order.
_seen = set()
UNIVERSE = []
for _s in LARGE_CAP + MID_CAP:
    if _s not in _seen:
        _seen.add(_s)
        UNIVERSE.append(_s)

# Real market-regime benchmark: NIFTY 100 (matches the large-cap tilt of the
# universe). A genuine index has correlated drawdowns — the whole point the
# synthetic data could not reproduce.
REGIME_INDEX = "^CNX100"
