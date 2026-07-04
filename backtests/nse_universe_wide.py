"""
nse_universe_wide.py — Widened NSE universe (~2x) to test whether more breadth
strengthens momentum. Adds a broad batch of liquid-ish mid/small-caps beyond
the original large+mid 104. Deliberately reaches DOWN the cap spectrum (which
is where momentum is academically strongest) — precisely so we can measure
whether the extra alpha is real signal or just survivorship + un-capturable
small-cap liquidity. Fetch filters out any that fail on Yahoo.
"""
from quant_engine.backtests.nse_universe import UNIVERSE as CORE

EXTRA = [
    # rails / defence / PSU capex
    "IRCTC", "IRFC", "RVNL", "HAL", "MAZDOCK", "COCHINSHIP", "BEML", "BDL",
    "RECLTD", "PFC", "IREDA", "NHPC", "SJVN", "NLCINDIA", "OIL", "HINDPETRO",
    "ADANIPOWER", "JSWENERGY", "TORNTPOWER", "CESC", "SUZLON", "INOXWIND", "THERMAX",
    # capital goods / industrials
    "KEC", "ELGIEQUIP", "GRINDWELL", "SKFINDIA", "TIMKEN", "SCHAEFFLER", "KSB",
    "CARBORUNIV", "AIAENG", "APLAPOLLO", "RATNAMANI", "WELCORP", "JINDALSAW",
    # tech / new-age
    "KPITTECH", "TATATECH", "LTTS", "TATAELXSI", "SONACOMS", "UNOMINDA", "MOTHERSON",
    "EXIDEIND", "DIXON", "AMBER", "KAYNES", "SYRMA", "POLICYBZR", "PAYTM",
    "DELHIVERY", "IEX", "ANGELONE", "CDSL", "CAMS", "KFINTECH", "JIOFIN", "BSE",
    # financials
    "BAJAJHLDNG", "SUNDARMFIN", "MANAPPURAM", "TATACOMM", "INDUSTOWER", "IGL", "MGL",
    # metals / materials
    "HINDZINC", "NATIONALUM", "GRAVITA", "KALYANKJIL",
    # consumer / retail
    "VBL", "UNITDSPR", "RADICO", "ABFRL", "KPRMILL", "TRIDENT", "VMART", "BATAINDIA",
    "RELAXO", "METROBRAND", "CAMPUS", "KANSAINER", "BERGEPAINT", "WHIRLPOOL",
    "BLUESTARCO", "CROMPTON", "VGUARD", "KEI", "FINCABLES",
    # pharma / healthcare
    "LAURUSLABS", "GLENMARK", "ALKEM", "IPCALAB", "JBCHEPHARM", "MANKIND", "ZYDUSLIFE",
    "ABBOTINDIA", "NATCOPHARM", "GRANULES", "AJANTPHARM", "SYNGENE", "METROPOLIS",
    "FORTIS", "MAXHEALTH", "LAURUSLABS", "ERIS",
]

_seen = set()
WIDE = []
for s in list(CORE) + EXTRA:
    if s not in _seen:
        _seen.add(s)
        WIDE.append(s)
