"""
data_provider.py — Market data interface.

Two implementations:
  1. BreezeDataProvider: thin wrapper around your existing ICICI Direct
     Breeze Connect setup (reuses the same auth pattern as your portfolio
     tracker) to fetch historical daily OHLCV for a given symbol.
  2. SyntheticDataProvider: generates realistic-ish fake OHLCV so every
     other module in this engine can be tested end-to-end without needing
     live market access. This is what the test script uses.

Both implementations return a pandas DataFrame with columns:
    date, open, high, low, close, volume
sorted ascending by date, with a plain RangeIndex (0..n-1) so the
integer-position logic in zones.py / vwap.py works directly.
"""

import os
import pickle
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional


class DataProvider(ABC):
    @abstractmethod
    def get_daily_ohlcv(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        ...


class YFinanceDataProvider(DataProvider):
    """
    Real NSE daily OHLCV via the `yfinance` library (Yahoo Finance).

    Yahoo blocks *server-side / robots-respecting* fetches, which is why the
    original design conversation (Claude-in-chat, no network) could only use
    SyntheticDataProvider. Run from a real local Python process with a normal
    user-agent, `yfinance` works — verify this is still true when you run.

    NSE symbols take the ".NS" Yahoo suffix (e.g. "RELIANCE" -> "RELIANCE.NS").
    Pass either the bare NSE code or the full ".NS" ticker; this class appends
    ".NS" if no suffix is present. Yahoo index tickers (e.g. "^CNX100",
    "^NSEI") are passed through unchanged.

    Prices are SPLIT- AND DIVIDEND-ADJUSTED (auto_adjust=True). This is
    deliberate: an unadjusted series shows a fake ~50% one-day "crash" on a
    2:1 split date, which would fire a spurious death-cross/stop exit in the
    backtest. Volume is left unadjusted by Yahoo.

    A small on-disk pickle cache (one file per symbol, keyed by the full date
    span requested) avoids re-hitting Yahoo across repeated backtest runs.
    Delete the cache dir to force a fresh pull.

    KNOWN LIMITATION — survivorship: Yahoo only serves symbols that still
    trade under their current name. Anything delisted/renamed during your
    window is silently absent, so a universe built from *today's* index
    constituents is survivorship-biased upward. This is a property of the
    data source, not a bug here — caveat any result accordingly.
    """

    def __init__(self, cache_dir: Optional[str] = None, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _to_yahoo(symbol: str) -> str:
        if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
            return symbol
        return f"{symbol}.NS"

    def _cache_path(self, symbol: str, from_date: str, to_date: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe = symbol.replace("^", "IDX_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}__{from_date}__{to_date}__adj{int(self.auto_adjust)}.pkl")

    def get_daily_ohlcv(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, from_date, to_date)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        import yfinance as yf

        ticker = self._to_yahoo(symbol)
        # yfinance's `end` is exclusive; add a day so to_date is included.
        end_inclusive = (pd.to_datetime(to_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(
            ticker,
            start=from_date,
            end=end_inclusive,
            progress=False,
            auto_adjust=self.auto_adjust,
        )

        if raw is None or len(raw) == 0:
            raise ValueError(f"No data returned for {ticker} between {from_date} and {to_date}")

        # yfinance returns a MultiIndex column frame ((field, ticker)) even for
        # a single ticker; collapse it to plain field names.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index().rename(
            columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "open", "high", "low", "close", "volume"]]

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
        return df


class BreezeDataProvider(DataProvider):
    """
    Wraps breeze_connect.BreezeConnect, same as your Breeze API portfolio
    tracker. Requires BREEZE_API_KEY, BREEZE_API_SECRET and a generated
    session token (Breeze sessions expire daily and need re-login via the
    ICICI Direct portal each morning).

    Usage:
        from breeze_connect import BreezeConnect
        breeze = BreezeConnect(api_key="...")
        breeze.generate_session(api_secret="...", session_token="...")
        provider = BreezeDataProvider(breeze)
        df = provider.get_daily_ohlcv("RELIANCE", "2023-01-01", "2026-06-30")
    """

    def __init__(self, breeze_client):
        self.breeze = breeze_client

    def get_daily_ohlcv(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        resp = self.breeze.get_historical_data_v2(
            interval="1day",
            from_date=f"{from_date}T00:00:00.000Z",
            to_date=f"{to_date}T00:00:00.000Z",
            stock_code=symbol,
            exchange_code="NSE",
            product_type="cash",
        )
        rows = resp.get("Success", [])
        if not rows:
            raise ValueError(f"No data returned for {symbol} between {from_date} and {to_date}")

        df = pd.DataFrame(rows)
        df = df.rename(
            columns={
                "datetime": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df = df.sort_values("date").reset_index(drop=True)
        return df[["date", "open", "high", "low", "close", "volume"]]


class SyntheticDataProvider(DataProvider):
    """
    Generates a synthetic OHLCV series with:
      - a random-walk trend component
      - a few injected volatility "squeeze then breakout" regimes so the
        technical.py squeeze detector has something real to find
      - a few injected "drop-base-rally" patterns so zones.py has order
        blocks to find
      - volume that spikes on breakout days

    This is for TESTING THE PIPELINE ONLY — it validates that your code
    runs end-to-end without errors and produces sane outputs. It proves
    nothing about whether the strategy is profitable; that requires real
    historical data and the backtest module.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_daily_ohlcv(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        start = pd.to_datetime(from_date)
        end = pd.to_datetime(to_date)
        dates = pd.bdate_range(start, end)
        n = len(dates)

        price = 500.0
        closes = []
        vols = []
        base_vol = 200_000

        i = 0
        while i < n:
            # Randomly decide regime: 70% normal drift, then occasionally
            # inject a squeeze (low vol for ~15 bars) followed by a breakout,
            # or a drop-base-rally pattern.
            regime_roll = self.rng.random()

            if regime_roll < 0.04 and i < n - 25:
                # Inject drop-base-rally + squeeze breakout combo
                drop_len = 5
                for _ in range(drop_len):
                    price *= 1 - abs(self.rng.normal(0.015, 0.005))
                    closes.append(price)
                    vols.append(base_vol * self.rng.uniform(0.8, 1.3))
                base_len = 8
                base_center = price
                for _ in range(base_len):
                    price = base_center * (1 + self.rng.normal(0, 0.006))
                    closes.append(price)
                    vols.append(base_vol * self.rng.uniform(0.6, 0.9))
                rally_len = 4
                for _ in range(rally_len):
                    price *= 1 + abs(self.rng.normal(0.02, 0.008))
                    closes.append(price)
                    vols.append(base_vol * self.rng.uniform(1.6, 2.4))
                i += drop_len + base_len + rally_len
            else:
                price *= 1 + self.rng.normal(0.0003, 0.014)
                closes.append(price)
                vols.append(base_vol * self.rng.uniform(0.7, 1.3))
                i += 1

        closes = np.array(closes[:n])
        vols = np.array(vols[:n])

        # Build OHLC around the close series with small realistic wicks
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) * (1 + np.abs(self.rng.normal(0, 0.004, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(self.rng.normal(0, 0.004, n)))

        df = pd.DataFrame(
            {
                "date": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": vols.astype(int),
            }
        )
        return df.reset_index(drop=True)
