"""
backtest.py — Walk-forward backtester for the recommendation engine.

WHY WALK-FORWARD, NOT A SINGLE IN-SAMPLE BACKTEST
--------------------------------------------------
If you tune your thresholds (margin of safety %, squeeze percentile, etc.)
and then test them on the SAME historical period you tuned them on, your
results are meaningless — you've just fit noise. Walk-forward testing
simulates only ever using information that would have been available at
the time:

  - At each weekly decision point (Saturday, per your spec), the engine
    only sees OHLCV data up to and including that point. No future bars
    leak into indicator calculations (this is enforced by literally
    slicing the DataFrame, not just "starting the loop late").
  - Entries are simulated at the NEXT bar's open (Monday), not the
    signal bar's close — you can't actually transact at Saturday's price.
  - A fixed holding period (or a stop/target exit) is enforced, with
    trades marked-to-market only using data at/after entry.

METRICS REPORTED
-----------------
  - CAGR (compound annual growth rate of the equity curve)
  - Sharpe ratio (using weekly returns, annualized)
  - Max drawdown
  - Win rate and average win/loss
  - Number of trades (small sample sizes should be treated with real
    suspicion — Indian mid/small caps clearing 5 simultaneous filters
    may simply not produce enough trades per year for the statistics
    to be reliable)

WHAT THIS DOES NOT DO
----------------------
  - Model realistic slippage/impact for less liquid names beyond a simple
    configurable slippage_pct - for real testing, model this per-stock
    based on average daily traded value.
  - Account for STT, brokerage, or taxes (add via `cost_pct_per_trade`).
  - Handle survivorship bias in your stock universe - if you backtest only
    on today's Nifty 500 constituents, you're silently excluding companies
    that got delisted/went bankrupt, which inflates results. Use a
    point-in-time constituent list if you can source one.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional
import pandas as pd
import numpy as np


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    num_trades: int


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def _sharpe_from_returns(returns: pd.Series, periods_per_year: int = 52) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def walk_forward_backtest(
    symbol_dfs: dict,  # {symbol: full ohlcv DataFrame with 'date' column}
    signal_fn: Callable[[str, pd.DataFrame], bool],
    start_date: str,
    end_date: str,
    holding_period_days: int = 15,
    stop_loss_pct: Optional[float] = 0.08,
    take_profit_pct: Optional[float] = None,
    cost_pct_per_trade: float = 0.001,
    slippage_pct: float = 0.0015,
    initial_capital: float = 1_000_000.0,
    max_concurrent_positions: int = 10,
) -> BacktestResult:
    """
    signal_fn(symbol, sliced_df) -> bool: should return True if the stock
    qualifies as a BUY at the last row of `sliced_df` (which contains ONLY
    data up to and including the current decision date - this is your
    look-ahead-bias firewall. Wire this to `engine.evaluate_stock` /
    `run_weekly_scan` by wrapping it, e.g.:

        def signal_fn(symbol, df):
            inputs = build_stock_inputs(symbol, df)  # your own mapping fn
            result = evaluate_stock(inputs, kb)
            return result.fully_qualified

    Decision dates are every Saturday between start_date and end_date.
    Entries fill at the next available bar's open. Exits: whichever of
    (stop_loss_pct, take_profit_pct, holding_period_days) triggers first.
    """
    decision_dates = pd.date_range(start_date, end_date, freq="W-SAT")

    open_positions = []  # list of dicts: symbol, entry_date, entry_price, exit_target_date
    closed_trades: List[Trade] = []
    cash = initial_capital
    equity_curve = []
    equity_dates = []

    for decision_date in decision_dates:
        # --- Step 1: manage exits for open positions ---
        still_open = []
        for pos in open_positions:
            df = symbol_dfs[pos["symbol"]]
            future_bars = df[(df["date"] > pos["entry_date"]) & (df["date"] <= decision_date)]
            exited = False
            for _, bar in future_bars.iterrows():
                ret_if_low = (bar["low"] - pos["entry_price"]) / pos["entry_price"]
                ret_if_high = (bar["high"] - pos["entry_price"]) / pos["entry_price"]

                if stop_loss_pct is not None and ret_if_low <= -stop_loss_pct:
                    exit_price = pos["entry_price"] * (1 - stop_loss_pct)
                    _close_trade(pos, bar["date"], exit_price, "stop_loss", closed_trades, cost_pct_per_trade)
                    cash += pos["shares"] * exit_price * (1 - cost_pct_per_trade)
                    exited = True
                    break
                if take_profit_pct is not None and ret_if_high >= take_profit_pct:
                    exit_price = pos["entry_price"] * (1 + take_profit_pct)
                    _close_trade(pos, bar["date"], exit_price, "take_profit", closed_trades, cost_pct_per_trade)
                    cash += pos["shares"] * exit_price * (1 - cost_pct_per_trade)
                    exited = True
                    break
                if (bar["date"] - pos["entry_date"]).days >= holding_period_days:
                    exit_price = bar["close"]
                    _close_trade(pos, bar["date"], exit_price, "time_exit", closed_trades, cost_pct_per_trade)
                    cash += pos["shares"] * exit_price * (1 - cost_pct_per_trade)
                    exited = True
                    break
            if not exited:
                still_open.append(pos)
        open_positions = still_open

        # --- Step 2: scan for new signals (only using data up to decision_date) ---
        if len(open_positions) < max_concurrent_positions:
            for symbol, df in symbol_dfs.items():
                if any(p["symbol"] == symbol for p in open_positions):
                    continue
                sliced = df[df["date"] <= decision_date].reset_index(drop=True)
                if len(sliced) < 120:  # need enough history for 100-day indicators
                    continue
                try:
                    qualifies = signal_fn(symbol, sliced)
                except Exception:
                    continue
                if qualifies:
                    entry_bars = df[df["date"] > decision_date]
                    if entry_bars.empty:
                        continue
                    entry_bar = entry_bars.iloc[0]
                    entry_price = entry_bar["open"] * (1 + slippage_pct)
                    position_value = cash / max_concurrent_positions
                    shares = position_value / entry_price
                    cash -= shares * entry_price * (1 + cost_pct_per_trade)
                    open_positions.append(
                        {
                            "symbol": symbol,
                            "entry_date": entry_bar["date"],
                            "entry_price": entry_price,
                            "shares": shares,
                        }
                    )
                    if len(open_positions) >= max_concurrent_positions:
                        break

        # --- Step 3: mark-to-market equity ---
        mtm = cash
        for pos in open_positions:
            df = symbol_dfs[pos["symbol"]]
            recent = df[df["date"] <= decision_date]
            if not recent.empty:
                mtm += pos["shares"] * recent["close"].iloc[-1]
        equity_curve.append(mtm)
        equity_dates.append(decision_date)

    equity = pd.Series(equity_curve, index=equity_dates)
    weekly_returns = equity.pct_change().dropna()

    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[0] > 0 else 0.0

    wins = [t.return_pct for t in closed_trades if t.return_pct > 0]
    losses = [t.return_pct for t in closed_trades if t.return_pct <= 0]

    return BacktestResult(
        trades=closed_trades,
        equity_curve=equity,
        cagr=cagr,
        sharpe=_sharpe_from_returns(weekly_returns),
        max_drawdown=_max_drawdown(equity),
        win_rate=len(wins) / len(closed_trades) if closed_trades else 0.0,
        avg_win_pct=float(np.mean(wins)) if wins else 0.0,
        avg_loss_pct=float(np.mean(losses)) if losses else 0.0,
        num_trades=len(closed_trades),
    )


def _close_trade(pos, exit_date, exit_price, reason, closed_trades, cost_pct_per_trade):
    ret = (exit_price * (1 - cost_pct_per_trade) - pos["entry_price"]) / pos["entry_price"]
    closed_trades.append(
        Trade(
            symbol=pos["symbol"],
            entry_date=pos["entry_date"],
            entry_price=pos["entry_price"],
            exit_date=exit_date,
            exit_price=exit_price,
            return_pct=ret,
            exit_reason=reason,
        )
    )
