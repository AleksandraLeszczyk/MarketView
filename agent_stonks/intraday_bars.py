"""Today's regular-session minute bars, in the shape a rule agent reads.

Shared by the hand-written (non-LLM) traders. Getting bars out of `AppState`
is integration, not strategy, and two agents doing it subtly differently would
show up in SimLab as a strategy difference that isn't one. Indicators stay in
each agent -- that part is what a head-to-head is actually comparing.
"""
from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from .clock import now as _now

ET = ZoneInfo("America/New_York")

# Regular trading hours. The stored/streamed tape runs 04:00-20:00 ET, and no
# session-anchored feature means anything outside it: VWAP anchors at the open,
# and thin pre-market prints would set the volume baseline a breakout or a
# pullback is measured against.
SESSION_START = time(9, 30)
SESSION_END = time(16, 0)

COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def session_frame(sym_state) -> pd.DataFrame:
    """The current session's bars as ts/OHLCV rows, oldest first.

    `AppState` keeps streamed minute bars as Alpaca dicts
    ({"t","o","h","l","c","v"}) in a lock-guarded deque -- the same container
    SimLab fills from the stored tape, so live and replay read identically.

    Only today is returned. Intraday features are session-local, and carrying
    yesterday into a rolling window would price the overnight gap as a
    one-minute move in ATR. Every agent using this wants ~50 bars of *today*
    before it acts anyway, so a shorter frame costs no tradeable bar.
    """
    with sym_state.lock:
        bars = list(sym_state.bars)

    if not bars:
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime([b["t"] for b in bars], utc=True, format="mixed"),
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b.get("v") or 0.0) for b in bars],
        }
    )

    local = frame["ts"].dt.tz_convert(ET)
    session = (
        (local.dt.date == _now().astimezone(ET).date())
        & (local.dt.time >= SESSION_START)
        & (local.dt.time < SESSION_END)
    )
    return frame[session].reset_index(drop=True)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR over ts/OHLCV rows."""
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, re-anchored each session date."""
    local_ts = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert(ET)
    session = local_ts.dt.date
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    price_volume = typical * frame["volume"]
    return price_volume.groupby(session).cumsum() / (
        frame["volume"].groupby(session).cumsum()
    )
