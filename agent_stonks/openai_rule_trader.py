"""TraderByChatGPT -- a rule-based agent with no LLM and no model in the loop.

Written by ChatGPT to the brief the other personalities get, and kept as it was
written: a long-only intraday breakout state machine on AAPL. Where Apple
Trader (`agent_stonks.apple_trader`) asks a saved classifier whether a momentum
change will hold, this one asks nothing -- the entry is a pure boolean over the
bar that just closed, so the same tape always produces the same trades.

The rules
---------
* BUY when one bar clears every condition at once: close above session VWAP,
  EMA9 > EMA21 > EMA50, a close through the highest high of the last 20 bars,
  volume at least 1.4x its 20-bar median, RSI inside 55-75 (weak momentum and
  blow-off chases are both refused), ATR between 0.25% and 2.5% of price, and a
  green bar. Entries are taken between 09:45 and 15:15 only.
* SIZE on risk, not on cash: the initial stop is 1.3 ATR below the entry, and
  the quantity is whatever puts `risk_pct` of equity between the fill and that
  stop -- capped at `max_position_pct` of cash, because a collapsing ATR would
  otherwise ask for an unbounded position.
* SELL on whichever comes first: the ATR stop (which ratchets up to 1.5 ATR
  under the running peak once the trade is 1R ahead), the optional percentage
  trail, a time stop after 60 bars, or the pre-close flatten.

A day that closes 1% down stops opening new trades; the position already on is
still managed to its stop.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from . import market_hours, scoring, observability as obs
from .agent import _log, stop_agent
from .clock import now as _now
from .config import (
    TRADER_BY_CHATGPT_BAR_LAG_SEC,
    TRADER_BY_CHATGPT_CYCLE_SEC,
    TRADER_BY_CHATGPT_FLATTEN_BEFORE_CLOSE_MIN,
    TRADER_BY_CHATGPT_POSITION_PCT,
    TRADER_BY_CHATGPT_TRAIL_PCT,
)
from .decisions import DecisionTracker
from .intraday_bars import ET, atr as _atr, session_frame, session_vwap as _session_vwap
from .state import AppState

TRADER_BY_CHATGPT_KEY = "trader_by_chatgpt"
TRADER_BY_CHATGPT_LABEL = "TraderByChatGPT (rule-based, no LLM)"
TRADER_BY_CHATGPT_AVATAR = "Multiavatar-TraderByChatGPT.png"
RULE_PROVIDER = "rules"
TICKER = "AAPL"


@dataclass(frozen=True)
class TraderByChatGPTConfig:
    """Deterministic strategy/risk controls.

    The strategy is deliberately long-only and single-position.
    """

    # Entry
    breakout_lookback: int = 20
    min_relative_volume: float = 1.40
    rsi_min: float = 55.0
    rsi_max: float = 75.0
    # The volatility regime the entry accepts, as ATR14 over price. These are
    # ONE-MINUTE bars: a liquid large cap runs a median 1-minute ATR near 0.08%
    # of price, two orders of magnitude under the daily figure the same
    # indicator gives. The floor rejects tape too dead for a breakout to travel
    # anywhere; the ceiling is a safety valve against disorderly bars, and on
    # this timeframe it does not bind.
    min_atr_pct: float = 0.0004
    max_atr_pct: float = 0.025
    entry_start: time = time(9, 45)
    entry_stop: time = time(15, 15)

    # Risk / exits
    risk_pct: float = 0.25
    max_position_pct: float = TRADER_BY_CHATGPT_POSITION_PCT
    initial_stop_atr: float = 1.30
    trail_atr: float = 1.50
    trail_activation_r: float = 1.00
    trail_pct: float = TRADER_BY_CHATGPT_TRAIL_PCT

    flatten_before_close_min: int = TRADER_BY_CHATGPT_FLATTEN_BEFORE_CLOSE_MIN

    # Safety valve for a stagnant trade.
    max_bars_in_trade: int = 60


def config_signature(config: Optional[TraderByChatGPTConfig] = None) -> str:
    """Stable identity for SimLab grouping/de-duplication.

    Carries every number a SimLab run can vary, so two runs that differ in one
    of them are two configurations to compare rather than one repeated.
    """
    c = config or TraderByChatGPTConfig()
    return (
        f"trader_by_chatgpt(bl={c.breakout_lookback},rv>={c.min_relative_volume:g},"
        f"rsi={c.rsi_min:g}-{c.rsi_max:g},vol>={c.min_atr_pct * 100:g}%,"
        f"atr={c.initial_stop_atr:g}/"
        f"{c.trail_atr:g},risk={c.risk_pct:g}%,cap={c.max_position_pct:g}%,"
        f"trail={c.trail_pct:g}%)"
    )


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, math.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _feature_frame(
    frame: pd.DataFrame, config: TraderByChatGPTConfig
) -> pd.DataFrame:
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"AAPL bars missing columns: {sorted(missing)}"
        )

    df = frame.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=sorted(required))
        .sort_values("ts")
        .drop_duplicates("ts")
    )
    if df.empty:
        return df

    # Trend
    df["ema9"] = df["close"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    df["ema21"] = df["close"].ewm(
        span=21, adjust=False, min_periods=21
    ).mean()
    df["ema50"] = df["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()

    # Momentum / volatility
    df["rsi14"] = _wilder_rsi(df["close"], 14)
    df["atr14"] = _atr(df, 14)
    df["vwap"] = _session_vwap(df)

    # IMPORTANT: shift(1) makes these baselines use only bars before
    # the bar being evaluated, preventing look-ahead leakage.
    df["prior_high"] = df["high"].shift(1).rolling(
        config.breakout_lookback,
        min_periods=config.breakout_lookback,
    ).max()

    df["volume_median20"] = df["volume"].shift(1).rolling(
        20, min_periods=20
    ).median()

    df["relative_volume"] = (
        df["volume"] / df["volume_median20"]
    )
    df["atr_pct"] = df["atr14"] / df["close"]

    local = df["ts"].dt.tz_convert(ET)
    df["local_time"] = local.dt.time
    bars_per_day = local.dt.date.value_counts()
    df["bars_today"] = local.dt.date.map(bars_per_day)

    return df


class TraderByChatGPT:
    """One deterministic AAPL long-only intraday state machine."""

    def __init__(self, config: Optional[TraderByChatGPTConfig] = None) -> None:
        self.config = config or TraderByChatGPTConfig()
        self.last_bar_ts = None
        self.entry: Optional[dict] = None
        self.starting_equity: Optional[float] = None

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        state: AppState,
        tracker: DecisionTracker,
    ) -> str:
        sym_state = state.sym(TICKER)

        if sym_state is None:
            _log(
                state,
                {
                    "type": "error",
                    "text": "AAPL is not being streamed; nothing to trade.",
                },
            )
            return "no_data"

        if not market_hours.is_market_open():
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        "Market closed -- TraderByChatGPT is not watching bars."
                    ),
                },
            )
            return "closed"

        raw = session_frame(sym_state)
        if raw.empty:
            return "no_data"

        frame = _feature_frame(raw, self.config)
        if frame.empty:
            return "no_data"

        row = frame.iloc[-1]
        read = row.to_dict()
        read["ts"] = row["ts"].to_pydatetime()

        fresh_bar = read["ts"] != self.last_bar_ts
        if fresh_bar:
            self.last_bar_ts = read["ts"]

        position = float(tracker.position_for(TICKER))

        # Adopt a pre-existing long if this process restarted.
        if position > 0 and self.entry is None:
            self.entry = {
                "price": float(read["close"]),
                "peak": float(read["high"]),
                "atr": float(read["atr14"]),
                "bars": 0,
            }
        elif position <= 0:
            self.entry = None

        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1
            self.entry["peak"] = max(
                self.entry["peak"],
                float(read["high"]),
            )

        _log(
            state,
            {
                "type": "analysis",
                "text": self._read_summary(read, position),
            },
        )

        # Daily loss guard: do not open fresh trades after -1%.
        snapshot = tracker.snapshot()
        equity = float(
            snapshot.get("equity", snapshot.get("cash", 0.0))
        )

        if self.starting_equity is None and equity > 0:
            self.starting_equity = equity

        daily_loss_guard = (
            self.starting_equity is not None
            and equity <= self.starting_equity * 0.99
        )

        # Exit always has priority over new entry logic.
        if position > 0:
            reason = self._exit_reason(read)

            if reason is not None:
                self._sell(
                    state,
                    tracker,
                    position,
                    read,
                    reason,
                )
                return "sold"

            return "hold"

        if not fresh_bar:
            return "hold"

        if daily_loss_guard:
            return "hold"

        if self._entry_signal(read):
            if self._closing_soon():
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            "AAPL entry signal ignored inside the "
                            "closing flatten window."
                        ),
                    },
                )
                return "hold"

            return (
                "bought"
                if self._buy(state, tracker, read)
                else "hold"
            )

        return (
            "warming_up"
            if self._warming_up(read)
            else "hold"
        )

    # ------------------------------------------------------------------
    # Entry rules
    # ------------------------------------------------------------------

    def _warming_up(self, read: dict) -> bool:
        needed = max(
            50,
            self.config.breakout_lookback + 20,
        )

        return (
            pd.isna(read.get("ema50"))
            or pd.isna(read.get("prior_high"))
            or pd.isna(read.get("relative_volume"))
            or int(read.get("bars_today", 0)) < needed
        )

    def _entry_signal(self, read: dict) -> bool:
        """One pure boolean predicate on the just-closed bar.

        No LLM/model score, no confirmation window and no future bars.
        The predicate deliberately asks for confluence rather than predicting
        direction from a single indicator.
        """
        if self._warming_up(read):
            return False

        local_time = read["local_time"]

        if (
            local_time < self.config.entry_start
            or local_time >= self.config.entry_stop
        ):
            return False

        values = (
            read["close"],
            read["vwap"],
            read["ema9"],
            read["ema21"],
            read["ema50"],
            read["prior_high"],
            read["relative_volume"],
            read["rsi14"],
            read["atr_pct"],
        )

        if any(pd.isna(v) for v in values):
            return False

        return (
            # Price regime: buyers are above session value.
            read["close"] > read["vwap"]
            # Trend: short > medium > long EMA.
            and read["ema9"] > read["ema21"] > read["ema50"]
            # Breakout: current bar closes through prior resistance.
            and read["close"] > read["prior_high"]
            # Participation: breakout must have unusual volume.
            and read["relative_volume"]
            >= self.config.min_relative_volume
            # Avoid weak momentum and extreme chase conditions.
            and self.config.rsi_min
            <= read["rsi14"]
            <= self.config.rsi_max
            # Avoid dead and disorderly volatility regimes.
            and self.config.min_atr_pct
            <= read["atr_pct"]
            <= self.config.max_atr_pct
            # Require bullish close.
            and read["close"] > read["open"]
        )

    # ------------------------------------------------------------------
    # Exit rules
    # ------------------------------------------------------------------

    def _stop_price(self) -> float:
        if not self.entry:
            return 0.0

        entry_price = float(self.entry["price"])
        peak = float(self.entry["peak"])
        entry_atr = float(self.entry["atr"])

        initial_stop = (
            entry_price
            - self.config.initial_stop_atr * entry_atr
        )

        trailing_stop = (
            peak - self.config.trail_atr * entry_atr
        )

        # Trail activates only after the trade earns 1R.
        if peak >= (
            entry_price
            + self.config.trail_activation_r * entry_atr
        ):
            return max(initial_stop, trailing_stop)

        return initial_stop

    def _exit_reason(self, read: dict) -> Optional[str]:
        if not self.entry:
            return None

        entry_price = float(self.entry["price"])
        peak = float(self.entry["peak"])
        bars = int(self.entry["bars"])
        price = float(read["close"])
        stop = self._stop_price()

        pnl_pct = (
            (price / entry_price - 1.0) * 100.0
        )

        if price <= stop:
            return (
                f"ATR stop: ${price:,.2f} <= ${stop:,.2f}; "
                f"entry ${entry_price:,.2f}, "
                f"peak ${peak:,.2f}, "
                f"P/L {pnl_pct:+.2f}%."
            )

        # Optional percentage safety stop from the running peak, if the
        # repository config sets one. This can only tighten the ATR trail.
        if self.config.trail_pct > 0:
            drawdown_pct = (
                (price / peak - 1.0) * 100.0
            )

            if drawdown_pct <= -abs(
                self.config.trail_pct
            ):
                return (
                    f"Trailing stop: ${price:,.2f} is "
                    f"{drawdown_pct:+.2f}% from "
                    f"${peak:,.2f}; P/L "
                    f"{pnl_pct:+.2f}%."
                )

        if bars >= self.config.max_bars_in_trade:
            return (
                f"Time stop after {bars} bars; "
                f"P/L {pnl_pct:+.2f}%."
            )

        if self._closing_soon():
            to_close = market_hours.seconds_to_close() or 0.0

            return (
                f"Session ends in {to_close / 60:.0f} min; "
                f"flattening AAPL instead of carrying the "
                f"intraday signal overnight "
                f"({pnl_pct:+.2f}%)."
            )

        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _buy(
        self,
        state: AppState,
        tracker: DecisionTracker,
        read: dict,
    ) -> bool:
        price = float(read["close"])
        atr = float(read["atr14"])

        if price <= 0 or atr <= 0:
            return False

        snapshot = tracker.snapshot()
        cash = float(snapshot.get("cash", 0.0))
        equity = float(snapshot.get("equity", cash))

        # Risk-budget sizing:
        # dollars-at-risk = equity * risk_pct.
        risk_budget = (
            equity * self.config.risk_pct / 100.0
        )

        stop_distance = (
            self.config.initial_stop_atr * atr
        )

        risk_qty = (
            risk_budget / stop_distance
        )

        # Hard notional cap protects against tiny ATR values.
        cap_qty = (
            cash
            * self.config.max_position_pct
            / 100.0
            / price
        )

        quantity = (
            math.floor(
                min(risk_qty, cap_qty) * 1e4
            ) / 1e4
        )

        if quantity <= 0:
            return False

        initial_stop = price - stop_distance

        reasoning = (
            f"BUY AAPL: close ${price:,.2f} above VWAP "
            f"with EMA9>EMA21>EMA50, "
            f"{self.config.breakout_lookback}-bar breakout, "
            f"relative volume {read['relative_volume']:.2f}x, "
            f"RSI {read['rsi14']:.1f}. "
            f"Initial stop ${initial_stop:,.2f}; "
            f"risk budget {self.config.risk_pct:.2f}% "
            f"of equity; position capped at "
            f"{self.config.max_position_pct:.1f}%."
        )

        decision = tracker.record_trade(
            TICKER,
            "buy",
            quantity,
            reasoning,
            state.api_key,
            state.api_secret,
            state.feed,
        )

        self._log_decision(
            state,
            decision,
            read,
        )

        if decision.status == "filled":
            fill_price = float(
                decision.price or price
            )

            # Peak starts from the fill; the pre-entry bar's high cannot
            # retroactively tighten the stop.
            self.entry = {
                "price": fill_price,
                "peak": fill_price,
                "atr": atr,
                "bars": 0,
            }

        return decision.status == "filled"

    def _sell(
        self,
        state: AppState,
        tracker: DecisionTracker,
        quantity: float,
        read: dict,
        reasoning: str,
    ) -> None:
        decision = tracker.record_trade(
            TICKER,
            "sell",
            quantity,
            reasoning,
            state.api_key,
            state.api_secret,
            state.feed,
        )

        self._log_decision(
            state,
            decision,
            read,
        )

        if decision.status == "filled":
            self.entry = None

    # ------------------------------------------------------------------
    # Logging / timing
    # ------------------------------------------------------------------

    def _closing_soon(self) -> bool:
        to_close = market_hours.seconds_to_close()

        return (
            to_close is not None
            and to_close
            <= self.config.flatten_before_close_min * 60
        )

    def _read_summary(
        self,
        read: dict,
        position: float,
    ) -> str:
        ts = read["ts"].astimezone(ET)

        parts = [
            f"AAPL {ts:%H:%M}",
            f"${read['close']:,.2f}",
        ]

        if not pd.isna(read.get("vwap")):
            parts.append(
                f"VWAP ${read['vwap']:,.2f}"
            )
        else:
            parts.append("VWAP n/a")

        if not pd.isna(read.get("ema50")):
            parts.append(
                "EMA9/21/50 "
                f"{read['ema9']:.2f}/"
                f"{read['ema21']:.2f}/"
                f"{read['ema50']:.2f}"
            )
        else:
            parts.append("EMA warming up")

        if not pd.isna(
            read.get("relative_volume")
        ):
            parts.append(
                f"RV {read['relative_volume']:.2f}x"
            )
        else:
            parts.append("RV n/a")

        if not pd.isna(read.get("rsi14")):
            parts.append(
                f"RSI {read['rsi14']:.1f}"
            )
        else:
            parts.append("RSI n/a")

        parts.append(f"pos {position:g}")

        if self.entry:
            parts.append(
                f"stop ${self._stop_price():,.2f}"
            )

        return " · ".join(parts)

    def _log_decision(
        self,
        state: AppState,
        decision,
        read: dict,
    ) -> None:
        _log(
            state,
            {
                "type": "decision",
                "action": decision.action,
                "symbol": decision.symbol,
                "status": decision.status,
                "price": decision.price,
                "quantity": decision.filled_quantity,
                "reasoning": decision.reasoning,
                "bar_ts": read["ts"].isoformat(),
            },
        )


def _seconds_to_next_bar(
    cycle_sec: int,
    lag: float = TRADER_BY_CHATGPT_BAR_LAG_SEC,
) -> float:
    """Align cycles to closed-minute bars."""
    ts = _now().timestamp()

    return (
        (math.floor(ts / cycle_sec) + 1) * cycle_sec
        + lag
        - ts
    )


def _trader_by_chatgpt_loop(
    state: AppState,
    tracker: DecisionTracker,
    config: TraderByChatGPTConfig,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    trader = TraderByChatGPT(config)

    _log(
        state,
        {
            "type": "status",
            "text": (
                "TraderByChatGPT armed: long-only "
                "20-bar breakout above VWAP with "
                "EMA9>EMA21>EMA50, relative-volume "
                "confirmation, RSI filter, "
                f"{config.initial_stop_atr:.2f} ATR "
                "initial stop, "
                f"{config.trail_atr:.2f} ATR "
                "ratcheting trail, "
                f"{config.risk_pct:.2f}% risk/trade, "
                f"{config.flatten_before_close_min} "
                "min pre-close flatten."
            ),
        },
    )

    while not stop_event.is_set():
        try:
            trader.run_cycle(
                state,
                tracker,
            )

        except Exception as exc:
            _log(
                state,
                {
                    "type": "error",
                    "text": (
                        f"TraderByChatGPT cycle failed: "
                        f"{exc}"
                    ),
                },
            )

        scoring.maybe_score_day(
            state,
            tracker,
        )

        stop_event.wait(
            _seconds_to_next_bar(cycle_sec)
        )

    scoring.end_session(
        state,
        tracker,
    )

    state.agent_running = False

    _log(
        state,
        {
            "type": "status",
            "text": "TraderByChatGPT stopped",
        },
    )

    obs.flush()


def launch_trader_by_chatgpt(
    state: AppState,
    tracker: DecisionTracker,
    config: Optional[TraderByChatGPTConfig] = None,
    cycle_sec: int = TRADER_BY_CHATGPT_CYCLE_SEC,
) -> None:
    """Stop any existing personality and launch TraderByChatGPT."""
    stop_agent(state)

    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True

    scoring.begin_session(
        state,
        TRADER_BY_CHATGPT_KEY,
        [TICKER],
    )

    threading.Thread(
        target=_trader_by_chatgpt_loop,
        args=(
            state,
            tracker,
            config or TraderByChatGPTConfig(),
            cycle_sec,
            stop_event,
        ),
        daemon=True,
    ).start()
