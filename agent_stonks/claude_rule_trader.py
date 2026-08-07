"""TraderByClaude -- a rule-based agent with no LLM and no model in the loop.

The deliberate counter-thesis to TraderByChatGPT
------------------------------------------------
Both agents are long-only intraday state machines on AAPL, both refuse to
trade against the session trend, and both size on distance-to-stop. They
disagree about *where in a trend you are allowed to buy*.

TraderByChatGPT buys strength: a close through the 20-bar high on unusual
volume. That entry is right at the extreme of the move, which puts the
structural stop far below and makes every failed breakout a full-width loss.
Intraday breakouts on a single large cap fail often -- the 20-bar high is where
resting sell orders live.

This one buys the *pullback inside a trend that is already established*. It
waits for price to come back to the 20-bar EMA and then take out the previous
bar's high, so the entry sits near support rather than at resistance:

* the stop goes under the pullback low, a few cents of ATR away instead of a
  fixed 1.3 ATR, so the same risk budget buys a bigger position on the same
  idea, and
* the invalidation is structural rather than arbitrary -- if the low that
  defined the pullback breaks, the reason for the trade is gone.

The rules
---------
* BUY when all of these hold on the bar that just closed: EMA20 > EMA50 and
  EMA20 rising over `trend_slope_bars`; close above session VWAP; price dipped
  to or through EMA20 somewhere in the last `pullback_lookback` bars; this bar
  closes back above EMA20 *and* above the previous bar's high; the pullback
  happened on below-median volume; the close is no more than
  `max_stretch_atr` ATR above EMA20; and the bar is green. Entries 09:45-15:15.
* The quiet-pullback test is the part that does the most work. A healthy
  pullback is participation drying up, not distribution -- if the dip came on
  heavy volume, sellers are working an order and the "support" is a queue.
* SIZE on risk: stop is `stop_buffer_atr` ATR under the pullback low, and the
  quantity is whatever puts `risk_pct` of equity between the fill and that
  stop, capped at `max_position_pct` of cash. A pullback further than
  `max_risk_atr` ATR away is skipped rather than sized down -- a stop that wide
  is a different trade than the one these rules describe.
* SELL on whichever comes first: the stop under the pullback low, a close back
  below EMA20 once the trade has had `structure_grace_bars` to work (the trend
  structure the entry was built on is gone), the time stop, or the pre-close
  flatten. Once the trade is `breakeven_at_r` R ahead the stop ratchets to the
  entry and never comes back down.

A day that is 1% down stops opening new trades; whatever is already on is
still managed to its stop.

What would falsify it
---------------------
The thesis is that near-support entries lose less per failure than
at-resistance entries, and that this outweighs the signals it misses by
refusing to chase. On a tape that trends hard in one direction all day the
breakout agent should win outright, because the pullback this one waits for
never comes. Two stored days cannot tell those apart -- see the note in the
SimLab Agents tab.
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
    TRADER_BY_CLAUDE_BAR_LAG_SEC,
    TRADER_BY_CLAUDE_CYCLE_SEC,
    TRADER_BY_CLAUDE_FLATTEN_BEFORE_CLOSE_MIN,
    TRADER_BY_CLAUDE_POSITION_PCT,
    TRADER_BY_CLAUDE_RISK_PCT,
)
from .decisions import DecisionTracker
from .intraday_bars import ET, atr as _atr, session_frame, session_vwap as _session_vwap
from .state import AppState

TRADER_BY_CLAUDE_KEY = "trader_by_claude"
TRADER_BY_CLAUDE_LABEL = "TraderByClaude (rule-based, no LLM)"
TRADER_BY_CLAUDE_AVATAR = "Multiavatar-TraderByClaude.png"
RULE_PROVIDER = "rules"
TICKER = "AAPL"


@dataclass(frozen=True)
class TraderByClaudeConfig:
    """Deterministic strategy/risk controls. Long-only, one position at a time."""

    # Trend: the regime that has to already exist before a pullback is
    # interesting. A dip in a downtrend is not a pullback, it is the trend.
    fast_ema: int = 20
    slow_ema: int = 50
    trend_slope_bars: int = 5

    # The pullback and its resolution.
    pullback_lookback: int = 10
    # How far above EMA20 the reclaim bar may close, in ATR. This is the
    # anti-chase rule: past it, the pullback has already been bought.
    max_stretch_atr: float = 0.75
    # Pullback volume as a multiple of the 20-bar median. Below 1.0 means
    # participation dried up on the dip, which is what makes it a pullback.
    quiet_pullback_ratio: float = 1.00

    entry_start: time = time(9, 45)
    entry_stop: time = time(15, 15)

    # Risk / exits
    risk_pct: float = TRADER_BY_CLAUDE_RISK_PCT
    max_position_pct: float = TRADER_BY_CLAUDE_POSITION_PCT
    stop_buffer_atr: float = 0.25
    # Skip rather than size down when the structural stop is this far away.
    max_risk_atr: float = 2.00
    breakeven_at_r: float = 1.00
    # Bars of room before a close under EMA20 counts as a broken structure --
    # without it the entry bar's own noise can close the trade it opened.
    structure_grace_bars: int = 3

    flatten_before_close_min: int = TRADER_BY_CLAUDE_FLATTEN_BEFORE_CLOSE_MIN
    max_bars_in_trade: int = 60


def config_signature(config: Optional[TraderByClaudeConfig] = None) -> str:
    """Stable identity for SimLab grouping/de-duplication.

    Carries every number a SimLab run can vary, so two runs that differ in one
    of them are two configurations to compare rather than one repeated.
    """
    c = config or TraderByClaudeConfig()
    return (
        f"trader_by_claude(ema={c.fast_ema}/{c.slow_ema},pb={c.pullback_lookback},"
        f"quiet<={c.quiet_pullback_ratio:g}x,stretch<={c.max_stretch_atr:g}atr,"
        f"stop=low-{c.stop_buffer_atr:g}atr(max {c.max_risk_atr:g}),"
        f"be@{c.breakeven_at_r:g}R,risk={c.risk_pct:g}%,cap={c.max_position_pct:g}%)"
    )


def _feature_frame(frame: pd.DataFrame, config: TraderByClaudeConfig) -> pd.DataFrame:
    """Every number the rules read, computed once per cycle.

    Baselines that a decision compares *against* are shifted by one bar, so
    nothing the entry test looks at includes the bar being tested: a rolling
    high that contains its own bar is a tautology, not a signal.
    """
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{TICKER} bars missing columns: {sorted(missing)}")

    df = frame.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=sorted(required)).sort_values("ts").drop_duplicates("ts")
    if df.empty:
        return df

    # Trend
    df["ema_fast"] = df["close"].ewm(
        span=config.fast_ema, adjust=False, min_periods=config.fast_ema
    ).mean()
    df["ema_slow"] = df["close"].ewm(
        span=config.slow_ema, adjust=False, min_periods=config.slow_ema
    ).mean()
    df["ema_slope"] = df["ema_fast"] - df["ema_fast"].shift(config.trend_slope_bars)

    # Volatility / value
    df["atr14"] = _atr(df, 14)
    df["vwap"] = _session_vwap(df)

    # The pullback: how deep it went and how busy it was. Both windows END on
    # the previous bar -- the dip is what happened BEFORE the bar that reclaims
    # it, and including the reclaim bar's own low would let a single wide bar
    # manufacture its own setup.
    df["pullback_low"] = df["low"].shift(1).rolling(
        config.pullback_lookback, min_periods=config.pullback_lookback
    ).min()
    df["pullback_volume"] = df["volume"].shift(1).rolling(
        config.pullback_lookback, min_periods=config.pullback_lookback
    ).mean()
    df["volume_median20"] = df["volume"].shift(1).rolling(20, min_periods=20).median()
    df["previous_high"] = df["high"].shift(1)

    df["stretch_atr"] = (df["close"] - df["ema_fast"]) / df["atr14"]

    local = df["ts"].dt.tz_convert(ET)
    df["local_time"] = local.dt.time
    df["bars_today"] = local.dt.date.map(local.dt.date.value_counts())
    return df


class TraderByClaude:
    """One deterministic AAPL long-only intraday state machine."""

    def __init__(self, config: Optional[TraderByClaudeConfig] = None) -> None:
        self.config = config or TraderByClaudeConfig()
        self.last_bar_ts = None
        # The open long: fill price, the structural stop, the running peak, the
        # per-share risk that defines 1R, and how many bars it has been held.
        self.entry: Optional[dict] = None
        self.starting_equity: Optional[float] = None

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def run_cycle(self, state: AppState, tracker: DecisionTracker) -> str:
        """Read the newest closed bar and act on it. Returns a short outcome
        tag ("bought", "sold", "hold", "warming_up", "closed", "no_data")."""
        sym_state = state.sym(TICKER)
        if sym_state is None:
            _log(state, {
                "type": "error",
                "text": f"{TICKER} is not being streamed; nothing to trade.",
            })
            return "no_data"

        if not market_hours.is_market_open():
            _log(state, {
                "type": "status",
                "text": "Market closed -- TraderByClaude is not watching bars.",
            })
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

        # Adopt a pre-existing long if this process restarted onto a live
        # ledger: the structure that justified it is not recoverable, so the
        # stop starts from here rather than from a pullback low never seen.
        if position > 0 and self.entry is None:
            risk = max(float(read["atr14"]) * self.config.stop_buffer_atr, 0.01)
            self.entry = {
                "price": float(read["close"]),
                "stop": float(read["close"]) - risk,
                "peak": float(read["high"]),
                "risk": risk,
                "bars": 0,
            }
        elif position <= 0:
            self.entry = None

        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1
            self.entry["peak"] = max(self.entry["peak"], float(read["high"]))
            self._ratchet_stop()

        _log(state, {"type": "analysis", "text": self._read_summary(read, position)})

        snapshot = tracker.snapshot()
        equity = float(snapshot.get("equity", snapshot.get("cash", 0.0)))
        if self.starting_equity is None and equity > 0:
            self.starting_equity = equity
        daily_loss_guard = (
            self.starting_equity is not None
            and equity <= self.starting_equity * 0.99
        )

        # Managing an open position always outranks looking for a new one.
        if position > 0:
            reason = self._exit_reason(read)
            if reason is not None:
                self._sell(state, tracker, position, read, reason)
                return "sold"
            return "hold"

        if not fresh_bar or daily_loss_guard:
            return "hold"

        if self._entry_signal(read):
            if self._closing_soon():
                _log(state, {
                    "type": "status",
                    "text": (
                        f"Pullback entry on the {read['ts'].astimezone(ET):%H:%M} bar, "
                        "but the session is inside its closing flatten window and the "
                        "position would be shut straight back out. Standing down."
                    ),
                })
                return "hold"
            return "bought" if self._buy(state, tracker, read) else "hold"

        return "warming_up" if self._warming_up(read) else "hold"

    # ------------------------------------------------------------------
    # Entry rules
    # ------------------------------------------------------------------

    def _warming_up(self, read: dict) -> bool:
        """Before this, the indicators exist but do not yet mean anything."""
        needed = max(self.config.slow_ema, self.config.pullback_lookback + 21)
        return (
            pd.isna(read.get("ema_slow"))
            or pd.isna(read.get("pullback_low"))
            or pd.isna(read.get("volume_median20"))
            or pd.isna(read.get("ema_slope"))
            or int(read.get("bars_today", 0)) < needed
        )

    def _entry_signal(self, read: dict) -> bool:
        """One pure boolean predicate on the just-closed bar.

        Reads in three parts: is there a trend, was there a pullback in it, and
        has this bar resolved the pullback upward without already running away.
        """
        if self._warming_up(read):
            return False

        local_time = read["local_time"]
        if local_time < self.config.entry_start or local_time >= self.config.entry_stop:
            return False

        values = (
            read["close"], read["open"], read["vwap"], read["ema_fast"],
            read["ema_slow"], read["ema_slope"], read["pullback_low"],
            read["pullback_volume"], read["volume_median20"],
            read["previous_high"], read["stretch_atr"], read["atr14"],
        )
        if any(pd.isna(v) for v in values):
            return False

        return (
            # Trend: the fast mean is above the slow one and still climbing.
            read["ema_fast"] > read["ema_slow"]
            and read["ema_slope"] > 0
            # Value: the session's buyers are in control.
            and read["close"] > read["vwap"]
            # Pullback: price actually came back to the mean recently...
            and read["pullback_low"] <= read["ema_fast"]
            # ...on drying-up participation, not on a seller working an order.
            and read["pullback_volume"]
            <= self.config.quiet_pullback_ratio * read["volume_median20"]
            # Resolution: this bar closes back above the mean and takes out the
            # previous bar's high -- the dip is over, not merely paused.
            and read["close"] > read["ema_fast"]
            and read["close"] > read["previous_high"]
            # Anti-chase: buy near the mean, not after the move has left it.
            and read["stretch_atr"] <= self.config.max_stretch_atr
            # Require a bullish close.
            and read["close"] > read["open"]
        )

    # ------------------------------------------------------------------
    # Exit rules
    # ------------------------------------------------------------------

    def _ratchet_stop(self) -> None:
        """Move the stop to breakeven once the trade is `breakeven_at_r` ahead.

        One-way only: a stop that can move down is not a stop.
        """
        if not self.entry:
            return
        entry_price = float(self.entry["price"])
        target = entry_price + self.config.breakeven_at_r * float(self.entry["risk"])
        if float(self.entry["peak"]) >= target:
            self.entry["stop"] = max(float(self.entry["stop"]), entry_price)

    def _exit_reason(self, read: dict) -> Optional[str]:
        if not self.entry:
            return None

        entry_price = float(self.entry["price"])
        stop = float(self.entry["stop"])
        bars = int(self.entry["bars"])
        price = float(read["close"])
        pnl_pct = (price / entry_price - 1.0) * 100.0
        r_multiple = (price - entry_price) / float(self.entry["risk"])

        if price <= stop:
            at_breakeven = stop >= entry_price
            return (
                f"{'Breakeven stop' if at_breakeven else 'Structural stop'}: "
                f"${price:,.2f} <= ${stop:,.2f}; entry ${entry_price:,.2f}, "
                f"peak ${self.entry['peak']:,.2f}, {r_multiple:+.2f}R "
                f"({pnl_pct:+.2f}%)."
            )

        # The trend structure the entry was built on. Given a few bars to work
        # first, because the entry bar's own pullback into the mean is noise,
        # not a failure.
        if bars >= self.config.structure_grace_bars and not pd.isna(read["ema_fast"]):
            if price < float(read["ema_fast"]):
                return (
                    f"Structure break: closed ${price:,.2f} back below the "
                    f"{self.config.fast_ema}-bar mean ${read['ema_fast']:,.2f} after "
                    f"{bars} bars; {r_multiple:+.2f}R ({pnl_pct:+.2f}%)."
                )

        if bars >= self.config.max_bars_in_trade:
            return (
                f"Time stop after {bars} bars; the continuation this entry "
                f"was paid to catch did not arrive. {r_multiple:+.2f}R "
                f"({pnl_pct:+.2f}%)."
            )

        if self._closing_soon():
            to_close = market_hours.seconds_to_close() or 0.0
            return (
                f"Session ends in {to_close / 60:.0f} min; flattening {TICKER} "
                f"rather than carrying an intraday signal overnight "
                f"({pnl_pct:+.2f}%)."
            )

        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, read: dict) -> bool:
        price = float(read["close"])
        atr = float(read["atr14"])
        if price <= 0 or atr <= 0:
            return False

        stop = float(read["pullback_low"]) - self.config.stop_buffer_atr * atr
        risk_per_share = price - stop
        # A stop this far away is a different trade than the rules describe, so
        # it is skipped rather than sized down into something unrecognisable.
        if risk_per_share <= 0 or risk_per_share > self.config.max_risk_atr * atr:
            return False

        snapshot = tracker.snapshot()
        cash = float(snapshot.get("cash", 0.0))
        equity = float(snapshot.get("equity", cash))

        risk_budget = equity * self.config.risk_pct / 100.0
        risk_qty = risk_budget / risk_per_share
        # Hard notional cap: a stop a few cents away would otherwise ask for a
        # position many times the account.
        cap_qty = cash * self.config.max_position_pct / 100.0 / price
        quantity = math.floor(min(risk_qty, cap_qty) * 1e4) / 1e4
        if quantity <= 0:
            return False

        reasoning = (
            f"BUY {TICKER}: pullback to the {self.config.fast_ema}-bar mean "
            f"${read['ema_fast']:,.2f} on "
            f"{read['pullback_volume'] / read['volume_median20']:.2f}x median volume, "
            f"reclaimed by a close at ${price:,.2f} through the prior bar's "
            f"${read['previous_high']:,.2f}, {read['stretch_atr']:.2f} ATR above the "
            f"mean, in an uptrend (EMA{self.config.fast_ema}>EMA{self.config.slow_ema}, "
            f"rising) above VWAP ${read['vwap']:,.2f}. Stop ${stop:,.2f} under the "
            f"pullback low ${read['pullback_low']:,.2f}; risking "
            f"{self.config.risk_pct:.2f}% of equity, "
            f"${risk_per_share:,.2f}/share to that stop."
        )

        decision = tracker.record_trade(
            TICKER, "buy", quantity, reasoning,
            state.api_key, state.api_secret, state.feed,
        )
        self._log_decision(state, decision, read)

        if decision.status == "filled":
            fill_price = float(decision.price or price)
            # 1R is measured from the fill to the stop that was actually set,
            # so a slipped fill makes the trade smaller in R, not luckier.
            self.entry = {
                "price": fill_price,
                "stop": stop,
                "peak": fill_price,
                "risk": max(fill_price - stop, 0.01),
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
            TICKER, "sell", quantity, reasoning,
            state.api_key, state.api_secret, state.feed,
        )
        self._log_decision(state, decision, read)
        if decision.status == "filled":
            self.entry = None

    # ------------------------------------------------------------------
    # Logging / timing
    # ------------------------------------------------------------------

    def _closing_soon(self) -> bool:
        to_close = market_hours.seconds_to_close()
        return (
            to_close is not None
            and to_close <= self.config.flatten_before_close_min * 60
        )

    def _read_summary(self, read: dict, position: float) -> str:
        ts = read["ts"].astimezone(ET)
        parts = [f"{TICKER} {ts:%H:%M}", f"${read['close']:,.2f}"]

        if not pd.isna(read.get("vwap")):
            parts.append(f"VWAP ${read['vwap']:,.2f}")
        else:
            parts.append("VWAP n/a")

        if not pd.isna(read.get("ema_slow")):
            parts.append(
                f"EMA{self.config.fast_ema}/{self.config.slow_ema} "
                f"{read['ema_fast']:.2f}/{read['ema_slow']:.2f}"
            )
        else:
            parts.append("EMA warming up")

        if not pd.isna(read.get("pullback_low")):
            parts.append(f"pb low ${read['pullback_low']:,.2f}")
        else:
            parts.append("pb n/a")

        if not pd.isna(read.get("stretch_atr")):
            parts.append(f"stretch {read['stretch_atr']:+.2f} ATR")

        parts.append(f"pos {position:g}")
        if self.entry:
            parts.append(f"stop ${float(self.entry['stop']):,.2f}")
        return " · ".join(parts)

    def _log_decision(self, state: AppState, decision, read: dict) -> None:
        _log(state, {
            "type": "decision",
            "action": decision.action,
            "symbol": decision.symbol,
            "status": decision.status,
            "price": decision.price,
            "quantity": decision.filled_quantity,
            "reasoning": decision.reasoning,
            "bar_ts": read["ts"].isoformat(),
        })


def _seconds_to_next_bar(
    cycle_sec: int, lag: float = TRADER_BY_CLAUDE_BAR_LAG_SEC
) -> float:
    """Align cycles to closed-minute bars."""
    ts = _now().timestamp()
    return (math.floor(ts / cycle_sec) + 1) * cycle_sec + lag - ts


def _trader_by_claude_loop(
    state: AppState,
    tracker: DecisionTracker,
    config: TraderByClaudeConfig,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    trader = TraderByClaude(config)
    _log(state, {
        "type": "status",
        "text": (
            f"TraderByClaude armed: long-only pullback continuation -- buys a dip "
            f"to the {config.fast_ema}-bar mean on quiet volume once it is "
            f"reclaimed, inside an EMA{config.fast_ema}>EMA{config.slow_ema} uptrend "
            f"above VWAP; stop {config.stop_buffer_atr:g} ATR under the pullback low, "
            f"breakeven at {config.breakeven_at_r:g}R, out on a close back below the "
            f"mean, {config.risk_pct:.2f}% risk/trade, "
            f"{config.flatten_before_close_min} min pre-close flatten."
        ),
    })

    while not stop_event.is_set():
        try:
            trader.run_cycle(state, tracker)
        except Exception as exc:
            _log(state, {
                "type": "error",
                "text": f"TraderByClaude cycle failed: {exc}",
            })
        scoring.maybe_score_day(state, tracker)
        stop_event.wait(_seconds_to_next_bar(cycle_sec))

    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "TraderByClaude stopped"})
    obs.flush()


def launch_trader_by_claude(
    state: AppState,
    tracker: DecisionTracker,
    config: Optional[TraderByClaudeConfig] = None,
    cycle_sec: int = TRADER_BY_CLAUDE_CYCLE_SEC,
) -> None:
    """Stop any existing personality and launch TraderByClaude."""
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, TRADER_BY_CLAUDE_KEY, [TICKER])
    threading.Thread(
        target=_trader_by_claude_loop,
        args=(state, tracker, config or TraderByClaudeConfig(), cycle_sec, stop_event),
        daemon=True,
    ).start()
