"""Apple Trader -- a rule-based agent with no LLM in the loop.

Every other personality in `agent_stonks.agent` is a system prompt handed to a
model that reasons its way to a decision. This one is a plain loop: once a
minute it scores the AAPL bar that just closed with the momentum-change
regressor from FinNotebooks/TimeToChange (see `agent_stonks.momentum_model`)
and applies a fixed set of rules to the number that comes back. Same paper
ledger, same fill path, same log -- only the decision-making is deterministic,
so the same tape always produces the same trades.

The rules
---------
The model predicts Δ momentum over the next 15 minutes in bps/min, so
`momentum + prediction` is where the smoothed momentum is heading, and
comparing that to the day's ±theta gives the regime it is heading INTO.

* BUY when the model calls a change into the positive regime -- negative ->
  positive or balanced -> positive -- with |prediction| at or above
  `threshold`, repeated on `confirm_minutes` consecutive closed bars.
* SELL when it calls the way back out -- positive -> balanced or positive ->
  negative -- under the same size and repetition test.

`threshold` is the parameter that decides how large a change has to be to count
at all: the model's output is a ranking, not a calibrated forecast, so the
operating point is a percentile of |prediction| rather than a physical level
(the default is the 90th percentile on the model's own training days). Raising
it trades less often and only on the loudest calls.

Checking its own calls
----------------------
The model's timing is much weaker than its direction (its `persist` window is
15 bars, and on the full minute grid the timing signal barely beats chance), so
every position is re-examined on each bar rather than left to a static bracket:

* the regime it predicted has `validation_minutes` bars to actually show up.
  If momentum never turns positive in that window the call was false and the
  position is cut at market, without waiting for a stop.
* a realized flip into the negative regime, held for `confirm_minutes` bars,
  closes the position even when the model hasn't called the change yet.
* a hard `stop_loss_pct` below the entry closes it immediately.
* everything is flattened before the close -- momentum, regimes and every
  feature the model uses are intraday and do not survive the overnight gap.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

from . import market_hours, momentum_model, scoring
from . import observability as obs
from .agent import _log, stop_agent
from .clock import now as _now
from .config import (
    APPLE_TRADER_BAR_LAG_SEC,
    APPLE_TRADER_CONFIRM_MIN,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN,
    APPLE_TRADER_HISTORY_DAYS,
    APPLE_TRADER_MIN_BARS_TODAY,
    APPLE_TRADER_POSITION_PCT,
    APPLE_TRADER_STOP_LOSS_PCT,
    APPLE_TRADER_THRESHOLD,
    APPLE_TRADER_VALIDATION_MIN,
)
from .decisions import DecisionTracker
from .state import AppState

APPLE_TRADER_KEY = "apple_trader"
APPLE_TRADER_LABEL = "Apple Trader (rule-based, no LLM)"
APPLE_TRADER_AVATAR = "Multiavatar-4bcbffe68af819e050.png"

# Stands where a provider name goes for the other agents (SimLab run records,
# result grouping): this one has no LLM behind it, its rules are the "model".
RULE_PROVIDER = "rules"

# The saved model is fitted per ticker, and only AAPL has one -- the personality
# is named after the single symbol it can trade.
TICKER = "AAPL"

# The two calls the rules act on.
SIGNAL_TO_POSITIVE = "to_positive"
SIGNAL_OUT_OF_POSITIVE = "out_of_positive"


@dataclass
class AppleTraderConfig:
    """Tunables of the loop. `threshold` is the one the user normally moves."""

    threshold: float = APPLE_TRADER_THRESHOLD
    confirm_minutes: int = APPLE_TRADER_CONFIRM_MIN
    # None -> the bundle's own `persist` (the horizon the model predicts over).
    validation_minutes: Optional[int] = APPLE_TRADER_VALIDATION_MIN
    stop_loss_pct: float = APPLE_TRADER_STOP_LOSS_PCT
    position_pct: float = APPLE_TRADER_POSITION_PCT
    history_days: int = APPLE_TRADER_HISTORY_DAYS
    min_bars_today: int = APPLE_TRADER_MIN_BARS_TODAY
    flatten_before_close_min: int = APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN


def config_signature(config: "AppleTraderConfig | None" = None) -> str:
    """Compact identity of one rule set, standing in for a model name.

    Two runs of this agent differ only in these numbers, so SimLab groups and
    de-duplicates runs on this string exactly as it does on `provider/model`
    for the LLM agents -- retuning the threshold is a new configuration to
    test, not a repeat of one already tested.
    """
    c = config or AppleTraderConfig()
    return (
        f"momentum_change_{TICKER}(thr={c.threshold:g},confirm={c.confirm_minutes},"
        f"validate={c.validation_minutes or 'persist'},stop={c.stop_loss_pct:g}%,"
        f"size={c.position_pct:g}%)"
    )


def classify(read: dict, threshold: float) -> "str | None":
    """The call this minute's prediction makes, or None for "nothing to do".

    A call needs three things at once: a prediction at least `threshold` big,
    a current regime it can leave, and a projected momentum that lands in a
    different regime. Predicting a bigger positive momentum while already in
    the positive regime is not a change, and is deliberately not a signal.
    """
    pred = read.get("pred")
    regime, projected = read.get("regime"), read.get("projected_regime")
    if pred is None or regime is None or projected is None:
        return None
    if pred >= threshold and regime in (-1, 0) and projected == 1:
        return SIGNAL_TO_POSITIVE
    if pred <= -threshold and regime == 1 and projected in (0, -1):
        return SIGNAL_OUT_OF_POSITIVE
    return None


class AppleTrader:
    """The state machine driving one run: signal confirmation and the checks on
    a position already taken. One instance per launched agent."""

    def __init__(self, config: "AppleTraderConfig | None" = None, persist: int = 15) -> None:
        self.config = config or AppleTraderConfig()
        self.persist = persist
        # Confirmation counter: which call is building, and over how many
        # DISTINCT closed bars it has now repeated.
        self.pending: "str | None" = None
        self.pending_bars = 0
        # The open long: entry price, bars held, whether the predicted positive
        # regime has actually shown up, and the run of realized negative bars.
        self.entry: "dict | None" = None
        self.negative_bars = 0
        # Timestamp of the last bar acted on, so a cycle that re-reads the same
        # bar (slow fetch, market data lag) can't count as fresh confirmation.
        self.last_bar_ts = None

    # --- confirmation -----------------------------------------------------

    def _confirm(self, signal: "str | None") -> bool:
        """Feed one bar's call into the counter; True once it has repeated
        `confirm_minutes` times in a row."""
        if signal is None:
            self.pending, self.pending_bars = None, 0
            return False
        if signal != self.pending:
            self.pending, self.pending_bars = signal, 1
        else:
            self.pending_bars += 1
        return self.pending_bars >= max(1, self.config.confirm_minutes)

    @property
    def validation_bars(self) -> int:
        return self.config.validation_minutes or self.persist

    # --- one cycle --------------------------------------------------------

    def run_cycle(self, bundle: dict, state: AppState, tracker: DecisionTracker) -> str:
        """Score the newest closed bar and act on it. Returns a short outcome
        tag ("bought", "sold", "hold", "warming_up", "closed", "no_data")."""
        sym_state = state.sym(TICKER)
        if sym_state is None:
            _log(state, {"type": "error", "text": f"{TICKER} is not being streamed; nothing to trade."})
            return "no_data"

        if not market_hours.is_market_open():
            _log(
                state,
                {"type": "status", "text": "Market closed -- Apple Trader is not scoring bars."},
            )
            return "closed"

        frame = momentum_model.minute_frame(sym_state, self.config.history_days)
        read = momentum_model.read_latest(bundle, frame) if len(frame) else None
        if read is None:
            # Either the frame is too short to run the pipeline at all, or the
            # newest bar has no momentum yet -- the trailing window needs the
            # session's first `window` minutes before it produces a number.
            _log(
                state,
                {"type": "status", "text": f"No scoreable {TICKER} bar yet; momentum is still forming."},
            )
            return "no_data"

        position = tracker.position_for(TICKER)
        fresh_bar = read["ts"] != self.last_bar_ts
        if fresh_bar:
            self.last_bar_ts = read["ts"]

        if position > 0 and self.entry is None:
            # A position without a remembered entry (agent restarted onto an
            # existing ledger): adopt it at the current price so the checks
            # below still have something to measure against.
            self.entry = {"price": read["price"], "bars": 0, "regime_confirmed": read["regime"] == 1}
        if position <= 0:
            self.entry, self.negative_bars = None, 0

        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1
            if read["regime"] == 1:
                self.entry["regime_confirmed"] = True
            self.negative_bars = self.negative_bars + 1 if read["regime"] == -1 else 0

        # Inside the first hour the trailing windows are still filling, and the
        # model's own notes say to ignore what it prints there. The bar is read
        # and logged, but nothing it says counts as a call -- so no confirmation
        # can be built on it either. The risk checks below still run: a position
        # carried into the warm-up window must not go unwatched.
        warming_up = bool(
            self.config.min_bars_today and read["bars_today"] < self.config.min_bars_today
        )
        signal = None if warming_up else classify(read, self.config.threshold)
        confirmed = self._confirm(signal) if fresh_bar else False
        _log(state, {"type": "analysis", "text": self._read_summary(read, signal, position, warming_up)})

        if position > 0:
            reason = self._exit_reason(read, confirmed)
            if reason is not None:
                self._sell(state, tracker, position, read, reason)
                return "sold"
            return "warming_up" if warming_up else "hold"

        if confirmed and self.pending == SIGNAL_TO_POSITIVE:
            return "bought" if self._buy(state, tracker, read) else "hold"
        return "warming_up" if warming_up else "hold"

    # --- the checks on an open position ------------------------------------

    def _exit_reason(self, read: dict, confirmed: bool) -> "str | None":
        """Why this long should be closed on this bar, or None to keep holding.

        Ordered by urgency: the loss cap first, then the end of the session,
        then the call being wrong, then the model calling the way out.
        """
        entry = self.entry or {}
        entry_price = entry.get("price") or 0.0
        pnl_pct = (read["price"] / entry_price - 1) * 100 if entry_price else 0.0

        if self.config.stop_loss_pct and pnl_pct <= -abs(self.config.stop_loss_pct):
            return (
                f"Stop: {pnl_pct:+.2f}% from the ${entry_price:,.2f} entry breaches the "
                f"{self.config.stop_loss_pct:.2f}% loss cap. Cutting the position at market."
            )

        to_close = market_hours.seconds_to_close()
        if to_close is not None and to_close <= self.config.flatten_before_close_min * 60:
            return (
                f"Session ends in {to_close / 60:.0f} min. Momentum, regimes and every model "
                "feature are intraday, so the position is flattened rather than carried overnight."
            )

        if not entry.get("regime_confirmed") and entry.get("bars", 0) >= self.validation_bars:
            return (
                f"False call: {self.validation_bars} bars after entry momentum is still "
                f"{momentum_model.regime_name(read['regime'])} ({read['mom']:+.2f} bps/min vs "
                f"theta {read['theta']:.2f}), so the predicted turn into the positive regime "
                f"never happened. Cutting losses at {pnl_pct:+.2f}%."
            )

        if self.negative_bars >= max(1, self.config.confirm_minutes):
            return (
                f"Momentum has been negative for {self.negative_bars} straight bars "
                f"({read['mom']:+.2f} bps/min vs theta {read['theta']:.2f}) -- the positive "
                f"regime is gone. Exiting at {pnl_pct:+.2f}%."
            )

        if confirmed and self.pending == SIGNAL_OUT_OF_POSITIVE:
            return (
                f"Model calls a persistent change out of the positive regime: Δmomentum "
                f"{read['pred']:+.2f} bps/min (|Δ| >= {self.config.threshold:.2f}) on "
                f"{self.pending_bars} consecutive bars, projecting "
                f"{momentum_model.regime_name(read['projected_regime'])}. Exiting at {pnl_pct:+.2f}%."
            )
        return None

    # --- orders ------------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, read: dict) -> bool:
        """Deploy `position_pct` of the cash balance; False if it buys nothing."""
        price = read["price"]
        cash = tracker.snapshot()["cash"]
        budget = cash * self.config.position_pct / 100.0
        quantity = math.floor(budget / price * 1e4) / 1e4 if price > 0 else 0.0
        if quantity <= 0:
            _log(
                state,
                {"type": "status", "text": f"Entry signal confirmed but ${cash:,.2f} cash buys no {TICKER}."},
            )
            return False

        reasoning = (
            f"Model calls a persistent change into the positive regime: Δmomentum "
            f"{read['pred']:+.2f} bps/min (>= {self.config.threshold:.2f}) on "
            f"{self.pending_bars} consecutive bars, taking momentum "
            f"{read['mom']:+.2f} -> {read['projected_mom']:+.2f} bps/min through theta "
            f"{read['theta']:.2f} out of the {momentum_model.regime_name(read['regime'])} regime."
        )
        decision = tracker.record_trade(
            TICKER, "buy", quantity, reasoning, state.api_key, state.api_secret, state.feed
        )
        self._log_decision(state, decision, read)
        if decision.status == "filled":
            self.entry = {
                "price": decision.price,
                "bars": 0,
                "regime_confirmed": read["regime"] == 1,
            }
            self.negative_bars = 0
        self.pending, self.pending_bars = None, 0
        return decision.status == "filled"

    def _sell(
        self, state: AppState, tracker: DecisionTracker, quantity: float, read: dict, reasoning: str
    ) -> None:
        decision = tracker.record_trade(
            TICKER, "sell", quantity, reasoning, state.api_key, state.api_secret, state.feed
        )
        self._log_decision(state, decision, read)
        if decision.status == "filled":
            self.entry, self.negative_bars = None, 0
        self.pending, self.pending_bars = None, 0

    # --- logging -----------------------------------------------------------

    def _read_summary(
        self, read: dict, signal: "str | None", position: float, warming_up: bool = False
    ) -> str:
        pred = read["pred"]
        pred_str = f"{pred:+.2f}" if pred is not None else "n/a"
        parts = [
            f"{TICKER} {read['ts']:%H:%M} ${read['price']:,.2f}",
            f"momentum {read['mom']:+.2f} bps/min vs theta {read['theta']:.2f} "
            f"({momentum_model.regime_name(read['regime'])})",
            f"Δmomentum {pred_str} (threshold {self.config.threshold:.2f}) -> "
            f"{momentum_model.regime_name(read['projected_regime'])}",
        ]
        if warming_up:
            parts.append(
                f"warming up ({read['bars_today']}/{self.config.min_bars_today} bars) -- not trading"
            )
        if signal is not None:
            parts.append(f"{signal} {self.pending_bars}/{self.config.confirm_minutes} confirmed")
        if position > 0 and self.entry:
            pnl = (read["price"] / self.entry["price"] - 1) * 100 if self.entry["price"] else 0.0
            held = self.entry["bars"]
            state_word = "confirmed" if self.entry["regime_confirmed"] else "unconfirmed"
            parts.append(
                f"long {position:g} sh @ ${self.entry['price']:,.2f} ({pnl:+.2f}%), "
                f"{held} bars, regime {state_word}"
            )
        return " · ".join(parts)

    def _log_decision(self, state: AppState, decision, read: dict) -> None:
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
                "regime": momentum_model.regime_name(read["regime"]),
            },
        )


# --- the loop ---------------------------------------------------------------

def _seconds_to_next_bar(cycle_sec: int, lag: float = APPLE_TRADER_BAR_LAG_SEC) -> float:
    """Seconds to wait so the next cycle lands just after a bar closes.

    Aligning to the clock (rather than sleeping a flat `cycle_sec` from
    wherever the last cycle finished) keeps one cycle per closed bar however
    long the scoring itself took.
    """
    ts = _now().timestamp()
    return (math.floor(ts / cycle_sec) + 1) * cycle_sec + lag - ts


def _apple_trader_loop(
    state: AppState,
    tracker: DecisionTracker,
    config: AppleTraderConfig,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    bundle = momentum_model.load_bundle(TICKER)
    if bundle is None:
        _log(
            state,
            {
                "type": "error",
                "text": (
                    f"No momentum-change model for {TICKER} at "
                    f"{momentum_model.model_path(TICKER)} (or scikit-learn/joblib are not "
                    f"installed). Apple Trader cannot run without it."
                ),
            },
        )
        scoring.end_session(state, tracker)
        state.agent_running = False
        return

    params = bundle["pipeline_params"]
    trader = AppleTrader(config, persist=params["persist"])
    train_days = bundle.get("train_days") or []
    _log(
        state,
        {
            "type": "status",
            "text": (
                f"Apple Trader armed on {bundle.get('model_name', 'the saved model')} "
                f"({len(train_days)} training days"
                + (f", {train_days[0]} → {train_days[-1]}" if train_days else "")
                + f"): scoring every {TICKER} minute bar, |Δmomentum| >= {config.threshold:.2f} "
                f"bps/min confirmed over {config.confirm_minutes} bars to trade, "
                f"{trader.validation_bars} bars to prove the regime, "
                f"{config.stop_loss_pct:.2f}% loss cap."
            ),
        },
    )

    while not stop_event.is_set():
        try:
            trader.run_cycle(bundle, state, tracker)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Apple Trader cycle failed: {exc}"})
        scoring.maybe_score_day(state, tracker)
        stop_event.wait(_seconds_to_next_bar(cycle_sec))

    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Apple Trader stopped"})
    obs.flush()


def launch_apple_trader(
    state: AppState,
    tracker: DecisionTracker,
    config: "AppleTraderConfig | None" = None,
    cycle_sec: int = APPLE_TRADER_CYCLE_SEC,
) -> None:
    """Stop any running agent for this state, then start the Apple Trader loop.

    It trades only `TICKER`, which must already be streamed. No LLM client, no
    tools and no tactics are involved -- the loop places its own orders through
    the same `DecisionTracker` as every other personality.
    """
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, APPLE_TRADER_KEY, [TICKER])
    threading.Thread(
        target=_apple_trader_loop,
        args=(state, tracker, config or AppleTraderConfig(), cycle_sec, stop_event),
        daemon=True,
    ).start()
