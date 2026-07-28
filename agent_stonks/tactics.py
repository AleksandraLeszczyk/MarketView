"""
Standing conditional trade plans ("tactics") for the trading agent.

A `Tactics` object is a set of conditional actions the agent arms via the
`set_tactics` tool instead of trading at the current price: "buy 10 shares if
last_price below X", "sell 20% of the position if last_price above Y and vix
below Z". Each action carries one or more conditions that must ALL hold at the
same moment for the action to fire.

`TacticsExecutor` is the matching engine: a background thread, nudged by every
stream tick (bars/trades/quotes) and backed by a slow fallback poll, that
evaluates the armed tactics against live state and executes the first action
whose conditions are met AND that can actually transact right now -- through
the same `DecisionTracker` path the agent's own buy/sell decisions take, so
fills, fees, logging, and charting behave identically. After one action
executes, the whole tactics set is disarmed and the agent is woken to
reevaluate with the fill in hand; it re-arms whatever still applies on the next
cycle.

Executability is part of the match, not an afterthought: a sell is dormant
while the position is flat and a buy is dormant with no cash. A protective stop
armed next to an entry trigger is meant to cover the position that entry will
open, so a dip through the stop level BEFORE the entry fires must leave both
armed -- otherwise the agent loses its breakout buy to its own stop and sits
out the move (see `_executable_now`).

While a sell bracket is armed on an open long position -- a take-profit
("sell when last_price above target") together with a protective stop ("sell
when last_price below stop", armed below the entry price) -- the executor also
trails the stop up mechanically as the price advances (see
`TacticsExecutor._ratchet_trailing_stop`). The take-profit level never moves.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

from . import clock, historical
from .config import TACTICS_POLL_SEC
from .state import (
    ALERTABLE_FIELDS,
    PRICE_AXIS_ALERT_FIELDS,
    alert_field_value,
    compare,
    format_alert,
)

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import SymbolState

logger = logging.getLogger(__name__)


# Every field a tactic condition can watch. All alertable fields (including the
# derived `momentum_pct`) are refreshed on the live price/quote stream; the one
# extra, `vix`, is evaluated on demand by the executor from the (cached)
# broad-market fetch.
TACTIC_CONDITION_FIELDS: dict[str, str] = {
    **ALERTABLE_FIELDS,
    "vix": "CBOE VIX index level (broad-market fear gauge, refreshed every few minutes)",
}


@dataclass
class TacticCondition:
    field: str  # one of TACTIC_CONDITION_FIELDS
    condition: str  # "above" (>=) | "below" (<=)
    value: float
    # Seconds the comparison must hold CONTINUOUSLY before it counts as met
    # (0 = met on the first tick that satisfies it). Lets an entry demand a
    # sustained cross instead of firing on a single wick tick.
    hold_sec: float = 0.0
    # The value the condition was armed at, recorded the first time the
    # trailing-stop ratchet moves `value` so the ratchet keeps interpolating
    # from the armed level. None until (unless) the condition is ever trailed.
    initial_value: Optional[float] = None
    # Monotonic timestamp of when the comparison started holding continuously
    # (runtime state for hold_sec; None while the comparison is unmet).
    met_since: Optional[float] = None


@dataclass
class TacticAction:
    action: str  # "buy" | "sell"
    # Exactly one of the two sizes is set. `quantity` is absolute shares;
    # `quantity_pct` (0-100] is resolved at execution time: percent of the
    # current position for a sell, percent of available cash for a buy.
    quantity: Optional[float]
    quantity_pct: Optional[float]
    conditions: list[TacticCondition] = field(default_factory=list)
    note: str = ""
    # Whether the automatic trailing-stop ratchet may move this action's
    # protective-stop conditions (sell stops only). False = leave my levels alone.
    trail: bool = True


@dataclass
class Tactics:
    ts: str
    symbol: str
    reasoning: str
    actions: list[TacticAction] = field(default_factory=list)
    status: str = "armed"  # "armed" | "executed" | "cancelled"
    # Highest last_price seen while armed; drives the trailing-stop ratchet.
    high_water: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_tactics(symbol: str, raw_actions: object, reasoning: str) -> "tuple[Tactics | None, str | None]":
    """Validate a raw set_tactics payload (from the LLM) into a `Tactics`, or
    return (None, error) describing exactly what to fix so the model can retry."""
    if not isinstance(raw_actions, list) or not raw_actions:
        return None, "'actions' must be a non-empty array (or [] to cancel armed tactics)."

    actions: list[TacticAction] = []
    for i, raw in enumerate(raw_actions):
        if not isinstance(raw, dict):
            return None, f"actions[{i}] must be an object."
        action = raw.get("action")
        if action not in ("buy", "sell"):
            return None, f"actions[{i}].action must be 'buy' or 'sell', got {action!r}."

        quantity = raw.get("quantity")
        quantity_pct = raw.get("quantity_pct")
        if (quantity is None) == (quantity_pct is None):
            return None, (
                f"actions[{i}] must set exactly one of 'quantity' (shares) or "
                "'quantity_pct' (percent of position for sell / of cash for buy)."
            )
        try:
            if quantity is not None:
                quantity = float(quantity)
                if quantity <= 0:
                    return None, f"actions[{i}].quantity must be > 0."
            else:
                quantity_pct = float(quantity_pct)
                if not 0 < quantity_pct <= 100:
                    return None, f"actions[{i}].quantity_pct must be in (0, 100]."
        except (TypeError, ValueError):
            return None, f"actions[{i}] has a non-numeric quantity."

        raw_conditions = raw.get("conditions")
        if not isinstance(raw_conditions, list) or not raw_conditions:
            return None, f"actions[{i}].conditions must be a non-empty array."
        conditions: list[TacticCondition] = []
        for j, cond in enumerate(raw_conditions):
            if not isinstance(cond, dict):
                return None, f"actions[{i}].conditions[{j}] must be an object."
            cfield = cond.get("field")
            if cfield not in TACTIC_CONDITION_FIELDS:
                return None, (
                    f"actions[{i}].conditions[{j}].field must be one of: "
                    f"{', '.join(TACTIC_CONDITION_FIELDS)}."
                )
            ccond = cond.get("condition")
            if ccond not in ("above", "below"):
                return None, f"actions[{i}].conditions[{j}].condition must be 'above' or 'below'."
            try:
                cvalue = float(cond.get("value"))
            except (TypeError, ValueError):
                return None, f"actions[{i}].conditions[{j}].value must be a number."
            raw_hold = cond.get("hold_sec", 0)
            try:
                hold_sec = float(raw_hold or 0)
            except (TypeError, ValueError):
                return None, f"actions[{i}].conditions[{j}].hold_sec must be a number of seconds."
            if not 0 <= hold_sec <= 600:
                return None, f"actions[{i}].conditions[{j}].hold_sec must be between 0 and 600 seconds."
            conditions.append(
                TacticCondition(field=cfield, condition=ccond, value=cvalue, hold_sec=hold_sec)
            )

        trail = raw.get("trail", True)
        if not isinstance(trail, bool):
            return None, f"actions[{i}].trail must be a boolean."
        actions.append(
            TacticAction(
                action=action,
                quantity=quantity,
                quantity_pct=quantity_pct,
                conditions=conditions,
                note=str(raw.get("note") or ""),
                trail=trail,
            )
        )

    tactics = Tactics(
        ts=clock.now().isoformat(),
        symbol=symbol,
        reasoning=reasoning,
        actions=actions,
    )
    return tactics, None


def format_condition(cond: TacticCondition) -> str:
    # Same {field, condition, value} shape as a wake-up alert.
    text = format_alert({"field": cond.field, "condition": cond.condition, "value": cond.value})
    if cond.hold_sec:
        text += f" held {cond.hold_sec:g}s"
    return text


def format_tactic_action(action: TacticAction, symbol: "str | None" = None) -> str:
    """One-line human-readable description, e.g.
    'buy 10 sh AAPL when last_price below 180 and vix below 20'."""
    if action.quantity is not None:
        size = f"{action.quantity:g} sh"
    else:
        of = "position" if action.action == "sell" else "cash"
        size = f"{action.quantity_pct:g}% of {of}"
    conds = " and ".join(format_condition(c) for c in action.conditions)
    sym = f" {symbol}" if symbol else ""
    text = f"{action.action} {size}{sym} when {conds}"
    if action.note:
        text += f" ({action.note})"
    return text


def tactics_summaries(tactics: "Tactics | None") -> list[str]:
    """Human-readable one-liner per armed action; [] when nothing is armed."""
    if tactics is None:
        return []
    return [format_tactic_action(a, symbol=tactics.symbol) for a in tactics.actions]


def tactic_price_levels(tactics: "Tactics | None") -> list[dict]:
    """Price-axis condition levels of the armed tactics, shaped for drawing as
    horizontal lines on the Live chart (like pending alerts). Conditions on
    non-price fields (volume, vix, momentum, ...) have no price level and are
    skipped."""
    if tactics is None:
        return []
    levels: list[dict] = []
    for action in tactics.actions:
        for cond in action.conditions:
            if cond.field not in PRICE_AXIS_ALERT_FIELDS:
                continue
            if action.quantity is not None:
                size = f"{action.quantity:g}sh"
            else:
                size = f"{action.quantity_pct:g}%"
            levels.append(
                {
                    "action": action.action,
                    "label": f"{action.action} {size}",
                    "field": cond.field,
                    "condition": cond.condition,
                    "value": cond.value,
                }
            )
    return levels


def fetch_vix_level() -> "float | None":
    """Latest VIX close from the cached broad-market fetch, or None if unavailable."""
    try:
        series = historical.fetch_market_indicators().get("vix")
    except Exception:
        return None
    if series is None or len(series) == 0:
        return None
    return float(series.iloc[-1])


class TacticsExecutor:
    """Background matching engine for one symbol's armed `state.tactics`.

    Each streamed symbol gets its own executor. The stream calls `notify()` on
    every tick of that symbol so a triggered condition executes within
    milliseconds of the move; a slow fallback poll (`poll_sec`) keeps the
    non-stream fields (vix, momentum) and REST-fallback sessions covered. Trades
    go through the same shared `DecisionTracker` as the agent's own decisions --
    the executor never picks its own fill price -- and every execution wakes the
    agent to reevaluate the situation.
    """

    def __init__(self, state: "SymbolState", tracker: "DecisionTracker", poll_sec: float = TACTICS_POLL_SEC) -> None:
        self.state = state
        self.tracker = tracker
        self.poll_sec = poll_sec
        self._check_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: "threading.Thread | None" = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._check_event.set()

    def notify(self) -> None:
        """Nudge the executor to re-check conditions now (called on stream ticks)."""
        self._check_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_event.wait(timeout=self.poll_sec)
            self._check_event.clear()
            if self._stop_event.is_set():
                return
            try:
                self.check_now()
            except Exception as exc:
                logger.warning("Tactics evaluation failed: %s", exc)

    def condition_value(self, cond_field: str) -> "float | None":
        if cond_field == "vix":
            return fetch_vix_level()
        # momentum_pct and the other derived fields are handled by
        # alert_field_value, which computes them from the same live state.
        return alert_field_value(self.state, cond_field)

    def _condition_met(self, cond: TacticCondition) -> bool:
        ok = compare(self.condition_value(cond.field), cond.condition, cond.value)
        if not cond.hold_sec:
            return ok
        # hold_sec: the comparison must hold continuously. Track when it first
        # held; any tick that fails it resets the clock. clock.monotonic() is
        # the wall clock live and the simulated clock under simlab, so hold
        # timers measure tape time either way.
        now = clock.monotonic()
        if not ok:
            cond.met_since = None
            return False
        if cond.met_since is None:
            cond.met_since = now
        return (now - cond.met_since) >= cond.hold_sec

    def _conditions_met(self, action: TacticAction) -> bool:
        # Evaluate every condition (no short-circuit) so each hold_sec timer
        # keeps tracking even while a sibling condition is unmet.
        results = [self._condition_met(c) for c in action.conditions]
        return all(results)

    def check_now(self) -> "TacticAction | None":
        """Evaluate the armed tactics once; execute and return the first action
        whose conditions all hold, or None when nothing fired. When nothing
        fired, trail any protective sell stops up behind the price (the ratchet
        runs after evaluation so a price touching the take-profit level fires
        the take-profit, never a stop ratcheted onto the same level)."""
        tactics = self.state.tactics
        if tactics is None or tactics.status != "armed":
            return None
        for action in tactics.actions:
            if self._conditions_met(action) and self._executable_now(action):
                self._execute(tactics, action)
                return action
        self._ratchet_trailing_stop(tactics)
        return None

    def _executable_now(self, action: TacticAction) -> bool:
        """Whether this action could actually transact at this moment.

        A protective stop armed alongside an entry trigger is DORMANT, not
        live, for as long as the agent is flat: it exists to cover the position
        the entry will open. Firing it while flat transacts nothing, and -- as
        `_execute` disarms the whole set -- used to destroy the entry trigger
        sitting next to it, so a breakout buy could be wiped out by its own
        stop before the breakout ever happened. Skipping it here leaves both
        armed: the stop stays available for the fill that is still to come.
        """
        snap = self.tracker.snapshot()
        if action.action == "sell":
            return snap["positions"].get(self.state.symbol, 0.0) > 0
        return (snap["cash"] - self.tracker.trade_cost) > 0

    def _entry_price(self, snap: dict) -> "float | None":
        """Fill price of the most recent filled buy in this symbol, or None."""
        for decision in reversed(snap["decisions"]):
            if (
                decision.symbol == self.state.symbol
                and decision.action == "buy"
                and decision.status == "filled"
                and decision.price
            ):
                return float(decision.price)
        return None

    def _ratchet_trailing_stop(self, tactics: Tactics) -> None:
        """Trail armed protective sell stops up as the price advances toward
        the take-profit target.

        Applies while the armed actions bracket an open long position with a
        take-profit ("sell when last_price above target") and a stop ("sell
        when last_price below stop" armed BELOW the entry price). Whenever the
        high-water mark of last_price since arming covers a fraction of the
        entry->target distance, each stop is raised to cover the same fraction
        of its own distance to the target (stop = armed + fraction * (target -
        armed)) -- locking in gains while always staying below the high that
        produced it. The target never moves, stops only ever move up, and
        stops armed at/above the entry (a manual breakeven/structure stop) are
        left alone, as is any action armed with trail=False.

        The ratchet ENGAGES only once the trade has paid one full R -- the
        high-water mark has cleared entry + (entry - armed stop). Before that
        the stop stays exactly where it was armed: for a breakout bracket the
        armed stop sits below the broken range while the entry sits just above
        it, so an early proportional trail would drag the stop up INTO the
        broken level and turn a perfectly normal retest of it into a stop-out.
        """
        with self.state.lock:
            price = self.state.last_price
        if not price or price <= 0:
            return
        if tactics.high_water is None or price > tactics.high_water:
            tactics.high_water = price

        sells = [a for a in tactics.actions if a.action == "sell"]
        targets = [
            c.value
            for a in sells
            for c in a.conditions
            if c.field == "last_price" and c.condition == "above"
        ]
        if not targets:
            return
        target = max(targets)

        snap = self.tracker.snapshot()
        if snap["positions"].get(self.state.symbol, 0.0) <= 0:
            return
        entry = self._entry_price(snap)
        if entry is None or entry >= target or tactics.high_water <= entry:
            return
        progress = min(1.0, (tactics.high_water - entry) / (target - entry))

        for action in sells:
            if not action.trail:
                continue
            for cond in action.conditions:
                if cond.field != "last_price" or cond.condition != "below":
                    continue
                armed_at = cond.initial_value if cond.initial_value is not None else cond.value
                if armed_at >= entry:
                    continue
                # Engage only after +1R: high-water must clear entry by the
                # armed stop distance before this stop is allowed to move.
                if tactics.high_water < entry + (entry - armed_at):
                    continue
                new_stop = round(armed_at + progress * (target - armed_at), 4)
                if new_stop > cond.value:
                    cond.initial_value = armed_at
                    logger.info(
                        "%s trailing stop ratcheted %.4f -> %.4f (high %.4f covers %.0f%% of entry %.4f -> target %.4f)",
                        tactics.symbol, cond.value, new_stop, tactics.high_water, progress * 100, entry, target,
                    )
                    cond.value = new_stop

    def _resolve_quantity(self, action: TacticAction) -> float:
        if action.quantity is not None:
            return action.quantity
        snap = self.tracker.snapshot()
        frac = (action.quantity_pct or 0.0) / 100.0
        if action.action == "sell":
            return snap["positions"].get(self.state.symbol, 0.0) * frac
        # Percent-of-cash buy: sized off available cash at the last seen price;
        # record_trade re-fetches the fill price and clamps to affordable anyway.
        with self.state.lock:
            price = self.state.last_price
        if not price or price <= 0:
            return 0.0
        return max(0.0, (snap["cash"] - self.tracker.trade_cost)) * frac / price

    def _drop_action(self, tactics: Tactics, action: TacticAction,
                     summary: str, conds: str) -> None:
        """Retire one action that fired but could not transact, leaving the
        rest of the set armed.

        Dropping (rather than disarming everything) keeps a sibling entry
        trigger alive, and terminates: each pass removes one action, so a
        condition that stays true cannot spin. The agent is only woken once
        the set empties out and there is genuinely nothing left armed.
        """
        try:
            tactics.actions.remove(action)
        except ValueError:
            pass
        entry = {
            "type": "tactics_execution",
            "ts": clock.now().isoformat(),
            "action": action.action,
            "symbol": tactics.symbol,
            "tactic": summary,
            "triggered_by": conds,
            "status": "skipped",
            "error": "resolved quantity is 0 (no position to sell / no cash to buy)",
        }
        app = self.state.app
        with app.lock:
            app.agent_log.append(entry)
        logger.info("%s tactic skipped (nothing to transact): %s", tactics.symbol, summary)
        if not tactics.actions:
            tactics.status = "executed"
            self.state.tactics = None
            app.agent_wake_reason = (
                f"Tactics exhausted: {summary} could not transact and nothing remains armed."
            )
            app.agent_wake_event.set()

    def _execute(self, tactics: Tactics, action: TacticAction) -> None:
        state = self.state
        conds = " and ".join(format_condition(c) for c in action.conditions)
        summary = format_tactic_action(action, symbol=tactics.symbol)

        # Size the trade BEFORE disarming: an action that resolves to nothing
        # never executed, so it must not take the rest of the set down with it
        # (`_executable_now` filters the common case; this covers dust
        # positions and cash that rounds to no affordable share).
        quantity = self._resolve_quantity(action)
        if quantity <= 0:
            self._drop_action(tactics, action, summary, conds)
            return

        # Disarm first so a slow fill can't be double-triggered by the next tick.
        tactics.status = "executed"
        state.tactics = None

        decision = None
        error = None
        try:
            decision = self.tracker.record_trade(
                tactics.symbol,
                action.action,
                quantity,
                f"Tactics triggered ({conds}): {action.note or tactics.reasoning}",
                state.api_key,
                state.api_secret,
                state.feed,
            )
        except Exception as exc:
            error = str(exc)

        entry: dict = {
            "type": "tactics_execution",
            "ts": clock.now().isoformat(),
            "action": action.action,
            "symbol": tactics.symbol,
            "tactic": summary,
            "triggered_by": conds,
        }
        if decision is not None:
            entry.update(
                {
                    "status": decision.status,
                    "price": decision.price,
                    "quantity": decision.filled_quantity,
                }
            )
            outcome = (
                f"{decision.status} {decision.filled_quantity:g} sh @ ${decision.price:,.4f}"
                if decision.price is not None
                else decision.status
            )
        else:
            entry.update({"status": "error", "error": error})
            outcome = f"failed: {error}"
        app = state.app
        with app.lock:
            app.agent_log.append(entry)

        # Wake the agent to reevaluate with the fill (or failure) in hand. Any
        # remaining armed actions were disarmed above -- the agent re-sets what
        # still applies once it has seen the new position. (An action that
        # resolved to no quantity never reaches here; see `_drop_action`.)
        app.agent_wake_reason = f"Tactics executed: {summary} -> {outcome}."
        app.agent_wake_event.set()
