"""The rule language Apple Trader 2 is configured in.

Apple Trader (the first one) ships two *whole strategies*, and picking a model
picks which of them runs -- the momentum rules or the day-range rules, each with
its own fixed shape of entry and exit. That is a good way to reproduce a
notebook and a bad way to ask a question the notebook did not ask. There is no
way to buy the anticipated turn but exit on the day-range model's predicted
high, or to scale into a dip in three pieces, or to take half off at the target
and trail the rest, because none of those is one of the two shapes.

This module is the other end of that trade-off: instead of two strategies it
defines a **vocabulary**, and a strategy is whatever list of rules is written in
it.

    signal          a named number read off the tape, a model, the position or
                    the clock -- `bar.price`, `nbeats.turn_proba`,
                    `dayrange.pred_high_dip_adr`, `pos.drawdown_pct`. `SIGNALS`
                    is the whole catalogue and nothing outside it can be named.
    condition       one signal against one number: `above` (>=) or `below` (<=).
    action item     buy or sell, how much, and the conditions that arm it --
                    joined by AND (`all`) or OR (`any`).
    rule set        an ordered list of action items. The first one that both
                    matches and can transact fires, and at most one fires per
                    bar.

The catalogue is one thing; what is *readable on a given instrument* is
another. Every model here was fitted on specific symbols and none of the
notebooks claims transfer, so `signals_for(ticker)` is the list a builder
offers: on AAPL it is everything, on GOOGL and INTC it drops the two momentum
models (TimeToChange2 only ever ran on AAPL) and keeps the day-range forecast,
and on anything else it is the model-free half -- the bar, the momentum regime,
the position and the clock, which are computed from the tape and mean the same
thing on every symbol. `ruleset_error` re-checks the same fact, so a rule set
carried over from another instrument is refused with the signal named rather
than quietly running with that condition permanently unmet.

What that buys, concretely: the two shipped strategies both come out of it as
ordinary rule sets (see `PRESETS`), so the vocabulary is at least as expressive
as what it replaces -- and everything between and around them is now sayable.

Deliberate limits
-----------------
* **One joiner per item, not a boolean expression tree.** `cond AND cond OR
  cond` has no meaning without precedence rules, and precedence rules in a
  point-and-click builder are a trap. An item is all-of or any-of; anything that
  genuinely needs a mix is two items, which is also how it reads to a human.
* **Two operators.** `above` is >= and `below` is <=, the same pair
  `agent_stonks.tactics` and the wake-up alerts use, so one comparison means one
  thing everywhere in the app. Boolean signals (`mom.to_positive`) are 1/0, so
  "above 0.5" is "is true" -- `Signal.flag` marks them and the builder renders a
  toggle rather than a number.
* **Order is precedence.** Rules are not scored against each other; the first
  match wins. A protective exit written above a scale-in is checked first, which
  is the only sane reading of a list.
* **A missing signal is not a match.** A model that was not asked, a forecast
  that does not exist before 9:35, a P&L with no position -- all read as None,
  and a condition on None is unmet in an AND *and* in an OR. Rules never fire on
  a fabricated zero.

What a rule set does NOT contain is the flatten-before-close: that one stays a
property of the agent (`AppleTrader2Config.flatten_before_close_min`), because
every feature every model here reads is intraday and a rule set that forgot to
close the book overnight would be a bug wearing the costume of a strategy.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from . import apple_models, persistence_model

# --- conditions -------------------------------------------------------------

OP_ABOVE = "above"
OP_BELOW = "below"
OPS = (OP_ABOVE, OP_BELOW)
OP_SYMBOL = {OP_ABOVE: ">=", OP_BELOW: "<="}

JOIN_ALL = "all"
JOIN_ANY = "any"
JOINS = (JOIN_ALL, JOIN_ANY)
JOIN_WORD = {JOIN_ALL: "and", JOIN_ANY: "or"}

BUY = "buy"
SELL = "sell"
ACTIONS = (BUY, SELL)

# How much. The basis follows the side, which is what makes one percent mode
# enough: a buy spends cash and a sell delivers stock, so "50%" is unambiguous
# in both directions and nobody has to pick between two percent modes that are
# each wrong half the time.
SIZE_PCT = "pct"
SIZE_CASH = "cash"
SIZE_SHARES = "shares"
SIZE_MODES = (SIZE_PCT, SIZE_CASH, SIZE_SHARES)
SIZE_LABEL = {
    SIZE_PCT: "% (of cash on a buy, of the position on a sell)",
    SIZE_CASH: "$ (spend on a buy, sell that much stock)",
    SIZE_SHARES: "shares",
}

# Orders are placed in fractional shares, floored to this many decimals -- the
# same rounding `apple_trader._order_quantity` uses, so a 95%-of-cash rule here
# and the original agent's 95% position size buy the same quantity.
SHARE_DECIMALS = 4


@dataclass(frozen=True)
class Signal:
    """One number a condition can be written about."""

    key: str
    label: str
    group: str
    # What the number is measured in, for the builder's suffix and the log.
    unit: str
    help: str
    # The `apple_models` key whose bundle has to be loaded before this signal
    # can be read at all, or None for signals that come off the tape, the
    # position or the clock. This is what makes a rule set that names no model
    # runnable with nothing installed.
    model: Optional[str] = None
    # Whether the bundle must be able to forecast (as opposed to score a change
    # bar). Checked against the loaded bundle by `ruleset_error`.
    needs_forecast: bool = False
    # 1/0 valued: the builder offers true/false instead of a number.
    flag: bool = False
    # What the builder seeds a fresh condition's value with.
    default: float = 0.0


GROUP_BAR = "Bar"
GROUP_MOMENTUM = "Momentum"
GROUP_MODEL = "Model forecasts"
GROUP_DAYRANGE = "Day range"
GROUP_POSITION = "Position"
GROUP_CLOCK = "Clock"


def _tape_signals() -> "dict[str, Signal]":
    """The bar, the momentum regime, the position and the clock.

    None of these needs a saved model. The momentum ones need the *settings* a
    bundle carries, but those settings have shipped defaults, so a rule set
    written entirely out of this group runs on a machine with no model files at
    all -- which is the cheapest way to sanity-check the plumbing.
    """
    return {
        s.key: s
        for s in (
            Signal("bar.price", "Price (bar close)", GROUP_BAR, "$",
                   "Close of the minute bar that just finished -- the price every rule "
                   "is priced off and, near enough, the price an order fills at.",
                   default=100.0),
            Signal("bar.high", "Bar high", GROUP_BAR, "$",
                   "High of the bar that just closed. Use it to ask whether price "
                   "*reached* a level rather than closed through it.", default=100.0),
            Signal("bar.low", "Bar low", GROUP_BAR, "$",
                   "Low of the bar that just closed -- the touch a resting buy would "
                   "have filled on.", default=100.0),
            Signal("bar.session_change_pct", "Change on the day", GROUP_BAR, "%",
                   "Percent from the session's first bar open to this bar's close."),
            Signal("mom.score", "Momentum score", GROUP_MOMENTUM, "σ",
                   "The 15-bar return in units of its own random-walk sigma, smoothed. "
                   "+1 means price drifted up about one sigma more than a coin flip "
                   "would. This is the raw number the regime trigger runs on."),
            Signal("mom.regime", "Regime (-1/0/+1)", GROUP_MOMENTUM, "",
                   "Negative (-1), balanced (0) or positive (+1), from a Schmitt "
                   "trigger over the momentum score -- so a value hovering on the line "
                   "cannot emit a burst of fake changes. 'above 1' is 'positive'.",
                   default=1.0),
            Signal("mom.bars_in_regime", "Bars in this regime", GROUP_MOMENTUM, "bars",
                   "How long the current regime has run as of this bar.", default=15.0),
            Signal("mom.pre_dwell", "Bars the old regime held", GROUP_MOMENTUM, "bars",
                   "On a bar the regime changed on: how long the regime it left had "
                   "run. Absent on every other bar. This is the observable gate that "
                   "decides half of what the persistence models know.", default=15.0),
            Signal("mom.regime_change", "Regime changed on this bar", GROUP_MOMENTUM, "",
                   "1 on a bar the regime changed, 0 otherwise.", flag=True),
            Signal("mom.to_positive", "Changed INTO positive", GROUP_MOMENTUM, "",
                   "1 on a bar the regime changed into positive -- the bar the "
                   "notebook's rule buys.", flag=True),
            Signal("pos.shares", "Shares held", GROUP_POSITION, "sh",
                   "The open position, 0 while flat. A sell never needs to test it (a "
                   "sell that cannot transact is skipped anyway); an entry usually "
                   "does — 'at or below 0' is 'only while flat', which is what turns a "
                   "buy rule that would keep adding on every bar its signal holds into "
                   "a single entry."),
            Signal("pos.pnl_pct", "Open P&L", GROUP_POSITION, "%",
                   "Percent from the (average) entry price to this bar's close. "
                   "Absent while flat."),
            Signal("pos.drawdown_pct", "Give-back from the peak", GROUP_POSITION, "%",
                   "Percent below the highest price the position has traded at since "
                   "it was opened -- always <= 0, and the peak only ratchets up. "
                   "'below -0.5' is a 0.5% trailing stop.", default=-0.5),
            Signal("pos.since_buy_pct", "Change since the last buy (vs its peak)",
                   GROUP_POSITION, "%",
                   "Percent from the best price since the last buy filled to this bar's "
                   "close -- the reference is max(that fill price, every high since), so "
                   "this is always <= 0 and the reference only ratchets up. The "
                   "difference from the give-back above is what it does at the exit: "
                   "that one disappears the moment the book goes flat, this one keeps "
                   "measuring from the same buy, which is what lets a rule wait for a "
                   "move away from where it last traded before buying back in. Absent "
                   "until this agent has bought once, and reset each morning.",
                   default=-0.5),
            Signal("pos.bars_held", "Bars held", GROUP_POSITION, "bars",
                   "Bars since the position was opened. Absent while flat.",
                   default=10.0),
            Signal("pos.value", "Position value", GROUP_POSITION, "$",
                   "Shares held times this bar's close."),
            Signal("pos.cash", "Cash", GROUP_POSITION, "$",
                   "The ledger's uninvested cash."),
            Signal("clock.minutes_from_open", "Minutes since the open", GROUP_CLOCK,
                   "min", "Minutes since 09:30 ET, from the bar's own timestamp.",
                   default=30.0),
            Signal("clock.minutes_to_close", "Minutes to the close", GROUP_CLOCK, "min",
                   "Minutes until 16:00 ET. The flatten-before-close rule already "
                   "covers the last few; use this to stop opening new risk earlier.",
                   default=30.0),
        )
    }


def _model_signals() -> "dict[str, Signal]":
    """One namespace per saved model, so a condition names *which* model.

    `persistence.proba` and `nbeats.proba` are the same question put to two
    models, and a rule set may ask both -- the cost is one loaded bundle each,
    paid only if some enabled condition names it. That is the whole reason the
    keys are namespaced rather than there being a single `proba` plus a
    model-key setting on the agent: "model A predicts X" is the condition, not
    the configuration.

    Only forecasting models get `turn_proba` and `reversal_proba`, because a
    classifier fitted on regime-change bars has nothing to say about a bar that
    is not one. `apple_models.AppleModel.anticipates` is the cheap flag used
    here; `ruleset_error` re-checks it against the bundle that actually loaded.
    """
    out: "dict[str, Signal]" = {}
    for model in apple_models.MODELS.values():
        if model.strategy != apple_models.STRATEGY_MOMENTUM:
            continue
        out[f"{model.key}.proba"] = Signal(
            f"{model.key}.proba", f"{model.label}: change holds", GROUP_MODEL, "prob",
            "On a bar the regime changed INTO positive: the model's probability that "
            "the change persists. Absent on every other bar -- this is the one "
            "question the models were fitted on.",
            model=model.key, default=0.5,
        )
        if not model.anticipates:
            continue
        out[f"{model.key}.turn_proba"] = Signal(
            f"{model.key}.turn_proba", f"{model.label}: turns positive next bar",
            GROUP_MODEL, "prob",
            "While the regime is still negative or balanced: the forecast probability "
            "that it turns positive on the very next bar and then holds. Absent once "
            "the regime IS positive. Buying on this is the original agent's "
            "'anticipate' entry, and it is selective -- most bars score near zero.",
            model=model.key, needs_forecast=True, default=0.05,
        )
        out[f"{model.key}.reversal_proba"] = Signal(
            f"{model.key}.reversal_proba", f"{model.label}: flips negative",
            GROUP_MODEL, "prob",
            "While the regime is positive: the forecast probability that it flips all "
            "the way to negative somewhere inside the 15-bar horizon. Read it as an "
            "early warning that the positive run is ending rather than a forecast of a "
            "downtrend -- on measured tape the literal flip almost never prints. Costs "
            "a full forecast on every positive bar it is asked on.",
            model=model.key, needs_forecast=True, default=0.30,
        )
    return out


def _dayrange_signals() -> "dict[str, Signal]":
    """TimeToChange3's one forecast, and the distances that make it a rule.

    The raw `pred_high` is a dollar price and comparing it to a constant is
    almost never what a rule wants. The three `_adr` distances are the same
    forecast expressed as "how far under the predicted high did this bar get,
    in average daily ranges" -- which is exactly the form notebook 05's rule is
    written in, and it scales with how wide the sessions have been instead of
    needing a new number every time AAPL moves $20.

    All of them are absent until the opening window closes at 9:35, and for the
    whole session if the forecast cannot be made.
    """
    key = apple_models.DAYRANGE_KEY
    return {
        s.key: s
        for s in (
            Signal(f"{key}.pred_high", "Predicted session high", GROUP_DAYRANGE, "$",
                   "Where the model expects today's high to land, forecast once at "
                   "9:35 and fixed for the day.", model=key, default=100.0),
            Signal(f"{key}.pred_low", "Predicted session low", GROUP_DAYRANGE, "$",
                   "The same forecast's low.", model=key, default=100.0),
            Signal(f"{key}.adr", "Average daily range", GROUP_DAYRANGE, "$",
                   "The trailing 14-day average daily range in dollars -- the unit the "
                   "distances below are measured in.", model=key, default=1.0),
            Signal(f"{key}.pred_high_dip_adr", "Dip below predicted high (bar low)",
                   GROUP_DAYRANGE, "×ADR",
                   "(predicted high − this bar's LOW) / ADR: how deep under the "
                   "predicted high the bar traded. The notebook's buy is 'above 0.75'.",
                   model=key, default=0.75),
            Signal(f"{key}.pred_high_reach_adr", "Reach toward predicted high (bar high)",
                   GROUP_DAYRANGE, "×ADR",
                   "(predicted high − this bar's HIGH) / ADR: how close to the "
                   "predicted high the bar got -- smaller is closer. The notebook's "
                   "sell is 'below 0.10'.", model=key, default=0.10),
            Signal(f"{key}.pred_high_gap_adr", "Gap to predicted high (close)",
                   GROUP_DAYRANGE, "×ADR",
                   "(predicted high − this bar's CLOSE) / ADR. The close-based version "
                   "of the two above: a bar has to finish through the level, not just "
                   "wick through it.", model=key, default=0.75),
            Signal(f"{key}.pred_low_gap_adr", "Gap above predicted low (close)",
                   GROUP_DAYRANGE, "×ADR",
                   "(this bar's CLOSE − predicted low) / ADR. Near zero means the day "
                   "is at the bottom of where the model expected it to trade.",
                   model=key, default=0.25),
        )
    }


SIGNALS: "dict[str, Signal]" = {
    **_tape_signals(),
    **_model_signals(),
    **_dayrange_signals(),
}

# The builder renders the catalogue in this order.
SIGNAL_GROUPS = (
    GROUP_BAR, GROUP_MOMENTUM, GROUP_MODEL, GROUP_DAYRANGE, GROUP_POSITION, GROUP_CLOCK,
)

# The symbol every ticker-aware entry point here defaults to, so the whole
# module still reads as "AAPL, unless told otherwise".
DEFAULT_TICKER = apple_models.DEFAULT_TICKER


def readable_on(key: str, ticker: "str | None" = None) -> bool:
    """Whether one signal can be read on this instrument.

    True for every tape, momentum, position and clock signal on every symbol --
    those are computed here, from bars. False for a model signal on a symbol
    that model was not fitted on, which is the whole ticker dimension.
    """
    spec = SIGNALS.get(key)
    if spec is None:
        return False
    return spec.model is None or apple_models.covers(spec.model, ticker)


def signals_for(ticker: "str | None" = None) -> "dict[str, Signal]":
    """The catalogue as it applies to one instrument, in catalogue order."""
    return {key: spec for key, spec in SIGNALS.items() if readable_on(key, ticker)}


def signals_in(group: str, ticker: "str | None" = None) -> "list[Signal]":
    return [s for s in signals_for(ticker).values() if s.group == group]


def signal(key: str) -> "Signal | None":
    return SIGNALS.get(key)


def signal_label(key: str) -> str:
    found = SIGNALS.get(key)
    return found.label if found else key


# --- the rule set ------------------------------------------------------------


@dataclass
class Condition:
    """One signal against one number."""

    field: str
    op: str = OP_ABOVE
    value: float = 0.0

    def __post_init__(self) -> None:
        self.value = float(self.value)


@dataclass
class ActionItem:
    """One rule: what to do, how much of it, and when.

    `cooldown_bars` is the one field that is about the *list* rather than about
    a single decision. Without it a rule like "buy 10% of cash while the model
    is above 0.4" fires on every bar the condition holds, laddering the whole
    balance in over ten minutes -- which is a real strategy, but it should be
    one somebody chose rather than the accidental consequence of writing a
    condition that stays true. Setting it to the length of a session makes a
    rule fire once a day.
    """

    action: str = BUY
    size_mode: str = SIZE_PCT
    size: float = 95.0
    conditions: "list[Condition]" = field(default_factory=list)
    join: str = JOIN_ALL
    # Free text, carried into the ledger's reasoning so a fill can be read back
    # to the rule that caused it.
    label: str = ""
    enabled: bool = True
    cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if self.size_mode not in SIZE_MODES:
            raise ValueError(f"size_mode must be one of {SIZE_MODES}, got {self.size_mode!r}")
        if self.join not in JOINS:
            raise ValueError(f"join must be one of {JOINS}, got {self.join!r}")
        if self.size <= 0:
            raise ValueError(f"size must be > 0, got {self.size!r}")
        if self.size_mode == SIZE_PCT and self.size > 100:
            raise ValueError(f"a percentage size must be <= 100, got {self.size!r}")
        self.conditions = [
            c if isinstance(c, Condition) else Condition(**c) for c in self.conditions
        ]
        for cond in self.conditions:
            if cond.field not in SIGNALS:
                raise ValueError(
                    f"unknown signal {cond.field!r}; see apple_rules.SIGNALS for the "
                    "catalogue"
                )
            if cond.op not in OPS:
                raise ValueError(f"op must be one of {OPS}, got {cond.op!r}")
        self.size = float(self.size)
        self.cooldown_bars = int(self.cooldown_bars)


@dataclass
class RuleSet:
    """An ordered list of action items -- the whole strategy.

    An empty list is allowed and means "never trade": a rule set is edited
    incrementally in a UI, and refusing to construct the half-built ones would
    make the builder fight the user. `ruleset_error` is what stops an empty or
    unrunnable set from being *launched*.
    """

    items: "list[ActionItem]" = field(default_factory=list)

    def __post_init__(self) -> None:
        self.items = [
            i if isinstance(i, ActionItem) else ActionItem(**i) for i in self.items
        ]

    @property
    def active(self) -> "list[ActionItem]":
        return [i for i in self.items if i.enabled]

    def fields(self) -> "list[str]":
        """Every signal any enabled rule reads, in catalogue order.

        This is what decides which models get loaded and which numbers the log
        line reports, so a disabled rule costs nothing at all.
        """
        used = {c.field for item in self.active for c in item.conditions}
        return [key for key in SIGNALS if key in used]

    def models(self) -> "list[str]":
        """The `apple_models` keys whose bundles this rule set needs loaded."""
        needed = {SIGNALS[f].model for f in self.fields()}
        return [k for k in apple_models.keys() if k in needed]

    def unreadable_on(self, ticker: "str | None") -> "list[str]":
        """The signals this rule set names that do not exist for this symbol.

        Non-empty means the set was written for a different instrument: a
        `nbeats.turn_proba` condition carried onto GOOGL would never be met and
        the run would finish clean and empty, which reads like a strategy result.
        `ruleset_error` turns this into a refusal.
        """
        return [key for key in self.fields() if not readable_on(key, ticker)]

    def reads_momentum(self) -> bool:
        """Whether anything here depends on the momentum pipeline warming up.

        A rule set written on price, the position and the clock is decidable on
        the session's first bar; one that reads a regime or a model is not, and
        the agent reports itself as warming up rather than as holding. Only the
        status tag depends on this -- an unwarmed signal is None either way, and
        None never fires a rule.
        """
        for key in self.fields():
            spec = SIGNALS[key]
            if key.startswith("mom."):
                return True
            if spec.model is not None and apple_models.is_momentum(spec.model):
                return True
        return False

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, raw: "dict | None") -> "RuleSet":
        return cls(**(raw or {}))


# --- evaluation --------------------------------------------------------------

# A rule set is evaluated against one of these: signal key -> current value, or
# None where the signal does not apply to this bar. `apple_trader2.SignalBus` is
# the live implementation; tests pass a dict's `.get`.
Values = Callable[[str], Optional[float]]


def condition_met(cond: Condition, values: Values) -> bool:
    """Whether one condition holds right now.

    A signal that is absent is never a match -- not in an AND and not in an OR.
    See the module docstring: the alternative is a rule firing on a model that
    was never asked.
    """
    current = values(cond.field)
    if current is None:
        return False
    if cond.op == OP_ABOVE:
        return current >= cond.value
    return current <= cond.value


def item_matches(item: ActionItem, values: Values) -> bool:
    """Whether an action item's conditions are satisfied.

    A rule with no conditions never fires. "Always buy" is not a strategy this
    agent should let somebody write by accident, and it is one deleted condition
    away from every rule set in the builder.
    """
    if not item.conditions:
        return False
    met = (condition_met(c, values) for c in item.conditions)
    return all(met) if item.join == JOIN_ALL else any(met)


def resolve_quantity(
    item: ActionItem, price: float, cash: float, shares: float
) -> float:
    """Shares this rule would trade right now -- 0 when it cannot transact.

    Every mode is clipped to what the ledger can actually do, so a rule is
    allowed to over-ask: "sell 200 shares" out of a 50-share position sells 50,
    and "spend $5,000" out of $300 spends $300. The clip is what makes a rule
    set portable across starting balances, and 0 here is what makes a rule
    dormant rather than an error -- a sell is simply skipped while flat, exactly
    as an armed tactic is.
    """
    if price <= 0:
        return 0.0
    if item.action == BUY:
        if item.size_mode == SIZE_PCT:
            budget = cash * item.size / 100.0
        elif item.size_mode == SIZE_CASH:
            budget = min(item.size, cash)
        else:
            budget = min(item.size * price, cash)
        return _floor_shares(max(budget, 0.0) / price)
    if shares <= 0:
        return 0.0
    if item.size_mode == SIZE_PCT:
        wanted = shares * item.size / 100.0
    elif item.size_mode == SIZE_CASH:
        wanted = item.size / price
    else:
        wanted = item.size
    # A sell that would leave a sliver behind takes the sliver with it: a
    # rounding remainder of 0.0001 shares is not a position anybody meant to
    # hold, and leaving it makes "sell 100%" look like it did not work.
    if wanted >= shares - 10 ** -SHARE_DECIMALS:
        return shares
    return _floor_shares(max(wanted, 0.0))


def _floor_shares(quantity: float) -> float:
    return math.floor(quantity * 10**SHARE_DECIMALS) / 10**SHARE_DECIMALS


@dataclass
class Match:
    """The rule that fired, and the size it resolved to."""

    index: int
    item: ActionItem
    quantity: float


def first_match(
    ruleset: RuleSet,
    values: Values,
    price: float,
    cash: float,
    shares: float,
    blocked: "Callable[[int], bool] | None" = None,
) -> "Match | None":
    """The first enabled rule that matches AND can transact, or None.

    Three ways a rule is passed over, and they are different: it is disabled, or
    its conditions do not hold, or it holds but resolves to no shares (a sell
    while flat, a buy with no cash). The third is the one worth being explicit
    about -- it means a protective sell written at the top of the list does not
    swallow the bar while the book is flat, so the entry below it still gets its
    turn. `blocked` is the cooldown, asked per index.
    """
    for index, item in enumerate(ruleset.items):
        if not item.enabled:
            continue
        if blocked is not None and blocked(index):
            continue
        if not item_matches(item, values):
            continue
        quantity = resolve_quantity(item, price, cash, shares)
        if quantity <= 0:
            continue
        return Match(index=index, item=item, quantity=quantity)
    return None


# --- validation --------------------------------------------------------------


def ruleset_error(
    ruleset: RuleSet,
    bundles: "dict[str, dict | None]",
    ticker: "str | None" = None,
) -> "str | None":
    """Why this rule set cannot run, or None if it can.

    Checked once before a run starts rather than per bar, for the reason
    `apple_trader.config_error` gives: every failure here would otherwise be
    silent. A model that did not load leaves its signals as None, None never
    matches, and the run finishes clean with an empty ledger that reads like a
    strategy result instead of a missing file.

    `bundles` is what `apple_models.load` returned for each key the rule set
    named -- None for one that could not be assembled. `ticker` is the
    instrument the run is configured for, which decides which signals exist at
    all.
    """
    symbol = (ticker or DEFAULT_TICKER).upper()
    if not ruleset.active:
        return (
            "This rule set has no enabled rules, so the agent would watch the tape and "
            "never trade. Add at least one buy."
        )
    empty = [
        i + 1 for i, item in enumerate(ruleset.items) if item.enabled and not item.conditions
    ]
    if empty:
        plural = "s" if len(empty) > 1 else ""
        return (
            f"Rule{plural} {', '.join(str(n) for n in empty)} ha{'ve' if plural else 's'} "
            "no conditions, so nothing can ever arm "
            f"{'them' if plural else 'it'}. Add a condition or disable the rule."
        )
    if not any(item.action == BUY for item in ruleset.active):
        return (
            "This rule set can only sell, so it can never open the position its sells "
            "are about. Add a buy rule."
        )

    # Before anything is loaded: signals that do not exist on this instrument.
    # This is the check a rule set carried over from another symbol trips, and
    # it comes first because "there is no such model for GOOGL" is a different
    # problem from "the GOOGL model is not installed" and the second message
    # would send the reader looking for a file that was never meant to exist.
    unreadable = ruleset.unreadable_on(symbol)
    if unreadable:
        # One model per message, even when two are missing: the fix is per
        # model, and the second one surfaces on the next check.
        key = SIGNALS[unreadable[0]].model
        model = apple_models.get(key)
        named = ", ".join(
            f"`{f}`" for f in unreadable if SIGNALS[f].model == key
        )
        return (
            f"{named} cannot be read on {symbol}: {model.label} was fitted on "
            f"{', '.join(model.tickers)} only. Drop those conditions, or switch the "
            f"instrument back to one the model covers."
        )

    for key in ruleset.models():
        bundle = bundles.get(key)
        if bundle is None:
            named = ", ".join(
                f"`{f}`" for f in ruleset.fields() if SIGNALS[f].model == key
            )
            return (
                f"{apple_models.unavailable_reason(key, symbol)} {named} cannot be read."
            )
        for field_key in ruleset.fields():
            spec = SIGNALS[field_key]
            if spec.model != key or not spec.needs_forecast:
                continue
            if not persistence_model.anticipates(bundle):
                return (
                    f"{apple_models.get(key).label} was fitted on regime-change bars "
                    f"only, so it cannot answer `{field_key}` -- that question is about "
                    "a bar which is not a change. Use a forecasting model's signal "
                    "instead."
                )
    return None


# --- rendering ---------------------------------------------------------------


def format_condition(cond: Condition) -> str:
    spec = SIGNALS.get(cond.field)
    if spec is not None and spec.flag:
        return f"{cond.field} is {'true' if cond.op == OP_ABOVE else 'false'}"
    return f"{cond.field} {OP_SYMBOL[cond.op]} {cond.value:g}"


def format_size(item: ActionItem) -> str:
    if item.size_mode == SIZE_PCT:
        return f"{item.size:g}% of {'cash' if item.action == BUY else 'the position'}"
    if item.size_mode == SIZE_CASH:
        return f"${item.size:,.0f}"
    return f"{item.size:g} sh"


def format_item(item: ActionItem) -> str:
    """One line: 'buy 95% of cash when nbeats.turn_proba >= 0.05'."""
    joiner = f" {JOIN_WORD[item.join]} "
    conds = joiner.join(format_condition(c) for c in item.conditions) or "never"
    text = f"{item.action} {format_size(item)} when {conds}"
    if item.cooldown_bars:
        text += f" [then wait {item.cooldown_bars} bars]"
    if item.label:
        text += f" ({item.label})"
    if not item.enabled:
        text = f"[off] {text}"
    return text


def describe(ruleset: RuleSet) -> "list[str]":
    """One line per rule, in the order they are checked."""
    return [f"{i + 1}. {format_item(item)}" for i, item in enumerate(ruleset.items)]


def _compact(item: ActionItem) -> str:
    size = (
        f"{item.size:g}%"
        if item.size_mode == SIZE_PCT
        else (f"${item.size:g}" if item.size_mode == SIZE_CASH else f"{item.size:g}sh")
    )
    joiner = "&" if item.join == JOIN_ALL else "|"
    conds = joiner.join(
        f"{c.field}{OP_SYMBOL[c.op]}{c.value:g}" for c in item.conditions
    )
    return f"{item.action} {size} if {conds}"


# Beyond this, a signature stops being something anybody reads in a results
# table and becomes a wall -- so it collapses to a count plus a digest, which
# still identifies the rule set exactly.
_SIGNATURE_MAX = 180


def signature(ruleset: RuleSet, ticker: str) -> str:
    """The compact identity SimLab groups and de-duplicates runs on.

    Stands where `provider/model` stands for the LLM agents, and where
    `apple_trader.config_signature`'s threshold string stands for the first
    Apple Trader. Two rule sets that differ by one number are two strategies to
    test, so the numbers are in the string; the hash fallback keeps that true
    for the long ones without making Results unreadable.
    """
    body = "; ".join(_compact(i) for i in ruleset.active)
    if not body:
        body = "no rules"
    if len(body) > _SIGNATURE_MAX:
        body = f"{len(ruleset.active)} rules #{digest(ruleset)}"
    return f"apple2_{ticker}({body})"


def digest(ruleset: RuleSet) -> str:
    """A short stable hash of the enabled rules -- the same rules always hash
    the same, and a disabled rule never changes it."""
    payload = "|".join(_compact(i) for i in ruleset.active)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# --- presets -----------------------------------------------------------------
#
# The two strategies Apple Trader ships (three configurations of them), written
# in this vocabulary, plus one that could not be written before. They are here
# to be loaded and edited: a strategy nobody can reproduce is a strategy nobody
# can compare against, and these are the comparison.
#
# All three reproductions were checked rather than asserted: replayed against
# `apple_trader` on the 2026-07-27 SIP session, through the real saved bundles,
# each preset produced the *same trades* -- same minutes, same quantities, same
# fills (6, 6 and 2 orders respectively). If one of these is edited, that is the
# check to re-run before believing the new one is still a reproduction.


def _momentum_anticipate() -> RuleSet:
    """Apple Trader's default rules: `anticipate`, 0.5% trail, reversal exit.

    Exactly the shipped configuration -- `nbeats,anticipate,p>=0.05,trail=0.5%,
    rev>=0.30,size=95%` -- and it is worth reading as three separate claims,
    because in this form they can be switched off one at a time. Sell first, buy
    second: an exit that lost a race with an entry would re-enter the position
    it was closing.

    The `pos.shares <= 0` condition on the entry is what the original agent gets
    for free by checking the position before it ever looks at a signal. Without
    it this is a subtly different strategy -- 95% of cash, then 95% of what is
    left, on every later bar the forecast clears the threshold -- so it is
    written down rather than assumed.
    """
    return RuleSet(
        items=[
            ActionItem(
                action=SELL, size_mode=SIZE_PCT, size=100.0, join=JOIN_ANY,
                label="Trailing stop or forecast reversal",
                conditions=[
                    Condition("pos.drawdown_pct", OP_BELOW, -0.5),
                    Condition("nbeats.reversal_proba", OP_ABOVE, 0.30),
                ],
            ),
            ActionItem(
                action=BUY, size_mode=SIZE_PCT, size=95.0,
                label="Buy the anticipated turn, while flat",
                conditions=[
                    Condition("pos.shares", OP_BELOW, 0.0),
                    Condition("nbeats.turn_proba", OP_ABOVE, 0.05),
                ],
            ),
        ]
    )


def _momentum_confirm() -> RuleSet:
    """The notebook's rule: buy the printed change, trail out.

    `mom.to_positive` is not redundant next to `persistence.proba` -- the
    probability is only ever non-None on a change bar, so the flag adds nothing
    to the *machine*. It is in the rule because the rule is meant to be read,
    and "buy when the regime turned positive and the model rates it 0.07+" is
    what this strategy is.
    """
    return RuleSet(
        items=[
            ActionItem(
                action=SELL, size_mode=SIZE_PCT, size=100.0,
                label="Trailing stop",
                conditions=[Condition("pos.drawdown_pct", OP_BELOW, -0.5)],
            ),
            ActionItem(
                action=BUY, size_mode=SIZE_PCT, size=95.0,
                label="Buy the confirmed change, while flat",
                conditions=[
                    Condition("pos.shares", OP_BELOW, 0.0),
                    Condition("mom.to_positive", OP_ABOVE, 0.5),
                    Condition("persistence.proba", OP_ABOVE, 0.07),
                ],
            ),
        ]
    )


def _dayrange_levels() -> RuleSet:
    """TimeToChange3's two resting levels, as two rules.

    The distances are the notebook's (0.75 and 0.10 average daily ranges below
    the predicted high) and the triggers are the notebook's too: the buy reads
    the bar's low and the sell reads its high, so a bar that only wicked through
    the level still counts -- which is what a resting limit order would have
    done.
    """
    key = apple_models.DAYRANGE_KEY
    return RuleSet(
        items=[
            ActionItem(
                action=SELL, size_mode=SIZE_PCT, size=100.0,
                label="Sell just under the predicted high",
                conditions=[Condition(f"{key}.pred_high_reach_adr", OP_BELOW, 0.10)],
            ),
            ActionItem(
                action=BUY, size_mode=SIZE_PCT, size=95.0,
                label="Buy the dip below the predicted high, while flat",
                conditions=[
                    Condition("pos.shares", OP_BELOW, 0.0),
                    Condition(f"{key}.pred_high_dip_adr", OP_ABOVE, 0.75),
                ],
            ),
        ]
    )


def _scale_in_take_half() -> RuleSet:
    """Neither of the above: scale in twice, take half off, trail the rest.

    Not a recommendation and not measured -- it is here because it is the
    shortest rule set that says something the first Apple Trader structurally
    cannot, and a vocabulary is worth judging on what it adds. Three things it
    adds: a partial exit (50% of the position), an entry sized to a fraction of
    cash so a second one can follow, and a cooldown so the second entry is a
    decision rather than the same bar's condition still being true.
    """
    return RuleSet(
        items=[
            ActionItem(
                action=SELL, size_mode=SIZE_PCT, size=100.0,
                label="Stop out the rest",
                conditions=[Condition("pos.drawdown_pct", OP_BELOW, -0.6)],
            ),
            ActionItem(
                action=SELL, size_mode=SIZE_PCT, size=50.0, cooldown_bars=390,
                label="Take half off into a 0.4% gain",
                conditions=[Condition("pos.pnl_pct", OP_ABOVE, 0.4)],
            ),
            ActionItem(
                action=BUY, size_mode=SIZE_PCT, size=40.0, cooldown_bars=5,
                label="Scale into the anticipated turn",
                conditions=[Condition("nbeats.turn_proba", OP_ABOVE, 0.05)],
            ),
        ]
    )


def _tape_turn_trail() -> RuleSet:
    """The same idea as the notebook's rule with the model taken out: buy the
    printed change into positive, trail out.

    The one preset that names no model, and therefore the one that runs on
    *any* instrument. What it drops is the whole question the models answer --
    whether this particular change is likely to hold — so it takes every change
    into positive rather than the rated ones, which is a materially different
    (and untested) strategy rather than a cheaper version of the same one. It is
    here because an instrument with no saved model still needs somewhere to
    start, and because it makes the models' contribution measurable: run this
    against `Momentum — confirm the turn` on AAPL and the difference is what the
    classifier is worth.
    """
    return RuleSet(
        items=[
            ActionItem(
                action=SELL, size_mode=SIZE_PCT, size=100.0,
                label="Trailing stop",
                conditions=[Condition("pos.drawdown_pct", OP_BELOW, -0.5)],
            ),
            ActionItem(
                action=BUY, size_mode=SIZE_PCT, size=95.0,
                label="Buy the change into positive, while flat",
                conditions=[
                    Condition("pos.shares", OP_BELOW, 0.0),
                    Condition("mom.to_positive", OP_ABOVE, 0.5),
                    Condition("mom.pre_dwell", OP_ABOVE, 15.0),
                ],
            ),
        ]
    )


PRESETS: "dict[str, Callable[[], RuleSet]]" = {
    "Momentum — anticipate the turn (Apple Trader's default)": _momentum_anticipate,
    "Momentum — confirm the turn (the notebook's rule)": _momentum_confirm,
    "Day range — two levels below the predicted high": _dayrange_levels,
    "Scale in, take half off, trail the rest": _scale_in_take_half,
    "Tape only — buy the regime turn, trail out (no model)": _tape_turn_trail,
}

DEFAULT_PRESET = "Momentum — anticipate the turn (Apple Trader's default)"
# What every instrument can run, whatever is installed and whatever was fitted.
MODEL_FREE_PRESET = "Tape only — buy the regime turn, trail out (no model)"


def presets_for(ticker: "str | None" = None) -> "list[str]":
    """The presets whose signals all exist on this instrument.

    A preset is a starting point, and one that cannot run on the instrument it
    is offered for is a trap rather than a starting point -- on GOOGL that is
    the day-range pair and the model-free one, on an unmodelled symbol only the
    model-free one.
    """
    return [
        name for name, build in PRESETS.items()
        if not build().unreadable_on(ticker)
    ]


def default_preset(ticker: "str | None" = None) -> str:
    """The preset a fresh builder starts on for this instrument."""
    available = presets_for(ticker)
    if DEFAULT_PRESET in available:
        return DEFAULT_PRESET
    return available[0] if available else MODEL_FREE_PRESET


def preset(name: "str | None" = None, ticker: "str | None" = None) -> RuleSet:
    """A named preset, freshly built (never a shared mutable instance).

    An unknown name -- or one that does not apply to this instrument -- falls
    back to `default_preset`, which is what makes switching the instrument in a
    picker a safe operation rather than one that can leave the builder holding
    a rule set it refuses to run.
    """
    build = PRESETS.get(name or "")
    if build is None or build().unreadable_on(ticker):
        build = PRESETS[default_preset(ticker)]
    return build()
