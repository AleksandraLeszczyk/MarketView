"""Apple Trader 2 -- the same loop, with the strategy taken out of the code.

Apple Trader runs one of two hard-coded strategies and picks between them by
picking a model. This one runs whatever list of rules it is handed. The tape,
the ledger, the fill path, the flatten-before-close and the once-a-minute
cadence are identical; the only difference is where the decision comes from --
`agent_stonks.apple_rules`, which defines the vocabulary, and a `RuleSet`, which
is the strategy.

    signals   `SignalBus` turns the bar that just closed, the momentum regime,
              the models a rule names, the open position and the clock into
              named numbers.
    rules     `apple_rules.first_match` walks the list in order and returns the
              first rule that both matches and can transact.
    orders    one market order per bar at most, through the same
              `DecisionTracker` as every other personality.

It also picks its own instrument, which Apple Trader cannot. There the strategy
*is* a saved AAPL model, so the symbol is not a choice; here a model is one
signal among many, and the rest of the vocabulary is computed from bars and
means the same thing on every symbol. So the config names a ticker, the
catalogue narrows to what exists for it (`apple_rules.signals_for`) and a rule
naming a model that symbol has none of is refused before the run rather than
sitting permanently unmet inside it.

Three properties worth stating, because they are what make a list of rules
behave predictably rather than merely expressively:

* **At most one action per closed bar.** The list is a priority order, not a
  budget. A bar where a stop and a scale-in both match is the stop's, because it
  is written above -- and the scale-in gets its turn on the next bar if it still
  applies. Cycles that run before a new bar has closed do nothing at all, so a
  slow fetch cannot double-fire a rule.
* **Nothing is computed that no rule reads.** The bundles loaded are the ones
  the enabled rules name (`RuleSet.models`), the day's high/low forecast is made
  only if some rule asks for it, and inside a rule the conditions short-circuit
  -- so an `all` rule whose cheap condition fails never pays for the forecast in
  the expensive one. Writing a broad rule set costs what it uses.
* **The flatten is not a rule.** Every feature every model here reads is
  intraday, and a rule set that forgot to close the book would carry overnight
  risk it was never measured on. So the closing flatten sits in the config
  (`flatten_before_close_min`), applies before the list is consulted, and cannot
  be deleted by editing rules -- the same reason it is unconditional in Apple
  Trader.

What this does not do
---------------------
It does not tune anything, and it does not know which rule sets are any good.
The four presets in `apple_rules` include the two shipped strategies precisely
so that a rule set written here can be compared against them in SimLab rather
than against an intuition -- and the honest summary of what those two are worth
is in `apple_trader`'s docstring, which this agent does nothing to improve.

Rules are also not backtested by construction: the vocabulary makes it easy to
write a set that fires on every bar, ladders a balance in over ten minutes, or
holds a position no exit can reach. `apple_rules.ruleset_error` catches the ones
that cannot work at all (no buy, no conditions, a model that cannot answer);
everything else is a strategy, and finding out whether it is a good one is what
the simulator is for.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import pandas as pd

from . import apple_models, apple_rules, historical, market_hours, persistence_model, scoring
from . import observability as obs
from .agent import _log, stop_agent
from .apple_rules import RuleSet
from .apple_trader import fetch_opening_window
from .clock import now as _now
from .config import (
    APPLE_TRADER_BAR_LAG_SEC,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN,
)
from .decisions import DecisionTracker
from .state import AppState

APPLE_TRADER2_KEY = "apple_trader2"
APPLE_TRADER2_LABEL = "Apple Trader 2 (adjustable rules)"
APPLE_TRADER2_AVATAR = "Multiavatar-088d131670fd0343f5.png"

# The symbol a run trades unless its config says otherwise. Unlike Apple Trader
# (which is AAPL-only because both its strategies *are* a saved model) the
# instrument here is configuration: a rule set written on the bar, the momentum
# regime, the position and the clock is computed from the tape and means the
# same thing on any symbol, and the models that are not symbol-agnostic are
# simply not offered elsewhere -- see `apple_rules.signals_for`.
DEFAULT_TICKER = apple_models.DEFAULT_TICKER


@dataclass
class AppleTrader2Config:
    """The whole configuration: an instrument, a rule set, and the one rule that
    is not in it.

    `flatten_before_close_min` is deliberately outside the list -- see the
    module docstring. Everything else a run does is in `rules`.

    `ticker` is not a signal and not a rule: it decides which symbol's bars the
    whole list is evaluated against, and therefore which model signals exist at
    all. Which is why it is validated together with the rules
    (`apple_rules.ruleset_error`) rather than on its own.
    """

    rules: RuleSet = field(default_factory=apple_rules.preset)
    flatten_before_close_min: int = APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN
    ticker: str = DEFAULT_TICKER

    def __post_init__(self) -> None:
        # A config reconstructed from a stored SimLab record arrives with plain
        # dicts where the dataclasses go.
        if not isinstance(self.rules, RuleSet):
            self.rules = RuleSet.from_record(self.rules)
        self.flatten_before_close_min = int(self.flatten_before_close_min)
        self.ticker = (self.ticker or DEFAULT_TICKER).strip().upper()

    def to_record(self) -> dict:
        return {
            "rules": self.rules.to_record(),
            "flatten_before_close_min": self.flatten_before_close_min,
            "ticker": self.ticker,
        }

    @classmethod
    def from_record(cls, raw: "dict | None") -> "AppleTrader2Config":
        raw = dict(raw or {})
        return cls(
            rules=RuleSet.from_record(raw.get("rules")),
            flatten_before_close_min=raw.get(
                "flatten_before_close_min", APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN
            ),
            # A record written before the instrument was configurable describes
            # an AAPL run, which is what the default decodes to -- the same
            # reason `rule_agents._APPLE_LEGACY` exists.
            ticker=raw.get("ticker") or DEFAULT_TICKER,
        )


def config_signature(config: "AppleTrader2Config | None" = None) -> str:
    """The compact identity SimLab groups runs on. See `apple_rules.signature`.

    The instrument is in it because it is part of the strategy: the same rules
    over GOOGL are a different experiment from the same rules over AAPL, and
    filing them together would average two tapes into one row.
    """
    config = config or AppleTrader2Config()
    return apple_rules.signature(config.rules, config.ticker)


def load_bundles(config: AppleTrader2Config) -> "dict[str, dict | None]":
    """Every model bundle the enabled rules name, for this config's instrument,
    loaded once.

    A rule set that names no model loads nothing at all and needs neither the
    saved artifacts nor PyTorch -- a set written entirely out of price, momentum
    and the position is a perfectly good strategy and should not pay for a
    200 MB dependency it never calls.
    """
    return {
        key: apple_models.load(key, config.ticker) for key in config.rules.models()
    }


def config_error(
    config: AppleTrader2Config, bundles: "dict[str, dict | None]"
) -> "str | None":
    """Why this configuration cannot run, or None. See `apple_rules.ruleset_error`."""
    return apple_rules.ruleset_error(config.rules, bundles, config.ticker)


def _dayrange():
    """`agent_stonks.dayrange_model`, imported on first use -- it pulls PyTorch
    and LightGBM in, and a rule set that never mentions the day-range model must
    not pay for them."""
    from . import dayrange_model

    return dayrange_model


class SessionForecaster:
    """TimeToChange3's one forecast a day, made only when a rule asks for it.

    The forecast is a statement about one session, built at 9:35 from a year of
    daily history plus the first five minutes, and it never updates -- so it is
    cached per date and re-made each morning. A session it cannot be made for is
    cached too: a daily history that is too short at 9:35 is still too short at
    14:00, and the alternative is failing (and logging) once a minute for six
    and a half hours.
    """

    def __init__(self, ticker: str = DEFAULT_TICKER) -> None:
        self.ticker = ticker
        self.plan: "dict | None" = None
        self.blocked: "dict | None" = None

    def forecast(self, bundle: dict, state: AppState, frame) -> "dict | None":
        """Today's forecast, or None while it does not (yet) exist."""
        dayrange = _dayrange()
        today = dayrange.market_date()
        if self.plan is not None and self.plan["date"] != today:
            self.plan = None
        if self.blocked is not None and self.blocked["date"] != today:
            self.blocked = None
        if self.plan is not None:
            return self.plan
        if self.blocked is not None:
            return None

        want = dayrange.opening_minutes(bundle)
        if len(frame) < want:
            return None
        try:
            opening = fetch_opening_window(state, frame, want, ticker=self.ticker)
            history = dayrange.daily_frame_from_bars(
                historical.fetch_daily_ohlc_bars(
                    self.ticker, days=dayrange.DAILY_HISTORY_DAYS
                )
            )
            forecast = dayrange.forecast_session(
                bundle, history, opening, today,
                open_price=historical.fetch_session_open(self.ticker),
            )
        except Exception as exc:
            self.blocked = {"date": today, "reason": str(exc)}
            _log(
                state,
                {
                    "type": "error",
                    "text": (
                        f"No {self.ticker} day-range forecast for this session, so every "
                        f"rule written on it is dormant today: {exc}"
                    ),
                },
            )
            return None

        self.plan = {"date": today, "opening_end": opening.index[-1], **forecast}
        warning = dayrange.volume_scale_warning(getattr(state, "feed", None))
        if warning:
            _log(state, {"type": "status", "text": f"Forecast caveat: {warning}"})
        _log(
            state,
            {
                "type": "analysis",
                "text": (
                    f"{self.ticker} forecast for the session: high "
                    f"${forecast['pred_high']:,.2f}, low ${forecast['pred_low']:,.2f}, "
                    f"14-day average range ${forecast['adr14_abs']:,.2f}."
                ),
            },
        )
        return self.plan


_UNSET = object()


class SignalBus:
    """Every number a condition can name, computed on demand for one bar.

    On demand is the whole design. The catalogue spans a bar's close (free), a
    Schmitt-triggered momentum regime (one pass over the session), a 25-feature
    window run through a saved model (a forecast, per bar, per model) and a
    day-range forecast (a year of daily history plus a network fetch). Computing
    all of it every minute to answer a rule set that reads three of them would
    make a cheap strategy pay a forecaster's bill -- so nothing is computed
    until `value` is asked for it, and every answer is cached until the bus is
    replaced on the next bar.

    A signal that does not apply is None, never a placeholder: the persistence
    question is only asked on a bar the regime changed into positive, the turn
    question only while it has not, the reversal question only while it has, and
    every day-range signal only after the opening window closes. That is the
    same gating `persistence_model.read_latest` applies -- a model asked about a
    bar it was not fitted for would answer, and the answer would mean nothing.
    """

    def __init__(
        self,
        frame,
        *,
        bundles: "dict[str, dict | None]",
        params: dict,
        position: float,
        entry: "dict | None",
        cash: float,
        last_buy: "dict | None" = None,
        plan_fn=None,
    ) -> None:
        self.frame = frame
        self.bundles = bundles
        self.params = params
        self.position = position
        self.entry = entry
        self.last_buy = last_buy
        self.cash = cash
        # Called at most once, and only if a rule reads a day-range signal.
        self._plan_fn = plan_fn
        self._plan = _UNSET
        self._scored_frame = _UNSET
        self._features = _UNSET
        self._sequences: dict = {}
        self._cache: "dict[str, float | None]" = {}

        self.last = frame.iloc[-1]
        self.ts = frame.index[-1]
        self.price = float(self.last["close"])
        self.high = float(self.last["high"])
        self.low = float(self.last["low"])

    # --- the one public entry point ---------------------------------------

    def value(self, key: str) -> "float | None":
        """The current value of one signal, or None where it does not apply."""
        if key in self._cache:
            return self._cache[key]
        namespace, _, name = key.partition(".")
        try:
            if namespace == "bar":
                value = self._bar(name)
            elif namespace == "mom":
                value = self._momentum(name)
            elif namespace == "pos":
                value = self._position(name)
            elif namespace == "clock":
                value = self._clock(name)
            elif namespace == apple_models.DAYRANGE_KEY:
                value = self._dayrange(name)
            else:
                value = self._model(namespace, name)
        except Exception:
            # A signal that blows up is a signal that is not available. Letting
            # it escape would take the whole cycle down, and a rule that cannot
            # be evaluated must not fire -- None is exactly "unmet".
            value = None
        self._cache[key] = value
        return value

    def computed(self) -> "dict[str, float | None]":
        """What was actually read this bar -- what the log line reports.

        Reporting the rule set's whole vocabulary instead would force every
        skipped condition to be evaluated, which is the cost the short-circuit
        exists to avoid.
        """
        return dict(self._cache)

    # --- the tape ----------------------------------------------------------

    def _bar(self, name: str) -> "float | None":
        if name == "price":
            return self.price
        if name == "high":
            return self.high
        if name == "low":
            return self.low
        if name == "session_change_pct":
            first_open = float(self.frame["open"].iloc[0])
            return (self.price / first_open - 1) * 100 if first_open else None
        return None

    def _scored(self):
        """The frame with momentum and regimes attached, or None.

        None while the session is too young for the trailing window to produce a
        number -- which is most of the first half hour, and is why a momentum
        rule simply does not fire then rather than firing on a zero.
        """
        if self._scored_frame is _UNSET:
            self._scored_frame = None
            if len(self.frame) >= self.params["horizon"] + 2:
                scored = persistence_model.add_momentum_regimes(self.frame, self.params)
                if not pd.isna(scored.iloc[-1]["mom"]):
                    self._scored_frame = scored
        return self._scored_frame

    def _momentum(self, name: str) -> "float | None":
        scored = self._scored()
        if scored is None:
            return None
        last = scored.iloc[-1]
        if name == "score":
            return float(last["mom"])
        if name == "regime":
            return float(int(last["regime"]))
        if name == "bars_in_regime":
            return float(int(last["bars_in_regime"]))
        if name == "pre_dwell":
            return None if pd.isna(last["pre_dwell"]) else float(int(last["pre_dwell"]))
        if name == "regime_change":
            return 1.0 if bool(last["regime_change"]) else 0.0
        if name == "to_positive":
            return 1.0 if bool(last["regime_change"]) and int(last["regime"]) == 1 else 0.0
        return None

    # --- the models --------------------------------------------------------

    def _sequence(self, model_key: str, bundle: dict):
        """The 20-bar feature window this model scores, or None if it is not
        complete. Cached per model because two bundles can want different
        columns, and the feature frame under them is shared."""
        if model_key not in self._sequences:
            if self._features is _UNSET:
                scored = self._scored()
                self._features = (
                    None if scored is None else persistence_model.compute_features(scored)
                )
            self._sequences[model_key] = (
                None
                if self._features is None
                else persistence_model.build_sequence(
                    self._features, bundle["feature_columns"], int(bundle["seq_len"])
                )
            )
        return self._sequences[model_key]

    def _model(self, model_key: str, name: str) -> "float | None":
        bundle = self.bundles.get(model_key)
        scored = self._scored()
        if bundle is None or scored is None:
            return None
        last = scored.iloc[-1]
        regime = int(last["regime"])
        to_positive = bool(last["regime_change"]) and regime == 1

        if name == "proba":
            if not to_positive:
                return None
            predict = persistence_model.predict_proba
        elif name == "turn_proba":
            if regime == 1 or not persistence_model.anticipates(bundle):
                return None
            predict = persistence_model.predict_turn_proba
        elif name == "reversal_proba":
            if regime != 1 or not persistence_model.forecasts_reversal(bundle):
                return None
            predict = persistence_model.predict_reversal_proba
        else:
            return None

        sequence = self._sequence(model_key, bundle)
        if sequence is None:
            return None
        return float(predict(bundle, sequence[None, ...])[0])

    # --- the day-range forecast --------------------------------------------

    def _plan_now(self) -> "dict | None":
        if self._plan is _UNSET:
            self._plan = self._plan_fn() if self._plan_fn is not None else None
        return self._plan

    def _dayrange(self, name: str) -> "float | None":
        plan = self._plan_now()
        if plan is None:
            return None
        # The bar the forecast was built on is the last bar of the opening
        # window, and it is never traded -- the notebook's `start_after`, and
        # the same refusal Apple Trader's day-range rules make.
        if self.ts <= plan["opening_end"]:
            return None
        if name == "pred_high":
            return float(plan["pred_high"])
        if name == "pred_low":
            return float(plan["pred_low"])
        adr = float(plan["adr14_abs"])
        if name == "adr":
            return adr
        if adr <= 0:
            return None
        if name == "pred_high_dip_adr":
            return (float(plan["pred_high"]) - self.low) / adr
        if name == "pred_high_reach_adr":
            return (float(plan["pred_high"]) - self.high) / adr
        if name == "pred_high_gap_adr":
            return (float(plan["pred_high"]) - self.price) / adr
        if name == "pred_low_gap_adr":
            return (self.price - float(plan["pred_low"])) / adr
        return None

    # --- the position and the clock ----------------------------------------

    def _position(self, name: str) -> "float | None":
        if name == "shares":
            return float(self.position)
        if name == "cash":
            return float(self.cash)
        if name == "value":
            return float(self.position) * self.price
        if name == "since_buy_pct":
            # Anchored on the last buy rather than on the position, so it
            # outlives the sell -- that is the whole point of it, and the reason
            # it is read before the flat-book gate below.
            return self._since_buy()
        entry = self.entry
        if not entry or self.position <= 0:
            # No position means no P&L, no give-back and no holding time. None
            # rather than 0: a rule reading "give-back below -0.5%" must not be
            # armed by a flat book.
            return None
        if name == "pnl_pct":
            price = entry.get("price") or 0.0
            return (self.price / price - 1) * 100 if price else None
        if name == "drawdown_pct":
            peak = entry.get("peak") or entry.get("price") or 0.0
            return (self.price / peak - 1) * 100 if peak else None
        if name == "bars_held":
            return float(entry.get("bars") or 0)
        return None

    def _since_buy(self) -> "float | None":
        """This bar's close against the best price since the last buy filled.

        `max(fill price, every high since)` is the reference the rule is
        specified against. The peak is maintained to be exactly that -- it
        starts at the fill and only ratchets -- so the `max` here is belt and
        braces rather than arithmetic that ever bites; it is written out because
        it is what the signal means.
        """
        anchor = self.last_buy
        if not anchor:
            return None
        reference = max(anchor.get("price") or 0.0, anchor.get("peak") or 0.0)
        return (self.price / reference - 1) * 100 if reference > 0 else None

    def _clock(self, name: str) -> "float | None":
        if name == "minutes_from_open":
            if "minutes_from_open" in self.frame.columns:
                return float(self.last["minutes_from_open"])
            return None
        if name == "minutes_to_close":
            seconds = market_hours.seconds_to_close()
            return None if seconds is None else seconds / 60.0
        return None


class AppleTrader2:
    """The state machine driving one run: the rule list, and the book it keeps.

    One instance per launched agent; `run_cycle` is called once per closed bar
    by the live loop and once per stored bar by SimLab.
    """

    def __init__(
        self, config: "AppleTrader2Config | None" = None,
        bundles: "dict[str, dict | None] | None" = None,
    ) -> None:
        self.config = config or AppleTrader2Config()
        self.bundles = bundles or {}
        # Read off the config once: every log line, order and state lookup below
        # is about this one symbol, and the config is not re-read mid-run.
        self.ticker = self.config.ticker
        # The momentum settings every regime signal is computed with. They come
        # from whichever momentum bundle the rules named -- the two carry the
        # same numbers -- and fall back to the shipped defaults when no rule
        # names one, which is what lets a price-only rule set run with no model
        # files present at all.
        self.params = persistence_model.momentum_params(self._momentum_bundle())
        self.forecaster = SessionForecaster(self.ticker)
        # The open long: average entry, the peak the give-back is measured from,
        # and how many bars it has been held. None while flat.
        self.entry: "dict | None" = None
        # The last buy that filled, and the highest price seen since it did --
        # the anchor `pos.since_buy_pct` measures from. Deliberately NOT the
        # same record as `entry`: this one survives the sell that clears the
        # position, so a rule can wait for price to move away from where it last
        # bought before buying again. Re-anchored by every buy, cleared each
        # morning (a reference from yesterday would be measured across the
        # overnight gap, which nothing else here does).
        self.last_buy: "dict | None" = None
        self.last_bar_ts = None
        # Bars seen this session, and the bar each rule last fired on -- the two
        # halves of `cooldown_bars`. Both reset each morning, which is what
        # makes a cooldown longer than a session mean "once a day".
        self.bar_count = 0
        self.fired_at: "dict[int, int]" = {}
        self.session_date = None

    def _momentum_bundle(self) -> "dict | None":
        for key, bundle in self.bundles.items():
            if bundle is not None and apple_models.is_momentum(key):
                return bundle
        return None

    # --- one cycle --------------------------------------------------------

    def run_cycle(self, state: AppState, tracker: DecisionTracker) -> str:
        """Read the newest closed bar and act on it. Returns a short outcome tag
        ("bought", "sold", "hold", "warming_up", "closed", "no_data")."""
        sym_state = state.sym(self.ticker)
        if sym_state is None:
            _log(
                state,
                {
                    "type": "error",
                    "text": f"{self.ticker} is not being streamed; nothing to trade.",
                },
            )
            return "no_data"

        if not market_hours.is_market_open():
            _log(
                state,
                {"type": "status", "text": "Market closed -- Apple Trader 2 is not watching bars."},
            )
            return "closed"

        frame = persistence_model.minute_frame(sym_state)
        if not len(frame):
            _log(state, {"type": "status", "text": f"No {self.ticker} bars yet today."})
            return "no_data"

        ts = frame.index[-1]
        self._roll_session(ts)
        fresh_bar = ts != self.last_bar_ts
        if fresh_bar:
            self.last_bar_ts = ts
            self.bar_count += 1

        position = tracker.position_for(self.ticker)
        self._reconcile(position, float(frame["close"].iloc[-1]))
        if fresh_bar:
            high = float(frame["high"].iloc[-1])
            if self.entry is not None:
                self.entry["bars"] += 1
                # The give-back is measured from the highest price the position
                # has TRADED at, so the peak comes from the bar's high, not its
                # close.
                self.entry["peak"] = max(self.entry["peak"], high)
            if self.last_buy is not None:
                # Ratchets on whether or not the position is still open: the
                # range this signal measures over is "since the last buy", and
                # selling does not end it.
                self.last_buy["peak"] = max(self.last_buy["peak"], high)

        snapshot = tracker.snapshot()
        bus = SignalBus(
            frame,
            bundles=self.bundles,
            params=self.params,
            position=position,
            entry=self.entry,
            cash=snapshot["cash"],
            last_buy=self.last_buy,
            plan_fn=lambda: self._forecast(state, frame),
        )

        if position > 0 and self._closing_soon():
            self._log_read(state, bus, position)
            self._flatten(state, tracker, position, bus)
            return "sold"

        if not fresh_bar:
            # Nothing has changed since the last decision, and re-running the
            # list on the same numbers could only produce the same answer twice.
            return "hold"

        match = apple_rules.first_match(
            self.config.rules,
            bus.value,
            price=bus.price,
            cash=snapshot["cash"],
            shares=position,
            blocked=self._cooling_down,
        )
        if match is not None and match.item.action == apple_rules.BUY and self._closing_soon():
            self._log_read(state, bus, position)
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"Rule {match.index + 1} fired on the {ts:%H:%M} bar, but the session "
                        f"is inside its last {self.config.flatten_before_close_min} min and "
                        "any position would be flattened straight back out. Standing down."
                    ),
                },
            )
            return "hold"

        self._log_read(state, bus, position)
        if match is None:
            # A cheap length test rather than asking the bus, which would force
            # the momentum pass on every bar of a rule set that never reads it.
            # It is a status tag, not a gate: nothing downstream branches on it.
            warming = (
                self.config.rules.reads_momentum()
                and len(frame) < self.params["horizon"] + 2
            )
            return "warming_up" if warming else "hold"
        return self._execute(state, tracker, match, bus)

    def _roll_session(self, ts) -> None:
        """Start each session with fresh cooldowns, bar count and buy anchor.

        The anchor goes with them because the alternative is measuring today's
        price against a reference set yesterday, across the overnight gap that
        every other signal here is careful not to cross.
        """
        date = ts.date()
        if self.session_date != date:
            self.session_date = date
            self.bar_count = 0
            self.fired_at = {}
            self.last_buy = None

    def _reconcile(self, position: float, price: float) -> None:
        """Keep the remembered entry in step with the ledger.

        A position with no remembered entry is an agent restarted onto an
        existing book: it is adopted at the current price, so the give-back is
        measured from here rather than from a peak this instance never saw.
        """
        if position > 0 and self.entry is None:
            self.entry = {"price": price, "peak": price, "bars": 0}
        elif position <= 0:
            self.entry = None

    def _cooling_down(self, index: int) -> bool:
        item = self.config.rules.items[index]
        if not item.cooldown_bars:
            return False
        fired = self.fired_at.get(index)
        return fired is not None and self.bar_count - fired < item.cooldown_bars

    def _closing_soon(self) -> bool:
        to_close = market_hours.seconds_to_close()
        return (
            to_close is not None
            and to_close <= self.config.flatten_before_close_min * 60
        )

    def _forecast(self, state: AppState, frame) -> "dict | None":
        bundle = self.bundles.get(apple_models.DAYRANGE_KEY)
        if bundle is None:
            return None
        return self.forecaster.forecast(bundle, state, frame)

    # --- orders ------------------------------------------------------------

    def _execute(
        self, state: AppState, tracker: DecisionTracker, match: apple_rules.Match, bus: SignalBus
    ) -> str:
        decision = tracker.record_trade(
            self.ticker, match.item.action, match.quantity, self._reasoning(match, bus),
            state.api_key, state.api_secret, state.feed,
        )
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
            },
        )
        if decision.status != "filled":
            return "hold"

        self.fired_at[match.index] = self.bar_count
        self._book(match.item.action, decision, tracker.position_for(self.ticker))
        return "bought" if match.item.action == apple_rules.BUY else "sold"

    def _book(self, action: str, decision, position: float) -> None:
        """Update the remembered entry and buy anchor after a fill.

        Partial fills in both directions are the point of this agent, so this is
        more than "set on buy, clear on sell": a second buy averages the entry
        price up or down (the P&L a rule reads has to be the position's, not the
        first slice's), while a partial sell leaves the entry and the peak
        alone -- taking half off does not restart the trailing stop on the rest.

        The buy anchor is the one record a sell does not touch, in either
        direction: every buy re-anchors it at that fill (it is the *last* buy,
        not the first), and it then keeps running through the exit until the
        next buy or the next session.
        """
        price = decision.price
        if action == apple_rules.BUY and price is not None:
            self.last_buy = {"price": price, "peak": price}
        if position <= 0:
            self.entry = None
            return
        if action == apple_rules.SELL or price is None:
            return
        filled = float(decision.filled_quantity or 0.0)
        if self.entry is None or filled <= 0:
            self.entry = {"price": price, "peak": price, "bars": 0}
            return
        held_before = max(position - filled, 0.0)
        total = held_before + filled
        self.entry["price"] = (
            (self.entry["price"] * held_before + price * filled) / total if total else price
        )
        self.entry["peak"] = max(self.entry["peak"], price)

    def _flatten(
        self, state: AppState, tracker: DecisionTracker, position: float, bus: SignalBus
    ) -> None:
        to_close = market_hours.seconds_to_close() or 0.0
        entry_price = (self.entry or {}).get("price") or 0.0
        pnl_pct = (bus.price / entry_price - 1) * 100 if entry_price else 0.0
        decision = tracker.record_trade(
            self.ticker, apple_rules.SELL, position,
            (
                f"Session ends in {to_close / 60:.0f} min. Every signal these rules read is "
                "intraday, so the book is flattened rather than carried overnight "
                f"({pnl_pct:+.2f}%). This is the agent's own rule, not one of the "
                f"{len(self.config.rules.items)} in the list."
            ),
            state.api_key, state.api_secret, state.feed,
        )
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
            },
        )
        if decision.status == "filled":
            self.entry = None

    def _reasoning(self, match: apple_rules.Match, bus: SignalBus) -> str:
        """Why this rule fired, with the numbers it fired on.

        The rule as written plus the values behind it, because a ledger entry
        that only quotes the rule cannot be checked afterwards and one that only
        quotes the numbers cannot be traced back to the rule that read them.
        """
        item = match.item
        joiner = f" {apple_rules.JOIN_WORD[item.join]} "
        met = joiner.join(
            f"{apple_rules.format_condition(c)} "
            f"({_render(bus.value(c.field))} now)"
            for c in item.conditions
        )
        label = f" — {item.label}" if item.label else ""
        return (
            f"Rule {match.index + 1}{label}: {item.action} "
            f"{apple_rules.format_size(item)} when {met}. "
            f"Sending {match.quantity:g} sh at market."
        )

    # --- logging -----------------------------------------------------------

    def _log_read(self, state: AppState, bus: SignalBus, position: float) -> None:
        _log(state, {"type": "analysis", "text": self._read_summary(bus, position)})

    def _read_summary(self, bus: SignalBus, position: float) -> str:
        parts = [f"{self.ticker} {bus.ts:%H:%M} ${bus.price:,.2f}"]
        # Only the signals the rules actually read this bar, in the order they
        # were read -- so the log shows what the decision was made on, including
        # which conditions were never reached. The two the position line below
        # already quotes are dropped rather than printed twice; everything else
        # about the book (the cash a rule sized against, the give-back since the
        # last buy) is worth having in the line whether or not one is open.
        parts += [
            f"{key} {_render(value)}"
            for key, value in bus.computed().items()
            if key not in ("pos.pnl_pct", "pos.drawdown_pct")
        ]
        if position > 0 and self.entry:
            entry_price = self.entry["price"]
            pnl = (bus.price / entry_price - 1) * 100 if entry_price else 0.0
            peak = self.entry["peak"]
            give_back = (bus.price / peak - 1) * 100 if peak else 0.0
            parts.append(
                f"long {position:g} sh @ ${entry_price:,.2f} ({pnl:+.2f}%), "
                f"{self.entry['bars']} bars, peak ${peak:,.2f} ({give_back:+.2f}% off)"
            )
        return " · ".join(parts)


def _render(value: "float | None") -> str:
    """A signal value for a log line: 'n/a' where the signal did not apply."""
    return "n/a" if value is None else f"{value:,.4g}"


def build_trader(config: AppleTrader2Config, bundles: "dict[str, dict | None]") -> AppleTrader2:
    """The state machine for this configuration.

    One seam for every launch path (the live loop below, SimLab's
    `rule_agents._build_apple2`), mirroring `apple_trader.build_trader` -- there
    is only one state machine here because the strategy is data.
    """
    return AppleTrader2(config, bundles)


# --- the loop ---------------------------------------------------------------


def armed_summary(config: AppleTrader2Config) -> str:
    """The line the log opens a run with: every rule, in order."""
    rules = "\n".join(apple_rules.describe(config.rules))
    models = ", ".join(apple_models.get(k).label for k in config.rules.models()) or "none"
    return (
        f"Apple Trader 2 armed on {len(config.rules.active)} rule(s) over {config.ticker} "
        f"(models needed: {models}). At most one fires per closed bar, first match "
        f"wins, and anything still open is flattened "
        f"{config.flatten_before_close_min} min before the close:\n{rules}"
    )


def _seconds_to_next_bar(cycle_sec: int, lag: float = APPLE_TRADER_BAR_LAG_SEC) -> float:
    """Seconds to wait so the next cycle lands just after a bar closes."""
    ts = _now().timestamp()
    return (math.floor(ts / cycle_sec) + 1) * cycle_sec + lag - ts


def _apple_trader2_loop(
    state: AppState,
    tracker: DecisionTracker,
    config: AppleTrader2Config,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    bundles = load_bundles(config)
    mismatch = config_error(config, bundles)
    if mismatch is not None:
        _log(state, {"type": "error", "text": mismatch})
        scoring.end_session(state, tracker)
        state.agent_running = False
        return

    trader = build_trader(config, bundles)
    _log(state, {"type": "status", "text": armed_summary(config)})

    while not stop_event.is_set():
        try:
            trader.run_cycle(state, tracker)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Apple Trader 2 cycle failed: {exc}"})
        scoring.maybe_score_day(state, tracker)
        stop_event.wait(_seconds_to_next_bar(cycle_sec))

    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Apple Trader 2 stopped"})
    obs.flush()


def launch_apple_trader2(
    state: AppState,
    tracker: DecisionTracker,
    config: "AppleTrader2Config | None" = None,
    cycle_sec: int = APPLE_TRADER_CYCLE_SEC,
) -> None:
    """Stop any running agent for this state, then start the Apple Trader 2 loop.

    It trades only the one symbol its config names, which must already be
    streamed. No LLM client, no tools and no tactics are involved -- the loop
    places its own orders through the same `DecisionTracker` as every other
    personality.
    """
    config = config or AppleTrader2Config()
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, APPLE_TRADER2_KEY, [config.ticker])
    threading.Thread(
        target=_apple_trader2_loop,
        args=(state, tracker, config, cycle_sec, stop_event),
        daemon=True,
    ).start()
