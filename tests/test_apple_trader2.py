"""Apple Trader 2: the rule vocabulary, and the loop that executes it.

Three layers, pinned separately because they fail separately:

* `apple_rules` on its own -- conditions, AND/OR, sizing, precedence, the
  validation that refuses a rule set which cannot work. No frames, no models.
* `SignalBus` -- that a signal is a real number when it applies and None when it
  does not, over a synthetic session.
* `AppleTrader2.run_cycle` -- what actually gets ordered, driven through rules
  written on price and the position alone, so the whole loop is exercised
  without depending on a saved model or on live market data.
"""

import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from agent_stonks import apple_models
from agent_stonks import apple_rules as ar
from agent_stonks import apple_trader2 as at2
from agent_stonks import clock, persistence_model
from agent_stonks.apple_rules import ActionItem, Condition, RuleSet
from agent_stonks.apple_trader2 import (
    DEFAULT_TICKER as TICKER,
)
from agent_stonks.apple_trader2 import AppleTrader2, AppleTrader2Config, SignalBus
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

# 10:30 ET on a Tuesday: mid-session, well clear of both the open and the close.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)
# 15:58 ET the same day: inside the default five-minute flatten window.
NEAR_CLOSE = datetime(2026, 7, 21, 19, 58, tzinfo=timezone.utc)


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price
        self.orders: list[tuple] = []

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        self.orders.append((symbol, side, quantity, price))
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


@pytest.fixture
def market_open():
    clock.set_simulated(MIDSESSION)
    yield
    clock.clear()


@pytest.fixture
def near_close():
    clock.set_simulated(NEAR_CLOSE)
    yield
    clock.clear()


@pytest.fixture
def state() -> AppState:
    state = AppState()
    state.set_symbols([TICKER])
    state.api_key = "k"
    state.api_secret = "s"
    state.feed = "iex"
    return state


def frame(closes: "list[float]", *, end=MIDSESSION, highs=None, lows=None):
    """A minute frame of `closes` ending on the bar that closed at `end`.

    Built through `persistence_model.frame_from_bars` so it carries the same
    session bookkeeping the live buffer does -- the momentum pipeline reads
    those columns.
    """
    bars = []
    for i, close in enumerate(closes):
        ts = end - timedelta(minutes=len(closes) - 1 - i)
        bars.append(
            {
                "t": ts.isoformat(),
                "o": close,
                "h": close if highs is None else highs[i],
                "l": close if lows is None else lows[i],
                "c": close,
                "v": 1000.0,
            }
        )
    return persistence_model.frame_from_bars(bars)


def values(mapping: dict):
    """A stand-in signal bus: everything not in the dict is an absent signal."""
    return mapping.get


# --------------------------------------------------------------- conditions


class TestConditions:
    def test_above_is_inclusive_and_below_is_inclusive(self):
        cond = Condition("bar.price", ar.OP_ABOVE, 100.0)
        assert ar.condition_met(cond, values({"bar.price": 100.0}))
        assert ar.condition_met(cond, values({"bar.price": 100.5}))
        assert not ar.condition_met(cond, values({"bar.price": 99.5}))

        cond = Condition("bar.price", ar.OP_BELOW, 100.0)
        assert ar.condition_met(cond, values({"bar.price": 100.0}))
        assert not ar.condition_met(cond, values({"bar.price": 100.5}))

    def test_all_needs_every_condition(self):
        item = ActionItem(
            action=ar.BUY,
            join=ar.JOIN_ALL,
            conditions=[
                Condition("bar.price", ar.OP_BELOW, 100.0),
                Condition("mom.score", ar.OP_ABOVE, 1.0),
            ],
        )
        assert ar.item_matches(item, values({"bar.price": 99.0, "mom.score": 1.5}))
        assert not ar.item_matches(item, values({"bar.price": 99.0, "mom.score": 0.5}))

    def test_any_needs_one(self):
        item = ActionItem(
            action=ar.SELL,
            join=ar.JOIN_ANY,
            conditions=[
                Condition("pos.drawdown_pct", ar.OP_BELOW, -0.5),
                Condition("nbeats.reversal_proba", ar.OP_ABOVE, 0.3),
            ],
        )
        assert ar.item_matches(item, values({"pos.drawdown_pct": -0.7}))
        assert ar.item_matches(item, values({"nbeats.reversal_proba": 0.4}))
        assert not ar.item_matches(
            item, values({"pos.drawdown_pct": -0.1, "nbeats.reversal_proba": 0.1})
        )

    def test_an_absent_signal_never_matches_in_either_join(self):
        """The model that was not asked is not a yes -- and in an OR it is not
        even a maybe. A rule set must never fire on a number that does not
        exist."""
        for join in ar.JOINS:
            item = ActionItem(
                action=ar.SELL,
                join=join,
                conditions=[Condition("nbeats.reversal_proba", ar.OP_ABOVE, 0.3)],
            )
            assert not ar.item_matches(item, values({}))

    def test_a_rule_with_no_conditions_never_fires(self):
        """One deleted condition away from 'always buy', so it is inert rather
        than unconditional."""
        assert not ar.item_matches(ActionItem(action=ar.BUY, conditions=[]), values({}))

    def test_a_flag_signal_reads_as_true_or_false(self):
        item = ActionItem(
            action=ar.BUY, conditions=[Condition("mom.to_positive", ar.OP_ABOVE, 0.5)]
        )
        assert ar.item_matches(item, values({"mom.to_positive": 1.0}))
        assert not ar.item_matches(item, values({"mom.to_positive": 0.0}))
        assert "is true" in ar.format_condition(item.conditions[0])

    def test_an_unknown_signal_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown signal"):
            ActionItem(action=ar.BUY, conditions=[Condition("bar.vibes", ar.OP_ABOVE, 1)])


# ------------------------------------------------------------------- sizing


class TestSizing:
    def test_percent_of_cash_on_a_buy(self):
        item = ActionItem(action=ar.BUY, size_mode=ar.SIZE_PCT, size=95.0)
        assert ar.resolve_quantity(item, price=100.0, cash=10_000.0, shares=0) == 95.0

    def test_percent_of_the_position_on_a_sell(self):
        item = ActionItem(action=ar.SELL, size_mode=ar.SIZE_PCT, size=50.0)
        assert ar.resolve_quantity(item, price=100.0, cash=0.0, shares=80.0) == 40.0

    def test_a_dollar_amount_is_clipped_to_the_cash_there_is(self):
        item = ActionItem(action=ar.BUY, size_mode=ar.SIZE_CASH, size=5_000.0)
        assert ar.resolve_quantity(item, price=100.0, cash=10_000.0, shares=0) == 50.0
        assert ar.resolve_quantity(item, price=100.0, cash=300.0, shares=0) == 3.0

    def test_a_share_count_is_clipped_to_the_position(self):
        item = ActionItem(action=ar.SELL, size_mode=ar.SIZE_SHARES, size=200.0)
        assert ar.resolve_quantity(item, price=100.0, cash=0.0, shares=50.0) == 50.0

    def test_selling_everything_leaves_no_rounding_sliver(self):
        """A remainder of 0.0001 shares is not a position anybody meant to hold,
        and leaving it makes 'sell 100%' look like it did not work."""
        item = ActionItem(action=ar.SELL, size_mode=ar.SIZE_PCT, size=100.0)
        shares = 33.333333
        assert ar.resolve_quantity(item, price=100.0, cash=0.0, shares=shares) == shares

    def test_a_rule_that_cannot_transact_resolves_to_nothing(self):
        sell = ActionItem(action=ar.SELL, size_mode=ar.SIZE_PCT, size=100.0)
        assert ar.resolve_quantity(sell, price=100.0, cash=10_000.0, shares=0.0) == 0.0
        buy = ActionItem(action=ar.BUY, size_mode=ar.SIZE_PCT, size=100.0)
        assert ar.resolve_quantity(buy, price=100.0, cash=0.0, shares=10.0) == 0.0


# ---------------------------------------------------------------- precedence


class TestFirstMatch:
    def _ruleset(self) -> RuleSet:
        return RuleSet(
            items=[
                ActionItem(
                    action=ar.SELL, size_mode=ar.SIZE_PCT, size=100.0,
                    conditions=[Condition("pos.drawdown_pct", ar.OP_BELOW, -0.5)],
                ),
                ActionItem(
                    action=ar.BUY, size_mode=ar.SIZE_PCT, size=95.0,
                    conditions=[Condition("bar.price", ar.OP_BELOW, 100.0)],
                ),
            ]
        )

    def test_the_first_matching_rule_wins(self):
        match = ar.first_match(
            self._ruleset(),
            values({"pos.drawdown_pct": -0.9, "bar.price": 99.0}),
            price=99.0, cash=10_000.0, shares=50.0,
        )
        assert match.index == 0 and match.item.action == ar.SELL

    def test_a_dormant_rule_does_not_swallow_the_bar(self):
        """The sell is written first and its condition is unreadable while flat,
        but even if it matched it could not transact -- so the entry below it
        still gets its turn on this bar rather than the next."""
        match = ar.first_match(
            self._ruleset(),
            values({"pos.drawdown_pct": -0.9, "bar.price": 99.0}),
            price=99.0, cash=10_000.0, shares=0.0,
        )
        assert match.index == 1 and match.item.action == ar.BUY

    def test_a_disabled_rule_is_skipped(self):
        rules = self._ruleset()
        rules.items[1].enabled = False
        assert ar.first_match(
            rules, values({"bar.price": 99.0}), price=99.0, cash=10_000.0, shares=0.0
        ) is None

    def test_the_cooldown_callback_blocks_a_rule(self):
        rules = self._ruleset()
        assert ar.first_match(
            rules, values({"bar.price": 99.0}), price=99.0, cash=10_000.0, shares=0.0,
            blocked=lambda index: index == 1,
        ) is None


# ---------------------------------------------------------------- validation


class TestValidation:
    def test_an_empty_rule_set_cannot_run(self):
        assert "no enabled rules" in ar.ruleset_error(RuleSet(), {})

    def test_a_rule_without_conditions_cannot_run(self):
        rules = RuleSet(items=[ActionItem(action=ar.BUY, conditions=[])])
        assert "no conditions" in ar.ruleset_error(rules, {})

    def test_a_set_that_can_only_sell_cannot_run(self):
        """Its sells are about a position nothing in the list can open."""
        rules = RuleSet(
            items=[
                ActionItem(
                    action=ar.SELL, size_mode=ar.SIZE_PCT, size=100.0,
                    conditions=[Condition("pos.pnl_pct", ar.OP_ABOVE, 1.0)],
                )
            ]
        )
        assert "only sell" in ar.ruleset_error(rules, {})

    def test_a_missing_bundle_is_reported_before_the_run_not_after(self):
        """Left unchecked this is the quiet failure: the signal stays None, None
        never matches, and the run finishes clean with an empty ledger that
        reads like a strategy result."""
        rules = ar.preset("Momentum — anticipate the turn (Apple Trader's default)")
        error = ar.ruleset_error(rules, {"nbeats": None})
        assert "nbeats.turn_proba" in error or "turn_proba" in error

    def test_a_classifier_cannot_be_asked_to_forecast(self, monkeypatch):
        """`persistence.proba` is a question about a change bar; there is no
        `persistence.turn_proba` in the catalogue at all."""
        assert "persistence.turn_proba" not in ar.SIGNALS
        assert "nbeats.turn_proba" in ar.SIGNALS

        rules = RuleSet(
            items=[
                ActionItem(
                    action=ar.BUY, size_mode=ar.SIZE_PCT, size=95.0,
                    conditions=[Condition("nbeats.turn_proba", ar.OP_ABOVE, 0.05)],
                )
            ]
        )
        monkeypatch.setattr(persistence_model, "anticipates", lambda bundle: False)
        error = ar.ruleset_error(rules, {"nbeats": {"model": "not a forecaster"}})
        assert "cannot answer" in error

    def test_the_shipped_presets_all_validate(self):
        """Every preset has to be launchable given its models, or it is not a
        preset -- it is a trap."""
        for name in ar.PRESETS:
            rules = ar.preset(name)
            bundles = {key: {"stub": True} for key in rules.models()}
            monkey = ar.ruleset_error(rules, bundles)
            # The only thing a stub bundle cannot satisfy is the forecast check.
            assert monkey is None or "cannot answer" in monkey


# ----------------------------------------------------------------- instrument
#
# What a symbol changes is which model signals exist for it -- nothing else.
# `UNMODELLED` stands for every ticker nobody fitted anything on, which is the
# case the tape-only half of the catalogue exists for.

UNMODELLED = "MSFT"
DAYRANGE_ONLY = "GOOGL"


class TestInstrument:
    def test_the_models_on_offer_follow_the_symbol(self):
        assert apple_models.keys_for(TICKER) == ["persistence", "nbeats", "dayrange"]
        assert apple_models.keys_for(DAYRANGE_ONLY) == ["dayrange"]
        assert apple_models.keys_for("INTC") == ["dayrange"]
        assert apple_models.keys_for(UNMODELLED) == []

    def test_a_model_is_not_loaded_for_a_symbol_it_was_not_fitted_on(self, monkeypatch):
        """Cheaper than the file check and more honest: there is no GOOGL
        N-BEATS file to be missing, because there is no GOOGL N-BEATS model."""
        called: list = []
        monkeypatch.setitem(
            apple_models.MODELS, "nbeats",
            replace(apple_models.MODELS["nbeats"], load=called.append),
        )
        assert apple_models.load("nbeats", DAYRANGE_ONLY) is None
        assert called == []
        # ...and the loader is still reached for the symbol it does cover.
        apple_models.load("nbeats", TICKER)
        assert called == [TICKER]
        reason = apple_models.unavailable_reason("nbeats", DAYRANGE_ONLY)
        assert "no" in reason.lower() and "AAPL" in reason

    def test_the_catalogue_narrows_but_never_below_the_tape(self):
        every = set(ar.signals_for(TICKER))
        dayrange_only = set(ar.signals_for(DAYRANGE_ONLY))
        unmodelled = set(ar.signals_for(UNMODELLED))

        assert every == set(ar.SIGNALS)
        assert "nbeats.turn_proba" not in dayrange_only
        assert "persistence.proba" not in dayrange_only
        assert "dayrange.pred_high_dip_adr" in dayrange_only
        assert not any("." in key and key.split(".")[0] in apple_models.MODELS
                       for key in unmodelled)
        # The model-free half is identical on every symbol: it is computed from
        # bars, and bars are bars.
        tape = {key for key, spec in ar.SIGNALS.items() if spec.model is None}
        assert unmodelled == tape and tape < dayrange_only

    def test_a_rule_naming_an_absent_model_is_refused_with_the_signal_named(self):
        rules = ar.preset("Momentum — anticipate the turn (Apple Trader's default)")
        error = ar.ruleset_error(rules, {}, DAYRANGE_ONLY)
        assert "nbeats.turn_proba" in error and DAYRANGE_ONLY in error
        # ...and on AAPL the same rules get past this check, on to the ones
        # about the bundle itself (which a stub cannot satisfy).
        assert "cannot be read on" not in (
            ar.ruleset_error(rules, {"nbeats": {"stub": True}}, TICKER) or ""
        )

    def test_the_missing_model_is_reported_before_the_missing_file(self):
        """Two different problems: 'GOOGL has no N-BEATS model' sends the reader
        to the instrument picker, 'the file is not installed' sends them looking
        for a file that was never meant to exist."""
        rules = ar.preset("Momentum — anticipate the turn (Apple Trader's default)")
        error = ar.ruleset_error(rules, {"nbeats": None}, DAYRANGE_ONLY)
        assert "cannot be read on" in error and "not installed" not in error

    def test_every_preset_offered_for_a_symbol_runs_on_it(self):
        for symbol in (TICKER, DAYRANGE_ONLY, UNMODELLED):
            offered = ar.presets_for(symbol)
            assert offered, symbol
            for name in offered:
                rules = ar.preset(name, symbol)
                bundles = {key: {"stub": True} for key in rules.models()}
                error = ar.ruleset_error(rules, bundles, symbol)
                assert error is None or "cannot answer" in error, (symbol, name, error)

    def test_an_unmodelled_symbol_still_has_somewhere_to_start(self):
        assert ar.presets_for(UNMODELLED) == [ar.MODEL_FREE_PRESET]
        assert ar.default_preset(UNMODELLED) == ar.MODEL_FREE_PRESET
        assert ar.preset(None, UNMODELLED).models() == []

    def test_a_preset_that_does_not_apply_falls_back_rather_than_failing(self):
        """The picker can be pointed at a symbol while holding another's preset
        name; the fallback is what keeps that from producing an unrunnable set."""
        rules = ar.preset(
            "Momentum — anticipate the turn (Apple Trader's default)", UNMODELLED
        )
        assert rules.unreadable_on(UNMODELLED) == []

    def test_the_bundles_loaded_are_the_configs_symbols(self, monkeypatch):
        asked: list = []
        monkeypatch.setattr(
            at2.apple_models, "load",
            lambda key, ticker=None: asked.append((key, ticker)),
        )
        config = AppleTrader2Config(
            rules=ar.preset("Day range — two levels below the predicted high"),
            ticker=DAYRANGE_ONLY,
        )
        at2.load_bundles(config)
        assert asked == [("dayrange", DAYRANGE_ONLY)]

    def test_the_configured_symbol_is_the_one_traded(self, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        state = AppState()
        state.set_symbols([DAYRANGE_ONLY])
        state.api_key, state.api_secret, state.feed = "k", "s", "iex"
        trader = trader_for(buy_below(99.5), ticker=DAYRANGE_ONLY)

        tape.bar(99.0)
        assert trader.run_cycle(state, tracker) == "bought"
        assert tracker.position_for(DAYRANGE_ONLY) > 0
        assert tracker.position_for(TICKER) == 0

    def test_a_symbol_that_is_not_streamed_says_so(self, state, market_open, monkeypatch):
        """`state` streams AAPL only."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        Tape(monkeypatch, None)
        trader = trader_for(buy_below(99.5), ticker=DAYRANGE_ONLY)
        assert trader.run_cycle(state, tracker) == "no_data"
        assert DAYRANGE_ONLY in state.agent_log[-1]["text"]


# -------------------------------------------------------- record and identity


class TestRecordAndSignature:
    def test_a_config_survives_the_json_round_trip(self):
        config = AppleTrader2Config(rules=ar.preset(), flatten_before_close_min=3)
        restored = AppleTrader2Config.from_record(config.to_record())
        assert isinstance(restored.rules, RuleSet)
        assert isinstance(restored.rules.items[0], ActionItem)
        assert isinstance(restored.rules.items[0].conditions[0], Condition)
        assert restored.flatten_before_close_min == 3
        assert restored.to_record() == config.to_record()

    def test_the_instrument_survives_the_round_trip(self):
        config = AppleTrader2Config(
            rules=ar.preset(None, DAYRANGE_ONLY), ticker=DAYRANGE_ONLY
        )
        assert AppleTrader2Config.from_record(config.to_record()).ticker == DAYRANGE_ONLY

    def test_a_record_written_before_the_instrument_existed_is_an_aapl_run(self):
        """A stored record describes a run that already happened, so a missing
        key has to decode to the behaviour of the day it was written."""
        legacy = {"rules": ar.preset().to_record(), "flatten_before_close_min": 5}
        assert AppleTrader2Config.from_record(legacy).ticker == TICKER

    def test_the_signature_carries_the_numbers(self):
        signature = at2.config_signature(AppleTrader2Config(rules=ar.preset()))
        assert signature.startswith("apple2_AAPL(")
        assert "nbeats.turn_proba>=0.05" in signature

    def test_the_same_rules_on_another_symbol_are_another_configuration(self):
        """Two tapes are two experiments; filing them together would average
        them into one row in Results."""
        rules = ar.preset("Day range — two levels below the predicted high")
        assert at2.config_signature(
            AppleTrader2Config(rules=rules, ticker=TICKER)
        ) != at2.config_signature(
            AppleTrader2Config(rules=rules, ticker=DAYRANGE_ONLY)
        )

    def test_moving_a_threshold_is_a_different_configuration(self):
        one = ar.preset()
        other = ar.preset()
        other.items[1].conditions[0].value = 0.10
        assert ar.signature(one, TICKER) != ar.signature(other, TICKER)

    def test_a_disabled_rule_does_not_change_the_identity(self):
        """Results should go on grouping a run with the ones it matches; a rule
        that is switched off did not run."""
        rules = ar.preset()
        digest = ar.digest(rules)
        rules.items.append(
            ActionItem(
                action=ar.BUY, enabled=False,
                conditions=[Condition("bar.price", ar.OP_BELOW, 1.0)],
            )
        )
        assert ar.digest(rules) == digest

    def test_a_long_rule_set_collapses_to_a_digest(self):
        rules = RuleSet(
            items=[
                ActionItem(
                    action=ar.BUY, size_mode=ar.SIZE_SHARES, size=1.0,
                    conditions=[
                        Condition("bar.price", ar.OP_BELOW, 100.0 + i),
                        Condition("mom.score", ar.OP_ABOVE, 0.1 * i),
                        Condition("clock.minutes_from_open", ar.OP_ABOVE, i),
                    ],
                )
                for i in range(8)
            ]
        )
        signature = ar.signature(rules, TICKER)
        assert len(signature) < 60 and ar.digest(rules) in signature

    def test_only_the_models_the_rules_name_are_needed(self):
        assert ar.preset("Day range — two levels below the predicted high").models() == [
            "dayrange"
        ]
        price_only = RuleSet(
            items=[
                ActionItem(
                    action=ar.BUY, conditions=[Condition("bar.price", ar.OP_BELOW, 100.0)]
                )
            ]
        )
        assert price_only.models() == []


# ------------------------------------------------------------------ signals


class TestSignalBus:
    def _bus(self, closes=None, **kwargs):
        closes = closes or [100.0 + i * 0.01 for i in range(60)]
        defaults = dict(
            bundles={}, params=persistence_model.momentum_params(None),
            position=0.0, entry=None, cash=10_000.0,
        )
        defaults.update(kwargs)
        return SignalBus(frame(closes), **defaults)

    def test_the_bar_reads_off_the_last_closed_bar(self, market_open):
        bus = self._bus([100.0, 101.0, 102.0])
        assert bus.value("bar.price") == 102.0
        assert bus.value("bar.session_change_pct") == pytest.approx(2.0)

    def test_momentum_is_absent_until_the_window_fills(self, market_open):
        assert self._bus([100.0, 100.5, 101.0]).value("mom.score") is None
        assert self._bus().value("mom.score") is not None

    def test_the_regime_and_its_dwell_are_readable(self, market_open):
        bus = self._bus([100.0 + i * 0.05 for i in range(80)])
        assert bus.value("mom.regime") in (-1.0, 0.0, 1.0)
        assert bus.value("mom.bars_in_regime") >= 0

    def test_a_model_signal_is_absent_without_its_bundle(self, market_open):
        assert self._bus().value("nbeats.turn_proba") is None

    def test_position_signals_are_absent_while_flat(self, market_open):
        bus = self._bus()
        assert bus.value("pos.shares") == 0.0
        assert bus.value("pos.pnl_pct") is None
        assert bus.value("pos.drawdown_pct") is None

    def test_position_signals_read_the_open_trade(self, market_open):
        bus = self._bus(
            [100.0, 99.0], position=10.0, entry={"price": 100.0, "peak": 101.0, "bars": 4}
        )
        assert bus.value("pos.pnl_pct") == pytest.approx(-1.0)
        assert bus.value("pos.drawdown_pct") == pytest.approx((99.0 / 101.0 - 1) * 100)
        assert bus.value("pos.bars_held") == 4.0
        assert bus.value("pos.value") == pytest.approx(990.0)

    def test_the_change_since_the_last_buy_is_measured_from_its_peak(self, market_open):
        """The reference is max(fill price, every high since), so the signal is
        a give-back and never a gain."""
        bus = self._bus([100.0, 99.0], last_buy={"price": 100.0, "peak": 102.0})
        assert bus.value("pos.since_buy_pct") == pytest.approx((99.0 / 102.0 - 1) * 100)

        flat = self._bus([100.0, 100.0], last_buy={"price": 100.0, "peak": 100.0})
        assert flat.value("pos.since_buy_pct") == pytest.approx(0.0)

    def test_it_reads_with_no_position_but_not_before_the_first_buy(self, market_open):
        """The point of it: the position-anchored give-back disappears at the
        exit, this one does not."""
        never_bought = self._bus([100.0, 99.0])
        assert never_bought.value("pos.since_buy_pct") is None
        assert never_bought.value("pos.drawdown_pct") is None

        sold_out = self._bus(
            [100.0, 99.0], position=0.0, entry=None,
            last_buy={"price": 100.0, "peak": 100.0},
        )
        assert sold_out.value("pos.drawdown_pct") is None
        assert sold_out.value("pos.since_buy_pct") == pytest.approx(-1.0)

    def test_the_clock_reads_the_session(self, market_open):
        bus = self._bus([100.0, 101.0])
        assert bus.value("clock.minutes_from_open") == pytest.approx(60.0)
        assert bus.value("clock.minutes_to_close") == pytest.approx(330.0, abs=1.0)

    def test_day_range_distances_are_measured_in_average_daily_ranges(self, market_open):
        plan = {
            "date": None, "opening_end": frame([100.0]).index[0] - timedelta(days=1),
            "pred_high": 110.0, "pred_low": 100.0, "adr14_abs": 4.0,
        }
        bus = self._bus([104.0, 102.0], plan_fn=lambda: plan)
        # (110 - 102) / 4 -- two average daily ranges under the predicted high.
        assert bus.value("dayrange.pred_high_gap_adr") == pytest.approx(2.0)
        assert bus.value("dayrange.adr") == 4.0

    def test_the_forecast_is_only_made_if_a_rule_asks(self, market_open):
        calls = []

        def plan_fn():
            calls.append(1)
            return None

        bus = self._bus(plan_fn=plan_fn)
        bus.value("bar.price")
        assert calls == []
        bus.value("dayrange.pred_high")
        bus.value("dayrange.pred_low")
        assert calls == [1]  # asked once, cached even when it returns nothing

    def test_a_signal_that_blows_up_is_absent_rather_than_fatal(self, market_open):
        def explode():
            raise RuntimeError("no history")

        assert self._bus(plan_fn=explode).value("dayrange.pred_high") is None

    def test_only_what_was_read_is_reported(self, market_open):
        bus = self._bus()
        bus.value("bar.price")
        assert list(bus.computed()) == ["bar.price"]


# -------------------------------------------------------------------- cycles


def buy_below(price: float, **kwargs) -> ActionItem:
    return ActionItem(
        action=ar.BUY, size_mode=ar.SIZE_PCT, size=95.0,
        conditions=[Condition("bar.price", ar.OP_BELOW, price)], **kwargs
    )


def sell_on_giveback(pct: float = -0.5, **kwargs) -> ActionItem:
    return ActionItem(
        action=ar.SELL, size_mode=ar.SIZE_PCT, size=100.0,
        conditions=[Condition("pos.drawdown_pct", ar.OP_BELOW, pct)], **kwargs
    )


class Tape:
    """Feeds the trader a scripted sequence of closed bars, one per cycle."""

    def __init__(self, monkeypatch, broker: "FakeBroker | None" = None, end=MIDSESSION):
        self.broker = broker
        self.end = end
        self.closes: list[float] = []
        self.highs: list[float] = []
        monkeypatch.setattr(
            at2.persistence_model, "minute_frame",
            lambda *a, **k: frame(
                self.closes, end=self.end + timedelta(minutes=len(self.closes) - 1),
                highs=self.highs, lows=self.closes,
            ),
        )

    def bar(self, close: float, high: "float | None" = None) -> None:
        """Append one closed bar; the broker fills at its close."""
        self.closes.append(close)
        self.highs.append(close if high is None else high)
        if self.broker is not None:
            self.broker.price = close


def trader_for(*items, flatten: int = 5, ticker: str = TICKER) -> AppleTrader2:
    return AppleTrader2(
        AppleTrader2Config(
            rules=RuleSet(items=list(items)), flatten_before_close_min=flatten,
            ticker=ticker,
        ),
        bundles={},
    )


class TestCycle:
    def test_a_matching_rule_buys(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(buy_below(99.5))

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "hold"
        tape.bar(99.0)
        assert trader.run_cycle(state, tracker) == "bought"
        assert tracker.position_for(TICKER) == pytest.approx(95.95, abs=0.05)

    def test_the_reasoning_names_the_rule_and_the_number(self, state, market_open, monkeypatch):
        broker = FakeBroker(99.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(buy_below(99.5, label="Buy the dip"))

        tape.bar(99.0)
        trader.run_cycle(state, tracker)
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Rule 1" in reasoning and "Buy the dip" in reasoning and "99" in reasoning

    def test_one_action_per_closed_bar(self, state, market_open, monkeypatch):
        """A cycle that runs before a new bar has closed re-reads the same
        numbers, and must not act on them twice."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(buy_below(99.5))

        tape.bar(99.0)
        assert trader.run_cycle(state, tracker) == "bought"
        held = tracker.position_for(TICKER)
        assert trader.run_cycle(state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_the_first_rule_in_the_list_wins_the_bar(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(sell_on_giveback(-0.5), buy_below(101.0))

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "bought"
        # Price gives back 1% from the peak: both rules match, the sell is first.
        tape.bar(99.0)
        assert trader.run_cycle(state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0

    def test_a_partial_sell_keeps_the_position_and_its_entry(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        take_half = ActionItem(
            action=ar.SELL, size_mode=ar.SIZE_PCT, size=50.0,
            conditions=[Condition("pos.pnl_pct", ar.OP_ABOVE, 1.0)],
        )
        trader = trader_for(take_half, buy_below(100.5))

        tape.bar(100.0)
        trader.run_cycle(state, tracker)
        bought = tracker.position_for(TICKER)
        tape.bar(102.0)
        assert trader.run_cycle(state, tracker) == "sold"
        assert tracker.position_for(TICKER) == pytest.approx(bought / 2, abs=0.01)
        # The trailing peak and the entry price survive a partial exit: taking
        # half off does not restart the stop on the rest.
        assert trader.entry is not None
        assert trader.entry["price"] == pytest.approx(100.0)
        assert trader.entry["peak"] == pytest.approx(102.0)

    def test_a_second_buy_averages_the_entry_price(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker, trade_cost=0.0)
        tape = Tape(monkeypatch, broker)
        scale_in = ActionItem(
            action=ar.BUY, size_mode=ar.SIZE_PCT, size=50.0,
            conditions=[Condition("bar.price", ar.OP_BELOW, 101.0)],
        )
        trader = trader_for(scale_in)

        tape.bar(100.0)
        trader.run_cycle(state, tracker)
        tape.bar(98.0)
        trader.run_cycle(state, tracker)
        # 50 sh at 100 then ~25.5 sh at 98 -- the P&L a rule reads is the
        # position's, not the first slice's.
        assert 98.0 < trader.entry["price"] < 100.0

    def test_a_cooldown_holds_a_rule_back(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker, trade_cost=0.0)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(
            ActionItem(
                action=ar.BUY, size_mode=ar.SIZE_PCT, size=10.0, cooldown_bars=3,
                conditions=[Condition("bar.price", ar.OP_BELOW, 101.0)],
            )
        )

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "bought"
        for _ in range(2):
            tape.bar(100.0)
            assert trader.run_cycle(state, tracker) == "hold"
        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "bought"

    def test_nothing_is_opened_inside_the_flatten_window(
        self, state, near_close, monkeypatch
    ):
        broker = FakeBroker(99.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker, end=NEAR_CLOSE)
        trader = trader_for(buy_below(99.5))

        tape.bar(99.0)
        assert trader.run_cycle(state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_an_open_position_is_flattened_before_the_close(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        # No sell rule at all: the flatten is the agent's, not the list's.
        trader = trader_for(buy_below(101.0))

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "bought"

        clock.set_simulated(NEAR_CLOSE)
        tape.end = NEAR_CLOSE
        tape.bar(101.0)
        assert trader.run_cycle(state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_market_being_closed_is_not_a_cycle(self, state, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 2, 0, tzinfo=timezone.utc))
        try:
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
            Tape(monkeypatch, None)
            trader = trader_for(buy_below(101.0))
            assert trader.run_cycle(state, tracker) == "closed"
        finally:
            clock.clear()

    def test_a_symbol_that_is_not_streamed_is_reported(self, market_open, monkeypatch):
        state = AppState()
        state.set_symbols(["TSLA"])
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        assert trader_for(buy_below(101.0)).run_cycle(state, tracker) == "no_data"

    def test_the_buy_anchor_survives_the_exit_and_arms_the_re_entry(
        self, state, market_open, monkeypatch
    ):
        """The rule the signal exists for: sell, then buy back only once price
        has come 1% off the best price since that buy."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker, trade_cost=0.0)
        tape = Tape(monkeypatch, broker)
        re_enter = ActionItem(
            action=ar.BUY, size_mode=ar.SIZE_PCT, size=95.0,
            conditions=[
                Condition("pos.shares", ar.OP_BELOW, 0.0),
                Condition("pos.since_buy_pct", ar.OP_BELOW, -1.0),
            ],
        )
        trader = trader_for(
            sell_on_giveback(-0.5), re_enter, buy_below(100.5, cooldown_bars=390)
        )

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "bought"  # the first entry
        tape.bar(101.0, high=101.0)
        assert trader.run_cycle(state, tracker) == "hold"
        tape.bar(100.4)  # 0.6% off the 101.00 peak -- the stop
        assert trader.run_cycle(state, tracker) == "sold"

        # Flat now, so the position-anchored give-back is gone, but the anchor
        # from the buy at 100.00 (peak 101.00) is not: -1% of 101.00 is 99.99.
        tape.bar(100.1)
        assert trader.run_cycle(state, tracker) == "hold"
        tape.bar(99.9)
        assert trader.run_cycle(state, tracker) == "bought"
        # ...and that buy re-anchors it, so the same rule cannot fire again
        # until price falls another 1% from here.
        assert trader.last_buy == {"price": 99.9, "peak": 99.9}
        tape.bar(99.5)
        assert trader.run_cycle(state, tracker) == "hold"

    def test_the_anchor_ratchets_on_bars_after_the_buy_only(
        self, state, market_open, monkeypatch
    ):
        """The bar the buy filled on happened before the position did, so its
        high is not part of the range -- the same rule the position peak uses."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(buy_below(100.5))

        trader = trader_for(buy_below(100.5, cooldown_bars=390))

        tape.bar(100.0, high=105.0)
        assert trader.run_cycle(state, tracker) == "bought"
        assert trader.last_buy == {"price": 100.0, "peak": 100.0}
        tape.bar(100.5, high=102.0)
        trader.run_cycle(state, tracker)
        assert trader.last_buy["peak"] == 102.0

    def test_the_anchor_is_dropped_at_the_session_roll(
        self, state, market_open, monkeypatch
    ):
        """Yesterday's reference would be measured across the overnight gap."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = trader_for(buy_below(100.5))

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "bought"
        assert trader.last_buy is not None

        # A new session, on a tape the buy rule wants nothing to do with.
        tape.closes, tape.highs = [], []
        tape.end = MIDSESSION + timedelta(days=1)
        clock.set_simulated(tape.end)
        tape.bar(200.0)
        trader.run_cycle(state, tracker)
        assert trader.last_buy is None

    def test_a_restart_onto_an_open_book_adopts_the_position(
        self, state, market_open, monkeypatch
    ):
        """The give-back is then measured from here, not from a peak this
        instance never saw."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tracker.record_trade(TICKER, "buy", 10.0, "seeded", "k", "s")
        tape = Tape(monkeypatch, broker)
        trader = trader_for(sell_on_giveback(-0.5), buy_below(90.0))

        tape.bar(100.0)
        assert trader.run_cycle(state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(100.0)
        tape.bar(99.0)
        assert trader.run_cycle(state, tracker) == "sold"


class TestLaunch:
    def test_launching_a_broken_rule_set_stops_the_run(self, state, market_open, monkeypatch):
        """The validation runs before the loop, so a configuration that cannot
        work reports itself instead of producing an empty ledger."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at2.scoring, "begin_session", lambda *a, **k: None)
        monkeypatch.setattr(at2.scoring, "end_session", lambda *a, **k: None)
        at2._apple_trader2_loop(
            state, tracker, AppleTrader2Config(rules=RuleSet()), 60, threading.Event()
        )
        assert any(entry.get("type") == "error" for entry in state.agent_log)
        assert state.agent_running is False
