"""Apple Trader: the rule-based loop over the momentum-persistence model.

Every cycle is driven through a stubbed model read, so these pin the RULES --
when it buys, when it refuses to, and every way it gets back out -- without
depending on the saved artifact or on live market data.
"""

import threading
from datetime import datetime, timezone

import pandas as pd
import pytest

from agent_stonks import apple_trader as at
from agent_stonks import clock
from agent_stonks.apple_trader import TICKER, AppleTrader, AppleTraderConfig, config_signature
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

# 10:30 ET on a Tuesday: mid-session, well clear of both the open and the close.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)

BUNDLE = {"pipeline": None, "feature_columns": [], "seq_len": 20, "threshold": 0.07}


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
def state() -> AppState:
    state = AppState()
    state.set_symbols([TICKER])
    state.api_key = "k"
    state.api_secret = "s"
    state.feed = "iex"
    return state


class Reads:
    """Feeds the trader a scripted sequence of model reads, one per cycle."""

    def __init__(self, monkeypatch, broker: "FakeBroker | None" = None):
        self.broker = broker
        self.minute = 0
        self.next_read: dict = {}
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(at.persistence_model, "read_latest", lambda *a, **k: self.next_read)

    def set(
        self,
        *,
        price: float = 100.0,
        high: "float | None" = None,
        mom: float = 1.2,
        regime: int = 1,
        prev_regime: "int | None" = 0,
        change: bool = False,
        proba: "float | None" = None,
        turn_proba: "float | None" = None,
        reversal_proba: "float | None" = None,
        pre_dwell: "int | None" = 20,
        bars_in_regime: int = 20,
        bars_today: int = 200,
        warming_up: bool = False,
        advance: bool = True,
    ) -> dict:
        """Stage the next bar. `advance=False` replays the SAME timestamp, the
        way a cycle that runs before a new bar has closed would see it."""
        if advance:
            self.minute += 1
        if self.broker is not None:
            self.broker.price = price
        self.next_read = {
            "ts": pd.Timestamp("2026-07-21 10:30", tz="America/New_York")
            + pd.Timedelta(minutes=self.minute),
            "price": price,
            "high": price if high is None else high,
            "mom": mom,
            "regime": regime,
            "prev_regime": prev_regime,
            "regime_change": change,
            "to_positive": change and regime == 1,
            "pre_dwell": pre_dwell if change else None,
            "bars_in_regime": bars_in_regime,
            "proba": proba,
            "turn_proba": turn_proba,
            "reversal_proba": reversal_proba,
            "bars_today": bars_today,
            "warming_up": warming_up,
        }
        return self.next_read

    def to_positive(self, *, proba: float, **kwargs) -> dict:
        """The bar `confirm` acts on: a change into positive, already printed."""
        return self.set(regime=1, prev_regime=0, change=True, proba=proba, **kwargs)

    def pre_turn(self, *, turn_proba: "float | None", regime: int = 0, **kwargs) -> dict:
        """The bar `anticipate` acts on: the regime has NOT turned positive, and
        the forecaster has been asked whether it is about to.

        `read_latest` only fills `turn_proba` in on such a bar, so a stub that
        set it beside `regime=1` would be testing a state the pipeline cannot
        produce.
        """
        return self.set(regime=regime, prev_regime=regime, turn_proba=turn_proba, **kwargs)


def confirm_config(**kwargs) -> AppleTraderConfig:
    """A rule set on the `confirm` entry, which is no longer the default.

    The trailing stop, the flatten rule and the guards are shared by both entry
    modes, so the suites below pin them through the mode whose trigger is the
    notebook's and whose stub is a single scripted bar.
    """
    return AppleTraderConfig(entry_mode=at.ENTRY_CONFIRM, **kwargs)


class TestEntry:
    """The `confirm` trigger: buy a change into positive that has already
    printed, if the model rates it likely to hold."""

    def _trader(self, **kwargs) -> AppleTrader:
        return AppleTrader(confirm_config(**kwargs), model_threshold=0.07)

    def test_buys_a_to_positive_change_the_model_backs(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=0.62)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0
        assert "62%" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_probability_below_the_threshold_is_not_a_buy(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=0.49)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_bundles_own_threshold_applies_when_none_is_configured(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(confirm_config(prob_threshold=None), model_threshold=0.07)
        assert trader.prob_threshold == pytest.approx(0.07)

        reads.to_positive(proba=0.10)  # under any sane default, over this model's
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"

    def test_a_change_out_of_positive_is_not_an_entry(self, state, market_open, monkeypatch):
        """Only changes INTO the positive regime are ever traded; the model is
        not even asked about the others, so `proba` is None on them."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        for regime, prev in ((0, 1), (-1, 1)):
            reads.set(regime=regime, prev_regime=prev, change=True, proba=None)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_sitting_in_the_positive_regime_is_not_a_change(self, state, market_open, monkeypatch):
        """The signal is the transition, not the state: momentum can be
        strongly positive for an hour without the loop ever buying."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        for _ in range(5):
            reads.set(regime=1, change=False, mom=2.5, proba=0.99)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_an_unscoreable_change_is_not_a_buy(self, state, market_open, monkeypatch):
        """A change whose 20-bar feature window hasn't warmed up leaves `proba`
        None. An unasked model is not a yes."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=None)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_does_not_trade_during_the_models_warm_up(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=None, warming_up=True, bars_today=12)
        assert trader.run_cycle(BUNDLE, state, tracker) == "warming_up"
        assert tracker.position_for(TICKER) == 0

    def test_re_reading_the_same_bar_does_not_re_enter(self, state, market_open, monkeypatch):
        """One closed bar is one decision. A cycle that runs before the next bar
        has arrived must not act on the same change twice."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5, trail_pct=0.5)

        reads.to_positive(proba=0.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        # Same bar, deeper drawdown than the stop allows: the exit fires...
        reads.to_positive(proba=0.9, price=99.0, advance=False)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        # ...and the stale bar cannot immediately buy the same change back.
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_position_size_follows_the_configured_share_of_cash(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0), trade_cost=0.0)
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5, position_pct=50.0)

        reads.to_positive(proba=0.9, price=100.0)
        trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == pytest.approx(50.0)

    def test_no_entry_inside_the_closing_flatten_window(self, state, market_open, monkeypatch):
        """The notebook's simulator makes no decision on the last bar of a
        session; the same reasoning ends this loop's entries once the flatten
        rule is in force, since the position would be sold straight back."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5, flatten_before_close_min=5)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.to_positive(proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        assert any("Standing down" in e.get("text", "") for e in state.agent_log)

    def test_a_signal_just_before_the_window_still_trades(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5, flatten_before_close_min=5)

        clock.set_simulated(datetime(2026, 7, 21, 19, 54, tzinfo=timezone.utc))  # 15:54 ET
        reads.to_positive(proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"

    def test_no_second_entry_while_already_long(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=0.9)
        trader.run_cycle(BUNDLE, state, tracker)
        size = tracker.position_for(TICKER)
        for _ in range(3):
            reads.to_positive(proba=0.9, price=100.5)
            trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == size


class TestAnticipateEntry:
    """The default trigger: buy while the regime is still negative or balanced,
    on the forecast that it turns positive next bar.

    The whole point of this mode is *where in the transition* the order goes
    in, so these pin which bars can and cannot produce one -- not just the
    threshold arithmetic.
    """

    def _trader(self, **kwargs) -> AppleTrader:
        return AppleTrader(
            AppleTraderConfig(entry_mode=at.ENTRY_ANTICIPATE, **kwargs),
            model_threshold=0.05,
        )

    def test_buys_a_balanced_bar_the_forecast_expects_to_turn(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=0.45, regime=0, mom=0.8)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_buys_out_of_the_negative_regime_too(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=0.31, regime=-1, mom=-0.5)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_the_regime_is_not_yet_positive_when_the_order_goes_in(
        self, state, market_open, monkeypatch
    ):
        """The regression this mode exists for. `confirm` can only ever buy a
        bar whose regime has already turned positive, which puts the entry
        after the momentum score has crossed its threshold and after the move
        that pushed it there. Here the ledger records a bar that has not."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2)

        read = reads.pre_turn(turn_proba=0.45, regime=0, mom=0.8, bars_in_regime=44)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert read["regime"] != 1
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "still balanced" in reasoning
        assert "held 44 bars" in reasoning
        assert "45%" in reasoning

    def test_a_forecast_below_the_threshold_is_not_a_buy(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=0.19)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_an_unscoreable_bar_is_not_a_buy(self, state, market_open, monkeypatch):
        """A bar whose 20-bar window hasn't warmed up, or a bundle that cannot
        forecast at all, leaves `turn_proba` None. An unasked model is not a
        yes -- and it must not fall through to the persistence answer."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=None, proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_confirmed_change_is_no_longer_an_entry(
        self, state, market_open, monkeypatch
    ):
        """Once the change has printed, this mode has missed it and says so by
        standing aside -- it does not chase the bar `confirm` would have
        bought."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.2)

        reads.to_positive(proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_no_entry_inside_the_closing_flatten_window(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2, flatten_before_close_min=5)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.pre_turn(turn_proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_re_reading_the_same_bar_does_not_re_enter(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2, trail_pct=0.5)

        reads.pre_turn(turn_proba=0.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        reads.pre_turn(turn_proba=0.9, price=99.0, advance=False)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_trailing_stop_is_the_exit_here_too(
        self, state, market_open, monkeypatch
    ):
        """Nothing about the exit changes with the entry mode: once the
        position is on, only price decides when it comes off."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2, trail_pct=0.5)

        reads.pre_turn(turn_proba=0.9, price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        reads.set(price=102.0, regime=1)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=101.4, regime=1)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"


def _enter(state, tracker, reads, config) -> AppleTrader:
    """Take a long at $100 so the exit rule has something to act on."""
    trader = AppleTrader(config, model_threshold=0.07)
    reads.to_positive(proba=0.9, price=100.0)
    assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
    return trader


class TestTrailingStop:
    def test_sells_once_price_gives_back_the_configured_percent(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=99.6)  # -0.4% from the peak: still inside the give-back
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=99.5)  # -0.5%: at the line
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Trailing stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_stop_trails_the_high_since_entry_not_the_entry(
        self, state, market_open, monkeypatch
    ):
        """A drop that would be harmless measured from the entry still sells
        once the position has been further ahead than that."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=102.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        # +1.4% on the trade, but 0.6% off the $102.00 peak.
        reads.set(price=101.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        exit_reason = tracker.snapshot()["decisions"][-1].reasoning
        assert "102.00 high" in exit_reason and "+1.40%" in exit_reason

    def test_the_peak_ratchets_up_and_never_down(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=1.0))

        for price in (101.0, 100.4, 103.0, 102.5):
            reads.set(price=price)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(103.0)

    def test_the_peak_comes_from_the_bar_high_not_its_close(
        self, state, market_open, monkeypatch
    ):
        """The stop trails the highest price the position actually traded at,
        so a bar that spiked and gave most of it back still raises the peak."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=100.5, high=100.6)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(100.6)
        # Still above the entry and only -0.45% off the last close: nothing but
        # the $100.60 print inside the previous bar explains this exit.
        reads.set(price=100.05, high=100.05)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "100.60 high" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_entry_bars_own_high_is_not_part_of_the_peak(
        self, state, market_open, monkeypatch
    ):
        """A spike earlier in the change bar happened before the position
        existed, so the stop does not trail from it."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = AppleTrader(confirm_config(trail_pct=0.5), model_threshold=0.07)

        reads.to_positive(proba=0.9, price=100.0, high=102.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert trader.entry["peak"] == pytest.approx(100.0)
        reads.set(price=99.7, high=99.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"

    def test_before_any_new_high_the_stop_sits_under_the_entry(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=99.51)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=99.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"

    def test_momentum_turning_negative_does_not_close_the_position(
        self, state, market_open, monkeypatch
    ):
        """The regime *having* turned is not an exit -- only price and the
        model's forecast are, and this bar carries neither.

        Momentum that has already gone negative is exactly the give-back the
        trailing stop is measuring, so acting on it as well would be the same
        rule twice at a worse level. What the forecast exit acts on is the bar
        *before* this one, which is the whole distinction (see TestReversalExit).
        """
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=2.0))

        for _ in range(5):
            reads.set(price=99.5, mom=-1.5, regime=-1, prev_regime=0, change=True)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_flattens_before_the_close(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=5.0))

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.set(price=100.2)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning

    def test_adopts_a_position_it_did_not_open(self, state, market_open, monkeypatch):
        """Restarted onto a ledger that already holds shares: the stop has no
        peak it ever saw, so it starts trailing from the current price."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tracker.record_trade(TICKER, "buy", 10, "seeded", "k", "s")
        reads = Reads(monkeypatch, broker)
        trader = AppleTrader(confirm_config(trail_pct=0.5), model_threshold=0.07)

        reads.set(price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(100.0)
        reads.set(price=99.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"


def reversal_config(**kwargs) -> AppleTraderConfig:
    """A rule set whose forecast exit is armed, on the `confirm` entry so the
    position can be taken with one scripted bar."""
    kwargs.setdefault("reversal_threshold", 0.30)
    return confirm_config(**kwargs)


class TestReversalExit:
    """The second exit: the model calling the end of the regime it bought.

    The stub supplies `reversal_proba` the way `read_latest` does -- filled in
    only on a positive bar reached while holding, and None everywhere else --
    so these pin the rule without depending on a 200 MB forecaster.
    """

    def test_sells_when_the_forecast_clears_the_threshold(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=5.0))

        reads.set(price=101.0, reversal_proba=0.29)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=101.0, reversal_proba=0.30)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Forecast reversal" in tracker.snapshot()["decisions"][-1].reasoning

    def test_sells_at_the_high_before_any_give_back(
        self, state, market_open, monkeypatch
    ):
        """The point of the rule: out while the trade is still at its peak,
        which the trailing stop can never do by construction."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=0.5))

        reads.set(price=103.0, reversal_proba=0.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "+3.00%" in reasoning and "90%" in reasoning

    def test_off_by_default_for_records_that_never_had_it(
        self, state, market_open, monkeypatch
    ):
        """`reversal_threshold=None` is the pre-existing strategy exactly: the
        forecast is ignored however loud it gets."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(
            state, tracker, reads, confirm_config(trail_pct=5.0, reversal_threshold=None)
        )

        reads.set(price=101.0, reversal_proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_an_unasked_model_is_not_a_sell(self, state, market_open, monkeypatch):
        """`read_latest` leaves the number None on any bar that does not pose
        the question. An absent probability is not a quiet zero *or* a quiet
        yes -- it is a bar with nothing to say."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=5.0))

        reads.set(price=101.0, reversal_proba=None)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_the_trailing_stop_still_wins_a_bar_they_both_fire_on(
        self, state, market_open, monkeypatch
    ):
        """Both exits close the position, so the only thing at stake is the
        ledger's account of why -- and a give-back that actually happened is a
        better explanation than a forecast that agreed with it."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=0.5))

        reads.set(price=99.0, reversal_proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "Trailing stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_question_is_only_asked_while_holding(
        self, state, market_open, monkeypatch
    ):
        """Every ask costs a full forecast and nothing acts on the answer with
        the book flat, so `run_cycle` must not pay for it when it is not long."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        asked: list[bool] = []
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        reads = Reads(monkeypatch, broker)

        def spy(bundle, frame, holding=False):
            asked.append(holding)
            return reads.next_read

        monkeypatch.setattr(at.persistence_model, "read_latest", spy)
        trader = AppleTrader(reversal_config(trail_pct=5.0), model_threshold=0.07)

        reads.set(price=100.0)  # flat
        trader.run_cycle(BUNDLE, state, tracker)
        reads.to_positive(proba=0.9, price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        reads.set(price=101.0)  # long
        trader.run_cycle(BUNDLE, state, tracker)
        assert asked == [False, False, True]

    def test_a_disarmed_rule_never_pays_for_the_forecast(
        self, state, market_open, monkeypatch
    ):
        """Holding is not enough: with the rule off the question is pointless,
        and asking it would make switching the rule off cost the same as
        leaving it on."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        asked: list[bool] = []
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        reads = Reads(monkeypatch, broker)

        def spy(bundle, frame, holding=False):
            asked.append(holding)
            return reads.next_read

        monkeypatch.setattr(at.persistence_model, "read_latest", spy)
        trader = _enter(
            state, tracker, reads, confirm_config(trail_pct=5.0, reversal_threshold=None)
        )
        reads.set(price=101.0)
        trader.run_cycle(BUNDLE, state, tracker)
        assert asked == [False, False]

    def test_an_out_of_range_threshold_is_refused(self):
        with pytest.raises(ValueError, match="not a probability"):
            AppleTraderConfig(reversal_threshold=1.5)


class TestGuards:
    def test_does_nothing_when_the_market_is_closed(self, state, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 2, 0, tzinfo=timezone.utc))  # 22:00 ET Monday
        try:
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
            reads = Reads(monkeypatch)
            reads.to_positive(proba=0.99)
            assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "closed"
            assert tracker.position_for(TICKER) == 0
        finally:
            clock.clear()

    def test_reports_when_the_ticker_is_not_streamed(self, market_open, monkeypatch):
        state = AppState()
        state.set_symbols(["MSFT"])
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        Reads(monkeypatch)
        assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "no_data"
        assert any(e["type"] == "error" for e in state.agent_log)

    def test_reports_when_there_is_not_enough_history(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(at.persistence_model, "read_latest", lambda *a, **k: None)
        assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "no_data"

    def test_missing_model_stops_the_loop_instead_of_trading(self, state, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.persistence_model, "load_bundle", lambda: None)
        stop_event = threading.Event()
        at._apple_trader_loop(
            state, tracker, confirm_config(model_key="persistence"), 60, stop_event
        )
        assert state.agent_running is False
        assert any("cannot run without it" in e.get("text", "") for e in state.agent_log)

    def test_anticipating_on_a_model_that_cannot_forecast_stops_the_loop(
        self, state, monkeypatch
    ):
        """The failure this prevents is the quiet one: `read_latest` leaves
        `turn_proba` None on a classifier, every bar reads as "not a buy", and
        the run finishes clean with an empty ledger that looks like a result."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.apple_models, "load", lambda key: BUNDLE)
        config = AppleTraderConfig(
            model_key="persistence", entry_mode=at.ENTRY_ANTICIPATE
        )
        at._apple_trader_loop(state, tracker, config, 60, threading.Event())
        assert state.agent_running is False
        assert any(
            "cannot forecast" in e.get("text", "") and e["type"] == "error"
            for e in state.agent_log
        )

    def test_the_reversal_exit_on_a_model_that_cannot_forecast_stops_the_loop(
        self, state, monkeypatch
    ):
        """Quieter than the entry version and caught for the same reason: the
        run would trade normally and exit everything on the trailing stop,
        which is indistinguishable from a rule that just never triggered."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.apple_models, "load", lambda key: BUNDLE)
        config = confirm_config(model_key="persistence", reversal_threshold=0.3)
        at._apple_trader_loop(state, tracker, config, 60, threading.Event())
        assert state.agent_running is False
        assert any(
            "cannot forecast the breakdown" in e.get("text", "") and e["type"] == "error"
            for e in state.agent_log
        )

    def test_clearing_the_reversal_exit_lets_that_model_run(self, state, monkeypatch):
        """The classifier is not disqualified -- only that one rule is."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.apple_models, "load", lambda key: BUNDLE)
        config = confirm_config(model_key="persistence", reversal_threshold=None)
        assert at.config_error(config, BUNDLE) is None
        stop = threading.Event()
        stop.set()
        at._apple_trader_loop(state, tracker, config, 60, stop)
        assert not any(e["type"] == "error" for e in state.agent_log)

    def test_the_loop_loads_the_model_the_config_names(self, state, monkeypatch):
        """A config naming an unavailable model stops on *that* model rather
        than quietly running the one that happens to be loadable."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(
            at.apple_models, "load", lambda key: None if key == "nbeats" else BUNDLE
        )
        at._apple_trader_loop(
            state, tracker, AppleTraderConfig(model_key="nbeats"), 60, threading.Event()
        )
        assert state.agent_running is False
        assert any(
            "timetochange2_nbeats.pt" in e.get("text", "") for e in state.agent_log
        )


class TestConfigSignature:
    def test_every_rule_that_changes_behaviour_is_in_the_signature(self):
        base = config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5))
        assert base != config_signature(AppleTraderConfig(prob_threshold=0.3, trail_pct=0.5))
        assert base != config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.8))
        assert base != config_signature(
            AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5, model_key="persistence")
        )
        # The same model answering the other question is a different experiment.
        assert base != config_signature(
            confirm_config(prob_threshold=0.2, trail_pct=0.5)
        )
        assert base == config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5))

    def test_arming_the_reversal_exit_is_a_new_configuration(self):
        off = config_signature(AppleTraderConfig(prob_threshold=0.2, reversal_threshold=None))
        armed = config_signature(AppleTraderConfig(prob_threshold=0.2, reversal_threshold=0.3))
        assert off != armed
        assert "rev>=0.3" in armed
        # Retuning it is a new configuration too.
        assert armed != config_signature(
            AppleTraderConfig(prob_threshold=0.2, reversal_threshold=0.4)
        )

    def test_a_rule_set_without_it_signs_as_it_always_did(self):
        """Runs recorded before this exit existed and runs configured without
        it now are the same strategy, so Results must go on grouping them."""
        off = config_signature(AppleTraderConfig(prob_threshold=0.2, reversal_threshold=None))
        assert off == "nbeats_AAPL(anticipate,p>=0.2,trail=0.5%,size=95%)"

    def test_an_unset_threshold_names_the_model_that_supplies_it(self):
        config = AppleTraderConfig(prob_threshold=None)
        assert "p>=model" in config_signature(config)
        assert "p>=0.07" in config_signature(config, model_threshold=0.07)


class TestCycleTiming:
    def test_wakes_just_after_the_next_bar_closes(self):
        clock.set_simulated(datetime(2026, 7, 21, 14, 30, 20, tzinfo=timezone.utc))
        try:
            # 40s to the boundary, plus the lag that lets the bar arrive.
            assert at._seconds_to_next_bar(60, lag=5.0) == pytest.approx(45.0)
        finally:
            clock.clear()

    def test_the_lag_is_added_on_top_of_the_boundary(self):
        """Just past a boundary it waits for the NEXT one, never skipping a bar
        by landing before the lag."""
        clock.set_simulated(datetime(2026, 7, 21, 14, 30, 3, tzinfo=timezone.utc))
        try:
            assert at._seconds_to_next_bar(60, lag=5.0) == pytest.approx(62.0)
        finally:
            clock.clear()


# --------------------------------------------------------------------------
# The day-range rules (TimeToChange3): one forecast, two resting levels.
#
# Driven through a stubbed forecast for the same reason the momentum suites
# stub the model read: these pin the RULES -- when a level is a buy, when it is
# a sell, and what the opening window and the closing bell override -- without
# depending on the saved bundle. `tests/test_dayrange_model.py` pins the
# forecast itself.
# --------------------------------------------------------------------------

DAYRANGE_BUNDLE = {"opening_minutes": 5}

# The forecast the stub returns: a $10 average daily range around a predicted
# high of $110, so at the shipped 0.75 / 0.10 the levels land on round numbers.
FORECAST = {
    "pred_high": 110.0,
    "pred_low": 95.0,
    "prev_avg": 102.0,
    "adr14_abs": 10.0,
    "or_high": 103.0,
    "or_low": 101.0,
}
BUY_LEVEL = 102.5   # 110 - 0.75 x 10
SELL_LEVEL = 109.0  # 110 - 0.10 x 10


class Tape:
    """A growing frame of today's minute bars, as `minute_frame` returns it.

    The first five bars are the 09:30 opening window the forecast is built on;
    everything after them is a tradable bar the test appends one at a time.
    """

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(self, monkeypatch, broker=None, minutes: int = 5):
        self.broker = broker
        self.rows: list[dict] = []
        self.index: list[pd.Timestamp] = []
        self.forecast_calls = 0
        for i in range(minutes):
            self.append(101.0 + i * 0.1, low=100.9, high=101.5, offset=i)

        dayrange = at._dayrange()
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: self.frame()
        )
        monkeypatch.setattr(at.historical, "fetch_daily_ohlc_bars", lambda *a, **k: [])
        monkeypatch.setattr(at.historical, "fetch_session_open", lambda *a, **k: None)
        monkeypatch.setattr(dayrange, "forecast_session", self._forecast)

    def _forecast(self, *args, **kwargs):
        self.forecast_calls += 1
        return dict(FORECAST)

    def append(self, close: float, low=None, high=None, offset=None):
        """One more closed bar. `offset` is minutes from the open; without it
        the bar lands at 10:30, comfortably past the opening window."""
        if offset is None:
            offset = 60 + len(self.rows)
        self.index.append(self.OPEN + pd.Timedelta(minutes=offset))
        self.rows.append(
            {
                "open": close,
                "high": close if high is None else high,
                "low": close if low is None else low,
                "close": close,
                "volume": 1.0e5,
                "minutes_from_open": float(offset),
            }
        )
        if self.broker is not None:
            self.broker.price = close

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))


def dayrange_config(**kwargs) -> AppleTraderConfig:
    return AppleTraderConfig(model_key="dayrange", **kwargs)


class TestDayRangeEntry:
    def _trader(self, **kwargs):
        return at.DayRangeTrader(dayrange_config(**kwargs))

    def test_a_bar_that_trades_down_to_the_buy_level_is_bought(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader()

        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "102.50" in reasoning and "110.00" in reasoning

    def test_a_bar_that_stays_above_the_buy_level_is_not(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader()

        tape.append(104.0, low=BUY_LEVEL + 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_levels_move_with_the_configured_distances(
        self, state, market_open, monkeypatch
    ):
        """The two knobs are the whole strategy: a shallower buy distance turns
        the same bar from a hold into a fill."""
        broker = FakeBroker(105.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(buy_k=0.4)  # buy level 106.0 rather than 102.5

        tape.append(105.0, low=105.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert trader.plan["buy_level"] == pytest.approx(106.0)
        assert trader.plan["sell_level"] == pytest.approx(SELL_LEVEL)

    def test_nothing_trades_before_the_opening_window_closes(
        self, state, market_open, monkeypatch
    ):
        """The forecast does not exist before 9:35, so neither does the rule --
        even on a bar that is below where the buy level will turn out to be."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        tape = Tape(monkeypatch, minutes=3)
        trader = self._trader()

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.plan is None
        assert tape.forecast_calls == 0
        assert tracker.position_for(TICKER) == 0

    def test_the_last_bar_of_the_opening_window_is_not_traded(
        self, state, market_open, monkeypatch
    ):
        """The forecast is built *from* that bar, so acting on it would be
        trading the same minute the model was just handed. The notebook skips
        it too (`start_after`)."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        tape = Tape(monkeypatch, minutes=5)
        tape.rows[-1]["low"] = BUY_LEVEL - 5  # deep enough to fill, if it counted
        trader = self._trader()

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.plan is not None  # the forecast IS made on that bar
        assert tracker.position_for(TICKER) == 0

    def test_the_forecast_is_made_once_and_reused_all_day(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader()

        for _ in range(6):
            tape.append(104.0)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 1

    def test_a_replayed_bar_does_not_buy_twice(self, state, market_open, monkeypatch):
        """A cycle that runs before a new bar closes sees the same one again."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader()

        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        held = tracker.position_for(TICKER)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held


class TestDayRangeExit:
    def _entered(self, state, tracker, tape, **kwargs):
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader

    def test_a_bar_that_trades_up_to_the_sell_level_closes_the_position(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_position_that_never_reaches_the_target_is_simply_held(
        self, state, market_open, monkeypatch
    ):
        """No stop, by design: the forecast says where the day tops out, and
        bailing on weakness would be a second, unmeasured rule."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        for price in (101.0, 99.0, 96.0, 94.0):
            tape.append(price)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_the_closing_bell_flattens_what_the_day_never_paid_out(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        tape.append(104.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_rule_re_arms_after_a_sale(self, state, market_open, monkeypatch):
        """The levels are resting orders, not a one-shot: a day that dips,
        recovers and dips again is traded twice."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        fills = [d for d in tracker.snapshot()["decisions"] if d.status == "filled"]
        assert [d.action for d in fills] == ["buy", "sell", "buy"]


class TestDayRangeGuards:
    def test_no_entry_inside_the_closing_flatten_window(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        # Warm the plan up while the session still has hours left.
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_a_forecast_that_cannot_be_made_stops_the_day_rather_than_the_bar(
        self, state, market_open, monkeypatch
    ):
        """Too little daily history at 9:35 is still too little at 14:00, so
        the refusal is logged once and the session is skipped -- not retried
        every minute for six and a half hours."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise ValueError("only 40 daily sessions of history")

        monkeypatch.setattr(at._dayrange(), "forecast_session", boom)
        trader = at.DayRangeTrader(dayrange_config())

        for _ in range(4):
            tape.append(103.0, low=BUY_LEVEL - 0.01)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "no_data"
        assert calls["n"] == 1
        assert tracker.position_for(TICKER) == 0
        errors = [e for e in state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1 and "40 daily sessions" in errors[0]["text"]

    def test_an_opening_window_the_buffer_never_saw_is_refused(
        self, state, market_open, monkeypatch
    ):
        """An agent started at 10:30 has a buffer that begins at 10:30. Taking
        its first five bars as "the open" would forecast confidently off the
        wrong five minutes."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch, minutes=0)
        for _ in range(6):
            tape.append(103.0, low=BUY_LEVEL - 0.01)
        monkeypatch.setattr(at.agent_mod, "fetch_bars_window", lambda *a, **k: [])
        trader = at.DayRangeTrader(dayrange_config())

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "no_data"
        assert tracker.position_for(TICKER) == 0
        assert any(
            "09:30 window" in e.get("text", "")
            for e in state.agent_log
            if e.get("type") == "error"
        )

    def test_a_new_session_forgets_yesterdays_levels(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 1

        clock.set_simulated(datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc))
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 2


class TestStrategySelection:
    def test_the_model_chooses_the_state_machine(self):
        assert isinstance(
            at.build_trader(dayrange_config(), DAYRANGE_BUNDLE), at.DayRangeTrader
        )
        assert isinstance(at.build_trader(AppleTraderConfig(), BUNDLE), AppleTrader)

    def test_the_momentum_pairing_checks_do_not_fire_on_the_other_strategy(self):
        """`anticipate` and the reversal exit are momentum concepts. A
        day-range config carries their defaults and must not be rejected for
        them -- the bundle it runs on cannot answer either question and is
        never asked."""
        assert at.config_error(dayrange_config(), DAYRANGE_BUNDLE) is None

    def test_the_levels_are_the_signature_and_the_momentum_knobs_are_not(self):
        base = config_signature(dayrange_config())
        assert base == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert base != config_signature(dayrange_config(buy_k=0.8))
        assert base != config_signature(dayrange_config(sell_k=0.2))
        # Inert knobs must not split one strategy's runs into two
        # configurations in Results.
        assert base == config_signature(dayrange_config(trail_pct=2.0, prob_threshold=0.9))

    def test_a_sell_level_below_the_buy_level_is_refused(self):
        """Both are distances *below* the predicted high, so the sell distance
        has to be the smaller number. The other way round the rule would sell
        under its own entry on every bar."""
        with pytest.raises(ValueError, match="sell_k"):
            AppleTraderConfig(buy_k=0.5, sell_k=0.5)
        with pytest.raises(ValueError, match="sell_k"):
            AppleTraderConfig(buy_k=0.2, sell_k=0.6)
