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
        pre_dwell: "int | None" = 20,
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
            "proba": proba,
            "bars_today": bars_today,
            "warming_up": warming_up,
        }
        return self.next_read

    def to_positive(self, *, proba: float, **kwargs) -> dict:
        """The one bar the model is ever asked about: a change into positive."""
        return self.set(regime=1, prev_regime=0, change=True, proba=proba, **kwargs)


class TestEntry:
    def _trader(self, **kwargs) -> AppleTrader:
        return AppleTrader(AppleTraderConfig(**kwargs), model_threshold=0.07)

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
        trader = AppleTrader(AppleTraderConfig(prob_threshold=None), model_threshold=0.07)
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
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=0.5))

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
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=0.5))

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
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=1.0))

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
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=0.5))

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
        trader = AppleTrader(AppleTraderConfig(trail_pct=0.5), model_threshold=0.07)

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
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=0.5))

        reads.set(price=99.51)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=99.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"

    def test_momentum_turning_negative_does_not_close_the_position(
        self, state, market_open, monkeypatch
    ):
        """Only price decides the exit. The model called the entry; it gets no
        vote on the way out, and neither does the regime it predicted."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=2.0))

        for _ in range(5):
            reads.set(price=99.5, mom=-1.5, regime=-1, prev_regime=0, change=True)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_flattens_before_the_close(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, AppleTraderConfig(trail_pct=5.0))

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
        trader = AppleTrader(AppleTraderConfig(trail_pct=0.5), model_threshold=0.07)

        reads.set(price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(100.0)
        reads.set(price=99.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"


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
        at._apple_trader_loop(state, tracker, AppleTraderConfig(), 60, stop_event)
        assert state.agent_running is False
        assert any("cannot run without it" in e.get("text", "") for e in state.agent_log)


class TestConfigSignature:
    def test_every_rule_that_changes_behaviour_is_in_the_signature(self):
        base = config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5))
        assert base != config_signature(AppleTraderConfig(prob_threshold=0.3, trail_pct=0.5))
        assert base != config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.8))
        assert base == config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5))

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
