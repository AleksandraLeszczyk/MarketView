"""Apple Trader: the rule-based loop over the momentum-change model.

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
from agent_stonks import momentum_model
from agent_stonks.apple_trader import (
    SIGNAL_OUT_OF_POSITIVE,
    SIGNAL_TO_POSITIVE,
    TICKER,
    AppleTrader,
    AppleTraderConfig,
    classify,
)
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

# 10:30 ET on a Tuesday: mid-session, well clear of both the open and the close.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)

BUNDLE = {"estimator": None, "feature_cols": [], "pipeline_params": {"persist": 15, "n_prev_changes": 3}}


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

    def __init__(self, monkeypatch, *, bars_today: int = 200):
        self.bars_today = bars_today
        self.minute = 0
        self.next_read: dict = {}
        monkeypatch.setattr(at.momentum_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]}))
        monkeypatch.setattr(at.momentum_model, "read_latest", lambda *a, **k: self.next_read)

    def set(
        self,
        *,
        pred: "float | None" = 0.0,
        regime: int = 0,
        mom: float = 0.0,
        theta: float = 0.5,
        price: float = 100.0,
        advance: bool = True,
        bars_today: "int | None" = None,
    ) -> dict:
        """Stage the next bar. `advance=False` replays the SAME timestamp, the
        way a cycle that runs before a new bar has closed would see it."""
        if advance:
            self.minute += 1
        projected = mom + pred if pred is not None else None
        self.next_read = {
            "ts": pd.Timestamp("2026-07-21 10:30", tz="America/New_York") + pd.Timedelta(minutes=self.minute),
            "price": price,
            "mom": mom,
            "theta": theta,
            "regime": regime,
            "pred": pred,
            "projected_mom": projected,
            "projected_regime": momentum_model._regime_of(projected, theta),
            "bars_today": self.bars_today if bars_today is None else bars_today,
        }
        return self.next_read


def _run(trader: AppleTrader, state: AppState, tracker: DecisionTracker, n: int = 1) -> str:
    outcome = ""
    for _ in range(n):
        outcome = trader.run_cycle(BUNDLE, state, tracker)
    return outcome


class TestClassify:
    """A call needs a big enough prediction AND an actual change of regime."""

    @pytest.mark.parametrize("regime", [-1, 0])
    def test_entry_from_negative_or_balanced(self, regime):
        read = {"pred": 1.0, "regime": regime, "projected_regime": 1}
        assert classify(read, threshold=0.75) == SIGNAL_TO_POSITIVE

    @pytest.mark.parametrize("projected", [0, -1])
    def test_exit_out_of_positive(self, projected):
        read = {"pred": -1.0, "regime": 1, "projected_regime": projected}
        assert classify(read, threshold=0.75) == SIGNAL_OUT_OF_POSITIVE

    def test_a_change_smaller_than_the_threshold_is_not_a_call(self):
        read = {"pred": 0.5, "regime": 0, "projected_regime": 1}
        assert classify(read, threshold=0.75) is None
        assert classify(read, threshold=0.4) == SIGNAL_TO_POSITIVE

    def test_more_of_the_same_regime_is_not_a_change(self):
        """Already positive and getting stronger is not a transition."""
        assert classify({"pred": 3.0, "regime": 1, "projected_regime": 1}, 0.75) is None

    def test_a_move_that_stays_inside_the_band_is_not_a_call(self):
        assert classify({"pred": 1.0, "regime": 0, "projected_regime": 0}, 0.75) is None

    def test_no_prediction_is_no_call(self):
        assert classify({"pred": None, "regime": 0, "projected_regime": None}, 0.75) is None


class TestEntry:
    def test_buys_only_after_the_call_repeats(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=3))

        for _ in range(2):
            reads.set(pred=1.0, regime=0, mom=0.2)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
            assert tracker.position_for(TICKER) == 0

        reads.set(pred=1.0, regime=0, mom=0.2)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_an_interrupted_run_restarts_the_count(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=3))

        reads.set(pred=1.0, regime=0, mom=0.2)
        trader.run_cycle(BUNDLE, state, tracker)
        reads.set(pred=0.1, regime=0, mom=0.2)  # quiet bar breaks the run
        trader.run_cycle(BUNDLE, state, tracker)
        for _ in range(2):
            reads.set(pred=1.0, regime=0, mom=0.2)
            trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == 0

        reads.set(pred=1.0, regime=0, mom=0.2)
        trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) > 0

    def test_re_reading_the_same_bar_is_not_fresh_confirmation(self, state, market_open, monkeypatch):
        """Two cycles over one closed bar are one observation, not two."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=2))

        reads.set(pred=1.0, regime=0, mom=0.2)
        trader.run_cycle(BUNDLE, state, tracker)
        reads.set(pred=1.0, regime=0, mom=0.2, advance=False)
        trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == 0

        reads.set(pred=1.0, regime=0, mom=0.2)
        trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) > 0

    def test_does_not_trade_during_the_warm_up_hour(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=2, min_bars_today=60))

        for _ in range(5):
            reads.set(pred=3.0, regime=0, mom=0.2, bars_today=30)
            assert trader.run_cycle(BUNDLE, state, tracker) == "warming_up"
        assert tracker.position_for(TICKER) == 0
        # The warm-up bars build no confirmation either, so a full run is still
        # needed once the windows are filled.
        reads.set(pred=3.0, regime=0, mom=0.2)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(pred=3.0, regime=0, mom=0.2)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"

    def test_position_size_follows_the_configured_share_of_cash(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0), trade_cost=0.0)
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=1, position_pct=50.0))

        reads.set(pred=1.0, regime=0, mom=0.2, price=100.0)
        trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == pytest.approx(50.0)

    def test_no_second_entry_while_already_long(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=1))

        reads.set(pred=1.0, regime=0, mom=0.2)
        trader.run_cycle(BUNDLE, state, tracker)
        size = tracker.position_for(TICKER)
        for _ in range(3):
            reads.set(pred=1.0, regime=0, mom=0.2)
            trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == size


def _enter(state, tracker, reads, config) -> AppleTrader:
    """Take a confirmed long at $100 so the exit rules have something to act on."""
    trader = AppleTrader(config)
    for _ in range(config.confirm_minutes):
        reads.set(pred=1.0, regime=0, mom=0.2, price=100.0)
        trader.run_cycle(BUNDLE, state, tracker)
    assert tracker.position_for(TICKER) > 0
    return trader


class TestExits:
    def test_sells_when_the_model_calls_the_way_out(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        config = AppleTraderConfig(threshold=0.75, confirm_minutes=2, validation_minutes=15)
        trader = _enter(state, tracker, reads, config)

        # The regime shows up, so the entry is not a false call.
        reads.set(pred=0.0, regime=1, mom=1.0)
        trader.run_cycle(BUNDLE, state, tracker)

        reads.set(pred=-1.0, regime=1, mom=1.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"  # one call is not enough
        reads.set(pred=-1.0, regime=1, mom=1.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0

    def test_cuts_a_call_whose_regime_never_showed_up(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        config = AppleTraderConfig(threshold=0.75, confirm_minutes=2, validation_minutes=5)
        trader = _enter(state, tracker, reads, config)

        for _ in range(config.validation_minutes - 1):
            reads.set(pred=0.0, regime=0, mom=0.2)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"

        reads.set(pred=0.0, regime=0, mom=0.2)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "False call" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_regime_that_does_show_up_survives_the_validation_window(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        config = AppleTraderConfig(threshold=0.75, confirm_minutes=2, validation_minutes=5)
        trader = _enter(state, tracker, reads, config)

        reads.set(pred=0.0, regime=1, mom=1.0)  # the predicted regime arrives
        trader.run_cycle(BUNDLE, state, tracker)
        for _ in range(config.validation_minutes + 3):
            reads.set(pred=0.0, regime=0, mom=0.2)
            trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) > 0

    def test_sustained_negative_momentum_closes_the_position(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        config = AppleTraderConfig(threshold=0.75, confirm_minutes=2, validation_minutes=50)
        trader = _enter(state, tracker, reads, config)

        reads.set(pred=0.0, regime=1, mom=1.0)
        trader.run_cycle(BUNDLE, state, tracker)
        reads.set(pred=0.0, regime=-1, mom=-1.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(pred=0.0, regime=-1, mom=-1.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "negative for 2 straight bars" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_loss_cap_fires_before_anything_else(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch)
        config = AppleTraderConfig(threshold=0.75, confirm_minutes=2, stop_loss_pct=1.0)
        trader = _enter(state, tracker, reads, config)

        broker.price = 98.5
        reads.set(pred=0.0, regime=1, mom=1.0, price=98.5)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "Stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_flattens_before_the_close(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        config = AppleTraderConfig(threshold=0.75, confirm_minutes=2, validation_minutes=50)
        trader = _enter(state, tracker, reads, config)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.set(pred=0.0, regime=1, mom=1.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning

    def test_adopts_a_position_it_did_not_open(self, state, market_open, monkeypatch):
        """Restarted onto a ledger that already holds shares: the checks still
        need something to measure against, so the position is adopted."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tracker.record_trade(TICKER, "buy", 10, "seeded", "k", "s")
        reads = Reads(monkeypatch)
        trader = AppleTrader(AppleTraderConfig(threshold=0.75, confirm_minutes=2, stop_loss_pct=1.0))

        reads.set(pred=0.0, regime=1, mom=1.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        broker.price = 98.0
        reads.set(pred=0.0, regime=1, mom=1.0, price=98.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"


class TestGuards:
    def test_does_nothing_when_the_market_is_closed(self, state, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 2, 0, tzinfo=timezone.utc))  # 22:00 ET Monday
        try:
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
            reads = Reads(monkeypatch)
            reads.set(pred=3.0, regime=0, mom=0.2)
            trader = AppleTrader(AppleTraderConfig(confirm_minutes=1))
            assert trader.run_cycle(BUNDLE, state, tracker) == "closed"
            assert tracker.position_for(TICKER) == 0
        finally:
            clock.clear()

    def test_reports_when_the_ticker_is_not_streamed(self, market_open, monkeypatch):
        state = AppState()
        state.set_symbols(["MSFT"])
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        Reads(monkeypatch)
        trader = AppleTrader()
        assert trader.run_cycle(BUNDLE, state, tracker) == "no_data"
        assert any(e["type"] == "error" for e in state.agent_log)

    def test_reports_when_there_is_not_enough_history(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.momentum_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]}))
        monkeypatch.setattr(at.momentum_model, "read_latest", lambda *a, **k: None)
        assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "no_data"

    def test_missing_model_stops_the_loop_instead_of_trading(self, state, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.momentum_model, "load_bundle", lambda ticker: None)
        stop_event = threading.Event()
        at._apple_trader_loop(state, tracker, AppleTraderConfig(), 60, stop_event)
        assert state.agent_running is False
        assert any("cannot run without it" in e.get("text", "") for e in state.agent_log)


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
