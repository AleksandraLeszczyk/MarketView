"""The momentum/regime/feature pipeline mirrored from mshift.

These pin the definitions the saved model was trained with -- a drift here is
the classic way a loaded model starts scoring nonsense while still "working".
The numbers themselves were checked bar-for-bar against mshift's own output on
the notebooks' cached AAPL history; what is pinned here is the behaviour that
check would catch changing.
"""

import numpy as np
import pandas as pd
import pytest

from agent_stonks import persistence_model as pm

PARAMS = pm.MOMENTUM_DEFAULTS


def _bars(day: str, closes, start: str = "09:30", seed: int = 0) -> list[dict]:
    """Minute bars in the {"t","o","h","l","c","v"} shape the streams deliver.

    Each bar gets a real body and wick and its own volume: several features
    divide by the bar range or z-score the volume, and a synthetic tape of
    identical flat bars would leave them undefined for reasons the pipeline
    itself is not responsible for.
    """
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([closes[:1], closes[:-1]])
    wick = closes * 3e-4
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    volumes = 1000.0 + rng.integers(0, 500, size=len(closes))
    idx = pd.date_range(
        f"{day} {start}", periods=len(closes), freq="1min", tz="America/New_York"
    ).tz_convert("UTC")
    return [
        {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": o, "h": h, "l": lo, "c": c, "v": v}
        for ts, o, h, lo, c, v in zip(idx, opens, highs, lows, closes, volumes)
    ]


def _flat_then_trend(n_flat: int, n_trend: int, step: float, start: float = 100.0, seed: int = 0):
    """Closes that oscillate with no net drift, then trend at `step` per minute.

    The flat half alternates rather than wandering: a random walk normalised by
    its own realised volatility crosses the +-0.90 entry band by chance often
    enough to make "did a regime change happen" a coin flip in a test.
    """
    rng = np.random.default_rng(seed)
    flat = np.resize([1e-4, -1e-4], n_flat) if n_flat else np.zeros(0)
    trend = np.full(n_trend, step) + rng.normal(0, 5e-5, size=n_trend)
    return start * np.exp(np.cumsum(np.concatenate([flat, trend])))


def _session(day: str = "2026-07-21", n_flat: int = 90, n_trend: int = 90, step: float = 3e-4):
    return pm._frame_from_bars(_bars(day, _flat_then_trend(n_flat, n_trend, step)))


class TestFrame:
    def test_keeps_only_the_regular_session_and_stamps_the_bookkeeping(self):
        frame = pm._frame_from_bars(_bars("2026-07-21", [100.0] * 180, start="08:00"))
        assert frame.index.min().strftime("%H:%M") == "09:30"
        assert frame.index.max().strftime("%H:%M") == "10:59"
        assert list(frame.columns) == [
            "open", "high", "low", "close", "volume",
            "session", "bar_of_day", "minutes_from_open",
        ]
        assert frame["bar_of_day"].iloc[0] == 0
        assert frame["minutes_from_open"].iloc[0] == 0.0
        assert frame["minutes_from_open"].iloc[-1] == 89.0

    def test_the_live_frame_is_todays_session_only(self, monkeypatch):
        """Nothing in the pipeline crosses the overnight gap, so yesterday's
        bars would only slow it down -- and they are dropped."""
        import threading
        from types import SimpleNamespace

        monkeypatch.setattr(pm.clock, "now", lambda: pd.Timestamp("2026-07-21 15:00", tz="UTC"))
        sym_state = SimpleNamespace(
            symbol="AAPL",
            lock=threading.Lock(),
            bars=_bars("2026-07-20", [100.0] * 390) + _bars("2026-07-21", [111.0] * 30),
        )
        frame = pm.minute_frame(sym_state)
        assert len(frame) == 30
        assert frame["close"].eq(111.0).all()
        assert frame["bar_of_day"].iloc[-1] == 29


class TestMomentum:
    def test_momentum_is_drift_in_units_of_its_own_random_walk_scale(self):
        frame = pm.compute_momentum(_session(), PARAMS)
        # The trailing return needs `horizon` bars and sigma needs 10, so the
        # first bars carry no momentum rather than a fabricated one.
        assert frame["mom_raw"].iloc[:10].isna().all()
        quiet, trending = frame["mom"].iloc[80], frame["mom"].iloc[-1]
        assert abs(quiet) < 0.9  # noise alone stays inside the entry band
        assert trending > 3.0  # a steady 4 bps/min drift is many sigmas of it

    def test_momentum_does_not_reach_across_the_overnight_gap(self):
        frame = pm._frame_from_bars(
            _bars("2026-07-20", [100.0] * 60) + _bars("2026-07-21", [130.0] * 60)
        )
        scored = pm.compute_momentum(frame, PARAMS)
        day2 = scored[scored["session"] == scored["session"].iloc[-1]]
        # The 30% overnight jump is invisible: day 2 opens with no return at all
        # and its own trailing windows have to fill from scratch.
        assert pd.isna(day2["ret_1"].iloc[0])
        assert day2["mom_raw"].iloc[:10].isna().all()


class TestRegimes:
    def test_the_schmitt_trigger_enters_high_and_leaves_low(self):
        """Entering a directional regime takes |mom| > 0.90; leaving it only
        needs a fall back below 0.40. The gap is what stops a score hovering on
        the line from emitting a burst of fake changes."""
        mom = np.array([0.0, 0.5, 0.89, 0.91, 0.60, 0.41, 0.39, -0.5, -0.95, -0.5, -0.39])
        starts = np.zeros(len(mom), dtype=bool)
        regimes = pm._hysteresis_regimes(mom, starts, enter=0.90, exit_=0.40)
        assert list(regimes) == [0, 0, 0, 1, 1, 1, 0, 0, -1, -1, 0]

    def test_each_session_starts_flat(self):
        mom = np.array([2.0, 2.0, 2.0, 2.0])
        starts = np.array([False, False, True, False])
        assert list(pm._hysteresis_regimes(mom, starts, 0.90, 0.40)) == [1, 1, 1, 1]
        # ...and a session that opens below the entry band starts balanced even
        # though the previous session ended in a regime.
        mom = np.array([2.0, 2.0, 0.5, 0.5])
        assert list(pm._hysteresis_regimes(mom, starts, 0.90, 0.40)) == [1, 1, 0, 0]

    def test_a_trend_produces_one_change_into_positive(self):
        scored = pm.add_momentum_regimes(_session(), PARAMS)
        changes = scored[scored["regime_change"]]
        to_positive = changes[changes["regime"] == 1]
        assert len(to_positive) == 1
        # It is exactly the bar where the regime column first turns positive.
        assert to_positive.index[0] == scored.index[scored["regime"] == 1][0]
        assert to_positive["prev_regime"].iloc[0] == 0

    def test_pre_dwell_is_how_long_the_old_regime_had_held(self):
        scored = pm.add_momentum_regimes(_session(), PARAMS)
        change = scored[scored["regime_change"]].iloc[0]
        pos = scored.index.get_loc(scored[scored["regime_change"]].index[0])
        assert change["pre_dwell"] == pos  # the balanced run started at bar 0
        assert scored["pre_dwell"][~scored["regime_change"]].isna().all()


class TestFeatures:
    def test_feature_table_matches_the_saved_contract(self):
        scored = pm.compute_features(pm.add_momentum_regimes(_session(), PARAMS))
        assert len(pm.FEATURE_COLUMNS) == 25
        assert set(pm.FEATURE_COLUMNS) <= set(scored.columns)
        # A warmed-up bar has every feature: nothing is left for an imputer to
        # invent, which is why an unwarmed window is dropped instead.
        assert scored[pm.FEATURE_COLUMNS].iloc[-1].notna().all()
        assert scored[pm.FEATURE_COLUMNS].iloc[0].isna().any()

    def test_the_dwell_feature_carries_the_observable_half_of_persistence(self):
        """On a change bar `f_bars_in_regime` is always 1, so `f_prev_dwell` is
        the only channel that can see how long the old regime had held -- the
        model's strongest single feature."""
        scored = pm.compute_features(pm.add_momentum_regimes(_session(), PARAMS))
        change = scored[scored["regime_change"]].iloc[0]
        assert change["f_bars_in_regime"] == pytest.approx(np.log1p(1))
        assert change["f_prev_dwell"] == pytest.approx(np.log1p(change["pre_dwell"]))

    def test_features_are_finite_or_absent_never_infinite(self):
        """A flat bar divides by a zero range; training replaced those with NaN
        (and dropped the sample) rather than letting an inf reach the model."""
        scored = pm.compute_features(pm.add_momentum_regimes(_session(), PARAMS))
        assert not np.isinf(scored[pm.FEATURE_COLUMNS].to_numpy(dtype=float)).any()


class TestSequences:
    def _features(self):
        return pm.compute_features(pm.add_momentum_regimes(_session(), PARAMS))

    def test_the_block_ends_at_the_last_bar(self):
        scored = self._features()
        block = pm.build_sequence(scored, pm.FEATURE_COLUMNS, 20)
        assert block.shape == (20, 25)
        assert block[-1] == pytest.approx(
            scored[pm.FEATURE_COLUMNS].iloc[-1].to_numpy(dtype=np.float32), rel=1e-6
        )

    def test_an_unwarmed_window_is_refused_rather_than_imputed(self):
        scored = self._features()
        assert pm.build_sequence(scored.iloc[:25], pm.FEATURE_COLUMNS, 20) is None
        assert pm.build_sequence(scored.iloc[:10], pm.FEATURE_COLUMNS, 20) is None

    def test_a_window_that_would_cross_a_session_is_refused(self):
        frame = pm._frame_from_bars(
            _bars("2026-07-20", _flat_then_trend(90, 90, 4e-4))
            + _bars("2026-07-21", _flat_then_trend(0, 5, 4e-4, start=120.0))
        )
        scored = pm.compute_features(pm.add_momentum_regimes(frame, PARAMS))
        assert pm.build_sequence(scored, pm.FEATURE_COLUMNS, 20) is None


class StubBundle(dict):
    """A bundle whose estimator records what it was asked, so `read_latest`'s
    contract can be pinned without shipping the fitted artifact."""

    def __init__(self, proba: float = 0.8):
        calls = self.calls = []

        class Pipeline:
            def predict_proba(self, X):
                calls.append(np.asarray(X))
                return np.column_stack([np.full(len(X), 1 - proba), np.full(len(X), proba)])

        super().__init__(
            pipeline=Pipeline(),
            feature_columns=list(pm.FEATURE_COLUMNS),
            seq_len=20,
            threshold=0.07,
            settings={"momentum": dict(PARAMS)},
        )


class TestReadLatest:
    def _upto(self, scored, ts):
        return scored.loc[:ts]

    def test_the_model_is_asked_only_about_a_change_into_positive(self):
        bundle = StubBundle()
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]

        read = pm.read_latest(bundle, frame.loc[:change_ts])
        assert read["to_positive"] and read["regime_change"]
        assert read["proba"] == pytest.approx(0.8)
        assert bundle.calls[-1].shape == (1, 20, 25)

        # The very next bar is the same positive regime, not a change: no call.
        after = frame.index[frame.index.get_loc(change_ts) + 1]
        read = pm.read_latest(bundle, frame.loc[:after])
        assert read["regime"] == 1 and not read["regime_change"]
        assert read["proba"] is None
        assert len(bundle.calls) == 1

    def test_an_opening_change_is_flagged_warming_up_but_gated_on_the_window(self):
        """The simulator (`mshift.backtest._signal_sequences`) gates the model
        on one thing only: a complete, all-finite 20-bar window. `warming_up`
        mirrors the cut `build_events` makes on the *training* set and is
        reported, not enforced -- it would never bind anyway, since the window
        rejects everything earlier."""
        bundle = StubBundle()
        # A drift starting at bar 16 puts the change well inside warmup_bars.
        frame = pm._frame_from_bars(_bars("2026-07-21", _flat_then_trend(16, 40, 1e-3)))
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]
        assert scored["bar_of_day"].loc[change_ts] < PARAMS["warmup_bars"]

        read = pm.read_latest(bundle, frame.loc[:change_ts])
        assert read["warming_up"] and read["to_positive"]
        assert read["proba"] is None
        assert bundle.calls == []

    def test_the_window_is_not_complete_until_long_after_the_warm_up(self):
        """Why the warm-up flag is never the binding constraint: the slowest
        trailing indicator in the feature set leaves NaNs deep into the
        session, so no bar before ~39 can carry 20 finite feature rows."""
        frame = _session()
        feat = pm.compute_features(pm.add_momentum_regimes(frame, PARAMS))
        first = next(
            i
            for i in range(len(feat))
            if pm.build_sequence(feat.iloc[: i + 1], pm.FEATURE_COLUMNS, 20) is not None
        )
        assert first > PARAMS["warmup_bars"]

    def test_a_frame_too_short_to_score_returns_nothing(self):
        frame = pm._frame_from_bars(_bars("2026-07-21", [100.0] * 5))
        assert pm.read_latest(StubBundle(), frame) is None

    def test_the_read_describes_the_bar_it_scored(self):
        bundle = StubBundle()
        frame = _session()
        read = pm.read_latest(bundle, frame)
        assert read["ts"] == frame.index[-1]
        assert read["price"] == pytest.approx(frame["close"].iloc[-1])
        assert read["high"] == pytest.approx(frame["high"].iloc[-1])
        assert read["bars_today"] == len(frame)
        assert read["regime"] == 1 and not read["warming_up"]

    def test_a_classifier_is_never_asked_the_anticipation_question(self):
        """It was fitted on change bars, so a bar that is not one is not a
        harder question for it -- it is a different one. `turn_proba` stays
        None on every bar rather than carrying an off-distribution number."""
        bundle = StubBundle()
        assert not pm.anticipates(bundle)
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        balanced = scored[scored["regime"] != 1].index[-1]

        read = pm.read_latest(bundle, frame.loc[:balanced])
        assert read["regime"] != 1
        assert read["turn_proba"] is None


class TurnStubBundle(StubBundle):
    """A bundle that can also forecast, i.e. one shaped like N-BEATS'."""

    def __init__(self, proba: float = 0.8, turn: float = 0.3):
        super().__init__(proba)
        self.turn_calls: list = []

        def score_turn(X):
            self.turn_calls.append(np.asarray(X))
            return np.full(len(X), turn)

        self["score_turn"] = score_turn


class TestAnticipationRead:
    """Which of the two questions a bar poses, and that it never poses both."""

    def test_a_bar_the_regime_has_not_turned_on_gets_the_turn_question(self):
        bundle = TurnStubBundle()
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        balanced = scored[scored["regime"] != 1].index[-1]

        read = pm.read_latest(bundle, frame.loc[:balanced])
        assert read["regime"] != 1
        assert read["turn_proba"] == pytest.approx(0.3)
        assert read["proba"] is None
        assert bundle.turn_calls[-1].shape == (1, 20, 25)
        assert bundle.calls == []

    def test_the_change_bar_still_gets_the_persistence_question(self):
        bundle = TurnStubBundle()
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]

        read = pm.read_latest(bundle, frame.loc[:change_ts])
        assert read["to_positive"]
        assert read["proba"] == pytest.approx(0.8)
        assert read["turn_proba"] is None
        assert bundle.turn_calls == []

    def test_sitting_inside_the_positive_regime_poses_neither(self):
        bundle = TurnStubBundle()
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]
        after = frame.index[frame.index.get_loc(change_ts) + 1]

        read = pm.read_latest(bundle, frame.loc[:after])
        assert read["regime"] == 1 and not read["regime_change"]
        assert read["proba"] is None and read["turn_proba"] is None

    def test_the_dwell_the_gate_reads_is_the_current_regimes_own_length(self):
        """On the bar before a change this is what `pre_dwell` will be on the
        change bar itself -- the same observable gate, one bar early."""
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]
        before = frame.index[frame.index.get_loc(change_ts) - 1]

        read = pm.read_latest(TurnStubBundle(), frame.loc[:before])
        assert read["bars_in_regime"] == int(scored["pre_dwell"].loc[change_ts])

    def test_an_unwarmed_window_leaves_the_turn_unanswered(self):
        bundle = TurnStubBundle()
        frame = pm._frame_from_bars(_bars("2026-07-21", _flat_then_trend(16, 40, 1e-3)))
        read = pm.read_latest(bundle, frame.iloc[:20])
        assert read["turn_proba"] is None
        assert bundle.turn_calls == []


class ReversalStubBundle(TurnStubBundle):
    """A forecasting bundle that answers the exit question too."""

    def __init__(self, proba: float = 0.8, turn: float = 0.3, reversal: float = 0.6):
        super().__init__(proba, turn)
        self.reversal_calls: list = []

        def score_reversal(X):
            self.reversal_calls.append(np.asarray(X))
            return np.full(len(X), reversal)

        self["score_reversal"] = score_reversal


class TestReversalRead:
    """The third question: asked on a positive bar, and only while holding."""

    def _positive_bar(self):
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]
        after = frame.index[frame.index.get_loc(change_ts) + 1]
        return frame.loc[:after]

    def test_a_held_positive_bar_gets_the_reversal_question(self):
        bundle = ReversalStubBundle()
        read = pm.read_latest(bundle, self._positive_bar(), holding=True)
        assert read["regime"] == 1
        assert read["reversal_proba"] == pytest.approx(0.6)
        assert bundle.reversal_calls[-1].shape == (1, 20, 25)

    def test_the_same_bar_is_not_asked_with_the_book_flat(self):
        """Every ask costs a full forecast and nothing acts on the answer when
        there is no position, so the default must not pay for it."""
        bundle = ReversalStubBundle()
        read = pm.read_latest(bundle, self._positive_bar())
        assert read["regime"] == 1
        assert read["reversal_proba"] is None
        assert bundle.reversal_calls == []

    def test_a_non_positive_bar_is_never_asked(self):
        """The question is about leaving a positive regime; a bar that is not
        in one has not posed it, holding or not."""
        bundle = ReversalStubBundle()
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        balanced = scored[scored["regime"] != 1].index[-1]

        read = pm.read_latest(bundle, frame.loc[:balanced], holding=True)
        assert read["regime"] != 1
        assert read["reversal_proba"] is None
        assert bundle.reversal_calls == []

    def test_the_change_bar_poses_the_entry_and_exit_questions_at_once(self):
        """Entering on the anticipated turn means holding into the bar the
        change prints on -- a bar that is both a to-positive change and a
        position to manage. The two questions are about different things and
        both get asked."""
        bundle = ReversalStubBundle()
        frame = _session()
        scored = pm.add_momentum_regimes(frame, PARAMS)
        change_ts = scored[scored["regime_change"] & (scored["regime"] == 1)].index[0]

        read = pm.read_latest(bundle, frame.loc[:change_ts], holding=True)
        assert read["to_positive"]
        assert read["proba"] == pytest.approx(0.8)
        assert read["reversal_proba"] == pytest.approx(0.6)
        assert read["turn_proba"] is None

    def test_a_classifier_is_never_asked_it(self):
        bundle = StubBundle()
        assert not pm.forecasts_reversal(bundle)
        read = pm.read_latest(bundle, self._positive_bar(), holding=True)
        assert read["reversal_proba"] is None

    def test_asking_a_model_that_cannot_answer_raises(self):
        with pytest.raises(ValueError, match="cannot forecast the breakdown"):
            pm.predict_reversal_proba(StubBundle(), np.zeros((1, 20, 25), dtype=np.float32))


class TestBundle:
    def test_the_path_is_overridable_by_environment(self, monkeypatch):
        assert pm.model_path().name == "apple_momentum_2.joblib"
        monkeypatch.setenv(pm.MODEL_PATH_ENV, "/tmp/elsewhere.joblib")
        assert str(pm.model_path()) == "/tmp/elsewhere.joblib"

    def test_a_missing_file_degrades_to_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv(pm.MODEL_PATH_ENV, str(tmp_path / "nope.joblib"))
        pm.reset_bundle_cache()
        try:
            assert pm.load_bundle() is None
        finally:
            pm.reset_bundle_cache()

    def test_the_pickled_summariser_resolves_to_the_mirror(self):
        """The bundle was pickled from the notebooks' own package, so loading it
        needs something importable at `mshift.model.SequenceSummarizer`. The
        stub module stands in for a package that would otherwise drag matplotlib
        and yfinance into the app."""
        import sys

        pm._register_unpickle_alias()
        assert sys.modules["mshift.model"].SequenceSummarizer is not None

    def test_the_summariser_compresses_a_sequence_into_shape_statistics(self):
        X = np.arange(2 * 20 * 25, dtype=float).reshape(2, 20, 25)
        out = pm.SequenceSummarizer().fit(X).transform(X)
        assert out.shape == (2, 25 * len(pm.SUMMARY_STATS))
        # "last" is the change bar itself, and it comes first.
        assert out[:, :25] == pytest.approx(X[:, -1, :])
        # Each channel here climbs by 25 per bar, so slope and delta say so.
        slope = out[:, 25 * 5 : 25 * 6]
        delta = out[:, 25 * 6 : 25 * 7]
        assert slope == pytest.approx(np.full((2, 25), 25.0))
        assert delta == pytest.approx(np.full((2, 25), 25.0 * 19))

    def test_defaults_stand_in_for_a_bundle_without_settings(self):
        assert pm.momentum_params({}) == pm.MOMENTUM_DEFAULTS
        assert pm.momentum_params({"settings": {"momentum": {"warmup_bars": 5}}})["warmup_bars"] == 5
        assert pm.model_threshold({}) == 0.5
        assert pm.model_threshold({"threshold": 0.07}) == pytest.approx(0.07)
