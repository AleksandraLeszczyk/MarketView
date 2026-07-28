"""The momentum/regime/feature pipeline mirrored from momlib.

These pin the definitions the saved model was trained with -- a drift here is
the classic way a loaded model starts scoring nonsense while still "working".
"""

import numpy as np
import pandas as pd
import pytest

from agent_stonks import momentum_model as mm

PARAMS = {"window": 15, "smooth_halflife": 8.0, "c": 0.4, "persist": 15, "n_prev_changes": 3}


def _session(day: str, closes, volume: float = 1000.0) -> pd.DataFrame:
    """A one-day minute frame in the shape momlib works on."""
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="1min", tz="America/New_York")
    closes = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(len(closes), volume),
        },
        index=idx,
    )
    df["day"] = df.index.date
    return df


def _regime_frame(regimes, day: str = "2026-07-20") -> pd.DataFrame:
    """A frame carrying a hand-built regime sequence, for testing the
    persistence rule without going through momentum at all."""
    frame = _session(day, np.full(len(regimes), 100.0))
    frame["regime"] = np.asarray(regimes, dtype=float)
    frame["mom"] = frame["regime"]
    frame["theta"] = 0.5
    return frame


def _flat_then_trend(n_flat: int, n_trend: int, step_bps: float, start: float = 100.0):
    """Closes that sit still, then drift at a constant `step_bps` per minute."""
    rets = np.concatenate([np.zeros(n_flat), np.full(n_trend, step_bps / mm.BPS)])
    # A dead-flat day has zero volatility, so theta would be 0 and every bar
    # would classify as a regime; a touch of noise keeps it realistic.
    rng = np.random.default_rng(0)
    rets = rets + rng.normal(0, 1e-5, size=len(rets))
    return start * np.exp(np.cumsum(rets))


class TestMomentumAndRegimes:
    def test_momentum_is_the_trailing_mean_log_return_in_bps(self):
        closes = 100 * np.exp(np.cumsum(np.full(60, 2.0 / mm.BPS)))
        out = mm.add_momentum(_session("2026-07-20", closes), window=15, smooth_halflife=8.0)
        # Constant 2 bps/min drift: the trailing mean is 2 bps/min once filled.
        assert out["mom_raw"].iloc[:14].isna().all()
        assert out["mom_raw"].iloc[-1] == pytest.approx(2.0, abs=1e-6)
        # `mom` is the EMA of that, so it lags but converges from below.
        assert 0 < out["mom"].iloc[-1] < out["mom_raw"].iloc[-1]

    def test_theta_comes_from_the_previous_day(self):
        rng = np.random.default_rng(1)
        quiet = 100 * np.exp(np.cumsum(rng.normal(0, 1e-5, 390)))
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 390)))
        df = pd.concat([_session("2026-07-20", quiet), _session("2026-07-21", wild)])
        out = mm.add_regimes(mm.add_momentum(df, window=15), window=15, c=0.4)

        sigma_day1 = out[out["day"] == out["day"].iloc[0]]["log_ret"].std()
        day2 = out[out["day"] == out["day"].iloc[-1]]
        assert day2["theta"].iloc[0] == pytest.approx(0.4 * sigma_day1 / np.sqrt(15))
        # Day 2 is 10x more volatile, but its threshold is still yesterday's:
        # known at the open, not fitted to the day it judges.
        assert day2["theta"].nunique() == 1

    def test_regime_is_momentum_against_plus_minus_theta(self):
        df = _session("2026-07-20", _flat_then_trend(120, 200, 8.0))
        out = mm.add_regimes(mm.add_momentum(df, window=15), window=15, c=0.4)
        last = out.iloc[-1]
        assert last["mom"] > last["theta"]
        assert last["regime"] == 1
        assert set(out["regime"].dropna().unique()) <= {-1.0, 0.0, 1.0}
        assert out["regime"].isna().iloc[0]  # momentum not computable yet


class TestPersistentChanges:
    def test_finds_the_balanced_to_positive_flip(self):
        df = _session("2026-07-20", _flat_then_trend(150, 150, 8.0))
        scored, events = mm.prepare(df, PARAMS)
        assert len(events) == 1
        event = events.iloc[0]
        assert event["transition"] == "balanced -> positive"
        assert event["mom_delta"] > 0
        # The flip is where the regime column first turns positive.
        first_positive = scored.index[scored["regime"] == 1][0]
        assert events.index[0] == first_positive

    def test_last_persist_bars_can_never_produce_an_event(self):
        """A change is only confirmable `persist` bars later, so the tail of a
        live frame is always event-free -- the delay a live loop has to accept."""
        df = _session("2026-07-20", _flat_then_trend(150, 150, 8.0))
        scored, events = mm.prepare(df, PARAMS)
        assert (events.index < scored.index[-PARAMS["persist"]]).all()

    def test_a_flip_that_does_not_hold_is_not_a_change(self):
        """The rule itself, on a hand-built regime sequence: a flip counts only
        when the old regime held `persist` bars before it and the new one holds
        `persist` bars after."""
        persist = PARAMS["persist"]
        regimes = [0.0] * 40 + [1.0] * (persist - 1) + [0.0] * 40  # too short to persist
        assert len(mm.find_persistent_changes(_regime_frame(regimes), persist=persist)) == 0

        # One bar longer and it qualifies -- both in and back out of the regime.
        held = [0.0] * 40 + [1.0] * persist + [0.0] * 40
        events = mm.find_persistent_changes(_regime_frame(held), persist=persist)
        assert list(events["transition"]) == ["balanced -> positive", "positive -> balanced"]
        assert list(events["bar_of_day"]) == [40, 40 + persist]

    def test_a_flip_across_the_overnight_gap_is_not_a_change(self):
        persist = PARAMS["persist"]
        frame = pd.concat(
            [
                _regime_frame([0.0] * 60, day="2026-07-20"),
                _regime_frame([1.0] * 60, day="2026-07-21"),
            ]
        )
        assert len(mm.find_persistent_changes(frame, persist=persist)) == 0


class TestFeatures:
    def test_feature_table_matches_the_saved_contract(self):
        df = pd.concat(
            [
                _session("2026-07-20", _flat_then_trend(200, 190, 6.0)),
                _session("2026-07-21", _flat_then_trend(150, 240, 8.0, start=101.0)),
            ]
        )
        scored, events = mm.prepare(df, PARAMS)
        X = mm.build_live_features(scored, events, n_prev_changes=3)

        assert list(X.columns) == mm.FEATURE_COLS
        assert len(X) == len(scored)
        last = X.iloc[-1]
        # Day 2 is fully warmed up: every trailing window is filled.
        assert last[["mom_5", "mom_15", "mom_60", "mom_slope", "mom_smooth_z"]].notna().all()
        assert last["minute_of_day"] == 389
        assert last["dist_prev_close"] == pytest.approx(
            np.log(scored["Close"].iloc[-1] / scored[scored["day"] == scored["day"].iloc[0]]["Close"].iloc[-1])
            * mm.BPS
        )
        # regime_before is the previous minute's regime, within the day.
        assert last["regime_before"] == scored["regime"].iloc[-2]

    def test_event_history_features_only_see_changes_already_past(self):
        df = _session("2026-07-20", _flat_then_trend(150, 240, 8.0))
        scored, events = mm.prepare(df, PARAMS)
        assert len(events) == 1
        X = mm.build_live_features(scored, events, n_prev_changes=3)

        change_pos = scored.index.get_loc(events.index[0])
        before, after = X.iloc[change_pos - 1], X.iloc[-1]
        # Nothing has happened yet on the bar before the change...
        assert np.isnan(before["dist_change_1"])
        assert np.isnan(before["bars_since_change"])
        # ...and afterwards the distance is measured back to its price.
        assert after["dist_change_1"] == pytest.approx(
            np.log(scored["Close"].iloc[-1] / events.iloc[0]["price"]) * mm.BPS
        )
        assert after["bars_since_change"] == len(scored) - 1 - change_pos
        # Only one change ever happened, so the older slots stay empty.
        assert np.isnan(after["dist_change_2"])

    def test_event_history_features_are_nan_without_any_events(self):
        df = _session("2026-07-20", _flat_then_trend(390, 0, 0.0))
        scored, _ = mm.prepare(df, PARAMS)
        X = mm.build_live_features(scored, pd.DataFrame(), n_prev_changes=3)
        assert X[["dist_change_1", "dist_change_2", "dist_change_3"]].isna().all().all()
        assert X["bars_since_change"].isna().all()


class TestScoring:
    def test_score_latest_agrees_with_score_bars(self):
        df = _session("2026-07-20", _flat_then_trend(150, 240, 8.0))
        scored, events = mm.prepare(df, PARAMS)

        class Estimator:
            """Stands in for the sklearn pipeline: sums two features so the
            call is verifiable without shipping the fitted artifact."""

            def predict(self, X):
                return (X["mom_15"] + X["mom_smooth_z"]).to_numpy()

        bundle = {
            "estimator": Estimator(),
            "feature_cols": mm.FEATURE_COLS,
            "pipeline_params": PARAMS,
        }
        preds = mm.score_bars(bundle, scored, events)
        assert mm.score_latest(bundle, scored, events) == pytest.approx(preds.iloc[-1])
        # Bars whose momentum windows haven't filled stay unscored rather than
        # getting a fabricated number.
        assert preds.iloc[:15].isna().all()

    def test_read_latest_projects_the_regime_the_model_points_at(self, monkeypatch):
        df = _session("2026-07-20", _flat_then_trend(150, 240, 8.0))

        class Estimator:
            def predict(self, X):
                return np.array([5.0])

        bundle = {
            "estimator": Estimator(),
            "feature_cols": mm.FEATURE_COLS,
            "pipeline_params": PARAMS,
        }
        read = mm.read_latest(bundle, df)
        assert read["pred"] == 5.0
        assert read["projected_mom"] == pytest.approx(read["mom"] + 5.0)
        assert read["projected_regime"] == 1
        assert read["bars_today"] == 390
        assert mm.regime_name(read["regime"]) in {"negative", "balanced", "positive"}

    def test_read_latest_returns_none_on_a_frame_too_short_to_score(self):
        bundle = {"estimator": None, "feature_cols": mm.FEATURE_COLS, "pipeline_params": PARAMS}
        assert mm.read_latest(bundle, _session("2026-07-20", [100.0] * 5)) is None


class TestMinuteFrame:
    def _bars(self, day: str, start_hhmm: str, n: int, price: float) -> list[dict]:
        idx = pd.date_range(
            f"{day} {start_hhmm}", periods=n, freq="1min", tz="America/New_York"
        ).tz_convert("UTC")
        return [
            {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": price, "h": price, "l": price,
             "c": price, "v": 100.0}
            for ts in idx
        ]

    def test_frame_from_bars_keeps_only_the_regular_session(self):
        bars = self._bars("2026-07-20", "08:00", 180, 100.0)  # 08:00 -> 10:59 ET
        frame = mm._frame_from_bars(bars)
        assert frame.index.min().strftime("%H:%M") == "09:30"
        assert frame.index.max().strftime("%H:%M") == "10:59"
        assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "day"]

    def test_live_bars_win_over_the_cached_history(self, monkeypatch):
        """Today comes from the live stream; the days before it from yfinance.
        Where the two overlap the live bar is the fresher one."""
        from types import SimpleNamespace
        import threading

        today = pd.Timestamp("2026-07-21").date()
        history = mm._frame_from_bars(
            self._bars("2026-07-20", "09:30", 390, 100.0) + self._bars("2026-07-21", "09:30", 5, 111.0)
        )
        monkeypatch.setattr(mm, "_previous_sessions", lambda *a, **k: history)
        monkeypatch.setattr(mm.clock, "now", lambda: pd.Timestamp("2026-07-21 15:00", tz="UTC"))

        sym_state = SimpleNamespace(
            symbol="AAPL", lock=threading.Lock(), bars=self._bars("2026-07-21", "09:30", 30, 222.0)
        )
        frame = mm.minute_frame(sym_state)
        assert frame[frame["day"] < today]["Close"].eq(100.0).all()
        assert frame[frame["day"] == today]["Close"].eq(222.0).all()
        assert len(frame[frame["day"] == today]) == 30
