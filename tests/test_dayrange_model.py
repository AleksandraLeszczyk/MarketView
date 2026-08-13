"""The TimeToChange3 day-range mirror: features, assembly, and the bundle.

Two kinds of test here, and the distinction matters.

The ones that need neither the saved model nor the notebooks pin the *seams* --
what the app has to get right to hand the model correct inputs: today's row
carrying only its open, a short history being refused rather than extrapolated,
the opening-range arithmetic matching the notebook's group-by form.

The one that needs both pins the *mirror itself*, against the number notebook
05 recorded for 2026-08-07. That is the test that would fail if
`dayrange/features.py` changed and this copy did not, and it is exact to the
last decimal rather than approximate -- the same contract
`tests/test_persistence_model.py` holds with `mshift`.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models, clock

D = pytest.importorskip("agent_stonks.dayrange_model")

NOTEBOOK = Path(
    "/Users/aleksandra/Documents/playground/Code/FinNotebooks/TimeToChange3"
)

# Every ticker TimeToChange3 was run for, from the registry rather than a list
# here -- adding one there should extend the mirror check rather than leave it
# silently covering the old set.
TICKERS = list(apple_models.DAYRANGE_TICKERS)

needs_model = pytest.mark.skipif(
    not D.model_path().exists(), reason="the TimeToChange3 bundle is not installed"
)


def _notebook_data(ticker: str) -> "tuple[Path, Path]":
    """Where the notebooks keep one ticker's minute and daily frames."""
    return (
        NOTEBOOK / "data" / ticker / "minute.parquet",
        NOTEBOOK / "data" / ticker / "daily.parquet",
    )


def synthetic_daily(n: int = 400, start: str = "2025-01-02") -> pd.DataFrame:
    """A daily series long enough to clear every trailing window.

    Deterministic and gently trending, so the feature table's last row is
    finite everywhere -- these tests are about plumbing, not about whether the
    numbers are any good.
    """
    dates = pd.bdate_range(start, periods=n, name="date")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0.006, 0.003, n))),
            "low": close * (1 - np.abs(rng.normal(0.006, 0.003, n))),
            "close": close,
            "volume": rng.uniform(4e7, 6e7, n),
        },
        index=dates,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


def synthetic_opening(day: pd.Timestamp, base: float = 120.0, n: int = 5) -> pd.DataFrame:
    """`n` one-minute bars from 09:30 on `day`."""
    index = pd.date_range(
        day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30),
        periods=n,
        freq="1min",
    )
    step = np.linspace(0, 0.4, n)
    return pd.DataFrame(
        {
            "open": base + step,
            "high": base + step + 0.15,
            "low": base + step - 0.1,
            "close": base + step + 0.05,
            "volume": np.full(n, 3.0e5),
        },
        index=index,
    )


class TestDailyFrameFromBars:
    def test_bars_become_a_date_indexed_ohlcv_frame(self):
        frame = D.daily_frame_from_bars(
            [
                {"t": "2026-08-05", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                {"t": "2026-08-06T05:00:00Z", "o": 2, "h": 3, "l": 1.5, "c": 2.5, "v": 20},
            ]
        )
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert list(frame.index) == [pd.Timestamp("2026-08-05"), pd.Timestamp("2026-08-06")]
        assert frame.loc[pd.Timestamp("2026-08-06"), "close"] == 2.5

    def test_a_repeated_day_keeps_the_last_copy(self):
        """A backfill overlapping the cache is the normal case, not an error;
        the later row is the corrected one."""
        frame = D.daily_frame_from_bars(
            [
                {"t": "2026-08-05", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                {"t": "2026-08-05", "o": 1, "h": 2, "l": 0.5, "c": 9.9, "v": 10},
            ]
        )
        assert len(frame) == 1
        assert frame["close"].iloc[0] == 9.9

    def test_no_bars_is_an_empty_frame_of_the_right_shape(self):
        frame = D.daily_frame_from_bars([])
        assert len(frame) == 0
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


class TestTodaysRow:
    """The seam that matters most: today's daily bar does not exist at 9:35."""

    def test_only_the_open_of_todays_row_reaches_the_features(self):
        """The row carries the opening window's high, low and close because it
        has to carry something. If any of them were read, this test would see
        the feature table move when they change -- and a feature reading
        today's high is a feature reading the answer."""
        day = pd.Timestamp("2026-08-07")
        history = synthetic_daily()
        history = history[history.index < day]
        opening = synthetic_opening(day)

        base = D.daily_features(
            D.session_daily_frame(history, opening, day, open_price=121.0)
        ).loc[day]

        wilder = opening.copy()
        wilder["high"] *= 1.05
        wilder["low"] *= 0.95
        wilder["close"] *= 1.03
        wilder["volume"] *= 10
        moved = D.daily_features(
            D.session_daily_frame(history, wilder, day, open_price=121.0)
        ).loc[day]

        model_inputs = [c for c in D.DAILY_FEATURE_COLS] + ["prev_avg", "adr14_abs", "advol14"]
        pd.testing.assert_series_equal(base[model_inputs], moved[model_inputs])

    def test_the_open_is_the_official_print_when_one_is_supplied(self):
        day = pd.Timestamp("2026-08-07")
        opening = synthetic_opening(day, base=120.0)
        frame = D.session_daily_frame(
            synthetic_daily(), opening, day, open_price=121.5
        )
        assert frame.loc[day, "open"] == pytest.approx(121.5)

    def test_the_first_minute_bar_stands_in_when_it_is_not(self):
        day = pd.Timestamp("2026-08-07")
        opening = synthetic_opening(day, base=120.0)
        frame = D.session_daily_frame(synthetic_daily(), opening, day, open_price=None)
        assert frame.loc[day, "open"] == pytest.approx(float(opening["open"].iloc[0]))

    def test_a_stale_row_for_today_in_the_history_is_replaced(self):
        """A daily feed that includes a partial row for today must not win over
        the row assembled here -- its high and low are the day so far."""
        day = pd.Timestamp("2026-08-07")
        history = synthetic_daily()
        history.loc[day] = [999.0, 999.0, 999.0, 999.0, 1.0]
        frame = D.session_daily_frame(
            history.sort_index(), synthetic_opening(day), day, open_price=121.5
        )
        assert frame.loc[day, "open"] == pytest.approx(121.5)
        assert frame.loc[day, "high"] != 999.0
        assert len(frame.loc[[day]]) == 1


class TestHistoryRequirement:
    def test_a_short_history_is_refused(self):
        problem = D.require_history(synthetic_daily(n=120))
        assert problem is not None
        assert "252" in problem

    def test_a_full_history_passes(self):
        assert D.require_history(synthetic_daily(n=400)) is None

    def test_the_refusal_reaches_the_caller_as_an_error(self):
        day = pd.Timestamp("2026-08-07")
        short = synthetic_daily(n=100)
        with pytest.raises(ValueError, match="252"):
            D.forecast_session(
                {"model": None, "opening_minutes": 5},
                short[short.index < day],
                synthetic_opening(day),
                day,
            )

    def test_an_incomplete_opening_window_is_refused(self):
        day = pd.Timestamp("2026-08-07")
        with pytest.raises(ValueError, match="first 5 minutes"):
            D.forecast_session(
                {"model": None, "opening_minutes": 5},
                synthetic_daily(),
                synthetic_opening(day, n=3),
                day,
            )


class TestOpeningFeatures:
    """One session's opening features, against the notebook's group-by form.

    `dayrange.features.opening_features` computes these for every stored day at
    once; live there is only one. The two must agree column for column, so this
    reimplements the group-by version here and compares.
    """

    def _grouped(self, opening: pd.DataFrame, row: pd.Series, n: int = 5) -> dict:
        first = opening.iloc[:n]
        o5 = first["open"].iloc[0]
        h5, l5 = first["high"].max(), first["low"].min()
        c5, v5 = first["close"].iloc[-1], first["volume"].sum()
        minute_ret = np.log(first["close"] / first["open"])
        ref, adr = row["prev_avg"], row["adr14"]
        typical = row["advol14"] * (n / 390)
        return {
            "or_high": np.log(h5 / ref),
            "or_low": np.log(l5 / ref),
            "or_close": np.log(c5 / ref),
            "or_ret": np.log(c5 / o5),
            "or_range": (h5 - l5) / ref,
            "or_range_vs_adr": ((h5 - l5) / ref) / adr,
            "or_up": (h5 - o5) / ref,
            "or_down": (o5 - l5) / ref,
            "or_close_pos": (c5 - l5) / (h5 - l5),
            "or_ret_std": minute_ret.std(),
            "or_up_bars": (minute_ret > 0).mean(),
            "or_volume_share": np.log(v5 / typical),
        }

    def test_every_column_matches_the_grouped_form(self):
        day = pd.Timestamp("2026-08-07")
        history = synthetic_daily()
        opening = synthetic_opening(day)
        row = D.daily_features(
            D.session_daily_frame(history[history.index < day], opening, day)
        ).loc[day]

        mine = D.opening_features(opening, row)
        for name, expected in self._grouped(opening, row).items():
            assert mine[name] == pytest.approx(float(expected)), name

    def test_a_flat_opening_range_scores_the_middle_rather_than_dividing_by_zero(self):
        """`_pos_in_range` returns 0.5 on an empty span; the scalar form has to
        do the same, and a five-minute window that never moves is a real (if
        rare) shape on a thin tape."""
        day = pd.Timestamp("2026-08-07")
        flat = synthetic_opening(day)
        for col in ("open", "high", "low", "close"):
            flat[col] = 120.0
        row = D.daily_features(
            D.session_daily_frame(synthetic_daily(), flat, day)
        ).loc[day]
        assert D.opening_features(flat, row)["or_close_pos"] == pytest.approx(0.5)


class TestVolumeScaleWarning:
    def test_iex_is_flagged(self):
        warning = D.volume_scale_warning("iex")
        assert warning is not None and "consolidated" in warning

    @pytest.mark.parametrize("feed", ["sip", "yfinance", None, ""])
    def test_consolidated_tapes_are_not(self, feed):
        assert D.volume_scale_warning(feed) is None


class TestMarketDate:
    def test_the_session_date_follows_the_simulated_clock(self):
        # 00:30 UTC is still the previous day in New York, which is the
        # boundary a UTC date would get wrong.
        clock.set_simulated(datetime(2026, 8, 8, 0, 30, tzinfo=timezone.utc))
        try:
            assert D.market_date() == pd.Timestamp("2026-08-07")
        finally:
            clock.clear()


@needs_model
class TestBundle:
    def test_all_three_daily_models_load(self):
        """The published error is the three-way blend's. A bundle that quietly
        came back with one voter would forecast something else."""
        bundle = D.load_bundle()
        assert bundle is not None
        assert sorted(bundle["daily_models"]) == ["lgbm", "nbeats", "nhits"]
        assert bundle["model"].correction is not None
        assert bundle["model"].use_constraint is True

    def test_the_weights_are_equal(self):
        assert len(set(D.load_bundle()["model"].weights.values())) == 1

    def test_a_missing_file_is_none_rather_than_an_exception(self, monkeypatch, tmp_path):
        monkeypatch.setenv(D.MODEL_PATH_ENV, str(tmp_path / "nope.joblib"))
        D.reset_bundle_cache()
        try:
            assert D.load_bundle() is None
        finally:
            monkeypatch.delenv(D.MODEL_PATH_ENV, raising=False)
            D.reset_bundle_cache()

    def test_a_missing_checkpoint_makes_the_bundle_unavailable(self, monkeypatch, tmp_path):
        """Not a degraded LightGBM-only model: two of three voters missing is a
        different predictor with different error, and it would report the
        blend's numbers."""
        joblib = pytest.importorskip("joblib")
        copy = tmp_path / "bundle.joblib"
        joblib.dump(joblib.load(D.model_path()), copy)  # no .pt files beside it
        monkeypatch.setenv(D.MODEL_PATH_ENV, str(copy))
        D.reset_bundle_cache()
        try:
            assert D.load_bundle() is None
        finally:
            monkeypatch.delenv(D.MODEL_PATH_ENV, raising=False)
            D.reset_bundle_cache()


@pytest.mark.parametrize("ticker", TICKERS)
class TestAgainstTheNotebook:
    """The mirror contract, pinned to notebook 05's recorded forecast -- once
    per ticker TimeToChange3 was fitted for.

    Exact rather than approximate: every step from the raw daily frame to the
    predicted price is deterministic, so any difference at all means this copy
    and `dayrange/features.py` have diverged. Running it for all three also
    pins the thing a single-ticker check cannot: that `model_path(ticker)`
    reaches the *right* bundle, since the GOOGL model reproducing GOOGL's
    recorded forecast is only possible if it was the one that loaded.

    The session each one is checked on is the one that bundle recorded
    (`sim_date_forecast.date`), which is its own held-out day rather than a
    date fixed here.
    """

    @pytest.fixture
    def inputs(self, ticker):
        minute_file, daily_file = _notebook_data(ticker)
        if not minute_file.exists():
            pytest.skip(f"the TimeToChange3 {ticker} notebook data is not on this machine")
        bundle = D.load_bundle(ticker)
        if bundle is None:
            pytest.skip(f"the TimeToChange3 {ticker} bundle is not installed")
        day = pd.Timestamp(bundle["metadata"]["sim_date_forecast"]["date"])
        minute = pd.read_parquet(minute_file)
        daily = pd.read_parquet(daily_file)[["open", "high", "low", "close", "volume"]]
        return bundle, day, minute[minute["date"] == day], daily

    def test_reproduces_the_recorded_forecast_exactly(self, inputs):
        bundle, day, bars, daily = inputs
        out = D.forecast_session(
            bundle,
            daily[daily.index < day],
            bars.iloc[: D.opening_minutes(bundle)],
            day,
            # The official opening print, which is what
            # `historical.fetch_session_open` supplies live and what the
            # notebook's daily frame carries.
            open_price=float(daily.loc[day, "open"]),
        )
        recorded = bundle["metadata"]["sim_date_forecast"]
        for key in ("prev_avg", "pred_high", "pred_low", "adr14_abs"):
            assert out[key] == pytest.approx(recorded[key], abs=1e-9), key

    def test_the_minute_open_fallback_costs_a_few_cents_not_a_different_answer(self, inputs):
        """Without the official print the first minute bar's open stands in.
        The two differ on about half of sessions; this pins the size of that
        substitution against the model's own mean error (about $2 on AAPL), so
        a change that made the fallback matter would show up here."""
        bundle, day, bars, daily = inputs
        recorded = bundle["metadata"]["sim_date_forecast"]
        fallback = D.forecast_session(
            bundle, daily[daily.index < day],
            bars.iloc[: D.opening_minutes(bundle)], day, open_price=None,
        )
        assert abs(fallback["pred_high"] - recorded["pred_high"]) < 0.25

    def test_the_bundle_that_loaded_is_the_one_that_was_asked_for(self, ticker):
        """A path bug that fell back to the default would otherwise show up as
        a forecast that is merely wrong rather than as a wrong model."""
        bundle = D.load_bundle(ticker)
        if bundle is None:
            pytest.skip(f"the TimeToChange3 {ticker} bundle is not installed")
        assert bundle["ticker"] == ticker
        assert D.model_path(ticker).name.endswith(f"_{ticker}.joblib")
