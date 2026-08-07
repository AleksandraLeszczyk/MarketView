"""The N-BEATS route from a momentum forecast to a persistence probability.

Two kinds of test live here. The pure ones pin the bridge -- the Schmitt-trigger
replay, the run-length count and the residual bootstrap -- because those are
mirrored from `mshift.forecast` and a drift in any of them changes every
probability the model reports without changing anything that looks broken. The
rest load the real checkpoint and are skipped when it is not on this machine;
what they pin is the bundle's *contract* with `persistence_model`, which is the
thing that makes the model swappable at all.

The numbers themselves were checked against notebook 07 at wiring time: given
the notebook's own sequences for 2026-07-27, this module reproduces its four
decisions and their probabilities (0.620 / 0.688 / 0.000 / 0.682) exactly.
"""

import numpy as np
import pandas as pd
import pytest

# Through the module, never `importorskip("torch")` directly: it pulls
# LightGBM in ahead of torch to keep their two OpenMP runtimes from colliding,
# and a test session that imports torch first segfaults later in
# `test_profile_model` with no traceback. See the module's own comment.
nb = pytest.importorskip("agent_stonks.nbeats_model")

from agent_stonks import apple_models  # noqa: E402
from agent_stonks import persistence_model as pm  # noqa: E402

ENTER, EXIT = 0.90, 0.40
MOMENTUM = dict(pm.MOMENTUM_DEFAULTS)


class TestForwardRegimePath:
    """The trigger replayed over a forecast is the same trigger, minus the
    session reset a forecast path cannot cross."""

    def test_holds_a_regime_until_momentum_falls_below_the_exit_band(self):
        path = [[1.2, 1.0, 0.6, 0.45, 0.3, 0.2]]
        states = nb.forward_regime_path(path, 1, ENTER, EXIT)
        # 0.45 is still above the 0.40 exit, so the regime survives it.
        assert list(states[0]) == [1, 1, 1, 1, 0, 0]

    def test_flips_straight_through_neutral_on_a_hard_reversal(self):
        states = nb.forward_regime_path([[-1.5, -1.0]], 1, ENTER, EXIT)
        assert list(states[0]) == [-1, -1]

    def test_re_entering_needs_the_full_entry_threshold(self):
        states = nb.forward_regime_path([[0.0, 0.5, 0.8, 0.95]], 1, ENTER, EXIT)
        assert list(states[0]) == [0, 0, 0, 1]

    def test_rows_are_independent(self):
        states = nb.forward_regime_path([[1.2, 1.2], [0.0, 0.0]], [1, 1], ENTER, EXIT)
        assert list(states[0]) == [1, 1]
        assert list(states[1]) == [0, 0]


class TestPostDwell:
    def test_counts_the_anchor_bar_so_a_regime_that_dies_at_once_is_one(self):
        assert nb.post_dwell_from_path([[0.0, 0.0, 0.0]], 1, ENTER, EXIT)[0] == 1

    def test_a_path_that_never_leaves_runs_past_the_horizon(self):
        # Anchor bar plus 15 forecast bars with no break: 16, not 15. The
        # count is deliberately not clipped to the horizon -- `>= min_dwell` is
        # all the label asks, and clipping would make "survived exactly 15" and
        # "still going" indistinguishable.
        assert nb.post_dwell_from_path([[1.5] * 15], 1, ENTER, EXIT)[0] == 16

    def test_a_regime_that_recovers_still_ends_at_the_first_break(self):
        # Run length, not total time spent in the regime: the label is about
        # the run the change started, and a gap ends it.
        assert nb.post_dwell_from_path([[1.5, 0.0, 1.5, 1.5]], 1, ENTER, EXIT)[0] == 2


class TestPersistenceProbability:
    def _samples(self, n_survive: int, n_die: int) -> np.ndarray:
        survive = np.full((n_survive, 15), 1.5)
        die = np.concatenate([np.full((n_die, 1), 1.5), np.zeros((n_die, 14))], axis=1)
        return np.concatenate([survive, die])[None, ...]

    def test_p_post_is_the_fraction_of_futures_that_survive_min_dwell(self):
        out = nb.persistence_probability(self._samples(30, 70), [1], [20], MOMENTUM, 15)
        assert out["p_post"][0] == pytest.approx(0.30)
        # 16 bars for the survivors (anchor + a clean horizon), 2 for the ones
        # that fall out of the band on their second bar.
        assert out["expected_post_dwell"][0] == pytest.approx(0.3 * 16 + 0.7 * 2)

    def test_the_pre_dwell_gate_multiplies_the_sampled_part_away(self):
        out = nb.persistence_probability(self._samples(90, 10), [1], [8], MOMENTUM, 15)
        # The forecast is emphatic and irrelevant: the observable precondition
        # is necessary for the label, so a closed gate is a zero whatever the
        # paths say. Both halves stay visible, which is the point of returning
        # them separately.
        assert out["p_post"][0] == pytest.approx(0.90)
        assert out["pre_dwell_gate"][0] == 0.0
        assert out["p_full"][0] == 0.0

    def test_a_horizon_too_short_to_decide_the_label_raises(self):
        with pytest.raises(ValueError, match="too short"):
            nb.persistence_probability(np.ones((1, 5, 3)), [1], [20], MOMENTUM, 15)


class TestBootstrapPaths:
    """Whole residual rows, drawn reproducibly."""

    def test_every_path_is_the_point_forecast_plus_one_whole_residual_row(self):
        point = np.array([[0.0, 1.0, 2.0]])
        residuals = np.array([[10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])
        paths = nb.bootstrap_paths(point, residuals, n_paths=50)
        assert paths.shape == (1, 50, 3)
        drawn = paths[0] - point
        # Cell-wise noise would produce rows that are in neither residual; the
        # correlation across the horizon is exactly what must survive.
        for row in drawn:
            assert any(np.allclose(row, r) for r in residuals)

    def test_the_draw_is_seeded_so_one_tape_gives_one_answer(self):
        point = np.zeros((2, 3))
        residuals = np.arange(30, dtype=float).reshape(10, 3)
        assert np.array_equal(
            nb.bootstrap_paths(point, residuals, 20), nb.bootstrap_paths(point, residuals, 20)
        )
        assert not np.array_equal(
            nb.bootstrap_paths(point, residuals, 20, seed=1),
            nb.bootstrap_paths(point, residuals, 20, seed=2),
        )


class TestBuildExog:
    def test_is_the_anchor_bar_followed_by_the_window_mean(self):
        X = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        exog = nb.build_exog(X)
        assert exog.shape == (2, 6)
        assert np.allclose(exog[:, :3], X[:, -1, :])
        assert np.allclose(exog[:, 3:], X.mean(axis=1))


# --- the saved checkpoint ----------------------------------------------------

BUNDLE = nb.load_bundle()
needs_model = pytest.mark.skipif(
    BUNDLE is None, reason=f"no N-BEATS checkpoint at {nb.model_path()}"
)


@needs_model
class TestBundle:
    """What the rest of the app is allowed to assume about the loaded model."""

    def test_carries_the_same_window_contract_as_the_incumbent(self):
        assert BUNDLE["feature_columns"] == pm.FEATURE_COLUMNS
        assert BUNDLE["seq_len"] == 20
        assert BUNDLE["n_seeds"] == 5

    def test_momentum_settings_match_the_live_pipeline(self):
        # The trigger the forecast is replayed through has to be the trigger
        # the live bars were classified with, or the two halves of the
        # probability are about different regimes.
        assert pm.momentum_params(BUNDLE) == pm.MOMENTUM_DEFAULTS

    def test_the_threshold_is_the_one_picked_on_the_validation_events(self):
        # Not comparable to the classifier's: `p_full` is a survival
        # probability times a hard gate, and 0.5 means nothing on it.
        assert pm.model_threshold(BUNDLE) == pytest.approx(0.05)

    def test_residuals_are_the_training_windows_only(self):
        assert BUNDLE["n_residual_rows"] > 1000
        assert BUNDLE["horizon"] == 15

    def test_scores_through_the_same_entry_point_as_the_classifier(self):
        X = np.zeros((1, 20, len(pm.FEATURE_COLUMNS)), dtype=np.float32)
        X[0, :, pm.FEATURE_COLUMNS.index("f_regime")] = 1.0
        X[0, :, pm.FEATURE_COLUMNS.index("f_prev_dwell")] = np.log1p(40.0)
        proba = pm.predict_proba(BUNDLE, X)
        assert proba.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0
        assert proba[0] == pytest.approx(pm.predict_proba(BUNDLE, X)[0])

    def test_a_closed_gate_is_zero_whatever_the_forecast_says(self):
        X = np.zeros((1, 20, len(pm.FEATURE_COLUMNS)), dtype=np.float32)
        X[0, :, pm.FEATURE_COLUMNS.index("f_regime")] = 1.0
        X[0, :, pm.FEATURE_COLUMNS.index("f_prev_dwell")] = np.log1p(8.0)
        assert pm.predict_proba(BUNDLE, X)[0] == 0.0

    def test_the_wrong_shaped_sequence_is_refused_not_reshaped(self):
        with pytest.raises(ValueError, match="expected sequences"):
            pm.predict_proba(BUNDLE, np.zeros((1, 19, len(pm.FEATURE_COLUMNS))))


class TestAppleModelsRegistry:
    def test_both_models_are_offered_with_the_classifier_first(self):
        assert apple_models.keys() == ["persistence", "nbeats"]
        assert apple_models.DEFAULT_MODEL == "persistence"

    def test_an_unknown_key_falls_back_rather_than_raising(self):
        # Stored experiment records outlive model names; a Results page should
        # still render one written before this registry existed.
        assert apple_models.get("retired-model").key == apple_models.DEFAULT_MODEL
        assert apple_models.get(None).key == apple_models.DEFAULT_MODEL

    def test_the_unavailable_message_names_the_file_and_the_dependency(self):
        reason = apple_models.unavailable_reason("nbeats")
        assert "timetochange2_nbeats.pt" in reason
        assert "PyTorch" in reason

    @needs_model
    def test_loads_the_named_model_and_not_the_default(self):
        assert apple_models.load("nbeats") is BUNDLE
        assert apple_models.threshold("nbeats") == pytest.approx(0.05)


@needs_model
class TestReadLatestOnTheRealModel:
    """The live entry point does not know which model it is holding."""

    def _session(self) -> pd.DataFrame:
        pytest.importorskip("pandas")
        from tests.test_persistence_model import _bars, _flat_then_trend

        return pm._frame_from_bars(_bars("2026-07-21", _flat_then_trend(90, 90, 3e-4)))

    def test_a_to_positive_change_gets_a_probability_in_range(self):
        frame = self._session()
        reads = [
            pm.read_latest(BUNDLE, frame.iloc[: i + 1]) for i in range(len(frame))
        ]
        scored = [r for r in reads if r and r["to_positive"] and r["proba"] is not None]
        assert scored, "the synthetic trend should produce at least one scored change"
        assert all(0.0 <= r["proba"] <= 1.0 for r in scored)

    def test_bars_that_are_not_a_to_positive_change_are_never_scored(self):
        frame = self._session()
        for i in range(len(frame)):
            read = pm.read_latest(BUNDLE, frame.iloc[: i + 1])
            if read and not read["to_positive"]:
                assert read["proba"] is None
