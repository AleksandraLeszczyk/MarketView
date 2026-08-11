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

    def _anticipation_window(self, regime: float, dwell: float) -> np.ndarray:
        X = np.zeros((1, 20, len(pm.FEATURE_COLUMNS)), dtype=np.float32)
        X[0, :, pm.FEATURE_COLUMNS.index("f_regime")] = regime
        X[0, :, pm.FEATURE_COLUMNS.index("f_bars_in_regime")] = np.log1p(dwell)
        return X

    def test_the_forecaster_answers_the_anticipation_question_too(self):
        proba = pm.predict_turn_proba(BUNDLE, self._anticipation_window(0.0, 40.0))
        assert proba.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0
        # Seeded bootstrap: the same tape has to give the same trades.
        assert proba[0] == pytest.approx(
            pm.predict_turn_proba(BUNDLE, self._anticipation_window(0.0, 40.0))[0]
        )

    def test_the_dwell_gate_closes_on_the_bar_before_the_change_too(self):
        """`anticipate` carries the same observable half of the persistence
        label as `confirm`, read one bar earlier: there it is `pre_dwell` on
        the change bar, here it is the current regime's own length."""
        assert pm.predict_turn_proba(BUNDLE, self._anticipation_window(0.0, 8.0))[0] == 0.0

    def test_the_wrong_shaped_sequence_is_refused_here_as_well(self):
        with pytest.raises(ValueError, match="expected sequences"):
            pm.predict_turn_proba(BUNDLE, np.zeros((1, 19, len(pm.FEATURE_COLUMNS))))

    def test_the_forecaster_answers_the_exit_question_too(self):
        X = self._anticipation_window(1.0, 40.0)
        proba = pm.predict_reversal_proba(BUNDLE, X)
        assert proba.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0
        assert proba[0] == pytest.approx(pm.predict_reversal_proba(BUNDLE, X)[0])

    def test_the_exit_question_carries_no_dwell_gate(self):
        """A dwell that would zero the entry questions leaves this one alone:
        a position taken on the turn is holding a young regime by design."""
        assert pm.predict_reversal_proba(BUNDLE, self._anticipation_window(1.0, 4.0))[0] > 0.0

    def test_a_bar_that_is_not_positive_is_not_asked(self):
        assert pm.predict_reversal_proba(BUNDLE, self._anticipation_window(0.0, 40.0))[0] == 0.0


class TestTurnProbability:
    """The forward-shifted twin of `persistence_probability`: the anchor bar is
    one the regime has NOT turned on yet."""

    def _paths(self, *rows) -> np.ndarray:
        return np.asarray([list(rows)], dtype=float)

    def test_a_path_positive_for_the_whole_horizon_counts(self):
        paths = self._paths([1.2] * 15)
        out = nb.turn_probability(paths, 0, 40, MOMENTUM, 15)
        assert out["p_turn"][0] == pytest.approx(1.0)
        assert out["p_full"][0] == pytest.approx(1.0)

    def test_a_turn_that_dies_inside_the_horizon_does_not(self):
        paths = self._paths([1.2] * 10 + [0.0] * 5)
        assert nb.turn_probability(paths, 0, 40, MOMENTUM, 15)["p_turn"][0] == 0.0

    def test_a_turn_that_arrives_late_does_not_count_either(self):
        """The horizon is exactly `min_dwell` long, so only a turn on the very
        first forecast bar leaves room to see it survive. A later one is not
        rejected as unlikely -- it simply cannot be checked."""
        paths = self._paths([0.5] + [1.2] * 14)
        assert nb.turn_probability(paths, 0, 40, MOMENTUM, 15)["p_turn"][0] == 0.0

    def test_a_short_dwell_closes_the_gate_whatever_the_paths_say(self):
        out = nb.turn_probability(self._paths([1.2] * 15), 0, 8, MOMENTUM, 15)
        assert out["p_turn"][0] == pytest.approx(1.0)
        assert out["p_full"][0] == 0.0

    def test_it_starts_from_the_negative_regime_as_readily_as_the_balanced_one(self):
        # From -1 the trigger needs `mom > enter` to reach +1 directly.
        out = nb.turn_probability(self._paths([1.2] * 15), -1, 40, MOMENTUM, 15)
        assert out["p_turn"][0] == pytest.approx(1.0)

    def test_a_horizon_too_short_to_decide_raises(self):
        with pytest.raises(ValueError, match="too short"):
            nb.turn_probability(self._paths([1.2] * 10), 0, 40, MOMENTUM, 15)


class TestReversalProbability:
    """The exit question: anchored on a positive bar, does the regime flip
    negative inside the horizon."""

    def _paths(self, *rows) -> np.ndarray:
        return np.asarray([list(rows)], dtype=float)

    def test_a_path_that_crashes_through_the_band_counts(self):
        # From +1 the trigger leaves below `exit_` and lands on -1 directly
        # when the same bar is also below `-enter`.
        out = nb.reversal_probability(self._paths([1.2] * 5 + [-1.5] * 10), 1, MOMENTUM)
        assert out["p_reversal"][0] == pytest.approx(1.0)
        assert out["expected_bars_to_reversal"][0] == pytest.approx(6.0)

    def test_a_path_that_only_fades_to_balanced_does_not(self):
        """The distinction the whole rule rests on: a move pausing is not a
        move reversing, and only the second is news the trailing stop does not
        already have."""
        out = nb.reversal_probability(self._paths([1.2] * 5 + [0.0] * 10), 1, MOMENTUM)
        assert out["p_reversal"][0] == 0.0
        assert out["expected_bars_to_reversal"][0] == pytest.approx(16.0)

    def test_a_slow_decline_through_neutral_still_counts(self):
        """Reaching negative by way of balanced is a reversal too -- what is
        excluded is stopping there, not passing through."""
        paths = self._paths([1.2, 0.6, 0.2, -0.3, -0.7] + [-1.4] * 10)
        out = nb.reversal_probability(paths, 1, MOMENTUM)
        assert out["p_reversal"][0] == pytest.approx(1.0)
        assert out["expected_bars_to_reversal"][0] == pytest.approx(6.0)

    def test_the_probability_is_the_fraction_of_futures_that_flip(self):
        paths = np.asarray([[[-1.5] * 15, [1.2] * 15, [-1.5] * 15, [0.0] * 15]])
        assert nb.reversal_probability(paths, 1, MOMENTUM)["p_reversal"][0] == pytest.approx(0.5)

    def test_there_is_no_dwell_gate(self):
        """A position entered on the turn holds a regime a few bars old by
        construction, so the gate the entry questions carry would switch this
        rule off exactly when it is needed."""
        out = nb.reversal_probability(self._paths([-1.5] * 15), 1, MOMENTUM)
        assert out["p_reversal"][0] == pytest.approx(1.0)
        assert "pre_dwell_gate" not in out

    def test_a_short_horizon_is_accepted(self):
        """Unlike the other two this decides nothing about survival, so there
        is no minimum length below which the answer is meaningless."""
        out = nb.reversal_probability(self._paths([-1.5] * 3), 1, MOMENTUM)
        assert out["p_reversal"][0] == pytest.approx(1.0)


class TestAppleModelsRegistry:
    def test_both_momentum_models_are_offered_with_the_forecaster_as_default(self):
        momentum = [k for k in apple_models.keys() if apple_models.is_momentum(k)]
        assert momentum == ["persistence", "nbeats"]
        # Apple Trader's default entry (`anticipate`) is a question only a
        # forecaster can answer, so the default model has to be one.
        assert apple_models.DEFAULT_MODEL == "nbeats"

    def test_only_the_forecaster_claims_it_can_anticipate(self):
        """The flag a picker reads, without importing 200 MB of PyTorch to
        find out. `persistence_model.anticipates` is the authority at run
        time; these two must not disagree.

        Only asked of the momentum models: the entry mode is not part of the
        day-range strategy's configuration, so its flag says nothing.
        """
        assert apple_models.get("nbeats").anticipates is True
        assert apple_models.get("persistence").anticipates is False

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
