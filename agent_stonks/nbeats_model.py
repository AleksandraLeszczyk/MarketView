"""N-BEATS -> momentum forecast -> persistence probability (TimeToChange2 nb 07).

The Apple Trader's incumbent brain (`agent_stonks.persistence_model`) is a
classifier: it is handed the 20 bars ending at a momentum-regime change and
returns one number. This module is the alternative TimeToChange2 notebook 07
puts in the driver's seat instead -- a *forecaster*, which answers the same
question by a longer route:

1. **Point forecast.** Five seeds of N-BEATS each predict the next 15 bars of
   the momentum score; the ensemble is their mean. On 19 sessions the gap
   between two seeds is comparable to the gap between two architectures, so a
   single seed is noise and the ensemble *is* the model.
2. **Sample paths.** N-BEATS emits a point, not a distribution, so it gets one
   the way every point model in notebook 6 did: whole **rows** of its training
   residuals are resampled and added. Rows, not cells -- forecast errors are
   strongly correlated across the horizon, and i.i.d. per-step noise would
   cross the regime threshold far more often than real paths do, inflating
   every run-length probability derived from them.
3. **Replay the trigger.** Each path runs through the same Schmitt trigger the
   live pipeline uses, giving `P(post_dwell >= 15)` by Monte Carlo.
4. **Apply the gate.** `pre_dwell >= 15` is observable at the change bar and
   necessary for persistence, so it multiplies in as the hard 0/1 it is.

Why this is a *different model* and not a tuning of the old one
--------------------------------------------------------------
Notebook 6 found the incumbent classifier is a coin flip (0.48 AUC) on the half
of the problem that matters -- the changes that already pass the observable
`pre_dwell >= 15` pre-condition -- and that this model is not: 0.67 +/- 0.07 on
that hard half across four walk-forward folds, the only entrant above chance on
all four, and the only one whose bootstrap interval on the held-out split
excludes chance (0.713, 95% CI [0.52, 0.89]). That is a real but small effect
measured on 35 events. It is the reason this model is offered, and the size of
the interval is the reason it is offered as an *option* rather than as the new
default.

What it shares with the incumbent, exactly
------------------------------------------
Everything up to the scoring step. Bars, sessions, momentum, regimes and all 25
features come from `persistence_model` unchanged, and the sequence handed to
the network is the identical `(20, 25)` block -- the same contract, and the same
way for it to go silently wrong if `mshift` ever moves. Only the last step
differs, so `read_latest` cannot tell the two apart and Apple Trader's rules do
not change when the model does.

Three things this loader needs that a joblib pipeline does not
-------------------------------------------------------------
* **PyTorch.** Optional, like scikit-learn is for the incumbent; without it
  this model reports itself unavailable rather than failing at trade time.
  `apple_models` imports this module lazily for exactly that reason.
* **The residual sidecar.** Step 2 needs the ensemble's residuals over the
  training windows, and the checkpoint does not carry them -- the notebook
  recomputes them from the whole dataset, which AgentStonks does not have. They
  are exported once to `<model>_residuals.npz` and validated against the
  checkpoint on load, because a residual set that no longer belongs to these
  weights would produce plausible, wrong probabilities in silence.
* **A seeded bootstrap.** The probability is a Monte-Carlo estimate, so it has
  a standard error (~0.015 at 500 paths). The draw is seeded, so the same tape
  gives the same trades -- which is the property the rest of Apple Trader is
  built on. Notebook 7.8 checked that no decision on its simulation day flipped
  across eight seeds at any path count; that is reassurance, not a guarantee
  for a candidate sitting on the threshold.

  One consequence worth knowing before comparing numbers: the seeded draw has
  shape `(n_rows, n_paths)`, so a row's paths depend on how many rows were
  scored *with* it. Live, `read_latest` asks about one bar at a time, and the
  notebook scores a whole day at once -- so the same candidate can come back
  0.65 here and 0.68 there. That is the sampling error above, not a
  disagreement: the decisions on notebook 07's four candidates are identical
  either way, and `p_post` before the draw matches it to the last decimal.

`tests/test_nbeats_model.py` pins the network, the bridge and the bundle shape.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from pathlib import Path

import numpy as np

# PyTorch and LightGBM each ship their own copy of the OpenMP runtime, and this
# environment has no system `libomp` for them to share. Loading a LightGBM
# model into a process that already imported torch segfaults outright -- no
# exception, no traceback -- which would take the whole app down the first time
# a chart asked `profile_model` for the LevelsML pack. Importing LightGBM
# first is enough to avoid it, so this module pulls it in ahead of torch
# whenever it is installed. It costs one import in the one process that would
# otherwise crash, and the alternative (KMP_DUPLICATE_LIB_OK) is documented as
# unsafe rather than as a fix.
try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from . import persistence_model  # noqa: E402

# The shared model store next to the AgentStonks checkout (Code/Models), where
# the incumbent bundle and the LevelsML pack already live.
MODEL_PATH_ENV = "APPLE_NBEATS_MODEL"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "Models" / "timetochange2_nbeats.pt"
)

# How many futures to sample per decision. Notebook 7.8 measured the sampling
# standard deviation at 0.013-0.016 here, against probabilities near 0.65.
N_PATHS = 500
# Fixed so one tape produces one answer. `benchmark._bootstrap_paths` is called
# with the seed as its `seed` argument; notebook 07 passes 0 for the ensemble.
BOOTSTRAP_SEED = 0

# `mshift.config.PersistenceParams` -- the label this model reconstructs. Only
# `min_dwell` is used: the fast-move clause fires 0 times on AAPL, which is
# precisely what lets a momentum forecast reproduce the label exactly.
PERSISTENCE_DEFAULTS = {"min_dwell": 15, "fast_move_bars": 10, "fast_move_pct": 0.01}

_lock = threading.Lock()
_cache: dict = {"path": None, "bundle": None}

DEVICE = torch.device("cpu")


# --- the network (mirrors mshift.deep) ---------------------------------------

class SeqStandardizer:
    """Per-channel standardisation of `(n, L, F)` sequences.

    Restored from the checkpoint rather than fitted here: the statistics are
    the *training rows'*, and recomputing them from live bars would quietly
    rescale every input the model was trained to read.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.std_ = np.asarray(std, dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.std_).astype(
            np.float32
        )


class TargetScaler:
    """Scalar standardisation of the forecast target, from the checkpoint."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean_ = float(mean)
        self.std_ = float(std) or 1.0

    def transform(self, Y) -> np.ndarray:
        return ((np.asarray(Y, dtype=np.float64) - self.mean_) / self.std_).astype(
            np.float32
        )

    def inverse(self, Y) -> np.ndarray:
        return np.asarray(Y, dtype=np.float64) * self.std_ + self.mean_


class _TrendBasis(nn.Module):
    """Polynomial basis of degree `p` over normalised time.

    Constrains a block to explain only smooth, low-order movement -- what makes
    the interpretable N-BEATS configuration interpretable, since whatever the
    trend stack cannot express is passed on rather than absorbed.
    """

    def __init__(self, degree: int, backcast_size: int, forecast_size: int) -> None:
        super().__init__()
        self.degree = degree
        t_back = np.linspace(-1.0, 0.0, backcast_size, dtype=np.float32)
        t_fore = np.linspace(0.0, 1.0, forecast_size, dtype=np.float32)
        self.register_buffer(
            "back", torch.tensor(np.stack([t_back**i for i in range(degree + 1)]))
        )
        self.register_buffer(
            "fore", torch.tensor(np.stack([t_fore**i for i in range(degree + 1)]))
        )

    @property
    def theta_dim(self) -> int:
        return 2 * (self.degree + 1)

    def forward(self, theta):
        d = self.degree + 1
        return theta[:, :d] @ self.back, theta[:, d:] @ self.fore


class _GenericBasis(nn.Module):
    """Unconstrained linear maps from theta to backcast and forecast."""

    def __init__(self, backcast_size: int, forecast_size: int, theta_size: int = 32) -> None:
        super().__init__()
        self.theta_size = theta_size
        self.back = nn.Linear(theta_size, backcast_size, bias=False)
        self.fore = nn.Linear(theta_size, forecast_size, bias=False)

    @property
    def theta_dim(self) -> int:
        return self.theta_size

    def forward(self, theta):
        return self.back(theta), self.fore(theta)


class _NBeatsBlock(nn.Module):
    """Four-layer ReLU MLP producing basis coefficients.

    `exog_dim > 0` widens the input with the summarised exogenous panel (the
    NBEATSx extension). The basis stays univariate: the 25 feature channels
    inform *how much* of each basis function to use, they never enter the
    output space.
    """

    def __init__(self, input_size: int, exog_dim: int, width: int, basis, dropout: float) -> None:
        super().__init__()
        self.basis = basis
        layers: "list[nn.Module]" = []
        in_dim = input_size + exog_dim
        for _ in range(4):
            layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = width
        self.mlp = nn.Sequential(*layers)
        self.theta = nn.Linear(width, basis.theta_dim)

    def forward(self, x, exog=None):
        h = x if exog is None else torch.cat([x, exog], dim=1)
        return self.basis(self.theta(self.mlp(h)))


class NBeats(nn.Module):
    """Doubly-residual stack of fully connected blocks (Oreshkin et al. 2020).

    Block *k* receives what its predecessors could not explain (backcast
    subtracted from the input) and the global forecast is the sum of every
    block's contribution. Reproduced here verbatim from `mshift.deep` so the
    saved `state_dict` restores onto the same parameter names -- a mismatch
    raises rather than loading a differently-shaped model.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()

        for kind in config["stacks"]:
            shared = None
            for _ in range(config["blocks_per_stack"]):
                if config["share_weights"] and shared is not None:
                    self.blocks.append(shared)
                    continue
                if kind == "trend":
                    basis = _TrendBasis(
                        config["trend_degree"], config["lookback"], config["horizon"]
                    )
                elif kind == "generic":
                    basis = _GenericBasis(
                        config["lookback"], config["horizon"], config["generic_theta"]
                    )
                else:
                    raise ValueError(f"unknown stack kind {kind!r}")
                block = _NBeatsBlock(
                    config["lookback"],
                    config["exog_dim"],
                    config["width"],
                    basis,
                    config["dropout"],
                )
                self.blocks.append(block)
                shared = block

    def forward(self, y_hist, exog=None):
        residual = y_hist
        forecast = y_hist.new_zeros(y_hist.shape[0], self.config["horizon"])
        for block in self.blocks:
            backcast, block_forecast = block(residual, exog)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class Seed:
    """One trained seed: weights plus the scalers that make them mean anything.

    Reloading weights without the scalers gives a model that predicts in the
    wrong units and fails silently, which is why the checkpoint stores both and
    why this class refuses to exist without them.
    """

    def __init__(self, model: NBeats, target_scaler: TargetScaler, exog_scaler, seed: int) -> None:
        self.model = model
        self.target_scaler = target_scaler
        self.exog_scaler = exog_scaler
        self.seed = seed


def build_exog(X: np.ndarray) -> np.ndarray:
    """Static exogenous vector per window: the anchor bar plus the window mean.

    50 numbers summarising 25 channels over 20 bars -- deliberately modest, the
    alternative being to hand a 6k-row training problem the full panel through
    a dense layer, which is how these architectures start memorising sessions.
    """
    X = np.asarray(X, dtype=np.float32)
    return np.concatenate([X[:, -1, :], X.mean(axis=1)], axis=1)


@torch.no_grad()
def predict_point(seed: Seed, X: np.ndarray, target_index: int) -> np.ndarray:
    """One seed's `(n, horizon)` momentum forecast, in the original units."""
    seed.model.eval()
    hist = torch.tensor(seed.target_scaler.transform(X[:, :, target_index]))
    exog = None
    if seed.exog_scaler is not None:
        e = build_exog(X)
        exog = torch.tensor(seed.exog_scaler.transform(e[:, None, :])[:, 0, :])
    return seed.target_scaler.inverse(seed.model(hist, exog).cpu().numpy())


def ensemble_forecast(seeds: "list[Seed]", X: np.ndarray, target_index: int) -> np.ndarray:
    """The seed-averaged point forecast. This, not any one seed, is the model."""
    X = np.asarray(X, dtype=np.float32)
    return np.mean([predict_point(s, X, target_index) for s in seeds], axis=0)


# --- the persistence bridge (mirrors mshift.forecast) ------------------------

def bootstrap_paths(
    point: np.ndarray, residuals: np.ndarray, n_paths: int = N_PATHS, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """`(n, n_paths, horizon)` futures: the point forecast plus residual rows.

    Whole rows are drawn, never individual cells -- see the module docstring.
    """
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(residuals), size=(len(point), n_paths))
    return point[:, None, :] + residuals[pick]


def forward_regime_path(mom_path: np.ndarray, start_state, enter: float, exit_: float) -> np.ndarray:
    """Replay the Schmitt trigger forward over forecast momentum paths.

    Vectorised over rows, sequential over the horizon -- the trigger is a state
    machine and cannot be unrolled. Mirrors
    `persistence_model._hysteresis_regimes` minus the session-start reset: a
    forecast path never crosses a session boundary by construction.
    """
    mom_path = np.atleast_2d(np.asarray(mom_path, dtype=np.float64))
    state = np.broadcast_to(np.asarray(start_state), (mom_path.shape[0],)).copy().astype(np.int8)
    out = np.empty(mom_path.shape, dtype=np.int8)

    for h in range(mom_path.shape[1]):
        m = mom_path[:, h]
        up = state == 1
        down = state == -1
        flat = ~(up | down)

        nxt = state.copy()
        # from +1: leave only below exit_, and jump straight to -1 below -enter
        leaving_up = up & (m < exit_)
        nxt[leaving_up] = np.where(m[leaving_up] < -enter, -1, 0)
        # from -1: mirror image
        leaving_down = down & (m > -exit_)
        nxt[leaving_down] = np.where(m[leaving_down] > enter, 1, 0)
        # from 0: enter on a big enough push
        nxt[flat & (m > enter)] = 1
        nxt[flat & (m < -enter)] = -1

        state = nxt
        out[:, h] = state
    return out


def post_dwell_from_path(mom_path: np.ndarray, start_regime, enter: float, exit_: float) -> np.ndarray:
    """Forward run length implied by a path, counting the anchor bar itself:
    a regime that dies on the very next bar has `post_dwell == 1`."""
    states = forward_regime_path(mom_path, start_regime, enter, exit_)
    start = np.broadcast_to(np.asarray(start_regime), (states.shape[0],))
    broken = states != start[:, None]
    first_break = np.where(broken.any(axis=1), broken.argmax(axis=1), states.shape[1])
    return (first_break + 1).astype(np.int32)


def persistence_probability(
    samples: np.ndarray, start_regime, pre_dwell, momentum: dict, min_dwell: int
) -> dict:
    """Turn `(n, paths, horizon)` sampled momentum paths into a probability.

    Returns both halves separately, because they are different questions and
    the project's headline result is that they have different answers:

        p_post   P(the new regime survives `min_dwell` bars) -- the hard half
        p_full   the complete label, with the observable `pre_dwell` gate
    """
    samples = np.asarray(samples, dtype=np.float64)
    n, n_paths, horizon = samples.shape
    if horizon < min_dwell - 1:
        raise ValueError(f"horizon {horizon} is too short to decide post_dwell >= {min_dwell}")

    enter = float(momentum["enter_threshold"])
    exit_ = float(momentum["exit_threshold"])
    dwell = post_dwell_from_path(
        samples.reshape(n * n_paths, horizon),
        np.repeat(np.asarray(start_regime), n_paths),
        enter,
        exit_,
    )
    survives = (dwell >= min_dwell).reshape(n, n_paths)

    p_post = survives.mean(axis=1)
    gate = (np.asarray(pre_dwell, dtype=np.float64) >= min_dwell).astype(float)
    return {
        "p_post": p_post,
        "p_full": p_post * gate,
        "pre_dwell_gate": gate,
        "expected_post_dwell": dwell.reshape(n, n_paths).mean(axis=1),
    }


def turn_probability(
    samples: np.ndarray, start_regime, current_dwell, momentum: dict, min_dwell: int
) -> dict:
    """`P(the regime turns positive on the next bar and stays there)`.

    The forward-shifted twin of `persistence_probability`. That one is anchored
    on a bar where the change has already happened and asks whether the new
    regime survives; this one is anchored on the bar *before* -- the regime is
    still negative or balanced -- and asks whether the change is about to
    happen at all. It is the question Apple Trader's `anticipate` entry mode is
    built on, and the reason that mode needs a forecaster: a classifier trained
    on change bars has nothing to say about a bar that is not one.

    "Stays there" is the whole horizon rather than `min_dwell` bars, and the two
    coincide by construction: the forecast is `horizon` bars long and
    `min_dwell` is 15 of them, so a path that is positive from the first
    forecast bar to the last has exactly cleared the dwell the persistence
    label demands. A turn any *later* in the horizon cannot be checked for
    survival inside it, which is why this asks only about the next bar -- see
    the `anticipate` note in `agent_stonks.apple_trader`.

    The gate is the same observable half of the persistence label, evaluated
    one bar earlier: `current_dwell` is how long the regime being left has run
    as of the anchor bar, which is precisely the `pre_dwell` the change bar
    would report. So both probabilities carry the same gate and mean the same
    kind of thing, which is what lets one threshold serve both modes.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n, n_paths, horizon = samples.shape
    if horizon < min_dwell:
        raise ValueError(
            f"horizon {horizon} is too short to see a turn hold for {min_dwell} bars"
        )

    states = forward_regime_path(
        samples.reshape(n * n_paths, horizon),
        np.repeat(np.asarray(start_regime), n_paths),
        float(momentum["enter_threshold"]),
        float(momentum["exit_threshold"]),
    )
    holds = (states == 1).all(axis=1).reshape(n, n_paths)

    p_turn = holds.mean(axis=1)
    gate = (np.asarray(current_dwell, dtype=np.float64) >= min_dwell).astype(float)
    return {
        "p_turn": p_turn,
        "p_full": p_turn * gate,
        "pre_dwell_gate": gate,
    }


# --- loading -----------------------------------------------------------------

def model_path() -> Path:
    """Where the saved checkpoint is expected to live."""
    return Path(os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH)


def residuals_path(path: "Path | None" = None) -> Path:
    """The sidecar beside the checkpoint, written by the export script."""
    path = Path(path or model_path())
    return path.with_name(f"{path.stem}_residuals.npz")


def metadata_path(path: "Path | None" = None) -> Path:
    """The sidecar JSON `mshift.deep.save_bundle` writes with the weights."""
    return Path(path or model_path()).with_suffix(".json")


def _restore_seed(payload: dict) -> Seed:
    config = dict(payload["config"])
    config["stacks"] = tuple(config["stacks"])
    module = NBeats(config)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    target = payload.get("target_scaler")
    exog = payload.get("exog_scaler")
    if target is None:
        raise ValueError("checkpoint has no target scaler; this is a classification head")
    return Seed(
        model=module,
        target_scaler=TargetScaler(target["mean"], target["std"]),
        exog_scaler=None if exog is None else SeqStandardizer(exog["mean"], exog["std"]),
        seed=int(payload.get("seed", 0)),
    )


def _load_residuals(path: Path, feature_columns: "list[str]", horizon: int) -> np.ndarray:
    """The exported training residuals, refused unless they match these weights.

    The rows are what turns a point forecast into a fan, so a stale set would
    not raise anything -- it would just widen or narrow every probability the
    model reports. The checks below are the only thing standing between that
    and a plausible wrong number.
    """
    with np.load(path, allow_pickle=False) as data:
        residuals = np.asarray(data["residuals"], dtype=np.float64)
        meta = json.loads(str(data["meta"]))
    if residuals.ndim != 2 or residuals.shape[1] != horizon:
        raise ValueError(
            f"residuals are {residuals.shape}, expected (n, {horizon}) for this checkpoint"
        )
    if list(meta.get("feature_columns") or []) != list(feature_columns):
        raise ValueError("residuals were exported against different feature columns")
    return residuals


def load_bundle() -> "dict | None":
    """The N-BEATS model in the shape `persistence_model.read_latest` expects.

    Same keys as the incumbent joblib bundle (`feature_columns`, `seq_len`,
    `threshold`, `settings`) plus a `score` callable in place of its `pipeline`
    -- which is the whole reason Apple Trader can swap one for the other
    without a branch anywhere in its rules.

    None when the checkpoint, the residual sidecar or the metadata is missing
    or does not agree with itself; the caller reports that rather than trading
    on a model it could not fully assemble. Cached per path.
    """
    path = model_path()
    with _lock:
        if _cache["path"] == path:
            return _cache["bundle"]
        _cache.update(path=path, bundle=None)
        try:
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            seeds = [_restore_seed(p) for p in checkpoint["seeds"]]
        except (OSError, KeyError, ValueError, RuntimeError, TypeError):
            return None
        if not seeds:
            return None

        first = checkpoint["seeds"][0]
        feature_columns = list(first["feature_columns"])
        seq_len = int(first["seq_len"])
        horizon = int(first["horizon"])
        settings = dict(first.get("settings") or {})
        try:
            target_index = feature_columns.index("f_mom")
            residuals = _load_residuals(residuals_path(path), feature_columns, horizon)
        except (OSError, KeyError, ValueError):
            return None

        meta = {}
        try:
            meta = json.loads(metadata_path(path).read_text())
        except (OSError, ValueError):
            pass
        metrics = ((meta.get("metrics") or {}).get("persistence")) or {}

        momentum = {
            **persistence_model.MOMENTUM_DEFAULTS,
            **(settings.get("momentum") or {}),
        }
        min_dwell = int(
            (settings.get("persistence") or {}).get(
                "min_dwell", PERSISTENCE_DEFAULTS["min_dwell"]
            )
        )
        i_regime = feature_columns.index("f_regime")
        i_dwell = feature_columns.index("f_prev_dwell")
        i_in_regime = feature_columns.index("f_bars_in_regime")

        def _sampled_paths(X: np.ndarray) -> np.ndarray:
            return bootstrap_paths(
                ensemble_forecast(seeds, X, target_index), residuals, N_PATHS, BOOTSTRAP_SEED
            )

        def score(X) -> np.ndarray:
            """`(n, seq_len, n_features)` sequences -> persistence probability.

            `regime` and `pre_dwell` are recovered from the sequence itself --
            `f_regime` is the regime and `f_prev_dwell` is `log1p` of the dwell
            as of the previous bar, which on a change bar *is* `pre_dwell`. So
            the scorer needs nothing beyond the array it is handed, and cannot
            reach for a column of the future even by accident.
            """
            X = np.asarray(X, dtype=np.float32)
            if len(X) == 0:
                return np.array([])
            result = persistence_probability(
                _sampled_paths(X),
                np.rint(X[:, -1, i_regime]).astype(int),
                np.expm1(X[:, -1, i_dwell]),
                momentum,
                min_dwell,
            )
            return np.clip(result["p_full"], 0.0, 1.0)

        def score_turn(X) -> np.ndarray:
            """`(n, seq_len, n_features)` sequences -> P(turns positive next bar).

            The `anticipate` half of the interface, and the reason the bundle
            carries two callables: the anchor here is a bar whose regime is
            still negative or balanced, so `score` above -- which asks whether
            the regime the anchor is *in* survives -- would be answering the
            wrong question entirely.

            Like `score`, everything comes out of the array itself:
            `f_bars_in_regime` is `log1p` of how long the current regime has
            run as of the anchor bar, which is the `pre_dwell` the change bar
            would report. Rows the gate closes are returned as a hard zero
            without forecasting them -- the same answer the full computation
            gives, for none of the five networks and 500 paths, which is what
            keeps a per-bar cycle affordable when most bars cannot qualify.
            """
            X = np.asarray(X, dtype=np.float32)
            if len(X) == 0:
                return np.array([])
            dwell = np.expm1(X[:, -1, i_in_regime])
            open_gate = dwell >= min_dwell
            out = np.zeros(len(X), dtype=float)
            if not open_gate.any():
                return out
            candidates = X[open_gate]
            result = turn_probability(
                _sampled_paths(candidates),
                np.rint(candidates[:, -1, i_regime]).astype(int),
                dwell[open_gate],
                momentum,
                min_dwell,
            )
            out[open_gate] = np.clip(result["p_full"], 0.0, 1.0)
            return out

        bundle = {
            "score": score,
            "score_turn": score_turn,
            "feature_columns": feature_columns,
            "seq_len": seq_len,
            "horizon": horizon,
            "n_seeds": len(seeds),
            "n_paths": N_PATHS,
            "n_residual_rows": int(len(residuals)),
            # `p_full` is a survival probability multiplied by a hard gate, not
            # a calibrated posterior: 0.5 means nothing on it, and roughly half
            # the events are exactly zero because the gate closed. The cut-off
            # is the one notebook 07 grid-searched on the validation events.
            "threshold": float(metrics.get("threshold", 0.05)),
            "settings": settings,
            "metrics": metrics,
            "forecast_metrics": (meta.get("metrics") or {}).get("forecast") or {},
            "trained_at": ", ".join(meta.get("train_sessions") or []) or "unknown",
            "excluded_sessions_from": meta.get("excluded_sessions_from", "?"),
        }
        _cache["bundle"] = bundle
        return bundle


def reset_bundle_cache() -> None:
    """Drop the cached bundle (used by tests that swap the checkpoint)."""
    with _lock:
        _cache.update(path=None, bundle=None)
