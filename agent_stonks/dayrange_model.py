"""Where the day's high and low will land, called at 9:35 (TimeToChange3).

Apple Trader's other two models answer a question about the *next fifteen
minutes*: a momentum regime just turned (or is about to), will it hold? This
one answers a question about the *whole session*, once, and then says nothing
else all day -- how far above and below yesterday's average price will today's
high and low print?

    prev_avg = (prev_open + prev_close) / 2
    y_high   = log(today_high / prev_avg)
    y_low    = log(today_low  / prev_avg)

The forecast is made from the daily history up to yesterday plus the first five
1-minute bars of this morning, and it exists from 9:35 onwards. Nothing about
it updates intraday, which is why the rules built on it are resting levels
rather than a per-bar signal -- see `apple_trader.DayRangeTrader`.

What "best model" means here
----------------------------
TimeToChange3 fitted three predictors on the same daily features and blends
them equally, then adds a second stage and a constraint. All four parts are in
the saved bundle and all four are loaded:

1. **LightGBM**, **N-BEATS** and **N-HiTS** each predict both targets from the
   daily table (and, for the two networks, a 32-day lookback of eight per-day
   channels). Equal-weight blend.
2. An **opening ridge** learns what the first five minutes add on top of the
   daily prediction, fitted to its residuals over the 20 sessions where minute
   data existed.
3. The prediction is **clipped to contain the observed 5-minute range** -- the
   day's high cannot come in below a high that has already printed.

On a 129-session test window the blend's mean absolute error is 0.0077 in log
units ($2.12), against 0.0110 for a 14-day rolling baseline and 0.0135 for
persistence -- 30% of the baseline's error removed, and the ordering holds
across all four walk-forward refits. That is the number worth quoting, and it
is a statement about the *width* of the day. Notebook 3 showed direction is not
predictable here before any model was fitted, and the feature importances
agreed afterwards; the trading rule is built as a mean-reversion bet for
exactly that reason.

The blend is the model, so a missing PyTorch makes this bundle unavailable
rather than quietly degrading to LightGBM alone -- two of the three voters
missing is a different predictor with different error, not the same one with a
smaller install.

The mirror contract
-------------------
Everything below `--- features` is a verbatim copy of `dayrange/features.py`
and the inference half of `dayrange/modeling.py` from
`FinNotebooks/TimeToChange3`. It has to be: the saved model is a function of
those exact column definitions, and a mirror that drifts produces confident
numbers off a different feature. If `dayrange` changes, retrain **and** update
this module -- the same contract `persistence_model` has with `mshift` and
`profile_model` has with `levelsml`.

Two consequences of that copy being here rather than imported:

* **The pickle is stamped `dayrange.modeling`.** `_register_unpickle_alias()`
  installs a stub module pointing at the mirrors of `LGBMRange` and
  `OpeningCorrection` below, so `joblib.load` resolves without importing the
  real package (which would drag matplotlib, plotly and yfinance into the app).
  If the real `dayrange` is genuinely importable, it wins.
* **LightGBM is imported before torch.** Both wheels bundle their own OpenMP
  runtime and this environment has no system `libomp`; loading a LightGBM model
  in a process that imported torch first segfaults with no traceback. This
  bundle needs both in one process, which no other model here does, so the
  ordering matters more than it does in `nbeats_model` -- same fix.

What the live path has to supply, and where it can go wrong
-----------------------------------------------------------
Unlike every other model in this app, this one does not run on today's tape
alone. It needs roughly **a year of daily bars** behind the session (the
longest window is the 252-day distance-from-extremes, and 126-day momentum and
the 32-day sequence sit inside it), which arrives via
`historical.fetch_daily_ohlc_bars` -- unadjusted, so it is on the same raw
price scale the model was fitted on. `require_history` refuses a short history
rather than predicting off a table full of NaNs.

The one input whose *scale* depends on the tape is `or_volume_share`, the
opening five minutes' volume against a 14-day daily average. The model was
fitted on consolidated (yfinance) minute volume. On Alpaca's IEX feed a minute
bar carries roughly 4% of that, which moves the feature by about log(0.04) =
-3.2 -- far outside anything it saw in training. `volume_scale_warning`
reports it; the fix is to run the live stream and any SimLab dataset on a
consolidated tape (`yfinance` or `sip`), not to patch the number.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pandas as pd

# See the module docstring: LightGBM must be imported before torch in this
# process or loading the LightGBM half of the bundle segfaults outright. Both
# wheels also honour OMP_NUM_THREADS, and capping it before either is loaded is
# what `dayrange/__init__.py` does in the notebooks for the same reason.
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# The import order above protects LightGBM and *arms the opposite trap*: with
# both OpenMP runtimes loaded and LightGBM's first, the next torch op that goes
# parallel takes the process down instead. It is not exotic -- `MaxPool1d` on a
# (1, 8, 32) tensor is enough, which is exactly what an N-HiTS block does on
# every forecast. Unlike the import order this one cannot be fixed by being
# careful here, because `nbeats_model` may already have imported both by the
# time anything asks for this model, so the cap is applied at runtime where it
# works regardless of who imported what.
#
# Process-global, and deliberately so: it is a property of this environment
# having no shared `libomp`, not of this model. The alternative
# (KMP_DUPLICATE_LIB_OK) is documented as unsafe rather than as a fix. Nothing
# here is large enough to miss the threads -- the whole forecast is three small
# networks over a 32x8 window, once a day.
torch.set_num_threads(1)

from . import market_hours  # noqa: E402

# The shared model store next to the AgentStonks checkout (Code/Models), where
# the momentum bundles and the LevelsML pack already live. The path names the
# joblib; the two checkpoints and the metadata sit beside it under the same
# stem, exactly as `dayrange.modeling.save_bundle` wrote them.
#
# One bundle per ticker, because TimeToChange3 fits one per ticker: the same
# pipeline was run over AAPL, GOOGL and INTC and each produced its own daily
# models, its own opening ridge and its own metadata. Nothing here is shared
# between them but the code, so the ticker is an argument rather than a setting
# -- `apple_models.DAYRANGE_TICKERS` is the list, and `apple_trader2` passes the
# one its config names.
MODEL_PATH_ENV = "APPLE_DAYRANGE_MODEL"
DEFAULT_TICKER = "AAPL"
MODEL_DIR = Path(__file__).resolve().parents[2] / "Models"
DEFAULT_MODEL_PATH = MODEL_DIR / f"timetochange3_dayrange_{DEFAULT_TICKER}.joblib"

# How many minutes of the open the forecast is allowed to look at. Not a
# tunable: the opening ridge was fitted on exactly this window.
OPENING_MINUTES = 5

# Calendar days of daily history to ask for. The longest window in the feature
# set is 252 *trading* days, which needs ~370 calendar days; 420 matches what
# SimLab already stores per dataset (`simlab.data.DAILY_LOOKBACK_DAYS`), so the
# live fetch and a replay see the same depth.
DAILY_HISTORY_DAYS = 420

# Trading days that must be present before a forecast is attempted: the
# 252-day extremes window plus today's row, which is the longest thing the
# feature table needs. The 32-day sequence lookback and the 126-day momentum
# both sit inside it, so this one bound covers everything.
#
# It is a real constraint rather than a formality. 420 calendar days is about
# 288 trading days, so the live fetch clears this by a month and no more -- the
# reason it is checked instead of assumed is that `dist_252high` carries
# `min_periods=60` and LightGBM eats NaNs, so a short history would produce a
# confident number off a 60-day extreme where the model was fitted on a
# 252-day one.
MIN_DAILY_SESSIONS = 253

EPS = 1e-12

_lock = threading.Lock()
# Keyed by path rather than a single slot, so a session that runs GOOGL after
# AAPL does not evict and re-load a 1 MB bundle each time it switches. Failures
# are cached under the same key (as None), which is what stops a per-minute loop
# from re-hitting the filesystem for a file that is not there.
_cache: "dict[Path, dict | None]" = {}


# --- features (mirrors dayrange.features) -----------------------------------

SHORT_WINDOWS = (7, 14, 28)
LONG_WINDOWS = (63, 126)

TARGETS = ["y_high", "y_low"]


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _pos_in_range(value, low, high):
    """Where `value` sits inside [low, high]; 0.5 when the range is empty."""
    span = high - low
    return ((value - low) / span.where(span > EPS)).fillna(0.5)


def add_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach the reference price and the two targets to a daily frame."""
    out = daily.copy()
    mid = (out["open"] + out["close"]) / 2
    out["mid"] = mid
    out["prev_avg"] = mid.shift(1)
    out["y_high"] = np.log(_safe_div(out["high"], out["prev_avg"]))
    out["y_low"] = np.log(_safe_div(out["low"], out["prev_avg"]))
    return out


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Features from the daily series, as known at 9:35.

    Every trailing statistic is rolled and then shifted a day, so the only
    same-day inputs are the ones taken from today's 9:30 open (`gap`,
    `gap_vs_adr`, `open_off`) and the calendar (`dow`). That is what makes it
    safe to hand this function a daily frame whose last row is today's,
    partially formed: nothing reads today's high, low or close.
    """
    df = add_targets(daily)
    o, h, l, c, v, mid = (df[k] for k in ["open", "high", "low", "close", "volume", "mid"])

    ret = np.log(mid / mid.shift(1))
    rng_rel = (h - l) / mid
    true_range = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)

    f = pd.DataFrame(index=df.index)

    # --- yesterday -------------------------------------------------------
    f["prev_range"] = rng_rel.shift(1)
    f["prev_true_range"] = (true_range / c.shift(1)).shift(1)
    f["prev_body"] = np.log(c / o).shift(1)
    f["prev_ret"] = ret.shift(1)
    f["prev_close_pos"] = _pos_in_range(c, l, h).shift(1)
    f["prev_gap"] = np.log(o / c.shift(1)).shift(1)
    f["prev_high_off"] = np.log(h / mid.shift(1)).shift(1)  # yesterday's own y_high
    f["prev_low_off"] = np.log(l / mid.shift(1)).shift(1)
    f["prev_volume_z"] = (
        (np.log(v) - np.log(v).rolling(28).mean()) / np.log(v).rolling(28).std()
    ).shift(1)

    # --- 7 / 14 / 28-day averages, from open and close -------------------
    for w in SHORT_WINDOWS:
        avg = mid.rolling(w).mean().shift(1)
        f[f"avg{w}_dist"] = np.log(mid.shift(1) / avg)
        f[f"vol{w}"] = ret.rolling(w).std().shift(1)
        f[f"adr{w}"] = rng_rel.rolling(w).mean().shift(1)
        f[f"high_off{w}"] = np.log(h / mid.shift(1)).rolling(w).mean().shift(1)
        f[f"low_off{w}"] = np.log(l / mid.shift(1)).rolling(w).mean().shift(1)

    f["adr_spread"] = f["adr7"] - f["adr28"]
    f["vol_spread"] = f["vol7"] - f["vol28"]
    f["avg_slope"] = f["avg7_dist"] - f["avg28_dist"]

    # --- long-term momentum ----------------------------------------------
    for w in LONG_WINDOWS:
        f[f"mom{w}"] = np.log(mid / mid.shift(w)).shift(1)
        f[f"vol{w}"] = ret.rolling(w).std().shift(1)
    f["mom28"] = np.log(mid / mid.shift(28)).shift(1)
    f["dist_252high"] = np.log(mid.shift(1) / h.rolling(252, min_periods=60).max().shift(1))
    f["dist_252low"] = np.log(mid.shift(1) / l.rolling(252, min_periods=60).min().shift(1))

    # --- today, known at 9:30 --------------------------------------------
    f["gap"] = np.log(o / c.shift(1))
    f["gap_vs_adr"] = f["gap"].abs() / f["adr14"]
    f["open_off"] = np.log(o / df["prev_avg"])
    f["dow"] = df.index.dayofweek

    # dollar values the trading rule needs, and a volume yardstick for the
    # opening features; not model inputs themselves
    f["prev_avg"] = df["prev_avg"]
    f["adr14_abs"] = (h - l).rolling(14).mean().shift(1)
    f["advol14"] = v.rolling(14).mean().shift(1)

    f[TARGETS] = df[TARGETS]
    return f


DAILY_FEATURE_COLS: list[str] = [
    "prev_range", "prev_true_range", "prev_body", "prev_ret", "prev_close_pos",
    "prev_gap", "prev_high_off", "prev_low_off", "prev_volume_z",
    "avg7_dist", "vol7", "adr7", "high_off7", "low_off7",
    "avg14_dist", "vol14", "adr14", "high_off14", "low_off14",
    "avg28_dist", "vol28", "adr28", "high_off28", "low_off28",
    "adr_spread", "vol_spread", "avg_slope",
    "mom28", "mom63", "vol63", "mom126", "vol126",
    "dist_252high", "dist_252low",
    "gap", "gap_vs_adr", "open_off", "dow",
]

OPENING_FEATURE_COLS: list[str] = [
    "or_high", "or_low", "or_close", "or_ret", "or_range", "or_range_vs_adr",
    "or_up", "or_down", "or_close_pos", "or_ret_std", "or_up_bars", "or_volume_share",
]


def opening_features(opening_bars: pd.DataFrame, daily_row: pd.Series) -> dict:
    """What the first `OPENING_MINUTES` bars of one session say.

    `dayrange.features.opening_features` computes this for every stored session
    at once, grouping a multi-day minute frame by date. Live there is only ever
    one session in hand, so this is the same arithmetic on one group -- the
    column-by-column equivalence is what `tests/test_dayrange_model.py` pins.

    Levels are expressed against yesterday's average price so they stay on the
    same scale as the targets.
    """
    o5 = float(opening_bars["open"].iloc[0])
    h5 = float(opening_bars["high"].max())
    l5 = float(opening_bars["low"].min())
    c5 = float(opening_bars["close"].iloc[-1])
    v5 = float(opening_bars["volume"].sum())

    minute_ret = np.log(opening_bars["close"] / opening_bars["open"])
    ref = float(daily_row["prev_avg"])
    adr = float(daily_row["adr14"])
    # typical volume for a 5-minute slice, from the 14-day daily average
    typical_v5 = float(daily_row["advol14"]) * (len(opening_bars) / 390)

    span = h5 - l5
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_share = np.log(v5 / typical_v5) if v5 > 0 and typical_v5 > 0 else np.nan

    return {
        "or_high": np.log(h5 / ref),
        "or_low": np.log(l5 / ref),
        "or_close": np.log(c5 / ref),
        "or_ret": np.log(c5 / o5),
        "or_range": span / ref,
        "or_range_vs_adr": (span / ref) / adr,
        "or_up": (h5 - o5) / ref,
        "or_down": (o5 - l5) / ref,
        "or_close_pos": (c5 - l5) / span if span > EPS else 0.5,
        "or_ret_std": float(minute_ret.std()),
        "or_up_bars": float((minute_ret > 0).mean()),
        "or_volume_share": volume_share,
        # kept for the constraint and the trading rule, not fed to the model
        "high5": h5, "low5": l5, "close5": c5, "open5": o5,
    }


# --- the sequence models' input (mirrors dayrange.modeling) ------------------

LOOKBACK = 32

SEQ_CHANNELS = [
    "ch_ret", "ch_range", "ch_gap", "ch_body",
    "ch_close_pos", "ch_volume_z", "ch_high_off", "ch_low_off",
]


def channel_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-day channels, all of them known once that day has closed."""
    o, h, l, c, v = (daily[k] for k in ["open", "high", "low", "close", "volume"])
    mid = (o + c) / 2
    logv = np.log(v.replace(0, np.nan))

    ch = pd.DataFrame(index=daily.index)
    ch["ch_ret"] = np.log(mid / mid.shift(1))
    ch["ch_range"] = (h - l) / mid
    ch["ch_gap"] = np.log(o / c.shift(1))
    ch["ch_body"] = np.log(c / o)
    ch["ch_close_pos"] = _pos_in_range(c, l, h)
    ch["ch_volume_z"] = (logv - logv.rolling(28).mean()) / logv.rolling(28).std()
    ch["ch_high_off"] = np.log(h / mid.shift(1))
    ch["ch_low_off"] = np.log(l / mid.shift(1))
    return ch[SEQ_CHANNELS]


def make_sequences(
    ch: pd.DataFrame, dates: pd.DatetimeIndex, lookback: int = LOOKBACK
) -> np.ndarray:
    """Stack of (lookback, channels) windows ending the day *before* each date."""
    pos = {d: i for i, d in enumerate(ch.index)}
    values = ch.to_numpy(dtype="float32")
    out = np.empty((len(dates), lookback, values.shape[1]), dtype="float32")
    for k, d in enumerate(dates):
        i = pos[d]
        if i < lookback:
            raise ValueError(f"{d.date()} has only {i} prior sessions, need {lookback}")
        out[k] = values[i - lookback : i]  # strictly past
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --- the estimators, restore-only (mirrors dayrange.modeling) ----------------

def _mlp(sizes: "list[int]", dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NBeatsBlock(nn.Module):
    """Generic N-BEATS block: shared trunk, one head backcasts, one forecasts."""

    def __init__(self, input_dim: int, hidden: int, theta_dim: int, horizon: int,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.trunk = _mlp([input_dim] + [hidden] * n_layers, dropout)
        self.theta_b = nn.Linear(hidden, theta_dim)
        self.theta_f = nn.Linear(hidden, theta_dim)
        self.to_backcast = nn.Linear(theta_dim, input_dim)
        self.to_forecast = nn.Linear(theta_dim, horizon)

    def forward(self, x):
        h = self.trunk(x)
        return self.to_backcast(self.theta_b(h)), self.to_forecast(self.theta_f(h))


class NHitsBlock(nn.Module):
    """N-HiTS block: pool the lookback, learn on the coarse view, interpolate back.

    One deviation from the paper, and it is the notebook's, not this mirror's:
    the forecast axis here is two numbers rather than a time series, so the
    hierarchical interpolation applies to the backcast only and the forecast
    head is direct. The multi-rate input pooling is unchanged.
    """

    def __init__(self, lookback: int, n_channels: int, hidden: int, pool_kernel: int,
                 horizon: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lookback, self.n_channels, self.pool_kernel = lookback, n_channels, pool_kernel
        self.pool = nn.MaxPool1d(pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_len = int(np.ceil(lookback / pool_kernel))
        self.pooled_dim = pooled_len * n_channels
        self.trunk = _mlp([self.pooled_dim] + [hidden] * n_layers, dropout)
        self.to_backcast = nn.Linear(hidden, self.pooled_dim)  # coarse, then upsampled
        self.to_forecast = nn.Linear(hidden, horizon)

    def forward(self, x):
        b = x.shape[0]
        seq = x.view(b, self.lookback, self.n_channels).transpose(1, 2)  # (B, C, L)
        pooled = self.pool(seq)
        h = self.trunk(pooled.flatten(1))
        coarse = self.to_backcast(h).view(b, self.n_channels, -1)
        backcast = nn.functional.interpolate(
            coarse, size=self.lookback, mode="linear", align_corners=False
        )
        return backcast.transpose(1, 2).reshape(b, -1), self.to_forecast(h)


class SeqNet(nn.Module):
    """Residual stack over the lookback, plus an MLP branch for the tabular features."""

    def __init__(self, kind: str, lookback: int, n_channels: int, n_exo: int,
                 horizon: int = 2, hidden: int = 64, n_blocks: int = 3,
                 theta_dim: int = 16, pool_kernels: "tuple[int, ...]" = (8, 4, 1),
                 dropout: float = 0.1):
        super().__init__()
        self.kind, self.lookback, self.n_channels = kind, lookback, n_channels
        input_dim = lookback * n_channels

        if kind == "nbeats":
            self.blocks = nn.ModuleList(
                [NBeatsBlock(input_dim, hidden, theta_dim, horizon, dropout=dropout)
                 for _ in range(n_blocks)]
            )
        elif kind == "nhits":
            kernels = list(pool_kernels)[:n_blocks] or [1]
            while len(kernels) < n_blocks:
                kernels.append(1)
            self.blocks = nn.ModuleList(
                [NHitsBlock(lookback, n_channels, hidden, k, horizon, dropout=dropout)
                 for k in kernels]
            )
        else:
            raise ValueError(f"unknown kind {kind!r}")

        self.exo = _mlp([n_exo, hidden, hidden], dropout) if n_exo else None
        self.exo_head = nn.Linear(hidden, horizon) if n_exo else None

    def forward(self, seq, exo):
        residual = seq.flatten(1)
        forecast = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        if self.exo is not None:
            forecast = forecast + self.exo_head(self.exo(exo))
        return forecast


class SeqRegressor:
    """A restored SeqNet plus its scalers. Inference only -- `fit` lives in the
    notebook package and nothing here retrains."""

    def __init__(self, **params):
        self.params = dict(params)
        self.net: "SeqNet | None" = None

    def _prep(self, seq, exo):
        return (
            torch.tensor((seq - self.seq_mean) / self.seq_std, dtype=torch.float32),
            torch.tensor((exo - self.exo_mean) / self.exo_std, dtype=torch.float32),
        )

    def predict(self, seq, exo) -> np.ndarray:
        self.net.eval()
        xs, xe = self._prep(np.asarray(seq, "float32"), np.asarray(exo, "float32"))
        with torch.no_grad():
            out = self.net(xs, xe).numpy()
        return out * self.y_std + self.y_mean


class LGBMRange:
    """One LightGBM per target.

    Mirrors `dayrange.modeling.LGBMRange` for unpickling: the fitted state is
    `models` and `feature_names`, both plain attributes, so the saved object
    restores onto this definition exactly.
    """

    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        self.models: dict = {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.models[t].predict(X[self.feature_names]) for t in TARGETS])


class OpeningCorrection:
    """Ridge on the opening-range features, fitted to the daily model's residuals.

    Mirrors `dayrange.modeling.OpeningCorrection` for unpickling; `models` and
    `cols` are the whole of its fitted state.
    """

    def __init__(self, alpha: float = 10.0, cols: "list[str] | None" = None):
        self.alpha = alpha
        self.cols = cols or OPENING_FEATURE_COLS
        self.models: dict = {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.models[t].predict(X[self.cols]) for t in TARGETS])

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {t: m.named_steps["ridge"].coef_ for t, m in self.models.items()}, index=self.cols
        )


def apply_open_constraint(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """The day's high cannot be below what already printed in the first 5 minutes."""
    out = pred.copy()
    out[:, 0] = np.maximum(out[:, 0], frame["or_high"].to_numpy())
    out[:, 1] = np.minimum(out[:, 1], frame["or_low"].to_numpy())
    return out


class DayRangeModel:
    """Daily blend + opening correction + the constraint, as one predictor."""

    def __init__(self, daily_models: dict, weights: dict, correction, use_constraint: bool,
                 lookback: int, daily_cols: "list[str]", metadata: dict):
        self.daily_models = daily_models
        self.weights = weights
        self.correction = correction
        self.use_constraint = use_constraint
        self.lookback = lookback
        self.daily_cols = list(daily_cols)
        self.metadata = metadata

    def predict_daily(self, X: pd.DataFrame, seq: "np.ndarray | None" = None) -> np.ndarray:
        total, wsum = 0.0, 0.0
        for name, model in self.daily_models.items():
            w = self.weights.get(name, 0.0)
            if w == 0:
                continue
            p = (
                model.predict(seq, X[self.daily_cols].to_numpy())
                if isinstance(model, SeqRegressor)
                else model.predict(X[self.daily_cols])
            )
            total, wsum = total + w * np.asarray(p), wsum + w
        return total / wsum

    def predict(self, X: pd.DataFrame, seq: "np.ndarray | None" = None) -> np.ndarray:
        pred = self.predict_daily(X, seq)
        if self.correction is not None:
            pred = pred + self.correction.predict(X)
        if self.use_constraint and "or_high" in X.columns:
            pred = apply_open_constraint(pred, X)
        return pred

    def predict_prices(self, X: pd.DataFrame, seq: "np.ndarray | None" = None) -> pd.DataFrame:
        pred = self.predict(X, seq)
        ref = X["prev_avg"].to_numpy()
        return pd.DataFrame(
            {
                "pred_high_rel": pred[:, 0],
                "pred_low_rel": pred[:, 1],
                "pred_high": ref * np.exp(pred[:, 0]),
                "pred_low": ref * np.exp(pred[:, 1]),
                "prev_avg": ref,
            },
            index=X.index,
        )


# --- the saved bundle --------------------------------------------------------

def _register_unpickle_alias() -> None:
    """Make `dayrange.modeling.{LGBMRange,OpeningCorrection}` resolvable for joblib.

    The bundle was pickled from the notebooks' own package, so both classes are
    stamped with their module path there. Importing the real `dayrange` would
    pull torch, lightgbm, matplotlib and yfinance in through its `__init__`, so
    a stub module pointing at the mirrors above stands in -- unless the real
    package is genuinely importable, in which case it wins.
    """
    if "dayrange.modeling" in sys.modules:
        return
    try:
        __import__("dayrange.modeling")
        return
    except Exception:
        sys.modules.pop("dayrange", None)
    package = types.ModuleType("dayrange")
    package.__path__ = []  # a package, so "dayrange.modeling" is a legal submodule
    module = types.ModuleType("dayrange.modeling")
    module.LGBMRange = LGBMRange
    module.OpeningCorrection = OpeningCorrection
    module.SeqRegressor = SeqRegressor
    module.SeqNet = SeqNet
    package.modeling = module
    sys.modules["dayrange"] = package
    sys.modules["dayrange.modeling"] = module


def model_path(ticker: str = DEFAULT_TICKER) -> Path:
    """Where one ticker's saved joblib is expected to live.

    One file per ticker: TimeToChange3 fits the whole pipeline per symbol and
    makes no claim that one transfers to another, so `timetochange3_dayrange_
    GOOGL.joblib` is a different model rather than the same one pointed
    elsewhere.

    Two env overrides, and the difference matters. `APPLE_DAYRANGE_MODEL_<TICKER>`
    relocates one ticker's bundle. The bare `APPLE_DAYRANGE_MODEL` names a single
    file, so it can only mean the default ticker's -- letting it answer for every
    symbol would hand a GOOGL run the AAPL model without saying so.
    """
    symbol = (ticker or DEFAULT_TICKER).upper()
    override = os.environ.get(f"{MODEL_PATH_ENV}_{symbol}")
    if not override and symbol == DEFAULT_TICKER:
        override = os.environ.get(MODEL_PATH_ENV)
    return Path(override or MODEL_DIR / f"timetochange3_dayrange_{symbol}.joblib")


def checkpoint_path(kind: str, path: "Path | None" = None) -> Path:
    """Where one sequence model's weights sit, beside the joblib."""
    base = path or model_path()
    return base.with_name(f"{base.stem}_{kind}.pt")


def metadata_path(path: "Path | None" = None) -> Path:
    base = path or model_path()
    return base.with_suffix(".json")


def _restore_seq(path: Path) -> SeqRegressor:
    blob = torch.load(path, weights_only=False)
    reg = SeqRegressor(**blob["params"])
    reg.net = SeqNet(
        kind=blob["params"]["kind"], lookback=blob["params"]["lookback"],
        n_channels=blob["n_channels"], n_exo=blob["n_exo"], horizon=2,
        hidden=blob["params"]["hidden"], n_blocks=blob["params"]["n_blocks"],
        dropout=blob["params"]["dropout"],
    )
    reg.net.load_state_dict(blob["state_dict"])
    reg.net.eval()
    for k in ("seq_mean", "seq_std", "exo_mean", "exo_std", "y_mean", "y_std"):
        setattr(reg, k, blob[k])
    return reg


def load_bundle(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    assembled.

    Refuses rather than degrades in three cases, each of which would otherwise
    produce a forecast that looks fine and is not the model that was measured:

    * a missing joblib, or a joblib that is not this bundle's shape;
    * a missing N-BEATS or N-HiTS checkpoint -- the published error is the
      three-way blend's, and dropping a voter silently changes it;
    * a checkpoint whose weights refuse to load onto these definitions, which
      is what a drifted mirror looks like from here.

    Cached per path after the first load, including the failure, so a loop that
    asks every minute doesn't re-hit the filesystem.
    """
    path = model_path(ticker)
    with _lock:
        if path in _cache:
            return _cache[path]
        _cache[path] = None
        try:
            import joblib
        except ImportError:
            return None
        _register_unpickle_alias()
        try:
            sk = joblib.load(path)
        except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError):
            return None
        if not isinstance(sk, dict) or {"lgbm", "weights", "daily_cols", "lookback"} - set(sk):
            return None

        models = {"lgbm": sk["lgbm"]}
        try:
            for kind in ("nbeats", "nhits"):
                models[kind] = _restore_seq(checkpoint_path(kind, path))
        except (OSError, KeyError, RuntimeError, ValueError, AttributeError):
            return None

        meta_file = metadata_path(path)
        try:
            metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        except (OSError, ValueError):
            metadata = {}

        model = DayRangeModel(
            daily_models=models, weights=sk["weights"], correction=sk.get("correction"),
            use_constraint=bool(sk.get("use_constraint", True)), lookback=int(sk["lookback"]),
            daily_cols=sk["daily_cols"], metadata=metadata,
        )
        _cache[path] = {
            "model": model,
            "metadata": metadata,
            "daily_models": list(models),
            "opening_minutes": int(metadata.get("opening_minutes") or OPENING_MINUTES),
            "lookback": model.lookback,
            "trained_at": metadata.get("created"),
            "path": str(path),
            # Which symbol this bundle was fitted on, as the file itself
            # recorded it -- not the ticker that asked for it. A mismatch is
            # worth being able to see rather than inferring from the filename.
            "ticker": str(metadata.get("ticker") or "").upper() or None,
        }
        return _cache[path]


def reset_bundle_cache() -> None:
    """Drop every cached bundle (used by tests that swap the model file)."""
    with _lock:
        _cache.clear()


def opening_minutes(bundle: "dict | None" = None) -> int:
    """The opening window the bundle's ridge was fitted on."""
    try:
        return int((bundle or {})["opening_minutes"])
    except (KeyError, TypeError, ValueError):
        return OPENING_MINUTES


# --- assembling one session's inputs -----------------------------------------

def daily_frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca/yfinance `{"t","o","h","l","c","v"}` daily bars -> the notebook's
    daily frame: a tz-naive midnight index named `date`, lower-case OHLCV.

    Rows dated on or after `through` are the caller's problem, not this
    function's -- see `session_daily_frame`, which is the only place today's row
    is allowed in and builds it from the minute tape instead.
    """
    columns = ["open", "high", "low", "close", "volume"]
    if not bars:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([str(b.get("t", ""))[:10] for b in bars]),
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b.get("v") or 0.0) for b in bars],
        }
    )
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return frame.sort_values("date").set_index("date")[columns]


def session_daily_frame(
    history: pd.DataFrame,
    opening_bars: pd.DataFrame,
    session_date,
    open_price: "float | None" = None,
) -> pd.DataFrame:
    """`history` with today appended, today's row assembled from what is known.

    Today's daily bar does not exist yet at 9:35, and `daily_features` needs a
    row for today anyway -- but it reads exactly one field out of it. Every
    trailing statistic there is rolled and then shifted a day, so the only
    same-day inputs are `gap`, `gap_vs_adr` and `open_off`, and all three are
    functions of today's **open** alone. The high, low and close carried here
    are the first five minutes', which the model is entitled to see and which
    nothing reads; the row being partial therefore cannot leak the day's
    outcome into the forecast.

    `open_price` is the official opening print when the caller could get one
    (`historical.fetch_session_open`). Without it the first regular-session
    minute bar's open stands in, which is the same number on about half of all
    sessions and within a few basis points otherwise -- the auction print and
    the first trade the feed happened to see. Measured on 2026-08-07 the
    substitution moves the predicted high from 318.16 to 318.10, against the
    model's own $2.12 mean error, so it is a fallback worth having rather than
    a reason to refuse.
    """
    day = pd.Timestamp(session_date).normalize()
    prior = history[history.index < day]
    first_open = float(opening_bars["open"].iloc[0])
    today = pd.DataFrame(
        {
            "open": [float(open_price) if open_price else first_open],
            "high": [float(opening_bars["high"].max())],
            "low": [float(opening_bars["low"].min())],
            "close": [float(opening_bars["close"].iloc[-1])],
            "volume": [float(opening_bars["volume"].sum())],
        },
        index=pd.DatetimeIndex([day], name="date"),
    )
    return pd.concat([prior, today])


def require_history(daily: pd.DataFrame) -> "str | None":
    """Why this daily frame cannot support a forecast, or None if it can.

    Checked before anything is computed, because the failure it prevents is the
    quiet one: `dist_252high` carries `min_periods=60` and LightGBM eats NaNs,
    so a short history produces a number rather than an error -- one built off
    a 60-day extreme where the model was fitted on a 252-day one, and off a
    ridge standardising features it never saw at that scale.
    """
    if len(daily) < MIN_DAILY_SESSIONS:
        return (
            f"only {len(daily)} daily sessions of history; the forecast needs "
            f"{MIN_DAILY_SESSIONS} for its 252-day distance-from-extremes window."
        )
    return None


def volume_scale_warning(feed: "str | None") -> "str | None":
    """Whether this tape's minute volume is on the scale the ridge was fitted on.

    `or_volume_share` is the only feature that reads minute *volume*, and the
    model saw consolidated volume. Alpaca's IEX feed is one venue at roughly 4%
    of it, which shifts the feature by about -3.2 in log space -- outside the
    range the ridge was fitted over, so its contribution stops being a
    correction and becomes a constant bias. SIP and yfinance are consolidated
    and fine.
    """
    if str(feed or "").lower() != "iex":
        return None
    return (
        "the IEX feed carries about 4% of consolidated minute volume, and the opening "
        "stage's `or_volume_share` feature was fitted on consolidated volume — its "
        "contribution to this forecast is biased. Run on the SIP or yfinance tape."
    )


def forecast_session(
    bundle: dict,
    history: pd.DataFrame,
    opening_bars: pd.DataFrame,
    session_date,
    open_price: "float | None" = None,
) -> dict:
    """The day's predicted high and low, plus everything the rule needs.

    `history` is completed daily bars (today's row is ignored if present),
    `opening_bars` the first `opening_minutes` regular-session 1-minute bars of
    `session_date`, `open_price` today's official opening print if one could be
    had. Returns `{"pred_high", "pred_low", "prev_avg", "adr14_abs", "or_high",
    "or_low"}` in dollars.

    Raises ValueError when the inputs cannot support a forecast, rather than
    returning a number built off NaNs -- the caller turns that into a logged
    refusal to trade.
    """
    want = opening_minutes(bundle)
    if len(opening_bars) < want:
        raise ValueError(
            f"the forecast is built on the first {want} minutes and only "
            f"{len(opening_bars)} bars have closed."
        )
    opening_bars = opening_bars.iloc[:want]

    daily = session_daily_frame(history, opening_bars, session_date, open_price)
    problem = require_history(daily)
    if problem is not None:
        raise ValueError(problem)

    feats = daily_features(daily)
    day = pd.Timestamp(session_date).normalize()
    row = feats.loc[[day]].copy()
    for name, value in opening_features(opening_bars, row.iloc[0]).items():
        row[name] = value

    model = bundle["model"]
    missing = [c for c in model.daily_cols + OPENING_FEATURE_COLS if not np.isfinite(row[c].iloc[0])]
    if missing:
        raise ValueError(
            "the feature table's last row is incomplete "
            f"({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}); "
            "the daily history is too short or has gaps."
        )

    seq = make_sequences(channel_frame(daily), row.index, model.lookback)
    pred = model.predict_prices(row, seq).iloc[0]
    return {
        "pred_high": float(pred["pred_high"]),
        "pred_low": float(pred["pred_low"]),
        "prev_avg": float(pred["prev_avg"]),
        "adr14_abs": float(row["adr14_abs"].iloc[0]),
        "or_high": float(row["high5"].iloc[0]),
        "or_low": float(row["low5"].iloc[0]),
    }


def session_bars(frame: pd.DataFrame, session_date) -> pd.DataFrame:
    """Regular-session minute bars belonging to one date.

    `persistence_model.minute_frame` already restricts the live buffer to
    09:30-15:59 of today, so this is a no-op there; it earns its keep on a
    frame assembled from stored bars.
    """
    day = pd.Timestamp(session_date).date()
    if not len(frame):
        return frame
    return frame[frame.index.date == day]


def market_date() -> "pd.Timestamp":
    """Today, in the exchange's timezone -- the session a forecast belongs to."""
    from .clock import now as _now

    return pd.Timestamp(_now().astimezone(market_hours.MARKET_TZ).date())
