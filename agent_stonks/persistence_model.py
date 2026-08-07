"""Intraday momentum-persistence classifier (TimeToChange2 / mshift).

Loads the classifier trained by FinNotebooks/TimeToChange2 notebook 04 and
saved as `Models/apple_momentum_2.joblib`: given the 20 bars ending at a
momentum-regime change it predicts the probability that the change is
**persistent** -- that the new regime survives, rather than the score flicking
back across the threshold a few minutes later.

Definitions here must mirror `mshift/momentum.py` and `mshift/features.py`
exactly, the same contract `profile_model.py` has with LevelsML: the bundle's
`feature_columns`, `seq_len` and `settings` are what makes the two agree, and a
feature pipeline that drifts from training is the classic way a saved model
goes silently wrong. `tests/test_persistence_model.py` pins the mirror.

Momentum, regimes and persistence, from mshift
----------------------------------------------
momentum   the `horizon`-bar log return divided by its own random-walk scale
           (sigma * sqrt(horizon)), then EWM-smoothed -- a dimensionless
           "sigmas of drift" score, comparable across quiet and busy parts of
           the day. Computed inside a trading session only.
regime     a Schmitt trigger over that score: ENTER a directional regime at
           |mom| > `enter_threshold`, leave it only once |mom| falls back below
           `exit_threshold`. The hysteresis is what stops a score hovering near
           the line from emitting a burst of fake changes.
persistent the old regime had held >= 15 bars and the new one holds >= 15 bars,
           or price moves >= 1% in the new regime's direction within 10 bars.
           Both halves are forward-looking -- which is what the model is for.

What the model is, honestly
---------------------------
TimeToChange2's own headline: out-of-fold AUC 0.82 over all regime changes, but
0.50 over the changes that already pass the observable `pre_dwell >= 15`
pre-condition. It is **a filter that separates the impossible from the
possible, not the likely from the unlikely** -- worth consulting before taking
a change, not worth reading as a forecast of how far price will go.

Everything is session-local: no indicator crosses the overnight gap, so the
live frame is today's streamed bars and nothing else. Two differences from
training are unavoidable and worth naming:

* the volatility floor (`sigma_floor`) is a per-session median, so live it is
  the median over the bars seen *so far* rather than the whole day. It only
  ever acts as a clip on a near-flat window, so it rarely binds at all.
* training bars came from the consolidated tape (yfinance); live bars come from
  the streamed feed. The volume features are all ratios and z-scores within the
  session, so a thinner feed mostly cancels out -- but not exactly.

Requires the optional `scikit-learn` / `joblib` dependencies (the estimator is
a pickled sklearn pipeline); everything degrades to None without them or
without the model file, exactly like the LevelsML pack.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pandas as pd

from . import clock, market_hours

REGIME_NAME = {-1: "negative", 0: "balanced", 1: "positive"}

# Default: the shared model store next to the AgentStonks checkout
# (Code/Models), the same directory the LevelsML pack lives in.
MODEL_PATH_ENV = "APPLE_MOMENTUM_MODEL"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "Models" / "apple_momentum_2.joblib"

# Regular-session window the model was trained on (mshift.data drops everything
# outside it before computing anything).
RTH_START = "09:30"
RTH_END = "15:59"

# mshift.config.MomentumParams -- the fallbacks if a bundle carries no settings.
MOMENTUM_DEFAULTS = {
    "horizon": 15,
    "vol_window": 30,
    "smooth_span": 7,
    "enter_threshold": 0.90,
    "exit_threshold": 0.40,
    "warmup_bars": 30,
}

_lock = threading.Lock()
_cache: dict = {"path": None, "bundle": None}


# --- the saved bundle -------------------------------------------------------

# Statistics the summariser extracts per feature channel, in output order.
SUMMARY_STATS = ("last", "mean", "std", "min", "max", "slope", "delta", "last_vs_mean")


class SequenceSummarizer:
    """Turn ``(n, L, F)`` sequences into ``(n, F * len(stats))`` tabular rows.

    A verbatim mirror of `mshift.model.SequenceSummarizer`, and the reason this
    class exists here at all: it is the first step of the pickled pipeline, so
    unpickling the bundle needs *something* importable at
    ``mshift.model.SequenceSummarizer``. Its state is only the constructor
    arguments, so the fitted pipeline restores onto this definition exactly.

    ``last`` is the value on the change bar itself, ``slope`` is the
    least-squares trend across the window and ``delta`` is last minus first.
    """

    def __init__(self, stats: tuple = SUMMARY_STATS, tail: "int | None" = None):
        self.stats = stats
        self.tail = tail

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"expected a 3-D array (n, L, F), got shape {X.shape}")
        self.n_features_in_ = X.shape[1] * X.shape[2]
        self.seq_len_ = X.shape[1] if self.tail is None else min(self.tail, X.shape[1])
        self.n_channels_ = X.shape[2]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"expected a 3-D array (n, L, F), got shape {X.shape}")
        if self.tail is not None:
            X = X[:, -self.tail :, :]
        L = X.shape[1]
        t_centred = np.arange(L, dtype=np.float64)
        t_centred -= t_centred.mean()
        denom = float((t_centred**2).sum()) or 1.0

        pieces = []
        for stat in self.stats:
            if stat == "last":
                pieces.append(X[:, -1, :])
            elif stat == "mean":
                pieces.append(X.mean(axis=1))
            elif stat == "std":
                pieces.append(X.std(axis=1))
            elif stat == "min":
                pieces.append(X.min(axis=1))
            elif stat == "max":
                pieces.append(X.max(axis=1))
            elif stat == "slope":
                pieces.append(np.einsum("l,nlf->nf", t_centred, X) / denom)
            elif stat == "delta":
                pieces.append(X[:, -1, :] - X[:, 0, :])
            elif stat == "last_vs_mean":
                pieces.append(X[:, -1, :] - X.mean(axis=1))
            else:
                raise ValueError(f"unknown summary statistic {stat!r}")
        return np.concatenate(pieces, axis=1)

    def get_feature_names_out(self, input_features=None):
        base = (
            list(input_features)
            if input_features is not None
            else [f"f{i}" for i in range(self.n_channels_)]
        )
        return np.array([f"{c}__{s}" for s in self.stats for c in base], dtype=object)


def _register_unpickle_alias() -> None:
    """Make `mshift.model.SequenceSummarizer` resolvable for `joblib.load`.

    The bundle was pickled from the notebooks' own package, so the class is
    stamped with its module path there. Importing the real `mshift` would drag
    in matplotlib, plotly and yfinance for a class that is 40 lines of numpy,
    so a stub module pointing at the mirror above stands in -- unless the real
    package is genuinely importable, in which case it wins.
    """
    if "mshift.model" in sys.modules:
        return
    try:
        __import__("mshift.model")
        return
    except Exception:
        sys.modules.pop("mshift", None)
    package = types.ModuleType("mshift")
    package.__path__ = []  # a package, so "mshift.model" is a legal submodule
    module = types.ModuleType("mshift.model")
    module.SequenceSummarizer = SequenceSummarizer
    module.SUMMARY_STATS = SUMMARY_STATS
    package.model = module
    sys.modules["mshift"] = package
    sys.modules["mshift.model"] = module


def model_path() -> Path:
    """Where the saved bundle is expected to live.

    One file, not one per ticker: TimeToChange2 fitted a single AAPL model and
    the honest reading of its own results does not transfer to other names.
    """
    return Path(os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH)


def load_bundle() -> "dict | None":
    """The saved bundle (pipeline + feature list + seq_len + threshold), or None.

    Cached after the first load; a missing file or a missing joblib/sklearn
    install is cached per path too, so a loop that asks every minute doesn't
    re-hit the filesystem.
    """
    path = model_path()
    with _lock:
        if _cache["path"] == path:
            return _cache["bundle"]
        _cache.update(path=path, bundle=None)
        try:
            import joblib  # noqa: F401  (also pulls in the sklearn unpickling path)
        except ImportError:
            return None
        _register_unpickle_alias()
        try:
            bundle = joblib.load(path)
        except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError):
            return None
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            return None
        if {"feature_columns", "seq_len"} - set(bundle):
            return None
        _cache["bundle"] = bundle
        return bundle


def reset_bundle_cache() -> None:
    """Drop the cached bundle (used by tests that swap the model file)."""
    with _lock:
        _cache.update(path=None, bundle=None)


def momentum_params(bundle: "dict | None" = None) -> dict:
    """The momentum/regime settings the bundle was fitted with."""
    settings = (bundle or {}).get("settings") or {}
    return {**MOMENTUM_DEFAULTS, **(settings.get("momentum") or {})}


def model_threshold(bundle: "dict | None" = None) -> float:
    """The probability cut-off chosen on the validation block at training time."""
    try:
        return float((bundle or {})["threshold"])
    except (KeyError, TypeError, ValueError):
        return 0.5


# --- momentum / regimes (mirrors mshift.momentum) ---------------------------

def compute_momentum(bars: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
    """Add `ret_1`, `sigma`, `mom_raw` and `mom`, per session.

    `mom` is the `horizon`-bar log return in units of its own random-walk
    standard deviation, EWM-smoothed: +1 means "price drifted up by about one
    sigma more than a coin flip would".
    """
    p = {**MOMENTUM_DEFAULTS, **(params or {})}
    out = bars.copy()
    logp = np.log(out["close"])
    log_grp = logp.groupby(out["session"], sort=False)

    out["ret_1"] = log_grp.diff()
    out["sigma"] = (
        out["ret_1"]
        .groupby(out["session"], sort=False)
        .transform(lambda s: s.rolling(p["vol_window"], min_periods=10).std())
    )
    # A dead-flat window would divide by ~0; floor sigma at a tiny fraction of
    # the session's own typical volatility.
    sigma_floor = out.groupby("session", sort=False)["sigma"].transform("median") * 0.10
    sigma = out["sigma"].clip(lower=sigma_floor).replace(0.0, np.nan)

    horizon_return = log_grp.transform(lambda s: s - s.shift(p["horizon"]))
    out["mom_raw"] = horizon_return / (sigma * np.sqrt(p["horizon"]))
    out["mom"] = (
        out["mom_raw"]
        .groupby(out["session"], sort=False)
        .transform(lambda s: s.ewm(span=p["smooth_span"], adjust=False).mean())
    )
    return out


def _hysteresis_regimes(
    mom: np.ndarray, session_start: np.ndarray, enter: float, exit_: float
) -> np.ndarray:
    """Schmitt-trigger classification of a momentum series.

    Entering a directional regime needs `|mom| > enter`; leaving it needs
    `|mom|` to fall back below `exit_`. Each session starts flat.
    """
    n = mom.shape[0]
    regimes = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(n):
        if session_start[i]:
            state = 0
        m = mom[i]
        if np.isnan(m):
            regimes[i] = 0
            state = 0
            continue
        if state == 1:
            if m < exit_:
                state = -1 if m < -enter else 0
        elif state == -1:
            if m > -exit_:
                state = 1 if m > enter else 0
        else:
            if m > enter:
                state = 1
            elif m < -enter:
                state = -1
        regimes[i] = state
    return regimes


def assign_regimes(bars: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
    """Add `regime`, `prev_regime`, `regime_change` and the dwell counters."""
    p = {**MOMENTUM_DEFAULTS, **(params or {})}
    out = bars.copy()

    out["regime"] = _hysteresis_regimes(
        out["mom"].to_numpy(dtype=float),
        out["bar_of_day"].to_numpy() == 0,
        p["enter_threshold"],
        p["exit_threshold"],
    )
    prev = out.groupby("session", sort=False)["regime"].shift(1)
    out["prev_regime"] = prev
    out["regime_change"] = (out["regime"] != prev) & prev.notna()

    # run_id increments on every change, so the dwell counter is a plain
    # cumulative count inside each run.
    out["run_id"] = out["regime_change"].cumsum() + out["session"].factorize()[0] * 10**6
    out["bars_in_regime"] = out.groupby("run_id", sort=False).cumcount() + 1
    # How long the OLD regime had run, on a change bar: the observable half of
    # the persistence definition, and the model's strongest single feature.
    prev_dwell = out["bars_in_regime"].groupby(out["session"], sort=False).shift(1)
    out["pre_dwell"] = prev_dwell.where(out["regime_change"])
    return out


def add_momentum_regimes(bars: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
    """Momentum + regimes in one call."""
    return assign_regimes(compute_momentum(bars, params), params)


# --- features (mirrors mshift.features) -------------------------------------

FEATURE_COLUMNS = [
    # momentum shape
    "f_mom", "f_mom_raw", "f_mom_delta", "f_mom_accel", "f_mom_abs",
    # normalised returns
    "f_ret_1", "f_ret_5", "f_ret_15", "f_ema_spread", "f_trend_5_30",
    # where price sits
    "f_rsi", "f_bb_pos", "f_vwap_dist", "f_day_range_pos",
    # volatility / candle shape
    "f_vol_ratio", "f_atr_rel", "f_body", "f_wick_skew",
    # volume
    "f_volume_z", "f_volume_ratio", "f_signed_volume",
    # context
    "f_regime", "f_bars_in_regime", "f_prev_dwell", "f_time_of_day",
]


def _rsi(close: pd.Series, session: pd.Series, window: int = 14) -> pd.Series:
    delta = close.groupby(session, sort=False).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(session, sort=False).transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    avg_loss = loss.groupby(session, sort=False).transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Add every `f_*` column to a frame that already has momentum and regimes.

    Every feature is causal -- it uses only information available at the close
    of its own bar -- and session-local. Most are normalised by the local
    volatility, so a value means the same thing at 09:45 on a wild day and at
    15:00 on a quiet one.
    """
    out = bars.copy()
    session = out["session"]
    close = out["close"]
    logp = np.log(close)
    sigma = out["sigma"].replace(0.0, np.nan)

    # --- momentum shape ---------------------------------------------------
    out["f_mom"] = out["mom"]
    out["f_mom_raw"] = out["mom_raw"]
    out["f_mom_delta"] = out["mom"].groupby(session, sort=False).diff()
    out["f_mom_accel"] = out["f_mom_delta"].groupby(session, sort=False).diff()
    out["f_mom_abs"] = out["mom"].abs()

    # --- normalised returns -----------------------------------------------
    out["f_ret_1"] = out["ret_1"] / sigma
    for k in (5, 15):
        r = logp.groupby(session, sort=False).transform(lambda s, k=k: s - s.shift(k))
        out[f"f_ret_{k}"] = r / (sigma * np.sqrt(k))

    ema_fast = close.groupby(session, sort=False).transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    ema_slow = close.groupby(session, sort=False).transform(
        lambda s: s.ewm(span=21, adjust=False).mean()
    )
    out["f_ema_spread"] = np.log(ema_fast / ema_slow) / sigma

    sma_fast = close.groupby(session, sort=False).transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    sma_slow = close.groupby(session, sort=False).transform(
        lambda s: s.rolling(30, min_periods=10).mean()
    )
    out["f_trend_5_30"] = np.log(sma_fast / sma_slow) / sigma

    # --- where price sits --------------------------------------------------
    out["f_rsi"] = (_rsi(close, session) - 50.0) / 50.0

    sma20 = close.groupby(session, sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    std20 = close.groupby(session, sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    out["f_bb_pos"] = ((close - sma20) / (2.0 * std20)).clip(-3, 3)

    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    pv = (typical * out["volume"]).groupby(session, sort=False).cumsum()
    cum_vol = out["volume"].groupby(session, sort=False).cumsum().replace(0.0, np.nan)
    out["vwap"] = pv / cum_vol
    out["f_vwap_dist"] = (np.log(close / out["vwap"]) / sigma).clip(-10, 10)

    day_high = out["high"].groupby(session, sort=False).cummax()
    day_low = out["low"].groupby(session, sort=False).cummin()
    span = (day_high - day_low).replace(0.0, np.nan)
    out["f_day_range_pos"] = (close - day_low) / span

    # --- volatility / candle shape -----------------------------------------
    sigma_fast = out["ret_1"].groupby(session, sort=False).transform(
        lambda s: s.rolling(10, min_periods=5).std()
    )
    sigma_slow = out["ret_1"].groupby(session, sort=False).transform(
        lambda s: s.rolling(60, min_periods=20).std()
    )
    out["f_vol_ratio"] = np.log(sigma_fast / sigma_slow.replace(0.0, np.nan)).clip(-3, 3)

    prev_close = close.groupby(session, sort=False).shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.groupby(session, sort=False).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    )
    out["f_atr_rel"] = (atr / close) / sigma

    bar_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["f_body"] = (close - out["open"]) / bar_range
    upper = out["high"] - out[["open", "close"]].max(axis=1)
    lower = out[["open", "close"]].min(axis=1) - out["low"]
    out["f_wick_skew"] = (upper - lower) / bar_range

    # --- volume -------------------------------------------------------------
    vol_mean = out["volume"].groupby(session, sort=False).transform(
        lambda s: s.rolling(30, min_periods=10).mean()
    )
    vol_std = out["volume"].groupby(session, sort=False).transform(
        lambda s: s.rolling(30, min_periods=10).std()
    )
    out["f_volume_z"] = (
        (out["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)
    ).clip(-5, 10)
    vol_fast = out["volume"].groupby(session, sort=False).transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    out["f_volume_ratio"] = np.log(vol_fast / vol_mean.replace(0.0, np.nan)).clip(-3, 3)
    out["f_signed_volume"] = (
        (np.sign(out["ret_1"]) * out["f_volume_z"])
        .groupby(session, sort=False)
        .transform(lambda s: s.ewm(span=5, adjust=False).mean())
    )

    # --- context ------------------------------------------------------------
    out["f_regime"] = out["regime"].astype(float)
    out["f_bars_in_regime"] = np.log1p(out["bars_in_regime"].astype(float))
    # How long the regime had already been running as of the PREVIOUS bar. On a
    # change bar this is exactly `pre_dwell`; `f_bars_in_regime` cannot carry
    # it, since on a change bar it is always 1.
    out["f_prev_dwell"] = np.log1p(
        out["bars_in_regime"].groupby(session, sort=False).shift(1).astype(float)
    )
    out["f_time_of_day"] = out["minutes_from_open"] / 390.0

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


# --- scoring ----------------------------------------------------------------

def build_sequence(
    features: pd.DataFrame, feature_columns: "list[str]", seq_len: int
) -> "np.ndarray | None":
    """The `(seq_len, n_features)` block ending at the LAST bar, or None.

    None means the model must not be asked: the window would start before the
    session opened, or some trailing indicator in it has not warmed up yet.
    Training dropped exactly these samples rather than imputing them.
    """
    if len(features) < seq_len:
        return None
    block = features[feature_columns].to_numpy(dtype=np.float32)[-seq_len:]
    bar_of_day = features["bar_of_day"].to_numpy()
    if bar_of_day[-seq_len] > bar_of_day[-1]:
        return None  # the window would cross into the previous session
    if not np.isfinite(block).all():
        return None
    return block


def predict_proba(bundle: dict, X) -> np.ndarray:
    """Persistence probability for a batch of `(n, seq_len, n_features)`.

    Two kinds of bundle answer this. The one loaded above carries a `pipeline`
    -- a pickled sklearn classifier that was trained on the label directly.
    `agent_stonks.nbeats_model` builds one carrying a `score` callable instead,
    which reaches the same number by forecasting momentum and replaying the
    regime rule over the forecast. Everything upstream of this line is
    identical for both, which is what lets Apple Trader swap them without a
    branch in its rules; see `agent_stonks.apple_models`.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None, ...]
    expected = (int(bundle["seq_len"]), len(bundle["feature_columns"]))
    if X.shape[1:] != expected:
        raise ValueError(f"expected sequences of shape {expected}, got {X.shape[1:]}")
    scorer = bundle.get("score")
    if scorer is not None:
        return np.asarray(scorer(X), dtype=float)
    return bundle["pipeline"].predict_proba(X)[:, 1]


# --- the live minute grid ---------------------------------------------------

def _frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca/yfinance {"t","o","h","l","c","v"} bars -> the mshift frame shape:
    an exchange-local index, lower-case OHLCV, and the session bookkeeping."""
    columns = [
        "open", "high", "low", "close", "volume",
        "session", "bar_of_day", "minutes_from_open",
    ]
    if not bars:
        return pd.DataFrame(columns=columns)
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="mixed").tz_convert(
        market_hours.MARKET_TZ
    )
    df = pd.DataFrame(
        {
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b.get("v") or 0.0) for b in bars],
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time(RTH_START, RTH_END)
    df = df[df["close"] > 0]
    return add_session_columns(df)


def add_session_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-day bookkeeping the rest of the pipeline groups on."""
    out = bars.copy()
    out["session"] = out.index.normalize()
    out["bar_of_day"] = out.groupby("session").cumcount()
    open_ts = out.index.normalize() + pd.Timedelta(f"{RTH_START}:00")
    out["minutes_from_open"] = (out.index - open_ts).total_seconds() / 60.0
    return out


def minute_frame(sym_state) -> pd.DataFrame:
    """The frame the pipeline runs on: today's session, live.

    Nothing in mshift crosses the overnight gap -- momentum, regimes and all 25
    features are computed inside a session -- so unlike the per-day-threshold
    model this one needs no history behind today at all.
    """
    today = clock.now().astimezone(market_hours.MARKET_TZ).date()
    with sym_state.lock:
        live_bars = list(sym_state.bars)
    live = _frame_from_bars(live_bars)
    if not len(live):
        return live
    return live[live.index.date == today]


def read_latest(bundle: dict, df: pd.DataFrame) -> "dict | None":
    """Score the newest bar of `df`: the regime it is in, whether it just
    changed, and -- only for a change INTO the positive regime -- the model's
    probability that the change is persistent.

    Returns `{"ts", "price", "high", "mom", "regime", "prev_regime",
    "regime_change", "to_positive", "pre_dwell", "proba", "bars_today",
    "warming_up"}`, or None when the frame is too short to score at all.

    The model is consulted on exactly the bars TimeToChange2's own simulator
    consults it on (`mshift.backtest._signal_sequences`): a to-positive regime
    change with a complete, all-finite 20-bar feature window behind it, inside
    one session. On every other bar `proba` is None -- an absent number rather
    than a fabricated one.

    `warming_up` reports the opening window `mshift.momentum.build_events`
    drops when assembling the *training* set, but it is not a second gate here
    any more than it is in the simulator -- and it never binds in practice:
    the trailing indicators inside the 20-bar window are not all finite until
    around bar 39 of a session, well past the 30-bar warm-up, so window
    completeness rejects every early change before the flag could.
    """
    p = momentum_params(bundle)
    if len(df) < p["horizon"] + 2:
        return None
    scored = add_momentum_regimes(df, p)
    last = scored.iloc[-1]
    if pd.isna(last["mom"]):
        return None

    regime = int(last["regime"])
    changed = bool(last["regime_change"])
    to_positive = changed and regime == 1
    warming_up = int(last["bar_of_day"]) < p["warmup_bars"]

    proba = None
    if to_positive:
        sequence = build_sequence(
            compute_features(scored), bundle["feature_columns"], int(bundle["seq_len"])
        )
        if sequence is not None:
            proba = float(predict_proba(bundle, sequence[None, ...])[0])

    prev = last["prev_regime"]
    return {
        "ts": scored.index[-1],
        "price": float(last["close"]),
        "high": float(last["high"]),
        "mom": float(last["mom"]),
        "regime": regime,
        "prev_regime": None if pd.isna(prev) else int(prev),
        "regime_change": changed,
        "to_positive": to_positive,
        "pre_dwell": None if pd.isna(last["pre_dwell"]) else int(last["pre_dwell"]),
        "proba": proba,
        "bars_today": int(last["bar_of_day"]) + 1,
        "warming_up": warming_up,
    }


def regime_name(regime: "int | None") -> str:
    return REGIME_NAME.get(regime, "unknown")
