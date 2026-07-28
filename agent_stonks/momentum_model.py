"""Intraday momentum-regime change model (TimeToChange / momlib).

Loads the regressor trained by FinNotebooks/TimeToChange notebook 03 and saved
as `Models/momentum_change_<TICKER>.joblib`: given the last minute of a live
session it predicts **Δ momentum over the next 15 bars** in bps/min -- the size
and direction of the momentum shift a persistent regime change would produce.

Definitions here must mirror `momlib/momentum.py` and `momlib/features.py`
exactly, the same contract `profile_model.py` has with LevelsML: the bundle's
`feature_cols` and `pipeline_params` are what makes the two agree, and a
feature pipeline that drifts from training is the classic way a saved model
goes silently wrong. `tests/test_momentum_model.py` pins the mirror by
re-deriving momlib's own numbers.

Momentum, regimes and persistent changes, from momlib
-----------------------------------------------------
momentum   mean 1-minute log return over the trailing `window` minutes in
           bps/min, EMA-smoothed (halflife `smooth_halflife`), computed inside
           a trading day only.
regime     +1 positive if momentum > theta, -1 negative if momentum < -theta,
           0 balanced in between, where theta = c * sigma(previous day) /
           sqrt(window) -- a per-day threshold known before the day starts.
persistent change   the regime flips at t, the old one held for the whole
           `persist` minutes before t and the new one holds for the `persist`
           minutes from t on.

Requires the optional `scikit-learn` / `joblib` dependencies (the estimator is
a pickled sklearn pipeline); everything degrades to None without them or
without the model file, exactly like the LevelsML pack.
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import clock, historical, market_hours

BPS = 1e4

REGIME_NAME = {-1: "negative", 0: "balanced", 1: "positive"}

# Default: the shared FinNotebooks model store, a sibling of the AgentStonks
# checkout (Code/FinNotebooks/Models). MODEL_DIR_ENV overrides the directory,
# MODEL_PATH_ENV a single ticker-independent file.
MODEL_DIR_ENV = "MOMENTUM_MODEL_DIR"
MODEL_PATH_ENV = "MOMENTUM_MODEL"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "FinNotebooks" / "Models"

# Regular-session window the model was trained on (momlib.data drops everything
# outside it before computing anything).
RTH_START = "09:30"
RTH_END = "15:59"

# A previous session with fewer bars than this is a half day / a partial
# yfinance chunk; momlib drops those days rather than letting them set `theta`.
MIN_BARS_PER_DAY = 300

_lock = threading.Lock()
_cache: dict = {"path": None, "bundle": None}
_history_cache: dict[tuple[str, int], dict] = {}


# --- the saved bundle -------------------------------------------------------

def model_path(ticker: str) -> Path:
    """Where the saved bundle for `ticker` is expected to live."""
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        return Path(override)
    model_dir = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
    return model_dir / f"momentum_change_{ticker.upper()}.joblib"


def load_bundle(ticker: str) -> "dict | None":
    """The saved bundle (estimator + feature list + pipeline params), or None.

    Cached after the first load; a missing file or a missing joblib/sklearn
    install is cached per path too, so a loop that asks every minute doesn't
    re-hit the filesystem.
    """
    path = model_path(ticker)
    with _lock:
        if _cache["path"] == path:
            return _cache["bundle"]
        _cache.update(path=path, bundle=None)
        try:
            import joblib  # noqa: F401  (also pulls in the sklearn unpickling path)
        except ImportError:
            return None
        try:
            bundle = joblib.load(path)
        except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError):
            return None
        if not isinstance(bundle, dict) or "estimator" not in bundle:
            return None
        _cache["bundle"] = bundle
        return bundle


# --- momentum / regimes / persistent changes (mirrors momlib.momentum) ------

def add_momentum(df: pd.DataFrame, window: int = 15, smooth_halflife: float = 8.0) -> pd.DataFrame:
    """Add log returns and trailing momentum (bps/min), per day.

    `mom_raw` is the plain trailing mean, `mom` its trailing EMA -- both use
    past bars only, so they are point-in-time safe.
    """
    out = df.copy()
    grp = out.groupby("day", sort=False)
    out["log_ret"] = grp["Close"].transform(lambda s: np.log(s).diff()) * BPS
    out["mom_raw"] = grp["log_ret"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    out["mom"] = out.groupby("day", sort=False)["mom_raw"].transform(
        lambda s: s.ewm(halflife=smooth_halflife, min_periods=1).mean()
    )
    return out


def add_regimes(df: pd.DataFrame, window: int = 15, c: float = 0.4) -> pd.DataFrame:
    """Classify every minute into a momentum regime with a per-day threshold.

    Day d's threshold uses day d-1's volatility so it is known at the open (the
    first day of the frame falls back to its own).
    """
    out = df.copy()
    sigma = out.groupby("day", sort=False)["log_ret"].std()
    prev_sigma = sigma.shift(1)
    prev_sigma.iloc[0] = sigma.iloc[0]
    theta_by_day = c * prev_sigma / np.sqrt(window)
    out["theta"] = pd.Series(out["day"], index=out.index).map(theta_by_day).astype(float)
    regime = np.select(
        [out["mom"] > out["theta"], out["mom"] < -out["theta"]], [1, -1], default=0
    ).astype(float)
    regime[out["mom"].isna().to_numpy()] = np.nan
    out["regime"] = regime
    return out


def find_persistent_changes(df: pd.DataFrame, persist: int = 15) -> pd.DataFrame:
    """Regime flips that held `persist` minutes on both sides, one row each.

    A change is only detectable `persist` bars after it happened, so the last
    `persist` minutes of a live frame never produce one -- which is exactly the
    delay a live system has to live with.
    """
    events = []
    for day, g in df.groupby("day", sort=True):
        reg = g["regime"].to_numpy()
        n = len(g)
        for t in range(persist, n - persist):
            r_new, r_prev = reg[t], reg[t - 1]
            if np.isnan(r_new) or np.isnan(r_prev) or r_new == r_prev:
                continue
            before = reg[t - persist : t]
            after = reg[t : t + persist]
            if np.isnan(before).any() or np.isnan(after).any():
                continue
            if (before == r_prev).all() and (after == r_new).all():
                events.append(
                    {
                        "timestamp": g.index[t],
                        "day": day,
                        "bar_of_day": t,
                        "price": g["Close"].iloc[t],
                        "prev_regime": int(r_prev),
                        "new_regime": int(r_new),
                        "mom_before": g["mom"].iloc[t - 1],
                        "mom_after": g["mom"].iloc[t + persist - 1],
                        "theta": g["theta"].iloc[t],
                    }
                )
    ev = pd.DataFrame(events)
    if len(ev):
        ev["transition"] = (
            ev["prev_regime"].map(REGIME_NAME) + " -> " + ev["new_regime"].map(REGIME_NAME)
        )
        ev["mom_delta"] = ev["mom_after"] - ev["mom_before"]
        ev = ev.set_index("timestamp").sort_index()
    return ev


def prepare(df: pd.DataFrame, params: "dict | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """momentum -> regimes -> persistent changes, with the bundle's parameters.

    Returns `(df_with_momentum, events)`. momlib's `prepare` also samples the
    stable points used as training negatives; scoring never needs them.
    """
    p = {"window": 15, "smooth_halflife": 8.0, "c": 0.4, "persist": 15, **(params or {})}
    df = add_momentum(df, window=p["window"], smooth_halflife=p["smooth_halflife"])
    df = add_regimes(df, window=p["window"], c=p["c"])
    return df, find_persistent_changes(df, persist=p["persist"])


# --- features (mirrors momlib.features) -------------------------------------

FEATURE_COLS = [
    "mom_5", "mom_15", "mom_60",
    "mom_z_5", "mom_z_15", "mom_z_60", "mom_smooth_z",
    "mom_slope",
    "rel_volume_15", "vol_accel",
    "volat_15", "volat_ratio",
    "dist_day_open", "range_pos_day", "dist_vwap",
    "dist_prev_close", "dist_sma5d", "ret_prev_day", "ret_5d",
    "minute_of_day", "theta",
    "dist_change_1", "dist_change_2", "dist_change_3", "bars_since_change",
    "regime_before",
]


def _per_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing per-bar features on the full minute grid."""
    out = pd.DataFrame(index=df.index)
    g = df.groupby("day", sort=False)

    # Momentum at several horizons, z-scored by the day's regime threshold so
    # that +-1 lands exactly on the regime boundary.
    for w in (5, 15, 60):
        m = g["log_ret"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        out[f"mom_{w}"] = m
        out[f"mom_z_{w}"] = m / df["theta"]
    out["mom_slope"] = out["mom_15"] - out["mom_15"].groupby(df["day"].to_numpy()).shift(5)
    out["mom_smooth_z"] = df["mom"] / df["theta"]

    # Volume behaviour.
    vol = df["Volume"].astype(float)
    v15 = vol.groupby(df["day"]).transform(lambda s: s.rolling(15, min_periods=15).mean())
    v60_prior = vol.groupby(df["day"]).transform(
        lambda s: s.rolling(60, min_periods=30).mean().shift(15)
    )
    out["rel_volume_15"] = v15 / v60_prior
    v5 = vol.groupby(df["day"]).transform(lambda s: s.rolling(5, min_periods=5).mean())
    out["vol_accel"] = v5 / v15

    # Volatility.
    s15 = g["log_ret"].transform(lambda s: s.rolling(15, min_periods=15).std())
    s60 = g["log_ret"].transform(lambda s: s.rolling(60, min_periods=30).std())
    out["volat_15"] = s15
    out["volat_ratio"] = s15 / s60

    # Position vs characteristic prices (log distance, bps).
    close = df["Close"]
    day_open = g["Open"].transform("first")
    out["dist_day_open"] = np.log(close / day_open) * BPS
    day_high = g["High"].transform(lambda s: s.cummax())
    day_low = g["Low"].transform(lambda s: s.cummin())
    rng = (day_high - day_low).replace(0.0, np.nan)
    out["range_pos_day"] = (close - day_low) / rng

    pv = (close * vol).groupby(df["day"]).transform("cumsum")
    cv = vol.groupby(df["day"]).transform("cumsum").replace(0.0, np.nan)
    out["dist_vwap"] = np.log(close / (pv / cv)) * BPS

    # Previous-day / multi-day context (all known before the day starts).
    daily_close = df.groupby("day", sort=True)["Close"].last()
    prev_close = daily_close.shift(1)
    sma5 = daily_close.rolling(5, min_periods=3).mean().shift(1)
    daily_ret = np.log(daily_close / prev_close) * BPS
    day_ser = pd.Series(df["day"], index=df.index)
    out["dist_prev_close"] = np.log(close / day_ser.map(prev_close).astype(float)) * BPS
    out["dist_sma5d"] = np.log(close / day_ser.map(sma5).astype(float)) * BPS
    out["ret_prev_day"] = day_ser.map(daily_ret.shift(1)).astype(float)
    ret_5d = (np.log(daily_close / daily_close.shift(5)) * BPS).shift(1)
    out["ret_5d"] = day_ser.map(ret_5d).astype(float)

    out["minute_of_day"] = g.cumcount()
    out["theta"] = df["theta"]
    return out


def _event_history_features(
    points: pd.DataFrame, events: pd.DataFrame, n_prev: int = 3
) -> pd.DataFrame:
    """Log distance to the prices of the last `n_prev` persistent changes
    strictly before each point, plus bars since the last one."""
    ev = events.sort_index()
    ev_ts = ev.index.to_numpy()
    ev_price = ev["price"].to_numpy(dtype=float)
    ev_pos = ev["global_pos"].to_numpy()

    rows = []
    for ts, row in points.iterrows():
        k = np.searchsorted(ev_ts, ts)  # events strictly before ts
        feat = {}
        for j in range(1, n_prev + 1):
            i = k - j
            feat[f"dist_change_{j}"] = (
                float(np.log(row["price"] / ev_price[i]) * BPS) if i >= 0 else np.nan
            )
        feat["bars_since_change"] = (
            float(row["global_pos"] - ev_pos[k - 1]) if k >= 1 else np.nan
        )
        rows.append(feat)
    return pd.DataFrame(rows, index=points.index)


def build_live_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    n_prev_changes: int = 3,
    confirm_lag: int = 0,
) -> pd.DataFrame:
    """The model's features for every bar of `df` (NaN while windows fill).

    `confirm_lag` delays each event's availability by that many bars. Training
    used 0, and 0 is also correct when only the newest bar is scored: an event
    can only be stamped `persist` bars after it happened, so nothing in the
    recent history was known any earlier than it really was.
    """
    bar_feats = _per_bar_features(df)
    pos = pd.Series(np.arange(len(df)), index=df.index, name="global_pos")
    points = pd.DataFrame({"price": df["Close"].astype(float)}).join(pos)

    if len(events):
        ev = events.join(pos)
        if confirm_lag:
            avail = np.minimum(ev["global_pos"].to_numpy() + confirm_lag, len(df) - 1)
            ev = ev.set_index(df.index[avail]).sort_index()
        hist = _event_history_features(points, ev, n_prev=n_prev_changes)
    else:
        cols = [f"dist_change_{j}" for j in range(1, n_prev_changes + 1)]
        hist = pd.DataFrame(np.nan, index=points.index, columns=cols + ["bars_since_change"])

    out = pd.concat([bar_feats, hist], axis=1)
    out["regime_before"] = df.groupby("day", sort=False)["regime"].shift(1)
    return out[FEATURE_COLS]


# The three features that carry the momentum signal: a prediction made before
# their trailing windows are filled is meaningless, whatever the imputer does.
_READY_COLS = ["mom_15", "mom_slope", "mom_smooth_z"]


def score_bars(bundle: dict, df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Predicted Δ momentum (bps/min) for every bar; NaN where not ready."""
    X = build_live_features(
        df, events, n_prev_changes=bundle["pipeline_params"]["n_prev_changes"]
    )[bundle["feature_cols"]]
    ready = X[_READY_COLS].notna().all(axis=1)
    pred = pd.Series(np.nan, index=df.index, name="pred_mom_delta")
    if ready.any():
        pred[ready] = bundle["estimator"].predict(X[ready])
    return pred


def score_latest(bundle: dict, df: pd.DataFrame, events: pd.DataFrame) -> "float | None":
    """Predicted Δ momentum for the LAST bar only, or None if it isn't ready.

    Same features as `score_bars`; only the final row goes through the
    estimator, since that is the only minute the live loop acts on.
    """
    X = build_live_features(
        df, events, n_prev_changes=bundle["pipeline_params"]["n_prev_changes"]
    )[bundle["feature_cols"]]
    row = X.iloc[[-1]]
    if not row[_READY_COLS].notna().all(axis=1).iloc[0]:
        return None
    return float(bundle["estimator"].predict(row)[0])


# --- the live minute grid ---------------------------------------------------

def _frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca/yfinance {"t","o","h","l","c","v"} bars -> the momlib frame shape:
    an exchange-local index, Open/High/Low/Close/Volume, and a `day` column."""
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "day"])
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="mixed").tz_convert(
        market_hours.MARKET_TZ
    )
    df = pd.DataFrame(
        {
            "Open": [float(b["o"]) for b in bars],
            "High": [float(b["h"]) for b in bars],
            "Low": [float(b["l"]) for b in bars],
            "Close": [float(b["c"]) for b in bars],
            "Volume": [float(b.get("v") or 0.0) for b in bars],
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time(RTH_START, RTH_END)
    df["day"] = df.index.date
    return df


def _previous_sessions(symbol: str, days: int, today) -> pd.DataFrame:
    """The last `days` completed sessions of 1-minute bars, from yfinance.

    Alpaca's IEX feed only carries IEX's own prints, so its minute bars are
    both thinner and lower-volume than the consolidated tape the model was
    trained on; the completed days that set `theta`, the previous close and the
    5-day return come from the same source as training. Cached per (symbol,
    days) for the trading day, since it only changes overnight.
    """
    key = (symbol, days)
    cached = _history_cache.get(key)
    if cached is not None and cached["day"] == today:
        return cached["frame"]

    frames = []
    back = 1
    # Walk back over calendar days until `days` sessions are collected; the
    # allowance covers weekends and holidays (which return no rows at all).
    while len(frames) < days and back <= days + 10:
        date = today - timedelta(days=back)
        back += 1
        try:
            bars = historical.fetch_intraday_bars_for_date(symbol, date.isoformat())
        except Exception:
            continue
        day_df = _frame_from_bars(bars)
        if len(day_df) >= MIN_BARS_PER_DAY:
            frames.append(day_df)

    frame = (
        pd.concat(reversed(frames)).sort_index()
        if frames
        else _frame_from_bars([])
    )
    _history_cache[key] = {"day": today, "frame": frame}
    return frame


def reset_history_cache() -> None:
    """Drop the cached previous sessions.

    They are keyed by (symbol, days) and the trading date only, so entering or
    leaving a simulation -- where the same symbol and date resolve to stored
    bars rather than yfinance -- must clear them (see `simlab.patches`).
    """
    _history_cache.clear()


def minute_frame(sym_state, history_days: int = 6) -> pd.DataFrame:
    """The frame the pipeline runs on: completed sessions plus today, live.

    Today comes from the streamed Alpaca bars (real time, and the only source
    that reaches the minute just closed); the days before it come from
    yfinance. Where both cover a timestamp the live bar wins.
    """
    today = clock.now().astimezone(market_hours.MARKET_TZ).date()
    with sym_state.lock:
        live_bars = list(sym_state.bars)
    live = _frame_from_bars(live_bars)
    live = live[live["day"] >= today] if len(live) else live

    history = _previous_sessions(sym_state.symbol, history_days, today)
    if not len(history):
        return live
    if not len(live):
        return history
    combined = pd.concat([history[history["day"] < today], live])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def read_latest(bundle: dict, df: pd.DataFrame) -> "dict | None":
    """Score the newest bar of `df`: what the regime is now and where the model
    says momentum is heading.

    Returns `{"ts", "price", "mom", "theta", "regime", "pred",
    "projected_mom", "projected_regime", "bars_today"}`, or None when the frame
    is too short to run the pipeline at all.

    `pred` is Δ momentum over the next `persist` bars in bps/min, so
    `mom + pred` is where the smoothed momentum is headed and comparing that to
    +-theta gives the regime the model is pointing at.
    """
    params = bundle["pipeline_params"]
    if len(df) < params["window"] + 2:
        return None
    scored, events = prepare(df, params)
    last = scored.iloc[-1]
    if pd.isna(last["mom"]) or pd.isna(last["theta"]):
        return None

    pred = score_latest(bundle, scored, events)
    mom = float(last["mom"])
    theta = float(last["theta"])
    projected = mom + pred if pred is not None else None
    return {
        "ts": scored.index[-1],
        "price": float(last["Close"]),
        "mom": mom,
        "theta": theta,
        "regime": int(last["regime"]) if not pd.isna(last["regime"]) else None,
        "pred": pred,
        "projected_mom": projected,
        "projected_regime": _regime_of(projected, theta),
        "bars_today": int((scored["day"] == scored["day"].iloc[-1]).sum()),
    }


def _regime_of(mom: "float | None", theta: float) -> "int | None":
    if mom is None:
        return None
    if mom > theta:
        return 1
    if mom < -theta:
        return -1
    return 0


def regime_name(regime: "int | None") -> str:
    return REGIME_NAME.get(regime, "unknown")
