import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .datalog import log_fetch, log_fetch_failure
from .market_hours import MARKET_TZ

logger = logging.getLogger(__name__)

HISTORICAL_PERIODS: dict[str, int] = {
    "7 Days": 7,
    "28 Days": 28,
    "Quarter": 91,
    "1 Year": 365,
    "5 Years": 1825,
}

VIX_SYMBOL = "^VIX"
VIX3M_SYMBOL = "^VIX3M"
SPY_SYMBOL = "SPY"

# Broad-market indicator series change slowly relative to the agent's cycle and
# are identical for every ticker, so cache them briefly to avoid re-hitting
# yfinance on each agent call.
_market_cache: dict = {"ts": None, "data": None}


def fetch_close_series(symbol: str, days: int) -> pd.Series:
    """Fetch daily close prices for `symbol` over the trailing `days`. Raises on failure."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    except Exception as exc:
        log_fetch_failure(
            "daily closes",
            [("yfinance", exc)],
            symbol=symbol,
            consequence=f"start={start:%Y-%m-%d}, end={end:%Y-%m-%d}",
        )
        raise
    if df.empty:
        log_fetch(
            "daily closes",
            "yfinance",
            symbol=symbol,
            detail=f"0 rows for start={start:%Y-%m-%d}, end={end:%Y-%m-%d}",
        )
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close = close.dropna()
    log_fetch("daily closes", "yfinance", symbol=symbol, detail=f"{len(close)} rows over {days}d")
    return close


def fetch_intraday_bars(symbol: str, interval: str = "1m") -> list[dict]:
    """Today's intraday bars from yfinance -- last-resort price fallback when both
    Alpaca's stream and REST API are unavailable. No API key required, but quotes
    are delayed (typically ~15 minutes) rather than real-time.

    Returns bars in the same {"t","o","h","l","c","v"} shape as Alpaca's REST/stream
    bars (UTC ISO timestamps) so callers don't need to branch on the source.
    """
    try:
        df = yf.download(symbol, period="1d", interval=interval, auto_adjust=False, progress=False)
    except Exception as exc:
        log_fetch_failure("intraday bars", [("yfinance", exc)], symbol=symbol)
        raise
    if df.empty:
        log_fetch("intraday bars", "yfinance (delayed)", symbol=symbol, detail="0 bars returned")
        return []
    log_fetch(
        "intraday bars",
        "yfinance (delayed)",
        symbol=symbol,
        detail=f"{len(df)} {interval} bars",
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    idx = idx.tz_convert("UTC")
    return [
        {
            "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": float(row.Open),
            "h": float(row.High),
            "l": float(row.Low),
            "c": float(row.Close),
            "v": float(row.Volume),
        }
        for ts, row in zip(idx, df.itertuples(index=False))
    ]


def fetch_intraday_bars_for_date(symbol: str, date: str, interval: str = "1m") -> list[dict]:
    """Intraday bars for a single past trading day (`date`, "YYYY-MM-DD") from
    yfinance. Used to build a volume-by-price profile of a prior session.

    yfinance only serves intraday history for roughly the last 60 days, and a
    non-trading day (weekend/holiday) returns no rows. Bars share the same
    {"t","o","h","l","c","v"} shape (UTC ISO timestamps) as fetch_intraday_bars,
    so callers don't branch on the source. Returns [] when the day has no data.
    """
    try:
        start = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"date must be 'YYYY-MM-DD', got {date!r}") from exc
    end = start + timedelta(days=1)
    try:
        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        log_fetch_failure("intraday bars (date)", [("yfinance", exc)], symbol=symbol)
        raise
    if df.empty:
        log_fetch("intraday bars (date)", "yfinance (delayed)", symbol=symbol, detail=f"0 bars for {date}")
        return []
    log_fetch(
        "intraday bars (date)",
        "yfinance (delayed)",
        symbol=symbol,
        detail=f"{len(df)} {interval} bars for {date}",
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    idx = idx.tz_convert("UTC")
    return [
        {
            "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": float(row.Open),
            "h": float(row.High),
            "l": float(row.Low),
            "c": float(row.Close),
            "v": float(row.Volume),
        }
        for ts, row in zip(idx, df.itertuples(index=False))
    ]


# Consolidated-tape volume (yfinance) is identical for every caller within a
# short window and each read is a full intraday/daily download, so cache briefly
# to avoid re-hitting yfinance on every analyze_volume tool call. Keyed by
# symbol; each entry is {"ts": datetime, "bars": list[dict]}.
_intraday_volume_cache: dict[str, dict] = {}
_daily_volume_cache: dict[str, dict] = {}
_VOLUME_CACHE_TTL_SEC = 60


def fetch_intraday_volume_bars(symbol: str, interval: str = "1m", ttl_sec: int = _VOLUME_CACHE_TTL_SEC) -> list[dict]:
    """Today's intraday bars from yfinance, for the volume-confirmation read.

    Alpaca's default (IEX) feed reports volume from a single venue -- a few
    percent of the consolidated tape -- so its absolute volume and volume ratios
    are unreliable. yfinance reports consolidated-tape volume across every
    exchange, making it the more accurate source for volume analysis even though
    its prices are ~15 minutes delayed. Bars share the {"t","o","h","l","c","v"}
    Alpaca shape. Cached for `ttl_sec`; returns [] (or the last good cache) on
    failure so callers can fall back to Alpaca volume.
    """
    now = datetime.now(timezone.utc)
    cached = _intraday_volume_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < ttl_sec:
        return cached["bars"]
    try:
        bars = fetch_intraday_bars(symbol, interval=interval)
    except Exception:
        return cached["bars"] if cached else []
    _intraday_volume_cache[symbol] = {"ts": now, "bars": bars}
    return bars


def fetch_daily_volume_bars(symbol: str, days: int = 90, ttl_sec: int = _VOLUME_CACHE_TTL_SEC) -> list[dict]:
    """Completed daily volumes from yfinance (consolidated tape), oldest-first.

    Supplies the average-daily-volume baseline that rvol_pace compares today's
    cumulative volume against. Sourcing both from yfinance keeps that ratio
    like-for-like; mixing a consolidated numerator with a single-venue Alpaca
    baseline would inflate rvol_pace by the inverse of IEX's tape share. Each bar
    is {"t": "YYYY-MM-DD", "v": float}. Cached for `ttl_sec`; [] on failure.
    """
    now = datetime.now(timezone.utc)
    cached = _daily_volume_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < ttl_sec:
        return cached["bars"]
    end = now
    start = end - timedelta(days=days)
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    except Exception as exc:
        log_fetch_failure("daily volume", [("yfinance", exc)], symbol=symbol)
        return cached["bars"] if cached else []
    if df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    bars = [
        {"t": ts.strftime("%Y-%m-%d"), "v": float(vol)}
        for ts, vol in zip(idx, df["Volume"])
        if pd.notna(vol)
    ]
    log_fetch("daily volume", "yfinance", symbol=symbol, detail=f"{len(bars)} days")
    _daily_volume_cache[symbol] = {"ts": now, "bars": bars}
    return bars


# A year of daily bars is a slow download and never changes during a session,
# so this one is cached for far longer than the volume reads above -- long
# enough that a per-minute trading loop asks yfinance once a day.
_daily_ohlc_cache: dict[str, dict] = {}
_DAILY_OHLC_CACHE_TTL_SEC = 3600


def fetch_daily_ohlc_bars(
    symbol: str, days: int = 420, ttl_sec: int = _DAILY_OHLC_CACHE_TTL_SEC
) -> list[dict]:
    """Completed daily OHLCV bars from yfinance, oldest-first, **unadjusted**.

    The long daily history behind a per-session model (see
    `agent_stonks.dayrange_model`), in the same {"t","o","h","l","c","v"} shape
    the rest of the app uses -- `t` is a plain "YYYY-MM-DD".

    Two properties this deliberately has:

    * `auto_adjust=False`, so prices are on the raw scale a live tape prints at
      and a dividend does not silently reprice the whole history. Any model
      fitted on unadjusted bars needs them; `fetch_close_series` adjusts, which
      is right for the return-based reads that use it and wrong here.
    * **today's partial row is dropped.** A daily bar mid-session has a real
      open and a high/low/close that are only true so far, and a caller that
      cannot tell the two apart will happily read the day's outcome out of it.
      Callers that legitimately need today's open ask `fetch_session_open`,
      which returns that one number and nothing else.

    Cached for `ttl_sec`; [] (or the last good cache) on failure.
    """
    now = datetime.now(timezone.utc)
    cached = _daily_ohlc_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < ttl_sec:
        return cached["bars"]
    start = now - timedelta(days=days)
    try:
        df = yf.download(
            symbol, start=start, end=now + timedelta(days=1), interval="1d",
            auto_adjust=False, progress=False,
        )
    except Exception as exc:
        log_fetch_failure("daily ohlc", [("yfinance", exc)], symbol=symbol)
        return cached["bars"] if cached else []
    if df is None or df.empty:
        return cached["bars"] if cached else []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    today = datetime.now(timezone.utc).astimezone(MARKET_TZ).date()
    bars = [
        {
            "t": ts.date().isoformat(),
            "o": float(row.Open), "h": float(row.High),
            "l": float(row.Low), "c": float(row.Close), "v": float(row.Volume),
        }
        for ts, row in zip(pd.to_datetime(df.index), df.itertuples(index=False))
        if row.Close == row.Close and ts.date() < today
    ]
    log_fetch("daily ohlc", "yfinance", symbol=symbol, detail=f"{len(bars)} days")
    _daily_ohlc_cache[symbol] = {"ts": now, "bars": bars}
    return bars


def fetch_session_open(symbol: str, ttl_sec: int = 300) -> Optional[float]:
    """Today's official opening print, or None before it exists.

    Split out from `fetch_daily_ohlc_bars` because it is the one part of
    today's daily bar that is *finished*: the open is set at 9:30 and never
    moves, while the same row's high, low and close keep changing until the
    bell. Returning it alone is what lets a model use today's open without any
    caller being able to reach the day's outcome through the same object.

    Sourced from the daily bar rather than from the first minute bar on
    purpose: it is the print models fitted on daily data were trained against,
    and the two differ by a few basis points on about half of all sessions
    (a minute bar's open is the first *trade* the feed saw, not the auction).
    """
    for bar in reversed(_todays_daily_row(symbol, ttl_sec)):
        return float(bar["o"])
    return None


_session_open_cache: dict[str, dict] = {}


def _todays_daily_row(symbol: str, ttl_sec: int) -> list[dict]:
    """Today's daily bar as a 0- or 1-element list, cached."""
    now = datetime.now(timezone.utc)
    cached = _session_open_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < ttl_sec:
        return cached["rows"]
    today = now.astimezone(MARKET_TZ).date()
    try:
        df = yf.download(
            symbol, start=today.isoformat(), end=(today + timedelta(days=1)).isoformat(),
            interval="1d", auto_adjust=False, progress=False,
        )
    except Exception as exc:
        log_fetch_failure("session open", [("yfinance", exc)], symbol=symbol)
        return cached["rows"] if cached else []
    rows: list[dict] = []
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for ts, row in zip(pd.to_datetime(df.index), df.itertuples(index=False)):
            if ts.date() == today and row.Open == row.Open:
                rows = [{"t": today.isoformat(), "o": float(row.Open)}]
    _session_open_cache[symbol] = {"ts": now, "rows": rows}
    return rows


def fetch_market_indicators(days: int = 365, ttl_sec: int = 300) -> dict:
    """Fetch the broad-market condition series (SPY, VIX, VIX3M) for `analyze_market`.

    Returns a dict of {"spy", "vix", "vix3m"} -> daily close Series. A failed or
    unavailable symbol yields an empty Series rather than raising, so one bad
    feed never sinks the whole read. Results are cached for `ttl_sec` seconds.
    """
    now = datetime.now(timezone.utc)
    cached = _market_cache["data"]
    cached_ts = _market_cache["ts"]
    if cached is not None and cached_ts is not None and (now - cached_ts).total_seconds() < ttl_sec:
        return cached

    data: dict[str, pd.Series] = {}
    for key, symbol in (("spy", SPY_SYMBOL), ("vix", VIX_SYMBOL), ("vix3m", VIX3M_SYMBOL)):
        try:
            data[key] = fetch_close_series(symbol, days)
        except Exception as exc:
            log_fetch_failure(
                "market indicators",
                [("yfinance", exc)],
                symbol=symbol,
                consequence=f"using empty {key} series",
            )
            data[key] = pd.Series(dtype=float)

    _market_cache["ts"] = now
    _market_cache["data"] = data
    return data


_beta_cache: dict[str, dict] = {}


def fetch_market_beta(symbol: str, days: int = 180, ttl_sec: int = 3600) -> "dict | None":
    """Long-term beta of `symbol` to the broad market (SPY) from daily returns.

    Beta is the slope of the ticker's daily returns regressed on SPY's over the
    trailing `days` -- the standard "how much of this name's move is just the
    market" coefficient. It anchors the market-neutral momentum read, which
    subtracts beta*market from the ticker's move to isolate its own push. Changes
    slowly, so results are cached per symbol for `ttl_sec`. Returns None when
    there isn't enough overlapping history to estimate it.
    """
    now = datetime.now(timezone.utc)
    cached = _beta_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < ttl_sec:
        return cached["value"]

    value: "dict | None" = None
    try:
        sym_close = fetch_close_series(symbol, days)
        mkt_close = fetch_close_series(SPY_SYMBOL, days)
        sym_ret = sym_close.pct_change().dropna()
        mkt_ret = mkt_close.pct_change().dropna()
        joined = pd.concat([sym_ret, mkt_ret], axis=1, join="inner").dropna()
        joined.columns = ["sym", "mkt"]
        if len(joined) >= 20:
            mkt_var = float(joined["mkt"].var())
            if mkt_var > 0:
                cov = float(joined["sym"].cov(joined["mkt"]))
                value = {"beta": cov / mkt_var, "window_days": days, "obs": len(joined)}
    except Exception as exc:
        log_fetch_failure("market beta", [("yfinance", exc)], symbol=symbol)
        value = None

    _beta_cache[symbol] = {"ts": now, "value": value}
    return value


def fetch_dividends(symbol: str, days: int) -> pd.Series:
    """Fetch dividend payouts for `symbol` over the trailing `days`. Raises on failure."""
    try:
        div = yf.Ticker(symbol).dividends
    except Exception as exc:
        log_fetch_failure("dividends", [("yfinance", exc)], symbol=symbol)
        raise
    if div.empty:
        log_fetch("dividends", "yfinance", symbol=symbol, detail="no payouts on record")
        return div
    cutoff = pd.Timestamp.now(tz=div.index.tz) - pd.Timedelta(days=days)
    div = div[div.index >= cutoff]
    log_fetch("dividends", "yfinance", symbol=symbol, detail=f"{len(div)} payouts over {days}d")
    return div


def fetch_earnings_dates(symbol: str, days: int) -> pd.DataFrame:
    """Fetch past and upcoming earnings dates for `symbol` over the trailing `days`."""
    try:
        earnings = yf.Ticker(symbol).get_earnings_dates(limit=20)
    except Exception as exc:
        log_fetch_failure(
            "earnings dates",
            [("yfinance", exc)],
            symbol=symbol,
            consequence="returning no earnings dates",
        )
        return pd.DataFrame()
    if earnings is None or earnings.empty:
        log_fetch("earnings dates", "yfinance", symbol=symbol, detail="none on record")
        return pd.DataFrame()
    log_fetch("earnings dates", "yfinance", symbol=symbol, detail=f"{len(earnings)} dates")
    cutoff = pd.Timestamp.now(tz=earnings.index.tz) - pd.Timedelta(days=days)
    return earnings[earnings.index >= cutoff]


def fetch_static_analysis(symbol: str) -> dict:
    """Fetch the raw inputs (P/E, growth, dividend yield) for a simple static valuation estimate."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        growth_rate = info.get("earningsGrowth") or info.get("revenueGrowth")
        log_fetch("fundamentals", "yfinance quoteSummary (.info)", symbol=symbol)
        return {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("trailingAnnualDividendYield"),
            "growth_rate": growth_rate,
        }
    except Exception as exc:
        # Yahoo Finance periodically restricts the quoteSummary endpoint; fall back
        # to computing dividend yield from the chart endpoint (less restricted).
        info_failure = ("yfinance quoteSummary (.info)", exc)
        ticker = None

    dividend_yield = None
    try:
        last_price = ticker.fast_info.last_price
        annual_div = float(ticker.dividends.last("365D").sum())
        if last_price and annual_div:
            dividend_yield = annual_div / last_price
        log_fetch(
            "fundamentals",
            "yfinance fast_info + dividends",
            symbol=symbol,
            detail="dividend yield only",
            failures=[info_failure],
        )
    except Exception as exc:
        log_fetch_failure(
            "fundamentals",
            [info_failure, ("yfinance fast_info + dividends", exc)],
            symbol=symbol,
            consequence="all fundamentals unavailable",
        )

    return {
        "pe_ratio": None,
        "forward_pe": None,
        "dividend_yield": dividend_yield,
        "growth_rate": None,
    }


PRICE_TARGET_TTL_SEC = 6 * 3600
_price_target_cache: dict = {}

_TARGET_COLUMNS = ["firm", "date", "target"]


def _fetch_target_actions(symbol: str) -> pd.DataFrame:
    """Raw dated analyst price-target actions for `symbol` from yfinance
    `upgrades_downgrades` (Yahoo's feed of rating/target changes). Analyst
    actions land at most a few times a day, so the parsed frame is cached
    for several hours. Raises on fetch failure; caller decides the fallback.
    """
    now = datetime.now(timezone.utc)
    cached = _price_target_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < PRICE_TARGET_TTL_SEC:
        return cached["data"]

    actions = yf.Ticker(symbol).upgrades_downgrades
    if actions is None or actions.empty or "currentPriceTarget" not in actions.columns:
        df = pd.DataFrame(columns=_TARGET_COLUMNS)
    else:
        df = actions.reset_index().rename(
            columns={"GradeDate": "date", "Firm": "firm", "currentPriceTarget": "target"}
        )
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        # Rows without a published target come through as 0.
        df = df[df["target"] > 0][_TARGET_COLUMNS].sort_values("date")

    _price_target_cache[symbol] = {"ts": now, "data": df}
    return df


def fetch_price_target_history(symbol: str, days: int, max_firms: int = 8) -> pd.DataFrame:
    """Piecewise history of expert (analyst firm) price targets for `symbol`
    over the trailing `days`.

    Returns a DataFrame with columns [firm, date, target]: each firm's target
    changes inside the window, plus the firm's standing target carried in at
    the window start so its line spans the whole shown range. Limited to the
    `max_firms` most recently active firms. Never raises -- returns an empty
    frame when the feed is unavailable.
    """
    try:
        actions = _fetch_target_actions(symbol)
    except Exception as exc:
        log_fetch_failure(
            "price targets",
            [("yfinance upgrades_downgrades", exc)],
            symbol=symbol,
            consequence="no expert target lines",
        )
        return pd.DataFrame(columns=_TARGET_COLUMNS)
    if actions.empty:
        log_fetch("price targets", "yfinance upgrades_downgrades", symbol=symbol, detail="no target actions on record")
        return pd.DataFrame(columns=_TARGET_COLUMNS)

    window_start = pd.Timestamp.now() - pd.Timedelta(days=days)
    latest_by_firm = actions.groupby("firm")["date"].max().sort_values(ascending=False)
    firms = latest_by_firm.head(max_firms).index

    rows: list[pd.DataFrame] = []
    for firm in firms:
        events = actions[actions["firm"] == firm]
        inside = events[events["date"] >= window_start]
        before = events[events["date"] < window_start]
        if not before.empty:
            # The firm's target standing when the window opens.
            carry = before.iloc[[-1]].copy()
            carry["date"] = window_start
            inside = pd.concat([carry, inside])
        if not inside.empty:
            rows.append(inside)

    if not rows:
        log_fetch("price targets", "yfinance upgrades_downgrades", symbol=symbol, detail=f"no targets within {days}d")
        return pd.DataFrame(columns=_TARGET_COLUMNS)
    result = pd.concat(rows).sort_values(["firm", "date"]).reset_index(drop=True)
    log_fetch(
        "price targets",
        "yfinance upgrades_downgrades",
        symbol=symbol,
        detail=f"{len(result)} target points from {result['firm'].nunique()} firms over {days}d",
    )
    return result


# Bulge-bracket firms whose standing price target we surface individually
# (alongside the yfinance consensus) in the pre-market read and the agent tool.
ANALYST_TARGET_FIRMS: tuple[str, ...] = ("UBS", "Morgan Stanley", "Barclays")

ANALYST_TARGETS_TTL_SEC = 6 * 3600
# Cache holds the RAW targets (consensus prices + per-firm targets + Yahoo's
# own current price); the price-relative upside fields are recomputed on every
# call from the freshest `current_price` the caller passes.
_analyst_targets_cache: dict = {}


def _upside_pct(target: Optional[float], price: Optional[float]) -> Optional[float]:
    """Percent upside from `price` to `target`, rounded, or None if either is missing."""
    if target is None or not price:
        return None
    return round((float(target) / float(price) - 1.0) * 100.0, 1)


def _latest_firm_targets(actions: pd.DataFrame, firms: tuple[str, ...]) -> dict:
    """Each named firm's most recent standing target from the actions feed
    (columns [firm, date, target]). Matches the firm name case-insensitively as
    a substring, so 'Barclays' also picks up 'Barclays Capital'."""
    result: dict = {}
    if actions is None or actions.empty:
        return result
    firm_series = actions["firm"].astype(str)
    for firm in firms:
        matched = actions[firm_series.str.contains(firm, case=False, na=False, regex=False)]
        if matched.empty:
            continue
        latest = matched.sort_values("date").iloc[-1]
        result[firm] = {
            "target": round(float(latest["target"]), 2),
            "date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
            "as_reported": str(latest["firm"]),
        }
    return result


def _fetch_raw_analyst_targets(symbol: str) -> dict:
    """Cached raw analyst-target inputs for `symbol`: the yfinance consensus
    (mean/median/high/low, analyst count, recommendation) from quoteSummary,
    Yahoo's own current price, and the standing target of each tracked firm from
    the upgrades/downgrades feed. Never raises -- missing pieces come back None/
    empty. Cached for several hours (targets move at most a few times a day)."""
    now = datetime.now(timezone.utc)
    cached = _analyst_targets_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < ANALYST_TARGETS_TTL_SEC:
        return cached["data"]

    consensus: dict = {
        "mean": None, "median": None, "high": None, "low": None,
        "num_analysts": None, "recommendation": None,
    }
    info_price = None
    failures: list[tuple[str, object]] = []
    try:
        info = yf.Ticker(symbol).info
        consensus = {
            "mean": info.get("targetMeanPrice"),
            "median": info.get("targetMedianPrice"),
            "high": info.get("targetHighPrice"),
            "low": info.get("targetLowPrice"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
        }
        info_price = info.get("currentPrice")
    except Exception as exc:
        failures.append(("yfinance quoteSummary (.info)", exc))

    try:
        actions = _fetch_target_actions(symbol)
    except Exception as exc:
        failures.append(("yfinance upgrades_downgrades", exc))
        actions = pd.DataFrame(columns=_TARGET_COLUMNS)
    firm_targets = _latest_firm_targets(actions, ANALYST_TARGET_FIRMS)

    have_consensus = consensus.get("mean") is not None
    if not have_consensus and not firm_targets:
        log_fetch_failure(
            "analyst targets",
            failures or [("yfinance", "no consensus or firm targets on record")],
            symbol=symbol,
            consequence="no analyst price targets",
        )
    else:
        log_fetch(
            "analyst targets",
            "yfinance (.info + upgrades_downgrades)",
            symbol=symbol,
            detail=f"consensus={'yes' if have_consensus else 'no'}, {len(firm_targets)} tracked firms",
            failures=failures,
        )

    data = {"consensus": consensus, "firm_targets": firm_targets, "info_current_price": info_price}
    _analyst_targets_cache[symbol] = {"ts": now, "data": data}
    return data


def fetch_analyst_targets(symbol: str, current_price: Optional[float] = None) -> dict:
    """Current analyst price targets for `symbol` with the actionable read a
    trader needs: the yfinance consensus (mean/median/high/low across every
    covering analyst) and the standing target from each tracked bulge-bracket
    firm (UBS, Morgan Stanley, Barclays), each annotated with the implied upside
    vs the current price. `current_price` overrides Yahoo's (pass the live
    streamed price); it falls back to Yahoo's own quote when omitted. Never
    raises. Returns a dict with `consensus`, `firms`, `current_price`, a
    one-line `summary`, and a list of actionable `insights`."""
    raw = _fetch_raw_analyst_targets(symbol)
    price = current_price if current_price else raw.get("info_current_price")

    c = raw["consensus"]
    consensus = {
        "mean": c.get("mean"),
        "median": c.get("median"),
        "high": c.get("high"),
        "low": c.get("low"),
        "num_analysts": c.get("num_analysts"),
        "recommendation": c.get("recommendation"),
        "mean_upside_pct": _upside_pct(c.get("mean"), price),
        "high_upside_pct": _upside_pct(c.get("high"), price),
        "low_upside_pct": _upside_pct(c.get("low"), price),
    }

    firms: dict = {}
    for name, ft in raw["firm_targets"].items():
        firms[name] = {**ft, "upside_pct": _upside_pct(ft.get("target"), price)}

    result = {
        "symbol": symbol,
        "current_price": price,
        "consensus": consensus,
        "firms": firms,
    }
    summary, insights = _summarize_analyst_targets(result)
    result["summary"] = summary
    result["insights"] = insights
    return result


def _summarize_analyst_targets(data: dict) -> "tuple[str, list[str]]":
    """Turn analyst targets into a one-line summary and a list of actionable
    insights (upside remaining, price outside the Street range, dispersion)."""
    price = data.get("current_price")
    cons = data.get("consensus") or {}
    mean, high, low = cons.get("mean"), cons.get("high"), cons.get("low")
    summary_parts: list[str] = []
    insights: list[str] = []

    if mean is not None:
        up = cons.get("mean_upside_pct")
        n = cons.get("num_analysts")
        head = f"Consensus mean {mean:g}"
        if up is not None:
            head += f" ({up:+.1f}%)"
        if n:
            head += f" from {n} analysts"
        summary_parts.append(head)
        if up is not None:
            if up <= 0:
                insights.append(
                    f"Price sits {abs(up):.1f}% ABOVE the consensus mean target ({mean:g}) -- "
                    "Street-implied upside is exhausted; treat further rallies as extended."
                )
            elif up >= 15:
                insights.append(
                    f"Consensus mean target implies {up:+.1f}% upside -- the Street still sees "
                    "meaningful room above the current price."
                )
            else:
                insights.append(
                    f"Only {up:+.1f}% to the consensus mean target -- limited Street upside remaining."
                )

    if high is not None and price and price > high:
        insights.append(
            f"Price is above the HIGHEST analyst target ({high:g}) -- no covering analyst sees "
            "further upside; richly valued vs the Street."
        )
    if low is not None and price and price < low:
        insights.append(
            f"Price is below the LOWEST analyst target ({low:g}) -- trading under the entire Street "
            "range; either a value gap or analysts are behind negative news."
        )
    if high is not None and low is not None and mean:
        dispersion = (high - low) / mean * 100
        if dispersion >= 40:
            insights.append(
                f"Wide target dispersion ({dispersion:.0f}% of mean, {low:g}-{high:g}) -- analysts "
                "strongly disagree; the consensus is a weak anchor."
            )

    for name, f in (data.get("firms") or {}).items():
        up = f.get("upside_pct")
        part = f"{name} {f['target']:g}"
        if up is not None:
            part += f" ({up:+.1f}%)"
        summary_parts.append(part)

    if not summary_parts:
        return "No analyst price targets available.", []
    return "; ".join(summary_parts) + ".", insights


SMART_MONEY_TTL_SEC = 6 * 3600
_smart_money_cache: dict = {}


def _net_insider_shares(purchases) -> "dict | None":
    """Parse yfinance `insider_purchases` (a 6-month buy/sell summary) into a net
    direction. The frame indexes rows like 'Total Shares Purchased' / 'Sold' /
    'Net Shares Purchased (Sold)' against a single value column."""
    try:
        frame = purchases
        if frame is None or getattr(frame, "empty", True):
            return None
        rows = {str(k).strip().lower(): v for k, v in frame.iloc[:, 0].items()}
        bought = rows.get("total shares purchased")
        sold = rows.get("total shares sold")
        net = rows.get("net shares purchased (sold)")
        if net is None and bought is not None and sold is not None:
            net = float(bought) - float(sold)
        if net is None:
            return None
        net = float(net)
        return {
            "net_shares_6mo": int(net),
            "bought_6mo": int(bought) if bought is not None else None,
            "sold_6mo": int(sold) if sold is not None else None,
            "direction": "buying" if net > 0 else ("selling" if net < 0 else "flat"),
        }
    except Exception:
        logger.warning("Could not parse insider purchases", exc_info=True)
        return None


def fetch_smart_money_flow(symbol: str) -> dict:
    """The institutional 'smart money' footprint for `symbol` from free yfinance data.

    Pulls three slow-moving but high-signal disclosures Yahoo aggregates from SEC
    filings: aggregate ownership breakdown (% held by insiders vs institutions),
    net insider buying/selling over the trailing 6 months (Form 4), and the
    largest institutional holders with their quarter-over-quarter share changes
    (13F). These are quarterly/Form-4 cadence -- not intraday signals -- so the
    result is cached for several hours. A net insider/institutional accumulation
    behind a bullish demand block corroborates the technical Smart Money read;
    distribution is a caution flag. Never raises -- missing fields come back None.
    """
    now = datetime.now(timezone.utc)
    cached = _smart_money_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < SMART_MONEY_TTL_SEC:
        return cached["data"]

    ticker = yf.Ticker(symbol)
    result: dict = {
        "symbol": symbol,
        "insiders_pct_held": None,
        "institutions_pct_held": None,
        "institutions_count": None,
        "insider_flow": None,
        "top_institutional_holders": [],
        "institutional_net_pct_change": None,
    }

    failures: list[tuple[str, object]] = []
    try:
        mh = ticker.major_holders
        if mh is not None and not mh.empty:
            col = mh.iloc[:, 0]
            for label, value in col.items():
                key = str(label).strip().lower()
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if key == "insiderspercentheld":
                    result["insiders_pct_held"] = round(val, 4)
                elif key == "institutionspercentheld":
                    result["institutions_pct_held"] = round(val, 4)
                elif key == "institutionscount":
                    result["institutions_count"] = int(val)
    except Exception as exc:
        failures.append(("yfinance major_holders", exc))

    try:
        result["insider_flow"] = _net_insider_shares(ticker.insider_purchases)
    except Exception as exc:
        failures.append(("yfinance insider_purchases", exc))

    try:
        inst = ticker.institutional_holders
        if inst is not None and not inst.empty:
            net_change = 0.0
            for _, row in inst.head(10).iterrows():
                pct_change = row.get("pctChange")
                if pct_change is not None and pd.notna(pct_change):
                    net_change += float(pct_change)
                result["top_institutional_holders"].append({
                    "holder": str(row.get("Holder", "")),
                    "shares": int(row["Shares"]) if pd.notna(row.get("Shares")) else None,
                    "pct_held": round(float(row["pctHeld"]), 4) if pd.notna(row.get("pctHeld")) else None,
                    "pct_change": round(float(pct_change), 4) if pct_change is not None and pd.notna(pct_change) else None,
                })
            result["top_institutional_holders"] = result["top_institutional_holders"][:5]
            result["institutional_net_pct_change"] = round(net_change, 4)
    except Exception as exc:
        failures.append(("yfinance institutional_holders", exc))

    if len(failures) == 3:
        log_fetch_failure(
            "smart money flow",
            failures,
            symbol=symbol,
            consequence="no institutional ownership data",
        )
    else:
        log_fetch(
            "smart money flow",
            "yfinance (SEC filings)",
            symbol=symbol,
            detail=f"{3 - len(failures)}/3 datasets",
            failures=failures,
        )

    result["summary"] = _summarize_smart_money_flow(result)
    _smart_money_cache[symbol] = {"ts": now, "data": result}
    return result


def _summarize_smart_money_flow(flow: dict) -> str:
    parts: list[str] = []
    inst_pct = flow.get("institutions_pct_held")
    ins_pct = flow.get("insiders_pct_held")
    if inst_pct is not None:
        parts.append(f"Institutions hold {inst_pct * 100:.1f}%" + (f" across {flow['institutions_count']} holders" if flow.get("institutions_count") else "") + ".")
    if ins_pct is not None:
        parts.append(f"Insiders hold {ins_pct * 100:.1f}%.")
    insider = flow.get("insider_flow")
    if insider:
        parts.append(f"Insiders net {insider['direction']} {abs(insider['net_shares_6mo']):,} shares over 6mo (Form 4).")
    net = flow.get("institutional_net_pct_change")
    if net is not None:
        lean = "accumulating" if net > 0 else ("distributing" if net < 0 else "flat")
        parts.append(f"Top institutions {lean} (net {net * 100:+.1f}% q/q, 13F).")
    return " ".join(parts) if parts else "No institutional ownership data available."


def estimate_total_return(dividend_yield: Optional[float], growth_rate: Optional[float]) -> Optional[float]:
    """Rough estimate of annual total return on the asset: dividend yield + earnings/revenue growth."""
    if dividend_yield is None or growth_rate is None:
        return None
    return dividend_yield + growth_rate


def estimate_dividend_return_10y(
    dividend_yield: Optional[float], growth_rate: Optional[float], years: int = 10
) -> Optional[float]:
    """Cumulative dividends collected over `years`, as a fraction of the current price.

    Assumes the dividend grows annually at `growth_rate` from today's yield.
    """
    if dividend_yield is None or growth_rate is None:
        return None
    return sum(dividend_yield * (1 + growth_rate) ** t for t in range(years))
