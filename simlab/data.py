"""Dataset download and storage for SimLab.

Everything a simulated session needs is downloaded once and kept in a local
file store, deduplicated at the (feed, symbol, trading day) level so
overlapping datasets never re-download or re-store the same day:

    data/simlab/
      store/
        bars/{FEED}/{SYMBOL}/{YYYY-MM-DD}.json.gz   1-minute bars, 04:00-20:00 ET
        daily/{FEED}/{SYMBOL}.json.gz               daily bars (range in the payload)
        news/{SYMBOL}/{YYYY-MM-DD}.json.gz          news articles created that day
        market/indicators.json.gz                   SPY/VIX/VIX3M daily closes
      datasets.json                                 named dataset manifest

A *dataset* is a named bundle: symbols + an inclusive date range + the feed its
bars came from. Creating one walks the range and fills only the store files
that are missing; deleting one only removes the manifest entry (the store is
shared).

Why the feed is part of the key
-------------------------------
`yfinance`, `iex` and `sip` are not three routes to the same numbers. IEX is
one venue -- about 4% of consolidated volume on a large-cap name -- so its
minute bars carry different closes (a few cents), different volumes (~25x
smaller) and sometimes an extra or missing bar. Any agent whose rules are
thresholds over those bars can trade a different day on each tape, and a model
reading volume or volatility features sees genuinely different inputs.

Keying the store on (symbol, day) alone made that invisible *and*
unfixable: a day already downloaded on IEX satisfied the "already stored"
check, so asking for `sip` silently reused the IEX bars and produced a dataset
that claimed a tape it did not have. The feed is therefore part of the path,
recorded on the manifest entry, and carried into the run record.

Days downloaded before feeds were tracked were all `iex` (it was the only
default), and they stay where they are -- the readers fall back to the pre-feed
layout for `iex` rather than forcing a re-download.

Bars/news keep Alpaca's native dict shapes ({"t","o","h","l","c","v",...}) so
everything downstream (SymbolState, technical_analysis) consumes them as-is;
the yfinance fetchers convert into that same shape.
"""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from agent_stonks.config import DATA_REST
from agent_stonks.market_hours import MARKET_TZ

SIMLAB_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab"
STORE_DIR = SIMLAB_DIR / "store"
MANIFEST_PATH = SIMLAB_DIR / "datasets.json"

# Stored minute-bar window per trading day, ET: full pre-market through
# post-market so premarket reads and opening tactics have real tape.
DAY_START_ET = time(4, 0)
DAY_END_ET = time(20, 0)

# Daily-bar history stored per symbol: enough for analyze_daily_trend (1y),
# order blocks, and the ADV/rvol baselines at any simulated day in range.
DAILY_LOOKBACK_DAYS = 420

MARKET_INDICATOR_SYMBOLS = {"spy": "SPY", "vix": "^VIX", "vix3m": "^VIX3M"}

# The tapes a dataset's bars can come from.
#
# `yfinance` is the default and the source of every newly downloaded minute
# bar: consolidated-tape OHLCV, free, no Alpaca subscription, and the same
# source the live volume tools read (agent_stonks.historical), so a simulated
# volume ratio is like-for-like with the live one. Its cost is reach -- Yahoo
# serves 1-minute history for the last 30 days only (see YF_MINUTE_WINDOW_DAYS)
# and publishes no per-bar VWAP, so bars carry no `vw` field.
#
# `iex` and `sip` are Alpaca's tapes, kept so every dataset downloaded before
# this still loads and can still be re-run. `iex` is the free single-venue feed
# and what every day stored before the feed was tracked contains. `sip` is
# Alpaca's consolidated tape and needs a paid data subscription; without one
# Alpaca rejects the request rather than quietly downgrading.
FEEDS = ("yfinance", "iex", "sip")
DEFAULT_FEED = "yfinance"

# The feed to assume where none was recorded: the pre-feed store layout and
# manifest entries written before `feed` existed. Those are all Alpaca IEX --
# it was the downloader's only option at the time -- and must never be read as
# anything else.
LEGACY_FEED = "iex"

# Yahoo serves 1-minute bars for roughly the last 30 calendar days and refuses
# anything older ("The requested range must be within the last 30 days"), so a
# yfinance dataset cannot reach further back than that. Alpaca has no such
# limit, which is why `iex`/`sip` remain the only way to build a dataset over
# an older window.
YF_MINUTE_WINDOW_DAYS = 30

_manifest_lock = threading.Lock()

ProgressCb = Callable[[str], None]


def _noop_progress(_msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Low-level store
# ---------------------------------------------------------------------------

def _read_gz(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write_gz(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(path)


def bars_path(symbol: str, day: date, feed: str = DEFAULT_FEED) -> Path:
    """Where a freshly downloaded day is written."""
    return STORE_DIR / "bars" / feed / symbol.upper() / f"{day.isoformat()}.json.gz"


def news_path(symbol: str, day: date) -> Path:
    # No feed: news is the same articles whichever tape the bars came from.
    return STORE_DIR / "news" / symbol.upper() / f"{day.isoformat()}.json.gz"


def daily_path(symbol: str, feed: str = DEFAULT_FEED) -> Path:
    return STORE_DIR / "daily" / feed / f"{symbol.upper()}.json.gz"


def market_path() -> Path:
    return STORE_DIR / "market" / "indicators.json.gz"


def _legacy_bars_path(symbol: str, day: date) -> Path:
    return STORE_DIR / "bars" / symbol.upper() / f"{day.isoformat()}.json.gz"


def _legacy_daily_path(symbol: str) -> Path:
    return STORE_DIR / "daily" / f"{symbol.upper()}.json.gz"


def stored_bars_path(symbol: str, day: date, feed: str = DEFAULT_FEED) -> Path:
    """Where this day actually is on disk, feed-scoped or pre-feed.

    Everything downloaded before the feed was tracked is IEX and sits in the
    old flat layout. Falling back to it (for `iex` only) keeps those datasets
    working without a re-download, and cannot mislabel anything: `sip` and
    `yfinance` never resolve to a file that was fetched as `iex`.
    """
    path = bars_path(symbol, day, feed)
    if not path.exists() and feed == LEGACY_FEED:
        legacy = _legacy_bars_path(symbol, day)
        if legacy.exists():
            return legacy
    return path


def stored_daily_path(symbol: str, feed: str = DEFAULT_FEED) -> Path:
    path = daily_path(symbol, feed)
    if not path.exists() and feed == LEGACY_FEED:
        legacy = _legacy_daily_path(symbol)
        if legacy.exists():
            return legacy
    return path


def load_day_bars(symbol: str, day: date, feed: str = DEFAULT_FEED) -> list[dict]:
    """Stored 1-minute bars for one (symbol, day, feed); [] when the day has no
    session (holiday) or hasn't been downloaded on that feed."""
    path = stored_bars_path(symbol, day, feed)
    if not path.exists():
        return []
    return _read_gz(path)  # type: ignore[return-value]


def load_daily_bars(symbol: str, feed: str = DEFAULT_FEED) -> list[dict]:
    path = stored_daily_path(symbol, feed)
    if not path.exists():
        return []
    return _read_gz(path).get("bars", [])  # type: ignore[union-attr]


def load_news(symbol: str, day: date) -> list[dict]:
    path = news_path(symbol, day)
    if not path.exists():
        return []
    return _read_gz(path)  # type: ignore[return-value]


def load_market_indicators() -> dict[str, list[dict]]:
    """{"spy"|"vix"|"vix3m": [{"date": "YYYY-MM-DD", "close": float}, ...]}"""
    path = market_path()
    if not path.exists():
        return {}
    return _read_gz(path)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    symbols: list[str]
    start: str  # inclusive, YYYY-MM-DD
    end: str  # inclusive, YYYY-MM-DD
    created_at: str = ""
    # Trading days (YYYY-MM-DD) that actually have bars for at least one
    # symbol -- weekends/holidays in the range are absent.
    days: list[str] = field(default_factory=list)
    # Which tape these bars are. Defaulted rather than required so manifest
    # entries written before feeds were tracked still load -- and `iex` is the
    # honest value for them, since it was the only feed the downloader used.
    feed: str = LEGACY_FEED

    def date_range(self) -> tuple[date, date]:
        return date.fromisoformat(self.start), date.fromisoformat(self.end)


def list_datasets() -> list[Dataset]:
    if not MANIFEST_PATH.exists():
        return []
    raw = json.loads(MANIFEST_PATH.read_text())
    return [Dataset(**entry) for entry in raw]


def get_dataset(name: str) -> Optional[Dataset]:
    return next((d for d in list_datasets() if d.name == name), None)


def _save_manifest(datasets: list[Dataset]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps([asdict(d) for d in datasets], indent=2))


def delete_dataset(name: str) -> None:
    """Remove a dataset from the manifest. Store files are shared across
    datasets and deliberately kept."""
    with _manifest_lock:
        _save_manifest([d for d in list_datasets() if d.name != name])


# ---------------------------------------------------------------------------
# Alpaca fetchers (range-based, paginated -- rest.py's live helpers are
# anchored to "now", which is exactly what a downloader must not be)
# ---------------------------------------------------------------------------

def _headers(key: str, secret: str) -> dict[str, str]:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _paged_get(url: str, params: dict, key: str, secret: str, item_key: str, symbol: str) -> list:
    """Follow Alpaca's next_page_token pagination until exhausted."""
    out: list = []
    token: Optional[str] = None
    while True:
        page_params = dict(params)
        if token:
            page_params["page_token"] = token
        r = requests.get(url, headers=_headers(key, secret), params=page_params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        container = payload.get(item_key) or {}
        items = container.get(symbol, []) if isinstance(container, dict) else container
        out.extend(items or [])
        token = payload.get("next_page_token")
        if not token:
            return out


def _fetch_minute_bars_day_alpaca(
    symbol: str, day: date, key: str, secret: str, feed: str
) -> list[dict]:
    start = datetime.combine(day, DAY_START_ET, tzinfo=MARKET_TZ).astimezone(timezone.utc)
    end = datetime.combine(day, DAY_END_ET, tzinfo=MARKET_TZ).astimezone(timezone.utc)
    return _paged_get(
        f"{DATA_REST}/v2/stocks/bars",
        dict(
            symbols=symbol,
            timeframe="1Min",
            start=start.isoformat(),
            end=end.isoformat(),
            limit=10000,
            feed=feed,
        ),
        key,
        secret,
        "bars",
        symbol,
    )


def _fetch_daily_bars_range_alpaca(
    symbol: str, start: date, end: date, key: str, secret: str, feed: str
) -> list[dict]:
    return _paged_get(
        f"{DATA_REST}/v2/stocks/bars",
        dict(
            symbols=symbol,
            timeframe="1Day",
            start=datetime.combine(start, time(0, 0), tzinfo=timezone.utc).isoformat(),
            end=datetime.combine(end, time(23, 59), tzinfo=timezone.utc).isoformat(),
            limit=10000,
            feed=feed,
        ),
        key,
        secret,
        "bars",
        symbol,
    )


# ---------------------------------------------------------------------------
# yfinance fetchers (the default source for new datasets)
#
# yfinance hands back a DataFrame; the store speaks Alpaca's bar dicts, so
# everything here converts. Two differences from an Alpaca day are permanent
# and worth knowing before comparing runs across feeds:
#
# - No `vw`. Yahoo publishes no per-bar VWAP, so yfinance bars omit the key
#   rather than carry a fabricated one. `technical_analysis.analyze_intraday`
#   and `charts` already branch on its presence, so the VWAP line/note is
#   simply absent on a yfinance run instead of being wrong.
# - Zero volume outside 09:30-16:00. `prepost=True` is required to reach the
#   stored 04:00-20:00 window, and Yahoo does return real, moving prices for
#   those minutes -- but reports every one of them at volume 0. The bars are
#   kept: their prices are what a pre-market read or an opening tactic acts on,
#   and dropping them would cut the stored day down to the regular session.
#   The consequence is that volume-threshold rules never fire outside regular
#   hours on this feed, where on Alpaca they can.
# ---------------------------------------------------------------------------

def _num(value: object, default: float = 0.0) -> float:
    """float(value), with NaN/None/junk collapsing to `default` -- `x or 0.0`
    does not, since NaN is truthy."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _yf_frame(symbol: str, start: date, end: date, interval: str, prepost: bool):
    """yfinance download, flattened to single-level columns. `end` exclusive."""
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,
        prepost=prepost,
        progress=False,
    )
    if frame is None or frame.empty:
        return None
    if hasattr(frame.columns, "get_level_values") and frame.columns.nlevels > 1:
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _fetch_minute_bars_day_yfinance(symbol: str, day: date) -> list[dict]:
    """All 1-minute bars for one trading day from yfinance, 04:00-20:00 ET.

    Returns [] for a non-trading day and for any day outside Yahoo's 30-day
    1-minute window -- the same "nothing stored for this day" signal a holiday
    produces, which `create_dataset` already handles.
    """
    frame = _yf_frame(symbol, day, day + timedelta(days=1), "1m", prepost=True)
    if frame is None:
        return []
    index = frame.index
    if index.tz is None:
        index = index.tz_localize(MARKET_TZ)
    index = index.tz_convert(timezone.utc)

    bars: list[dict] = []
    for ts, row in zip(index, frame.itertuples(index=False)):
        if row.Close != row.Close:  # NaN: no price for this minute at all
            continue
        # Yahoo can serve a few minutes either side of the requested day around
        # DST changes; the store's contract is one ET day per file.
        if ts.astimezone(MARKET_TZ).date() != day:
            continue
        bars.append(
            {
                "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": float(row.Open),
                "h": float(row.High),
                "l": float(row.Low),
                "c": float(row.Close),
                "v": _num(row.Volume),
            }
        )
    return bars


def _fetch_daily_bars_range_yfinance(symbol: str, start: date, end: date) -> list[dict]:
    """Daily bars over [start, end] inclusive, in the stored bar shape.

    Sourced from yfinance so a yfinance dataset's daily baselines (ADV, rvol,
    prev close) are measured on the same tape as its minute bars.
    """
    frame = _yf_frame(symbol, start, end + timedelta(days=1), "1d", prepost=False)
    if frame is None:
        return []
    return [
        {
            # Alpaca stamps a daily bar at the session open in UTC; the store
            # only ever reads the date part, and `SimMarket.daily_bars_at`
            # builds today's partial bar with the same 05:00Z stamp.
            "t": f"{ts.date().isoformat()}T05:00:00Z",
            "o": float(row.Open),
            "h": float(row.High),
            "l": float(row.Low),
            "c": float(row.Close),
            "v": _num(row.Volume),
        }
        for ts, row in zip(frame.index, frame.itertuples(index=False))
        if row.Close == row.Close  # drop NaN rows (non-trading days)
    ]


# ---------------------------------------------------------------------------
# Feed dispatch -- the one place that decides which vendor a feed means
# ---------------------------------------------------------------------------

def fetch_minute_bars_day(
    symbol: str, day: date, key: str = "", secret: str = "", feed: str = DEFAULT_FEED
) -> list[dict]:
    """All 1-minute bars for one trading day, 04:00-20:00 ET, from `feed`.

    `key`/`secret` are only used by the Alpaca feeds; the yfinance default
    needs no credentials.
    """
    if feed == "yfinance":
        return _fetch_minute_bars_day_yfinance(symbol, day)
    return _fetch_minute_bars_day_alpaca(symbol, day, key, secret, feed)


def fetch_daily_bars_range(
    symbol: str, start: date, end: date, key: str = "", secret: str = "",
    feed: str = DEFAULT_FEED,
) -> list[dict]:
    if feed == "yfinance":
        return _fetch_daily_bars_range_yfinance(symbol, start, end)
    return _fetch_daily_bars_range_alpaca(symbol, start, end, key, secret, feed)


def fetch_news_day(symbol: str, day: date, key: str, secret: str) -> list[dict]:
    """News articles for `symbol` created during `day` (UTC)."""
    start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    r = requests.get(
        f"{DATA_REST}/v1beta1/news",
        headers=_headers(key, secret),
        params=dict(
            symbols=symbol, start=start.isoformat(), end=end.isoformat(), limit=50, sort="desc"
        ),
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("news", [])


def fetch_market_indicator_closes(start: date, end: date) -> dict[str, list[dict]]:
    """Daily closes for SPY/VIX/VIX3M over [start, end] via yfinance."""
    import yfinance as yf

    out: dict[str, list[dict]] = {}
    for name, ticker in MARKET_INDICATOR_SYMBOLS.items():
        try:
            frame = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
            )
            closes = frame["Close"].dropna()
            out[name] = [
                {"date": idx.date().isoformat(), "close": float(value)}
                for idx, value in closes.items()
            ]
        except Exception:
            out[name] = []
    return out


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def weekdays(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def coverage(
    symbols: list[str], start: date, end: date, feed: str = DEFAULT_FEED
) -> dict[str, dict[str, bool]]:
    """{day -> {symbol -> already stored on this feed}} for every weekday."""
    return {
        day.isoformat(): {
            sym: stored_bars_path(sym, day, feed).exists() for sym in symbols
        }
        for day in weekdays(start, end)
    }


def _daily_covers(symbol: str, start: date, end: date, feed: str = DEFAULT_FEED) -> bool:
    """Whether the stored daily-bar file spans [start - lookback, end]."""
    path = stored_daily_path(symbol, feed)
    if not path.exists():
        return False
    meta = _read_gz(path)
    try:
        have_start = date.fromisoformat(meta["start"])
        have_end = date.fromisoformat(meta["end"])
    except (KeyError, ValueError, TypeError):
        return False
    return have_start <= start - timedelta(days=DAILY_LOOKBACK_DAYS) and have_end >= end


def _market_covers(start: date, end: date) -> bool:
    data = load_market_indicators()
    spy = data.get("spy") or []
    if not spy:
        return False
    dates = [row["date"] for row in spy]
    return dates[0] <= (start - timedelta(days=DAILY_LOOKBACK_DAYS)).isoformat() and dates[-1] >= (
        end - timedelta(days=4)
    ).isoformat()


def create_dataset(
    name: str,
    symbols: list[str],
    start: date,
    end: date,
    key: str = "",
    secret: str = "",
    feed: str = DEFAULT_FEED,
    progress: ProgressCb = _noop_progress,
) -> Dataset:
    """Create (or refresh) a named dataset, downloading only what the store
    is missing. Returns the manifest entry with its resolved trading days.

    `key`/`secret` are Alpaca credentials. On the default yfinance feed they
    are only needed for news, which is skipped (stored empty) without them; the
    Alpaca feeds require them for bars.
    """
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("dataset needs at least one symbol")
    if end < start:
        raise ValueError("dataset end date is before its start date")
    if feed not in FEEDS:
        raise ValueError(f"unknown feed {feed!r}; expected one of {', '.join(FEEDS)}")
    if feed != "yfinance" and not (key and secret):
        raise ValueError(f"the {feed!r} feed needs Alpaca credentials")

    # Yahoo's 1-minute history stops ~30 days back. Say so up front rather than
    # letting the run finish with a handful of silently empty days: those days
    # are indistinguishable from holidays once stored.
    if feed == "yfinance":
        horizon = date.today() - timedelta(days=YF_MINUTE_WINDOW_DAYS)
        if start < horizon:
            progress(
                f"warning: yfinance serves 1-minute bars only back to {horizon} "
                f"({YF_MINUTE_WINDOW_DAYS} days); days before that will store empty. "
                "Use the 'sip' feed (paid Alpaca data) for an older window."
            )

    # Daily bars first: they double as the trading-day calendar for the range.
    daily_start = start - timedelta(days=DAILY_LOOKBACK_DAYS)
    for sym in symbols:
        if _daily_covers(sym, start, end, feed):
            progress(f"daily bars {sym} [{feed}]: already stored")
            continue
        progress(f"daily bars {sym} [{feed}]: downloading {daily_start} … {end}")
        bars = fetch_daily_bars_range(sym, daily_start, end, key, secret, feed)
        _write_gz(
            daily_path(sym, feed),
            {"symbol": sym, "start": daily_start.isoformat(), "end": end.isoformat(),
             "feed": feed, "bars": bars},
        )

    if _market_covers(start, end):
        progress("market indicators (SPY/VIX/VIX3M): already stored")
    else:
        progress("market indicators (SPY/VIX/VIX3M): downloading")
        _write_gz(market_path(), fetch_market_indicator_closes(daily_start, end))

    session_days: list[str] = []
    for day in weekdays(start, end):
        day_has_bars = False
        for sym in symbols:
            # The *resolved* path, so a day already held in the pre-feed layout
            # counts as stored instead of being downloaded a second time.
            if stored_bars_path(sym, day, feed).exists():
                day_has_bars = day_has_bars or bool(load_day_bars(sym, day, feed))
                continue
            progress(f"minute bars {sym} {day} [{feed}]: downloading")
            bars = fetch_minute_bars_day(sym, day, key, secret, feed)
            _write_gz(bars_path(sym, day, feed), bars)
            day_has_bars = day_has_bars or bool(bars)

            npath = news_path(sym, day)
            if not npath.exists():
                if not (key and secret):
                    # News is Alpaca-only; a credential-free yfinance dataset
                    # simply has none rather than failing the download.
                    _write_gz(npath, [])
                    continue
                try:
                    _write_gz(npath, fetch_news_day(sym, day, key, secret))
                except Exception as exc:  # news is nice-to-have, never fatal
                    progress(f"news {sym} {day}: failed ({exc}); storing empty")
                    _write_gz(npath, [])
        if not day_has_bars:
            # Existing empty files (holiday) or fresh empty downloads.
            day_has_bars = any(load_day_bars(sym, day, feed) for sym in symbols)
        if day_has_bars:
            session_days.append(day.isoformat())

    dataset = Dataset(
        name=name,
        symbols=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        created_at=datetime.now(timezone.utc).isoformat(),
        days=session_days,
        feed=feed,
    )
    with _manifest_lock:
        existing = [d for d in list_datasets() if d.name != name]
        _save_manifest([*existing, dataset])
    progress(f"dataset '{name}' ready: {len(session_days)} trading day(s)")
    return dataset


def store_size_bytes() -> int:
    if not STORE_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in STORE_DIR.rglob("*") if p.is_file())
