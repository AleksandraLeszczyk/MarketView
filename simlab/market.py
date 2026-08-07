"""Time-windowed views over a stored dataset.

`SimMarket` owns the raw stored series for one simulation (minute bars, daily
bars, news, market indicators) and answers every "as of simulated time t"
question the engine and the patched fetchers need. All methods are pure reads
over in-memory lists -- the simulation never touches the network.

Time convention: a minute bar stamped T covers [T, T+60) -- true of both
sources the store holds, Alpaca and yfinance. It is *completed* -- and
therefore visible to the agent -- once t >= T + 60. The engine steps the clock
to exactly those completion moments.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from agent_stonks.market_hours import MARKET_TZ

from . import data

BAR_SEC = 60.0


def parse_ts(raw: object) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class _SymbolSeries:
    """One symbol's stored series for the simulated day range, pre-parsed."""

    def __init__(self, symbol: str, days: list[date], feed: str = data.DEFAULT_FEED) -> None:
        self.symbol = symbol
        self.feed = feed
        self.minute_bars: list[dict] = []
        for day in days:
            self.minute_bars.extend(data.load_day_bars(symbol, day, feed))
        self.minute_ts: list[datetime] = []
        bars_clean: list[dict] = []
        for bar in self.minute_bars:
            ts = parse_ts(bar.get("t"))
            if ts is not None:
                bars_clean.append(bar)
                self.minute_ts.append(ts)
        self.minute_bars = bars_clean

        self.daily_bars: list[dict] = data.load_daily_bars(symbol, feed)
        self.news: list[dict] = []
        for day in days:
            self.news.extend(data.load_news(symbol, day))
        self.news.sort(key=lambda a: str(a.get("created_at") or ""))
        self.news_ts: list[Optional[datetime]] = [parse_ts(a.get("created_at")) for a in self.news]


class SimMarket:
    """Dataset-backed market data for one simulation run.

    `feed` names which tape to read, and is not cosmetic: the same day on
    `yfinance`, on `iex` and on `sip` is a different set of bars, so it belongs
    to the identity of a run rather than to how the data happened to be
    fetched. See the `simlab.data` module docstring.
    """

    def __init__(
        self, symbols: list[str], days: list[date], feed: str = data.DEFAULT_FEED
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.days = sorted(days)
        self.feed = feed
        self.series: dict[str, _SymbolSeries] = {
            sym: _SymbolSeries(sym, self.days, feed) for sym in self.symbols
        }
        self.indicators = data.load_market_indicators()

    # --- timeline ---------------------------------------------------------

    def step_times(self, day: date) -> list[datetime]:
        """Every bar-completion moment (T + 60s) across all symbols on `day`,
        deduplicated and ordered -- the engine's discrete timeline."""
        out: set[datetime] = set()
        for series in self.series.values():
            for ts in series.minute_ts:
                if ts.astimezone(MARKET_TZ).date() == day:
                    out.add(ts + timedelta(seconds=BAR_SEC))
        return sorted(out)

    def session_open(self, day: date) -> datetime:
        return datetime.combine(day, time(9, 30), tzinfo=MARKET_TZ).astimezone(timezone.utc)

    def session_close(self, day: date) -> datetime:
        return datetime.combine(day, time(16, 0), tzinfo=MARKET_TZ).astimezone(timezone.utc)

    # --- per-symbol views at time t --------------------------------------

    def completed_bars(self, symbol: str, t: datetime) -> list[dict]:
        """Minute bars completed by `t`, oldest first (the SymbolState buffer)."""
        series = self.series[symbol]
        cutoff = t - timedelta(seconds=BAR_SEC)
        idx = bisect_right(series.minute_ts, cutoff)
        return series.minute_bars[:idx]

    def price_at(self, symbol: str, t: datetime) -> Optional[float]:
        """Close of the last bar completed by `t` -- the simulated tape price."""
        bars = self.completed_bars(symbol, t)
        if not bars:
            return None
        try:
            return float(bars[-1]["c"])
        except (KeyError, TypeError, ValueError):
            return None

    def daily_bars_at(self, symbol: str, t: datetime) -> list[dict]:
        """Daily bars visible at `t`: completed days strictly before t's
        trading date, plus today's partial bar rebuilt from the minute tape --
        mirroring what the live REST daily fetch shows mid-session."""
        today = t.astimezone(MARKET_TZ).date().isoformat()
        series = self.series[symbol]
        out = [b for b in series.daily_bars if str(b.get("t", ""))[:10] < today]
        todays = [
            b
            for b, ts in zip(series.minute_bars, series.minute_ts)
            if ts.astimezone(MARKET_TZ).date().isoformat() == today
            and ts + timedelta(seconds=BAR_SEC) <= t
        ]
        if todays:
            out.append(
                {
                    "t": f"{today}T05:00:00Z",
                    "o": float(todays[0]["o"]),
                    "h": max(float(b["h"]) for b in todays),
                    "l": min(float(b["l"]) for b in todays),
                    "c": float(todays[-1]["c"]),
                    "v": sum(float(b.get("v") or 0.0) for b in todays),
                }
            )
        return out

    def prev_close(self, symbol: str, t: datetime) -> Optional[float]:
        today = t.astimezone(MARKET_TZ).date().isoformat()
        prior = [b for b in self.series[symbol].daily_bars if str(b.get("t", ""))[:10] < today]
        if not prior:
            return None
        try:
            return float(prior[-1]["c"])
        except (KeyError, TypeError, ValueError):
            return None

    def news_at(self, symbol: str, t: datetime) -> list[dict]:
        """Articles published by `t`, newest first (the SymbolState news list)."""
        series = self.series[symbol]
        out = [
            article
            for article, ts in zip(series.news, series.news_ts)
            if ts is not None and ts <= t
        ]
        return list(reversed(out))

    def fresh_news(self, symbol: str, after: datetime, until: datetime) -> list[dict]:
        """Articles published in (after, until] -- the wake-the-agent interrupt."""
        series = self.series[symbol]
        return [
            article
            for article, ts in zip(series.news, series.news_ts)
            if ts is not None and after < ts <= until
        ]

    # --- market indicators ------------------------------------------------

    def indicator_closes(self, name: str, t: datetime) -> list[tuple[str, float]]:
        """(date, close) pairs for SPY/VIX/VIX3M up to and including t's date."""
        today = t.astimezone(timezone.utc).date().isoformat()
        return [
            (row["date"], float(row["close"]))
            for row in self.indicators.get(name, [])
            if row["date"] <= today
        ]

    def bars_window(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        """Bars with start <= T < end (the patched fetch_bars_window)."""
        series = self.series[symbol]
        return [
            bar
            for bar, ts in zip(series.minute_bars, series.minute_ts)
            if start <= ts < end
        ]
