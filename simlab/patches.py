"""The simulation context: reroute every live-fetch call site to the dataset.

`simulation_context(market)` is the single place that knows which parts of
``agent_stonks`` would otherwise touch the network (or the wall clock) during
an agent cycle, and swaps each one for a dataset-backed equivalent for the
duration of a simulation:

- ``clock``                       -> pinned by the engine per step (cleared here on exit)
- ``agent.fetch_bars_window``     -> stored minute bars (opening-range recovery)
- ``agent.fetch_corporate_actions`` -> none scheduled (not part of the dataset)
- ``historical.fetch_market_indicators`` -> stored SPY/VIX/VIX3M closes,
  clipped to the simulated date (also covers ``tactics.fetch_vix_level`` and
  the ``vix`` tactic condition, which read through the same function)
- ``historical.fetch_analyst_targets`` / ``fetch_smart_money_flow`` ->
  honest "not available in simulation" notes (point-in-time histories of
  these aren't stored; a note keeps the agent reasoning on real data instead
  of on today's targets leaking into the past)
- ``historical.fetch_intraday_volume_bars`` / ``fetch_intraday_bars_for_date``
  / ``fetch_daily_volume_bars`` (the volume tools' consolidated-tape reads)
  -> stored minute/daily bars clipped to the simulated clock. Live these hit
  yfinance for *wall-clock today*, which inside a simulation is a different
  day entirely; and a dated fetch of the simulated day would return bars from
  hours in the simulated future. Both leaks corrupted every stored
  volume_detective run before this patch existed.
Keeping every patch point in this one module means a new live fetch added to
the app fails loudly here (the setattr asserts the attribute exists) instead
of silently leaking real-time data into simulated sessions.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd

from agent_stonks import agent as agent_mod
from agent_stonks import clock, historical
from agent_stonks.market_hours import MARKET_TZ

from .market import BAR_SEC, SimMarket


def _series(pairs: list[tuple[str, float]]) -> pd.Series:
    if not pairs:
        return pd.Series(dtype=float)
    dates, closes = zip(*pairs)
    return pd.Series(closes, index=pd.to_datetime(dates))


@contextmanager
def simulation_context(market: SimMarket) -> Iterator[None]:
    def fake_bars_window(symbol, timeframe, start, end, key, secret, feed="iex", limit=200):
        return market.bars_window(str(symbol).upper(), start, end)

    def fake_corporate_actions(symbol, key, secret, days_ahead=14):
        return []

    def fake_market_indicators(days: int = 365, ttl_sec: int = 300) -> dict:
        now = clock.now()
        return {
            name: _series(market.indicator_closes(name, now))
            for name in ("spy", "vix", "vix3m")
        }

    def fake_analyst_targets(symbol, current_price=None):
        return {
            "note": (
                "analyst price targets are not available in simulation "
                "(no point-in-time history stored) -- weigh the technical read instead"
            )
        }

    def fake_smart_money_flow(symbol):
        return {
            "note": (
                "insider/institutional ownership data is not available in simulation "
                "(no point-in-time history stored) -- weigh the technical read instead"
            )
        }

    def _bars_for_day(symbol: str, day, now: datetime) -> list[dict]:
        """Stored minute bars for one ET `day`, completed by `now` -- a full
        day for a past session, clipped mid-day for the simulated one."""
        series = market.series.get(str(symbol).upper())
        if series is None:
            return []
        return [
            bar
            for bar, ts in zip(series.minute_bars, series.minute_ts)
            if ts.astimezone(MARKET_TZ).date() == day
            and ts + timedelta(seconds=BAR_SEC) <= now
        ]

    def fake_intraday_volume_bars(symbol, interval="1m", ttl_sec=0):
        now = clock.now()
        return _bars_for_day(symbol, now.astimezone(MARKET_TZ).date(), now)

    def fake_intraday_bars_for_date(symbol, date, interval="1m"):
        # Same contract as the live fetch: bad format raises ValueError, a day
        # with no stored bars returns [] (the tool turns that into its honest
        # "no intraday bars for <day>" note).
        try:
            day = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"date must be 'YYYY-MM-DD', got {date!r}") from exc
        return _bars_for_day(symbol, day, clock.now())

    def fake_daily_volume_bars(symbol, days=90, ttl_sec=0):
        # Live shape: {"t": "YYYY-MM-DD", "v": volume}, today's partial row
        # included (downstream ADV helpers exclude today themselves).
        return [
            {"t": str(bar.get("t", ""))[:10], "v": float(bar.get("v") or 0.0)}
            for bar in market.daily_bars_at(str(symbol).upper(), clock.now())
        ]

    patches = [
        (agent_mod, "fetch_bars_window", fake_bars_window),
        (agent_mod, "fetch_corporate_actions", fake_corporate_actions),
        (historical, "fetch_market_indicators", fake_market_indicators),
        (historical, "fetch_analyst_targets", fake_analyst_targets),
        (historical, "fetch_smart_money_flow", fake_smart_money_flow),
        (historical, "fetch_intraday_volume_bars", fake_intraday_volume_bars),
        (historical, "fetch_intraday_bars_for_date", fake_intraday_bars_for_date),
        (historical, "fetch_daily_volume_bars", fake_daily_volume_bars),
    ]
    saved = []
    for module, name, replacement in patches:
        saved.append((module, name, getattr(module, name)))  # raises if renamed upstream
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)
        clock.clear()
