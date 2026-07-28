from datetime import datetime, timezone

from agent_stonks.market_hours import (
    MARKET_TZ,
    is_market_open,
    next_market_open,
    seconds_until_next_open,
    session_open,
)


def _et(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ)


# 2026-07-06 is a Monday; 2026-07-04 a Saturday; 2026-07-10 a Friday.
class TestIsMarketOpen:
    def test_open_mid_session(self):
        assert is_market_open(_et(2026, 7, 6, 10, 0))

    def test_closed_just_before_the_bell(self):
        assert not is_market_open(_et(2026, 7, 6, 9, 29))

    def test_open_exactly_at_the_bell(self):
        assert is_market_open(_et(2026, 7, 6, 9, 30))

    def test_closed_at_the_close(self):
        assert not is_market_open(_et(2026, 7, 6, 16, 0))

    def test_closed_on_the_weekend(self):
        assert not is_market_open(_et(2026, 7, 4, 12, 0))

    def test_accepts_utc_input(self):
        # 14:00 UTC on a summer Monday = 10:00 ET -> open.
        assert is_market_open(datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc))


class TestNextMarketOpen:
    def test_pre_open_same_day(self):
        nxt = next_market_open(_et(2026, 7, 6, 8, 0)).astimezone(MARKET_TZ)
        assert (nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute) == (2026, 7, 6, 9, 30)

    def test_mid_session_rolls_to_next_day(self):
        nxt = next_market_open(_et(2026, 7, 6, 10, 0)).astimezone(MARKET_TZ)
        assert (nxt.day, nxt.hour, nxt.minute) == (7, 9, 30)

    def test_friday_evening_rolls_to_monday(self):
        nxt = next_market_open(_et(2026, 7, 10, 17, 0)).astimezone(MARKET_TZ)
        assert (nxt.day, nxt.weekday()) == (13, 0)

    def test_returned_in_utc(self):
        assert next_market_open(_et(2026, 7, 6, 8, 0)).tzinfo == timezone.utc


class TestSessionOpen:
    def test_open_of_session_in_progress(self):
        opened = session_open(_et(2026, 7, 6, 11, 0))
        assert opened is not None
        et = opened.astimezone(MARKET_TZ)
        assert (et.day, et.hour, et.minute) == (6, 9, 30)

    def test_none_when_closed(self):
        assert session_open(_et(2026, 7, 6, 8, 0)) is None


class TestSecondsUntilNextOpen:
    def test_two_minutes_before_the_bell(self):
        assert seconds_until_next_open(_et(2026, 7, 6, 9, 28)) == 120.0

    def test_always_positive(self):
        assert seconds_until_next_open(_et(2026, 7, 6, 12, 0)) > 0


class TestSecondsToClose:
    def test_mid_session(self):
        from agent_stonks.market_hours import seconds_to_close

        assert seconds_to_close(_et(2026, 7, 6, 15, 0)) == 3600.0

    def test_final_minutes(self):
        from agent_stonks.market_hours import seconds_to_close

        assert seconds_to_close(_et(2026, 7, 6, 15, 50)) == 600.0

    def test_closed_returns_none(self):
        from agent_stonks.market_hours import seconds_to_close

        assert seconds_to_close(_et(2026, 7, 6, 16, 30)) is None
        assert seconds_to_close(_et(2026, 7, 4, 12, 0)) is None  # Saturday
