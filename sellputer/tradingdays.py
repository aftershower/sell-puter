"""NYSE trading-day check, so scheduled runs skip market holidays."""

from datetime import date, datetime, timedelta

import exchange_calendars as xcals

_NYSE = xcals.get_calendar("XNYS")


def is_trading_day(d: date) -> bool:
    return bool(_NYSE.is_session(d.isoformat()))


def last_closed_session(now: datetime) -> date:
    """The latest NYSE session whose closing prices are in (half an hour after the 16:00 close)."""
    d = now.date()
    if is_trading_day(d) and (now.hour, now.minute) >= (16, 30):
        return d
    return _NYSE.date_to_session((d - timedelta(days=1)).isoformat(), direction="previous").date()
