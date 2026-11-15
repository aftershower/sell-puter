"""NYSE trading-day check, so scheduled runs skip market holidays."""

from datetime import date

import exchange_calendars as xcals

_NYSE = xcals.get_calendar("XNYS")


def is_trading_day(d: date) -> bool:
    return bool(_NYSE.is_session(d.isoformat()))
