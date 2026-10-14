import datetime

import pytz
from smolagents import tool


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Fetch the current local time in a specified timezone.

    Args:
        timezone: A valid timezone, such as 'Asia/Taipei'.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return f"The current local time in {timezone} is: {local_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}"
