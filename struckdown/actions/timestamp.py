"""Built-in @timestamp action for getting current time."""

from datetime import datetime

from . import Actions


@Actions.register("timestamp")
def timestamp_action(context: dict, format: str | None = None) -> str:
    """Get the current timestamp.

    Usage:
        [[@timestamp]]  # outputs current datetime in ISO format
        [[@timestamp|format="%Y-%m-%d"]]
        [[@timestamp:current_time|format="%H:%M"]]  # outputs time + saves in context as current_time

    Args:
        context: Struckdown context dict
        format: strftime format string (default: ISO format)

    Returns:
        Formatted timestamp
    """
    now = datetime.now()
    if format is None:
        return now.isoformat()
    return now.strftime(format)
