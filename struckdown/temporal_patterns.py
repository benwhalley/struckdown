"""
Temporal pattern expansion using RRULE.

This module handles converting natural language date/datetime patterns
into concrete dates using the dateutil.rrule library.
"""

import logging
from typing import Any, List, Optional, Union

import anyio

logger = logging.getLogger(__name__)


async def expand_temporal_pattern(
    pattern_string: str,
    action_type: str,
    is_single_value: bool,
    quantifier: Optional[tuple],
    llm: Any,
    credentials: Any,
    accumulated_context: dict,
) -> tuple[Union[Any, List[Any]], List[Any]]:
    """
    Expand a temporal pattern string into concrete date(s) using RRULE.

    Args:
        pattern_string: Natural language pattern (e.g., "first 2 Tuesdays in October")
        action_type: Either "date" or "datetime"
        is_single_value: True if expecting single value, False for list
        quantifier: Optional (min_items, max_items) tuple for validation
        llm: LLM instance for chatter call
        credentials: LLM credentials
        accumulated_context: Context dict with temporal information

    Returns:
        Tuple of (final_result, interim_steps) where:
        - final_result: Single date/datetime if is_single_value=True, otherwise list of dates/datetimes
        - interim_steps: List of SegmentResults from intermediate LLM calls

    Raises:
        ValueError: If pattern cannot be expanded or doesn't meet constraints
    """
    logger.debug(f"Detected {action_type} pattern: {pattern_string}")

    # Import at function level to avoid circular dependency
    from struckdown import SegmentResult
    from struckdown import chatter as chatter_func

    # Track interim steps
    interim_steps = []

    # Call chatter recursively to convert pattern to RRULE dict
    rule_prompt = f"{pattern_string} [[date_rule:rule]]"
    rule_result = await anyio.to_thread.run_sync(
        lambda: chatter_func(
            rule_prompt,
            model=llm,
            credentials=credentials,
            context=accumulated_context,  # Pass temporal context
        ),
        abandon_on_cancel=True,
    )

    # Extract RRULE parameters from the result
    # DateRuleResponse has flat fields, convert to dict
    rrule_model = rule_result.results["rule"].output
    rrule_dict = rrule_model.model_dump(exclude_none=True)
    logger.debug(f"RRULE spec: {rrule_dict}")

    # Store the date_rule extraction as an interim step
    interim_steps.append(rule_result.results["rule"])

    # Expand RRULE to list of dates
    try:
        from datetime import date as date_type
        from datetime import datetime as datetime_type

        from dateutil.rrule import DAILY, MONTHLY, WEEKLY, YEARLY, rrule

        # Map string freq to constants
        freq_map = {
            "DAILY": DAILY,
            "WEEKLY": WEEKLY,
            "MONTHLY": MONTHLY,
            "YEARLY": YEARLY,
        }

        # Prepare kwargs for rrule
        rrule_kwargs = {}
        rrule_kwargs["freq"] = freq_map[rrule_dict["freq"]]

        # Parse dtstart from string to date/datetime
        if "dtstart" in rrule_dict:
            dtstart_str = rrule_dict["dtstart"]
            if action_type == "date":
                # For date type, parse as date but rrule needs datetime
                if "T" in dtstart_str:
                    # It's already a datetime string
                    rrule_kwargs["dtstart"] = datetime_type.fromisoformat(dtstart_str)
                else:
                    # It's a date string, convert to datetime
                    d = date_type.fromisoformat(dtstart_str)
                    rrule_kwargs["dtstart"] = datetime_type(d.year, d.month, d.day)
            else:  # datetime type
                rrule_kwargs["dtstart"] = datetime_type.fromisoformat(dtstart_str)

        # Parse until if present
        if "until" in rrule_dict:
            until_str = rrule_dict["until"]
            if "T" in until_str:
                rrule_kwargs["until"] = datetime_type.fromisoformat(until_str)
            else:
                d = date_type.fromisoformat(until_str)
                rrule_kwargs["until"] = datetime_type(d.year, d.month, d.day)

        # Copy other parameters
        for key in [
            "count",
            "interval",
            "byweekday",
            "bymonth",
            "bymonthday",
            "bysetpos",
            "byhour",
            "byminute",
        ]:
            if key in rrule_dict:
                rrule_kwargs[key] = rrule_dict[key]

        # Generate dates using rrule
        rule = rrule(**rrule_kwargs)
        expanded_dates = list(rule)

        # Convert to date if needed (rrule returns datetimes)
        if action_type == "date":
            expanded_dates = [
                d.date() if hasattr(d, "date") else d for d in expanded_dates
            ]

        # Handle single value vs list
        if is_single_value:
            # For single values, take first date and warn if multiple
            if len(expanded_dates) > 1:
                logger.warning(
                    f"Pattern '{pattern_string}' generated {len(expanded_dates)} dates "
                    f"but only one was requested. Using first date: {expanded_dates[0]}. "
                    f"Consider using [[{action_type}*:var]] for multiple dates."
                )
            result = expanded_dates[0] if expanded_dates else None
        else:
            # For lists, validate against quantifier constraints
            if quantifier:
                min_items, max_items = quantifier
                if max_items and len(expanded_dates) > max_items:
                    logger.warning(
                        f"RRULE generated {len(expanded_dates)} dates but max is {max_items}. Truncating."
                    )
                    expanded_dates = expanded_dates[:max_items]
                if min_items and len(expanded_dates) < min_items:
                    raise ValueError(
                        f"RRULE generated only {len(expanded_dates)} dates but minimum is {min_items}"
                    )
            result = expanded_dates

        logger.debug(
            f"Expanded pattern to {len(expanded_dates) if isinstance(result, list) else 1} date(s): {result}"
        )
        return result, interim_steps

    except Exception as e:
        logger.error(f"Failed to expand {action_type} pattern: {e}")
        raise ValueError(
            f"Could not expand {action_type} pattern '{pattern_string}': {e}"
        )
