"""
Number validation logic for struckdown.

Handles parsing and validation of numeric constraints (min/max) and the required flag.
"""

from typing import Any, List, Optional, Tuple, Union


def parse_number_options(options: Optional[List[str]]) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Parse numeric options to extract min, max, and required flag.

    Args:
        options: List of option strings like ["min=0", "max=100", "required"]

    Returns:
        Tuple of (min_val, max_val, is_required)
    """
    min_val = None
    max_val = None
    is_required = False

    if not options:
        return min_val, max_val, is_required

    for opt in options:
        if opt.strip() == "required":
            is_required = True
        elif "min=" in opt or "max=" in opt:
            parts = opt.split(",")
            for part in parts:
                part = part.strip()
                if part.startswith("min="):
                    try:
                        min_val = float(part.split("=", 1)[1])
                    except (ValueError, IndexError):
                        pass
                elif part.startswith("max="):
                    try:
                        max_val = float(part.split("=", 1)[1])
                    except (ValueError, IndexError):
                        pass

    return min_val, max_val, is_required


def validate_number_constraints(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    is_required: bool = False,
) -> Any:
    """
    Validate numeric value(s) against constraints.

    Behavior:
    - If required=False (default) and value is None: returns None
    - If required=False and value violates constraints: returns None (lenient)
    - If required=True and value is None: raises ValueError
    - If required=True and value violates constraints: raises ValueError (strict)
    - If value passes constraints: returns value unchanged

    Args:
        value: The extracted value (can be None, int, float, or list)
        field_name: Name of the field for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        is_required: Whether the field is required

    Returns:
        The validated value (possibly modified to None if not required and violated)

    Raises:
        ValueError: If required=True and validation fails
    """
    # Check if value is required
    if is_required and value is None:
        raise ValueError(
            f"Required numeric field '{field_name}' could not be extracted from the input text."
        )

    # If no value, nothing to validate
    if value is None:
        return None

    # If no constraints, return value as-is
    if min_val is None and max_val is None:
        return value

    # Prepare values to check
    values_to_check = []
    if isinstance(value, list):
        values_to_check = value
    else:
        values_to_check = [value]

    # Check if any value violates constraints
    has_violation = False
    for val in values_to_check:
        if val is not None:  # Skip None values in lists
            if min_val is not None and val < min_val:
                has_violation = True
                if is_required:
                    raise ValueError(
                        f"Numeric value {val} for field '{field_name}' is below minimum {min_val}"
                    )
            if max_val is not None and val > max_val:
                has_violation = True
                if is_required:
                    raise ValueError(
                        f"Numeric value {val} for field '{field_name}' exceeds maximum {max_val}"
                    )

    # If not required and there's a violation, return None instead
    if has_violation and not is_required:
        return None

    return value
