"""Unified validation and option parsing for struckdown.

This module consolidates:
- Option parsing (from options.py)
- Numeric constraint validation (from number_validation.py)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


@dataclass
class ParsedOptions:
    """Parsed type options with common fields extracted.

    Attributes:
        required: Whether the field is required (no empty response allowed)
        ge: Greater than or equal constraint (also accepts 'min')
        le: Less than or equal constraint (also accepts 'max')
        gt: Greater than constraint
        lt: Less than constraint
        min_length: Minimum string/list length
        max_length: Maximum string/list length
        positional: Non-keyword options (e.g., enum values)
        kwargs: All key=value pairs as a dict
    """

    required: bool = False
    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    positional: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a kwarg value by key."""
        return self.kwargs.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if a kwarg key exists."""
        return key in self.kwargs

    def is_required_flag_set(self) -> bool:
        """Check if required flag is set (via required=true OR bare 'required').

        For types where bare 'required' should be a keyword (like date/datetime),
        use this method. For types where 'required' is a valid option value
        (like pick), use the `required` attribute directly.
        """
        if self.required:
            return True
        # check for bare "required" in positional
        return any(p.lower() == "required" for p in self.positional)

    def get_selection_options(self) -> list[str]:
        """Get positional options, excluding 'required' keyword.

        For 'pick' type, filters out bare 'required' from options.
        """
        return [p for p in self.positional if p.lower() != "required"]


def parse_options(options: list[str] | None) -> ParsedOptions:
    """Parse type options into a structured object.

    Handles both key=value pairs and positional arguments.
    Common options (required, min, max, etc.) are extracted to named fields.

    Args:
        options: List of option strings, e.g., ['min=0', 'max=100', 'required']

    Returns:
        ParsedOptions with extracted fields and raw kwargs

    Examples:
        >>> opts = parse_options(['min=0', 'max=100', 'required'])
        >>> opts.ge
        0.0
        >>> opts.le
        100.0
        >>> opts.required
        True

        >>> opts = parse_options(['apple', 'banana', 'cherry'])
        >>> opts.positional
        ['apple', 'banana', 'cherry']
    """
    result = ParsedOptions()
    if not options:
        return result

    for opt in options:
        # handle both OptionValue namedtuples and legacy string options
        if hasattr(opt, 'key') and hasattr(opt, 'value'):
            # OptionValue namedtuple
            key = opt.key.lower() if opt.key else None
            value = opt.value
            value_str = str(value)
        else:
            # legacy string option
            opt_str = str(opt).strip()
            if not opt_str:
                continue
            if "=" in opt_str:
                key, value_str = opt_str.split("=", 1)
                key = key.strip().lower()
                value_str = value_str.strip()
                value = _parse_value(value_str)
            else:
                key = None
                value = opt_str
                value_str = opt_str

        if key:
            parsed = _parse_value(value_str) if isinstance(value, str) else value
            result.kwargs[key] = parsed

            # extract common fields with aliases
            match key:
                case "required":
                    result.required = _parse_bool(value_str) if isinstance(value_str, str) else bool(value)
                case "min" | "ge":
                    result.ge = float(parsed) if parsed is not None else None
                case "max" | "le":
                    result.le = float(parsed) if parsed is not None else None
                case "gt":
                    result.gt = float(parsed) if parsed is not None else None
                case "lt":
                    result.lt = float(parsed) if parsed is not None else None
                case "min_length" | "minlength":
                    result.min_length = int(parsed) if parsed is not None else None
                case "max_length" | "maxlength":
                    result.max_length = int(parsed) if parsed is not None else None
        else:
            # positional value (bare keywords like "required" are just values for pick type)
            result.positional.append(value)

    return result


def _parse_bool(value: str) -> bool:
    """Parse a string as a boolean."""
    return value.lower() in ("true", "t", "1", "yes")


def _parse_value(value: str) -> Any:
    """Convert string to appropriate Python type.

    Tries boolean, int, float, falls back to string.
    """
    # boolean
    if value.lower() in ("true", "t", "1", "yes"):
        return True
    if value.lower() in ("false", "f", "0", "no"):
        return False

    # integer
    try:
        return int(value)
    except ValueError:
        pass

    # float
    try:
        return float(value)
    except ValueError:
        pass

    # string
    return value


def validate_number_constraints(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    is_required: bool = False,
) -> Any:
    """Validate numeric value(s) against constraints.

    Behaviour:
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
    # check if value is required
    if is_required and value is None:
        raise ValueError(
            f"Required numeric field '{field_name}' could not be extracted from the input text."
        )

    # if no value, nothing to validate
    if value is None:
        return None

    # if no constraints, return value as-is
    if min_val is None and max_val is None:
        return value

    # prepare values to check
    values_to_check = []
    if isinstance(value, list):
        values_to_check = value
    else:
        values_to_check = [value]

    # check if any value violates constraints
    has_violation = False
    for val in values_to_check:
        if val is not None:  # skip None values in lists
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

    # if not required and there's a violation, return None instead
    if has_violation and not is_required:
        return None

    return value
