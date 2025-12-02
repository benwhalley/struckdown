# Simplify validation.py

## Overview

Refactor `struckdown/validation.py` to use more functional, declarative patterns.

## Changes

### 1. `_parse_value` -- replace try/except with parser chain

Current code uses try/except for control flow. Replace with a functional parser chain:

```python
def _try_bool(value: str) -> bool | None:
    match value.lower():
        case "true" | "t" | "1" | "yes": return True
        case "false" | "f" | "0" | "no": return False
        case _: return None

def _try_int(value: str) -> int | None:
    return int(value) if value.lstrip("-").isdigit() else None

def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None

def _parse_value(value: str) -> Any:
    parsers = [_try_bool, _try_int, _try_float]
    for parser in parsers:
        if (result := parser(value)) is not None:
            return result
    return value
```

### 2. `validate_number_constraints` -- simplify with match and any()

Replace imperative violation checking:

```python
def validate_number_constraints(
    value: Any,
    field_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    is_required: bool = False,
) -> Any:
    # handle None cases first
    match (value, is_required):
        case (None, True):
            raise ValueError(
                f"Required numeric field '{field_name}' could not be extracted."
            )
        case (None, False):
            return None

    # no constraints means valid
    if min_val is None and max_val is None:
        return value

    # normalise to list
    values = value if isinstance(value, list) else [value]

    def check_violation(val: Any) -> str | None:
        if val is None:
            return None
        if min_val is not None and val < min_val:
            return f"value {val} below minimum {min_val}"
        if max_val is not None and val > max_val:
            return f"value {val} exceeds maximum {max_val}"
        return None

    violations = [v for val in values if (v := check_violation(val))]

    if violations:
        if is_required:
            raise ValueError(f"Numeric field '{field_name}': {violations[0]}")
        return None

    return value
```

### 3. Keep `_parse_bool` and `ParsedOptions` as-is

These are already clean and readable.

### 4. `parse_options` -- minor cleanup only

The current implementation with match-case is already good. Only change: extract the field-setting logic if desired, but not essential.

## Testing

Run existing tests to verify behaviour unchanged:

```bash
uv run python -m pytest struckdown/tests/ -v -k validation
```

If no validation-specific tests exist, run the full suite to catch regressions.
