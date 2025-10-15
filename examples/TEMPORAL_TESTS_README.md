# Temporal Extraction Test Suite

Comprehensive test suite for validating temporal type extractions (dates, datetimes, times, durations) in Struckdown.

## Running the Tests

### Basic Usage

```bash
uv run python examples/temporal_test_cases.py
```

### With Verbose Output

See detailed results for each test case:

```bash
uv run python examples/temporal_test_cases.py --verbose
# or
uv run python examples/temporal_test_cases.py -v
```

### Stop on First Error

Useful for debugging:

```bash
uv run python examples/temporal_test_cases.py --stop-on-error
# or
uv run python examples/temporal_test_cases.py -x
```

### Combined Options

```bash
uv run python examples/temporal_test_cases.py --verbose --stop-on-error
```

## Test Coverage

The test suite includes **38 comprehensive test cases** covering:

### 1. Single Date Extractions (4 tests)
- Explicit dates (e.g., "January 15, 2024")
- ISO format dates
- Relative dates (e.g., "tomorrow", "next Monday")

### 2. Recurring Patterns - Single Value (2 tests)
- Patterns like "first Tuesday in November"
- Should warn about truncation when pattern generates multiple dates

### 3. Recurring Patterns - Lists (4 tests)
- "first 2 Tuesdays in September" `[[date*:dates]]`
- "first 3 Mondays in October"
- "every Wednesday in September 2025"
- All using RRULE expansion

### 4. Multiple Explicit Dates (2 tests)
- Lists of explicit dates
- Mixed date formats

### 5. Quantifiers (4 tests)
- `[[date{2}:dates]]` - exactly 2 dates
- `[[date{1,3}:dates]]` - 1 to 3 dates
- `[[date+:dates]]` - at least 1 date
- `[[date*:dates]]` - zero or more dates

### 6. DateTime Extractions (4 tests)
- Explicit datetimes with time components
- ISO datetime format
- Relative datetimes (e.g., "tomorrow at 2pm")
- Lists of datetimes

### 7. Time Extractions (4 tests)
- Explicit times (12-hour and 24-hour format)
- Lists of times
- Ambiguous times (without AM/PM)

### 8. Duration Extractions (5 tests)
- Simple durations ("2 hours", "30 minutes")
- Complex durations ("1 week and 3 days", "2 hours 45 minutes")
- Lists of durations

### 9. Mixed Temporal Types (2 tests)
- Extracting different temporal types from same text
- Date + duration, datetime + duration

### 10. Edge Cases (3 tests)
- No date in text (returns None)
- Empty lists
- Ambiguous month references

### 11. Year Inference (2 tests)
- Month without year uses current context
- Patterns with explicit years

### 12. Complex Patterns (2 tests)
- "Every other week" patterns
- "All Mondays in September"

## Test Results Format

```
================================================================================
TEMPORAL EXTRACTION TEST SUITE
================================================================================

Total test cases: 38
Categories: 12

================================================================================
CATEGORY: Single Dates - Explicit
================================================================================

[1/2] Extract explicit date (Jan 15, 2024)... ✓ PASSED
[2/2] Extract ISO format date... ✓ PASSED

...

================================================================================
TEST SUMMARY
================================================================================

✓ Passed: 38/38
✗ Failed: 0/38
⚠ Errors: 0/38

================================================================================
SUCCESS RATE: 100.0%
================================================================================
```

## Key Features Tested

### 1. Pattern Detection & RRULE Expansion
Tests verify that natural language patterns like "first 2 Tuesdays in September" are:
- Detected as patterns (not explicit dates)
- Converted to RRULE specifications
- Expanded to concrete dates using `dateutil.rrule`
- Properly handle year inference (defaults to current/next year)

### 2. Year Context
Tests verify that:
- Patterns without explicit years use the current date context
- "September" in October 2025 refers to September 2025, not 2024
- Explicit years in patterns are respected

### 3. Single vs List Extraction
Tests verify:
- `[[date:var]]` extracts single date, warns if pattern generates multiple
- `[[date*:vars]]` extracts list of dates
- Quantifiers properly constrain list sizes

### 4. Temporal Context Injection
Tests verify that `date_rule` extractions receive temporal context hints, ensuring proper year inference.

## Adding New Tests

To add a new test case, append to the `TEST_CASES` list in `temporal_test_cases.py`:

```python
{
    "category": "Your Category",
    "description": "Brief description of what's being tested",
    "prompt": "Your prompt with [[date:var]] or other temporal slots",
    "expected_type": date,  # or datetime, time, timedelta, list, etc.
    "validate": lambda r: (
        # Your validation logic
        isinstance(r["var"], date)
        and r["var"].month == 9
    ),
},
```

### Validation Tips

- Use `isinstance(r["varname"], expected_type)` for type checking
- For dates: check `.year`, `.month`, `.day`, `.weekday()`
- For times: check `.hour`, `.minute`
- For durations: accept both `timedelta` and `str` (LLM may return strings)
- For lists: check `len()` and use `all()` for element validation

## Cache Management

The test suite clears the Struckdown cache before each run. If you encounter issues:

```bash
rm -rf ~/.struckdown/cache
```

## Exit Codes

- `0`: All tests passed
- `1`: At least one test failed or error occurred

Perfect for CI/CD integration!

## Performance

Running all 38 tests typically takes 2-5 minutes depending on:
- LLM API response times
- Cache status
- Network conditions

Use `--stop-on-error` for faster debugging iterations.
