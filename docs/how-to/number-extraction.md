---
layout: default
title: Number Extraction
parent: How-To Guides
nav_order: 3
---

# Number Extraction in Struckdown

Flexible numeric extraction supporting both integers and floats with optional min/max constraints.

## Quick Start

### Basic Usage

```
The answer is 42 [[number:value]]
```

Returns: `value: 42` (int)

```
The price is $19.99 [[number:price]]
```

Returns: `price: 19.99` (float)

### With Constraints

```
Your score: 85 [[number:score|min=0,max=100]]
```

Returns: `score: 85` (int)

Validation: Ensures the extracted value is between 0 and 100 (inclusive).

### List Extraction

```
Test scores: 85, 92, 78, 95 [[number*:scores]]
```

Returns: `scores: [85, 92, 78, 95]` (list)

### List with Constraints

```
Ratings: 4, 5, 3 [[number*:ratings|min=0,max=5]]
```

`min` and `max` on a list slot bound each value. A value outside them fails validation.

## Syntax

The general syntax for number extraction is:

```
[[number:varname|options]]
```

Or with quantifiers for lists:

```
[[number*:varname|options]]
[[number{n}:varname|options]]
[[number{min,max}:varname|options]]
```

### Options

- `min=X` - Minimum value (e.g., `min=0`)
- `max=Y` - Maximum value (e.g., `max=100`)
- `min=X,max=Y` - Both constraints (e.g., `min=0,max=100`)
- `required` (or a `!` prefix, `[[!number:value]]`) - the answer may not be null

**Validation Behavior:**

`min` and `max` become constraints on the response schema (`ge` / `le`), whether or not the slot is required. An answer outside the range fails validation; pydantic-ai asks the model again, and if the retries run out the call raises. An out-of-range answer never turns into `None`.

`required` only controls whether null is allowed. Without it, the model may answer null when there is no number to find, and the slot's value is `None`.

### Quantifiers

- `*` - Zero or more numbers (e.g., `[[number*:values]]`)
- `+` - One or more numbers (e.g., `[[number+:values]]`)
- `?` - Zero or one number (e.g., `[[number?:value]]`)
- `{n}` - Exactly n numbers (e.g., `[[number{3}:rgb]]`)
- `{min,max}` - Between min and max numbers (e.g., `[[number{2,5}:scores]]`)
- `{min,}` - At least min numbers (e.g., `[[number{2,}:values]]`)

## Examples

See `examples/10_number_extraction.sd` for worked examples.

## How It Works

1. **LLM Extraction**: The LLM extracts numeric values (int or float) from the text based on the prompt and hints provided in the constraints.

2. **Type Selection**: The response model accepts `Union[int, float]` for single values or `List[Union[int, float]]` for lists, allowing the LLM to choose the most appropriate type.

3. **Validation**: `min` and `max` are part of the response schema, so an out-of-range answer fails validation and is retried by pydantic-ai; the range is also written into the field's description, which the model sees.

## Test Suite

Run the comprehensive test suite with:

```bash
uv run python examples/number_test_cases.py
```

Options:
- `--verbose` or `-v`: Show detailed output for each test
- `--stop-on-error` or `-x`: Stop on first failure

The test suite includes 43 test cases covering:
- Basic extraction (integers, floats, negative numbers, scientific notation)
- Min/max constraints (single and combined)
- List extraction (basic and with constraints)
- Quantifiers (all variants)
- Edge cases (None values, zero, very large/small numbers)
- Practical use cases (prices, measurements, percentages, ratings, temperatures)
- Null answers with and without `required`

## Technical Details

### Implementation Files

- `struckdown/return_type_models.py` - `number_response_model()` and `integer_response_model()`, both built by `_build_numeric_response_model()`

### Key Design Decisions

1. **Union Type**: Using `Union[int, float]` allows the LLM to choose the most natural type for each value.

2. **Hints and Validation**: Constraints appear in the field description the model reads, and are enforced by the schema, so an answer outside them is sent back for another attempt.

3. **Flexible Quantifiers**: Supports the same quantifier syntax as other struckdown types (pick, date, etc.) for consistency.

## Comparison with `int` Type

`int` and `number` are built the same way and take the same options and quantifiers. The difference is the value type:

| Feature | `int` | `number` |
|---------|-------|----------|
| Integers | yes | yes |
| Floats | no | yes |
| Min/max constraints | yes | yes |
| Lists and quantifiers | yes | yes |

Use `int` when you need an integer.

## Common Patterns

### Financial Data

```
Q1 Revenue: $1,234,567.89
Q2 Revenue: $1,456,789.01

Extract quarterly revenues [[number*:revenues]]
```

### Ratings

```
Product: 4.7 out of 5 stars [[number:rating|min=0,max=5]]
```

### Test Scores

```
Exam scores: 85, 92, 78, 95 [[number*:scores]]
```

### Temperature

```
Current: -12.5°C [[number:temp]]
```

### Percentage

```
Progress: 67.5% complete [[number:progress|min=0,max=100]]
```

## Error Handling

An answer outside `min` / `max` is a validation failure. pydantic-ai returns the error to the model and asks again; if the answer is still out of range after its retries, the call raises. A slot without `required` can still come back as `None`, but only when the model answers null, not because a value was out of range.

```
Give me a number greater than 10 [[number:mynum|max=10]]
```

Here the prompt and the constraint conflict. The model is asked for a value of at most 10; if it keeps answering above 10, the call fails rather than returning `None`.

- **Omit `required`** when the text may not contain a number: the model can answer null.
- **Use `required`** (or `!`) when a number must be present.

## Future Enhancements

Potential improvements:
- Support for currency symbols (auto-extract from "$19.99")
- Support for percentage symbols (auto-extract from "75%")
- More constraint types (e.g., `step=5` for multiples of 5)
- Statistical validation (e.g., `mean`, `std`, `outliers`)
