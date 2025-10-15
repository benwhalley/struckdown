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
Ratings: 4.5, 3.8, 4.9 [[number*:ratings|min=0,max=5]]
```

Returns: `ratings: [4.5, 3.8, 4.9]` (list)

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
- `required` - Makes the field required and enforces strict validation

**Validation Behavior:**

By default (`required` not specified):
- If the LLM returns `None` or cannot extract a number, returns `None` (no error)
- If the LLM returns a number within constraints, returns that number
- If the LLM returns a number outside constraints, returns `None` (lenient mode)

With `required` flag:
- If the LLM returns `None`, raises `ValueError`
- If the LLM returns a number within constraints, returns that number
- If the LLM returns a number outside constraints, raises `ValueError` (strict mode)

### Quantifiers

- `*` - Zero or more numbers (e.g., `[[number*:values]]`)
- `+` - One or more numbers (e.g., `[[number+:values]]`)
- `?` - Zero or one number (e.g., `[[number?:value]]`)
- `{n}` - Exactly n numbers (e.g., `[[number{3}:rgb]]`)
- `{min,max}` - Between min and max numbers (e.g., `[[number{2,5}:scores]]`)
- `{min,}` - At least min numbers (e.g., `[[number{2,}:values]]`)

## Examples

See `/Users/benwhalley/dev/struckdown/examples/13_number_extraction.sd` for 15 practical examples.

## How It Works

1. **LLM Extraction**: The LLM extracts numeric values (int or float) from the text based on the prompt and hints provided in the constraints.

2. **Type Selection**: The response model accepts `Union[int, float]` for single values or `List[Union[int, float]]` for lists, allowing the LLM to choose the most appropriate type.

3. **Post-Extraction Validation**: After extraction, Python code validates that all values satisfy the min/max constraints. If any value is out of range, a `ValueError` is raised with a helpful error message.

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
- **Validation errors with `required` flag** (strict validation)
- **Lenient behavior without `required` flag** (returns None for violations)

**Current Test Results: 43/43 passed (100% success rate)**

## Technical Details

### Implementation Files

- `struckdown/return_type_models.py` - Contains `number_response_model()` factory function
- `struckdown/__init__.py` - Post-extraction validation logic (lines 374-413)

### Key Design Decisions

1. **Union Type**: Using `Union[int, float]` allows the LLM to choose the most natural type for each value.

2. **Hints vs Validation**: Constraints are provided as suggestions in the prompt but strictly enforced after extraction. This approach:
   - Guides the LLM towards correct values
   - Catches edge cases where the LLM might misinterpret the text
   - Provides clear error messages when validation fails

3. **Flexible Quantifiers**: Supports the same quantifier syntax as other struckdown types (pick, date, etc.) for consistency.

## Comparison with `int` Type

The `number` type is more flexible than the existing `int` type:

| Feature | `int` | `number` |
|---------|-------|----------|
| Integers | ✓ | ✓ |
| Floats | ✗ | ✓ |
| Min/max constraints | ✗ | ✓ |
| Lists | ✗ | ✓ |
| Quantifiers | ✗ | ✓ |

The `int` type remains available for backward compatibility and cases where you specifically need an integer.

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
Exam scores: 85, 92, 78, 95 [[number*:scores|min=0,max=100]]
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

If a value violates constraints, a `ValueError` is raised with a clear error message:

```python
ValueError: Numeric value -10 for field 'score' is below minimum 0.0
```

```python
ValueError: Numeric value 150 for field 'score' exceeds maximum 100.0
```

### Important: Constraint Validation and `required` Flag

The constraints are provided as **hints** to the LLM, and validation behavior depends on the `required` flag:

#### Default Behavior (Lenient - `required` not set)

```
Give me a number greater than 10 [[number:mynum|max=10]]
```

The LLM will likely return a number > 10 (following the prompt), but since it violates `max=10`:
- **Result:** `mynum = None` (no error raised)
- This is lenient mode: constraint violations return `None` instead of raising errors

#### Strict Behavior (with `required` flag)

```
Give me a number greater than 10 [[number:mynum|max=10,required]]
```

With the `required` flag, constraint violations raise errors:
- **Result:** `ValueError: Numeric value 11 for field 'mynum' exceeds maximum 10.0`
- This is strict mode: constraint violations and missing values both raise errors

#### When to Use `required`

- **Omit `required` (default):** For optional extractions where you want graceful handling
  - Example: Parsing user input that might not contain numbers
  - Constraint violations return `None` instead of crashing

- **Use `required`:** When you need guaranteed valid data
  - Example: Processing structured data where numbers are mandatory
  - Ensures no invalid data passes through

This two-stage approach (hints + validation) ensures:
- The LLM is guided toward correct values
- You control whether violations are errors or handled gracefully
- Flexibility for both optional and required extractions

## Future Enhancements

Potential improvements:
- Support for currency symbols (auto-extract from "$19.99")
- Support for percentage symbols (auto-extract from "75%")
- More constraint types (e.g., `step=5` for multiples of 5)
- Statistical validation (e.g., `mean`, `std`, `outliers`)
