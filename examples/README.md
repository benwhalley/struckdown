# Struckdown Examples

This directory contains example files demonstrating Struckdown features.

## Example `.sd` Files

### Basic Features

- **`01_basic_completion.sd`** -- Simple LLM completion
- **`02_simple_chain.sd`** -- Chaining completions with `¡OBLIVIATE`
- **`03_template_variables.sd`** -- Using `{{variables}}` for dynamic content
- **`04_shared_header.sd`** -- System messages and headers with `¡SYSTEM`/`¡HEADER`
- **`05_return_types.sd`** -- Type-specific completions (bool, int, pick)

### Advanced Features

- **`06_list_completions.sd`** -- List generation (`3*[[item]]`, `*[[item]]`)
- **`07_complex_workflow.sd`** -- Multi-step data analysis pipeline
- **`08_template_tags.sd`** -- Dynamic content generation
- **`09_edge_cases.sd`** -- Error handling and special cases
- **`11_variables_in_header.sd`** -- Using variables in shared headers
- **`12_temporal_extraction.sd`** -- Date/time extraction examples
- **`13_number_extraction.sd`** -- Number extraction with constraints

## Demo Scripts

### Feature Tests & Demos

- **`temporal_test_cases.py`** -- Comprehensive date/time extraction tests
- **`number_test_cases.py`** -- Number validation and constraint tests
- **`temporal_extraction_demo.py`** -- Interactive temporal extraction demo
- **`temporal_optional_required_demo.py`** -- Optional vs required temporal fields

### Custom Actions Demos

- **`test_expertise_simple.py`** -- RAG pattern: parsing and syntax demo
- **`test_expertise_function.py`** -- RAG pattern: simple search examples
- **`test_context_flow.py`** -- RAG pattern: detailed context flow visualization
- **`test_examples.py`** -- Run all .sd example files

## Running Examples

### Test Individual Files

```bash
# Run an example
python -c "
from struckdown import chatter
template = open('examples/01_basic_completion.sd').read()
result = chatter(template)
print(result.response)
"
```

### Run Demo Scripts

```bash
# Temporal extraction tests
uv run python examples/temporal_test_cases.py

# Number extraction tests
uv run python examples/number_test_cases.py

# Custom Actions / RAG demos
uv run python examples/test_expertise_simple.py       # Parsing only
uv run python examples/test_expertise_function.py     # With mocked expertise
uv run python examples/test_context_flow.py           # Context flow visualization

# Run all .sd examples
uv run python examples/test_examples.py
```

## Key Concepts Demonstrated

### Completions

```
[[variable]]              # Basic text completion
[[type:variable]]         # Typed completion
[[3*item]]                # Exactly 3 items
[[2:4*item]]              # Between 2-4 items
[[*item]]                 # Any number of items
```

### Types

```
[[bool:answer]]           # True/False
[[int:count]]             # Integer
[[number:price]]          # Int or float
[[pick:color|red,blue]]   # Choose from options
[[date:when]]             # Date extraction
[[datetime:timestamp]]    # DateTime extraction
```

### Memory Management

```
Context...
[[extract1]]

¡OBLIVIATE              # Memory boundary

Use {{extract1}}        # Only variables carry forward
[[extract2]]
```

### Shared Headers

```
You are an expert.

¡BEGIN                  # Header ends here

Prompt 1: [[result1]]

¡OBLIVIATE

Prompt 2: [[result2]]  # Header is prepended automatically
```

## Example Workflows

### 1. Simple Extraction

```bash
sd chat -p examples/03_template_variables.sd
```

### 2. Multi-Step Analysis

```bash
sd chat -p examples/07_complex_workflow.sd
```

### 3. Batch Processing

```bash
# Create test files
echo "Price: $29.99" > test1.txt
echo "Cost: $45.50" > test2.txt

# Extract prices
sd batch test*.txt "Price: [[number:price]]"
```

## Documentation

- **[QuickStart](../docs/QUICKSTART.md)** -- Get started quickly
- **[CLI Usage](../docs/CLI_USAGE.md)** -- Command reference
- **[Custom Actions](../docs/CUSTOM_ACTIONS.md)** -- Python plugins
- **[Main README](../README.md)** -- Project overview
