# Struckdown Documentation

Complete documentation for Struckdown.

## Getting Started

- **[QuickStart](QUICKSTART.md)** -- Get started in 5 minutes
- **[Main README](../README.md)** -- Project overview with examples

## Tutorials

- **[Building a RAG System](TUTORIAL_RAG.md)** -- Learn the Extract → Search → Generate pattern
- **[Custom Actions Guide](CUSTOM_ACTIONS.md)** -- Extend Struckdown with Python plugins

## User Guides

- **[CLI Usage](CLI_USAGE.md)** -- Complete command-line reference
- **[Temperature and Model Overrides](TEMPERATURE_AND_MODEL_OVERRIDES.md)** -- Per-slot configuration

## Feature Documentation

- **[Number Extraction](NUMBER_EXTRACTION_README.md)** -- Extract and validate numbers
- **[Temporal Extraction](TEMPORAL_TESTS_README.md)** -- Work with dates, times, and durations

## Reference

- **[Examples](../examples/)** -- Example `.sd` files and test cases
- **[Security](../SECURITY.md)** -- Security best practices

## Quick Reference

### Basic Syntax

```
[[variable]]                    # Basic completion
[[type:variable]]               # Typed completion
{{variable}}                    # Reference previous extraction
<checkpoint>                    # Memory boundary
<system>...</system>            # System message block
<system local>...</system>      # System message (segment-scoped)
```

### Types

| Type | Syntax | Example |
|------|--------|---------|
| Text | `[[name]]` | Default text response |
| Boolean | `[[bool:answer]]` | True/False |
| Integer | `[[int:count]]` | Whole numbers |
| Number | `[[number:price]]` | Int or float |
| Pick | `[[pick:color\|red,blue]]` | Choose from options |
| Date | `[[date:when]]` | Date extraction |
| DateTime | `[[datetime:timestamp]]` | DateTime with timezone |
| Time | `[[time:meeting]]` | Time extraction |
| Duration | `[[duration:length]]` | Time spans |

### Quantifiers

| Syntax | Meaning |
|--------|---------|
| `[[3*item]]` | Exactly 3 items |
| `[[2:4*item]]` | Between 2-4 items |
| `[[*item]]` | Zero or more |
| `[[+item]]` | One or more |
| `[[?item]]` | Zero or one |
| `[[{n}*item]]` | Exactly n items |
| `[[{n,}*item]]` | At least n items |

### Required Fields

```
[[!date:deadline]]              # ! makes field required
[[date:optional]]               # Optional (default)
[[date:required|required]]      # Explicit required option
```

### Constraints

```
[[number:age|min=0,max=120]]    # Number range
[[pick:size|S,M,L|required]]    # Required selection
```

### Custom Actions

```
[[@action:var|param=value]]      # Call Python function
[[@search:results|query={{q}}]]  # Use template variables
```

### CLI

```bash
# Interactive
sd chat "Prompt: [[var]]"
sd chat -p prompt.sd

# Batch
sd batch *.txt "[[extract]]" -o results.json
sd batch *.txt -p prompt.sd -o results.csv

# Options
-o, --output FILE               # Output file (json/csv/xlsx)
-p, --prompt FILE               # Prompt file
-k, --keep-inputs               # Include input fields
-q, --quiet                     # Suppress progress
--verbose                       # Debug logging
```

## Architecture

### Processing Pipeline

1. **Parse** -- Template is parsed into segments and completions
2. **Render** -- Variables are substituted into prompts
3. **Execute** -- LLM calls or custom actions are executed
4. **Extract** -- Responses are validated and stored
5. **Repeat** -- Next segment uses previous results as context

### Memory Model

```
<system> (global system prompt, persistent across segments)
    ↓
Segment 1: prompt → [[var1]]
    ↓
<checkpoint> (memory boundary)
    ↓
Segment 2: {{var1}} → [[var2]]
    ↓
<checkpoint>
    ↓
Segment 3: {{var1}} {{var2}} → [[var3]]
```

Global system messages persist across `<checkpoint>` boundaries.
Only user/assistant message history is cleared. Local system prompts (`<system local>`) are cleared at each checkpoint.

### Caching

- LLM responses cached to `~/.struckdown/cache`
- Cache key includes: prompt, model, parameters
- Credentials excluded from cache key
- Automatic LRU eviction when size limit exceeded

## Environment Variables

```bash
# Required
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"

# Optional
export DEFAULT_LLM="gpt-4o-mini"          # Default model
export STRUCKDOWN_CACHE="~/.struckdown/cache"  # Cache directory
export STRUCKDOWN_CACHE_SIZE=10240        # Cache size (MB)
```

## Advanced Topics

### Parallel Execution

Struckdown automatically parallelizes independent segments:

```
Segment A: [[var_a]]
Segment B: [[var_b]]          # Runs in parallel with A
Segment C: {{var_a}} [[var_c]]  # Waits for A
```

### Pattern Expansion

Date patterns are automatically expanded:

```
[[date*:dates]]  # Input: "first 3 Tuesdays in October"
                 # Output: [date1, date2, date3]
```

### Type Coercion

Parameters in custom actions are automatically coerced:

```python
@Actions.register('add')
def add(context, a: int, b: int):
    return str(a + b)

# "10" and "5" are converted to int automatically
[[@add:result|a=10,b=5]]
```

## Troubleshooting

### Common Issues

**"No module named 'struckdown'"**
```bash
# Make sure it's installed
uv pip list | grep struckdown

# Reinstall if needed
uv pip install git+https://github.com/benwhalley/struckdown
```

**"Set LLM_API_KEY and LLM_API_BASE environment variables"**
```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"
```

**Cache corruption**
```bash
# Clear cache
rm -rf ~/.struckdown/cache

# Or disable caching
export STRUCKDOWN_CACHE=0
```

**Template parsing errors**
- Ensure `[[completions]]` are at the end of segments
- Check for matching `{{` and `}}`
- Verify `<checkpoint>` is on its own line

## Contributing

See issues at [github.com/benwhalley/struckdown](https://github.com/benwhalley/struckdown)

## License

MIT
