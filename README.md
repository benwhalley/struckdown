# struckdown

Markdown-like syntax for structured conversations with language models.

## QuickStart

```bash
# Install
uv tool install git+https://github.com/benwhalley/struckdown

# Configure
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"

# Try it
sd chat "Tell me a joke: [[joke]]"
```

**[→ Full QuickStart Guide](docs/QUICKSTART.md)**

## What is Struckdown?

Struckdown makes it easy to extract structured data from text using LLMs with a simple, markdown-inspired syntax.

### Example: Batch Processing

Imagine you have product descriptions:

```bash
% sd batch *.txt "Purpose, <5 words: [[purpose]]"
[
  {
    "filename": "butter_robot.txt",
    "purpose": "Pass butter, question existence."
  },
  {
    "filename": "plumbus.txt",
    "purpose": "Household universal utility device."
  },
  {
    "filename": "portal_gun.txt",
    "purpose": "Interdimensional travel device."
  }
]
```

### Example: Type Extraction

Extract structured data with type constraints:

```bash
% sd batch *.txt "Price: [[number:price]] Currency? [[pick:currency|schmeckles,brapples,flurbos]]"
[
  {
    "filename": "butter_robot.txt",
    "price": 18,
    "currency": "schmeckles"
  },
  {
    "filename": "plumbus.txt",
    "price": 6.5,
    "currency": "brapples"
  }
]
```

### Example: Chaining Operations

Batch operations accept JSON, so you can chain commands:

```bash
% sd batch *.txt "Purpose: [[purpose]] Name: [[name]]" | \
  sd batch "Most similar on Amazon: [[product]]" -k
```

## Key Features

- **Simple syntax** -- `[[variable]]` for completions, `{{variable}}` for references
- **Type safety** -- Extract booleans, numbers, dates, or pick from options
- **Memory management** -- Use `¡OBLIVIATE` to save tokens between steps
- **Batch processing** -- Process hundreds of files with progress bars
- **Caching** -- Automatic disk caching saves money and time
- **Custom actions** -- Extend with Python functions (RAG, APIs, databases)
- **Multiple outputs** -- JSON, CSV, Excel, or stdout

## Documentation

### Getting Started
- **[QuickStart](docs/QUICKSTART.md)** -- Get started in 5 minutes
- **[CLI Usage](docs/CLI_USAGE.md)** -- Complete command reference

### Tutorials
- **[Building a RAG System](docs/TUTORIAL_RAG.md)** -- Extract → Search → Generate pattern
- **[Custom Actions](docs/CUSTOM_ACTIONS.md)** -- Extend with Python plugins

### Reference
- **[Examples](examples/)** -- Real-world examples and test cases
- **[Security](SECURITY.md)** -- Security guidelines and best practices

## Installation

Requires [UV](https://docs.astral.sh/uv/):

```bash
# Install as a tool (recommended)
uv tool install git+https://github.com/benwhalley/struckdown

# Or install in current environment
uv pip install git+https://github.com/benwhalley/struckdown
```

## Configuration

Set these environment variables:

```bash
export LLM_API_KEY="sk-..."              # Your API key
export LLM_API_BASE="https://api.openai.com/v1"  # API endpoint
export DEFAULT_LLM="gpt-4o-mini"         # Model name
```

### VSCode Extension

Syntax highlighting for `.sd` files:

```bash
cd vscode-extension && ./install.sh
```

Select theme: **Cmd/Ctrl+Shift+P** → "Color Theme" → "Struckdown Dark"

## Basic Syntax

### Completions (Slots)

Use `[[slot]]` to mark where the LLM should respond:

```bash
sd chat "Explain quantum physics: [[explanation]]"
```

### Typed Completions

Specify the type of response:

```bash
# Boolean
sd chat "Is the sky blue? [[bool:answer]]"

# Pick from options
sd chat "Choose [[pick:color|red,green,blue]]"

# Numbers
sd chat "Price: $19.99 [[number:price]]"

# Dates
sd chat "Meeting on Jan 15, 2024 [[date:meeting]]"
```

### Variables

Reference previous extractions with `{{variable}}`:

```
Extract name: [[name]]

¡OBLIVIATE

Hello {{name}}, how are you? [[response]]
```

### Memory Boundaries

Use `¡OBLIVIATE` to create memory boundaries and save tokens:

```
Long expensive context...

Summary: [[summary]]

¡OBLIVIATE

Translate {{summary}} to Spanish: [[translation]]
```

Everything before `¡OBLIVIATE` is forgotten -- only extracted variables carry forward.

## CLI Commands

### `sd chat` - Interactive Mode

```bash
sd chat "Tell me a joke: [[joke]]"
sd chat -p prompt.sd
echo "Process this" | sd chat
```

### `sd batch` - Batch Processing

```bash
# Basic usage
sd batch *.txt "Extract [[name]]" -o results.json

# With prompt file
sd batch *.txt -p prompt.sd -o results.csv

# Keep input fields
sd batch *.txt "[[summary]]" -k

# Chain operations
sd batch *.txt "[[purpose]]" | sd batch "Similar: [[product]]" -k
```

**Output formats** (auto-detected from extension):
- `.json` -- JSON array
- `.csv` -- CSV file
- `.xlsx` -- Excel spreadsheet
- None -- Pretty-printed to stdout

## Caching

Struckdown automatically caches LLM responses to disk:

```bash
# Default cache location
~/.struckdown/cache  # 10 GB limit (LRU eviction)

# Disable caching
export STRUCKDOWN_CACHE=0

# Custom cache directory
export STRUCKDOWN_CACHE=/path/to/cache

# Custom size limit (MB)
export STRUCKDOWN_CACHE_SIZE=5120  # 5 GB
```

## Advanced Features

### List Completions

Extract multiple items:

```bash
# Exactly 3 items
sd chat "Name 3 fruits: [[3*pick:fruit|apple,banana,orange]]"

# Between 2 and 4 items
sd chat "Name 2-4 animals: [[2:4*extract:animals]]"

# Any number
sd chat "List all mentioned: [[*extract:items]]"
```

### Date/Time Extraction

```bash
# Single date
sd chat "Meeting Jan 15 [[date:when]]"

# Date range with pattern expansion
sd chat "Every Tuesday in October [[date*:dates]]"

# With validation
sd chat "Deadline [[!date:deadline]]"  # ! makes it required
```

### Number Extraction

```bash
# With constraints
sd chat "Age (0-120): [[number:age|min=0,max=120]]"

# Required numbers
sd chat "Price: [[!number:price]]"

# Multiple numbers
sd chat "Extract all prices: [[number*:prices]]"
```

### Custom Actions

Extend Struckdown with Python functions:

```python
from struckdown import Actions, chatter

@Actions.register('uppercase')
def uppercase_text(context, text: str):
    return text.upper()

# Use in template
result = chatter("[[uppercase:loud|text={{input}}]]")
```

See **[Custom Actions Guide](docs/CUSTOM_ACTIONS.md)** for details.

### Shared Headers

Use `¡BEGIN` to define a shared header for all segments:

```
You are an expert analyst.

¡BEGIN

First analysis: [[analysis1]]

¡OBLIVIATE

Second analysis: [[analysis2]]
```

The header is prepended to every segment after `¡OBLIVIATE`.

### Model/Temperature Overrides

Override per-slot settings:

```
# Different temperature
[[think:reasoning|temperature=0.3]]

# Different model
[[pick:choice|red,blue|model=gpt-4]]

# Combine
[[extract:data|model=gpt-4,temperature=0.0]]
```

## Examples

See **[examples/](examples/)** for:
- Basic completions
- Multi-step workflows
- RAG with custom actions
- Batch processing patterns
- Date/time extraction
- Number validation

## Contributing

Issues and pull requests welcome at [github.com/benwhalley/struckdown](https://github.com/benwhalley/struckdown)

## License

MIT
