# struckdown

Markdown-based syntax for ***structured*** conversations with language models.


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


# Using a prompt file with actions

```
# sets a config variable to run a web search for "oranges" and 
# summarise the results
sd chat "[[@search|oranges]]  Provide a 2-3 sentence summary [[summary]]"
```

**[→ Full QuickStart Guide](docs/QUICKSTART.md)**

## What is Struckdown?

Struckdown makes it easy to extract structured data from text using LLMs with a simple, markdown-inspired syntax.

### Example: Batch Processing

Imagine you have unstructured data stored in free text. You can make it structured like this:

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
- **System messages** -- Control LLM behavior with `<system>` tags
- **Type safety** -- Extract booleans, numbers, dates, or pick from options
- **Token management** -- Use `<checkpoint>` to save tokens between steps
- **Batch processing** -- Process hundreds of files with progress bars
- **Caching** -- Automatic disk caching saves money and time
- **Custom actions** -- Extend with Python functions (RAG, APIs, databases)
- **Multiple outputs** -- JSON, CSV, Excel, or stdout
- **Web search and URL fetching** -- Extract data directly from web pages



### Command Line

```bash
# Extract product data from a web page
sd chat "{{source}} Extract the product name and price [[product:data]]" \
  -s https://www.example.com/product/12345

# Fetch raw HTML (no readability processing)
sd chat "{{source}} Analyze the HTML structure [[analysis]]" \
  -s https://example.com --raw
```

### In Templates (for Batch Processing)

Use the `@fetch` action to fetch URLs dynamically within templates:

```
[[@fetch:page_content|product_url]]

Based on this product page:
{{page_content}}

Extract:
- Product name: [[name]]
- Price: [[number:price]]
```

With an input spreadsheet containing a `product_url` column:

```bash
sd batch products.xlsx template.sd -o results.xlsx
```

Each row's URL will be fetched, processed with readability to extract the main content, and converted to markdown before being passed to the LLM.

#### @fetch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `url` | required | URL to fetch (unquoted = variable, quoted = literal) |
| `raw` | `false` | Return raw HTML instead of markdown |
| `timeout` | `30` | Request timeout in seconds |
| `max_chars` | `32000` | Max characters (0 = no limit) |

Example with parameters:
```
[[@fetch:content|url,raw=true,timeout=60,max_chars=0]]
```

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

### Claude Code Skill

Struckdown includes a skill for [Claude Code](https://claude.ai/code) that helps you write well-engineered prompts:

```bash
# Install the skill
sd install-skill

# Then in Claude Code, use:
# /struckdown extract contact details from business cards
# /struckdown analyse sentiment with slots: sentiment, urgency
```

The skill guides you through:
- Gathering requirements and clarifying intent
- Choosing appropriate slot types and constraints
- Testing prompts with sample data
- Suggesting batch processing commands

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

# JSON (any valid JSON value)
sd chat "Return data as JSON [[json:data]]"

# Record (JSON object with string keys)
sd chat "Extract as key-value pairs [[record:info]]"
```

### Variables

Reference previous extractions with `{{variable}}`:

```
Extract name: [[name]]

<checkpoint>

Hello {{name}}, how are you? [[response]]
```

### Memory Boundaries

Use `<checkpoint>` to create memory boundaries and save tokens:

```
Long expensive context...

Summary: [[summary]]

<checkpoint>

Translate {{summary}} to Spanish: [[translation]]
```

Everything before `<checkpoint>` is forgotten -- only extracted variables carry forward.

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

### `sd check` - Validate Prompts

Check prompt syntax and display execution plan:

```bash
# Validate and show structure
sd check prompt.sd
```

Shows system prompt info, sections, completions, dependencies, and line numbers.

### `sd graph` - Visualize Prompts

Generate dependency graph visualizations:

```bash
# Generate HTML visualization (default)
sd graph prompt.sd

# Generate Mermaid diagram text
sd graph prompt.sd -o diagram.mmd
```

Creates interactive diagrams showing sections, completions, dependencies, and execution flow.

### `sd flat` - Flatten Templates

Resolve all `{% include %}` directives and output flattened template:

```bash
# Output to stdout
sd flat prompt.sd

# Save to file
sd flat prompt.sd -o flattened.sd
```

Useful for debugging includes or creating self-contained templates.

## File Includes

Struckdown supports file includes using Jinja2's `{% include %}` syntax:

```struckdown
{# Include shared system prompt #}
{% include 'common/system.sd' %}

{# Include evaluation rubric #}
{% include 'rubrics/essay_criteria.txt' %}

Process: {{input}}
Result: [[output]]
```

**Search paths** (in priority order):
1. Same directory as template file
2. `templates/` subdirectory relative to template file
3. `./includes/` (project-local includes)
4. `./templates/` (project-local templates)
5. `~/.struckdown/includes/` (global user includes)

**Advanced includes:**

```struckdown
{# Conditional includes #}
{% if verbose %}
  {% include 'detailed_instructions.sd' %}
{% else %}
  {% include 'brief_instructions.sd' %}
{% endif %}

{# Dynamic includes with variables #}
{% set rubric = 'rubrics/' + grade_level + '.sd' %}
{% include rubric %}
```

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

## Embeddings

Generate text embeddings using API or local models:

```python
from struckdown import get_embedding

# API embeddings (default)
embeddings = get_embedding(["text 1", "text 2"])

# Local embeddings (requires: uv pip install struckdown[local])
embeddings = get_embedding(texts, model="local/all-MiniLM-L6-v2")
```

Use `local/model-name` prefix for any sentence-transformers model. API embeddings use `LLM_API_KEY` and `LLM_API_BASE` environment variables.

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

### Pattern Matching

Constrain text extraction with regex patterns:

```bash
# Module code: 4 letters followed by digits
sd chat 'Module code: [[x|pattern="\w{4}\d+"]]'

# UK postcode pattern
sd chat 'Postcode: [[postcode|pattern="[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}"]]'

# Email-like pattern
sd chat 'Email: [[email|pattern="[^@]+@[^@]+\.[^@]+"]]'
```

Note: Patterns must be quoted strings. Use `\\` for literal backslashes.

### Custom Actions

Extend Struckdown with Python functions:

```python
from struckdown import Actions, chatter

@Actions.register('uppercase')
def uppercase_text(context, text: str):
    return text.upper()

# Use in template - unquoted 'input' is a variable reference
result = chatter("[[@uppercase:loud|text=input]]", context={"input": "hello"})
```

See **[Custom Actions Guide](docs/CUSTOM_ACTIONS.md)** for details.

### System Messages

Control system messages using XML-style `<system>` tags:

```
<system>You are an expert data analyst with 10 years of experience.</system>

<system local>Always provide concise, data-driven responses.</system>

First analysis: [[analysis1]]

<checkpoint>

Second analysis: [[analysis2]]
```

**Global system messages** (`<system>`) set the LLM's role and persist across all checkpoints.
**Local system messages** (`<system local>`) provide instructions that only apply to the current segment.

Multiple `<system>` tags append to the system message by default. Use modifiers:

```
<system>Base instructions.</system>
<system>Additional instructions.</system>        <!-- appends -->
<system replace>Replace all previous.</system>  <!-- replaces -->

<system local>This segment only.</system>       <!-- cleared after checkpoint -->
<system local replace>New local only.</system>  <!-- replaces local -->
```

All support template variables: `{{variable}}`.

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
