# QuickStart Guide

Get started with Struckdown in minutes.

## Installation

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install struckdown
uv tool install git+https://github.com/benwhalley/struckdown
```

## Configuration

Set your LLM credentials:

```bash
export LLM_API_KEY="sk-..."  # Your API key
export LLM_API_BASE="https://api.openai.com/v1"
export DEFAULT_LLM="gpt-4o-mini"
```

## Your First Prompt

Test the CLI:

```bash
sd chat "Tell me a joke: [[joke]]"
```

This simple prompt asks the LLM to tell a joke and stores it in a variable called `joke`.

## Basic Syntax

### Completions (Slots)

Struckdown uses `[[slot]]` syntax to tell the LLM where to generate responses:

```bash
sd chat "Explain quantum physics: [[explanation]]"
```

The final slot can be omitted in simple prompts:

```bash
sd chat "Tell me a joke"  # Automatically becomes [[response]]
```

### Typed Completions

Specify the type of response you want:

```bash
# Boolean (true/false)
sd chat "Is the sky blue? [[bool:answer]]"

# Pick from options
sd chat "Choose a color [[pick:color|red,green,blue]]"

# Number extraction
sd chat "Price: $19.99 [[number:price]]"

# Integer
sd chat "Count the words: 'hello world' [[int:count]]"
```

### Template Variables

Use `{{variable}}` to reference previous results:

```bash
sd chat "Name a fruit: [[fruit]]

¡OBLIVIATE

Tell me a joke about {{fruit}}: [[joke]]"
```

## Batch Processing

Process multiple files at once:

```bash
# Process text files
sd batch *.txt "Summarize in 5 words: [[summary]]" -o results.json

# Extract structured data
sd batch reviews/*.txt "Sentiment [[pick:sentiment|positive,negative,neutral]]" -o sentiment.csv

# Chain multiple operations
sd batch *.txt "Extract name: [[name]]" | \
  sd batch "Generate email for {{name}}: [[email]]" -k
```

## Memory Management with ¡OBLIVIATE

Use `¡OBLIVIATE` to create memory boundaries and save tokens:

```bash
sd chat "Long context...

Generate summary: [[summary]]

¡OBLIVIATE

Translate {{summary}} to Spanish: [[translation]]"
```

Everything before `¡OBLIVIATE` is forgotten -- only the extracted variables (`{{summary}}`) are available in the next section.

## Next Steps

- **[Tutorial](TUTORIAL.md)** -- Learn Struckdown step-by-step
- **[Reference](REFERENCE.md)** -- Complete syntax documentation
- **[Custom Actions](CUSTOM_ACTIONS.md)** -- Extend Struckdown with plugins
- **[Examples](../examples/)** -- Real-world examples
- **[CLI Usage](CLI_USAGE.md)** -- Complete CLI reference

## Common Patterns

### Extract structured data from files

```bash
sd batch documents/*.txt "
Name: [[name]]
Email: [[email]]
Phone: [[number:phone]]
" -o contacts.csv
```

### Multi-step analysis

```bash
sd chat "Analyze this code:

\`\`\`python
def hello(): print('hi')
\`\`\`

Issues: [[issues]]

¡OBLIVIATE

Issues found: {{issues}}

Suggested fixes: [[fixes]]"
```

### Classification pipeline

```bash
sd batch emails/*.txt "
Sentiment: [[pick:sentiment|positive,negative,neutral]]
Urgent: [[bool:urgent]]
Category: [[pick:category|sales,support,billing,other]]
" -o classified.json
```
