# struckdown

Markdown-based syntax for structured conversations with language models.

## Installation

```bash
pip install struckdown
```

## Quick Example

```bash
# Configure
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"

# Extract structured data
sd chat "Tell me a joke: [[joke]]"
sd batch *.txt "Purpose: [[purpose]] Price: [[number:price]]"
```

## Documentation

Full documentation, examples, and tutorials:

**https://github.com/benwhalley/struckdown**

## License

MIT
