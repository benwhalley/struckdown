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
sd batch -i '*.txt' "Purpose: [[purpose]] Price: [[number:price]]"
```

## Images

Wrap an image with `attach()` and it goes where its variable is rendered:

```python
from struckdown import complete, attach

complete("Read this sheet: {{ sheet }} [[extract:answers]]", {"sheet": attach("scan.jpg")})
```

Images are resized, turned upright and stripped of metadata before sending. From the command
line, use `sd chat --attach name=path` or `sd batch --as-image`. Install with
`pip install 'struckdown[vision]'`; see
[Images](https://github.com/benwhalley/struckdown/blob/main/docs/how-to/images.md).

## Documentation

Full documentation, examples, and tutorials:

**https://github.com/benwhalley/struckdown**

## License

MIT
