# File Includes Example

This directory demonstrates using `{% include %}` directives in struckdown templates.

## Files

- **common_system.sd** - Shared system prompt
- **evaluation_rubric.txt** - Shared evaluation criteria
- **main_template.sd** - Main template that includes the others

## Usage

```bash
# Test the template
cd examples/includes
echo '{"task": "Explain quantum computing"}' | sd chat -p main_template.sd

# View flattened template (with includes resolved)
sd flat main_template.sd
```

## How It Works

The `main_template.sd` uses:
```struckdown
{% include 'common_system.sd' %}
{% include 'evaluation_rubric.txt' %}
```

These files are found because they're in the same directory as the template.

## Search Paths

Struckdown looks for includes in:
1. Same directory as template (this directory)
2. `templates/` subdirectory
3. `./includes/` (project root)
4. `./templates/` (project root)
5. `~/.struckdown/includes/` (global)

## Advanced: Conditional Includes

You can also use Jinja2 features:

```struckdown
{% if mode == 'detailed' %}
  {% include 'detailed_rubric.txt' %}
{% else %}
  {% include 'brief_rubric.txt' %}
{% endif %}
```
