# Custom Actions Guide

Extend Struckdown with custom Python functions that bypass the LLM.

## Overview

Actions allow you to register Python functions that can be called from templates using `[[action:var|params]]` syntax. This is useful for:

- **RAG (Retrieval-Augmented Generation)** -- Search databases, vector stores, or APIs
- **Data transformations** -- Format, validate, or process extracted data
- **External integrations** -- Call APIs, query databases, read files
- **Performance** -- Skip LLM calls for deterministic operations

## Basic Example

### Register an Action

```python
from struckdown import Actions, chatter

@Actions.register('uppercase')
def uppercase_text(context, text: str):
    """Convert text to uppercase"""
    return text.upper()

# Use in template
result = chatter("[[uppercase:loud|text=hello world]]")
print(result['loud'])  # "HELLO WORLD"
```

### Template Syntax

Actions use `[[action:var|params]]` instead of `[[type:var]]`:

```python
# LLM completion (calls AI)
[[pick:color|red,blue,green]]

# Custom action (calls Python function)
[[uppercase:result|text=hello]]
```

## Action Parameters

### Parameter Passing

Parameters are passed as `key=value` pairs:

```python
@Actions.register('greet')
def greet(context, name: str, greeting: str = "Hello"):
    """Greet someone"""
    return f"{greeting}, {name}!"

# Use it
chatter("[[greet:message|name=Alice,greeting=Hi]]")
# Output: "Hi, Alice!"
```

### Using Context Variables

Reference previous extractions with `{{variable}}`:

```python
template = """
Extract name: [[name]]

¡OBLIVIATE

Greet them: [[greet:greeting|name={{name}}]]
"""

result = chatter(template, context={"input": "My name is Bob"})
print(result['greeting'])  # "Hello, Bob!"
```

### Type Coercion

Struckdown automatically converts string parameters to the correct type based on function signature:

```python
@Actions.register('multiply')
def multiply(context, value: int, factor: int = 2):
    """Multiply value by factor"""
    return str(value * factor)

# String "10" is automatically converted to int 10
chatter("[[multiply:result|value=10,factor=5]]")
# Output: "50"
```

Supported types:
- `str` (default)
- `int`, `float`
- `bool` ("true"/"false" converted)
- `List[T]`, `Dict[str, T]` (JSON parsing)

## Accessing Context

The `context` parameter provides access to all previously extracted variables:

```python
@Actions.register('count_extractions')
def count_extractions(context):
    """Count how many variables have been extracted"""
    return f"Extracted {len(context)} variables: {', '.join(context.keys())}"

template = """
Name: [[name]]
Age: [[int:age]]

¡OBLIVIATE

Summary: [[count_extractions:summary]]
"""
```

## Error Handling

Control how errors are handled with the `on_error` parameter:

### Propagate (Default)

Raises exceptions immediately:

```python
@Actions.register('strict_action', on_error='propagate')
def strict_action(context):
    raise ValueError("Something went wrong")

# This will raise ValueError
```

### Return Empty

Returns empty string on error:

```python
@Actions.register('safe_action', on_error='return_empty')
def safe_action(context, url: str):
    try:
        return fetch_data(url)
    except Exception:
        raise  # Will be caught and return ""

# If fetch fails, continues with empty string
```

### Log and Continue

Logs error and returns empty string:

```python
@Actions.register('logged_action', on_error='log_and_continue')
def logged_action(context, query: str):
    """Search database"""
    # If this fails, error is logged but execution continues
    return database.search(query)
```

## Real-World Examples

### RAG with Vector Search

```python
from struckdown import Actions, chatter
import chromadb

# Initialize your vector database
db = chromadb.Client()
collection = db.get_or_create_collection("docs")

@Actions.register('search_docs', on_error='return_empty')
def search_docs(context, query: str, n: int = 3):
    """Search documentation using vector similarity"""
    results = collection.query(
        query_texts=[query],
        n_results=n
    )

    # Format results
    docs = results['documents'][0]
    return "\n\n".join(f"- {doc}" for doc in docs)

# Use in template
template = """
User question: {{question}}

Relevant docs:
[[search_docs:context|query={{question}},n=5]]

¡OBLIVIATE

Based on this context:
{{context}}

Answer the question: {{question}}

[[answer]]
"""

result = chatter(template, context={"question": "How do I use actions?"})
```

### Database Query

```python
import sqlite3

@Actions.register('query_users', on_error='return_empty')
def query_users(context, email: str):
    """Look up user by email"""
    conn = sqlite3.connect('users.db')
    cursor = conn.execute(
        "SELECT name, role FROM users WHERE email = ?",
        (email,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return f"Name: {row[0]}, Role: {row[1]}"
    return "User not found"

# Use it
template = """
Email from logs: [[extract:email]]

¡OBLIVIATE

User info: [[query_users:user|email={{email}}]]

Personalized response for {{user}}: [[response]]
"""
```

### API Integration

```python
import requests

@Actions.register('weather', on_error='return_empty')
def get_weather(context, city: str, units: str = "metric"):
    """Fetch current weather"""
    api_key = os.getenv("WEATHER_API_KEY")

    response = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": units}
    )
    response.raise_for_status()

    data = response.json()
    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f"Temperature: {temp}°C, Conditions: {desc}"

# Use it
template = """
Extract city: [[city]]

¡OBLIVIATE

Weather: [[weather:conditions|city={{city}}]]

Travel advice for {{city}} given {{conditions}}: [[advice]]
"""
```

### Data Transformation

```python
from datetime import datetime

@Actions.register('format_date')
def format_date(context, iso_date: str, format: str = "%B %d, %Y"):
    """Convert ISO date to readable format"""
    dt = datetime.fromisoformat(iso_date)
    return dt.strftime(format)

@Actions.register('calculate_age')
def calculate_age(context, birth_date: str):
    """Calculate age from birth date"""
    birth = datetime.fromisoformat(birth_date)
    today = datetime.now()
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return str(age)

# Use them
template = """
Extract birth date (ISO format): [[date:birth]]

¡OBLIVIATE

Birth date: [[format_date:formatted|iso_date={{birth}},format=%d/%m/%Y]]
Age: [[calculate_age:age|birth_date={{birth}}]]

Birthday message for someone aged {{age}}: [[message]]
"""
```

## Best Practices

### 1. Type Hints

Always use type hints for automatic parameter validation:

```python
# Good
@Actions.register('add')
def add(context, a: int, b: int):
    return str(a + b)

# Bad (parameters are strings)
@Actions.register('add')
def add(context, a, b):
    return str(int(a) + int(b))  # Manual conversion
```

### 2. Return Strings

Actions should always return strings (they're inserted into templates):

```python
# Good
@Actions.register('count')
def count(context, items: List[str]):
    return str(len(items))

# Bad (returns int, will cause errors)
@Actions.register('count')
def count(context, items: List[str]):
    return len(items)
```

### 3. Error Handling

Use `on_error='return_empty'` for non-critical operations:

```python
# Non-critical -- if fetch fails, continue anyway
@Actions.register('fetch_metadata', on_error='return_empty')
def fetch_metadata(context, url: str):
    return requests.get(url).json()

# Critical -- if validation fails, stop immediately
@Actions.register('validate_license', on_error='propagate')
def validate_license(context, key: str):
    if not is_valid(key):
        raise ValueError("Invalid license")
    return "Valid"
```

### 4. Descriptive Names

Use clear, verb-based names:

```python
# Good
@Actions.register('search_documents')
@Actions.register('calculate_score')
@Actions.register('format_currency')

# Bad
@Actions.register('docs')
@Actions.register('score')
@Actions.register('money')
```

### 5. Documentation

Add docstrings -- they help users understand your actions:

```python
@Actions.register('search_products')
def search_products(context, query: str, limit: int = 10):
    """
    Search product database using fuzzy matching.

    Args:
        query: Search term
        limit: Maximum results to return (default: 10)

    Returns:
        Formatted list of products with prices
    """
    # implementation
```

## Security Considerations

⚠️ **Important**: Custom actions execute arbitrary Python code. See [SECURITY.md](../SECURITY.md) for guidelines.

**Never do this**:

```python
# DANGEROUS -- arbitrary code execution
@Actions.register('eval_code')
def eval_code(context, code: str):
    return str(eval(code))  # NEVER USE eval()!

# DANGEROUS -- command injection
@Actions.register('run_command')
def run_command(context, cmd: str):
    import subprocess
    return subprocess.run(cmd, shell=True, capture_output=True).stdout  # NEVER use shell=True!
```

**Do this instead**:

```python
# Safe -- limited operations only
@Actions.register('calculate')
def calculate(context, expression: str):
    """Safe arithmetic calculator"""
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    node = ast.parse(expression, mode='eval')

    # Validate only safe operations
    for n in ast.walk(node):
        if isinstance(n, ast.operator) and type(n) not in allowed_ops:
            raise ValueError(f"Unsafe operation: {type(n)}")

    return str(eval(compile(node, '<string>', 'eval')))
```

## Testing Actions

Test your actions before using them in production:

```python
import unittest
from struckdown import Actions, chatter

class TestMyActions(unittest.TestCase):
    def setUp(self):
        # Register actions
        @Actions.register('double')
        def double(context, value: int):
            return str(value * 2)

    def test_double_action(self):
        """Test double action with direct execution"""
        model = Actions.create_action_model('double', ['value=5'], None, False)
        result, _ = model._executor(context={}, rendered_prompt="")
        self.assertEqual(result.response, "10")

    def test_double_in_template(self):
        """Test double action in template"""
        result = chatter("[[double:result|value=7]]")
        self.assertEqual(result['result'], "14")
```

## Listing Registered Actions

Check which actions are available:

```python
from struckdown import Actions

# List all registered actions
print(Actions.list_registered())

# Check if specific action exists
if Actions.is_registered('search_docs'):
    print("Search action is available")
```

## Advanced: Stateful Actions

Actions can maintain state across invocations:

```python
class SearchCache:
    def __init__(self):
        self.cache = {}

    def search(self, context, query: str):
        """Cached search"""
        if query in self.cache:
            return self.cache[query]

        result = expensive_search(query)
        self.cache[query] = result
        return result

# Create instance
search_cache = SearchCache()

# Register the bound method
Actions.register('cached_search')(search_cache.search)
```

## See Also

- **[Tutorial](TUTORIAL.md)** -- Learn Struckdown basics
- **[Reference](REFERENCE.md)** -- Complete syntax reference
- **[Examples](../examples/)** -- More examples
- **[Security](../SECURITY.md)** -- Security guidelines
