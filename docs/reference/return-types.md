---
layout: default
title: Return Types
parent: Reference
nav_order: 4
---

# Return Types

Slots can specify return types to control how the LLM's response is parsed and validated.

## Syntax

```
[[name]]                # Default: string
[[type:name]]           # Built-in type
[[type:name|opts]]      # With constraints
[[CustomModel:name]]    # Pydantic model from context
```

## Built-in Types

### str / respond (default)

String output. This is the default if no type is specified.

```
[[response]]
[[respond:answer]]
```

### extract

Verbatim text extraction -- captures exact text from the input.

```
[[extract:quote]]
[[extract:name]]
```

### think

Internal reasoning -- for chain-of-thought before final answers.

```
[[think:analysis]]
```

### int

Integer output.

```
[[int:count]]
[[int:age|min=0,max=150]]
```

### number / float

Decimal or integer number.

```
[[number:price]]
[[number:score|min=0.0,max=1.0]]
```

### bool

Boolean value. The LLM returns `true` or `false`.

```
[[bool:is_valid]]
[[bool:should_continue]]
```

### pick

Choose from predefined options.

```
[[pick:sentiment|positive,negative,neutral]]
[[pick:priority|low,medium,high,critical]]
```

### date / datetime / time

Temporal extraction.

```
[[date:deadline]]
[[datetime:appointment]]
[[time:start_time]]
```

### json

Arbitrary JSON value.

```
[[json:data]]
[[json:metadata]]
```

Returns a Python dict or list.

### record

JSON object with string keys.

```
[[record:person]]
[[record:info]]
```

## Type Constraints

Add constraints after the type with `|`:

### Numeric Constraints

```
[[int:score|min=1,max=10]]
[[number:rating|min=0.0,max=5.0]]
[[int:count|min=0]]
```

### Required Fields

```
[[!number:price]]           # ! prefix = required
[[number:price|required]]   # Explicit required option
```

### Pattern Constraints

```
[[extract:code|pattern="\w{4}\d+"]]
[[extract:postcode|pattern="[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}"]]
```

## Quantifiers (Lists)

Extract multiple items:

```
[[type*:var]]           # Zero or more items
[[type+:var]]           # One or more items
[[type?:var]]           # Zero or one item
[[type{3}:var]]         # Exactly 3 items
[[type{2,5}:var]]       # Between 2 and 5 items
```

Examples:

```
[[extract+:points]]         # At least one point
[[pick{3}:colours|red,blue,green,yellow]]  # Exactly 3 picks
[[date*:holidays]]          # Zero or more dates
```

## Custom Pydantic Models

Pass Pydantic models in the context to use complex types:

```python
from pydantic import BaseModel, Field
from typing import List
from struckdown import chatter

class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    occupation: str

class Team(BaseModel):
    name: str
    members: List[Person]

result = chatter("""
Extract the team information:

{{text}}

[[Team:team]]
""", context={
    "text": "The Alpha team has John (30, engineer) and Jane (25, designer)",
    "Team": Team,
    "Person": Person,
})

team = result["team"]
print(team.name)              # "Alpha"
print(team.members[0].name)   # "John"
```

### Nested Models

Models can reference other models:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address
    employee_count: int

result = chatter("""
Extract company info: {{text}}
[[Company:company]]
""", context={
    "text": "Acme Inc at 123 Main St, NYC, USA has 500 employees",
    "Company": Company,
    "Address": Address,
})
```

### Optional Fields

```python
from typing import Optional

class Product(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
```

### Field Validation

Use Pydantic's `Field` for additional validation:

```python
from pydantic import Field

class Review(BaseModel):
    rating: int = Field(ge=1, le=5, description="Star rating 1-5")
    text: str = Field(min_length=10, max_length=500)
    verified: bool = False
```

## Special Return Types

### theme / themes

For qualitative analysis, extract themes:

```
[[theme:main_theme]]
[[themes:all_themes]]
```

Returns structured theme objects with name, description, and supporting quotes.

### code / codes

Extract qualitative codes:

```
[[code:primary_code]]
[[codes:all_codes]]
```

Returns code objects with name, description, and evidence.

## Error Handling

If the LLM response cannot be parsed into the requested type:

1. Struckdown retries with validation feedback (up to `max_retries`)
2. If all retries fail, raises a validation error

```python
from struckdown import chatter

try:
    result = chatter("Give me a number [[int:num]]")
except Exception as e:
    print(f"Failed to parse: {e}")
```

### Validation Messages

When validation fails, the error message is sent back to the LLM:

```
Validation error: value is not a valid integer
Please provide a valid integer.
```

This allows the model to self-correct.

## Type Coercion

Struckdown attempts reasonable type coercion:

| Input | Target | Result |
|-------|--------|--------|
| `"42"` | `int` | `42` |
| `"3.14"` | `float` | `3.14` |
| `"true"` | `bool` | `True` |
| `"yes"` | `bool` | `True` |
| `["a", "b"]` | `list` | `["a", "b"]` |
