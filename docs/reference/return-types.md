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
[[name]]              # Default: string
[[name:type]]         # Built-in type
[[name:type|opts]]    # With constraints
[[name:CustomModel]]  # Pydantic model from context
```

## Built-in Types

### str (default)

String output. This is the default if no type is specified.

```
[[response]]
[[response:str]]
```

### int

Integer output.

```
[[count:int]]
[[age:int|min=0,max=150]]
```

### float

Decimal number.

```
[[price:float]]
[[score:float|min=0.0,max=1.0]]
```

### bool

Boolean value.

```
[[is_valid:bool]]
[[should_continue:bool]]
```

The LLM will return `true` or `false`.

### list

List of strings.

```
[[items:list]]
[[keywords:list|max_items=10]]
```

### json

Arbitrary JSON object.

```
[[data:json]]
[[metadata:json]]
```

Returns a Python dict.


## Type Constraints

Add constraints after the type with `|`:

### String Constraints

```
[[summary:str|max_length=100]]
[[title:str|min_length=5,max_length=50]]
```

### Numeric Constraints

```
[[score:int|min=1,max=10]]
[[rating:float|min=0.0,max=5.0]]
[[count:int|min=0]]
```

### List Constraints

```
[[items:list|max_items=5]]
[[tags:list|min_items=1,max_items=10]]
```

### Enum Constraints

```
[[sentiment:str|enum=positive,negative,neutral]]
[[priority:str|enum=low,medium,high,critical]]
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

[[team:Team]]
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
[[company:Company]]
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
[[main_theme:theme]]
[[all_themes:themes]]
```

Returns structured theme objects with name, description, and supporting quotes.

### code / codes

Extract qualitative codes:

```
[[primary_code:code]]
[[codes:codes]]
```

Returns code objects with name, description, and evidence.


## Error Handling

If the LLM response cannot be parsed into the requested type:

1. Struckdown retries with validation feedback (up to `max_retries`)
2. If all retries fail, raises a validation error

```python
from struckdown import chatter

try:
    result = chatter("Give me a number [[num:int]]")
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
