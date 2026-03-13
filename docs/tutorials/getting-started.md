---
layout: default
title: Getting Started
parent: Tutorials
nav_order: 1
---

# Getting Started with Struckdown

Struckdown lets you extract structured data from text using LLMs. Instead of parsing free-form responses, you define exactly what you want and get validated, typed results.

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
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"
export DEFAULT_LLM="gpt-4o-mini"
```

## Your First Extraction

The core idea: use `[[slot]]` to mark where the LLM should respond, and get back structured data.

```bash
sd chat "The sky is blue. Is this true? [[bool:is_true]]"
```

Output:
```json
{"is_true": true}
```

Compare this to a raw LLM call -- you'd get "Yes, that's correct!" and have to parse it yourself.

## Why Struckdown?

### 1. Typed Extractions

Get exactly the data type you need:

```bash
# Boolean
sd chat "Is Python a programming language? [[bool:answer]]"
# {"answer": true}

# Number with constraints
sd chat "Rate this product (1-10): 'Amazing quality!' [[int:rating|min=1,max=10]]"
# {"rating": 9}

# Pick from options
sd chat "Classify: 'I hate waiting' [[pick:sentiment|positive,negative,neutral]]"
# {"sentiment": "negative"}

# Date extraction
sd chat "Meeting scheduled for next Tuesday [[date:when]]"
# {"when": "2024-01-16"}
```

### 2. Batch Processing

Process hundreds of files with one command:

```bash
# Summarise all text files
sd batch *.txt "Summarise in 5 words: [[summary]]" -o summaries.json

# Extract structured data from documents
sd batch documents/*.txt "
Name: [[extract:name]]
Email: [[extract:email]]
Phone: [[extract:phone]]
" -o contacts.csv

# Classify with multiple fields
sd batch reviews/*.txt "
Sentiment: [[pick:sentiment|positive,negative,neutral]]
Urgent: [[bool:urgent]]
Topic: [[pick:topic|billing,support,sales,other]]
" -o classified.xlsx
```

### 3. Multi-Step Reasoning

Use `<checkpoint>` to break complex tasks into stages, saving tokens:

```bash
sd chat "
Read this article and identify the main argument:
{{input}}

Main argument: [[extract:argument]]

<checkpoint>

Now critique this argument: {{argument}}

Critique: [[critique]]
" -s article.txt
```

Everything before `<checkpoint>` is forgotten -- only `{{argument}}` carries forward. This saves tokens on long documents.

### 4. Required Fields and Validation

Ensure you always get what you need:

```bash
# ! prefix makes the field required
sd chat "Extract the price: 'Contact us for pricing' [[!number:price]]"
# Will indicate no valid price found rather than guessing

# Pattern matching
sd chat 'Find the module code: "PSYC2001 is great" [[extract:code|pattern="\w{4}\d+"]]'
# {"code": "PSYC2001"}
```

## Common Patterns

### Extract Structured Data from Files

```bash
sd batch invoices/*.pdf "
Invoice number: [[extract:invoice_no]]
Date: [[date:date]]
Total: [[number:total]]
Paid: [[bool:paid]]
" -o invoices.csv
```

### Classify and Route

```bash
sd batch emails/*.txt "
Priority: [[pick:priority|high,medium,low]]
Department: [[pick:dept|sales,support,billing,hr]]
Requires response: [[bool:needs_reply]]
" -o routing.json
```

### Chain Operations

Pipe JSON output through multiple processing steps:

```bash
sd batch *.txt "Extract company name: [[extract:company]]" | \
  sd batch "Find {{company}} stock ticker: [[extract:ticker]]" -k
```

The `-k` flag keeps input fields, so you get both `company` and `ticker` in the output.

### Web Research

Fetch and process web content:

```bash
# Fetch a URL and extract data
sd chat "{{source}} Extract the main product and price [[extract:product]] [[number:price]]" \
  -s https://example.com/product

# Or use the @search action for web search
sd chat "[[@search:results|query='best python testing frameworks']] Summarise the top 3: [[summary]]"
```

## Using Prompt Files

For complex prompts, save them as `.sd` files:

```
{# feedback_classifier.sd #}
<system>
You are a customer feedback analyst.
Be objective and precise.
</system>

Customer feedback:
{{input}}

Analysis:
- Sentiment: [[pick:sentiment|positive,negative,neutral,mixed]]
- Topics: [[pick+:topics|product,service,price,delivery,quality]]
- Urgency: [[int:urgency|min=1,max=5]]
- Summary: [[extract:summary]]
```

Run with:

```bash
sd batch feedback/*.txt -p feedback_classifier.sd -o analysis.xlsx
```

## Python API

Use struckdown programmatically:

```python
from struckdown import chatter

result = chatter("""
Analyse this customer review:
{{review}}

Sentiment: [[pick:sentiment|positive,negative,neutral]]
Rating: [[int:rating|min=1,max=5]]
Key points: [[extract+:points]]
""", context={"review": "Great product but shipping was slow"})

print(result["sentiment"])  # "positive"
print(result["rating"])     # 4
print(result["points"])     # ["Good product quality", "Slow shipping"]
print(result.total_cost)    # 0.0001 (USD)
```

For async processing:

```python
from struckdown import chatter_async
import asyncio

async def process_many(reviews):
    tasks = [
        chatter_async("Sentiment: [[pick:sentiment|pos,neg]] {{r}}", context={"r": r})
        for r in reviews
    ]
    return await asyncio.gather(*tasks)
```

## Next Steps

- **[Template Syntax](../explanation/template-syntax.md)** -- Complete syntax reference
- **[CLI Reference](../reference/cli.md)** -- All CLI commands and options
- **[Custom Actions](../how-to/custom-actions.md)** -- Extend with Python plugins
- **[Caching](../explanation/caching.md)** -- How caching works
