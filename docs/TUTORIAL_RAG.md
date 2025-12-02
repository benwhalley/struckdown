# Tutorial: RAG with Custom Actions

Learn the "Extract → Search → Generate" pattern for building RAG systems with Struckdown.

## The Pattern

1. **Extract** -- LLM identifies what information is needed: `[[query]]`
2. **Search** -- Custom function retrieves context: `[[@search:context|query={{query}}]]`
3. **Generate** -- LLM generates answer using context: `[[answer]]`

## Example: Knowledge Base Search

### Register the Action

```python
from struckdown import Actions, chatter

@Actions.register('search_docs', on_error='return_empty')
def search_docs(context, query: str, n: int = 3):
    """Search knowledge base for relevant docs"""
    # In production: query vector database, API, etc.
    # For demo: simple mock
    knowledge = {
        "insomnia": "CBT-I is the first-line treatment for insomnia",
        "anxiety": "Grounding techniques: 5-4-3-2-1 sensory awareness",
    }

    for key, info in knowledge.items():
        if key in query.lower():
            return info

    return "No relevant information found"
```

### Use in Template

```python
template = """
User question: "How do I treat insomnia?"

Extract topic: [[topic]]

<checkpoint>

Relevant info:
[[@search_docs:context|query={{topic}},n=3]]

<checkpoint>

Context: {{context}}

Answer the question: [[answer]]
"""

result = chatter(template)
print(result['answer'])
```

## How Context Flows

### Segment 1: Extract
```python
Extract topic: [[topic]]
```
**Result**: `topic = "insomnia treatment"`

### Segment 2: Search (No LLM call!)
```python
<checkpoint>

[[@search_docs:context|query={{topic}},n=3]]
```

The function receives `query="insomnia treatment"` and returns docs.

**Result**: `context = "CBT-I is the first-line treatment..."`

### Segment 3: Generate
```python
<checkpoint>

Context: {{context}}

Answer: [[answer]]
```

The LLM sees the retrieved context and generates a grounded answer.

## Production Example with Vector DB

```python
import chromadb

db = chromadb.Client()
collection = db.get_or_create_collection("docs")

@Actions.register('search', on_error='return_empty')
def search(context, query: str, n: int = 5):
    """Vector similarity search"""
    results = collection.query(query_texts=[query], n_results=n)
    docs = results['documents'][0]
    return "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))

# Use it
template = """
Question: {{question}}

Extract key terms: [[terms]]

<checkpoint>

Retrieved docs:
[[@search:docs|query={{terms}},n=5]]

<checkpoint>

Based on: {{docs}}

Answer: [[answer]]
"""
```

## Why Use Checkpoints?

Saves tokens and cost:

**Without checkpoints**:
- LLM sees: [Long question (500 tokens]] + [Retrieved docs (1000 tokens]] = 1500 tokens

**With checkpoints**:
- Segment 1: Extract terms (500 tokens) → `[[terms]]`
- Segment 2: Function call (0 LLM tokens!) → retrieves docs
- Segment 3: Generate answer (1000 tokens only)

You send 500 fewer tokens to the LLM!

## Common Patterns

### Multiple Searches
```python
[[@search:context1|query={{topic1}}]]
[[@search:context2|query={{topic2}}]]
```

Both execute in the same segment, results available to next segment.

### Conditional Search
```python
{% if needs_context %}
[[@search:docs|query={{query}}]]
{% endif %}
```

## See Also

- **[Custom Actions Guide](CUSTOM_ACTIONS.md)** -- Complete Actions API
- **[Examples](../examples/)** -- Working RAG examples
