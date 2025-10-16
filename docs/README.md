# Struckdown Examples

This directory contains comprehensive examples demonstrating all features of the Struckdown templating library.

## Running Examples

Test all examples:
```bash
python test_examples.py
```

## Example Files

### Basic Features

1. **`01_basic_completion.sd`** - Simple LLM completion
2. **`02_simple_chain.sd`** - Chaining with `¡OBLIVIATE`
3. **`03_template_variables.sd`** - Dynamic content with `{{variables}}`

### Advanced Features

4. **`04_shared_header.sd`** - Shared context with `¡BEGIN`
5. **`05_return_types.sd`** - Type-specific completions (`bool`, `int`, `json`)
6. **`06_list_completions.sd`** - List generation (`3*[[item]]`, `2:4*[[item]]`, `*[[item]]`)

### Complex Workflows

7. **`07_complex_workflow.sd`** - Multi-step data analysis pipeline
8. **`08_template_tags.sd`** - Dynamic content generation
9. **`09_edge_cases.sd`** - Error handling and special cases
10. **`10_creative_writing.sd`** - Story generation pipeline

### Special Examples

11. **`common_header.sd`** - Original shared header example

## Key Syntax Features Demonstrated

### Completions
- `[[variable]]` - Basic text completion
- `[[type:variable]]` - Typed completion (bool, int, json)
- `3*[[item]]` - Fixed count list
- `2:4*[[item]]` - Range count list
- `*[[item]]` - Unlimited list
- **Optional in final segment** - The final segment can omit `[[placeholder]]` and it will automatically become `[[response]]`. Example: `Tell me a joke` (no placeholder needed!)

### Control Flow
- `¡OBLIVIATE` - Memory boundary between sections
- `¡BEGIN` - Start of template after shared header

### Variables
- `{{variable}}` - Template variable substitution
- Context passed via `chatter(template, context={...})`

### Shared Context
```markdown
You are an expert assistant

¡BEGIN

First prompt: [[result1]]

¡OBLIVIATE

Second prompt using {{result1}}: [[result2]]
```

The shared header is prepended to every prompt section, ensuring consistent LLM behavior across memory boundaries.

## Test Results

All examples parse successfully and demonstrate:
- ✅ Grammar validation
- ✅ Variable extraction  
- ✅ Execution with sample data
- ✅ Multi-section workflows
- ✅ Shared header functionality