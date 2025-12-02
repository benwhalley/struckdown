Refactor the struckdown parser module to eliminate all regex-based parsing where the Lark grammar already provides full coverage.

Goals:
1. Remove the following regex-driven behaviours:
   - SLOT_PATTERN and all code built on it (extract_slot_key, find_slots_with_positions).
   - Regex-based detection of `[[...]]` completions.
   - Regex-based checkpoint detection in _add_default_completion_if_needed.
   - Regex-based logic in split_by_checkpoint.

2. Replace them with grammar-driven parsing using the existing Lark grammar:
   - single_completion and completion_body rules already identify completion blocks.
   - checkpoint tags (<checkpoint>, <obliviate>) are already lexed and parsed.
   - Use parse tree traversal instead of regex.

3. The following functions should be rewritten to depend entirely on Lark:
   - extract_slot_key(inner_text)
   - find_slots_with_positions(text)  → rename or remove if unused
   - split_by_checkpoint(template_text)
   - _add_default_completion_if_needed(template)

4. Remove `_slot_body_grammar` and its parser entirely unless strictly needed.
   If a fast key extractor is still desired, reuse the main grammar:
       - parse `[[body]]` using full grammar
       - locate completion_body node
       - extract the variable key

5. New approach for default completion injection:
   - After full parsing, inspect the last NamedSegment.
   - If last segment contains no PromptParts, append a synthetic [[response]] completion.
   - Do not use regex to inspect the raw template.

6. New approach for checkpoint segmentation:
   - Use the full grammar transformer's produced `sections` list.
   - Checkpoint tags already trigger _flush_checkpoint.
   - Remove all splitting logic based on regex or manual string slicing.
   - split_by_checkpoint should either:
       (a) be removed entirely, or
       (b) delegate to MindframeTransformer output.

7. For any function that previously returned start/end positions:
   - Replace with Lark token metadata (line, column) via propagate_positions=True.
   - Only keep positional metadata if actually used; otherwise remove positional return values.

8. Ensure all behaviour remains equivalent at the semantic level:
   - LLM options handling, action_call_* and typed_completion_* must remain unchanged.
   - All error messages for ambiguous typed completions must stay identical.
   - No observable change in PromptPart construction.

9. Do not modify mindframe_grammar.lark or the MindframeTransformer API unless necessary.
   Prefer transforming parse-tree consumers rather than grammar producers.

10. Output should be a clean, working refactor that:
    - Removes regex duplication
    - Uses the grammar for all structural parsing
    - Reduces complexity while preserving exact semantics

Return the full revised code for the affected functions and any necessary adjustments to imports or parser initialisation.
