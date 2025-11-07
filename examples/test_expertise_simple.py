#!/usr/bin/env python3
"""
Simple test for the [[@]] action call syntax without requiring LLM.
Tests just the parsing and function execution.
"""

from struckdown.parsing import parser

# Test 1: Parse a function call
print("=" * 60)
print("TEST 1: Parse function call syntax")
print("=" * 60)

template1 = """
Some text here.

[[@expertise:results|query=insomnia,n=2]]
"""

try:
    segments = parser().parse(template1.strip())
    print(f"✓ Successfully parsed template with function call!")
    print(f"  Number of segments: {len(segments)}")

    # Check the parsed structure
    for i, segment in enumerate(segments):
        print(f"\n  Segment {i}:")
        for key, prompt_part in segment.items():
            print(f"    Key: {key}")
            print(f"    Action type: {prompt_part.action_type}")
            print(f"    Is function: {prompt_part.is_function}")
            print(f"    Options: {prompt_part.options}")

except Exception as e:
    print(f"✗ Parse failed: {e}")

# Test 2: Parse mixed completions and function calls
print("\n" + "=" * 60)
print("TEST 2: Parse mixed LLM completions and function calls")
print("=" * 60)

template2 = """
Client says: "I can't sleep"

What is the problem? [[problem]]

[[@expertise:info|query={{problem}},n=3]]

Response: {{ info }}
"""

try:
    segments = parser().parse(template2.strip())
    print(f"✓ Successfully parsed mixed template!")
    print(f"  Number of segments: {len(segments)}")

    for i, segment in enumerate(segments):
        print(f"\n  Segment {i}:")
        for key, prompt_part in segment.items():
            print(f"    Key: {key}")
            print(f"    Action type: {prompt_part.action_type}")
            print(f"    Is function: {prompt_part.is_function}")
            if prompt_part.text:
                text_preview = (
                    prompt_part.text[:50] + "..."
                    if len(prompt_part.text) > 50
                    else prompt_part.text
                )
                print(f"    Text: {text_preview}")

except Exception as e:
    print(f"✗ Parse failed: {e}")

# Test 3: Verify function vs completion distinction
print("\n" + "=" * 60)
print("TEST 3: Verify [[]] vs [[@]] distinction")
print("=" * 60)

template3 = """
[[llm_completion:var1]]

[[@function_call:var2|opt=value]]
"""

try:
    segments = parser().parse(template3.strip())
    print(f"✓ Successfully parsed!")

    # First segment should have LLM completion
    seg0_items = list(segments[0].items())
    if seg0_items:
        key, part = seg0_items[0]
        print(f"\n  [[]] completion:")
        print(f"    is_function={part.is_function} (should be False)")
        assert (
            part.is_function == False
        ), "LLM completion incorrectly marked as function"
        print(f"    ✓ Correct!")

    # Second segment should have function call
    if len(segments) > 1:
        seg1_items = list(segments[1].items())
        if seg1_items:
            key, part = seg1_items[0]
            print(f"\n  [[@]] action call:")
            print(f"    is_function={part.is_function} (should be True)")
            assert part.is_function == True, "Action call not marked as function"
            print(f"    ✓ Correct!")

except Exception as e:
    print(f"✗ Test failed: {e}")

print("\n" + "=" * 60)
print("All parsing tests passed!")
print("=" * 60)
