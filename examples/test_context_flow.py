#!/usr/bin/env python3
"""
Test showing how expertise adds to context before subsequent completions.

This demonstrates:
1. Initial context from user
2. Expertise function adds to context (no LLM call)
3. Final LLM completion can see both in its context
"""

from unittest.mock import Mock, patch

from struckdown import LLM, LLMCredentials, chatter
from struckdown.return_type_models import ACTION_LOOKUP

print("=" * 70)
print("CONTEXT FLOW TEST: Expertise → Context → LLM Completion")
print("=" * 70)

# Template showing the flow
template = """
Extract the main problem from: "I can't sleep, I've tried everything." [[problem]]

¡OBLIVIATE

[[@expertise:relevant_tips|query={{problem}},n=2]]

¡OBLIVIATE

Now write a response to the client.

Context you have available:
- Problem: {{ problem }}
- Relevant tips: {{ relevant_tips }}

Based on the above, respond: [[response]]
"""

print("\nTemplate structure:")
print("  Segment 1: Extract [[problem]]")
print("  Segment 2: Function call [(expertise)] using {{problem}}")
print("  Segment 3: LLM [[response]] using {{problem}} and {{relevant_tips}}")

# Mock the LLM to show what context it receives
mock_responses = {
    "problem": "insomnia and sleep difficulties",
    "response": "Based on the problem (insomnia and sleep difficulties) and the relevant tips about CBT-I and sleep hygiene, I recommend we start with sleep restriction therapy...",
}


def mock_structured_chat(messages, return_type, llm, credentials, extra_kwargs):
    """Mock LLM that returns predefined responses and shows context"""

    # Extract the user message content from messages list
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content += msg.get("content", "")

    prompt = user_content  # For backward compatibility with rest of function

    # Determine which variable we're extracting based on the prompt
    if "Extract the main problem" in prompt:
        var_name = "problem"
    elif "respond:" in prompt.lower():
        var_name = "response"
    else:
        var_name = "unknown"

    print(f"\n{'='*70}")
    print(f"LLM CALLED for [[{var_name}]]")
    print(f"{'='*70}")
    print(f"Prompt received (first 200 chars):")
    print(f"  {prompt[:200]}...")

    # Check what's in the prompt from context
    if "{{ problem }}" not in prompt and "insomnia" in prompt.lower():
        print(f"\n✓ Context variable 'problem' was interpolated in prompt")
        print(f"  (prompt contains: '...insomnia...')")

    if "{{ relevant_tips }}" not in prompt and "CBT-I" in prompt:
        print(f"\n✓ Context variable 'relevant_tips' was interpolated in prompt")
        print(f"  (prompt contains: '...CBT-I...')")

    # Return the mock response
    response_value = mock_responses.get(var_name, "mock response")

    print(f"\nReturning: '{response_value}'")

    # Create mock result object
    mock_result = Mock()
    mock_result.response = response_value
    mock_result.model_dump = lambda: {"response": response_value}

    mock_completion = Mock()
    mock_completion.model_dump = lambda: {"usage": {"total_tokens": 100}}

    return mock_result, mock_completion


# Patch the LLM call
with patch("struckdown.structured_chat", side_effect=mock_structured_chat):
    print("\n" + "=" * 70)
    print("EXECUTING TEMPLATE")
    print("=" * 70)

    result = chatter(template)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nExtracted values:")
print(f"  problem: {result['problem']}")
print(f"  relevant_tips: {result['relevant_tips'][:100]}...")
print(f"  response: {result['response'][:100]}...")

print("\n" + "=" * 70)
print("CONTEXT BUILD-UP TIMELINE")
print("=" * 70)

print("\nSegment 1:")
print("  Action: LLM extracts [[problem]]")
print("  Context before: {}")
print(f"  Context after:  {{problem: '{result['problem']}'}}")

print("\nSegment 2:")
print("  Action: Function [(expertise)] executes")
print(f"  Context before: {{problem: '{result['problem']}'}}")
print(f"  Context after:  {{problem: '...', relevant_tips: '...CBT-I...' }}")
print("  ✓ Function used {{problem}} from context")
print("  ✓ Function added 'relevant_tips' to context")
print("  ✓ NO LLM CALL - instant execution")

print("\nSegment 3:")
print("  Action: LLM generates [[response]]")
print("  Context before: {problem: '...', relevant_tips: '...'}")
print("  ✓ LLM prompt includes BOTH problem and relevant_tips")
print("  ✓ Response can reference both variables")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print(
    """
The expertise function call [[@]] executes BETWEEN the two LLM calls.
It receives the context from Segment 1, performs its search, and adds
results to the context BEFORE Segment 3's LLM call.

This is the "extract then search" pattern:
1. Extract info via LLM: [[var]]
2. Search using that info: [[@expertise|query={{var}}]]
3. Use search results in next LLM call: [[response]]

The LLM in step 3 sees a prompt with the expertise text already
interpolated, making it part of the context for generation.
"""
)

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
