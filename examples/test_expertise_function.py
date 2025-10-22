#!/usr/bin/env python3
"""
Test the new [[@]] function call syntax for expertise retrieval.

This demonstrates the "extract then search" pattern where:
1. We extract information via LLM using [[var]]
2. We search for expertise using that extracted info via [[@expertise:var|...]]
"""

from struckdown import chatter

# Test 1: Basic expertise function call with literal query
print("=" * 60)
print("TEST 1: Basic expertise lookup with literal query")
print("=" * 60)

template1 = """
The client says: "I've been having trouble sleeping lately. I lie awake for hours."

[[@expertise:sleep_tips|query=insomnia,n=2]]

Here's what I found:
{{ sleep_tips }}
"""

result1 = chatter(template1)
print(f"\nResult:\n{result1.response}")
print(f"\nExtracted sleep_tips:\n{result1['sleep_tips']}")

# Test 2: Extract then search pattern
print("\n" + "=" * 60)
print("TEST 2: Extract then search pattern")
print("=" * 60)

template2 = """
The client says: "I drink too much and I know it's a problem, but I just can't seem to stop."

What is the main issue the client is facing? [[problem]]

[[@expertise:relevant_info|query={{problem}},n=2]]

Based on the problem being "{{ problem }}", here's relevant expertise:

{{ relevant_info }}

Now respond to the client. [[response]]
"""

result2 = chatter(template2)
print(f"\nExtracted problem: {result2['problem']}")
print(f"\nRelevant expertise:\n{result2['relevant_info']}")
print(f"\nFinal response: {result2.response}")

# Test 3: Multiple extractions with expertise lookup
print("\n" + "=" * 60)
print("TEST 3: Multiple extractions with expertise lookup")
print("=" * 60)

template3 = """
The client says: "I'm in the contemplation stage. I know I should change my drinking, but I'm not sure I'm ready yet."

What stage of change is the client in? [[stage]]

What is their concern? [[concern]]

[[@expertise:stage_info|query={{stage}},n=1]]
[[@expertise:concern_info|query={{concern}},n=1]]

Stage information:
{{ stage_info }}

Concern information:
{{ concern_info }}

Based on the above, what technique should the therapist use? [[technique]]
"""

result3 = chatter(template3)
print(f"\nStage: {result3['stage']}")
print(f"\nConcern: {result3['concern']}")
print(f"\nStage info: {result3['stage_info']}")
print(f"\nConcern info: {result3['concern_info']}")
print(f"\nRecommended technique: {result3['technique']}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
