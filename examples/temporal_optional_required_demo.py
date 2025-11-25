"""
Demo script showing optional vs required temporal fields in Struckdown.

By default, temporal fields are optional and return None if no valid
date/time/duration can be extracted. Use |required to make them mandatory.
"""

from struckdown import chatter

# Example 1: Optional date (default) - returns None when no date found
print("=" * 60)
print("Example 1: Optional date (returns None when no date found)")
print("=" * 60)

result = chatter("This text has no dates in it [[date:event_date]]")
print(f"Extracted date: {result['event_date']}")
print(f"Type: {type(result['event_date'])}")
print("Note: Returns None because no date was found")
print()

# Example 2: Optional date with actual date - extracts successfully
print("=" * 60)
print("Example 2: Optional date with actual date")
print("=" * 60)

result = chatter("The meeting is on January 15, 2024 [[date:event_date]]")
print(f"Extracted date: {result['event_date']}")
print(f"Type: {type(result['event_date'])}")
print()

# Example 3: Required date - will raise error if no date found
print("=" * 60)
print("Example 3: Required date (will error if no date found)")
print("=" * 60)

try:
    result = chatter("This text has no dates [[date:event_date|required]]")
    print(f"Extracted date: {result['event_date']}")
except Exception as e:
    print(f"Error (as expected): {type(e).__name__}")
    print("The LLM will be forced to extract a date or fail validation")
print()

# Example 4: Required date with actual date - works fine
print("=" * 60)
print("Example 4: Required date with actual date")
print("=" * 60)

result = chatter("The deadline is March 30, 2024 [[date:deadline|required]]")
print(f"Extracted date: {result['deadline']}")
print(f"Type: {type(result['deadline'])}")
print()

# Example 5: Optional datetime (default)
print("=" * 60)
print("Example 5: Optional datetime (returns None when not found)")
print("=" * 60)

result = chatter("Some random text without any datetime [[datetime:event_time]]")
print(f"Extracted datetime: {result['event_time']}")
print()

# Example 6: Optional time (default)
print("=" * 60)
print("Example 6: Optional time (returns None when not found)")
print("=" * 60)

result = chatter("Text without time information [[time:meeting_time]]")
print(f"Extracted time: {result['meeting_time']}")
print()

# Example 7: Optional duration (default)
print("=" * 60)
print("Example 7: Optional duration (returns None when not found)")
print("=" * 60)

result = chatter("No duration mentioned here [[duration:length]]")
print(f"Extracted duration: {result['length']}")
print()

# Example 8: All temporal types work with |required
print("=" * 60)
print("Example 8: All temporal types support |required")
print("=" * 60)

template = """The conference is on January 15, 2024 at 9:00 AM and lasts 2 hours.

Extract date [[date:conf_date|required]]

<checkpoint>

Extract time [[time:conf_time|required]]

<checkpoint>

Extract duration [[duration:conf_length|required]]"""

result = chatter(template)
print(f"Conference date: {result['conf_date']}")
print(f"Conference time: {result['conf_time']}")
print(f"Conference length: {result['conf_length']}")
print()

print("=" * 60)
print("Summary:")
print("=" * 60)
print("- By default, temporal fields are OPTIONAL (return None if not found)")
print("- Use |required to make them MANDATORY (will error if not found)")
print("- This applies to all temporal types: date, datetime, time, duration")
print("=" * 60)
