"""
Demo script showing how to use the new temporal response types in Struckdown.

This demonstrates extracting dates, times, datetimes, and durations from text
using the new [[date:var]], [[time:var]], [[datetime:var]], and [[duration:var]] syntax.
"""

from datetime import date, datetime, time, timedelta

from struckdown import chatter

# Example 1: Extract a date
print("=" * 60)
print("Example 1: Extract a date")
print("=" * 60)

result = chatter("The meeting is scheduled for next Tuesday [[date:meeting_date]]")
print(f"Extracted date: {result['meeting_date']}")
print(f"Type: {type(result['meeting_date'])}")
print()

# Example 2: Extract a time
print("=" * 60)
print("Example 2: Extract a time")
print("=" * 60)

result = chatter("The event starts at 3:30 PM [[time:event_time]]")
print(f"Extracted time: {result['event_time']}")
print(f"Type: {type(result['event_time'])}")
print()

# Example 3: Extract a datetime
print("=" * 60)
print("Example 3: Extract a datetime")
print("=" * 60)

result = chatter(
    "The conference begins on January 15, 2024 at 2:00 PM [[datetime:conference_start]]"
)
print(f"Extracted datetime: {result['conference_start']}")
print(f"Type: {type(result['conference_start'])}")
print()

# Example 4: Extract a duration
print("=" * 60)
print("Example 4: Extract a duration")
print("=" * 60)

result = chatter("The flight takes 2 hours and 30 minutes [[duration:flight_duration]]")
print(f"Extracted duration: {result['flight_duration']}")
print(f"Type: {type(result['flight_duration'])}")
print()

# Example 5: Multiple temporal extractions
print("=" * 60)
print("Example 5: Multiple temporal extractions")
print("=" * 60)

template = """The project meeting is scheduled for {{project_name}}.

Extract the date [[date:meeting_date]]

<checkpoint>

Extract the start time [[time:start_time]]

<checkpoint>

How long will the meeting last? [[duration:meeting_length]]"""

result = chatter(
    template,
    context={"project_name": "Project Alpha on March 15 at 10:00 AM for 90 minutes"},
)
print(f"Meeting date: {result['meeting_date']}")
print(f"Start time: {result['start_time']}")
print(f"Meeting length: {result['meeting_length']}")
print()

# Example 6: Relative dates with context
print("=" * 60)
print("Example 6: Relative dates (uses current date context)")
print("=" * 60)

result = chatter("The deadline is in 5 days [[date:deadline]]")
print(f"Deadline (relative): {result['deadline']}")
print(f"Note: This is calculated relative to today's date")
print()

# Example 7: Multiple dates with quantifiers
print("=" * 60)
print("Example 7: Extract multiple dates")
print("=" * 60)

result = chatter(
    "Important dates: January 1, February 15, and March 30 [[date*:important_dates]]"
)
print(f"Extracted dates: {result['important_dates']}")
print(f"Type: {type(result['important_dates'])}")
print()

print("=" * 60)
print("All examples completed successfully!")
print("=" * 60)
