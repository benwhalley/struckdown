"""
Comprehensive test cases for temporal extraction functionality.

This file contains diverse examples of date, datetime, time, and duration extractions
with various syntaxes (single values, lists, patterns, quantifiers).

Run with: python examples/temporal_test_cases.py
"""

from datetime import date, datetime, time, timedelta
from struckdown import chatter, LLM, LLMCredentials

# Test case format: (description, prompt, expected_type, validation_function)
TEST_CASES = [
    # ===== SINGLE DATE EXTRACTIONS =====
    {
        "category": "Single Dates - Explicit",
        "description": "Extract explicit date (Jan 15, 2024)",
        "prompt": "The meeting is on January 15, 2024 [[date:d]]",
        "expected_type": date,
        "validate": lambda r: r["d"] == date(2024, 1, 15),
    },
    {
        "category": "Single Dates - Explicit",
        "description": "Extract ISO format date",
        "prompt": "Launch date: 2025-03-20 [[date:d]]",
        "expected_type": date,
        "validate": lambda r: r["d"] == date(2025, 3, 20),
    },
    {
        "category": "Single Dates - Relative",
        "description": "Extract 'tomorrow'",
        "prompt": "See you tomorrow [[date:d]]",
        "expected_type": date,
        "validate": lambda r: isinstance(r["d"], date) and r["d"] > date.today(),
    },
    {
        "category": "Single Dates - Relative",
        "description": "Extract 'next Monday'",
        "prompt": "The deadline is next Monday [[date:d]]",
        "expected_type": date,
        "validate": lambda r: isinstance(r["d"], date) and r["d"].weekday() == 0,  # Monday
    },

    # ===== RECURRING PATTERNS - SINGLE VALUE =====
    {
        "category": "Patterns - Single (with warning)",
        "description": "Pattern 'first tuesday in november' (single) - should warn about truncation",
        "prompt": "first tuesday in november [[date:d]]",
        "expected_type": date,
        "validate": lambda r: isinstance(r["d"], date) and r["d"].month == 11 and r["d"].weekday() == 1,
    },
    {
        "category": "Patterns - Single (with warning)",
        "description": "Pattern 'last friday in december' (single)",
        "prompt": "last friday in december [[date:d]]",
        "expected_type": date,
        "validate": lambda r: isinstance(r["d"], date) and r["d"].month == 12 and r["d"].weekday() == 4,
    },

    # ===== RECURRING PATTERNS - LISTS =====
    {
        "category": "Patterns - Lists",
        "description": "Pattern 'first 2 tuesdays in september' (list)",
        "prompt": "first 2 tuesdays in september [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) == 2
            and all(d.month == 9 and d.weekday() == 1 for d in r["dates"])
        ),
    },
    {
        "category": "Patterns - Lists",
        "description": "Pattern 'first 3 mondays in october' (list)",
        "prompt": "first 3 mondays in october [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) == 3
            and all(d.month == 10 and d.weekday() == 0 for d in r["dates"])
        ),
    },
    {
        "category": "Patterns - Lists",
        "description": "Pattern 'first 4 fridays in november' (list)",
        "prompt": "first 4 fridays in november 2025 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) == 4
            and all(d.month == 11 and d.weekday() == 4 for d in r["dates"])
        ),
    },
    {
        "category": "Patterns - Lists",
        "description": "Pattern 'every wednesday in september 2025' (list)",
        "prompt": "every wednesday in september 2025 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) >= 4  # At least 4 Wednesdays in September
            and all(d.year == 2025 and d.month == 9 and d.weekday() == 2 for d in r["dates"])
        ),
    },

    # ===== MULTIPLE EXPLICIT DATES (LISTS) =====
    {
        "category": "Lists - Explicit Dates",
        "description": "Extract list of explicit dates",
        "prompt": "Important dates: Jan 15, Feb 20, and March 10 2024 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) == 3
            and r["dates"][0].month == 1
            and r["dates"][1].month == 2
            and r["dates"][2].month == 3
        ),
    },
    {
        "category": "Lists - Explicit Dates",
        "description": "Extract list with mixed formats",
        "prompt": "Dates: 2024-01-15, January 20th, and Feb 1 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list) and len(r["dates"]) == 3,
    },

    # ===== QUANTIFIERS =====
    {
        "category": "Quantifiers",
        "description": "Exactly 2 dates required [[date{2}:dates]]",
        "prompt": "First and last day: Jan 1 and Dec 31, 2024 [[date{2}:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list) and len(r["dates"]) == 2,
    },
    {
        "category": "Quantifiers",
        "description": "1-3 dates [[date{1,3}:dates]]",
        "prompt": "Q1 months: Jan, Feb, March [[date{1,3}:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list) and 1 <= len(r["dates"]) <= 3,
    },
    {
        "category": "Quantifiers",
        "description": "At least 1 date [[date+:dates]]",
        "prompt": "Holidays: Christmas and New Year [[date+:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list) and len(r["dates"]) >= 1,
    },
    {
        "category": "Quantifiers",
        "description": "Zero or more dates [[date*:dates]]",
        "prompt": "No dates mentioned here [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list),  # Can be empty
    },

    # ===== DATETIME EXTRACTIONS =====
    {
        "category": "DateTimes - Explicit",
        "description": "Extract datetime with time",
        "prompt": "Meeting at January 15, 2024 at 3:30 PM [[datetime:dt]]",
        "expected_type": datetime,
        "validate": lambda r: isinstance(r["dt"], datetime) and r["dt"].hour == 15 and r["dt"].minute == 30,
    },
    {
        "category": "DateTimes - Explicit",
        "description": "Extract ISO datetime",
        "prompt": "Scheduled: 2024-12-25T18:00:00 [[datetime:dt]]",
        "expected_type": datetime,
        "validate": lambda r: isinstance(r["dt"], datetime) and r["dt"].hour == 18,
    },
    {
        "category": "DateTimes - Relative",
        "description": "Extract 'tomorrow at 2pm'",
        "prompt": "Call me tomorrow at 2pm [[datetime:dt]]",
        "expected_type": datetime,
        "validate": lambda r: isinstance(r["dt"], datetime) and r["dt"].hour == 14,
    },
    {
        "category": "DateTimes - Lists",
        "description": "Extract list of datetimes",
        "prompt": "Meetings: Monday 9am, Tuesday 2pm, Wednesday 11am [[datetime*:dts]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dts"], list) and len(r["dts"]) >= 2,
    },

    # ===== TIME EXTRACTIONS =====
    {
        "category": "Times - Explicit",
        "description": "Extract time '3:30 PM'",
        "prompt": "The show starts at 3:30 PM [[time:t]]",
        "expected_type": time,
        "validate": lambda r: isinstance(r["t"], time) and r["t"].hour == 15 and r["t"].minute == 30,
    },
    {
        "category": "Times - Explicit",
        "description": "Extract 24-hour time",
        "prompt": "Train departs at 18:45 [[time:t]]",
        "expected_type": time,
        "validate": lambda r: isinstance(r["t"], time) and r["t"].hour == 18 and r["t"].minute == 45,
    },
    {
        "category": "Times - Lists",
        "description": "Extract multiple times",
        "prompt": "Available times: 9am, 11am, 2pm, and 4pm [[time*:times]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["times"], list) and len(r["times"]) >= 3,
    },
    {
        "category": "Times - Ambiguous",
        "description": "Extract time without AM/PM (context)",
        "prompt": "Lunch meeting at 12:30 [[time:t]]",
        "expected_type": time,
        "validate": lambda r: isinstance(r["t"], time) and r["t"].hour == 12,
    },

    # ===== DURATION EXTRACTIONS =====
    {
        "category": "Durations - Simple",
        "description": "Extract duration '2 hours'",
        "prompt": "The movie lasts 2 hours [[duration:d]]",
        "expected_type": timedelta,
        "validate": lambda r: isinstance(r["d"], (timedelta, str)) and (
            r["d"] == timedelta(hours=2) if isinstance(r["d"], timedelta) else "2 hour" in r["d"].lower()
        ),
    },
    {
        "category": "Durations - Simple",
        "description": "Extract duration '30 minutes'",
        "prompt": "Meeting duration: 30 minutes [[duration:d]]",
        "expected_type": timedelta,
        "validate": lambda r: isinstance(r["d"], (timedelta, str)) and (
            r["d"] == timedelta(minutes=30) if isinstance(r["d"], timedelta) else "30 minute" in r["d"].lower()
        ),
    },
    {
        "category": "Durations - Complex",
        "description": "Extract duration '1 week and 3 days'",
        "prompt": "Project timeline: 1 week and 3 days [[duration:d]]",
        "expected_type": timedelta,
        "validate": lambda r: isinstance(r["d"], (timedelta, str)) and (
            r["d"] == timedelta(weeks=1, days=3) if isinstance(r["d"], timedelta) else ("week" in r["d"].lower() and "day" in r["d"].lower())
        ),
    },
    {
        "category": "Durations - Complex",
        "description": "Extract duration '2 hours 45 minutes'",
        "prompt": "Flight time: 2 hours 45 minutes [[duration:d]]",
        "expected_type": timedelta,
        "validate": lambda r: isinstance(r["d"], (timedelta, str)) and (
            r["d"] == timedelta(hours=2, minutes=45) if isinstance(r["d"], timedelta) else ("2 hour" in r["d"].lower() or "45 minute" in r["d"].lower())
        ),
    },
    {
        "category": "Durations - Lists",
        "description": "Extract multiple durations",
        "prompt": "Task durations: 30 minutes, 1 hour, 2.5 hours [[duration*:durations]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["durations"], list) and len(r["durations"]) >= 2,
    },

    # ===== MIXED TEMPORAL TYPES =====
    {
        "category": "Mixed Types",
        "description": "Extract date and duration from same text",
        "prompt": "Project starts March 20, 2024 [[date:start]] and will take 3 weeks [[duration:length]]",
        "expected_type": "mixed",
        "validate": lambda r: (
            isinstance(r["start"], date)
            and r["start"].month == 3
            and isinstance(r["length"], (timedelta, str))
        ),
    },
    {
        "category": "Mixed Types",
        "description": "Extract datetime and duration (simple)",
        "prompt": "Meeting at 2pm on Jan 15 [[datetime:start]] for 2 hours [[duration:length]]",
        "expected_type": "mixed",
        "validate": lambda r: (
            isinstance(r["start"], datetime)
            and isinstance(r["length"], (timedelta, str))
        ),
    },

    # ===== EDGE CASES =====
    {
        "category": "Edge Cases",
        "description": "No date in text (should return None)",
        "prompt": "There is no date information here [[date:d]]",
        "expected_type": type(None),
        "validate": lambda r: r["d"] is None,
    },
    {
        "category": "Edge Cases",
        "description": "No dates in list (should return empty list)",
        "prompt": "No temporal information present [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list) and len(r["dates"]) == 0,
    },
    {
        "category": "Edge Cases",
        "description": "Ambiguous month reference defaults to current/next year",
        "prompt": "See you in December [[date:d]]",
        "expected_type": date,
        "validate": lambda r: isinstance(r["d"], date) and r["d"].month == 12,
    },

    # ===== YEAR INFERENCE =====
    {
        "category": "Year Inference",
        "description": "Month without year (should use current year context)",
        "prompt": "first monday in november [[date:d]]",
        "expected_type": date,
        "validate": lambda r: isinstance(r["d"], date) and r["d"].month == 11 and r["d"].year >= 2025,
    },
    {
        "category": "Year Inference",
        "description": "Pattern with explicit year",
        "prompt": "first 2 tuesdays in september 2026 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) == 2
            and all(d.year == 2026 for d in r["dates"])
        ),
    },

    # ===== COMPLEX PATTERNS =====
    {
        "category": "Complex Patterns",
        "description": "Every other week pattern",
        "prompt": "every other monday for 4 weeks starting oct 1 2025 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["dates"], list) and len(r["dates"]) >= 2,
    },
    {
        "category": "Complex Patterns",
        "description": "All weekdays in a month",
        "prompt": "all mondays in september 2025 [[date*:dates]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["dates"], list)
            and len(r["dates"]) >= 4  # September has 4-5 Mondays
            and all(d.weekday() == 0 for d in r["dates"])
        ),
    },
]


def run_tests(verbose=False, stop_on_error=False):
    """
    Run all test cases and report results.

    Args:
        verbose: If True, print detailed output for each test
        stop_on_error: If True, stop on first failure
    """
    model = LLM()
    credentials = LLMCredentials()

    results = {
        "passed": [],
        "failed": [],
        "errors": [],
    }

    # Group tests by category
    by_category = {}
    for test in TEST_CASES:
        category = test["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(test)

    print("=" * 80)
    print("TEMPORAL EXTRACTION TEST SUITE")
    print("=" * 80)
    print(f"\nTotal test cases: {len(TEST_CASES)}")
    print(f"Categories: {len(by_category)}\n")

    for category, tests in by_category.items():
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 80}")

        for i, test in enumerate(tests, 1):
            desc = test["description"]
            prompt = test["prompt"]
            validate = test["validate"]

            if verbose:
                print(f"\n[{i}/{len(tests)}] {desc}")
                print(f"  Prompt: {prompt}")
            else:
                print(f"\n[{i}/{len(tests)}] {desc}...", end=" ")

            try:
                # Run the extraction
                result = chatter(prompt, model=model, credentials=credentials)

                # Validate the result
                if validate(result):
                    results["passed"].append(test)
                    if verbose:
                        print(f"  ✓ PASSED")
                        print(f"  Result: {result}")
                    else:
                        print("✓ PASSED")
                else:
                    results["failed"].append((test, result, "Validation failed"))
                    if verbose:
                        print(f"  ✗ FAILED - Validation failed")
                        print(f"  Result: {result}")
                    else:
                        print("✗ FAILED")

                    if stop_on_error:
                        print("\nStopping on first failure.")
                        break

            except Exception as e:
                results["errors"].append((test, str(e)))
                if verbose:
                    print(f"  ✗ ERROR: {e}")
                else:
                    print(f"✗ ERROR: {e}")

                if stop_on_error:
                    print("\nStopping on first error.")
                    break

        if stop_on_error and (results["failed"] or results["errors"]):
            break

    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n✓ Passed: {len(results['passed'])}/{len(TEST_CASES)}")
    print(f"✗ Failed: {len(results['failed'])}/{len(TEST_CASES)}")
    print(f"⚠ Errors: {len(results['errors'])}/{len(TEST_CASES)}")

    if results["failed"]:
        print(f"\n{'=' * 80}")
        print("FAILED TESTS")
        print(f"{'=' * 80}")
        for test, result, reason in results["failed"]:
            print(f"\n✗ {test['description']}")
            print(f"  Prompt: {test['prompt']}")
            print(f"  Reason: {reason}")
            print(f"  Result: {result}")

    if results["errors"]:
        print(f"\n{'=' * 80}")
        print("ERRORS")
        print(f"{'=' * 80}")
        for test, error in results["errors"]:
            print(f"\n⚠ {test['description']}")
            print(f"  Prompt: {test['prompt']}")
            print(f"  Error: {error}")

    success_rate = len(results["passed"]) / len(TEST_CASES) * 100
    print(f"\n{'=' * 80}")
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    stop_on_error = "--stop-on-error" in sys.argv or "-x" in sys.argv

    print("\nUsage: python examples/temporal_test_cases.py [--verbose|-v] [--stop-on-error|-x]\n")

    results = run_tests(verbose=verbose, stop_on_error=stop_on_error)

    # Exit with error code if any tests failed
    if results["failed"] or results["errors"]:
        sys.exit(1)
    else:
        sys.exit(0)
