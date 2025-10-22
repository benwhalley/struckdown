"""
Comprehensive test cases for number extraction functionality.

This file tests the flexible `number` type that accepts both integers and floats
with optional min/max validation.

Run with: uv run python examples/number_test_cases.py
"""

from struckdown import chatter, LLM, LLMCredentials

# Test case format: (description, prompt, expected_type, validation_function)
TEST_CASES = [
    # ===== BASIC NUMBER EXTRACTION =====
    {
        "category": "Basic Extraction - Integers",
        "description": "Extract simple integer",
        "prompt": "The answer is 42 [[number:value]]",
        "expected_type": int,
        "validate": lambda r: r["value"] == 42,
    },
    {
        "category": "Basic Extraction - Integers",
        "description": "Extract negative integer",
        "prompt": "Temperature: -15 degrees [[number:temp]]",
        "expected_type": int,
        "validate": lambda r: r["temp"] == -15,
    },
    {
        "category": "Basic Extraction - Floats",
        "description": "Extract simple float",
        "prompt": "The price is $19.99 [[number:price]]",
        "expected_type": float,
        "validate": lambda r: abs(r["price"] - 19.99) < 0.01,
    },
    {
        "category": "Basic Extraction - Floats",
        "description": "Extract negative float",
        "prompt": "Balance: -127.50 [[number:balance]]",
        "expected_type": float,
        "validate": lambda r: abs(r["balance"] - (-127.50)) < 0.01,
    },
    {
        "category": "Basic Extraction - Floats",
        "description": "Extract scientific notation",
        "prompt": "Value: 1.5e-3 [[number:val]]",
        "expected_type": float,
        "validate": lambda r: isinstance(r["val"], (int, float)) and r["val"] > 0,
    },

    # ===== MIN CONSTRAINT =====
    {
        "category": "Min Constraint",
        "description": "Extract with min=0 (valid positive)",
        "prompt": "Score: 85 [[number:score|min=0]]",
        "expected_type": int,
        "validate": lambda r: r["score"] == 85 and r["score"] >= 0,
    },
    {
        "category": "Min Constraint",
        "description": "Extract with min=0 (exactly at boundary)",
        "prompt": "Value: 0 [[number:val|min=0]]",
        "expected_type": int,
        "validate": lambda r: r["val"] == 0,
    },
    {
        "category": "Min Constraint",
        "description": "Extract with min=-10.5 (float boundary)",
        "prompt": "Temperature: -5.5 degrees [[number:temp|min=-10.5]]",
        "expected_type": float,
        "validate": lambda r: abs(r["temp"] - (-5.5)) < 0.01 and r["temp"] >= -10.5,
    },

    # ===== MAX CONSTRAINT =====
    {
        "category": "Max Constraint",
        "description": "Extract with max=100 (valid)",
        "prompt": "Percentage: 75 [[number:percent|max=100]]",
        "expected_type": int,
        "validate": lambda r: r["percent"] == 75 and r["percent"] <= 100,
    },
    {
        "category": "Max Constraint",
        "description": "Extract with max=100 (exactly at boundary)",
        "prompt": "Completion: 100% [[number:completion|max=100]]",
        "expected_type": int,
        "validate": lambda r: r["completion"] == 100,
    },
    {
        "category": "Max Constraint",
        "description": "Extract with max=5.0 (float boundary)",
        "prompt": "Rating: 4.8 out of 5 [[number:rating|max=5.0]]",
        "expected_type": float,
        "validate": lambda r: abs(r["rating"] - 4.8) < 0.01 and r["rating"] <= 5.0,
    },

    # ===== MIN/MAX COMBINED =====
    {
        "category": "Min/Max Combined",
        "description": "Extract with min=0,max=100 (mid-range)",
        "prompt": "Score: 67 out of 100 [[number:score|min=0,max=100]]",
        "expected_type": int,
        "validate": lambda r: r["score"] == 67 and 0 <= r["score"] <= 100,
    },
    {
        "category": "Min/Max Combined",
        "description": "Extract with min=0,max=100 (at min boundary)",
        "prompt": "Score: 0 [[number:score|min=0,max=100]]",
        "expected_type": int,
        "validate": lambda r: r["score"] == 0,
    },
    {
        "category": "Min/Max Combined",
        "description": "Extract with min=0,max=100 (at max boundary)",
        "prompt": "Score: 100 [[number:score|min=0,max=100]]",
        "expected_type": int,
        "validate": lambda r: r["score"] == 100,
    },
    {
        "category": "Min/Max Combined",
        "description": "Extract with min=-273.15,max=100.0 (float range)",
        "prompt": "Temperature: 37.5 degrees Celsius [[number:temp|min=-273.15,max=100.0]]",
        "expected_type": float,
        "validate": lambda r: abs(r["temp"] - 37.5) < 0.01 and -273.15 <= r["temp"] <= 100.0,
    },

    # ===== LIST EXTRACTION =====
    {
        "category": "Lists - Basic",
        "description": "Extract list of integers",
        "prompt": "Scores: 10, 20, 30 [[number*:scores]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["scores"], list)
            and len(r["scores"]) == 3
            and r["scores"] == [10, 20, 30]
        ),
    },
    {
        "category": "Lists - Basic",
        "description": "Extract list of floats",
        "prompt": "Measurements: 1.5, 2.3, 4.7 meters [[number*:measurements]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["measurements"], list)
            and len(r["measurements"]) == 3
            and all(isinstance(x, (int, float)) for x in r["measurements"])
        ),
    },
    {
        "category": "Lists - Basic",
        "description": "Extract mixed int/float list",
        "prompt": "Values: 10, 20.5, 30, 40.75 [[number*:values]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["values"], list)
            and len(r["values"]) == 4
        ),
    },

    # ===== QUANTIFIERS =====
    {
        "category": "Quantifiers",
        "description": "Exactly 2 numbers [[number{2}:values]]",
        "prompt": "Dimensions: 10.5 x 20.75 [[number{2}:values]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["values"], list) and len(r["values"]) == 2,
    },
    {
        "category": "Quantifiers",
        "description": "1-3 numbers [[number{1,3}:values]]",
        "prompt": "Ratings: 4.5, 3.8 [[number{1,3}:values]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["values"], list) and 1 <= len(r["values"]) <= 3,
    },
    {
        "category": "Quantifiers",
        "description": "At least 1 number [[number+:values]]",
        "prompt": "Scores: 85, 90, 95 [[number+:values]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["values"], list) and len(r["values"]) >= 1,
    },
    {
        "category": "Quantifiers",
        "description": "Zero or more numbers [[number*:values]]",
        "prompt": "No numbers here [[number*:values]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["values"], list),  # Can be empty
    },

    # ===== LIST WITH CONSTRAINTS =====
    {
        "category": "Lists with Constraints",
        "description": "List with min=0",
        "prompt": "Scores: 10, 25, 50 [[number*:scores|min=0]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["scores"], list)
            and all(x >= 0 for x in r["scores"])
        ),
    },
    {
        "category": "Lists with Constraints",
        "description": "List with max=100",
        "prompt": "Percentages: 25, 50, 75 [[number*:percentages|max=100]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["percentages"], list)
            and all(x <= 100 for x in r["percentages"])
        ),
    },
    {
        "category": "Lists with Constraints",
        "description": "List with min/max range",
        "prompt": "Ratings: 3.5, 4.0, 4.5 out of 5 [[number*:ratings|min=0,max=5]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["ratings"], list)
            and all(0 <= x <= 5 for x in r["ratings"])
        ),
    },

    # ===== QUANTIFIER + CONSTRAINTS =====
    {
        "category": "Quantifier + Constraints",
        "description": "2-5 numbers with min/max [[number{2,5}:scores|min=0,max=100]]",
        "prompt": "Test scores: 85, 92, 78, 95 [[number{2,5}:scores|min=0,max=100]]",
        "expected_type": list,
        "validate": lambda r: (
            isinstance(r["scores"], list)
            and 2 <= len(r["scores"]) <= 5
            and all(0 <= x <= 100 for x in r["scores"])
        ),
    },

    # ===== EDGE CASES =====
    {
        "category": "Edge Cases",
        "description": "No number in text (should return None)",
        "prompt": "There are no numbers here [[number:value]]",
        "expected_type": type(None),
        "validate": lambda r: r["value"] is None,
    },
    {
        "category": "Edge Cases",
        "description": "No numbers in list (should return empty list)",
        "prompt": "No numeric data available [[number*:values]]",
        "expected_type": list,
        "validate": lambda r: isinstance(r["values"], list) and len(r["values"]) == 0,
    },
    {
        "category": "Edge Cases",
        "description": "Zero as valid number",
        "prompt": "Count: 0 items [[number:count]]",
        "expected_type": int,
        "validate": lambda r: r["count"] == 0,
    },
    {
        "category": "Edge Cases",
        "description": "Very large number",
        "prompt": "Population: 7800000000 [[number:population]]",
        "expected_type": int,
        "validate": lambda r: r["population"] > 1000000000,
    },
    {
        "category": "Edge Cases",
        "description": "Very small decimal",
        "prompt": "Precision: 0.0001 [[number:precision]]",
        "expected_type": float,
        "validate": lambda r: 0 < r["precision"] < 0.001,
    },

    # ===== PRACTICAL USE CASES =====
    {
        "category": "Practical - Prices",
        "description": "Extract currency amount",
        "prompt": "Total: $1,234.56 [[number:total]]",
        "expected_type": float,
        "validate": lambda r: abs(r["total"] - 1234.56) < 0.01,
    },
    {
        "category": "Practical - Measurements",
        "description": "Extract dimension with unit",
        "prompt": "Length: 15.5 inches [[number:length]]",
        "expected_type": float,
        "validate": lambda r: abs(r["length"] - 15.5) < 0.01,
    },
    {
        "category": "Practical - Percentages",
        "description": "Extract percentage value",
        "prompt": "Progress: 67.5% complete [[number:progress|min=0,max=100]]",
        "expected_type": float,
        "validate": lambda r: abs(r["progress"] - 67.5) < 0.01,
    },
    {
        "category": "Practical - Ratings",
        "description": "Extract star rating",
        "prompt": "Rating: 4.7 out of 5 stars [[number:rating|min=0,max=5]]",
        "expected_type": float,
        "validate": lambda r: abs(r["rating"] - 4.7) < 0.01 and 0 <= r["rating"] <= 5,
    },
    {
        "category": "Practical - Temperatures",
        "description": "Extract temperature (can be negative)",
        "prompt": "Current temperature: -12.3°C [[number:temp]]",
        "expected_type": float,
        "validate": lambda r: abs(r["temp"] - (-12.3)) < 0.01,
    },
    # ===== VALIDATION ERRORS (with required flag) =====
    {
        "category": "Validation Errors - Required",
        "description": "Exceed max constraint with required (should raise ValueError)",
        "prompt": "Give me a number greater than 10 [[number:mynum|max=10,required]]",
        "expected_type": ValueError,
        "validate": lambda r: True,  # Should not reach here
    },
    {
        "category": "Validation Errors - Required",
        "description": "Below min constraint with required (should raise ValueError)",
        "prompt": "Give me a negative number [[number:mynum|min=0,required]]",
        "expected_type": ValueError,
        "validate": lambda r: True,  # Should not reach here
    },
    {
        "category": "Validation Errors - Required",
        "description": "List with value exceeding max (should raise ValueError)",
        "prompt": "Give me numbers: 50, 150, 75 [[number*:values|max=100,required]]",
        "expected_type": ValueError,
        "validate": lambda r: True,  # Should not reach here
    },
    {
        "category": "Validation Errors - Required",
        "description": "List with value below min (should raise ValueError)",
        "prompt": "Give me numbers: 10, -5, 20 [[number*:values|min=0,required]]",
        "expected_type": ValueError,
        "validate": lambda r: True,  # Should not reach here
    },
    # ===== LENIENT BEHAVIOR (without required flag) =====
    {
        "category": "Lenient Behavior",
        "description": "Out of range without required flag (should return None)",
        "prompt": "Give me a number greater than 10 [[number:mynum|max=10]]",
        "expected_type": type(None),
        "validate": lambda r: r["mynum"] is None,
    },
    {
        "category": "Lenient Behavior",
        "description": "Negative without required flag (should return None)",
        "prompt": "Give me a negative number [[number:mynum|min=0]]",
        "expected_type": type(None),
        "validate": lambda r: r["mynum"] is None,
    },
    {
        "category": "Lenient Behavior",
        "description": "No number present without required (should return None)",
        "prompt": "There are no numbers here at all [[number:mynum|min=0,max=100]]",
        "expected_type": type(None),
        "validate": lambda r: r["mynum"] is None,
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
    print("NUMBER EXTRACTION TEST SUITE")
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
                # Check if this test expects an exception
                expects_exception = test["expected_type"] == ValueError

                if expects_exception:
                    # For validation error tests, we expect an exception
                    try:
                        result = chatter(prompt, model=model, credentials=credentials)
                        # If we got here, the validation didn't raise an error (TEST FAILED)
                        results["failed"].append((test, result, "Expected ValueError but got successful result"))
                        if verbose:
                            print(f"  ✗ FAILED - Expected ValueError but got result")
                            print(f"  Result: {result}")
                        else:
                            print("✗ FAILED (no error raised)")

                        if stop_on_error:
                            print("\nStopping on first failure.")
                            break
                    except (ValueError, ExceptionGroup) as e:
                        # Check if it's a ValueError or an ExceptionGroup containing ValueError
                        is_validation_error = False
                        error_msg = ""

                        if isinstance(e, ValueError):
                            is_validation_error = True
                            error_msg = str(e)
                        elif isinstance(e, ExceptionGroup):
                            # Check if any sub-exception is a ValueError
                            for exc in e.exceptions:
                                if isinstance(exc, ValueError):
                                    is_validation_error = True
                                    error_msg = str(exc)
                                    break

                        if is_validation_error:
                            # Expected exception was raised (TEST PASSED)
                            results["passed"].append(test)
                            if verbose:
                                print(f"  ✓ PASSED (ValueError raised as expected)")
                                print(f"  Error: {error_msg}")
                            else:
                                print("✓ PASSED")
                        else:
                            # Wrong type of exception
                            results["failed"].append((test, str(e), "Expected ValueError but got different exception"))
                            if verbose:
                                print(f"  ✗ FAILED - Wrong exception type")
                                print(f"  Exception: {e}")
                            else:
                                print("✗ FAILED")

                            if stop_on_error:
                                print("\nStopping on first failure.")
                                break
                else:
                    # Normal test - run extraction and validate
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

    print("\nUsage: uv run python examples/number_test_cases.py [--verbose|-v] [--stop-on-error|-x]\n")

    results = run_tests(verbose=verbose, stop_on_error=stop_on_error)

    # Exit with error code if any tests failed
    if results["failed"] or results["errors"]:
        sys.exit(1)
    else:
        sys.exit(0)
